"""
scratch/profile_runtime_breakdown.py
Etapa 1 do Plano de Otimização do LEGION:
Profiling Granular de Baixa Sobrecarga via CUDA Events e Perf Counter.
Mede com precisão cirúrgica onde cada milissegundo de decodificação é gasto:
  1. Embedding Lookup
  2. Atenção & Convolução Causal (40 camadas: 30 Linear Attention + 10 Full Attention)
  3. Roteador MoE Gate Top-8 (40 camadas)
  4. MoE Slot Cache Lookup e I/O de Disco
  5. MoE Rotações Hadamard (rot_2048 e rot_512 em 40 camadas)
  6. MoE Kernels de Computação (Triton Parallel Gate-Up e Down em 40 camadas)
  7. MoE Especialistas Promovidos em 3 bits (Projeções lineares)
  8. Projeção Final LM Head (Vocabulário gigante de 248.320 tokens)
  9. Amostragem / Seleção do Próximo Token
"""
import os, sys, time, json
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legion.engine import LegionEngine

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_profiling():
    print("=" * 80)
    print("ETAPA 1: PROFILING GRANULAR DO RUNTIME LEGION (ARTEFATO B2)")
    print("=" * 80)

    # 1. Inicializa o motor com B2
    print("[1/3] Inicializando Motor LEGION...")
    engine = LegionEngine()
    tokenizer = engine.tokenizer

    test_prompt = "A inteligência artificial generativa baseada em arquiteturas Mixture-of-Experts permite que"
    inputs = tokenizer(test_prompt, return_tensors="pt")["input_ids"].to(DEVICE)

    # 2. Aquecimento (Warmup de 5 tokens para compilar kernels Triton e estabilizar cache)
    print("\n[2/3] Executando Aquecimento (Warmup de 5 tokens)...")
    with torch.no_grad():
        out = engine.model(input_ids=inputs, use_cache=True)
        past_kv = out.past_key_values
        next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()

    curr = torch.tensor([[next_tok]], device=DEVICE)
    for _ in range(5):
        with torch.no_grad():
            out = engine.model(input_ids=curr, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()
            curr = torch.tensor([[next_tok]], device=DEVICE)

    torch.cuda.synchronize()
    print("      [OK] Warmup concluído com sucesso!")

    # 3. Instrumentação Granular
    print("\n[3/3] Instrumentando 15 passos de decodificação autoregressiva...")

    # Instrumenta PagedMoEExpert para registrar seus componentes internos
    moe_timings = defaultdict(float)

    orig_forwards = {}
    for l_idx, layer in enumerate(engine.layers):
        orig_fwd = layer.forward

        def make_instrumented_fwd(l_mod):
            def instrumented_forward(hidden_states, topk_idx, topk_weights):
                T = hidden_states.shape[0]
                if T == 1:
                    topk_ids_list = topk_idx.view(-1).tolist()
                    topk_w = topk_weights.view(-1).bfloat16()

                    # A. I/O e Cache Slot Lookup
                    t0 = time.perf_counter()
                    promoted_in = [e for e in topk_ids_list if e in l_mod.promoted_3b]
                    reg_in = [e for e in topk_ids_list if e not in l_mod.promoted_3b]
                    reg_slots = [l_mod._get_slot(e) for e in reg_in]
                    moe_timings["io_lookup"] += (time.perf_counter() - t0) * 1000

                    # B. Especialistas Promovidos (3 bits)
                    promoted_out = torch.zeros(1, 2048, device=DEVICE, dtype=torch.bfloat16)
                    if promoted_in:
                        t0 = time.perf_counter()
                        xt = hidden_states.bfloat16()
                        for k_idx, e_id in enumerate(topk_ids_list):
                            if e_id in l_mod.promoted_3b:
                                W_gu = l_mod.promoted_3b[e_id]["gu"]
                                W_dn = l_mod.promoted_3b[e_id]["dn"]
                                y_gu = F.linear(xt, W_gu)
                                g, u = y_gu[:, :512], y_gu[:, 512:]
                                act = (F.silu(g.float()) * u.float()).bfloat16()
                                promoted_out += F.linear(act, W_dn) * topk_w[k_idx]
                        moe_timings["promoted_3b"] += (time.perf_counter() - t0) * 1000

                    # C. Rotação Hadamard rot_2048
                    if reg_slots:
                        ev_s = torch.cuda.Event(enable_timing=True)
                        ev_e = torch.cuda.Event(enable_timing=True)
                        ev_s.record()
                        xt = hidden_states.bfloat16()
                        xr = l_mod.rot_2048.fwd(xt.view(-1))
                        ev_e.record()
                        torch.cuda.synchronize()
                        moe_timings["rot_2048"] += ev_s.elapsed_time(ev_e)

                        # D. Triton Parallel Gate-Up
                        slots_t = torch.tensor(reg_slots, dtype=torch.int32, device=DEVICE)
                        reg_weights = [topk_w[k] for k, e in enumerate(topk_ids_list) if e in reg_in]
                        reg_w = torch.stack(reg_weights)
                        K_reg = len(reg_slots)
                        y_gu = l_mod.y_gu[:K_reg]

                        from vq2.kernels import vq_moe_parallel_gate_up, vq_moe_parallel_down
                        ev_s.record()
                        vq_moe_parallel_gate_up(
                            xr, slots_t, l_mod.pool_gu_idx, l_mod.cb16, l_mod.pool_gu_scale, y_gu,
                            OUT=1024, NG=256, rs=l_mod.RS_2048
                        )
                        ev_e.record()
                        torch.cuda.synchronize()
                        moe_timings["triton_gate_up"] += ev_s.elapsed_time(ev_e)

                        # E. Ativação SiLU e Rotação rot_512
                        ev_s.record()
                        gate = y_gu[:, :512]
                        up = y_gu[:, 512:]
                        act = (F.silu(gate.float()) * up.float()).bfloat16()
                        act_r = l_mod.act_r[:K_reg]
                        act_r.copy_(l_mod.rot_512.fwd(act))
                        ev_e.record()
                        torch.cuda.synchronize()
                        moe_timings["act_rot_512"] += ev_s.elapsed_time(ev_e)

                        # F. Triton Parallel Down
                        y_down = l_mod.y_down[:K_reg]
                        ev_s.record()
                        vq_moe_parallel_down(
                            act_r, slots_t, l_mod.pool_dn_idx, l_mod.cb16, l_mod.pool_dn_scale, y_down,
                            OUT=2048, NG=64, rs=l_mod.RS_512
                        )
                        ev_e.record()
                        torch.cuda.synchronize()
                        moe_timings["triton_down"] += ev_s.elapsed_time(ev_e)

                        # G. Soma ponderada
                        ev_s.record()
                        reg_out = (y_down * reg_w[:, None]).sum(dim=0, keepdim=True)
                        out_final = promoted_out + reg_out
                        ev_e.record()
                        torch.cuda.synchronize()
                        moe_timings["combine"] += ev_s.elapsed_time(ev_e)

                        return out_final
                    return promoted_out
                else:
                    return orig_fwd(hidden_states, topk_idx, topk_weights)

            return instrumented_forward

        orig_forwards[l_idx] = orig_fwd
        layer.forward = make_instrumented_fwd(layer)

    # Instrumenta Roteadores Top-8
    router_time_ms = [0.0]
    for layer in engine.model.model.layers:
        gate_mod = layer.mlp.gate
        orig_gate = gate_mod.forward
        def make_gate_fwd(g_mod, o_fn):
            def gate_instrumented(x):
                t0 = time.perf_counter()
                res = o_fn(x)
                torch.cuda.synchronize()
                router_time_ms[0] += (time.perf_counter() - t0) * 1000
                return res
            return gate_instrumented
        gate_mod.forward = make_gate_fwd(gate_mod, orig_gate)

    step_timings = []
    num_steps = 15

    for step in range(num_steps):
        t_step0 = time.perf_counter()

        # 1. Embed Tokens
        t_embed_s = time.perf_counter()
        with torch.no_grad():
            emb = engine.model.model.embed_tokens(curr).to(dtype=torch.bfloat16)
        torch.cuda.synchronize()
        t_embed = (time.perf_counter() - t_embed_s) * 1000

        # Zera contadores de MoE e Router para este step
        for k in moe_timings:
            moe_timings[k] = 0.0
        router_time_ms[0] = 0.0

        # 2. Forward das 40 Camadas (Atenção + MoE + Norms)
        t_layers_s = time.perf_counter()
        with torch.no_grad():
            out = engine.model.model(inputs_embeds=emb, past_key_values=past_kv, use_cache=True)
            last_hidden = out.last_hidden_state[:, -1:, :]
            past_kv = out.past_key_values
        torch.cuda.synchronize()
        t_layers = (time.perf_counter() - t_layers_s) * 1000

        # 3. LM Head (Projeção no vocabulário de 248.320 tokens)
        t_lmhead_s = time.perf_counter()
        with torch.no_grad():
            logits = F.linear(last_hidden, engine.model.lm_head.weight).squeeze(0).squeeze(0).float()
        torch.cuda.synchronize()
        t_lmhead = (time.perf_counter() - t_lmhead_s) * 1000

        # 4. Sampling
        t_samp_s = time.perf_counter()
        next_tok = torch.argmax(logits).item()
        curr = torch.tensor([[next_tok]], device=DEVICE)
        t_samp = (time.perf_counter() - t_samp_s) * 1000

        t_total_step = (time.perf_counter() - t_step0) * 1000

        # Subtrai MoE e Router do tempo das camadas para obter o tempo puro de Atenção/Linear Conv1D
        moe_total_step = sum(moe_timings.values())
        router_total_step = router_time_ms[0]
        attn_total_step = max(0.0, t_layers - moe_total_step - router_total_step)

        record = {
            "step": step + 1,
            "total_ms": t_total_step,
            "embed_ms": t_embed,
            "attention_ms": attn_total_step,
            "router_ms": router_total_step,
            "moe_io_ms": moe_timings["io_lookup"],
            "moe_rot2048_ms": moe_timings["rot_2048"],
            "moe_gate_up_ms": moe_timings["triton_gate_up"],
            "moe_act_rot512_ms": moe_timings["act_rot_512"],
            "moe_down_ms": moe_timings["triton_down"],
            "moe_combine_ms": moe_timings["combine"],
            "moe_promoted_ms": moe_timings["promoted_3b"],
            "lm_head_ms": t_lmhead,
            "sampling_ms": t_samp
        }
        step_timings.append(record)

        print(f"  Token {step+1:2d}/15: {t_total_step:.1f} ms ({1000/t_total_step:.2f} tok/s) | "
              f"Attn: {attn_total_step:.1f}ms | MoE Kernels: {moe_timings['triton_gate_up']+moe_timings['triton_down']:.1f}ms | "
              f"LM Head: {t_lmhead:.1f}ms", flush=True)

    # 4. Consolidação Estatística de Profiling
    print("\n" + "=" * 80)
    print("RELATÓRIO DE PROFILING: DECOMPOSIÇÃO TEMPORAL POR TOKEN DECODE (15 TOKENS):")
    print("=" * 80)

    n = len(step_timings)
    avg_total = sum(r["total_ms"] for r in step_timings) / n

    def get_avg(k): return sum(r[k] for r in step_timings) / n

    components = [
        ("Atenção & Convolução Causal (40 Camadas)", get_avg("attention_ms")),
        ("Projeção LM Head (Vocab 248.320 x 2048)", get_avg("lm_head_ms")),
        ("Triton Parallel Gate-Up Kernel (40L)", get_avg("moe_gate_up_ms")),
        ("Triton Parallel Down Kernel (40L)", get_avg("moe_down_ms")),
        ("MoE Rotações Hadamard (rot_2048 + rot_512)", get_avg("moe_rot2048_ms") + get_avg("moe_act_rot512_ms")),
        ("Roteador MoE Gate Top-8 (40L)", get_avg("router_ms")),
        ("MoE Cache Slot Lookup & I/O de Disco", get_avg("moe_io_ms")),
        ("MoE 16 Especialistas 3b (Projeções)", get_avg("moe_promoted_ms")),
        ("MoE Combinação Ponderada (Soma 8 exp)", get_avg("moe_combine_ms")),
        ("Embedding Lookup", get_avg("embed_ms")),
        ("Amostragem / Argmax", get_avg("sampling_ms")),
    ]

    components.sort(key=lambda x: x[1], reverse=True)

    print(f"{'Estágio / Operação do Pipeline':<45} | {'Tempo Médio (ms)':<16} | {'Impacto (%)':<10}")
    print("-" * 80)
    for comp, t_ms in components:
        pct = (t_ms / avg_total) * 100
        bar = "█" * int(pct / 3)
        print(f"{comp:<45} | {t_ms:7.2f} ms       | {pct:5.1f}%  {bar}")
    print("-" * 80)
    print(f"{'TEMPO TOTAL MÉDIO POR TOKEN':<45} | {avg_total:7.2f} ms       | 100.0%")
    print(f"VELOCIDADE EFETIVA ATUAL: {1000 / avg_total:.2f} tokens/segundo")
    print("=" * 80)

    # 5. Diagnóstico de Gargalos Prioritários
    print("\n" + "=" * 80)
    print("DIAGNÓSTICO TÉCNICO: ONDE ESTÃO OS REAIS GARGALOS PARA CHEGAR A 30 TOK/S:")
    print("=" * 80)
    top1 = components[0]
    top2 = components[1]
    top3 = components[2]
    print(f"1º Maior Gargalo: {top1[0]} -> {top1[1]:.1f} ms ({top1[1]/avg_total*100:.1f}%)")
    print(f"2º Maior Gargalo: {top2[0]} -> {top2[1]:.1f} ms ({top2[1]/avg_total*100:.1f}%)")
    print(f"3º Maior Gargalo: {top3[0]} -> {top3[1]:.1f} ms ({top3[1]/avg_total*100:.1f}%)")
    print("=" * 80)

    # Salva JSON
    profile_out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "avg_total_step_ms": avg_total,
        "current_tok_per_sec": 1000 / avg_total,
        "target_30_tok_s_budget_ms": 1000 / 30.0, # 33.3 ms
        "components": [
            {"name": c, "time_ms": t, "percent": (t / avg_total) * 100} for c, t in components
        ],
        "detailed_steps": step_timings
    }
    with open(ROOT / "scratch" / "profiling_breakdown_results.json", "w", encoding="utf-8") as f:
        json.dump(profile_out, f, indent=2, ensure_ascii=False)
    print(f"[OK] Dados salvos em: {ROOT / 'scratch' / 'profiling_breakdown_results.json'}")

if __name__ == "__main__":
    run_profiling()
