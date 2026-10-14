"""
scratch/bench_alternating_a_b.py
Benchmark Científico com Execuções Alternadas de A e B:
- Separa aquecimento (warmup), prefill TTFT (ms) e velocidade de decodificação (tokens/s).
- 3 rodadas alternadas (A->B, B->A, A->B) sobre prompts inéditos.
- Amostragem gananciosa (greedy) pura sem penalidade artificial de repetição.
"""
import sys, time, json
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vq2 import vq
from vq2.kernels import vq_moe_parallel_gate_up, vq_moe_parallel_down
from legion.engine import LegionEngine

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ANCHOR_EXPERTS_L39 = [133, 56, 24, 234, 204, 123]

BENCHMARK_PROMPTS = [
    # Prompt 1: Raciocínio Físico
    "<|im_start|>user\nDescreva os quatro principais testes experimentais da relatividade geral propostos por Einstein.<|im_end|>\n<|im_start|>assistant\n",
    # Prompt 2: Algoritmos e Otimização
    "<|im_start|>user\nExplique o funcionamento do algoritmo de Dijkstra para caminhos mínimos e sua complexidade de tempo.<|im_end|>\n<|im_start|>assistant\n",
    # Prompt 3: Filosofia da Ciência
    "<|im_start|>user\nO que define o critério de falseabilidade de Karl Popper na demarcação científica?<|im_end|>\n<|im_start|>assistant\n"
]

def load_3bit_anchors():
    cbs_data = torch.load(ROOT / "vq2_out" / "codebooks.pt", weights_only=False)
    cb16 = cbs_data["16"].to(DEVICE, torch.bfloat16)
    cb8 = cbs_data["16_8"].to(DEVICE, torch.bfloat16)
    rot_2048 = vq.get_rot("moe_expert_2048", 2048, DEVICE)
    rot_512 = vq.get_rot("moe_expert_512", 512, DEVICE)

    # Lê do container B recém-compilado!
    from vq2.paged_runtime import PagedModelStore
    store_b = PagedModelStore(ROOT / "vq2_35b_out" / "qwen35b_mixed_b.vq2m", use_mmap=False)

    weights_3b = {}
    for e_id in ANCHOR_EXPERTS_L39:
        off_gui, sz_gui, off_gus, sz_gus, off_dni, sz_dni, off_dns, sz_dns = store_b.moe_offsets[39][e_id]
        store_b.file.seek(off_gui)
        raw_gui = store_b.file.read(sz_gui)
        idx0_gu = torch.frombuffer(raw_gui[:524288], dtype=torch.int16).view(1024, 256).to(DEVICE)
        idx1_gu = torch.frombuffer(raw_gui[524288:], dtype=torch.uint8).view(1024, 256).to(DEVICE)

        store_b.file.seek(off_gus)
        raw_gus = store_b.file.read(sz_gus)
        sc_gu = torch.frombuffer(raw_gus, dtype=torch.float16).to(DEVICE, torch.bfloat16)

        W_rec_gu = rot_2048.inv(((cb16[idx0_gu.long()] + cb8[idx1_gu.long()]).view(1024, 2048) * sc_gu[:, None]).to(torch.bfloat16))

        store_b.file.seek(off_dni)
        raw_dni = store_b.file.read(sz_dni)
        idx0_dn = torch.frombuffer(raw_dni[:262144], dtype=torch.int16).view(2048, 64).to(DEVICE)
        idx1_dn = torch.frombuffer(raw_dni[262144:], dtype=torch.uint8).view(2048, 64).to(DEVICE)

        store_b.file.seek(off_dns)
        raw_dns = store_b.file.read(sz_dns)
        sc_dn = torch.frombuffer(raw_dns, dtype=torch.float16).to(DEVICE, torch.bfloat16)

        W_rec_dn = rot_512.inv(((cb16[idx0_dn.long()] + cb8[idx1_dn.long()]).view(2048, 512) * sc_dn[:, None]).to(torch.bfloat16))
        weights_3b[e_id] = {"gu": W_rec_gu, "dn": W_rec_dn}

    store_b.close()
    return weights_3b

def generate_measured(engine, prompt, custom_weights=None, max_new_tokens=60):
    l39 = engine.layers[39]
    orig_forward = l39.forward

    if custom_weights:
        def custom_fwd(hidden_states, topk_idx, topk_weights):
            T = hidden_states.shape[0]
            if T == 1:
                ids_list = topk_idx.view(-1).tolist()
                w_t = topk_weights.view(-1).bfloat16()
                xt = hidden_states.bfloat16()
                token_out = torch.zeros(1, 2048, device=DEVICE, dtype=torch.bfloat16)
                for k_idx, e_id in enumerate(ids_list):
                    if e_id in custom_weights:
                        W_gu = custom_weights[e_id]["gu"]
                        W_dn = custom_weights[e_id]["dn"]
                        y_gu = F.linear(xt, W_gu)
                        g, u = y_gu[:, :512], y_gu[:, 512:]
                        act = (F.silu(g.float()) * u.float()).bfloat16()
                        token_out += F.linear(act, W_dn) * w_t[k_idx]
                    else:
                        slot = l39._get_slot(e_id)
                        slots_t = torch.tensor([slot], dtype=torch.int32, device=DEVICE)
                        xr = l39.rot_2048.fwd(xt.view(-1))
                        y_gu = torch.empty((1, 1024), device=DEVICE, dtype=torch.bfloat16)
                        vq_moe_parallel_gate_up(xr, slots_t, l39.pool_gu_idx, l39.cb16, l39.pool_gu_scale, y_gu, 1024, 256, l39.RS_2048)
                        g, u = y_gu[:, :512], y_gu[:, 512:]
                        act = (F.silu(g.float()) * u.float()).bfloat16()
                        act_r = l39.rot_512.fwd(act)
                        y_dn = torch.empty((1, 2048), device=DEVICE, dtype=torch.bfloat16)
                        vq_moe_parallel_down(act_r, slots_t, l39.pool_dn_idx, l39.cb16, l39.pool_dn_scale, y_dn, 2048, 64, l39.RS_512)
                        token_out += y_dn * w_t[k_idx]
                return token_out
            else:
                return orig_forward(hidden_states, topk_idx, topk_weights)
        l39.forward = custom_fwd

    input_ids = engine.tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    
    # 1. Mede Prefill TTFT
    torch.cuda.synchronize()
    t0_prefill = time.perf_counter()
    with torch.no_grad():
        out = engine.model(input_ids, use_cache=True)
        past_key_values = out.past_key_values
        next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()
    torch.cuda.synchronize()
    ttft_ms = (time.perf_counter() - t0_prefill) * 1000

    # 2. Mede Decode Steps
    generated = [next_tok]
    curr_input = torch.tensor([[next_tok]], device=DEVICE, dtype=torch.long)
    
    torch.cuda.synchronize()
    t0_decode = time.perf_counter()
    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            out = engine.model(curr_input, past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()
            if next_tok in engine.eos_ids:
                break
            generated.append(next_tok)
            curr_input = torch.tensor([[next_tok]], device=DEVICE, dtype=torch.long)
    torch.cuda.synchronize()
    decode_duration = time.perf_counter() - t0_decode
    decode_tokens = len(generated) - 1
    tokens_per_sec = decode_tokens / max(decode_duration, 1e-4)

    l39.forward = orig_forward
    text = engine.tokenizer.decode(generated, skip_special_tokens=True)
    return ttft_ms, tokens_per_sec, len(generated), text

def run_alternating_bench():
    print("=" * 80)
    print("BENCHMARK DE PERFORMANCE: EXECUÇÕES ALTERNADAS (A vs B)")
    print("Métricas: TTFT (Prefill ms) e Velocidade de Decodificação (tokens/s)")
    print("=" * 80)

    print("\n[1/3] Inicializando Motor e Carregando Especialistas 3b do Artefato B...")
    weights_3b = load_3bit_anchors()
    engine = LegionEngine()

    # Aquecimento (Warmup)
    print("\n[2/3] Executando Warmup para JIT kernels e cache GPU...")
    generate_measured(engine, "Warmup prompt", custom_weights=None, max_new_tokens=5)
    generate_measured(engine, "Warmup prompt", custom_weights=weights_3b, max_new_tokens=5)
    print("      Warmup concluído com sucesso!")

    # Execuções Alternadas
    print("\n[3/3] Iniciando 3 Rodadas Alternadas...")
    print("-" * 80)

    records_A = []
    records_B = []

    for round_idx, prompt in enumerate(BENCHMARK_PROMPTS, 1):
        print(f"\n>>> RODADA {round_idx} (Prompt: '{prompt[16:55]}...'):")
        
        if round_idx % 2 == 1:
            # Ordem: A depois B
            ttft_a, tok_s_a, n_a, _ = generate_measured(engine, prompt, custom_weights=None, max_new_tokens=50)
            ttft_b, tok_s_b, n_b, _ = generate_measured(engine, prompt, custom_weights=weights_3b, max_new_tokens=50)
        else:
            # Ordem: B depois A
            ttft_b, tok_s_b, n_b, _ = generate_measured(engine, prompt, custom_weights=weights_3b, max_new_tokens=50)
            ttft_a, tok_s_a, n_a, _ = generate_measured(engine, prompt, custom_weights=None, max_new_tokens=50)

        records_A.append((ttft_a, tok_s_a))
        records_B.append((ttft_b, tok_s_b))

        print(f"  [A - 2 bits]  TTFT: {ttft_a:>6.1f} ms | Decode: {tok_s_a:>5.2f} tok/s ({n_a} tokens)")
        print(f"  [B - 3 bits]  TTFT: {ttft_b:>6.1f} ms | Decode: {tok_s_b:>5.2f} tok/s ({n_b} tokens)")

    mean_ttft_a = sum(r[0] for r in records_A) / len(records_A)
    mean_ttft_b = sum(r[1] for r in records_B) / len(records_B) # wait, r[0] is ttft
    mean_ttft_b = sum(r[0] for r in records_B) / len(records_B)
    mean_speed_a = sum(r[1] for r in records_A) / len(records_A)
    mean_speed_b = sum(r[1] for r in records_B) / len(records_B)

    print("\n" + "=" * 80)
    print("RESULTADO CONSOLIDADO DO BENCHMARK DE VELOCIDADE:")
    print("=" * 80)
    print(f"{'Métrica':<35} | {'Artefato A (2 bits)':<20} | {'Artefato B (Misto 3b)':<20}")
    print("-" * 80)
    print(f"{'TTFT Médio (Prefill)':<35} | {f'{mean_ttft_a:.1f} ms':<20} | {f'{mean_ttft_b:.1f} ms':<20}")
    print(f"{'Velocidade Média de Decode':<35} | {f'{mean_speed_a:.2f} tok/s':<20} | {f'{mean_speed_b:.2f} tok/s':<20}")
    delta_spd = (mean_speed_b - mean_speed_a) / mean_speed_a * 100
    print(f"{'Variação de Throughput':<35} | {'Baseline':<20} | {f'{delta_spd:+.1f}%':<20}")
    print("=" * 80)

if __name__ == "__main__":
    run_alternating_bench()
