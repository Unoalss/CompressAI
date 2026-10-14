import sys, os, time
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legion.engine import LegionEngine
from vq2.kernels import vq_moe_parallel_gate_up, vq_moe_parallel_down, triton_fused_add_rmsnorm

def test_all_layers_graph():
    print("=" * 80)
    print("TESTE DE CAPTURA DE TODAS AS 40 CAMADAS EM CUDA GRAPHS")
    print("=" * 80)

    engine = LegionEngine()
    device = "cuda"

    prompt = "A inteligência artificial avançada permite"
    prompt_ids = engine.tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()

    with torch.no_grad():
        out = engine.model(input_ids=prompt_ids, use_cache=True)
        past_kv = out.past_key_values
        next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()

    curr = torch.tensor([[next_tok]], device=device)
    # Warmup do cache MoE para estado quente
    for _ in range(15):
        with torch.no_grad():
            out = engine.model(input_ids=curr, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()
            curr = torch.tensor([[next_tok]], device=device)
    torch.cuda.synchronize()

    # Prepara buffers estáticos compartilhados
    N_LAYERS = 40
    print(f"\n[1/3] Inicializando e capturando os grafos das {N_LAYERS} camadas...")
    
    # Cada camada terá seus buffers e grafos
    graphs_a = []
    graphs_b = []
    buffers = []

    pos_step = prompt_ids.shape[1] + 15
    pos_ids_3d = torch.zeros(3, 1, 1, dtype=torch.long, device=device) + pos_step
    
    # Pre-calcula pos_emb estático
    dummy_x = torch.zeros(1, 1, 2048, dtype=torch.bfloat16, device=device)
    static_cos, static_sin = engine.model.model.rotary_emb(dummy_x, pos_ids_3d)

    pinned_slots = torch.empty(8, dtype=torch.int32, pin_memory=True)

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())

    t0_cap = time.time()
    for l_idx in range(N_LAYERS):
        layer = engine.model.model.layers[l_idx]
        paged_expert = layer.mlp.experts

        buf = {
            "x_in": torch.zeros(1, 1, 2048, dtype=torch.bfloat16, device=device),
            "slots": torch.zeros(8, dtype=torch.int32, device=device),
            "topk_weights": torch.zeros(8, dtype=torch.bfloat16, device=device),
            "res": None,
            "post_norm": None,
            "sh_final": None,
            "rw": None,
            "ri": None,
            "x_out": None
        }

        # Definição Parte A
        def make_part_a(l, b):
            is_linear = (l.block_type == "linear_attention")
            def part_a():
                x = b["x_in"]
                residual = x
                normed = l.input_layernorm(x)
                if is_linear:
                    attn_out = l.linear_attn(normed, cache_params=past_kv)
                else:
                    attn_out, _ = l.self_attn(normed, position_embeddings=(static_cos, static_sin), past_key_values=past_kv)
                res, post_norm = triton_fused_add_rmsnorm(
                    residual, attn_out, l.post_attention_layernorm.weight, l.post_attention_layernorm.eps
                )
                sh_out = l.mlp.shared_expert(post_norm.view(-1, 2048))
                sh_gate = F.sigmoid(l.mlp.shared_expert_gate(post_norm.view(-1, 2048)))
                sh_final = (sh_gate * sh_out).view(1, 1, 2048)
                _, r_weights, r_indices = l.mlp.gate(post_norm.view(-1, 2048))
                return res, post_norm, sh_final, r_weights, r_indices
            return part_a

        # Definição Parte B
        def make_part_b(l, b, exp):
            def part_b():
                K = 8
                xt = b["post_norm"].view(-1).bfloat16()
                xr = exp.rot_2048.fwd(xt)
                y_gu = exp.y_gu[:K]
                vq_moe_parallel_gate_up(
                    xr, b["slots"], exp.pool_gu_idx, exp.cb16, exp.pool_gu_scale, y_gu,
                    OUT=1024, NG=256, rs=exp.RS_2048
                )
                gate = y_gu[:, :512]
                up = y_gu[:, 512:]
                act = (F.silu(gate.float()) * up.float()).bfloat16()
                act_r = exp.rot_512.fwd(act)
                y_down = exp.y_down[:K]
                vq_moe_parallel_down(
                    act_r, b["slots"], exp.pool_dn_idx, exp.cb16, exp.pool_dn_scale, y_down,
                    OUT=2048, NG=64, rs=exp.RS_512
                )
                expert_out = (y_down * b["topk_weights"][:, None]).sum(dim=0, keepdim=True).view(1, 1, 2048)
                return b["res"] + b["sh_final"] + expert_out
            return part_b

        fn_a = make_part_a(layer, buf)
        fn_b = make_part_b(layer, buf, paged_expert)

        # Warmup e Captura Grafo A
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                with torch.no_grad():
                    res, pn, sh, rw, ri = fn_a()
        torch.cuda.current_stream().wait_stream(capture_stream)

        ga = torch.cuda.CUDAGraph()
        with torch.cuda.graph(ga, stream=capture_stream):
            buf["res"], buf["post_norm"], buf["sh_final"], buf["rw"], buf["ri"] = fn_a()

        # Warmup e Captura Grafo B
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                with torch.no_grad():
                    out_b = fn_b()
        torch.cuda.current_stream().wait_stream(capture_stream)

        gb = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gb, stream=capture_stream):
            buf["x_out"] = fn_b()

        graphs_a.append(ga)
        graphs_b.append(gb)
        buffers.append(buf)

    print(f"      [CONCLUÍDO] 40 camadas capturadas com sucesso em {time.time() - t0_cap:.2f}s!")

    # 2. Benchmark Comparativo: Eager vs CUDA Graph Decode (30 tokens)
    print("\n[2/3] Executando Benchmark Comparativo de Decodificação (30 tokens)...")
    
    # 2.1 Eager Forward
    eager_latencies = []
    curr_eager = curr.clone()
    for _ in range(30):
        ev_s = torch.cuda.Event(enable_timing=True)
        ev_e = torch.cuda.Event(enable_timing=True)
        ev_s.record()
        with torch.no_grad():
            out = engine.model(input_ids=curr_eager, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_t = torch.argmax(out.logits[:, -1, :], dim=-1).item()
            curr_eager = torch.tensor([[next_t]], device=device)
        ev_e.record()
        torch.cuda.synchronize()
        eager_latencies.append(ev_s.elapsed_time(ev_e))

    # 2.2 CUDA Graph Pipeline Forward
    graph_latencies = []
    curr_graph = curr.clone()
    norm_weight = engine.model.model.norm.weight
    lm_head_weight = engine.model.lm_head.weight

    for _ in range(30):
        ev_s = torch.cuda.Event(enable_timing=True)
        ev_e = torch.cuda.Event(enable_timing=True)
        ev_s.record()
        with torch.no_grad():
            # Embed
            h = engine.model.model.embed_tokens(curr_graph)
            # 40 Camadas via Grafos
            for l_idx in range(N_LAYERS):
                buf = buffers[l_idx]
                buf["x_in"].copy_(h)
                
                # Replay A
                graphs_a[l_idx].replay()
                
                # Bridge CPU (Resolve slots)
                exp = engine.model.model.layers[l_idx].mlp.experts
                topk_ids = buf["ri"].view(-1).tolist()
                for i, e in enumerate(topk_ids):
                    pinned_slots[i] = exp._get_slot(e)
                buf["slots"].copy_(pinned_slots, non_blocking=True)
                buf["topk_weights"].copy_(buf["rw"].view(-1), non_blocking=True)
                
                # Replay B
                graphs_b[l_idx].replay()
                h = buf["x_out"]

            # Head
            normed = engine.model.model.norm(h)
            logits = F.linear(normed, lm_head_weight).squeeze(0).squeeze(0).float()
            next_t = torch.argmax(logits, dim=-1).item()
            curr_graph = torch.tensor([[next_t]], device=device)
        ev_e.record()
        torch.cuda.synchronize()
        graph_latencies.append(ev_s.elapsed_time(ev_e))

    eager_p50 = sorted(eager_latencies)[15]
    graph_p50 = sorted(graph_latencies)[15]
    eager_tps = 1000.0 / eager_p50
    graph_tps = 1000.0 / graph_p50

    print("\n" + "=" * 80)
    print("RESULTADOS DO BENCHMARK END-TO-END (30 TOKENS):")
    print("=" * 80)
    print(f"Modo Padrão Eager: {eager_p50:.2f} ms/token ({eager_tps:.2f} tok/s)")
    print(f"Modo CUDA Graphs:  {graph_p50:.2f} ms/token ({graph_tps:.2f} tok/s)")
    gain_pct = ((graph_tps - eager_tps) / eager_tps) * 100.0
    print(f"Ganho Relativo:    {gain_pct:+.2f}% ({eager_p50 - graph_p50:+.2f} ms economizados por token!)")
    print("=" * 80)

if __name__ == "__main__":
    test_all_layers_graph()
