import sys, os, time
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legion.engine import LegionEngine
from vq2.kernels import vq_moe_parallel_gate_up, vq_moe_parallel_down

def test_pipeline():
    print("=" * 80)
    print("PROTÓTIPO EXPERIMENTAL: CUDA GRAPH DECODE PIPELINE")
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
    # Warmup do cache MoE
    for _ in range(5):
        with torch.no_grad():
            out = engine.model(input_ids=curr, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()
            curr = torch.tensor([[next_tok]], device=device)
    torch.cuda.synchronize()

    # Vamos construir o protótipo de grafo para a Camada 0 (DeltaNet)
    layer = engine.model.model.layers[0]
    paged_expert = layer.mlp.experts

    # Buffers estáticos dedicados na GPU
    static_x_in = torch.zeros(1, 1, 2048, dtype=torch.bfloat16, device=device)
    static_slots = torch.zeros(8, dtype=torch.int32, device=device)
    static_topk_weights = torch.zeros(8, dtype=torch.bfloat16, device=device)

    # 1. Função da Parte A: Norm + DeltaNet Attention + Fused Add/Norm + Router + Shared Expert
    def part_a_fn(x_in):
        residual = x_in
        normed = layer.input_layernorm(x_in)
        attn_out = layer.linear_attn(normed, cache_params=past_kv)
        from vq2.kernels import triton_fused_add_rmsnorm
        res, post_norm = triton_fused_add_rmsnorm(
            residual, attn_out, layer.post_attention_layernorm.weight, layer.post_attention_layernorm.eps
        )
        # Shared expert
        sh_out = layer.mlp.shared_expert(post_norm.view(-1, 2048))
        sh_gate = F.sigmoid(layer.mlp.shared_expert_gate(post_norm.view(-1, 2048)))
        sh_final = (sh_gate * sh_out).view(1, 1, 2048)

        # Router
        _, r_weights, r_indices = layer.mlp.gate(post_norm.view(-1, 2048))
        return res, post_norm, sh_final, r_weights, r_indices

    # 2. Função da Parte B: MoE Triton Kernels + Shared Add + Residual Add
    def part_b_fn(post_norm, slots, topk_w, sh_final, res):
        K = 8
        xt = post_norm.view(-1).bfloat16()
        xr = paged_expert.rot_2048.fwd(xt)
        y_gu = paged_expert.y_gu[:K]
        vq_moe_parallel_gate_up(
            xr, slots, paged_expert.pool_gu_idx, paged_expert.cb16, paged_expert.pool_gu_scale, y_gu,
            OUT=1024, NG=256, rs=paged_expert.RS_2048
        )
        gate = y_gu[:, :512]
        up = y_gu[:, 512:]
        act = (F.silu(gate.float()) * up.float()).bfloat16()
        act_r = paged_expert.rot_512.fwd(act)
        y_down = paged_expert.y_down[:K]
        vq_moe_parallel_down(
            act_r, slots, paged_expert.pool_dn_idx, paged_expert.cb16, paged_expert.pool_dn_scale, y_down,
            OUT=2048, NG=64, rs=paged_expert.RS_512
        )
        expert_out = (y_down * topk_w[:, None]).sum(dim=0, keepdim=True).view(1, 1, 2048)
        final_layer_out = res + sh_final + expert_out
        return final_layer_out

    # Stream de captura
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())

    # Warmup Parte A
    with torch.cuda.stream(s):
        for _ in range(3):
            with torch.no_grad():
                res_a, pn_a, sh_a, rw_a, ri_a = part_a_fn(static_x_in)
    torch.cuda.current_stream().wait_stream(s)

    # Captura Parte A
    g_a = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_a, stream=s):
        res_a, pn_a, sh_a, rw_a, ri_a = part_a_fn(static_x_in)
    print("  [SUCESSO] Grafo A (Attention + Router + Shared Expert) capturado!")

    # Warmup Parte B
    with torch.cuda.stream(s):
        for _ in range(3):
            with torch.no_grad():
                out_b = part_b_fn(pn_a, static_slots, static_topk_weights, sh_a, res_a)
    torch.cuda.current_stream().wait_stream(s)

    # Captura Parte B
    g_b = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_b, stream=s):
        out_b = part_b_fn(pn_a, static_slots, static_topk_weights, sh_a, res_a)
    print("  [SUCESSO] Grafo B (MoE Triton Kernels + Residual Add) capturado!")

    # Benchmark: Comparar tempo de 100 execuções da Camada 0
    # Modo Padrão (PyTorch Eager) vs Modo Grafo (Replay A + Bridge + Replay B)
    N_ITERS = 100
    
    # 1. Medição Eager
    torch.cuda.synchronize()
    ev_eager_s = torch.cuda.Event(enable_timing=True)
    ev_eager_e = torch.cuda.Event(enable_timing=True)
    ev_eager_s.record()
    for _ in range(N_ITERS):
        with torch.no_grad():
            res_a, pn_a, sh_a, rw_a, ri_a = part_a_fn(static_x_in)
            # Simula bridge
            topk_ids = ri_a.view(-1).tolist()
            for i, e in enumerate(topk_ids):
                static_slots[i] = paged_expert._get_slot(e)
            static_topk_weights.copy_(rw_a.view(-1))
            out_b = part_b_fn(pn_a, static_slots, static_topk_weights, sh_a, res_a)
    ev_eager_e.record()
    torch.cuda.synchronize()
    t_eager_ms = ev_eager_s.elapsed_time(ev_eager_e) / N_ITERS

    # 2. Medição com CUDA Graphs
    pinned_slots = torch.empty(8, dtype=torch.int32, pin_memory=True)
    torch.cuda.synchronize()
    ev_graph_s = torch.cuda.Event(enable_timing=True)
    ev_graph_e = torch.cuda.Event(enable_timing=True)
    ev_graph_s.record()
    for _ in range(N_ITERS):
        # 1. Replay Grafo A (GPU pura, 0 dispatch overhead)
        g_a.replay()
        # 2. Bridge CPU (leitura e resolução dos slots)
        topk_ids = ri_a.view(-1).tolist()
        for i, e in enumerate(topk_ids):
            pinned_slots[i] = paged_expert._get_slot(e)
        static_slots.copy_(pinned_slots, non_blocking=True)
        static_topk_weights.copy_(rw_a.view(-1), non_blocking=True)
        # 3. Replay Grafo B (GPU pura, 0 dispatch overhead)
        g_b.replay()
    ev_graph_e.record()
    torch.cuda.synchronize()
    t_graph_ms = ev_graph_s.elapsed_time(ev_graph_e) / N_ITERS

    print(f"\n--- RESULTADOS NA CAMADA 0 (100 iterações) ---")
    print(f"Modo Padrão Eager: {t_eager_ms:.4f} ms por execução")
    print(f"Modo CUDA Graphs:  {t_graph_ms:.4f} ms por execução")
    speedup = ((t_eager_ms - t_graph_ms) / t_eager_ms) * 100
    print(f"Ganho na camada:   {speedup:+.2f}% ({t_eager_ms - t_graph_ms:+.4f} ms por camada)")
    print(f"Economia projetada em 40 camadas: {(t_eager_ms - t_graph_ms) * 40:+.2f} ms por token!")

if __name__ == "__main__":
    test_pipeline()
