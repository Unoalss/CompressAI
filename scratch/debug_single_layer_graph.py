import sys, os, time
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legion.engine import LegionEngine
from vq2.kernels import vq_moe_parallel_gate_up, vq_moe_parallel_down, triton_fused_add_rmsnorm

def debug_single_layer():
    print("=" * 80)
    print("DEBUG: COMPARAÇÃO BIT A BIT NA CAMADA 0 (EAGER VS GRAPH)")
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
    layer = engine.model.model.layers[0]
    paged_expert = layer.mlp.experts

    h_in = engine.model.model.embed_tokens(curr)

    pos_ids_3d = torch.zeros(3, 1, 1, dtype=torch.long, device=device) + prompt_ids.shape[1]
    pos_emb = engine.model.model.rotary_emb(h_in, pos_ids_3d)

    # 1. Executa Camada 0 Eager
    with torch.no_grad():
        out_eager = layer(h_in, pos_emb, past_key_values=past_kv)
        if isinstance(out_eager, tuple):
            out_eager = out_eager[0]

    print(f"Eager out shape: {out_eager.shape}, min={out_eager.min():.4f}, max={out_eager.max():.4f}, mean={out_eager.mean():.4f}")

    # 2. Executa Camada 0 com Grafo
    # Prepara buffers
    buf_x_in = h_in.clone()
    buf_slots = torch.zeros(8, dtype=torch.int32, device=device)
    buf_weights = torch.zeros(8, dtype=torch.bfloat16, device=device)

    # Função A
    def fn_a():
        residual = buf_x_in
        normed = layer.input_layernorm(buf_x_in)
        attn_out = layer.linear_attn(normed, cache_params=past_kv)
        res, post_norm = triton_fused_add_rmsnorm(
            residual, attn_out, layer.post_attention_layernorm.weight, layer.post_attention_layernorm.eps
        )
        sh_out = layer.mlp.shared_expert(post_norm.view(-1, 2048))
        sh_gate = F.sigmoid(layer.mlp.shared_expert_gate(post_norm.view(-1, 2048)))
        sh_final = (sh_gate * sh_out).view(1, 1, 2048)
        _, r_weights, r_indices = layer.mlp.gate(post_norm.view(-1, 2048))
        return res, post_norm, sh_final, r_weights, r_indices

    # Warmup
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            with torch.no_grad():
                res, pn, sh, rw, ri = fn_a()
    torch.cuda.current_stream().wait_stream(s)

    ga = torch.cuda.CUDAGraph()
    with torch.cuda.graph(ga, stream=s):
        res, pn, sh, rw, ri = fn_a()

    # Função B
    def fn_b():
        K = 8
        xt = pn.view(-1).bfloat16()
        xr = paged_expert.rot_2048.fwd(xt)
        y_gu = paged_expert.y_gu[:K]
        vq_moe_parallel_gate_up(
            xr, buf_slots, paged_expert.pool_gu_idx, paged_expert.cb16, paged_expert.pool_gu_scale, y_gu,
            OUT=1024, NG=256, rs=paged_expert.RS_2048
        )
        gate = y_gu[:, :512]
        up = y_gu[:, 512:]
        act = (F.silu(gate.float()) * up.float()).bfloat16()
        act_r = paged_expert.rot_512.fwd(act)
        y_down = paged_expert.y_down[:K]
        vq_moe_parallel_down(
            act_r, buf_slots, paged_expert.pool_dn_idx, paged_expert.cb16, paged_expert.pool_dn_scale, y_down,
            OUT=2048, NG=64, rs=paged_expert.RS_512
        )
        expert_out = (y_down * buf_weights[:, None]).sum(dim=0, keepdim=True).view(1, 1, 2048)
        return res + sh + expert_out

    with torch.cuda.stream(s):
        for _ in range(3):
            with torch.no_grad():
                out_b = fn_b()
    torch.cuda.current_stream().wait_stream(s)

    gb = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gb, stream=s):
        out_b = fn_b()

    # Agora faz o Replay do Grafo
    ga.replay()
    topk_ids = ri.view(-1).tolist()
    for i, e in enumerate(topk_ids):
        buf_slots[i] = paged_expert._get_slot(e)
    buf_weights.copy_(rw.view(-1))
    gb.replay()
    torch.cuda.synchronize()

    print(f"Graph out shape: {out_b.shape}, min={out_b.min():.4f}, max={out_b.max():.4f}, mean={out_b.mean():.4f}")
    cos = F.cosine_similarity(out_eager.view(1, -1).float(), out_b.view(1, -1).float()).item()
    diff = torch.max(torch.abs(out_eager - out_b)).item()
    print(f"Cosine Similarity: {cos:.6f}")
    print(f"Max Absolute Diff: {diff:.6f}")

if __name__ == "__main__":
    debug_single_layer()
