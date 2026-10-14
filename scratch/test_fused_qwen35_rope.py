import torch
import triton
import triton.language as tl
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

@triton.jit
def _fused_qwen35_rmsnorm_rope_kernel(
    x_ptr, w_ptr, cos_ptr, sin_ptr, out_ptr,
    stride_xh, stride_xd,
    stride_oh, stride_od,
    stride_cos_d, stride_sin_d,
    eps: tl.constexpr,
):
    """
    Fused RMSNorm (zero-centered: 1.0 + w) + Partial RoPE (first 64 dims of 256).
    Grid: (H,) onde H é o número de cabeças (ex: 16 para Q, 2 para K).
    D = 256, ROTARY_DIM = 64, HALF_ROTARY = 32, PASS_DIM = 192.
    """
    pid = tl.program_id(0)

    # Divisão em potências de 2: 32 + 32 + 64 + 128 = 256
    idx_32 = tl.arange(0, 32)
    idx_64 = tl.arange(0, 64)
    idx_128 = tl.arange(0, 128)

    # 1. Carrega x em 4 blocos
    x0 = tl.load(x_ptr + pid * stride_xh + idx_32 * stride_xd).to(tl.float32)        # 0..31
    x1 = tl.load(x_ptr + pid * stride_xh + (idx_32 + 32) * stride_xd).to(tl.float32) # 32..63
    x2 = tl.load(x_ptr + pid * stride_xh + (idx_64 + 64) * stride_xd).to(tl.float32) # 64..127
    x3 = tl.load(x_ptr + pid * stride_xh + (idx_128 + 128) * stride_xd).to(tl.float32) # 128..255

    # 2. Carrega w (zero-centered: 1.0 + w)
    w0 = 1.0 + tl.load(w_ptr + idx_32).to(tl.float32)
    w1 = 1.0 + tl.load(w_ptr + (idx_32 + 32)).to(tl.float32)
    w2 = 1.0 + tl.load(w_ptr + (idx_64 + 64)).to(tl.float32)
    w3 = 1.0 + tl.load(w_ptr + (idx_128 + 128)).to(tl.float32)

    # 3. RMSNorm sobre todos os 256 elementos
    sum_sq = (
        tl.sum(x0 * x0, axis=0) +
        tl.sum(x1 * x1, axis=0) +
        tl.sum(x2 * x2, axis=0) +
        tl.sum(x3 * x3, axis=0)
    )
    rms = tl.rsqrt((sum_sq / 256.0) + eps)

    # Escala normalizada
    v0 = x0 * rms * w0
    v1 = x1 * rms * w1
    v2 = x2 * rms * w2
    v3 = x3 * rms * w3

    # 4. RoPE nas primeiras 64 dimensões (v0 e v1)
    cos0 = tl.load(cos_ptr + idx_32 * stride_cos_d).to(tl.float32)
    cos1 = tl.load(cos_ptr + (idx_32 + 32) * stride_cos_d).to(tl.float32)
    sin0 = tl.load(sin_ptr + idx_32 * stride_sin_d).to(tl.float32)
    sin1 = tl.load(sin_ptr + (idx_32 + 32) * stride_sin_d).to(tl.float32)

    y0 = (v0 * cos0) - (v1 * sin0)
    y1 = (v1 * cos1) + (v0 * sin1)
    # v2 e v3 passam sem rotação

    # 5. Salva diretamente no tensor de saída
    tl.store(out_ptr + pid * stride_oh + idx_32 * stride_od, y0.to(tl.bfloat16))
    tl.store(out_ptr + pid * stride_oh + (idx_32 + 32) * stride_od, y1.to(tl.bfloat16))
    tl.store(out_ptr + pid * stride_oh + (idx_64 + 64) * stride_od, v2.to(tl.bfloat16))
    tl.store(out_ptr + pid * stride_oh + (idx_128 + 128) * stride_od, v3.to(tl.bfloat16))

def triton_fused_rmsnorm_rope_qwen(x_states, weight, cos, sin, eps=1e-6):
    """
    x_states: (1, 1, H, 256)
    weight: (256,)
    cos, sin: (1, 1, 64) ou (1, 64)
    Retorna: (1, H, 1, 256)
    """
    H = x_states.shape[2]
    D = x_states.shape[3]
    assert D == 256, f"head_dim esperado 256, recebido {D}"
    
    out = torch.empty((1, H, 1, D), device=x_states.device, dtype=torch.bfloat16)
    x_2d = x_states.view(H, D)
    out_2d = out.view(H, D)
    cos_1d = cos.view(-1)
    sin_1d = sin.view(-1)
    
    grid = (H,)
    _fused_qwen35_rmsnorm_rope_kernel[grid](
        x_2d, weight, cos_1d, sin_1d, out_2d,
        x_2d.stride(0), x_2d.stride(1),
        out_2d.stride(0), out_2d.stride(1),
        cos_1d.stride(0), sin_1d.stride(0),
        eps=eps,
        num_warps=2
    )
    return out

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def ref_qwen_norm_rope(x_states, norm_weight, cos, sin, eps=1e-6):
    # x_states: (1, 1, H, 256)
    # 1. zero-centered RMSNorm
    x_f = x_states.float()
    norm = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
    norm = norm * (1.0 + norm_weight.float())
    norm = norm.to(torch.bfloat16).transpose(1, 2) # (1, H, 1, 256)
    
    # 2. partial RoPE
    cos_u = cos.unsqueeze(1) # (1, 1, 1, 64)
    sin_u = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1] # 64
    q_rot, q_pass = norm[..., :rotary_dim], norm[..., rotary_dim:]
    q_embed = (q_rot * cos_u) + (rotate_half(q_rot) * sin_u)
    return torch.cat([q_embed, q_pass], dim=-1)

def test_unit():
    print("Testando kernel unitário Triton para Qwen 3.5 Full Attention...")
    torch.manual_seed(42)
    device = "cuda"
    
    for H in [16, 2]: # Q tem 16 cabeças, K tem 2 cabeças
        x = torch.randn((1, 1, H, 256), device=device, dtype=torch.bfloat16)
        w = torch.zeros(256, device=device, dtype=torch.bfloat16) # zero-centered
        cos = torch.randn((1, 1, 64), device=device, dtype=torch.bfloat16)
        sin = torch.randn((1, 1, 64), device=device, dtype=torch.bfloat16)
        
        ref_out = ref_qwen_norm_rope(x, w, cos, sin)
        tri_out = triton_fused_rmsnorm_rope_qwen(x, w, cos, sin)
        
        diff = (ref_out - tri_out).abs().max().item()
        cos_sim = torch.nn.functional.cosine_similarity(ref_out.float().flatten(), tri_out.float().flatten(), dim=0).item()
        print(f"H={H}: Max Diff = {diff:.6e}, Cosine Sim = {cos_sim:.8f}")
        assert cos_sim > 0.9999, f"Similaridade insuficiente: {cos_sim}"

    print("\nMicro-benchmark de latência para as 10 camadas de Full Attention (Q e K por camada)...")
    # Para 10 camadas de Full Attention, temos por token:
    # 10 chamadas de Q (H=16) e 10 chamadas de K (H=2)
    q_x = torch.randn((1, 1, 16, 256), device=device, dtype=torch.bfloat16)
    q_w = torch.zeros(256, device=device, dtype=torch.bfloat16)
    k_x = torch.randn((1, 1, 2, 256), device=device, dtype=torch.bfloat16)
    k_w = torch.zeros(256, device=device, dtype=torch.bfloat16)
    cos = torch.randn((1, 1, 64), device=device, dtype=torch.bfloat16)
    sin = torch.randn((1, 1, 64), device=device, dtype=torch.bfloat16)

    # Warmup
    for _ in range(50):
        ref_qwen_norm_rope(q_x, q_w, cos, sin)
        ref_qwen_norm_rope(k_x, k_w, cos, sin)
        triton_fused_rmsnorm_rope_qwen(q_x, q_w, cos, sin)
        triton_fused_rmsnorm_rope_qwen(k_x, k_w, cos, sin)
    torch.cuda.synchronize()

    N = 200
    # PyTorch reference
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        for _ in range(10):
            q_out = ref_qwen_norm_rope(q_x, q_w, cos, sin)
            k_out = ref_qwen_norm_rope(k_x, k_w, cos, sin)
    torch.cuda.synchronize()
    t_ref_ms = (time.perf_counter() - t0) * 1000 / N

    # Triton Fused
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        for _ in range(10):
            q_out = triton_fused_rmsnorm_rope_qwen(q_x, q_w, cos, sin)
            k_out = triton_fused_rmsnorm_rope_qwen(k_x, k_w, cos, sin)
    torch.cuda.synchronize()
    t_tri_ms = (time.perf_counter() - t0) * 1000 / N

    print(f"PyTorch Original (10 camadas Q+K): {t_ref_ms:.3f} ms/token")
    print(f"Triton Fundido   (10 camadas Q+K): {t_tri_ms:.3f} ms/token")
    print(f"Economia isolada:                  {t_ref_ms - t_tri_ms:.3f} ms/token ({((t_ref_ms - t_tri_ms)/t_ref_ms)*100:.1f}% mais rápido)")

if __name__ == "__main__":
    import time
    test_unit()
