import torch
import triton
import triton.language as tl
import time

@triton.jit
def _fused_add_rmsnorm_zero_centered_kernel(
    x_ptr, delta_ptr, res_out_ptr, norm_out_ptr, w_ptr,
    eps: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    delta = tl.load(delta_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # 1. Soma residual
    res = x + delta
    tl.store(res_out_ptr + cols, res.to(tl.bfloat16), mask=mask)

    # 2. RMSNorm zero-centered sobre res
    var = tl.sum(res * res, axis=0) / D
    rsqrt_val = tl.rsqrt(var + eps)

    # Peso zero-centered: 1.0 + w
    w_eff = 1.0 + w
    normed = res * rsqrt_val * w_eff
    tl.store(norm_out_ptr + cols, normed.to(tl.bfloat16), mask=mask)

def triton_fused_add_rmsnorm(x, delta, weight, eps=1e-6):
    D = x.shape[-1]
    res_out = torch.empty_like(x)
    norm_out = torch.empty_like(x)
    BLOCK_D = triton.next_power_of_2(D)
    _fused_add_rmsnorm_zero_centered_kernel[(1,)](
        x, delta, res_out, norm_out, weight,
        eps=eps, D=D, BLOCK_D=BLOCK_D, num_warps=2
    )
    return res_out, norm_out

def ref_add_rmsnorm(x, delta, weight, eps=1e-6):
    res = x + delta
    var = res.float().pow(2).mean(-1, keepdim=True)
    normed = (res.float() * torch.rsqrt(var + eps)) * (1.0 + weight.float())
    return res, normed.to(x.dtype)

def test():
    print("Testando Fused Add + RMSNorm Zero-Centered...")
    torch.manual_seed(42)
    device = "cuda"
    D = 2048

    x = torch.randn(1, 1, D, device=device, dtype=torch.bfloat16)
    delta = torch.randn(1, 1, D, device=device, dtype=torch.bfloat16)
    w = torch.zeros(D, device=device, dtype=torch.bfloat16)

    ref_res, ref_norm = ref_add_rmsnorm(x, delta, w)
    tri_res, tri_norm = triton_fused_add_rmsnorm(x, delta, w)

    diff_res = (ref_res - tri_res).abs().max().item()
    diff_norm = (ref_norm - tri_norm).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(ref_norm.float().flatten(), tri_norm.float().flatten(), dim=0).item()

    print(f"Residual Max Diff: {diff_res:.6e}")
    print(f"Norm Max Diff:     {diff_norm:.6e}")
    print(f"Cosine Similarity: {cos_sim * 100:.6f}%")
    assert cos_sim > 0.99999, "Similaridade insuficiente!"

    # Benchmark: 80 operações por token (2 por camada x 40 camadas)
    # Warmup
    for _ in range(50):
        ref_add_rmsnorm(x, delta, w)
        triton_fused_add_rmsnorm(x, delta, w)
    torch.cuda.synchronize()

    N = 200
    # PyTorch ref
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        for _ in range(80):
            r, n = ref_add_rmsnorm(x, delta, w)
    torch.cuda.synchronize()
    t_ref_ms = (time.perf_counter() - t0) * 1000 / N

    # Triton fused
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        for _ in range(80):
            r, n = triton_fused_add_rmsnorm(x, delta, w)
    torch.cuda.synchronize()
    t_tri_ms = (time.perf_counter() - t0) * 1000 / N

    print("\n" + "=" * 60)
    print("BENCHMARK PARA 80 RESIDUAL ADDS + RMSNORMS (40 CAMADAS):")
    print("=" * 60)
    print(f"PyTorch Original (80x): {t_ref_ms:.3f} ms/token")
    print(f"Triton Fundido   (80x): {t_tri_ms:.3f} ms/token")
    print(f"Economia Estimada:      {t_ref_ms - t_tri_ms:.3f} ms/token ({((t_ref_ms - t_tri_ms)/t_ref_ms)*100:.1f}% mais rápido)")
    print("=" * 60)

if __name__ == "__main__":
    test()
