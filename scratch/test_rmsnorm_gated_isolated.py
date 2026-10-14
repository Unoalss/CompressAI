import sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:
    HAVE_TRITON = False

if HAVE_TRITON:
    @triton.jit
    def _triton_rmsnorm_gated_kernel(
        x_ptr, gate_ptr, w_ptr, out_ptr,
        stride_xm, stride_xd,
        stride_gm, stride_gd,
        stride_om, stride_od,
        eps: tl.constexpr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        cols = tl.arange(0, BLOCK_D)
        mask = cols < D

        off_x = pid * stride_xm + cols * stride_xd
        off_g = pid * stride_gm + cols * stride_gd
        off_o = pid * stride_om + cols * stride_od

        x = tl.load(x_ptr + off_x, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.bfloat16)
        g = tl.load(gate_ptr + off_g, mask=mask, other=0.0).to(tl.float32)

        # 1. Variância em float32
        var = tl.sum(x * x, axis=0) / D
        rsqrt_val = tl.math.rsqrt(var + eps)

        # 2. Normalização e cast para bfloat16
        norm_x = (x * rsqrt_val).to(tl.bfloat16)
        weighted = norm_x * w

        # 3. SiLU do gate em float32: g * sigmoid(g)
        silu_g = g * tl.sigmoid(g)

        # 4. Produto final em float32 convertido para bfloat16
        out = (weighted.to(tl.float32) * silu_g).to(tl.bfloat16)
        tl.store(out_ptr + off_o, out, mask=mask)

def triton_rmsnorm_gated(hidden_states: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    D = hidden_states.shape[-1]
    orig_shape = hidden_states.shape
    x_2d = hidden_states.view(-1, D)
    g_2d = gate.view(-1, D)
    M = x_2d.shape[0]

    out = torch.empty_like(x_2d)
    grid = (M,)
    BLOCK_D = triton.next_power_of_2(D)

    _triton_rmsnorm_gated_kernel[grid](
        x_2d, g_2d, weight, out,
        x_2d.stride(0), x_2d.stride(1),
        g_2d.stride(0), g_2d.stride(1),
        out.stride(0), out.stride(1),
        eps=eps, D=D, BLOCK_D=BLOCK_D,
        num_warps=2
    )
    return out.view(orig_shape)

def pytorch_rmsnorm_gated(hidden_states: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hs_f32 = hidden_states.to(torch.float32)
    variance = hs_f32.pow(2).mean(-1, keepdim=True)
    hs_norm = hs_f32 * torch.rsqrt(variance + eps)
    hs_w = weight * hs_norm.to(input_dtype)
    gate_act = F.silu(gate.to(torch.float32))
    return (hs_w * gate_act).to(input_dtype)

def test_isolated_rmsnorm_gated():
    print("=" * 80)
    print("TESTE NUMÉRICO E DE LATÊNCIA ISOLADA: RMSNormGated (Triton vs PyTorch)")
    print("=" * 80)

    assert HAVE_TRITON, "Triton não está disponível!"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "CUDA é obrigatório!"

    # Dimensões exatas do Qwen 3.5 35B DeltaNet durante decode
    # 32 cabeças, cada uma com 128 dimensões
    M = 32
    D = 128
    eps = 1e-6

    torch.manual_seed(42)
    # Testa 10 sementes diferentes para rigor estatístico
    max_diffs = []
    cos_sims = []

    for seed in range(10):
        torch.manual_seed(seed * 100 + 7)
        x = torch.randn((M, D), device=device, dtype=torch.bfloat16)
        gate = torch.randn((M, D), device=device, dtype=torch.bfloat16)
        w = torch.randn(D, device=device, dtype=torch.bfloat16)

        out_py = pytorch_rmsnorm_gated(x, gate, w, eps=eps)
        out_tr = triton_rmsnorm_gated(x, gate, w, eps=eps)

        assert not torch.isnan(out_tr).any(), "NaN detectado na saída do Triton!"
        assert not torch.isinf(out_tr).any(), "Inf detectado na saída do Triton!"

        diff = (out_py.float() - out_tr.float()).abs().max().item()
        cos = F.cosine_similarity(out_py.float().view(-1), out_tr.float().view(-1), dim=0).item()
        max_diffs.append(diff)
        cos_sims.append(cos)

    print("Resultados de Fidelidade Numérica (10 baterias aleatórias):")
    print(f"  Erro absoluto máximo:     {max(max_diffs):.6f}")
    print(f"  Similaridade Cosseno:     {min(cos_sims) * 100:.6f}%")
    print(f"  Ausência de NaNs/Infs:    100% LIMPO")

    # Benchmark de Latência Micro-operacional isolada
    # Simulando 30 camadas (30 chamadas sequenciais por token)
    N_LAYERS = 30
    N_ITERS = 1000

    x_list = [torch.randn((M, D), device=device, dtype=torch.bfloat16) for _ in range(N_LAYERS)]
    g_list = [torch.randn((M, D), device=device, dtype=torch.bfloat16) for _ in range(N_LAYERS)]
    w_list = [torch.randn(D, device=device, dtype=torch.bfloat16) for _ in range(N_LAYERS)]

    # Warmup
    for i in range(100):
        for l in range(N_LAYERS):
            _ = pytorch_rmsnorm_gated(x_list[l], g_list[l], w_list[l], eps=eps)
            _ = triton_rmsnorm_gated(x_list[l], g_list[l], w_list[l], eps=eps)
    torch.cuda.synchronize()

    # Mede PyTorch (30 chamadas = 1 token equivalente)
    ev_s = torch.cuda.Event(enable_timing=True)
    ev_e = torch.cuda.Event(enable_timing=True)
    ev_s.record()
    for _ in range(N_ITERS):
        for l in range(N_LAYERS):
            _ = pytorch_rmsnorm_gated(x_list[l], g_list[l], w_list[l], eps=eps)
    ev_e.record()
    torch.cuda.synchronize()
    time_py_ms = ev_s.elapsed_time(ev_e) / N_ITERS

    # Mede Triton (30 chamadas = 1 token equivalente)
    ev_s.record()
    for _ in range(N_ITERS):
        for l in range(N_LAYERS):
            _ = triton_rmsnorm_gated(x_list[l], g_list[l], w_list[l], eps=eps)
    ev_e.record()
    torch.cuda.synchronize()
    time_tr_ms = ev_s.elapsed_time(ev_e) / N_ITERS

    print("\nBenchmark Micro-operacional (Custo acumulado nas 30 camadas DeltaNet):")
    print(f"  PyTorch Original (10 ops/camada = 300 kernels):  {time_py_ms:.3f} ms/token ({time_py_ms / 30 * 1000:.1f} us/camada)")
    print(f"  Triton Fundido   (1 kernel/camada = 30 kernels): {time_tr_ms:.3f} ms/token ({time_tr_ms / 30 * 1000:.1f} us/camada)")
    print(f"  Economia Isolada:                                -{time_py_ms - time_tr_ms:.3f} ms/token ({((time_py_ms - time_tr_ms)/time_py_ms)*100:.1f}% mais rápido!)")
    print("=" * 80)

if __name__ == "__main__":
    test_isolated_rmsnorm_gated()
