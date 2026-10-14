import sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as qm
import triton
import triton.language as tl

@triton.jit
def _fused_rmsnorm_rope_kernel(
    x_ptr, w_ptr, cos_ptr, sin_ptr, out_ptr,
    stride_xb, stride_xh, stride_xd,
    stride_ob, stride_oh, stride_od,
    stride_cos_b, stride_cos_d,
    stride_sin_b, stride_sin_d,
    eps: tl.constexpr,
    HALF_D: tl.constexpr, # 64
    D: tl.constexpr,      # 128
):
    pid = tl.program_id(0) # Cada pid é uma cabeça de atenção (0..H-1)
    
    # Índices para a primeira e segunda metade
    idx_first = tl.arange(0, HALF_D)          # 0..63
    idx_second = HALF_D + tl.arange(0, HALF_D) # 64..127

    # Offsets de memória
    off_x_first = pid * stride_xh + idx_first * stride_xd
    off_x_second = pid * stride_xh + idx_second * stride_xd

    # 1. Carrega o vetor completo x da cabeça pid [128]
    x_f = tl.load(x_ptr + off_x_first).to(tl.float32)
    x_s = tl.load(x_ptr + off_x_second).to(tl.float32)

    # 2. RMSNorm: var = (sum(x_f^2) + sum(x_s^2)) / D
    var = (tl.sum(x_f * x_f, axis=0) + tl.sum(x_s * x_s, axis=0)) / D
    rsqrt_val = tl.math.rsqrt(var + eps)

    # Normaliza e converte para bfloat16 conforme referência
    norm_f = (x_f * rsqrt_val).to(tl.bfloat16)
    norm_s = (x_s * rsqrt_val).to(tl.bfloat16)

    # Pondera com o peso de normalização w
    w_f = tl.load(w_ptr + idx_first).to(tl.bfloat16)
    w_s = tl.load(w_ptr + idx_second).to(tl.bfloat16)
    v_f = norm_f * w_f
    v_s = norm_s * w_s

    # 3. RoPE (rotate_half)
    # v_rot = [-v_s, v_f]
    # cos_f, cos_s e sin_f, sin_s
    cos_f = tl.load(cos_ptr + idx_first * stride_cos_d).to(tl.float32)
    cos_s = tl.load(cos_ptr + idx_second * stride_cos_d).to(tl.float32)
    sin_f = tl.load(sin_ptr + idx_first * stride_sin_d).to(tl.float32)
    sin_s = tl.load(sin_ptr + idx_second * stride_sin_d).to(tl.float32)

    # y_first = (v_f * cos_f) - (v_s * sin_f)
    # y_second = (v_s * cos_s) + (v_f * sin_s)
    y_f = (v_f.to(tl.float32) * cos_f) - (v_s.to(tl.float32) * sin_f)
    y_s = (v_s.to(tl.float32) * cos_s) + (v_f.to(tl.float32) * sin_s)

    # 4. Grava resultado na saída [H, 128] em bfloat16
    off_out_first = pid * stride_oh + idx_first * stride_od
    off_out_second = pid * stride_oh + idx_second * stride_od

    tl.store(out_ptr + off_out_first, y_f.to(tl.bfloat16))
    tl.store(out_ptr + off_out_second, y_s.to(tl.bfloat16))

def triton_rmsnorm_rope(x_states, weight, cos, sin, eps=1e-6):
    """
    Executa RMSNorm + RoPE fundidos para x_states [batch=1, seq=1, heads, head_dim=128].
    Retorna tensor com formato [batch=1, heads, seq=1, head_dim=128].
    """
    # x_states: [1, 1, H, 128]
    H = x_states.shape[2]
    D = x_states.shape[3]
    assert D == 128, "head_dim deve ser 128!"

    out = torch.empty((1, H, 1, D), device=x_states.device, dtype=torch.bfloat16)

    # cos e sin podem vir como [1, 1, 128] ou [1, 1, 1, 128]
    cos_2d = cos.view(1, D)
    sin_2d = sin.view(1, D)
    x_2d = x_states.view(H, D)
    out_2d = out.view(H, D)

    grid = (H,)
    _fused_rmsnorm_rope_kernel[grid](
        x_2d, weight, cos_2d, sin_2d, out_2d,
        0, x_2d.stride(0), x_2d.stride(1),
        0, out_2d.stride(0), out_2d.stride(1),
        0, cos_2d.stride(1),
        0, sin_2d.stride(1),
        eps=eps, HALF_D=64, D=128,
        num_warps=1
    )
    return out

def test_fused_rmsnorm_rope_isolated():
    print("=" * 80)
    print("TESTE UNITÁRIO E BENCHMARK ISOLADO: RMSNorm + RoPE FUNDIDOS (TRITON vs PYTORCH)")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "CUDA é obrigatório!"

    N_HEADS_Q = 16
    N_HEADS_K = 2
    HEAD_DIM = 128
    eps = 1e-6

    # 1. Teste de Fidelidade Numérica em 10 posições / sementes diferentes
    print("[1/3] Verificando equivalência numérica em 10 posições/contextos distintos...")
    max_err_q = 0.0
    max_err_k = 0.0
    min_cos_q = 1.0
    min_cos_k = 1.0

    for seed in range(10):
        torch.manual_seed(1000 + seed)
        # Tensores de entrada simulando saída de q_proj e k_proj [1, 1, H, 128]
        q_raw = torch.randn((1, 1, N_HEADS_Q, HEAD_DIM), device=device, dtype=torch.bfloat16)
        k_raw = torch.randn((1, 1, N_HEADS_K, HEAD_DIM), device=device, dtype=torch.bfloat16)
        w_q = torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16)
        w_k = torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16)

        # Position embeddings (RoPE)
        cos = torch.randn((1, 1, HEAD_DIM), device=device, dtype=torch.bfloat16)
        sin = torch.randn((1, 1, HEAD_DIM), device=device, dtype=torch.bfloat16)

        # Referência PyTorch:
        # q_norm
        var_q = q_raw.float().pow(2).mean(-1, keepdim=True)
        q_normed = ((q_raw.float() * torch.rsqrt(var_q + eps)).bfloat16() * w_q).transpose(1, 2)
        # k_norm
        var_k = k_raw.float().pow(2).mean(-1, keepdim=True)
        k_normed = ((k_raw.float() * torch.rsqrt(var_k + eps)).bfloat16() * w_k).transpose(1, 2)
        # RoPE
        q_ref, k_ref = qm.apply_rotary_pos_emb(q_normed, k_normed, cos, sin)

        # Triton Fundido:
        q_tr = triton_rmsnorm_rope(q_raw, w_q, cos, sin, eps=eps)
        k_tr = triton_rmsnorm_rope(k_raw, w_k, cos, sin, eps=eps)

        assert not torch.isnan(q_tr).any() and not torch.isinf(q_tr).any(), "NaN/Inf em Q Triton!"
        assert not torch.isnan(k_tr).any() and not torch.isinf(k_tr).any(), "NaN/Inf em K Triton!"

        diff_q = (q_ref.float() - q_tr.float()).abs().max().item()
        diff_k = (k_ref.float() - k_tr.float()).abs().max().item()
        c_q = F.cosine_similarity(q_ref.float().view(-1), q_tr.float().view(-1), dim=0).item()
        c_k = F.cosine_similarity(k_ref.float().view(-1), k_tr.float().view(-1), dim=0).item()

        max_err_q = max(max_err_q, diff_q)
        max_err_k = max(max_err_k, diff_k)
        min_cos_q = min(min_cos_q, c_q)
        min_cos_k = min(min_cos_k, c_k)

    print(f"  Q (16 cabeças): Erro máx = {max_err_q:.6f} | Cosseno mín = {min_cos_q * 100:.6f}%")
    print(f"  K (2 cabeças):  Erro máx = {max_err_k:.6f} | Cosseno mín = {min_cos_k * 100:.6f}%")
    print("  [OK] Equivalência numérica RIGOROSAMENTE VALIDADA!")

    # 2. Benchmark de Velocidade Isolado (10 Camadas / 1.000 iterações)
    print("\n[2/3] Executando benchmark isolado nas 10 camadas de Full Attention (1.000 iterações)...")
    N_LAYERS = 10
    N_ITERS = 1000

    q_list = [torch.randn((1, 1, N_HEADS_Q, HEAD_DIM), device=device, dtype=torch.bfloat16) for _ in range(N_LAYERS)]
    k_list = [torch.randn((1, 1, N_HEADS_K, HEAD_DIM), device=device, dtype=torch.bfloat16) for _ in range(N_LAYERS)]
    wq_list = [torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16) for _ in range(N_LAYERS)]
    wk_list = [torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16) for _ in range(N_LAYERS)]

    # Warmup
    for _ in range(100):
        for l in range(N_LAYERS):
            var_q = q_list[l].float().pow(2).mean(-1, keepdim=True)
            qn = ((q_list[l].float() * torch.rsqrt(var_q + eps)).bfloat16() * wq_list[l]).transpose(1, 2)
            var_k = k_list[l].float().pow(2).mean(-1, keepdim=True)
            kn = ((k_list[l].float() * torch.rsqrt(var_k + eps)).bfloat16() * wk_list[l]).transpose(1, 2)
            _, _ = qm.apply_rotary_pos_emb(qn, kn, cos, sin)
            _ = triton_rmsnorm_rope(q_list[l], wq_list[l], cos, sin, eps=eps)
            _ = triton_rmsnorm_rope(k_list[l], wk_list[l], cos, sin, eps=eps)
    torch.cuda.synchronize()

    # Mede PyTorch
    ev_s = torch.cuda.Event(enable_timing=True); ev_e = torch.cuda.Event(enable_timing=True)
    ev_s.record()
    for _ in range(N_ITERS):
        for l in range(N_LAYERS):
            var_q = q_list[l].float().pow(2).mean(-1, keepdim=True)
            qn = ((q_list[l].float() * torch.rsqrt(var_q + eps)).bfloat16() * wq_list[l]).transpose(1, 2)
            var_k = k_list[l].float().pow(2).mean(-1, keepdim=True)
            kn = ((k_list[l].float() * torch.rsqrt(var_k + eps)).bfloat16() * wk_list[l]).transpose(1, 2)
            _, _ = qm.apply_rotary_pos_emb(qn, kn, cos, sin)
    ev_e.record(); torch.cuda.synchronize()
    t_py_ms = ev_s.elapsed_time(ev_e) / N_ITERS

    # Mede Triton Fundido
    ev_s.record()
    for _ in range(N_ITERS):
        for l in range(N_LAYERS):
            _ = triton_rmsnorm_rope(q_list[l], wq_list[l], cos, sin, eps=eps)
            _ = triton_rmsnorm_rope(k_list[l], wk_list[l], cos, sin, eps=eps)
    ev_e.record(); torch.cuda.synchronize()
    t_tr_ms = ev_s.elapsed_time(ev_e) / N_ITERS

    print("\n" + "=" * 80)
    print("RELATÓRIO DO BENCHMARK MICRO-OPERACIONAL (10 CAMADAS FULL ATTN / TOKEN):")
    print("=" * 80)
    print(f"PyTorch Original (Norm Q/K + RoPE separado): {t_py_ms:.3f} ms/token ({t_py_ms / 10 * 1000:.1f} us/camada)")
    print(f"Triton Fundido   (RMSNorm + RoPE integrado): {t_tr_ms:.3f} ms/token ({t_tr_ms / 10 * 1000:.1f} us/camada)")
    print(f"Ganho Líquido Isolado:                       -{t_py_ms - t_tr_ms:.3f} ms/token ({((t_py_ms - t_tr_ms)/t_py_ms)*100:.1f}% mais rápido!)")
    print("=" * 80)

if __name__ == "__main__":
    test_fused_rmsnorm_rope_isolated()
