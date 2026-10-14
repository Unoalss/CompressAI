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
    def _fused_delta_gates_kernel(
        b_ptr, a_ptr, A_exp_ptr, dt_bias_ptr, beta_ptr, g_ptr,
        N: tl.constexpr, BLOCK: tl.constexpr
    ):
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        
        b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        a = tl.load(a_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        A_exp = tl.load(A_exp_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        dt_bias = tl.load(dt_bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        # beta = sigmoid(b)
        beta = tl.sigmoid(b)
        tl.store(beta_ptr + cols, beta.to(tl.bfloat16), mask=mask)

        # g = -A_exp * softplus(a + dt_bias)
        # softplus(x) = log(1 + exp(x))
        x = a + dt_bias
        # Estabilidade numérica clássica para softplus: se x > 20, softplus(x) ≈ x
        sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
        g = -A_exp * sp
        tl.store(g_ptr + cols, g, mask=mask)

def fused_delta_gates_triton(b, a, A_exp, dt_bias):
    N = b.shape[-1]
    beta = torch.empty_like(b)
    g = torch.empty((b.shape[0], b.shape[1], N), device=b.device, dtype=torch.float32)
    _fused_delta_gates_kernel[(1,)](
        b, a, A_exp, dt_bias, beta, g, N=N, BLOCK=64, num_warps=1
    )
    return beta, g

def profile_gates_micro():
    print("=" * 80)
    print("AUDITORIA CIRÚRGICA: DECOMPOSIÇÃO DOS GATES E OPERAÇÕES AUXILIARES DELTANET")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "CUDA é obrigatório!"

    N_HEADS_V = 32
    N_HEADS_K = 16
    HEAD_DIM = 128
    N_LAYERS = 30
    N_ITERS = 1000

    torch.manual_seed(42)
    # Tensores para as 30 camadas
    b_list = [torch.randn((1, 1, N_HEADS_V), device=device, dtype=torch.bfloat16) for _ in range(N_LAYERS)]
    a_list = [torch.randn((1, 1, N_HEADS_V), device=device, dtype=torch.bfloat16) for _ in range(N_LAYERS)]
    A_log_list = [torch.empty(N_HEADS_V, device=device).uniform_(0.01, 16).log() for _ in range(N_LAYERS)]
    A_exp_list = [A_log.float().exp().contiguous() for A_log in A_log_list]
    dt_bias_list = [torch.ones(N_HEADS_V, device=device) for _ in range(N_LAYERS)]

    # Tensores de Q, K, V
    mixed_qkv_list = [torch.randn((1, 8192, 1), device=device, dtype=torch.bfloat16) for _ in range(N_LAYERS)]

    print("[1/4] Medindo custo isolado de cada sub-operação em 30 camadas (1.000 iterações)...")

    # A. Custo de beta = b.sigmoid()
    ev_s = torch.cuda.Event(enable_timing=True); ev_e = torch.cuda.Event(enable_timing=True)
    ev_s.record()
    for _ in range(N_ITERS):
        for l in range(N_LAYERS):
            _ = b_list[l].sigmoid()
    ev_e.record(); torch.cuda.synchronize()
    t_beta_ms = ev_s.elapsed_time(ev_e) / N_ITERS

    # B. Custo de A_log.float().exp() repetido
    ev_s.record()
    for _ in range(N_ITERS):
        for l in range(N_LAYERS):
            _ = A_log_list[l].float().exp()
    ev_e.record(); torch.cuda.synchronize()
    t_exp_ms = ev_s.elapsed_time(ev_e) / N_ITERS

    # C. Custo de softplus(a.float() + dt_bias)
    ev_s.record()
    for _ in range(N_ITERS):
        for l in range(N_LAYERS):
            _ = F.softplus(a_list[l].float() + dt_bias_list[l])
    ev_e.record(); torch.cuda.synchronize()
    t_sp_ms = ev_s.elapsed_time(ev_e) / N_ITERS

    # D. Custo de g = -A_exp * softplus
    ev_s.record()
    for _ in range(N_ITERS):
        for l in range(N_LAYERS):
            sp = F.softplus(a_list[l].float() + dt_bias_list[l])
            _ = -A_log_list[l].float().exp() * sp
    ev_e.record(); torch.cuda.synchronize()
    t_g_orig_ms = ev_s.elapsed_time(ev_e) / N_ITERS

    # E. Custo de torch.split + reshapes de Q, K, V
    ev_s.record()
    for _ in range(N_ITERS):
        for l in range(N_LAYERS):
            m = mixed_qkv_list[l].transpose(1, 2)
            q, k, v = torch.split(m, [2048, 2048, 4096], dim=-1)
            q = q.reshape(1, 1, -1, HEAD_DIM)
            k = k.reshape(1, 1, -1, HEAD_DIM)
            v = v.reshape(1, 1, -1, HEAD_DIM)
    ev_e.record(); torch.cuda.synchronize()
    t_split_ms = ev_s.elapsed_time(ev_e) / N_ITERS

    # F. Custo de repeat_interleave(2, dim=2)
    q_sample = torch.randn((1, 1, 16, 128), device=device, dtype=torch.bfloat16)
    k_sample = torch.randn((1, 1, 16, 128), device=device, dtype=torch.bfloat16)
    ev_s.record()
    for _ in range(N_ITERS):
        for l in range(N_LAYERS):
            _ = q_sample.repeat_interleave(2, dim=2)
            _ = k_sample.repeat_interleave(2, dim=2)
    ev_e.record(); torch.cuda.synchronize()
    t_repeat_ms = ev_s.elapsed_time(ev_e) / N_ITERS

    # G. Alternativa rápida ao repeat_interleave (expand + reshape / view)
    ev_s.record()
    for _ in range(N_ITERS):
        for l in range(N_LAYERS):
            # repeat_interleave via expand
            _ = q_sample.unsqueeze(3).expand(1, 1, 16, 2, 128).reshape(1, 1, 32, 128)
            _ = k_sample.unsqueeze(3).expand(1, 1, 16, 2, 128).reshape(1, 1, 32, 128)
    ev_e.record(); torch.cuda.synchronize()
    t_expand_ms = ev_s.elapsed_time(ev_e) / N_ITERS

    # H. Custo de Fused Gates (Triton)
    ev_s.record()
    for _ in range(N_ITERS):
        for l in range(N_LAYERS):
            _ = fused_delta_gates_triton(b_list[l], a_list[l], A_exp_list[l], dt_bias_list[l])
    ev_e.record(); torch.cuda.synchronize()
    t_fused_gates_ms = ev_s.elapsed_time(ev_e) / N_ITERS

    # I. Validação numérica de Fused Gates vs PyTorch
    print("\n[2/4] Validação de equivalência numérica dos Gates (Triton vs PyTorch):")
    max_err_beta = 0.0
    max_err_g = 0.0
    for l in range(N_LAYERS):
        beta_py = b_list[l].sigmoid()
        g_py = -A_log_list[l].float().exp() * F.softplus(a_list[l].float() + dt_bias_list[l])
        beta_tr, g_tr = fused_delta_gates_triton(b_list[l], a_list[l], A_exp_list[l], dt_bias_list[l])

        err_b = (beta_py.float() - beta_tr.float()).abs().max().item()
        err_g = (g_py - g_tr).abs().max().item()
        max_err_beta = max(max_err_beta, err_b)
        max_err_g = max(max_err_g, err_g)

    print(f"  Erro absoluto máximo em beta: {max_err_beta:.8f}")
    print(f"  Erro absoluto máximo em g:    {max_err_g:.8f}")

    print("\n" + "=" * 80)
    print("RELATÓRIO DA DECOMPOSIÇÃO DOS GATES E OPERAÇÕES AUXILIARES (30 CAMADAS / TOKEN):")
    print("=" * 80)
    print(f"1. Operações de Gates (Original):")
    print(f"   - beta = b.sigmoid():                     {t_beta_ms:.3f} ms/tok ({t_beta_ms/30*1000:.1f} us/cam)")
    print(f"   - A_log.exp() repetido:                   {t_exp_ms:.3f} ms/tok ({t_exp_ms/30*1000:.1f} us/cam)")
    print(f"   - softplus(a + dt_bias):                  {t_sp_ms:.3f} ms/tok ({t_sp_ms/30*1000:.1f} us/cam)")
    print(f"   - g completo original (4 ops):            {t_g_orig_ms:.3f} ms/tok ({t_g_orig_ms/30*1000:.1f} us/cam)")
    print(f"   Total Gates Original (beta + g):          {t_beta_ms + t_g_orig_ms:.3f} ms/tok")
    print(f"   Total Gates Triton Fundido:               {t_fused_gates_ms:.3f} ms/tok")
    print(f"   -> Ganho nos Gates:                       -{ (t_beta_ms + t_g_orig_ms) - t_fused_gates_ms:.3f} ms/tok")
    print("--------------------------------------------------------------------------------")
    print(f"2. Operações de Fatiamento e Repetição de Q, K, V:")
    print(f"   - split + transposes + reshapes:          {t_split_ms:.3f} ms/tok ({t_split_ms/30*1000:.1f} us/cam)")
    print(f"   - repeat_interleave(2) (Q e K):           {t_repeat_ms:.3f} ms/tok ({t_repeat_ms/30*1000:.1f} us/cam)")
    print(f"   - expand + reshape (Zero-Copy):           {t_expand_ms:.3f} ms/tok ({t_expand_ms/30*1000:.1f} us/cam)")
    print(f"   -> Ganho no repeat_interleave:            -{t_repeat_ms - t_expand_ms:.3f} ms/tok")
    print("--------------------------------------------------------------------------------")
    print(f"POTENCIAL LÍQUIDO COMBINADO DO PASSO 3:      -{((t_beta_ms + t_g_orig_ms) - t_fused_gates_ms) + (t_repeat_ms - t_expand_ms):.3f} ms/tok")
    print("=" * 80)

if __name__ == "__main__":
    profile_gates_micro()
