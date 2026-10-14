import sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as qm

def profile_full_attention_micro():
    print("=" * 80)
    print("FASE 4C - PASSO 1: DIAGNÓSTICO GRANULAR DA FULL ATTENTION (10 CAMADAS)")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "CUDA é obrigatório!"

    # Dimensões da Full Attention do Qwen 3.5 35B
    # 10 camadas
    # Q: 16 cabeças x 128 head_dim
    # K: 2 cabeças x 128 head_dim
    # V: 2 cabeças x 128 head_dim
    N_LAYERS = 10
    N_HEADS_Q = 16
    N_HEADS_KV = 2
    HEAD_DIM = 128
    N_ITERS = 1000

    torch.manual_seed(42)
    # Tensores para as 10 camadas simulando entrada após QKV proj no decode (batch=1, seq=1)
    q_list = [torch.randn((1, 1, N_HEADS_Q, HEAD_DIM), device=device, dtype=torch.bfloat16) for _ in range(N_LAYERS)]
    k_list = [torch.randn((1, 1, N_HEADS_KV, HEAD_DIM), device=device, dtype=torch.bfloat16) for _ in range(N_LAYERS)]
    w_q_norm = [torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16) for _ in range(N_LAYERS)]
    w_k_norm = [torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16) for _ in range(N_LAYERS)]

    # Position embeddings (RoPE): cos e sin para 1 token
    cos = torch.randn((1, 1, HEAD_DIM), device=device, dtype=torch.bfloat16)
    sin = torch.randn((1, 1, HEAD_DIM), device=device, dtype=torch.bfloat16)

    # 1. Medição isolada de cada sub-etapa
    print("[1/3] Medindo custo isolado de cada sub-etapa em 10 camadas (1.000 iterações)...")

    # A. Custo de q_norm (16 cabeças de 128)
    ev_s = torch.cuda.Event(enable_timing=True); ev_e = torch.cuda.Event(enable_timing=True)
    ev_s.record()
    for _ in range(N_ITERS):
        for l in range(N_LAYERS):
            q = q_list[l]
            var = q.float().pow(2).mean(-1, keepdim=True)
            _ = (q.float() * torch.rsqrt(var + 1e-6)).bfloat16() * w_q_norm[l]
    ev_e.record(); torch.cuda.synchronize()
    t_q_norm_ms = ev_s.elapsed_time(ev_e) / N_ITERS

    # B. Custo de k_norm (2 cabeças de 128)
    ev_s.record()
    for _ in range(N_ITERS):
        for l in range(N_LAYERS):
            k = k_list[l]
            var = k.float().pow(2).mean(-1, keepdim=True)
            _ = (k.float() * torch.rsqrt(var + 1e-6)).bfloat16() * w_k_norm[l]
    ev_e.record(); torch.cuda.synchronize()
    t_k_norm_ms = ev_s.elapsed_time(ev_e) / N_ITERS

    # C. Custo de transpose(1, 2) em Q e K
    q_normed = q_list[0]
    k_normed = k_list[0]
    ev_s.record()
    for _ in range(N_ITERS):
        for l in range(N_LAYERS):
            _ = q_normed.transpose(1, 2)
            _ = k_normed.transpose(1, 2)
    ev_e.record(); torch.cuda.synchronize()
    t_trans_ms = ev_s.elapsed_time(ev_e) / N_ITERS

    # D. Custo de apply_rotary_pos_emb original
    q_trans = q_normed.transpose(1, 2)
    k_trans = k_normed.transpose(1, 2)
    ev_s.record()
    for _ in range(N_ITERS):
        for l in range(N_LAYERS):
            _ = qm.apply_rotary_pos_emb(q_trans, k_trans, cos, sin)
    ev_e.record(); torch.cuda.synchronize()
    t_rope_ms = ev_s.elapsed_time(ev_e) / N_ITERS

    # E. Pipeline completo original: q_norm + k_norm + transposes + RoPE
    ev_s.record()
    for _ in range(N_ITERS):
        for l in range(N_LAYERS):
            # q_norm
            q = q_list[l]
            var_q = q.float().pow(2).mean(-1, keepdim=True)
            qn = ((q.float() * torch.rsqrt(var_q + 1e-6)).bfloat16() * w_q_norm[l]).transpose(1, 2)
            # k_norm
            k = k_list[l]
            var_k = k.float().pow(2).mean(-1, keepdim=True)
            kn = ((k.float() * torch.rsqrt(var_k + 1e-6)).bfloat16() * w_k_norm[l]).transpose(1, 2)
            # RoPE
            _, _ = qm.apply_rotary_pos_emb(qn, kn, cos, sin)
    ev_e.record(); torch.cuda.synchronize()
    t_total_orig_ms = ev_s.elapsed_time(ev_e) / N_ITERS

    print("\n" + "=" * 80)
    print("RELATÓRIO DO DIAGNÓSTICO GRANULAR DA FULL ATTENTION (10 CAMADAS / TOKEN):")
    print("=" * 80)
    print(f"1. q_norm (16 cabeças x 128 dims):        {t_q_norm_ms:.3f} ms/tok ({t_q_norm_ms / 10 * 1000:.1f} us/cam)")
    print(f"2. k_norm (2 cabeças x 128 dims):         {t_k_norm_ms:.3f} ms/tok ({t_k_norm_ms / 10 * 1000:.1f} us/cam)")
    print(f"3. Transposes (reshape para [1, H, 1, D]):{t_trans_ms:.3f} ms/tok ({t_trans_ms / 10 * 1000:.1f} us/cam)")
    print(f"4. apply_rotary_pos_emb (RoPE):           {t_rope_ms:.3f} ms/tok ({t_rope_ms / 10 * 1000:.1f} us/cam)")
    print("--------------------------------------------------------------------------------")
    print(f"TOTAL PIPELINE ORIGINAL (Q/K Norm + RoPE):{t_total_orig_ms:.3f} ms/tok ({t_total_orig_ms / 10 * 1000:.1f} us/cam)")
    print("=" * 80)

if __name__ == "__main__":
    profile_full_attention_micro()
