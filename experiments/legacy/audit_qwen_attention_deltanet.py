import os
import torch
from safetensors import safe_open
from test_qwen_ffn_single_layer import quantize_matrix_h5a_batched

MODEL_DIR = "E:/agente/modelo"
SAFETENSORS_PATH = os.path.join(MODEL_DIR, "model.safetensors-00001-of-00001.safetensors")

print("=" * 90)
print("FASE 4: DIAGNÓSTICO E VIABILIDADE DE QUANTIZAÇÃO: GATED DELTANET & FULL ATTENTION")
print("=" * 90)

with safe_open(SAFETENSORS_PATH, framework="pt", device="cpu") as f:
    # 1. Inspecionar Gated DeltaNet (Camada 0)
    print("\n--- 1. ANÁLISE DETALHADA DO GATED DELTANET (Camada 0) ---")
    dnet_keys = [k for k in f.keys() if "layers.0.linear_attn" in k]
    for k in sorted(dnet_keys):
        t = f.get_tensor(k)
        print(f"{k.split('.')[-1]:<15} | Shape: {str(list(t.shape)):<16} | Dtype: {str(t.dtype):<15} | Mean: {t.float().mean().item():+.5f} | Std: {t.float().std().item():.5f} | Min/Max: [{t.float().min().item():.3f}, {t.float().max().item():.3f}]")
        
    # 2. Inspecionar Full Attention (Camada 3)
    print("\n--- 2. ANÁLISE DETALHADA DO FULL ATTENTION (Camada 3) ---")
    attn_keys = [k for k in f.keys() if "layers.3.self_attn" in k]
    for k in sorted(attn_keys):
        t = f.get_tensor(k)
        print(f"{k.split('.')[-1]:<15} | Shape: {str(list(t.shape)):<16} | Dtype: {str(t.dtype):<15} | Mean: {t.float().mean().item():+.5f} | Std: {t.float().std().item():.5f} | Min/Max: [{t.float().min().item():.3f}, {t.float().max().item():.3f}]")

    # 3. Teste de Quantização H5-A a 3.05 bpp nas Grandes Projeções de Atenção
    print("\n--- 3. TESTE DE QUANTIZAÇÃO H5-A (3.05 bpp) EM MATRIZES DE ATENÇÃO ---")
    
    # DeltaNet in_proj_qkv (6144, 2048) -> 12.58M parâmetros
    w_qkv = f.get_tensor("model.language_model.layers.0.linear_attn.in_proj_qkv.weight")
    rec_qkv, bpp_qkv, mse_qkv, cos_qkv, t_qkv = quantize_matrix_h5a_batched(w_qkv, num_ctrl=3, num_spec=3, spec_mode='sensitivity', p_crit=0.05)
    print(f"DeltaNet in_proj_qkv (6144x2048): Cosine = {cos_qkv:.6f} | MSE = {mse_qkv:.8f} | Tempo = {t_qkv:.1f}s")
    
    # DeltaNet out_proj (2048, 2048) -> 4.19M parâmetros
    w_out = f.get_tensor("model.language_model.layers.0.linear_attn.out_proj.weight")
    rec_out, bpp_out, mse_out, cos_out, t_out = quantize_matrix_h5a_batched(w_out, num_ctrl=3, num_spec=3, spec_mode='sensitivity', p_crit=0.05)
    print(f"DeltaNet out_proj    (2048x2048): Cosine = {cos_out:.6f} | MSE = {mse_out:.8f} | Tempo = {t_out:.1f}s")

    # Full Attention q_proj (4096, 2048) -> 8.39M parâmetros
    w_q = f.get_tensor("model.language_model.layers.3.self_attn.q_proj.weight")
    rec_q, bpp_q, mse_q, cos_q, t_q = quantize_matrix_h5a_batched(w_q, num_ctrl=3, num_spec=3, spec_mode='sensitivity', p_crit=0.05)
    print(f"Full Attention q_proj(4096x2048): Cosine = {cos_q:.6f} | MSE = {mse_q:.8f} | Tempo = {t_q:.1f}s")

    # Full Attention o_proj (2048, 2048) -> 4.19M parâmetros
    w_o = f.get_tensor("model.language_model.layers.3.self_attn.o_proj.weight")
    rec_o, bpp_o, mse_o, cos_o, t_o = quantize_matrix_h5a_batched(w_o, num_ctrl=3, num_spec=3, spec_mode='sensitivity', p_crit=0.05)
    print(f"Full Attention o_proj(2048x2048): Cosine = {cos_o:.6f} | MSE = {mse_o:.8f} | Tempo = {t_o:.1f}s")

print("\n" + "=" * 90)
print("DIAGNÓSTICO CONCLUÍDO COM SUCESSO!")
print("=" * 90)
