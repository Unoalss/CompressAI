import sys, os
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legion.engine import LegionEngine
import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as qm
from scratch.test_fused_qwen35_rope import triton_fused_rmsnorm_rope_qwen

engine = LegionEngine()
layer = engine.model.model.layers[3].self_attn

# Crie tensores de entrada reais
prompt_ids = engine.tokenizer("Teste de atenção", return_tensors="pt")["input_ids"].cuda()

with torch.no_grad():
    out = engine.model(input_ids=prompt_ids, use_cache=True)
    past_kv = out.past_key_values
    
    # 1 token passo de decodificação
    curr = torch.tensor([[prompt_ids[0, -1].item()]], device="cuda")
    
    # Vamos interceptar o forward do layer
    captured = {}
    orig_forward = layer.forward
    
    def debug_forward(hidden_states, position_embeddings, **kwargs):
        captured['hidden_states'] = hidden_states
        captured['position_embeddings'] = position_embeddings
        return orig_forward(hidden_states, position_embeddings=position_embeddings, **kwargs)
        
    layer.forward = debug_forward
    engine.model(input_ids=curr, past_key_values=past_kv, use_cache=True)

hs = captured['hidden_states']
cos, sin = captured['position_embeddings']

print("hidden_states shape:", hs.shape)
print("cos shape:", cos.shape)
print("sin shape:", sin.shape)
print("head_dim:", layer.head_dim)

input_shape = hs.shape[:-1]
hidden_shape = (*input_shape, -1, layer.head_dim)

# Modo PyTorch:
query_raw, gate = torch.chunk(layer.q_proj(hs).view(*input_shape, -1, layer.head_dim * 2), 2, dim=-1)
q_norm_pt = layer.q_norm(query_raw.view(hidden_shape)).transpose(1, 2)
k_raw = layer.k_proj(hs).view(hidden_shape)
k_norm_pt = layer.k_norm(k_raw).transpose(1, 2)
q_rope_pt, k_rope_pt = qm.apply_rotary_pos_emb(q_norm_pt, k_norm_pt, cos, sin)

# Modo Triton:
q_tri = triton_fused_rmsnorm_rope_qwen(query_raw.view(hidden_shape), layer.q_norm.weight, cos, sin, eps=layer.q_norm.eps)
k_tri = triton_fused_rmsnorm_rope_qwen(k_raw, layer.k_norm.weight, cos, sin, eps=layer.k_norm.eps)

print("\n--- COMPARAÇÃO Q ---")
print("q_rope_pt shape:", q_rope_pt.shape)
print("q_tri shape:", q_tri.shape)
cos_q = torch.nn.functional.cosine_similarity(q_rope_pt.float().flatten(), q_tri.float().flatten(), dim=0).item()
diff_q = (q_rope_pt - q_tri).abs().max().item()
print(f"Cosine Sim Q: {cos_q:.8f}, Max Diff: {diff_q:.6f}")

print("\n--- COMPARAÇÃO K ---")
print("k_rope_pt shape:", k_rope_pt.shape)
print("k_tri shape:", k_tri.shape)
cos_k = torch.nn.functional.cosine_similarity(k_rope_pt.float().flatten(), k_tri.float().flatten(), dim=0).item()
diff_k = (k_rope_pt - k_tri).abs().max().item()
print(f"Cosine Sim K: {cos_k:.8f}, Max Diff: {diff_k:.6f}")

# Detalhe dos primeiros 10 elementos de Q
print("\nPrimeiros 10 elementos de Q[0, 0, 0, :10] PyTorch:", q_rope_pt[0, 0, 0, :10].tolist())
print("Primeiros 10 elementos de Q[0, 0, 0, :10] Triton: ", q_tri[0, 0, 0, :10].tolist())
print("Elementos 60..70 de Q[0, 0, 0, 60:70] PyTorch:   ", q_rope_pt[0, 0, 0, 60:70].tolist())
print("Elementos 60..70 de Q[0, 0, 0, 60:70] Triton:    ", q_tri[0, 0, 0, 60:70].tolist())
