import sys, time
sys.path.insert(0, r"e:\compressAI")
import torch
import torch.nn.functional as F
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer,
    l2norm
)
from transformers.cache_utils import DynamicCache
from accelerate import init_empty_weights

from vq2.kernels import patch_transformers_kernels
from vq2.load_and_test_full_35b import dequant_matrix, set_nested_param

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
patch_transformers_kernels()

cfg = AutoConfig.from_pretrained(r"e:\compressAI\modelo_35b", trust_remote_code=True)
text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg.text_config
with init_empty_weights():
    layer0 = Qwen3_5MoeDecoderLayer(text_cfg, layer_idx=0)

cbs_data = torch.load("e:/compressAI/vq2_out/codebooks.pt", weights_only=False)
s1 = torch.load("e:/compressAI/vq2_35b_out/quant_model-00001-of-00021.safetensors.pt", weights_only=False, map_location="cpu")

for raw_name, tensor in s1["dense"].items():
    if "layers.0." in raw_name:
        clean = raw_name.replace("model.language_model.layers.0.", "")
        set_nested_param(layer0, clean, tensor.to(device=DEVICE, dtype=torch.bfloat16))

for raw_name, q_data in s1["quant"].items():
    if "layers.0." in raw_name and "experts" not in raw_name:
        clean = raw_name.replace("model.language_model.layers.0.", "")
        W = dequant_matrix(q_data, cbs_data, device=DEVICE)
        set_nested_param(layer0, clean, W)

layer0.eval()
linear_attn = layer0.linear_attn

# Prefill 10 tokens para inicializar cache
x_prefill = torch.randn(1, 10, 2048, device=DEVICE, dtype=torch.bfloat16)
cache = DynamicCache(config=text_cfg)
with torch.no_grad():
    _ = layer0.linear_attn(x_prefill, cache_params=cache, use_cache=True)

x_token = torch.randn(1, 1, 2048, device=DEVICE, dtype=torch.bfloat16)

# Mede tempo original
print("\n--- MEDINDO LINEAR_ATTN ORIGINAL ---")
N = 200
for _ in range(10):
    with torch.no_grad():
        _ = linear_attn(x_token, cache_params=cache, use_cache=True)
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(N):
    with torch.no_grad():
        _ = linear_attn(x_token, cache_params=cache, use_cache=True)
torch.cuda.synchronize()
dt_orig = (time.perf_counter() - t0) / N * 1000
print(f"linear_attn Original: {dt_orig:.3f} ms")

# Agora vamos implementar a versao OTIMIZADA do forward de linear_attn
orig_forward = linear_attn.forward

def fast_linear_attn_forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    batch_size, seq_len, _ = hidden_states.shape
    use_precomputed_states = cache_params is not None and cache_params.has_previous_state(self.layer_idx, state_idx=0)
    
    if not (use_precomputed_states and seq_len == 1 and not cache_params.layers[self.layer_idx].record_past):
        return orig_forward(hidden_states, cache_params=cache_params, attention_mask=attention_mask, **kwargs)
        
    # Caminho ultra-rápido de decodificação de 1 token:
    # 1. Projeções de entrada
    mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2) # (1, 8192, 1)
    z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)
    
    # 2. Causal conv1d in-place update
    conv_state = cache_params.layers[self.layer_idx].conv_states[0]
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import causal_conv1d_update
    mixed_qkv = causal_conv1d_update(
        mixed_qkv,
        conv_state,
        self.conv1d.weight.squeeze(1),
        self.conv1d.bias,
        self.activation,
    )
        
    mixed_qkv = mixed_qkv.transpose(1, 2)
        
    # Split Q, K, V
    query, key, value = torch.split(
        mixed_qkv,
        [self.key_dim, self.key_dim, self.value_dim],
        dim=-1,
    )
    q = query.reshape(batch_size, seq_len, -1, self.head_k_dim) # (1, 1, 16, 128)
    k = key.reshape(batch_size, seq_len, -1, self.head_k_dim) # (1, 1, 16, 128)
    v = value.reshape(batch_size, seq_len, -1, self.head_v_dim) # (1, 1, 32, 128)
    
    # Repeat interleave para 32 heads
    q = q.repeat_interleave(2, dim=2) # (1, 1, 32, 128)
    k = k.repeat_interleave(2, dim=2) # (1, 1, 32, 128)
    
    beta = b.sigmoid().reshape(32, 1) # (32, 1)
    g = (-self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)).reshape(32, 1) # (32, 1)
    
    q_norm = l2norm(q.squeeze(1).float(), dim=-1) * (1.0 / (self.head_k_dim ** 0.5)) # (1, 32, 128)
    k_norm = l2norm(k.squeeze(1).float(), dim=-1) # (1, 32, 128)
    v_f = v.squeeze(1).float() # (1, 32, 128)
    
    # Estado recorrente
    recurrent_state = cache_params.layers[self.layer_idx].recurrent_states[0] # (1, 32, 128, 128)
    S = recurrent_state[0] # (32, 128, 128)
    S.mul_(g.exp().unsqueeze(-1))
    k_h = k_norm[0].unsqueeze(1) # (32, 1, 128)
    kv_mem = torch.bmm(k_h, S).squeeze(1) # (32, 128)
    
    # 3. delta = (v - kv_mem) * beta
    delta = (v_f[0] - kv_mem) * beta # (32, 128)
    
    # 4. S += k^T @ delta
    torch.baddbmm(S, k_h.transpose(1, 2), delta.unsqueeze(1), out=S)
    
    # 5. core_attn_out = q @ S -> (32, 1, 128) @ (32, 128, 128) -> (32, 1, 128)
    q_h = q_norm[0].unsqueeze(1)
    core_attn_out = torch.bmm(q_h, S) # (32, 1, 128)
    
    core_attn_out = core_attn_out.squeeze(1).to(hidden_states.dtype) # (32, 128)
    core_attn_out = self.norm(core_attn_out, z.reshape(-1, self.head_v_dim))
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
    
    return self.out_proj(core_attn_out)

import types
linear_attn.forward = types.MethodType(fast_linear_attn_forward, linear_attn)

# Mede tempo OTIMIZADO
print("\n--- MEDINDO LINEAR_ATTN OTIMIZADO ---")
for _ in range(10):
    with torch.no_grad():
        _ = linear_attn(x_token, cache_params=cache, use_cache=True)
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(N):
    with torch.no_grad():
        _ = linear_attn(x_token, cache_params=cache, use_cache=True)
torch.cuda.synchronize()
dt_fast = (time.perf_counter() - t0) / N * 1000
print(f"linear_attn OTIMIZADO: {dt_fast:.3f} ms | Aceleração: {dt_orig/dt_fast:.2f}x!")
print(f"Economia por camada: {dt_orig - dt_fast:.3f} ms")
print(f"Economia total em 40 camadas: {(dt_orig - dt_fast)*40:.2f} ms!")
