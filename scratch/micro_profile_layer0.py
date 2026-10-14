import sys, time
sys.path.insert(0, r"e:\compressAI")
import torch
import torch.nn.functional as F

from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeTextConfig
)
from transformers.cache_utils import DynamicCache
from accelerate import init_empty_weights

from vq2 import vq
from vq2.kernels import patch_transformers_kernels
from vq2.load_and_test_full_35b import (
    PackedMoEExpertUniversal,
    dequant_matrix,
    set_nested_param
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
patch_transformers_kernels()

cfg = AutoConfig.from_pretrained(r"e:\compressAI\modelo_35b", trust_remote_code=True)
text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg.text_config
with init_empty_weights():
    layer0 = Qwen3_5MoeDecoderLayer(text_cfg, layer_idx=0)

cbs_data = torch.load("e:/compressAI/vq2_out/codebooks.pt", weights_only=False)
cb16 = cbs_data["16"].to(device=DEVICE, dtype=torch.bfloat16)

s1 = torch.load("e:/compressAI/vq2_35b_out/quant_model-00001-of-00021.safetensors.pt", weights_only=False, map_location="cpu")
q_gu = s1["quant"]["model.language_model.layers.0.mlp.experts.gate_up_proj"]
q_down = s1["quant"]["model.language_model.layers.0.mlp.experts.down_proj"]

layer0.mlp.experts = PackedMoEExpertUniversal(q_gu, q_down, cb16, device=DEVICE)

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

# Preparar tensores
x = torch.randn(1, 1, 2048, device=DEVICE, dtype=torch.bfloat16)
cache = DynamicCache(config=text_cfg)

from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding
rotary = Qwen3_5MoeTextRotaryEmbedding(text_cfg).to(DEVICE)
pos_ids = torch.zeros(3, 1, 1, dtype=torch.long, device=DEVICE)
pos_embeds = rotary(x, pos_ids)

# Warmup
for _ in range(20):
    _ = layer0(x, position_embeddings=pos_embeds, past_key_values=cache, use_cache=True)
torch.cuda.synchronize()

# Medição precisa com CUDA Events
start_ev = torch.cuda.Event(enable_timing=True)
end_ev = torch.cuda.Event(enable_timing=True)

def measure(fn, N=200):
    torch.cuda.synchronize()
    start_ev.record()
    for _ in range(N):
        fn()
    end_ev.record()
    torch.cuda.synchronize()
    return start_ev.elapsed_time(end_ev) / N # ms

# 1. Total Layer
t_total = measure(lambda: layer0(x, position_embeddings=pos_embeds, past_key_values=cache, use_cache=True))

# 2. linear_attn componentes
attn = layer0.linear_attn
norm1 = layer0.input_layernorm
h_norm1 = norm1(x)

t_norm1 = measure(lambda: norm1(x))
t_in_proj_qkv = measure(lambda: attn.in_proj_qkv(h_norm1))
t_in_proj_z = measure(lambda: attn.in_proj_z(h_norm1))
t_in_proj_b = measure(lambda: attn.in_proj_b(h_norm1))
t_in_proj_a = measure(lambda: attn.in_proj_a(h_norm1))

# Executa in_projs para ter tensores intermediários
mixed_qkv = attn.in_proj_qkv(h_norm1).transpose(1, 2)
z = attn.in_proj_z(h_norm1).reshape(1, 1, -1, attn.head_v_dim)
conv_state = cache.layers[0].conv_states[0]

from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import causal_conv1d_update
t_conv1d = measure(lambda: causal_conv1d_update(mixed_qkv, conv_state, attn.conv1d.weight.squeeze(1), attn.conv1d.bias, attn.activation))

# DeltaNet recurrent rule
query, key, value = torch.split(mixed_qkv.transpose(1, 2), [attn.key_dim, attn.key_dim, attn.value_dim], dim=-1)
query = query.reshape(1, 1, -1, attn.head_k_dim).repeat_interleave(2, dim=2)
key = key.reshape(1, 1, -1, attn.head_k_dim).repeat_interleave(2, dim=2)
value = value.reshape(1, 1, -1, attn.head_v_dim)
b = attn.in_proj_b(h_norm1)
a = attn.in_proj_a(h_norm1)
beta = b.sigmoid()
g = -attn.A_log.float().exp() * F.softplus(a.float() + attn.dt_bias)
recurrent_state = cache.layers[0].recurrent_states[0]

from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import torch_recurrent_gated_delta_rule
t_recurrent = measure(lambda: torch_recurrent_gated_delta_rule(query, key, value, g=g, beta=beta, initial_state=recurrent_state, output_final_state=True))

core_out, _ = torch_recurrent_gated_delta_rule(query, key, value, g=g, beta=beta, initial_state=recurrent_state, output_final_state=True)
core_2d = core_out.reshape(-1, attn.head_v_dim)
z_2d = z.reshape(-1, attn.head_v_dim)
t_norm_gated = measure(lambda: attn.norm(core_2d, z_2d))
normed_out = attn.norm(core_2d, z_2d).reshape(1, 1, -1)
t_out_proj = measure(lambda: attn.out_proj(normed_out))

# 3. MLP componentes
norm2 = layer0.post_attention_layernorm
h_norm2 = norm2(x)
t_norm2 = measure(lambda: norm2(x))
mlp = layer0.mlp
t_gate = measure(lambda: mlp.gate(h_norm2))
router_logits = mlp.gate(h_norm2)
if isinstance(router_logits, tuple): router_logits = router_logits[0]
routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
topk_weights, topk_ids = torch.topk(routing_weights, 8, dim=-1)
topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
topk_weights = topk_weights.to(torch.bfloat16)

t_shared = measure(lambda: mlp.shared_expert(h_norm2))
t_experts = measure(lambda: mlp.experts(h_norm2, topk_ids, topk_weights))

print("=" * 70)
print(f"MICRO-BENCHMARK DE UMA CAMADA (Layer 0) - TEMPO TOTAL: {t_total:.3f} ms")
print("=" * 70)
print(f"1. RMSNorm 1:                   {t_norm1*1000:7.1f} µs")
print(f"2. in_proj_qkv (2048 -> 8192):   {t_in_proj_qkv*1000:7.1f} µs")
print(f"3. in_proj_z   (2048 -> 4096):   {t_in_proj_z*1000:7.1f} µs")
print(f"4. in_proj_b   (2048 -> 32):     {t_in_proj_b*1000:7.1f} µs")
print(f"5. in_proj_a   (2048 -> 32):     {t_in_proj_a*1000:7.1f} µs")
print(f"6. causal_conv1d_update:        {t_conv1d*1000:7.1f} µs")
print(f"7. torch_recurrent_delta_rule:  {t_recurrent*1000:7.1f} µs  <--")
print(f"8. norm_gated:                  {t_norm_gated*1000:7.1f} µs")
print(f"9. out_proj    (4096 -> 2048):   {t_out_proj*1000:7.1f} µs")
print(f"10. RMSNorm 2:                  {t_norm2*1000:7.1f} µs")
print(f"11. MoE Router Gate:            {t_gate*1000:7.1f} µs")
print(f"12. MoE Shared Expert:          {t_shared*1000:7.1f} µs")
print(f"13. MoE Triton V2 Experts (8):  {t_experts*1000:7.1f} µs")
print("=" * 70)
