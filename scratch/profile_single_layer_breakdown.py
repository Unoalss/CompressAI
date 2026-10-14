import sys, time
sys.path.insert(0, r"e:\compressAI")
import torch
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
OUT_35B = r"e:\compressAI\vq2_35b_out"
MODEL_DIR = r"e:\compressAI\modelo_35b"

patch_transformers_kernels()

cfg = AutoConfig.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg.text_config
with init_empty_weights():
    layer0 = Qwen3_5MoeDecoderLayer(text_cfg, layer_idx=0)

cbs_data = torch.load("e:/compressAI/vq2_out/codebooks.pt", weights_only=False)
cb16 = cbs_data["16"].to(device=DEVICE, dtype=torch.bfloat16)

# Carrega shard 1
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

# Mede cada bloco da camada 0:
x = torch.randn(1, 1, 2048, device=DEVICE, dtype=torch.bfloat16)
cache = DynamicCache(config=text_cfg)

from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding
rotary = Qwen3_5MoeTextRotaryEmbedding(text_cfg).to(DEVICE)
pos_ids = torch.zeros(3, 1, 1, dtype=torch.long, device=DEVICE)
pos_embeds = rotary(x, pos_ids)

# Warmup
for _ in range(10):
    _ = layer0(x, position_embeddings=pos_embeds, past_key_values=cache, use_cache=True)
torch.cuda.synchronize()

N = 200
# 1. Total da camada
t0 = time.perf_counter()
for _ in range(N):
    _ = layer0(x, position_embeddings=pos_embeds, past_key_values=cache, use_cache=True)
torch.cuda.synchronize()
dt_total = (time.perf_counter() - t0) / N * 1000
print(f"Total Camada 0: {dt_total:.3f} ms")

# 2. Apenas self_attn / linear_attn
t0 = time.perf_counter()
for _ in range(N):
    _ = layer0.linear_attn(layer0.input_layernorm(x), past_key_values=cache, use_cache=True)
torch.cuda.synchronize()
dt_attn = (time.perf_counter() - t0) / N * 1000
print(f"  linear_attn (DeltaNet + Conv1d + Layernorm): {dt_attn:.3f} ms")

# 3. Apenas MLP
h_norm = layer0.post_attention_layernorm(x)
t0 = time.perf_counter()
for _ in range(N):
    _ = layer0.mlp(h_norm)
torch.cuda.synchronize()
dt_mlp = (time.perf_counter() - t0) / N * 1000
print(f"  MLP completo (Router + MoE + Shared Expert): {dt_mlp:.3f} ms")
