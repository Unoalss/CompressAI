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

print("Layer 0 linear_attn structure:")
print(layer0.linear_attn)

import inspect
print("\nForward source code of linear_attn:")
print(inspect.getsource(layer0.linear_attn.forward))
