import sys, time
from pathlib import Path
sys.path.insert(0, r"e:\compressAI")
import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeTextConfig
)
from accelerate import init_empty_weights
from vq2 import vq
from vq2.inferencia_35b import PackedMoEExpert

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

cfg = AutoConfig.from_pretrained("e:/compressAI/modelo_35b", trust_remote_code=True)
text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg.text_config
if isinstance(text_cfg, dict):
    text_cfg = Qwen3_5MoeTextConfig(**text_cfg)

print("1. Instanciando Camada 0...")
with init_empty_weights():
    layer0 = Qwen3_5MoeDecoderLayer(text_cfg, layer_idx=0)

# Carrega Shard 1
print("2. Carregando Shard 1...")
s1 = torch.load("e:/compressAI/vq2_35b_out/quant_model-00001-of-00021.safetensors.pt", weights_only=False, map_location="cpu")
cbs_data = torch.load("e:/compressAI/vq2_out/codebooks.pt", weights_only=False)
cb16 = cbs_data["16"].to(device=DEVICE, dtype=torch.bfloat16)

# Carrega MoE experts
q_down = s1["quant"]["model.language_model.layers.0.mlp.experts.down_proj"]
q_gate_up = s1["quant"]["model.language_model.layers.0.mlp.experts.gate_up_proj"]
moe_experts = PackedMoEExpert(q_gate_up, q_down, cb16, device=DEVICE)
layer0.mlp.experts = moe_experts

print("3. Inspecionando parametros necessarios da camada 0...")
needed = []
for name, param in layer0.named_parameters():
    if "experts" not in name:
        needed.append((name, param.shape))
print(f"Total de outros parametros na camada 0: {len(needed)}")
for n, sh in needed[:10]:
    full_name = f"model.language_model.layers.0.{n}"
    in_dense = full_name in s1["dense"]
    in_quant = full_name in s1["quant"]
    print(f"  {n}: shape={sh} | dense={in_dense} | quant={in_quant}")

