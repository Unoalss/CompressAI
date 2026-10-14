import sys, time
from pathlib import Path
sys.path.insert(0, r"e:\compressAI")
import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeTextConfig
)
from accelerate import init_empty_weights
from vq2 import vq
from vq2.inferencia_35b import PackedMoEExpert

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("1. Configurando teste de 4 camadas (3 DeltaNet + 1 Full Attention)...")
cfg = AutoConfig.from_pretrained("e:/compressAI/modelo_35b", trust_remote_code=True)
text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg.text_config
if isinstance(text_cfg, dict):
    text_cfg = Qwen3_5MoeTextConfig(**text_cfg)

# Configura teste com 4 camadas para validar a pipeline completa
text_cfg.num_hidden_layers = 4
text_cfg.layer_types = text_cfg.layer_types[:4]

with init_empty_weights():
    model = Qwen3_5MoeForCausalLM(text_cfg)

print("Esqueleto de 4 camadas criado!")
print("Camada 0 attn:", type(getattr(model.model.layers[0], "self_attn", getattr(model.model.layers[0], "linear_attn", None))))
print("Camada 3 attn:", type(getattr(model.model.layers[3], "self_attn", getattr(model.model.layers[3], "linear_attn", None))))
