import sys
sys.path.insert(0, r"e:\compressAI")
import torch
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeTextConfig
)
from transformers.cache_utils import DynamicCache
from accelerate import init_empty_weights

cfg = AutoConfig.from_pretrained("e:/compressAI/modelo_35b", trust_remote_code=True)
text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg.text_config

dc = DynamicCache(config=text_cfg)
print("Configured DynamicCache:", type(dc), "len(layers):", len(dc.layers) if hasattr(dc, "layers") else "no layers attr")
