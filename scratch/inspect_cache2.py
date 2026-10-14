import sys
sys.path.insert(0, r"e:\compressAI")
import torch
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeTextConfig
)
from accelerate import init_empty_weights

cfg = AutoConfig.from_pretrained("e:/compressAI/modelo_35b", trust_remote_code=True)
text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg.text_config

import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as qm
import inspect
print([n for n in dir(qm) if "Cache" in n or "cache" in n])

# Let's inspect how modeling_qwen3_5_moe.py handles use_cache in forward:
lines = inspect.getsource(qm.Qwen3_5MoeModel.forward)
for l in lines.split("\n"):
    if "cache" in l.lower():
        print(l)
