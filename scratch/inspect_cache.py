import sys
sys.path.insert(0, r"e:\compressAI")
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeTextConfig
)
from accelerate import init_empty_weights

cfg = AutoConfig.from_pretrained("e:/compressAI/modelo_35b", trust_remote_code=True)
text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg.text_config
with init_empty_weights():
    model = Qwen3_5MoeForCausalLM(text_cfg)

import inspect
print("Forward cache params:", [p.name for p in inspect.signature(model.model.forward).parameters.values()])
print("layer_types:", text_cfg.layer_types[:5])
