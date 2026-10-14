import torch
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig
from accelerate import init_empty_weights

cfg = AutoConfig.from_pretrained("e:/compressAI/modelo_35b", trust_remote_code=True)
text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg.text_config
if isinstance(text_cfg, dict):
    text_cfg = Qwen3_5MoeTextConfig(**text_cfg)

text_cfg.num_hidden_layers = 2

with init_empty_weights():
    model = Qwen3_5MoeForCausalLM(text_cfg)

keys = list(model.state_dict().keys())
print("Total keys in 2-layer model:", len(keys))
print("Sample keys:")
for k in keys[:15]:
    print(" ", k)
