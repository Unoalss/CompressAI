import sys
sys.path.insert(0, r"e:\compressAI")
import torch
from transformers import AutoConfig, StaticCache
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextConfig

cfg = AutoConfig.from_pretrained("e:/compressAI/modelo_35b", trust_remote_code=True)
text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg.text_config

try:
    cache = StaticCache(config=text_cfg, max_batch_size=1, max_cache_len=256, device="cuda", dtype=torch.bfloat16)
    print("StaticCache initialized successfully for Qwen3.5 MoE!")
except Exception as e:
    print("StaticCache failed for Qwen3.5 MoE:", e)
