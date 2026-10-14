import sys
sys.path.insert(0, r"e:\compressAI")
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer
from accelerate import init_empty_weights

cfg = AutoConfig.from_pretrained(r"e:\compressAI\modelo_35b", trust_remote_code=True)
tc = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg.text_config
with init_empty_weights():
    l = Qwen3_5MoeDecoderLayer(tc, 0)
attn = l.linear_attn
print("key_dim:", attn.key_dim)
print("value_dim:", attn.value_dim)
print("head_k_dim:", attn.head_k_dim)
print("head_v_dim:", attn.head_v_dim)
print("num_k_heads:", attn.num_k_heads)
print("num_v_heads:", attn.num_v_heads)
print("conv_kernel_size:", attn.conv_kernel_size)
