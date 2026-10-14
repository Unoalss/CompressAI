import sys
sys.path.insert(0, ".")
import time, torch
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer, Qwen3_5MoeTextConfig
from transformers.cache_utils import DynamicCache
from accelerate import init_empty_weights
from vq2.kernels import patch_transformers_kernels
from vq2.paged_runtime import PagedModelStore
from scratch.test_zero_copy_layer import FastPagedMoEExpert

device = "cuda"
patch_transformers_kernels()
store = PagedModelStore("E:/compressAI/vq2_35b_out/qwen35b_full.vq2m")
cbs_data = torch.load("E:/compressAI/vq2_out/codebooks.pt", weights_only=False)
cb16 = cbs_data["16"].to(device=device, dtype=torch.bfloat16)

cfg = AutoConfig.from_pretrained("E:/compressAI/modelo_35b", trust_remote_code=True)
text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg.text_config
if isinstance(text_cfg, dict): text_cfg = Qwen3_5MoeTextConfig(**text_cfg)

with init_empty_weights():
    layer0 = Qwen3_5MoeDecoderLayer(text_cfg, layer_idx=0)

fast_expert = FastPagedMoEExpert(0, store, cb16, capacity=96, device=device)
layer0.mlp.experts = fast_expert

def _set_param(m, param_name, tensor):
    parts = param_name.split(".")
    obj = m
    for p in parts[:-1]:
        if not hasattr(obj, p): return False
        obj = getattr(obj, p)
    setattr(obj, parts[-1], torch.nn.Parameter(tensor, requires_grad=False))
    return True

for name, info in store.other_tensors.items():
    if "layers.0." in name:
        clean = name.replace("model.language_model.layers.0.", "")
        if info.get("type") == "dense":
            _set_param(layer0, clean, store.get_dense_tensor(name, device=device))
        elif info.get("type") == "quant_linear":
            W = store.get_quant_linear(name, cbs_data=cbs_data, device=device)
            _set_param(layer0, clean, W)

layer0.eval()

# Mede subcomponentes com cache
cache = DynamicCache(config=text_cfg)
x = torch.randn(1, 1, 2048, device=device, dtype=torch.bfloat16)
pos_embeds = (torch.zeros(1, 1, 64, device=device), torch.ones(1, 1, 64, device=device))
with torch.inference_mode():
    for _ in range(5):
        _ = layer0(x, position_embeddings=pos_embeds, past_key_values=cache, use_cache=True)
torch.cuda.synchronize()

N = 200
def bench(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / N * 1000

t_attn = bench(lambda: layer0.linear_attn(layer0.input_layernorm(x), cache_params=cache, use_cache=True))
h_norm2 = layer0.post_attention_layernorm(x)
t_gate = bench(lambda: layer0.mlp.gate(h_norm2))
router_logits = layer0.mlp.gate(h_norm2)
if isinstance(router_logits, tuple): router_logits = router_logits[0]
routing_weights = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
topk_weights, topk_ids = torch.topk(routing_weights, 8, dim=-1)
topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
topk_weights = topk_weights.to(torch.bfloat16)

t_shared = bench(lambda: layer0.mlp.shared_expert(h_norm2))
t_experts = bench(lambda: layer0.mlp.experts(h_norm2, topk_ids, topk_weights))
t_total_layer = bench(lambda: layer0(x, position_embeddings=pos_embeds, past_key_values=cache, use_cache=True))

print("=" * 60)
print(f"TEMPO TOTAL DA CAMADA: {t_total_layer:.3f} ms (40 camadas = {t_total_layer * 40:.1f} ms)")
print("=" * 60)
print(f"1. Linear Attention + Norm: {t_attn:.3f} ms ({(t_attn/t_total_layer)*100:.1f}%)")
print(f"2. Router Gate:             {t_gate:.3f} ms ({(t_gate/t_total_layer)*100:.1f}%)")
print(f"3. Shared Expert:           {t_shared:.3f} ms ({(t_shared/t_total_layer)*100:.1f}%)")
print(f"4. MoE Experts (8):         {t_experts:.3f} ms ({(t_experts/t_total_layer)*100:.1f}%)")
print(f"5. Overhead / Residuals:    {t_total_layer - (t_attn + t_gate + t_shared + t_experts):.3f} ms")
print("=" * 60)
