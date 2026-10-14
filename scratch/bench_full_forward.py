import sys
sys.path.insert(0, ".")
import time, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeTextConfig
)
from transformers.cache_utils import DynamicCache
from accelerate import init_empty_weights

from vq2 import vq
from vq2.kernels import patch_transformers_kernels
from vq2.paged_runtime import PagedModelStore
from scratch.test_zero_copy_layer import FastPagedMoEExpert

device = "cuda" if torch.cuda.is_available() else "cpu"
patch_transformers_kernels()

MODEL_DIR = Path("E:/compressAI/modelo_35b")
MODEL_PATH = Path("E:/compressAI/vq2_35b_out/qwen35b_full.vq2m")

print("Carregando modelo e store...")
store = PagedModelStore(MODEL_PATH)
cbs_data = torch.load("E:/compressAI/vq2_out/codebooks.pt", weights_only=False)
cb16 = cbs_data["16"].to(device=device, dtype=torch.bfloat16)

cfg = AutoConfig.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg.text_config
if isinstance(text_cfg, dict):
    text_cfg = Qwen3_5MoeTextConfig(**text_cfg)

with init_empty_weights():
    model = Qwen3_5MoeForCausalLM(text_cfg)

layers = []
for l_idx in range(40):
    layer = FastPagedMoEExpert(layer_idx=l_idx, store=store, cb16=cb16, capacity=96, device=device)
    model.model.layers[l_idx].mlp.experts = layer
    layers.append(layer)

# Carrega tensores densos
def _set_param(m, param_name, tensor):
    if "visual" in param_name or "mtp" in param_name: return False
    parts = param_name.split(".")
    obj = m
    for p in parts[:-1]:
        if not hasattr(obj, p): return False
        obj = getattr(obj, p)
    setattr(obj, parts[-1], torch.nn.Parameter(tensor, requires_grad=False))
    return True

for name, info in store.other_tensors.items():
    if info.get("type") == "dense":
        t = store.get_dense_tensor(name, device=device)
        _set_param(model, name, t)
    elif info.get("type") == "quant_linear":
        W = store.get_quant_linear(name, cbs_data=cbs_data, device=device)
        if W is not None:
            _set_param(model, name, W)

model.model.embed_tokens.to(device=device)
model.lm_head.to(device=device, dtype=torch.bfloat16)
model.model.rotary_emb.to(device)
model.eval()

# Mede 1 decode step com cache quente
past_kv = DynamicCache(config=text_cfg)
x_in = torch.tensor([[100]], device=device)
emb = model.model.embed_tokens(x_in).to(device=device, dtype=torch.bfloat16)

# Warmup 5 tokens
with torch.inference_mode():
    for _ in range(5):
        out = model.model(inputs_embeds=emb, past_key_values=past_kv, use_cache=True)
        logits = torch.nn.functional.linear(out.last_hidden_state[:, -1:, :], model.lm_head.weight)
torch.cuda.synchronize()

print("\n--- MEDINDO VELOCIDADE DE 1 DECODE STEP COMPLETO (40 CAMADAS) ---")
N = 25
torch.cuda.synchronize()
t0 = time.perf_counter()
with torch.inference_mode():
    for _ in range(N):
        out = model.model(inputs_embeds=emb, past_key_values=past_kv, use_cache=True)
        logits = torch.nn.functional.linear(out.last_hidden_state[:, -1:, :], model.lm_head.weight)
torch.cuda.synchronize()
dt_step = (time.perf_counter() - t0) / N * 1000

print(f"Tempo por token (1 step completo): {dt_step:.2f} ms")
print(f"Velocidade Base Atingida: {1000.0 / dt_step:.1f} tokens/s!")
vram_gb = torch.cuda.memory_allocated() / 1024**3
print(f"VRAM alocada: {vram_gb:.2f} GB (RTX 3060 tem 12 GB)")
