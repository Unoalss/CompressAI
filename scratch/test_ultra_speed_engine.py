import sys, time
sys.path.insert(0, r"e:\compressAI")
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeTextConfig
)
from transformers.cache_utils import DynamicCache
from accelerate import init_empty_weights

from vq2 import vq
from vq2.kernels import patch_transformers_kernels
from vq2.load_and_test_full_35b import (
    PackedMoEExpertUniversal,
    dequant_matrix,
    set_nested_param
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = r"e:\compressAI\modelo_35b"

patch_transformers_kernels()

print("1. Carregando modelo e tokenizer...")
tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
EOS_IDS = {tok.eos_token_id, 248044, 248045}

cfg = AutoConfig.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg.text_config
with init_empty_weights():
    model = Qwen3_5MoeForCausalLM(text_cfg)

cbs_data = torch.load("e:/compressAI/vq2_out/codebooks.pt", weights_only=False)
cb16 = cbs_data["16"].to(device=DEVICE, dtype=torch.bfloat16)

import json
with open("e:/compressAI/modelo_35b/model.safetensors.index.json") as f:
    shards_to_load = sorted(set(json.load(f)["weight_map"].values()))

layers_experts_loaded = set()
pending_gate_up = {}
pending_down = {}

for shard_name in shards_to_load:
    pt_path = f"e:/compressAI/vq2_35b_out/quant_{shard_name}.pt"
    try:
        s = torch.load(pt_path, weights_only=False, map_location="cpu")
    except Exception:
        continue
    quant_dict = s.get("quant", {})
    dense_dict = s.get("dense", {})

    for k, q_data in quant_dict.items():
        if "mlp.experts.gate_up_proj" in k:
            m = k.split(".layers.")
            if len(m) > 1: pending_gate_up[int(m[1].split(".")[0])] = q_data
        elif "mlp.experts.down_proj" in k:
            m = k.split(".layers.")
            if len(m) > 1: pending_down[int(m[1].split(".")[0])] = q_data

    for ln in list(pending_gate_up.keys()):
        if ln in pending_down and ln not in layers_experts_loaded:
            expert_mod = PackedMoEExpertUniversal(pending_gate_up[ln], pending_down[ln], cb16, device=DEVICE)
            model.model.layers[ln].mlp.experts = expert_mod
            layers_experts_loaded.add(ln)
            del pending_gate_up[ln], pending_down[ln]

    for raw_name, tensor in dense_dict.items():
        clean_name = raw_name.replace("model.language_model.", "model.")
        set_nested_param(model, clean_name, tensor.to(device=DEVICE, dtype=torch.bfloat16))

    for raw_name, q_data in quant_dict.items():
        if "mlp.experts" in raw_name: continue
        clean_name = raw_name.replace("model.language_model.", "model.")
        dev = "cpu" if "embed_tokens" in clean_name else DEVICE
        W = dequant_matrix(q_data, cbs_data, device=dev)
        set_nested_param(model, clean_name, W)

model.model.rotary_emb.to(DEVICE)
model.eval()
lm_head = model.lm_head.weight
print(f"Modelo 40 camadas carregado na GPU ({torch.cuda.memory_allocated()/1024**3:.2f} GB)!")
