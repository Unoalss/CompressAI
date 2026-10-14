import os, sys, time, json, gc
from pathlib import Path
sys.path.insert(0, r"e:\compressAI")
import torch
from transformers import AutoTokenizer, AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeTextConfig
)
from accelerate import init_empty_weights

from vq2.kernels import HAVE_TRITON
from vq2.load_and_test_full_35b import PackedMoEExpertUniversal, dequant_matrix, set_nested_param

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_35B = Path(r"e:\compressAI\vq2_35b_out")
MODEL_DIR = Path(r"e:\compressAI\modelo_35b")

cfg = AutoConfig.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg.text_config
if isinstance(text_cfg, dict):
    text_cfg = Qwen3_5MoeTextConfig(**text_cfg)

with init_empty_weights():
    model = Qwen3_5MoeForCausalLM(text_cfg)

cbs_data = torch.load("e:/compressAI/vq2_out/codebooks.pt", weights_only=False)
cb16 = cbs_data["16"].to(device=DEVICE, dtype=torch.bfloat16)

index_file = MODEL_DIR / "model.safetensors.index.json"
with open(index_file) as f:
    weight_map = json.load(f)["weight_map"]
shards_to_load = sorted(set(weight_map.values()))

layers_experts_loaded = set()
pending_gate_up = {}
pending_down = {}

print("Carregando 40 camadas...")
for shard_idx, shard_name in enumerate(shards_to_load, 1):
    pt_path = OUT_35B / f"quant_{shard_name}.pt"
    if not pt_path.exists(): continue
    shard_data = torch.load(pt_path, weights_only=False, map_location="cpu")
    quant_dict = shard_data.get("quant", {})
    dense_dict = shard_data.get("dense", {})
    
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
        dev = "cpu" if ("embed_tokens" in clean_name or "lm_head" in clean_name) else DEVICE
        W = dequant_matrix(q_data, cbs_data, device=dev)
        set_nested_param(model, clean_name, W)
        
    del shard_data
    gc.collect()

model.model.rotary_emb.to(DEVICE)
model.eval()

tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
prompt = "Olá! Quem é você?"
input_ids = tok.encode(prompt, return_tensors="pt")

print(f"\nIniciando geração de 10 tokens reais para o prompt: '{prompt}'...")
generated_ids = list(input_ids[0].tolist())
lm_head_weight = model.lm_head.weight.float()

t_start = time.perf_counter()
n_generate = 10

# Loop autoregressivo
for step in range(n_generate):
    curr_input_ids = torch.tensor([generated_ids], dtype=torch.long)
    inputs_embeds = model.model.embed_tokens(curr_input_ids).to(device=DEVICE, dtype=torch.bfloat16)
    
    with torch.no_grad():
        out = model.model(inputs_embeds=inputs_embeds)
        last_hidden = out.last_hidden_state[:, -1:, :].cpu().float()
        logits = torch.matmul(last_hidden, lm_head_weight.T)
        next_tok = torch.argmax(logits, dim=-1).item()
        
    generated_ids.append(next_tok)
    tok_str = tok.decode([next_tok])
    print(f"Token {step+1}: '{tok_str}' (id: {next_tok})", flush=True)

dt = time.perf_counter() - t_start
print(f"\nTempo total para {n_generate} tokens: {dt:.2f}s | Velocidade média: {n_generate/dt:.2f} tokens/s")
full_text = tok.decode(generated_ids)
print(f"Texto gerado completo:\n{full_text}")
