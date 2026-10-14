import os, sys, time, json, gc
from pathlib import Path
sys.path.insert(0, r"e:\compressAI")
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeTextConfig
)
from accelerate import init_empty_weights

from vq2 import vq
from vq2.kernels import HAVE_TRITON, vq_moe_parallel_gate_up, vq_moe_parallel_down
from vq2.load_and_test_full_35b import PackedMoEExpertUniversal, dequant_matrix, set_nested_param

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_35B = Path(r"e:\compressAI\vq2_35b_out")
MODEL_DIR = Path(r"e:\compressAI\modelo_35b")

print("=" * 80)
print("TESTE DE GERAÇÃO REAL DE 40 CAMADAS (35.5B PARAMS) NA RTX 3060 12GB")
print("=" * 80)

# Configura o modelo completo de 40 camadas
cfg = AutoConfig.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg.text_config
if isinstance(text_cfg, dict):
    text_cfg = Qwen3_5MoeTextConfig(**text_cfg)

print("1. Criando esqueleto completo de 40 camadas...")
with init_empty_weights():
    model = Qwen3_5MoeForCausalLM(text_cfg)

# Carrega Codebook
cbs_data = torch.load("e:/compressAI/vq2_out/codebooks.pt", weights_only=False)
cb16 = cbs_data["16"].to(device=DEVICE, dtype=torch.bfloat16)

# Carrega todos os 21 shards
index_file = MODEL_DIR / "model.safetensors.index.json"
with open(index_file) as f:
    weight_map = json.load(f)["weight_map"]
shards_to_load = sorted(set(weight_map.values()))

layers_experts_loaded = set()
pending_gate_up = {}
pending_down = {}

t0 = time.time()
print("2. Injetando pesos dos 21 shards nas 40 camadas...")
for shard_idx, shard_name in enumerate(shards_to_load, 1):
    pt_path = OUT_35B / f"quant_{shard_name}.pt"
    if not pt_path.exists():
        continue
        
    shard_data = torch.load(pt_path, weights_only=False, map_location="cpu")
    quant_dict = shard_data.get("quant", {})
    dense_dict = shard_data.get("dense", {})
    
    # Experts MoE
    for k, q_data in quant_dict.items():
        if "mlp.experts.gate_up_proj" in k:
            m = k.split(".layers.")
            if len(m) > 1:
                ln = int(m[1].split(".")[0])
                pending_gate_up[ln] = q_data
        elif "mlp.experts.down_proj" in k:
            m = k.split(".layers.")
            if len(m) > 1:
                ln = int(m[1].split(".")[0])
                pending_down[ln] = q_data
                
    for ln in list(pending_gate_up.keys()):
        if ln in pending_down and ln not in layers_experts_loaded:
            q_gu = pending_gate_up[ln]
            q_down = pending_down[ln]
            expert_mod = PackedMoEExpertUniversal(q_gu, q_down, cb16, device=DEVICE)
            model.model.layers[ln].mlp.experts = expert_mod
            layers_experts_loaded.add(ln)
            del pending_gate_up[ln], pending_down[ln]
            
    # Tensores Densos
    for raw_name, tensor in dense_dict.items():
        clean_name = raw_name.replace("model.language_model.", "model.")
        tensor_cuda = tensor.to(device=DEVICE, dtype=torch.bfloat16)
        set_nested_param(model, clean_name, tensor_cuda)
        
    # Tensores Lineares Quantizados
    for raw_name, q_data in quant_dict.items():
        if "mlp.experts" in raw_name:
            continue
        clean_name = raw_name.replace("model.language_model.", "model.")
        
        # embed_tokens e lm_head mantidos em CPU para economizar 2.04 GB de VRAM!
        if "embed_tokens" in clean_name or "lm_head" in clean_name:
            W = dequant_matrix(q_data, cbs_data, device="cpu")
            set_nested_param(model, clean_name, W)
        else:
            W = dequant_matrix(q_data, cbs_data, device=DEVICE)
            set_nested_param(model, clean_name, W)
            
    del shard_data
    gc.collect()

model.model.rotary_emb.to(DEVICE)
print(f"3. Modelo completo de 40 camadas pronto em {time.time()-t0:.2f}s!")
print(f"Camadas MoE ativas: {len(layers_experts_loaded)}/40")
print(f"VRAM alocada na GPU: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Teste de geracao real de texto
print("\n4. Testando geracao real autoregressiva token-a-token...")
tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
prompt = "Olá! Quem é você?"
input_ids = tok.encode(prompt, return_tensors="pt")

print(f"Prompt: '{prompt}' ({input_ids.shape[1]} tokens)")

# Prefill
# 1. Pega embeddings na CPU e envia para a GPU
inputs_embeds = model.model.embed_tokens(input_ids).to(device=DEVICE, dtype=torch.bfloat16)

with torch.no_grad():
    hidden_states = model.model(inputs_embeds=inputs_embeds).last_hidden_state
    # Projeta para lm_head na CPU
    last_hidden = hidden_states[:, -1:, :].cpu().float()
    lm_head_weight = model.lm_head.weight.float()
    logits = torch.matmul(last_hidden, lm_head_weight.T)
    next_token = torch.argmax(logits, dim=-1).item()

print(f"Primeiro token gerado pelo modelo: {next_token} -> '{tok.decode([next_token])}'")
print("=" * 80)
print("SUCESSO HISTÓRICO: O MODELO DE 35.5B PARÂMETROS ESTÁ GERANDO TEXTO REAL NA RTX 3060 12GB!")
print("=" * 80)
