import os
import sys
import time
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from test_qwen_ffn_single_layer import quantize_matrix_h5a_batched

MODEL_DIR = "E:/agente/modelo"

print("=" * 90)
print("MAPA DE SENSIBILIDADE DO MODELO INTEIRO: QWEN3.5-2B")
print("Objetivo: Medir o impacto isolado de quantizar cada tipo de matriz em camadas representativas")
print("=" * 90)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

# Frases de calibração para medir perplexidade / loss
eval_text = "The solar system formed approximately 4.6 billion years ago from the gravitational collapse of a molecular cloud. Quantum mechanics describes nature at the smallest scales."
inp = tokenizer(eval_text, return_tensors="pt")

print("Carregando modelo BF16...")
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
model.eval()

with torch.no_grad():
    base_loss = model(**inp, labels=inp.input_ids).loss.item()
    base_ppl = torch.exp(torch.tensor(base_loss)).item()

print(f"Base BF16: Loss={base_loss:.4f}, PPL={base_ppl:.2f}\n")

# Testar matrizes em camadas de início (0), meio (11) e fim (23)
test_targets = [
    # Camada 0 (Fronteira Inicial - DeltaNet)
    ("L0 DeltaNet in_proj_qkv", 0, "linear_attn", "in_proj_qkv"),
    ("L0 DeltaNet in_proj_z",   0, "linear_attn", "in_proj_z"),
    ("L0 DeltaNet out_proj",    0, "linear_attn", "out_proj"),
    ("L0 FFN gate_proj",        0, "mlp", "gate_proj"),
    ("L0 FFN up_proj",          0, "mlp", "up_proj"),
    ("L0 FFN down_proj",        0, "mlp", "down_proj"),
    
    # Camada 3 (Self-Attention)
    ("L3 Attention q_proj",     3, "self_attn", "q_proj"),
    ("L3 Attention o_proj",     3, "self_attn", "o_proj"),
    ("L3 FFN down_proj",        3, "mlp", "down_proj"),
    
    # Camada 11 (Self-Attention Meio)
    ("L11 Attention q_proj",    11, "self_attn", "q_proj"),
    ("L11 FFN down_proj",       11, "mlp", "down_proj"),
    
    # Camada 12 (DeltaNet Meio)
    ("L12 DeltaNet in_proj_qkv",12, "linear_attn", "in_proj_qkv"),
    ("L12 FFN down_proj",       12, "mlp", "down_proj"),
    
    # Camada 23 (Fronteira Final - Self-Attention)
    ("L23 Attention q_proj",    23, "self_attn", "q_proj"),
    ("L23 Attention o_proj",    23, "self_attn", "o_proj"),
    ("L23 FFN down_proj",       23, "mlp", "down_proj")
]

sensitivity_map = []

print(f"{'Componente Testado':<30} | {'Loss Inicial':<12} | {'Loss H5-A':<12} | {'Delta Loss':<12} | {'PPL Resultante':<15} | {'Sensibilidade'}")
print("-" * 105)

for label, layer_idx, mod_type, weight_name in test_targets:
    layer = model.model.layers[layer_idx]
    module = getattr(layer, mod_type)
    target_layer = getattr(module, weight_name)
    
    # Guardar peso original
    orig_w = target_layer.weight.data.clone()
    
    # Quantizar com H5-A a 3.05 BPW
    w_q, _, _, _, _ = quantize_matrix_h5a_batched(orig_w, num_ctrl=3, num_spec=3, p_crit=0.05, seed=42)
    target_layer.weight.data.copy_(w_q)
    
    # Medir loss
    with torch.no_grad():
        new_loss = model(**inp, labels=inp.input_ids).loss.item()
        delta_loss = new_loss - base_loss
        new_ppl = torch.exp(torch.tensor(new_loss)).item()
        
    if delta_loss > 0.05:
        tier = "CRÍTICA (5.0 BPW)"
    elif delta_loss > 0.015:
        tier = "ALTA (4.0 BPW)"
    elif delta_loss > 0.005:
        tier = "MÉDIA (3.05 BPW)"
    else:
        tier = "BAIXA (2.5 BPW)"
        
    print(f"{label:<30} | {base_loss:<12.4f} | {new_loss:<12.4f} | {delta_loss:<+12.4f} | {new_ppl:<15.2f} | {tier}")
    sensitivity_map.append({
        "label": label,
        "layer": layer_idx,
        "module": mod_type,
        "weight": weight_name,
        "delta_loss": delta_loss,
        "new_ppl": new_ppl,
        "tier": tier
    })
    
    # Restaurar peso original
    target_layer.weight.data.copy_(orig_w)

print("=" * 105)
with open("e:/agente/sensitivity_scan_results.json", "w") as f:
    json.dump(sensitivity_map, f, indent=2)

print("Mapa de sensibilidade salvo em e:/agente/sensitivity_scan_results.json")
