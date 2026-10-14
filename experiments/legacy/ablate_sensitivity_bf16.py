import os
import sys
import time
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from test_qwen_ffn_single_layer import quantize_matrix_h5a_batched

MODEL_DIR = "E:/agente/modelo"

print("=" * 80)
print("AUDITORIA DE SENSIBILIDADE ISOLADA: O QUE IMPACTA A PPL NO BF16?")
print("=" * 80)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

domains = {
    "Ciencia": "Quantum mechanics describes nature at the smallest scales of energy levels of atoms and subatomic particles.",
    "Codigo": "def binary_search(array, target): low = 0; high = len(array) - 1; while low <= high: mid = (low + high) // 2",
    "Historia": "The ancient library of Alexandria was once the largest repository of knowledge in the Hellenistic world.",
    "Enciclopedia": "The solar system formed approximately 4.6 billion years ago from the gravitational collapse of a molecular cloud."
}

def eval_ppl(m):
    scores = {}
    with torch.no_grad():
        for d, txt in domains.items():
            inputs = tokenizer(txt, return_tensors="pt")
            loss = m(**inputs, labels=inputs.input_ids).loss
            scores[d] = torch.exp(loss).item()
    scores["Media"] = sum(scores.values()) / len(scores)
    return scores

print("Carregando modelo BF16...")
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
model.eval()

ppl_base = eval_ppl(model)
print(f"PPL Base BF16: {ppl_base['Media']:.2f} (Codigo: {ppl_base['Codigo']:.2f})")

# 1. Teste Isolado: Embeddings com Top 32k tokens em BF16 + Resto em H5-A (3.05 BPW)
print("\n--- Testando Embeddings Híbridos (Top 32k em BF16 + Resto 3.05 BPW) ---")
w_emb_orig = model.model.embed_tokens.weight.data.clone()

# Quantizar apenas do token 32768 até o final
chunk_size = 50000
total_tokens = w_emb_orig.shape[0]
for start in range(32768, total_tokens, chunk_size):
    end = min(start + chunk_size, total_tokens)
    rec, _, _, _, _ = quantize_matrix_h5a_batched(w_emb_orig[start:end], num_ctrl=3, num_spec=3, p_crit=0.05, seed=500+start)
    model.model.embed_tokens.weight.data[start:end].copy_(rec)

ppl_emb_hybrid = eval_ppl(model)
print(f"PPL com Embeddings Híbridos (Top 32k BF16): {ppl_emb_hybrid['Media']:.2f} (Codigo: {ppl_emb_hybrid['Codigo']:.2f})")

# Restaurar embeddings
model.model.embed_tokens.weight.data.copy_(w_emb_orig)

# 2. Teste Isolado: DeltaNet 54 matrizes a 4.05 BPW vs 3.05 BPW
print("\n--- Testando DeltaNet 54 matrizes em 3 camadas de amostra (camadas 0, 6, 12) ---")
from run_qwen_optimal_bf16_h5a import quantize_matrix_h5a_4bpw

dnet_samples = [0, 6, 12]
saved_dnet = {}
for i in dnet_samples:
    l = model.model.layers[i]
    saved_dnet[f"{i}.qkv"] = l.linear_attn.in_proj_qkv.weight.data.clone()
    saved_dnet[f"{i}.z"] = l.linear_attn.in_proj_z.weight.data.clone()
    saved_dnet[f"{i}.out"] = l.linear_attn.out_proj.weight.data.clone()
    
    rqkv, _, _ = quantize_matrix_h5a_4bpw(saved_dnet[f"{i}.qkv"], num_ctrl=3, num_spec=3, p_crit=0.05, seed=200+i)
    rz, _, _ = quantize_matrix_h5a_4bpw(saved_dnet[f"{i}.z"], num_ctrl=3, num_spec=3, p_crit=0.05, seed=200+i)
    rout, _, _ = quantize_matrix_h5a_4bpw(saved_dnet[f"{i}.out"], num_ctrl=3, num_spec=3, p_crit=0.05, seed=200+i)
    
    l.linear_attn.in_proj_qkv.weight.data.copy_(rqkv)
    l.linear_attn.in_proj_z.weight.data.copy_(rz)
    l.linear_attn.out_proj.weight.data.copy_(rout)

ppl_dnet_sample = eval_ppl(model)
print(f"PPL com DeltaNet (amostra 4.05 BPW): {ppl_dnet_sample['Media']:.2f}")

# Restaurar
for i in dnet_samples:
    l = model.model.layers[i]
    l.linear_attn.in_proj_qkv.weight.data.copy_(saved_dnet[f"{i}.qkv"])
    l.linear_attn.in_proj_z.weight.data.copy_(saved_dnet[f"{i}.z"])
    l.linear_attn.out_proj.weight.data.copy_(saved_dnet[f"{i}.out"])

print("\nConcluído!")
