import os
import json
from collections import defaultdict
import torch
from safetensors import safe_open

MODEL_DIR = "E:/agente/modelo"
SAFETENSORS_PATH = os.path.join(MODEL_DIR, "model.safetensors-00001-of-00001.safetensors")

print("=" * 80)
print("DIAGNÓSTICO ESTRUTURAL E MATRICIAL: Qwen3.5-2B")
print("=" * 80)

total_params = 0
total_bytes = 0

cat_params = defaultdict(int)
cat_bytes = defaultdict(int)
shapes_by_cat = defaultdict(set)

with safe_open(SAFETENSORS_PATH, framework="pt", device="cpu") as f:
    keys = list(f.keys())
    print(f"Total de chaves no safetensors: {len(keys)}")
    
    # Categorização detalhada
    for k in keys:
        slice_obj = f.get_slice(k)
        shape = tuple(slice_obj.get_shape())
        dtype = slice_obj.get_dtype()
        
        num_el = 1
        for d in shape:
            num_el *= d
            
        bytes_per_elem = 2 if dtype in ("bfloat16", "float16") else (4 if dtype == "float32" else 1)
        tensor_bytes = num_el * bytes_per_elem
        
        total_params += num_el
        total_bytes += tensor_bytes
        
        # Categorizar
        if "visual" in k:
            cat = "Visual Encoder"
        elif "embed_tokens" in k or "lm_head" in k:
            cat = "Embeddings & Head"
        elif "language_model.layers" in k:
            if "mlp.gate_proj" in k:
                cat = "FFN Gate Proj (Expansão)"
            elif "mlp.up_proj" in k:
                cat = "FFN Up Proj (Expansão)"
            elif "mlp.down_proj" in k:
                cat = "FFN Down Proj (Redução)"
            elif "linear_attn" in k or "delta" in k:
                cat = "DeltaNet / Linear Attention"
            elif "self_attn" in k:
                cat = "Full Attention"
            elif "norm" in k:
                cat = "LayerNorm / RMSNorm"
            else:
                cat = "Language Model Outros"
        else:
            cat = "Outros / Metadados"
            
        cat_params[cat] += num_el
        cat_bytes[cat] += tensor_bytes
        shapes_by_cat[cat].add(shape)

print("\n--- DISTRIBUIÇÃO DE PARÂMETROS E TAMANHO EM BYTES ---")
print(f"{'Categoria':<35} | {'Parâmetros':<15} | {'Tamanho (MB)':<12} | {'% do Total':<10}")
print("-" * 80)

for cat in sorted(cat_params.keys(), key=lambda c: cat_bytes[c], reverse=True):
    p = cat_params[cat]
    b = cat_bytes[cat] / (1024 * 1024)
    pct = (cat_bytes[cat] / total_bytes) * 100
    print(f"{cat:<35} | {p:<15,d} | {b:<12.2f} | {pct:<9.2f}%")

print("-" * 80)
print(f"{'TOTAL DO MODELO':<35} | {total_params:<15,d} | {total_bytes / (1024**2):<12.2f} | 100.00%")

print("\n--- FORMATOS (SHAPES) POR CATEGORIA ---")
for cat, shapes in sorted(shapes_by_cat.items()):
    print(f"{cat}:")
    for s in sorted(shapes):
        print(f"   Shape: {s}")

print("\n--- ANÁLISE DETALHADA DO FFN ---")
ffn_total_params = cat_params["FFN Gate Proj (Expansão)"] + cat_params["FFN Up Proj (Expansão)"] + cat_params["FFN Down Proj (Redução)"]
ffn_total_mb = (cat_bytes["FFN Gate Proj (Expansão)"] + cat_bytes["FFN Up Proj (Expansão)"] + cat_bytes["FFN Down Proj (Redução)"]) / (1024**2)
ffn_pct = (ffn_total_mb * (1024**2) / total_bytes) * 100

print(f"Parâmetros FFN Totais: {ffn_total_params:,d} ({ffn_total_mb:.2f} MB em BF16, {ffn_pct:.2f}% dos pesos totais)")
print(f"Número de camadas FFN: 24 camadas x 3 matrizes = 72 matrizes")
print(f"Tamanho teórico a 3.05 bpp no FFN:")
ffn_compressed_mb = ffn_total_mb * (3.05 / 16.0)
print(f"   BF16 Original: {ffn_total_mb:.2f} MB ({ffn_total_mb/1024:.2f} GB)")
print(f"   H5-A (3.05 bpp): {ffn_compressed_mb:.2f} MB ({ffn_compressed_mb/1024:.2f} GB)")
print(f"   Economia teórica no FFN: {ffn_total_mb - ffn_compressed_mb:.2f} MB ({100 * (1 - 3.05/16):.2f}% de redução)")

print("\n" + "=" * 80)
