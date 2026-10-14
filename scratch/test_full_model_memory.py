import sys, time
from pathlib import Path
sys.path.insert(0, r"e:\compressAI")
import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeTextConfig
)
from accelerate import init_empty_weights

print("=== TESTE DE ORÇAMENTO DE VRAM DO MODELO 35B ===")
print("GPU:", torch.cuda.get_device_name(0))
vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"VRAM Total: {vram_total:.2f} GB")

# 1. MoE Experts (40 layers x 192 MB) em int16
moe_bytes = 40 * (128 + 64) * 1024**2
print(f"MoE Experts (256 experts x 40 camadas VQ2 int16): {moe_bytes / 1024**3:.2f} GB")

# 2. Non-MoE Attention + Norms (sem embed e sem lm_head)
# Total non-moe = 2.93B params. Menos embed (508M) e lm_head (508M) = 1.91B params!
# 1.91B params em BF16 = 3.82 GB!
non_moe_layers_bytes = 1.91 * 1e9 * 2
print(f"Atenção + Normas (40 camadas em BF16): {non_moe_layers_bytes / 1024**3:.2f} GB")

# 3. Embeddings e LM Head em CPU ou VQ2
embed_bytes_cpu = 508 * 1e6 * 2
print(f"Embeddings + LM Head (na CPU ou VQ2): {embed_bytes_cpu / 1024**3:.2f} GB (fora da VRAM!)")

total_vram_needed = (moe_bytes + non_moe_layers_bytes) / 1024**3
print(f"\nTotal de VRAM necessária na RTX 3060: {total_vram_needed:.2f} GB")
print(f"VRAM Livre restante: {vram_total - total_vram_needed:.2f} GB")

if total_vram_needed < vram_total:
    print(">>> SUCESSO: O MODELO COMPLETO CABE NA GPU COM FOLGA!")
else:
    print(">>> ALERTA: Excede a VRAM!")
