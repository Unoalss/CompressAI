import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vq2 import kernels
from vq2.runtime import load_packed

p = ROOT / "VQ2_MARCO_FINAL" / "qwen3.5_2b_vq2_560mb.vq2r"
model = load_packed(str(p), device="cuda")

# Run 1 forward pass
input_ids = torch.tensor([[100]], device="cuda")
model(input_ids=input_ids)

print("Best configs for _vq_gemv_16_8:")
if hasattr(kernels._vq_gemv_16_8, "best_config"):
    print("best_config:", kernels._vq_gemv_16_8.best_config)
if hasattr(kernels._vq_gemv_16_8, "cache"):
    for k, v in kernels._vq_gemv_16_8.cache.items():
        print(f"Key {k}: {v}")
