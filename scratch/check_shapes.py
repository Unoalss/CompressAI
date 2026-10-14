import sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vq2.runtime import load_packed

p = ROOT / "VQ2_MARCO_FINAL" / "qwen3.5_2b_vq2_560mb.vq2r"
model = load_packed(str(p), device="cuda")

# Let's inspect the shapes of PackedLinear modules
stats = {}
for name, mod in model.named_modules():
    if hasattr(mod, "cfg") and hasattr(mod, "out"):
        stats.setdefault((mod.cfg, mod.out, mod.n), []).append(name)

print("Unique (cfg, out, in) combinations:")
for (cfg, out, n), names in sorted(stats.items(), key=lambda x: -x[0][1]):
    print(f"cfg={cfg}, shape=({out}, {n}) : {len(names)} layers. Example: {names[0]}")
