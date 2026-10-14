import json
from pathlib import Path

ROOT = Path(r"e:\compressAI")
with open(ROOT / "vq2_out" / "alloc_560mb.json") as f:
    alloc = json.load(f)

with open(ROOT / "vq2_out" / "hessian_distortion_sweep.json") as f:
    sweep = json.load(f)

downgraded = [name for name, cfg in alloc["allocation"].items() if tuple(cfg) == (16,)]
print(f"Total de camadas em 2.0 bpw: {len(downgraded)}")

# Look up their penalty_per_mb in sweep
penalties = []
if isinstance(sweep, dict):
    for name in downgraded:
        if name in sweep:
            item = sweep[name]
            penalties.append((item.get("penalty_per_mb", 0), item.get("delta_distortion", 0), item.get("saved_mb", 0), name))
else:
    for name in downgraded:
        for item in sweep:
            if item.get("name") == name:
                penalties.append((item.get("penalty_per_mb", 0), item.get("delta_distortion", 0), item.get("saved_mb", 0), name))
                break

penalties.sort(reverse=True)
print("\nTop 15 camadas com MAIOR penalidade entre as 39 em 2.0 bpw:")
for p, d, s, name in penalties[:15]:
    print(f"penalty={p:.6f} | delta_dist={d:.4f} | saved={s:.1f}MB | {name}")

print(f"\nCamada de embed_tokens:")
for p, d, s, name in penalties:
    if "embed" in name:
        print(f"penalty={p:.6f} | delta_dist={d:.4f} | saved={s:.1f}MB | {name}")
