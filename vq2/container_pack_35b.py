"""
vq2/container_pack_35b.py - Empacotador e validador do container consolidado para Qwen3.8-35B-A3B VQ2.
Combina os 22 shards quantizados em um indice unico pronto para inferencia de alto desempenho em VRAM.
"""
import os, sys, time, json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_35B = ROOT / "vq2_35b_out"
MODEL_DIR = ROOT / "modelo_35b"

def inspect_shards_progress():
    index_file = MODEL_DIR / "model.safetensors.index.json"
    with open(index_file) as f:
        weight_map = json.load(f)["weight_map"]
    all_shards = sorted(set(weight_map.values()))
    
    completed = []
    total_bytes = 0
    for shard in all_shards:
        pt_file = OUT_35B / f"quant_{shard}.pt"
        if pt_file.exists():
            size = pt_file.stat().st_size
            total_bytes += size
            completed.append((shard, size))
            
    print("=" * 80)
    print(f"PROGRESSO DE QUANTIZACAO 35B: {len(completed)}/{len(all_shards)} shards prontos")
    print(f"Tamanho total acumulado: {total_bytes / 1024**3:.2f} GB ({total_bytes / 1024**2:.1f} MB)")
    if completed:
        avg_mb = (total_bytes / len(completed)) / 1024**2
        est_total_gb = (avg_mb * len(all_shards)) / 1024
        print(f"Media por shard: {avg_mb:.1f} MB | Projecao tamanho final: {est_total_gb:.2f} GB")
    print("=" * 80)
    return len(completed), len(all_shards)

if __name__ == "__main__":
    inspect_shards_progress()
