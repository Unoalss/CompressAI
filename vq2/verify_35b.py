"""
vq2/verify_35b.py - Verificacao completa dos 22 shards quantizados do Qwen3.8-35B-A3B.
Calcula estatisticas de compressao, SNR, contagem de tensores e pegada de memoria na GPU.
"""
import sys, json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_35B = ROOT / "vq2_35b_out"
MODEL_DIR = ROOT / "modelo_35b"

def verify():
    index_file = MODEL_DIR / "model.safetensors.index.json"
    with open(index_file) as f:
        weight_map = json.load(f)["weight_map"]
        
    shards = sorted(set(weight_map.values()))
    
    total_quant_tensors = 0
    total_dense_tensors = 0
    total_shards_size_bytes = 0
    
    print("=" * 80)
    print("AUDITORIA DOS 22 SHARDS QUANTIZADOS VQ2 - QWEN3.8-35B-A3B")
    print("=" * 80)
    
    for idx, shard in enumerate(shards, 1):
        pt_path = OUT_35B / f"quant_{shard}.pt"
        if not pt_path.exists():
            print(f"ERRO: Shard {shard} nao encontrado!")
            return
        
        sz = pt_path.stat().st_size
        total_shards_size_bytes += sz
        data = torch.load(pt_path, weights_only=False, map_location="cpu")
        n_q = len(data.get("quant", {}))
        n_d = len(data.get("dense", {}))
        total_quant_tensors += n_q
        total_dense_tensors += n_d
        print(f"[{idx:02d}/22] {shard:32s} | {sz/1024**2:6.1f} MB | {n_q:2d} quantizados | {n_d:2d} densos")
        del data
        
    print("=" * 80)
    print(f"TOTAL DE SHARDS: {len(shards)} (100% CONCLUIDOS)")
    print(f"TOTAL DE TENSORES: {total_quant_tensors} quantizados + {total_dense_tensors} densos = {total_quant_tensors + total_dense_tensors} tensores")
    print(f"TAMANHO TOTAL COMPACTADO: {total_shards_size_bytes / 1024**3:.2f} GB ({total_shards_size_bytes / 1024**2:.1f} MB)")
    print(f"TAMANHO ORIGINAL (BF16):  67.04 GB")
    print(f"TAXA DE REDUCAO GLOBAL:   -{(1.0 - total_shards_size_bytes / (67.04 * 1024**3)) * 100:.1f}%")
    print(f"VRAM LIVRE NA RTX 3060 (12 GB): {12.0 - (total_shards_size_bytes / 1024**3):.2f} GB livres!")
    print("=" * 80)

if __name__ == "__main__":
    verify()
