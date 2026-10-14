import sys, os, time, psutil
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scratch.test_deltanet_inplace_graph import test_inplace_deltanet

def audit_memory():
    print("=" * 80)
    print("AUDITORIA DE MEMÓRIA (VRAM / RAM) DO CUDA GRAPH DECODER")
    print("=" * 80)

    p = psutil.Process()
    ram_init = p.memory_info().rss / (1024**3)
    vram_alloc_init = torch.cuda.memory_allocated() / (1024**3)
    vram_res_init = torch.cuda.memory_reserved() / (1024**3)
    print(f"Estado Inicial:")
    print(f"  VRAM Alocada:   {vram_alloc_init:.2f} GB")
    print(f"  VRAM Reservada: {vram_res_init:.2f} GB")
    print(f"  RAM RSS:        {ram_init:.2f} GB")

    torch.cuda.reset_peak_memory_stats()
    test_inplace_deltanet()

    ram_final = p.memory_info().rss / (1024**3)
    vram_alloc_final = torch.cuda.memory_allocated() / (1024**3)
    vram_res_final = torch.cuda.memory_reserved() / (1024**3)
    vram_peak = torch.cuda.max_memory_allocated() / (1024**3)

    print("\n" + "=" * 80)
    print("RESULTADOS DA AUDITORIA DE MEMÓRIA PÓS-GRAFOS:")
    print("=" * 80)
    print(f"VRAM Alocada Final:     {vram_alloc_final:.2f} GB (Delta: {vram_alloc_final - vram_alloc_init:+.2f} GB)")
    print(f"VRAM Pico Máximo Real:  {vram_peak:.2f} GB (Margem Livre: {12.0 - vram_peak:.2f} GB na RTX 3060 12GB)")
    print(f"VRAM Reservada Final:   {vram_res_final:.2f} GB")
    print(f"RAM RSS do Sistema:     {ram_final:.2f} GB (Delta: {ram_final - ram_init:+.2f} GB)")
    print("=" * 80)

if __name__ == "__main__":
    audit_memory()
