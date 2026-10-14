"""
scratch/test_optimized_legion_engine.py - Teste das 3 otimizações para atingir 30+ tokens/s no LEGION.
1. Fast Top-K sampling (50x mais rápido que full sort de 248k).
2. Dirty-checking no PagedMoEExpert (evita cópias redundantes de slots).
3. Especulação Lookahead Multi-Token (burst de 2-3 tokens por passo).
"""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from legion.engine import LegionEngine

print("Testando motor LEGION com otimizacoes de velocidade...")
engine = LegionEngine(cache_capacity=64)

prompt = "Escreva uma funcao em Python para calcular o fatorial de um numero e explique o codigo passo a passo."

print(f"\nPrompt de teste: '{prompt}'")
print("Iniciando geracao...")

t0 = time.perf_counter()
tokens = []
stats_list = []

for chunk in engine.generate_stream(prompt=prompt, max_tokens=100):
    tokens.append(chunk["token"])
    stats_list.append(chunk["stats"])

dt = time.perf_counter() - t0
n_tok = len(tokens)
tok_s = n_tok / dt if dt > 0 else 0.0

print("\n" + "=" * 80)
print(f"RESULTADO DO TESTE BASE:")
print(f"Total de Tokens: {n_tok} em {dt:.2f}s")
print(f"Velocidade Media: {tok_s:.1f} tokens/segundo")
print(f"Texto gerado: {''.join(tokens[:30])}...")
print("=" * 80)
