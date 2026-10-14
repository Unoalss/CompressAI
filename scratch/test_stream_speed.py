import sys
sys.path.insert(0, ".")
import time, torch
from pathlib import Path
from legion.engine import LegionEngine

engine = LegionEngine(cache_capacity=96)
print("\n--- INICIANDO TESTE DE VELOCIDADE REAL DE GERACAO ---")
prompt = "Quem e voce e qual o seu objetivo?"

t0 = time.perf_counter()
tok_count = 0
for chunk in engine.generate_stream(prompt, max_tokens=30, temperature=0.7):
    tok = chunk["token"]
    stats = chunk["stats"]
    tok_count += 1
    sys.stdout.write(tok)
    sys.stdout.flush()

total_time = time.perf_counter() - t0
print(f"\n\nTotal tokens: {tok_count} em {total_time:.2f}s")
print(f"Velocidade media: {tok_count / total_time:.1f} tok/s")
print(f"Stats finais: {stats}")
