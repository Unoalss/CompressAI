import time
from legion.engine import LegionEngine

engine = LegionEngine()
print("Engine pronta! Iniciando geracao...")
t0 = time.perf_counter()
for chunk in engine.generate_stream("Ola, tudo bem? Me fale sobre inteligencia artificial.", max_tokens=15):
    print(f"Token: {chunk['token']!r:15s} | Step: {chunk['stats']['step_ms']:6.1f} ms | tok/s: {chunk['stats']['tok_s']:5.1f} | HitRate: {chunk['stats']['hit_rate_pct']}%")
print(f"Tempo total: {time.perf_counter() - t0:.2f}s")
