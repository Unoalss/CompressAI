import sys
sys.path.insert(0, ".")
import time, torch
from legion.engine import LegionEngine

print("Inicializando LEGION com Top-4 Adaptativo (Modo Turbo)...")
engine = LegionEngine(cache_capacity=96)

# Configura todas as 40 camadas para top_k = 4
for layer in engine.model.model.layers:
    if hasattr(layer.mlp, "gate") and hasattr(layer.mlp.gate, "top_k"):
        layer.mlp.gate.top_k = 4

print(f"Top-K configurado para: {engine.model.model.layers[0].mlp.gate.top_k}")

# Teste de geracao real
prompt = "Ola Legion, explique o que e computacao quantica e como ela funciona em 2 frases."
print(f"\nPergunta: {prompt}")
print("Resposta gerada:")

t0 = time.perf_counter()
tok_count = 0
for chunk in engine.generate_stream(prompt, max_tokens=60, temperature=0.7):
    tok = chunk["token"]
    tok_count += 1
    sys.stdout.write(tok)
    sys.stdout.flush()

total_time = time.perf_counter() - t0
print(f"\n\nTotal: {tok_count} tokens em {total_time:.2f}s ({tok_count/total_time:.1f} tok/s)")
print("Stats:", chunk["stats"])
