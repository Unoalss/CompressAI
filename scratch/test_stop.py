import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legion.engine import LegionEngine

engine = LegionEngine(cache_capacity=96)
prompt = "O que e um modelo Mixture of Experts (MoE)? Responda de forma direta em 2 frases."
print(f"\nTestando prompt: '{prompt}'")
full_text = ""
token_count = 0
for chunk in engine.generate_stream(prompt=prompt, max_tokens=150, temperature=0.7):
    token = chunk["token"]
    full_text += token
    token_count += 1
    print(token, end="", flush=True)

print(f"\n\n--- FIM DO TESTE ---")
print(f"Total tokens gerados: {token_count}")
