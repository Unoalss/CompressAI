import sys, os, time, psutil
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legion.engine import LegionEngine

def test():
    print("=" * 80)
    print("TESTE DE INICIALIZAÇÃO DO LEGION EM CPU (REFERÊNCIA)")
    print("=" * 80)

    p = psutil.Process()
    rss_before = p.memory_info().rss / 1024**3
    print(f"RAM RSS antes de carregar: {rss_before:.2f} GB")

    t0 = time.perf_counter()
    engine = LegionEngine(device="cpu", cache_capacity=16)
    t_load = time.perf_counter() - t0
    rss_after = p.memory_info().rss / 1024**3
    print(f"Tempo de carregamento: {t_load:.2f} s")
    print(f"RAM RSS após carregar: {rss_after:.2f} GB (Delta: +{rss_after - rss_before:.2f} GB)")

    prompt = "Olá, quem é você?"
    print(f"\nTestando geração na CPU com prompt: '{prompt}' (max_tokens=3)...")
    t_gen0 = time.perf_counter()
    tokens_out = []
    for chunk in engine.generate_stream(prompt, max_tokens=3, temperature=0.0):
        tokens_out.append(chunk["token"])
        print(f"  Token gerado: {chunk['token']!r} | stats: tok/s={chunk['stats']['tok_s']}, RSS={chunk['stats']['rss_gb']} GB")
    
    t_total = time.perf_counter() - t_gen0
    rss_end = p.memory_info().rss / 1024**3
    print(f"\nTexto gerado: {''.join(tokens_out)!r}")
    print(f"Tempo total de geração (3 tokens): {t_total:.2f} s")
    print(f"RAM RSS final: {rss_end:.2f} GB")
    print("TESTE DE CPU EXECUTADO COM SUCESSO!")

if __name__ == "__main__":
    test()
