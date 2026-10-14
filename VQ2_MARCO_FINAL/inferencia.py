"""
inferencia.py - Script de teste e geracao usando o modelo compactado VQ2 (567.9 MB).
Execucao: .\\.venv\\Scripts\\python.exe VQ2_MARCO_FINAL\\inferencia.py
"""
import os, sys, time
from pathlib import Path
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vq2.runtime import load_packed, report_memory
from vq2.generate import FastGenerator

MODELO_560 = Path(__file__).resolve().parent / "qwen3.5_2b_vq2_560mb.vq2r"
MODELO_680 = Path(__file__).resolve().parent / "qwen3.5_2b_vq2_680mb.vq2r"
CAMINHO_FINAL = MODELO_560 if MODELO_560.exists() else MODELO_680
MODELO_DIR = ROOT / "modelo"

print("=" * 80)
print("INFERENCIA COM MODELO VQ2 (COMPRIMIDO NA GPU) - QWEN3.5-2B")
print(f"Container selecionado: {CAMINHO_FINAL.name} ({CAMINHO_FINAL.stat().st_size / 2**20:.1f} MiB)")
print("=" * 80)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo de execucao: {device.upper()}")

t0 = time.time()
model = load_packed(str(CAMINHO_FINAL), device=device)   # pesos CONTINUAM comprimidos na GPU
print(f"Modelo carregado em {time.time()-t0:.2f} segundos!")
report_memory(model, " apos load")

tokenizer = AutoTokenizer.from_pretrained(str(MODELO_DIR), trust_remote_code=True)
generator = FastGenerator(model, device=device)
generator.set_tokenizer(tokenizer)

prompts = [
    ("Codigo Python", "def fibonacci(n):\n    \"\"\"Return nth Fibonacci number\"\"\"\n"),
    ("Conhecimento Geral", "The capital of France is"),
    ("Fisica", "Newton's second law of motion states that")
]

print("\n" + "-" * 80)
print("GERANDO RESPOSTAS COM O MODELO COMPACTADO...")
print("-" * 80)

# Warmup rapido
_ = generator.generate("warmup", max_new_tokens=5)
torch.cuda.synchronize()

for categoria, p in prompts:
    print(f"\n[Prompt - {categoria}]:\n{p}")
    
    t_gen0 = time.perf_counter()
    res, n_tokens = generator.generate(p, max_new_tokens=40)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t_gen0
    
    print(f"\n[Saida]:\n{res}")
    print(f"(Gerou {n_tokens} tokens em {dt:.2f}s | Velocidade: {n_tokens/dt:.1f} tok/s)")
    print("-" * 40)

report_memory(model, " final")
print("\n" + "=" * 80)
print("TESTE DE INFERENCIA CONCLUIDO COM SUCESSO!")
print("=" * 80)
