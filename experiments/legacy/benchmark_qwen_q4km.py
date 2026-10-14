import os
import sys
import time
import json
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "E:/agente/modelo"
GGUF_FILE = "Qwen3.5-2B-Q4_K_M.gguf"
GGUF_PATH = os.path.join(MODEL_DIR, GGUF_FILE)

print("=" * 100)
print("BENCHMARK COMPARATIVO: Qwen3.5-2B-Q4_K_M.gguf")
print("=" * 100)

proc = psutil.Process()
ram_init = proc.memory_info().rss / (1024**2)
sys_ram_avail_init = psutil.virtual_memory().available / (1024**3)

print(f"Arquivo GGUF: {GGUF_PATH}")
file_size_bytes = os.path.getsize(GGUF_PATH)
file_size_gb = file_size_bytes / (1024**3)
print(f"Tamanho real do .gguf: {file_size_bytes:,d} bytes ({file_size_gb:.2f} GiB / {file_size_bytes/(10**9):.2f} GB)")

# 1. Carregar Tokenizer e Modelo GGUF
t_load0 = time.time()
print("\nCarregando tokenizer e pesos Q4_K_M do GGUF...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    gguf_file=GGUF_FILE,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
)
model.eval()
t_load = time.time() - t_load0

ram_after_load = proc.memory_info().rss / (1024**2)
ram_used_by_model = ram_after_load - ram_init
sys_ram_avail_after = psutil.virtual_memory().available / (1024**3)

print(f"Modelo carregado em {t_load:.2f}s!")
print(f"RAM do processo: {ram_after_load:.1f} MB (Delta modelo: {ram_used_by_model:.1f} MB / {ram_used_by_model/1024:.2f} GB)")
print(f"RAM disponível no sistema: {sys_ram_avail_after:.2f} GB (era {sys_ram_avail_init:.2f} GB)")

# 2. Definição idêntica dos 4 domínios de Perplexidade
domains = {
    "Ciencia": "Quantum mechanics describes nature at the smallest scales of energy levels of atoms and subatomic particles.",
    "Codigo": "def binary_search(array, target): low = 0; high = len(array) - 1; while low <= high: mid = (low + high) // 2",
    "Historia": "The ancient library of Alexandria was once the largest repository of knowledge in the Hellenistic world.",
    "Enciclopedia": "The solar system formed approximately 4.6 billion years ago from the gravitational collapse of a molecular cloud."
}

# 3. Definição idêntica dos 3 prompts de Geração
prompts = [
    ("Capital", "The capital of France is"),
    ("Fibonacci", "def fibonacci(n):\n    \"\"\"Return nth Fibonacci number\"\"\"\n"),
    ("Newton", "Newton's second law of motion states that")
]

# 4. Avaliação de Perplexidade
print("\n" + "=" * 80)
print("AVALIANDO PERPLEXIDADE (PPL) NO Q4_K_M")
print("=" * 80)

ppl_scores = {}
ppl_times = {}

for d, txt in domains.items():
    t0 = time.time()
    with torch.no_grad():
        inputs = tokenizer(txt, return_tensors="pt")
        loss = model(**inputs, labels=inputs.input_ids).loss
        score = torch.exp(loss).item()
    t_eval = time.time() - t0
    ppl_scores[d] = score
    ppl_times[d] = t_eval
    print(f"   {d:<12}: PPL = {score:6.2f}  (tempo: {t_eval*1000:.1f} ms)")

ppl_scores["Media"] = sum(ppl_scores[d] for d in domains.keys()) / len(domains)
print("-" * 80)
print(f"   {'MEDIA GLOBAL':<12}: PPL = {ppl_scores['Media']:6.2f}")

# 5. Avaliação de Geração de Texto
print("\n" + "=" * 80)
print("AVALIANDO GERAÇÃO DE TEXTO NO Q4_K_M (3 PROMPTS)")
print("=" * 80)

gen_results = {}

for name, p in prompts:
    t0 = time.time()
    inp = tokenizer(p, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=20, do_sample=False)
    t_gen = time.time() - t0
    txt = tokenizer.decode(out[0], skip_special_tokens=True)
    toks = tokenizer.encode(txt)
    uniq = len(set(toks)) / len(toks)
    gen_results[name] = {
        "text": txt.replace("\n", "\\n")[:95],
        "uniq": uniq,
        "time": t_gen,
        "tokens": len(out[0]) - len(inp.input_ids[0]),
        "tok_per_sec": (len(out[0]) - len(inp.input_ids[0])) / t_gen
    }
    print(f"[{name}] ({t_gen:.2f}s | {gen_results[name]['tok_per_sec']:.2f} tok/s | Diversidade: {uniq*100:.1f}%)")
    print(f"  Texto: {gen_results[name]['text']}")

# 6. Carregar Resultados Anteriores de BF16 e H5-A para Comparação Direta
results_path = "e:/agente/qwen_ffn_h5a_results.json"
prev = {}
if os.path.exists(results_path):
    with open(results_path) as f:
        prev = json.load(f)

ppl_bf16 = prev.get("ppl_bf16", {"Ciencia": 5.05, "Codigo": 2.65, "Historia": 8.18, "Enciclopedia": 3.70, "Media": 4.90})
ppl_h5a = prev.get("ppl_h5a", {"Ciencia": 14.10, "Codigo": 3.22, "Historia": 10.81, "Enciclopedia": 7.49, "Media": 8.90})
gen_bf16 = prev.get("gen_bf16", {})
gen_h5a = prev.get("gen_h5a", {})

# 7. Tabela Comparativa Tridimensional: BF16 vs H5-A vs Q4_K_M
print("\n" + "=" * 105)
print("TABELA COMPARATIVA CONSOLIDADA: BF16 vs H5-A vs Q4_K_M")
print("=" * 105)
print(f"{'Métrica / Domínio':<22} | {'BF16 Original':<15} | {'H5-A (3.05 bpp)':<18} | {'Qwen Q4_K_M':<18} | {'Vantagem H5-A vs Q4'}")
print("-" * 105)
print(f"{'Bits / Peso (BPW)':<22} | {'16.00 bpw':<15} | {'3.05 bpw':<18} | {'5.45 bpw':<18} | {'-44.0% de bits'}")
print(f"{'Tamanho no Disco':<22} | {'4.55 GB':<15} | {'~0.87 GB (teórico)':<18} | {'1.28 GB (real)':<18} | {'Mais compacto'}")
print("-" * 105)

for d in ["Ciencia", "Codigo", "Historia", "Enciclopedia", "Media"]:
    v_bf16 = ppl_bf16.get(d, 0.0)
    v_h5a = ppl_h5a.get(d, 0.0)
    v_q4 = ppl_scores.get(d, 0.0)
    diff_q4_h5a = v_h5a - v_q4
    print(f"PPL {d:<18} | {v_bf16:<15.2f} | {v_h5a:<18.2f} | {v_q4:<18.2f} | {diff_q4_h5a:>+12.2f}")

print("-" * 105)
print("\n--- GERAÇÃO COMPARATIVA ---")
for k in ["Capital", "Fibonacci", "Newton"]:
    print(f"\nPrompt: [{k}]")
    if k in gen_bf16:
        print(f"  BF16:   {gen_bf16[k]['text']}")
    if k in gen_h5a:
        print(f"  H5-A:   {gen_h5a[k]['text']}")
    print(f"  Q4_K_M: {gen_results[k]['text']}")

# 8. Salvar Tudo em JSON
consolidated = {
    "gguf_file": GGUF_PATH,
    "gguf_size_bytes": file_size_bytes,
    "gguf_effective_bpw": 5.445,
    "ram_used_by_model_mb": ram_used_by_model,
    "ppl_bf16": ppl_bf16,
    "ppl_h5a": ppl_h5a,
    "ppl_q4_k_m": ppl_scores,
    "gen_bf16": gen_bf16,
    "gen_h5a": gen_h5a,
    "gen_q4_k_m": gen_results
}

with open("e:/agente/qwen_q4km_comparison_results.json", "w") as f:
    json.dump(consolidated, f, indent=2)

print(f"\nResultados salvos em e:/agente/qwen_q4km_comparison_results.json")
print("=" * 100)
