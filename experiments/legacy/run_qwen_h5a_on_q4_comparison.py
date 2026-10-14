import os
import sys
import time
import json
import gc
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from test_qwen_ffn_single_layer import quantize_matrix_h5a_batched

MODEL_DIR = "E:/agente/modelo"
GGUF_FILE = "Qwen3.5-2B-Q4_K_M.gguf"
print("=" * 100)
print("EXPERIMENTO: APLICANDO H5-A (3.05 BPW) SOBRE O MODELO Q4_K_M")
print("Objetivo: Reduzir o modelo Q4 com H5-A e comparar os 3 (Q4 vs H5-A(BF16) vs H5-A(Q4))")
print("=" * 100)

device = "cpu"
torch.set_num_threads(os.cpu_count() or 8)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

domains = {
    "Ciencia": "Quantum mechanics describes nature at the smallest scales of energy levels of atoms and subatomic particles.",
    "Codigo": "def binary_search(array, target): low = 0; high = len(array) - 1; while low <= high: mid = (low + high) // 2",
    "Historia": "The ancient library of Alexandria was once the largest repository of knowledge in the Hellenistic world.",
    "Enciclopedia": "The solar system formed approximately 4.6 billion years ago from the gravitational collapse of a molecular cloud."
}

prompts = [
    ("Capital", "The capital of France is"),
    ("Fibonacci", "def fibonacci(n):\n    \"\"\"Return nth Fibonacci number\"\"\"\n"),
    ("Newton", "Newton's second law of motion states that")
]

def eval_ppl(m):
    scores = {}
    with torch.no_grad():
        for d, txt in domains.items():
            inputs = tokenizer(txt, return_tensors="pt")
            loss = m(**inputs, labels=inputs.input_ids).loss
            scores[d] = torch.exp(loss).item()
    scores["Media"] = sum(scores.values()) / len(scores)
    return scores

def eval_gen(m, max_tokens=20):
    gens = {}
    with torch.no_grad():
        for name, p in prompts:
            inp = tokenizer(p, return_tensors="pt")
            out = m.generate(**inp, max_new_tokens=max_tokens, do_sample=False)
            txt = tokenizer.decode(out[0], skip_special_tokens=True)
            toks = tokenizer.encode(txt)
            uniq = len(set(toks)) / len(toks)
            gens[name] = {"text": txt.replace("\n", "\\n")[:80], "uniq": uniq}
    return gens

# 1. Carregar o Modelo Q4_K_M Completo
print("\nCarregando modelo Q4_K_M de E:/agente/modelo...")
t_load0 = time.time()
model_q4 = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    gguf_file=GGUF_FILE,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
)
model_q4.eval()
print(f"Modelo Q4 carregado em {time.time()-t_load0:.2f}s!")

# 2. Aplicar H5-A (3.05 bpp) nas 72 matrizes FFN do modelo Q4
print("\n" + "=" * 90)
print("APLICANDO H5-A (3.05 bpp) NAS 72 MATRIZES FFN DO MODELO Q4_K_M...")
print("Configuração Asimétrica: gate_proj (2+4-D), up_proj (2+4-D), down_proj (3+3), p_crit=5%")
print("=" * 90)

t_quant0 = time.time()
layer_stats = []

for i in range(24):
    t_l0 = time.time()
    l = model_q4.model.layers[i]
    
    # Pegar pesos Q4 de cada matriz FFN
    w_q4_gate = l.mlp.gate_proj.weight.data.clone()
    w_q4_up = l.mlp.up_proj.weight.data.clone()
    w_q4_down = l.mlp.down_proj.weight.data.clone()
    
    # Quantizar com H5-A a 3.05 bpp
    rec_gate, bpp_g, mse_g, cos_g, _ = quantize_matrix_h5a_batched(w_q4_gate, num_ctrl=2, num_spec=4, spec_mode='sensitivity', p_crit=0.05, seed=42+i)
    l.mlp.gate_proj.weight.data.copy_(rec_gate)
    
    rec_up, bpp_u, mse_u, cos_u, _ = quantize_matrix_h5a_batched(w_q4_up, num_ctrl=2, num_spec=4, spec_mode='sensitivity', p_crit=0.05, seed=42+i)
    l.mlp.up_proj.weight.data.copy_(rec_up)
    
    rec_down, bpp_d, mse_d, cos_d, _ = quantize_matrix_h5a_batched(w_q4_down, num_ctrl=3, num_spec=3, spec_mode='sensitivity', p_crit=0.05, seed=42+i)
    l.mlp.down_proj.weight.data.copy_(rec_down)
    
    t_layer = time.time() - t_l0
    layer_stats.append({
        "layer": i,
        "cos_g": cos_g, "mse_g": mse_g,
        "cos_u": cos_u, "mse_u": mse_u,
        "cos_d": cos_d, "mse_d": mse_d,
        "time": t_layer
    })
    
    if (i + 1) % 6 == 0 or i == 0 or i == 23:
        print(f"Camada {i:>2}/23 ({t_layer:.1f}s) | Gate Cos: {cos_g:.4f} | Up Cos: {cos_u:.4f} | Down Cos: {cos_d:.4f}")

total_quant_time = time.time() - t_quant0
print(f"\nQuantização de 72 matrizes Q4 com H5-A concluída em {total_quant_time:.1f}s ({total_quant_time/60:.2f} min)!")

# 3. Avaliar PPL e Geração do Modelo Q4 + H5-A
print("\nAvaliando Perplexidade (PPL) do modelo Q4 comprimido via H5-A...")
ppl_q4_h5a = eval_ppl(model_q4)
print("PPL Q4 + H5-A (3.05 bpp):")
for k, v in ppl_q4_h5a.items():
    print(f"   {k:<12}: {v:.2f}")

print("\nAvaliando Geração do modelo Q4 comprimido via H5-A (3 Prompts)...")
gen_q4_h5a = eval_gen(model_q4)
for k, v in gen_q4_h5a.items():
    print(f"   [{k}]: {v['text']} (Div: {v['uniq']*100:.1f}%)")

# 4. Carregar Resultados Anteriores para a Comparação dos 3
with open("e:/agente/qwen_q4km_comparison_results.json") as f:
    res_comp = json.load(f)

ppl_bf16 = res_comp["ppl_bf16"]
ppl_q4_orig = res_comp["ppl_q4_k_m"]
ppl_h5a_bf16 = res_comp["ppl_h5a"]

gen_bf16 = res_comp["gen_bf16"]
gen_q4_orig = res_comp["gen_q4_k_m"]
gen_h5a_bf16 = res_comp["gen_h5a"]

# 5. TABELA COMPARATIVA DOS 3 MODELOS
print("\n" + "=" * 115)
print("COMPARAÇÃO DIRETA DOS 3 MODELOS: Q4_K_M vs H5-A(BF16) vs H5-A(Q4)")
print("=" * 115)
print(f"{'Métrica / Domínio':<20} | {'1. Q4_K_M Original':<20} | {'2. H5-A (via BF16)':<20} | {'3. H5-A (sobre Q4)':<20} | {'Ref: BF16'}")
print("-" * 115)
print(f"{'Bits / Peso (BPW)':<20} | {'5.45 bpw':<20} | {'3.05 bpw':<20} | {'3.05 bpw':<20} | {'16.0 bpw'}")
print(f"{'Tamanho FFN Relativo':<20} | {'100%':<20} | {'-44% bits':<20} | {'-44% bits':<20} | {'Ref'}")
print("-" * 115)

for d in ["Ciencia", "Codigo", "Historia", "Enciclopedia", "Media"]:
    v_q4 = ppl_q4_orig.get(d, 0.0)
    v_h5a_bf = ppl_h5a_bf16.get(d, 0.0)
    v_h5a_q4 = ppl_q4_h5a.get(d, 0.0)
    v_bf = ppl_bf16.get(d, 0.0)
    print(f"PPL {d:<16} | {v_q4:<20.2f} | {v_h5a_bf:<20.2f} | {v_h5a_q4:<20.2f} | {v_bf:<8.2f}")

print("-" * 115)
print("\n--- GERAÇÃO COMPARATIVA DOS 3 ---")
for k in ["Capital", "Fibonacci", "Newton"]:
    print(f"\nPrompt: [{k}]")
    print(f"  1. Q4_K_M Original : {gen_q4_orig[k]['text']}")
    print(f"  2. H5-A (via BF16) : {gen_h5a_bf16[k]['text']}")
    print(f"  3. H5-A (sobre Q4) : {gen_q4_h5a[k]['text']}")

# 6. Salvar Resultados Estruturados
out_data = {
    "ppl_bf16": ppl_bf16,
    "ppl_q4_orig": ppl_q4_orig,
    "ppl_h5a_via_bf16": ppl_h5a_bf16,
    "ppl_h5a_over_q4": ppl_q4_h5a,
    "gen_bf16": gen_bf16,
    "gen_q4_orig": gen_q4_orig,
    "gen_h5a_via_bf16": gen_h5a_bf16,
    "gen_h5a_over_q4": gen_q4_h5a,
    "layers": layer_stats,
    "quant_time_sec": total_quant_time
}

with open("e:/agente/qwen_h5a_on_q4_results.json", "w") as f:
    json.dump(out_data, f, indent=2)

print("\nResultados salvos em e:/agente/qwen_h5a_on_q4_results.json")
print("=" * 100)
