import os
import sys
import time
import json
import gc
import psutil
import torch
import gguf
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
from test_qwen_ffn_single_layer import quantize_matrix_h5a_batched

MODEL_DIR = "E:/agente/modelo"
GGUF_FILE = "Qwen3.5-2B-Q4_K_M.gguf"
GGUF_PATH = os.path.join(MODEL_DIR, GGUF_FILE)

print("=" * 105)
print("EXPERIMENTO H5-A NO MODELO COMPLETO: EXP. A (FFN + SELF-ATTN) vs EXP. B (+ DELTANET LINEAR)")
print("=" * 105)

device = "cpu"
torch.set_num_threads(os.cpu_count() or 8)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

# -----------------------------------------------------------------------------
# PASSO 1: DIAGNÓSTICO EXATO E MAPEAMENTO DE MEBIBYTES ANTES DA QUANTIZAÇÃO
# -----------------------------------------------------------------------------
print("\n" + "=" * 90)
print("1. MAPEAMENTO DETALHADO DE COMPONENTES E ONDE ESTÃO OS MEGABYTES")
print("=" * 90)

reader = gguf.GGUFReader(GGUF_PATH)
gguf_sizes = {}
for tensor in reader.tensors:
    gguf_sizes[tensor.name] = int(tensor.n_bytes)

# Contagem analítica no modelo
ffn_params = 24 * (6144 * 2048 * 2 + 2048 * 6144) # 905,969,664
attn_params = 6 * (4096 * 2048 + 512 * 2048 * 2 + 2048 * 2048) # 88,080,384
deltanet_large_params = 18 * (6144 * 2048 + 2048 * 2048 * 2) # 377,487,360
deltanet_crit_params = 18 * (16 * 2048 * 2 + 6144 * 4 + 16 * 2 + 128) # 1,624,896
embed_params = 248320 * 2048 # 508,559,360
norms_params = 49 * 2048 # ~100k

# Tamanhos em bytes
b_ffn_bf16 = ffn_params * 2
b_attn_bf16 = attn_params * 2
b_dnet_large_bf16 = deltanet_large_params * 2
b_dnet_crit_bf16 = deltanet_crit_params * 2
b_embed_bf16 = embed_params * 2
b_norms_bf16 = norms_params * 2
b_total_bf16 = b_ffn_bf16 + b_attn_bf16 + b_dnet_large_bf16 + b_dnet_crit_bf16 + b_embed_bf16 + b_norms_bf16
total_params = ffn_params + attn_params + deltanet_large_params + deltanet_crit_params + embed_params + norms_params

# Estimativas no Q4
b_ffn_q4 = 548536320
b_attn_q4 = int(88080384 * 4.8 / 8) # ~52.8 MB
b_dnet_large_q4 = int(377487360 * 4.8 / 8) # ~226.5 MB
b_dnet_crit_q4 = b_dnet_crit_bf16 # mantido em alta precisão
b_embed_q4 = int(embed_params * 4.5 / 8) # ~286 MB
b_norms_q4 = b_norms_bf16
b_total_q4 = 1269873920

# H5-A a 3.05 BPW
b_ffn_h5a = int(ffn_params * 3.05 / 8)
b_attn_h5a = int(attn_params * 3.05 / 8)
b_dnet_large_h5a = int(deltanet_large_params * 3.05 / 8)

# Totais das estratégias
# Exp A: FFN + Self-Attn H5-A (DeltaNet + Embed em BF16)
b_total_exp_a = b_ffn_h5a + b_attn_h5a + b_dnet_large_bf16 + b_dnet_crit_bf16 + b_embed_bf16 + b_norms_bf16
# Exp B: FFN + Self-Attn + DeltaNet Grandes H5-A (DeltaNet Crítica + Embed em BF16)
b_total_exp_b = b_ffn_h5a + b_attn_h5a + b_dnet_large_h5a + b_dnet_crit_bf16 + b_embed_bf16 + b_norms_bf16

print(f"{'Componente':<30} | {'Parâmetros':<14} | {'Tamanho BF16':<14} | {'Tamanho Q4':<14} | {'H5-A (3.05 BPW)':<16} | {'Tratamento Proposto'}")
print("-" * 115)
print(f"{'FFN (24 camadas, 72 mat)':<30} | {ffn_params:<14,d} | {b_ffn_bf16/(1024**2):<11.1f} MB | {b_ffn_q4/(1024**2):<11.1f} MB | {b_ffn_h5a/(1024**2):<13.1f} MB | H5-A (3.05 bpw)")
print(f"{'Self-Attention (6 cam, 24 mat)':<30} | {attn_params:<14,d} | {b_attn_bf16/(1024**2):<11.1f} MB | {b_attn_q4/(1024**2):<11.1f} MB | {b_attn_h5a/(1024**2):<13.1f} MB | H5-A (3.05 bpw)")
print(f"{'DeltaNet Grandes (18 cam, 54)':<30} | {deltanet_large_params:<14,d} | {b_dnet_large_bf16/(1024**2):<11.1f} MB | {b_dnet_large_q4/(1024**2):<11.1f} MB | {b_dnet_large_h5a/(1024**2):<13.1f} MB | H5-A no Exp. B")
print(f"{'DeltaNet Crítica (conv/decay)':<30} | {deltanet_crit_params:<14,d} | {b_dnet_crit_bf16/(1024**2):<11.2f} MB | {b_dnet_crit_q4/(1024**2):<11.2f} MB | {b_dnet_crit_bf16/(1024**2):<13.2f} MB | Blindado em BF16")
print(f"{'Embeddings & LM Head (vocab)':<30} | {embed_params:<14,d} | {b_embed_bf16/(1024**2):<11.1f} MB | {b_embed_q4/(1024**2):<11.1f} MB | {b_embed_bf16/(1024**2):<13.1f} MB | Blindado em BF16")
print(f"{'Normalizações (RMSNorm)':<30} | {norms_params:<14,d} | {b_norms_bf16/(1024**2):<11.2f} MB | {b_norms_q4/(1024**2):<11.2f} MB | {b_norms_bf16/(1024**2):<13.2f} MB | Blindado em BF16")
print("-" * 115)
print(f"{'TOTAL BACKBONE':<30} | {total_params:<14,d} | {b_total_bf16/(1024**2):<11.1f} MB | {b_total_q4/(1024**2):<11.1f} MB | {b_total_exp_b/(1024**2):<13.1f} MB | ")
print(f"\nOnde estão os megabytes que impedem o arquivo de ser menor que ~1 GB?")
print(f" -> O Embedding possui 248.320 tokens x 2048 = 508.559.360 parâmetros.")
print(f"    Em BF16, ele sozinho consome {b_embed_bf16/(1024**2):.1f} MB (quase 1 GB!). No Q4, ele foi quantizado para {b_embed_q4/(1024**2):.1f} MB.")
print(f" -> O Backbone sem Embeddings em BF16 consome {(b_total_bf16 - b_embed_bf16)/(1024**2):.1f} MB.")
print(f" -> Com H5-A a 3.05 bpw (Exp. B), todo o Backbone computacional cai para apenas {(b_ffn_h5a + b_attn_h5a + b_dnet_large_h5a)/(1024**2):.1f} MB!")

# -----------------------------------------------------------------------------
# PIPELINE DE AVALIAÇÃO
# -----------------------------------------------------------------------------
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
            t0 = time.time()
            out = m.generate(**inp, max_new_tokens=max_tokens, do_sample=False)
            t_gen = time.time() - t0
            txt = tokenizer.decode(out[0], skip_special_tokens=True)
            toks = tokenizer.encode(txt)
            uniq = len(set(toks)) / len(toks)
            n_tokens = len(out[0]) - len(inp.input_ids[0])
            gens[name] = {
                "text": txt.replace("\n", "\\n")[:80],
                "uniq": uniq,
                "tok_per_sec": n_tokens / max(t_gen, 1e-4)
            }
    return gens

# Carregar modelo base BF16
print("\nCarregando modelo base BF16...")
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
model.eval()

# Armazenar cópia das matrizes originais
print("Guardando cópias dos pesos originais para permitir os experimentos controlados...")
orig_weights = {}
for i in range(24):
    l = model.model.layers[i]
    orig_weights[f"l{i}.gate"] = l.mlp.gate_proj.weight.data.clone()
    orig_weights[f"l{i}.up"] = l.mlp.up_proj.weight.data.clone()
    orig_weights[f"l{i}.down"] = l.mlp.down_proj.weight.data.clone()
    
    if hasattr(l, "self_attn"): # Camadas 3, 7, 11, 15, 19, 23
        orig_weights[f"l{i}.q_proj"] = l.self_attn.q_proj.weight.data.clone()
        orig_weights[f"l{i}.k_proj"] = l.self_attn.k_proj.weight.data.clone()
        orig_weights[f"l{i}.v_proj"] = l.self_attn.v_proj.weight.data.clone()
        orig_weights[f"l{i}.o_proj"] = l.self_attn.o_proj.weight.data.clone()
    elif hasattr(l, "linear_attn"): # 18 camadas DeltaNet
        orig_weights[f"l{i}.in_proj_qkv"] = l.linear_attn.in_proj_qkv.weight.data.clone()
        orig_weights[f"l{i}.in_proj_z"] = l.linear_attn.in_proj_z.weight.data.clone()
        orig_weights[f"l{i}.out_proj"] = l.linear_attn.out_proj.weight.data.clone()

# -----------------------------------------------------------------------------
# EXPERIMENTO A: FFN + SELF-ATTENTION H5-A (3.05 BPW)
# DeltaNet e Embeddings em BF16
# -----------------------------------------------------------------------------
print("\n" + "=" * 90)
print("EXPERIMENTO A: FFN (72 MATRIZES) + SELF-ATTENTION (24 MATRIZES) -> H5-A (3.05 BPW)")
print("Total: 96 matrizes quantizadas com H5-A (~994 Milhões de parâmetros)")
print("DeltaNet (18 camadas) e Embeddings mantidos em BF16")
print("=" * 90)

t0_exp_a = time.time()
attn_layers = [3, 7, 11, 15, 19, 23]

# 1. Quantizar FFN nas 24 camadas
for i in range(24):
    l = model.model.layers[i]
    rg, _, _, _, _ = quantize_matrix_h5a_batched(orig_weights[f"l{i}.gate"], num_ctrl=2, num_spec=4, p_crit=0.05, seed=42+i)
    ru, _, _, _, _ = quantize_matrix_h5a_batched(orig_weights[f"l{i}.up"], num_ctrl=2, num_spec=4, p_crit=0.05, seed=42+i)
    rd, _, _, _, _ = quantize_matrix_h5a_batched(orig_weights[f"l{i}.down"], num_ctrl=3, num_spec=3, p_crit=0.05, seed=42+i)
    l.mlp.gate_proj.weight.data.copy_(rg)
    l.mlp.up_proj.weight.data.copy_(ru)
    l.mlp.down_proj.weight.data.copy_(rd)

# 2. Quantizar Self-Attention nas 6 camadas
for i in attn_layers:
    l = model.model.layers[i]
    rq, _, _, _, _ = quantize_matrix_h5a_batched(orig_weights[f"l{i}.q_proj"], num_ctrl=3, num_spec=3, p_crit=0.05, seed=100+i)
    rk, _, _, _, _ = quantize_matrix_h5a_batched(orig_weights[f"l{i}.k_proj"], num_ctrl=3, num_spec=3, p_crit=0.05, seed=100+i)
    rv, _, _, _, _ = quantize_matrix_h5a_batched(orig_weights[f"l{i}.v_proj"], num_ctrl=3, num_spec=3, p_crit=0.05, seed=100+i)
    ro, _, _, _, _ = quantize_matrix_h5a_batched(orig_weights[f"l{i}.o_proj"], num_ctrl=3, num_spec=3, p_crit=0.05, seed=100+i)
    l.self_attn.q_proj.weight.data.copy_(rq)
    l.self_attn.k_proj.weight.data.copy_(rk)
    l.self_attn.v_proj.weight.data.copy_(rv)
    l.self_attn.o_proj.weight.data.copy_(ro)

t_quant_exp_a = time.time() - t0_exp_a
print(f"Quantização do Experimento A (96 matrizes) concluída em {t_quant_exp_a:.1f}s ({t_quant_exp_a/60:.2f} min)!")

# Avaliar Exp A
print("Avaliando Perplexidade e Geração do Experimento A...")
ppl_exp_a = eval_ppl(model)
gen_exp_a = eval_gen(model)

print("PPL Experimento A (FFN + Self-Attn H5-A):")
for k, v in ppl_exp_a.items():
    print(f"   {k:<12}: {v:.2f}")

# -----------------------------------------------------------------------------
# EXPERIMENTO B: FFN + SELF-ATTN + GRANDES PROJEÇÕES DELTANET H5-A (3.05 BPW)
# DeltaNet Crítica (A_log, dt_bias, conv1d, in_proj_a/b) e Embeddings em BF16
# -----------------------------------------------------------------------------
print("\n" + "=" * 90)
print("EXPERIMENTO B: FFN (72) + SELF-ATTN (24) + DELTANET GRANDES (54) -> H5-A (3.05 BPW)")
print("Total: 150 MATRIZES QUANTIZADAS COM H5-A (~1,37 BILHÃO DE PARÂMETROS)")
print("DeltaNet Crítica (SSM Recorrência) e Embeddings mantidos blindados em BF16")
print("=" * 90)

t0_exp_b = time.time()
deltanet_layers = [i for i in range(24) if i not in attn_layers]

# Quantizar as 54 grandes matrizes de DeltaNet (in_proj_qkv, in_proj_z, out_proj)
for i in deltanet_layers:
    l = model.model.layers[i]
    rqkv, _, _, _, _ = quantize_matrix_h5a_batched(orig_weights[f"l{i}.in_proj_qkv"], num_ctrl=3, num_spec=3, p_crit=0.05, seed=200+i)
    rz, _, _, _, _ = quantize_matrix_h5a_batched(orig_weights[f"l{i}.in_proj_z"], num_ctrl=3, num_spec=3, p_crit=0.05, seed=200+i)
    rout, _, _, _, _ = quantize_matrix_h5a_batched(orig_weights[f"l{i}.out_proj"], num_ctrl=3, num_spec=3, p_crit=0.05, seed=200+i)
    l.linear_attn.in_proj_qkv.weight.data.copy_(rqkv)
    l.linear_attn.in_proj_z.weight.data.copy_(rz)
    l.linear_attn.out_proj.weight.data.copy_(rout)

t_quant_exp_b = time.time() - t0_exp_b
print(f"Quantização adicional de 54 matrizes DeltaNet concluída em {t_quant_exp_b:.1f}s!")

# Avaliar Exp B
print("Avaliando Perplexidade e Geração do Experimento B...")
ppl_exp_b = eval_ppl(model)
gen_exp_b = eval_gen(model)

print("PPL Experimento B (FFN + Self-Attn + DeltaNet Grandes H5-A):")
for k, v in ppl_exp_b.items():
    print(f"   {k:<12}: {v:.2f}")

# -----------------------------------------------------------------------------
# TABELA COMPARATIVA GERAL E SÍNTESE
# -----------------------------------------------------------------------------
with open("e:/agente/qwen_rate_distortion_sweep_results.json") as f:
    r_prev = json.load(f)

ppl_bf16 = r_prev["ppl_bf16"]
ppl_q4_ffn = r_prev["ppl_q4_ffn"]
ppl_h5a_ffn = r_prev["ppl_h5a_305"]

with open("e:/agente/qwen_q4km_comparison_results.json") as f:
    r_q4_full = json.load(f)
ppl_q4_full = r_q4_full["ppl_q4_k_m"]

print("\n" + "=" * 120)
print("QUADRO COMPARATIVO CONSOLIDADO: DA COMPRESSÃO FFN AO MODELO COMPLETO H5-A")
print("=" * 120)
print(f"{'Configuração':<32} | {'Matrizes H5-A':<14} | {'Params H5-A':<12} | {'PPL Ciência':<11} | {'PPL Código':<11} | {'PPL Média':<10}")
print("-" * 120)
print(f"{'BF16 Original':<32} | {'0':<14} | {'0 M':<12} | {ppl_bf16['Ciencia']:<11.2f} | {ppl_bf16['Codigo']:<11.2f} | {ppl_bf16['Media']:<10.2f}")
print(f"{'Q4_K_M (Modelo GGUF Completo)':<32} | {'0 (Q4)':<14} | {'1.88 B (Q4)':<12} | {ppl_q4_full['Ciencia']:<11.2f} | {ppl_q4_full['Codigo']:<11.2f} | {ppl_q4_full['Media']:<10.2f}")
print(f"{'H5-A FFN-Only (3.05 bpw)':<32} | {'72':<14} | {'906 M':<12} | {ppl_h5a_ffn['Ciencia']:<11.2f} | {ppl_h5a_ffn['Codigo']:<11.2f} | {ppl_h5a_ffn['Media']:<10.2f}")
print(f"{'Exp. A: FFN + Self-Attn H5-A':<32} | {'96':<14} | {'994 M':<12} | {ppl_exp_a['Ciencia']:<11.2f} | {ppl_exp_a['Codigo']:<11.2f} | {ppl_exp_a['Media']:<10.2f}")
print(f"{'Exp. B: FFN+Attn+DeltaNet H5-A':<32} | {'150':<14} | {'1.371 M':<12} | {ppl_exp_b['Ciencia']:<11.2f} | {ppl_exp_b['Codigo']:<11.2f} | {ppl_exp_b['Media']:<10.2f}")
print("=" * 120)

print("\n--- GERAÇÃO COMPARATIVA NOS 3 PROMPTS ---")
for k in ["Capital", "Fibonacci", "Newton"]:
    print(f"\nPrompt: [{k}]")
    print(f"  Exp. A (FFN + Self-Attn): {gen_exp_a[k]['text']} ({gen_exp_a[k]['tok_per_sec']:.2f} tok/s)")
    print(f"  Exp. B (Modelo Completo): {gen_exp_b[k]['text']} ({gen_exp_b[k]['tok_per_sec']:.2f} tok/s)")

results_full = {
    "exp_a_ppl": ppl_exp_a,
    "exp_b_ppl": ppl_exp_b,
    "exp_a_gen": gen_exp_a,
    "exp_b_gen": gen_exp_b,
    "time_quant_a": t_quant_exp_a,
    "time_quant_b": t_quant_exp_b
}

with open("e:/agente/qwen_full_model_h5a_results.json", "w") as f:
    json.dump(results_full, f, indent=2)

print("\nResultados salvos em e:/agente/qwen_full_model_h5a_results.json")
print("=" * 105)
