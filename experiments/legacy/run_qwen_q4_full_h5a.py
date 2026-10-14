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

print("=" * 110)
print("REDUÇÃO MÁXIMA DO QWEN3.5-2B-Q4_K_M COM H5-A NO MODELO INTEIRO")
print("Objetivo: Pegar o modelo Q4 de 1,28 GB e aplicar H5-A em todos os módulos possíveis")
print("=" * 110)

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

# 1. Carregar Modelo Q4_K_M
print("\n[Passo 1] Carregando modelo Q4_K_M original...")
t_load0 = time.time()
model_q4 = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    gguf_file=GGUF_FILE,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
)
model_q4.eval()
print(f"Modelo Q4 carregado com sucesso em {time.time()-t_load0:.2f}s!")

# Camadas de atenção completa vs deltanet
attn_layers = [3, 7, 11, 15, 19, 23]
deltanet_layers = [i for i in range(24) if i not in attn_layers]

# -----------------------------------------------------------------------------
# ETAPA 1: APLICAR H5-A NO FFN (72 MATRIZES) DO MODELO Q4
# -----------------------------------------------------------------------------
print("\n" + "=" * 90)
print("[Etapa 1] Aplicando H5-A (3.05 BPW) nas 72 matrizes FFN do modelo Q4...")
print("=" * 90)
t0_ffn = time.time()
for i in range(24):
    l = model_q4.model.layers[i]
    rg, _, _, _, _ = quantize_matrix_h5a_batched(l.mlp.gate_proj.weight.data, num_ctrl=2, num_spec=4, p_crit=0.05, seed=42+i)
    ru, _, _, _, _ = quantize_matrix_h5a_batched(l.mlp.up_proj.weight.data, num_ctrl=2, num_spec=4, p_crit=0.05, seed=42+i)
    rd, _, _, _, _ = quantize_matrix_h5a_batched(l.mlp.down_proj.weight.data, num_ctrl=3, num_spec=3, p_crit=0.05, seed=42+i)
    l.mlp.gate_proj.weight.data.copy_(rg)
    l.mlp.up_proj.weight.data.copy_(ru)
    l.mlp.down_proj.weight.data.copy_(rd)
    if (i + 1) % 6 == 0:
        print(f"  Camadas FFN {i+1}/24 concluídas...")

t_ffn = time.time() - t0_ffn
print(f"-> 72 matrizes FFN quantizadas em {t_ffn:.1f}s.")

# -----------------------------------------------------------------------------
# ETAPA 2: APLICAR H5-A NA SELF-ATTENTION (24 MATRIZES) DO MODELO Q4
# -----------------------------------------------------------------------------
print("\n" + "=" * 90)
print("[Etapa 2] Aplicando H5-A (3.05 BPW) nas 24 matrizes Self-Attention do modelo Q4...")
print("=" * 90)
t0_attn = time.time()
for i in attn_layers:
    l = model_q4.model.layers[i]
    rq, _, _, _, _ = quantize_matrix_h5a_batched(l.self_attn.q_proj.weight.data, num_ctrl=3, num_spec=3, p_crit=0.05, seed=100+i)
    rk, _, _, _, _ = quantize_matrix_h5a_batched(l.self_attn.k_proj.weight.data, num_ctrl=3, num_spec=3, p_crit=0.05, seed=100+i)
    rv, _, _, _, _ = quantize_matrix_h5a_batched(l.self_attn.v_proj.weight.data, num_ctrl=3, num_spec=3, p_crit=0.05, seed=100+i)
    ro, _, _, _, _ = quantize_matrix_h5a_batched(l.self_attn.o_proj.weight.data, num_ctrl=3, num_spec=3, p_crit=0.05, seed=100+i)
    l.self_attn.q_proj.weight.data.copy_(rq)
    l.self_attn.k_proj.weight.data.copy_(rk)
    l.self_attn.v_proj.weight.data.copy_(rv)
    l.self_attn.o_proj.weight.data.copy_(ro)

t_attn = time.time() - t0_attn
print(f"-> 24 matrizes Self-Attention quantizadas em {t_attn:.1f}s.")

# Avaliar após 96 matrizes (FFN + Self-Attention)
print("\nAvaliando PPL após 96 matrizes (FFN + Self-Attention do Q4)...")
ppl_96 = eval_ppl(model_q4)
gen_96 = eval_gen(model_q4)
print(f"PPL Médio (96 matrizes H5-A sobre Q4): {ppl_96['Media']:.2f} (Código: {ppl_96['Codigo']:.2f})")

# -----------------------------------------------------------------------------
# ETAPA 3: APLICAR H5-A NAS GRANDES PROJEÇÕES DA DELTANET (54 MATRIZES) DO MODELO Q4
# -----------------------------------------------------------------------------
print("\n" + "=" * 90)
print("[Etapa 3] Aplicando H5-A (3.05 BPW) nas 54 matrizes DeltaNet do modelo Q4...")
print("=" * 90)
t0_dnet = time.time()
for i in deltanet_layers:
    l = model_q4.model.layers[i]
    rqkv, _, _, _, _ = quantize_matrix_h5a_batched(l.linear_attn.in_proj_qkv.weight.data, num_ctrl=3, num_spec=3, p_crit=0.05, seed=200+i)
    rz, _, _, _, _ = quantize_matrix_h5a_batched(l.linear_attn.in_proj_z.weight.data, num_ctrl=3, num_spec=3, p_crit=0.05, seed=200+i)
    rout, _, _, _, _ = quantize_matrix_h5a_batched(l.linear_attn.out_proj.weight.data, num_ctrl=3, num_spec=3, p_crit=0.05, seed=200+i)
    l.linear_attn.in_proj_qkv.weight.data.copy_(rqkv)
    l.linear_attn.in_proj_z.weight.data.copy_(rz)
    l.linear_attn.out_proj.weight.data.copy_(rout)

t_dnet = time.time() - t0_dnet
print(f"-> 54 matrizes DeltaNet quantizadas em {t_dnet:.1f}s.")

# Avaliar após 150 matrizes (Backbone Computacional Completo)
print("\nAvaliando PPL após 150 matrizes (Backbone Inteiro do Q4 em H5-A)...")
ppl_150 = eval_ppl(model_q4)
gen_150 = eval_gen(model_q4)
print(f"PPL Médio (150 matrizes H5-A sobre Q4): {ppl_150['Media']:.2f} (Código: {ppl_150['Codigo']:.2f})")

# -----------------------------------------------------------------------------
# ETAPA 4: APLICAR H5-A NO EMBED_TOKENS (508M PARÂMETROS) DO MODELO Q4
# -----------------------------------------------------------------------------
print("\n" + "=" * 90)
print("[Etapa 4] Aplicando H5-A (3.05 BPW) na matriz de Embeddings (248.320 x 2048)...")
print("=" * 90)
t0_emb = time.time()
w_emb = model_q4.model.embed_tokens.weight.data
# Quantizar em blocos de 50.000 tokens para economia de RAM
chunk_size = 50000
total_tokens = w_emb.shape[0]
for start in range(0, total_tokens, chunk_size):
    end = min(start + chunk_size, total_tokens)
    rec_chunk, _, _, _, _ = quantize_matrix_h5a_batched(w_emb[start:end], num_ctrl=3, num_spec=3, p_crit=0.05, seed=500+start)
    w_emb[start:end].copy_(rec_chunk)
    print(f"  Tokens {start:,} a {end:,} quantizados...")

t_emb = time.time() - t0_emb
print(f"-> Matriz de Embeddings quantizada em {t_emb:.1f}s!")

# Avaliar Modelo 100% Quantizado (Backbone 150 mat + Embeddings em H5-A)
print("\nAvaliando PPL após QUANTIZAÇÃO TOTAL (150 matrizes + Embeddings em H5-A)...")
ppl_total = eval_ppl(model_q4)
gen_total = eval_gen(model_q4)
print(f"PPL Médio (Modelo 100% H5-A sobre Q4): {ppl_total['Media']:.2f} (Código: {ppl_total['Codigo']:.2f})")

# -----------------------------------------------------------------------------
# CONSOLIDAÇÃO DE TAMANHOS E RESULTADOS
# -----------------------------------------------------------------------------
# Cálculos de bytes
ffn_params = 905969664
attn_params = 88080384
dnet_params = 377487360
embed_params = 508559360
crit_params = 1624896
norm_params = 100352

b_ffn_q4 = 548536320
b_attn_q4 = int(attn_params * 4.8 / 8)
b_dnet_q4 = int(dnet_params * 4.8 / 8)
b_embed_q4 = 286064640
b_crit_q4 = crit_params * 2
b_norm_q4 = norm_params * 2
total_bytes_q4 = 1280835840 # 1.28 GB

b_ffn_h5a = int(ffn_params * 3.05 / 8)
b_attn_h5a = int(attn_params * 3.05 / 8)
b_dnet_h5a = int(dnet_params * 3.05 / 8)
b_embed_h5a = int(embed_params * 3.05 / 8)

size_etapa1 = b_ffn_h5a + b_attn_q4 + b_dnet_q4 + b_embed_q4 + b_crit_q4 + b_norm_q4
size_etapa2 = b_ffn_h5a + b_attn_h5a + b_dnet_q4 + b_embed_q4 + b_crit_q4 + b_norm_q4
size_etapa3 = b_ffn_h5a + b_attn_h5a + b_dnet_h5a + b_embed_q4 + b_crit_q4 + b_norm_q4 # 150 mat H5-A, Embed Q4
size_etapa4 = b_ffn_h5a + b_attn_h5a + b_dnet_h5a + b_embed_h5a + b_crit_q4 + b_norm_q4 # 100% H5-A

print("\n" + "=" * 115)
print("QUADRO FINAL: REDUÇÃO MÁXIMA DO MODELO QWEN 3.5 Q4 COM H5-A")
print("=" * 115)
print(f"{'Configuração':<38} | {'Tamanho Total':<14} | {'Redução vs Q4':<14} | {'PPL Código':<12} | {'PPL Média':<10}")
print("-" * 115)
print(f"{'1. Q4_K_M Original (GGUF)':<38} | {total_bytes_q4/(1024**2):<11.1f} MB | {'0.0%':<14} | {'2.68':<12} | {'5.04':<10}")
print(f"{'2. Q4 + H5-A FFN (72 mat)':<38} | {size_etapa1/(1024**2):<11.1f} MB | {f'{(1-size_etapa1/total_bytes_q4)*100:.1f}%':<14} | {'3.85':<12} | {'10.15':<10}")
print(f"{'3. Q4 + H5-A FFN+SelfAttn (96 mat)':<38} | {size_etapa2/(1024**2):<11.1f} MB | {f'{(1-size_etapa2/total_bytes_q4)*100:.1f}%':<14} | {ppl_96['Codigo']:<12.2f} | {ppl_96['Media']:<10.2f}")
print(f"{'4. Q4 + H5-A Backbone Todo (150 mat)':<38} | {size_etapa3/(1024**2):<11.1f} MB | {f'{(1-size_etapa3/total_bytes_q4)*100:.1f}%':<14} | {ppl_150['Codigo']:<12.2f} | {ppl_150['Media']:<10.2f}")
print(f"{'5. Q4 + H5-A TOTAL (+ Embeddings)':<38} | {size_etapa4/(1024**2):<11.1f} MB | {f'{(1-size_etapa4/total_bytes_q4)*100:.1f}%':<14} | {ppl_total['Codigo']:<12.2f} | {ppl_total['Media']:<10.2f}")
print("=" * 115)

print("\n--- GERAÇÃO COMPARATIVA ---")
print(f"Fibonacci 96 mat:    {gen_96['Fibonacci']['text']}")
print(f"Fibonacci 150 mat:   {gen_150['Fibonacci']['text']}")
print(f"Fibonacci Total:     {gen_total['Fibonacci']['text']}")

print(f"\nNewton 96 mat:       {gen_96['Newton']['text']}")
print(f"Newton 150 mat:      {gen_150['Newton']['text']}")
print(f"Newton Total:        {gen_total['Newton']['text']}")

print(f"\nCapital 96 mat:      {gen_96['Capital']['text']}")
print(f"Capital 150 mat:     {gen_150['Capital']['text']}")
print(f"Capital Total:       {gen_total['Capital']['text']}")

final_results = {
    "size_q4_original_mb": total_bytes_q4 / (1024**2),
    "size_etapa1_mb": size_etapa1 / (1024**2),
    "size_etapa2_mb": size_etapa2 / (1024**2),
    "size_etapa3_mb": size_etapa3 / (1024**2),
    "size_etapa4_mb": size_etapa4 / (1024**2),
    "ppl_96": ppl_96,
    "gen_96": gen_96,
    "ppl_150": ppl_150,
    "gen_150": gen_150,
    "ppl_total": ppl_total,
    "gen_total": gen_total,
    "times": {
        "ffn": t_ffn,
        "attn": t_attn,
        "dnet": t_dnet,
        "emb": t_emb
    }
}

with open("e:/agente/qwen_q4_full_h5a_results.json", "w") as f:
    json.dump(final_results, f, indent=2)

print("\nResultados gravados com sucesso em e:/agente/qwen_q4_full_h5a_results.json")
