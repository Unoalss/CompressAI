import os
import sys
import time
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from test_qwen_ffn_single_layer import quantize_matrix_h5a_batched

MODEL_DIR = "E:/agente/modelo"

print("=" * 115)
print("EXPERIMENTO EXCLUSIVO: H5-A 3.05 + 2% CANAIS CRÍTICOS EM BF16 -> MODELO COMPLETO -> PPL")
print("Objetivo: Testar se a proteção dos 2% canais de maior erro derruba a PPL no modelo completo")
print("=" * 115)

device = "cpu"
torch.set_num_threads(os.cpu_count() or 8)

# -----------------------------------------------------------------------------
# FUNÇÃO DE QUANTIZAÇÃO H5-A 3.05 BPW COM 2% DE CANAIS CRÍTICOS EM BF16
# -----------------------------------------------------------------------------
def quantize_h5a_protect_channels(W, p_channel=0.02, num_ctrl=3, num_spec=3, p_crit=0.05, seed=42):
    orig_dtype = W.dtype
    # 1. Quantização H5-A base a 3.05 BPW
    W_rec, bpp, mse, cos, _ = quantize_matrix_h5a_batched(
        W, num_ctrl=num_ctrl, num_spec=num_spec, p_crit=p_crit, seed=seed
    )
    
    # 2. Identificar os 2% canais de entrada (colunas) que concentram maior energia de erro
    E = W.float() - W_rec.float()
    col_err = torch.sum(E ** 2, dim=0) # shape: (in_features,)
    col_norm = torch.norm(W.float(), dim=0)
    col_sens = col_err * (col_norm + 1e-5) # Sensibilidade ponderada por norma
    
    in_features = W.shape[1]
    num_crit = max(1, int(p_channel * in_features))
    crit_cols = torch.topk(col_sens, k=num_crit).indices
    
    # 3. Restaurar as colunas críticas em BF16 exato
    W_rec[:, crit_cols] = W[:, crit_cols]
    
    effective_bpw = (1.0 - p_channel) * 3.05 + p_channel * 16.0
    return W_rec.to(orig_dtype), effective_bpw

# -----------------------------------------------------------------------------
# PIPELINE DE AVALIAÇÃO
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# 1. CARREGAR MODELO BF16 ORIGINAL
# -----------------------------------------------------------------------------
print("\n[Passo 1] Carregando modelo base Qwen3.5-2B em BF16...")
t0_load = time.time()
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
model.eval()
print(f"-> Modelo carregado em {time.time()-t0_load:.2f}s!")

attn_layers = [3, 7, 11, 15, 19, 23]
deltanet_layers = [i for i in range(24) if i not in attn_layers]

# -----------------------------------------------------------------------------
# 2. QUANTIZAR FFN (72 MATRIZES): H5-A 3.05 + 2% CANAIS BF16
# -----------------------------------------------------------------------------
print("\n[Passo 2] Quantizando 72 matrizes FFN (H5-A 3.05 + 2% canais BF16)...")
t0_ffn = time.time()
for i in range(24):
    l = model.model.layers[i]
    rg, _ = quantize_h5a_protect_channels(l.mlp.gate_proj.weight.data, p_channel=0.02, num_ctrl=2, num_spec=4, p_crit=0.05, seed=42+i)
    ru, _ = quantize_h5a_protect_channels(l.mlp.up_proj.weight.data, p_channel=0.02, num_ctrl=2, num_spec=4, p_crit=0.05, seed=42+i)
    rd, _ = quantize_h5a_protect_channels(l.mlp.down_proj.weight.data, p_channel=0.02, num_ctrl=3, num_spec=3, p_crit=0.05, seed=42+i)
    l.mlp.gate_proj.weight.data.copy_(rg)
    l.mlp.up_proj.weight.data.copy_(ru)
    l.mlp.down_proj.weight.data.copy_(rd)
    if (i + 1) % 6 == 0:
        print(f"  Camadas FFN {i+1}/24 concluídas...")
t_ffn = time.time() - t0_ffn
print(f"-> 72 matrizes FFN concluídas em {t_ffn:.1f}s.")

# -----------------------------------------------------------------------------
# 3. QUANTIZAR SELF-ATTENTION (24 MATRIZES): H5-A 3.05 + 2% CANAIS BF16
# -----------------------------------------------------------------------------
print("\n[Passo 3] Quantizando 24 matrizes Self-Attention (H5-A 3.05 + 2% canais BF16)...")
t0_attn = time.time()
for i in attn_layers:
    l = model.model.layers[i]
    rq, _ = quantize_h5a_protect_channels(l.self_attn.q_proj.weight.data, p_channel=0.02, num_ctrl=3, num_spec=3, p_crit=0.05, seed=100+i)
    rk, _ = quantize_h5a_protect_channels(l.self_attn.k_proj.weight.data, p_channel=0.02, num_ctrl=3, num_spec=3, p_crit=0.05, seed=100+i)
    rv, _ = quantize_h5a_protect_channels(l.self_attn.v_proj.weight.data, p_channel=0.02, num_ctrl=3, num_spec=3, p_crit=0.05, seed=100+i)
    ro, _ = quantize_h5a_protect_channels(l.self_attn.o_proj.weight.data, p_channel=0.02, num_ctrl=3, num_spec=3, p_crit=0.05, seed=100+i)
    l.self_attn.q_proj.weight.data.copy_(rq)
    l.self_attn.k_proj.weight.data.copy_(rk)
    l.self_attn.v_proj.weight.data.copy_(rv)
    l.self_attn.o_proj.weight.data.copy_(ro)
t_attn = time.time() - t0_attn
print(f"-> 24 matrizes Self-Attention concluídas em {t_attn:.1f}s.")

# -----------------------------------------------------------------------------
# 4. QUANTIZAR DELTANET (54 MATRIZES): H5-A 3.05 + 2% CANAIS BF16
# -----------------------------------------------------------------------------
print("\n[Passo 4] Quantizando 54 matrizes DeltaNet (H5-A 3.05 + 2% canais BF16)...")
t0_dnet = time.time()
for i in deltanet_layers:
    l = model.model.layers[i]
    rqkv, _ = quantize_h5a_protect_channels(l.linear_attn.in_proj_qkv.weight.data, p_channel=0.02, num_ctrl=3, num_spec=3, p_crit=0.05, seed=200+i)
    rz, _ = quantize_h5a_protect_channels(l.linear_attn.in_proj_z.weight.data, p_channel=0.02, num_ctrl=3, num_spec=3, p_crit=0.05, seed=200+i)
    rout, _ = quantize_h5a_protect_channels(l.linear_attn.out_proj.weight.data, p_channel=0.02, num_ctrl=3, num_spec=3, p_crit=0.05, seed=200+i)
    l.linear_attn.in_proj_qkv.weight.data.copy_(rqkv)
    l.linear_attn.in_proj_z.weight.data.copy_(rz)
    l.linear_attn.out_proj.weight.data.copy_(rout)
t_dnet = time.time() - t0_dnet
print(f"-> 54 matrizes DeltaNet concluídas em {t_dnet:.1f}s.")

# -----------------------------------------------------------------------------
# 5. AVALIAÇÃO: MODELO COMPLETO (150 MATRIZES H5-A 3.05 + 2% CANAIS BF16)
# -----------------------------------------------------------------------------
print("\n" + "=" * 90)
print("AVALIANDO PPL DO BACKBONE COMPLETO (150 MATRIZES COM 2% CANAIS BF16)...")
print("=" * 90)
ppl_pure_backbone = eval_ppl(model)
gen_pure_backbone = eval_gen(model)

print("\n--- PPL (150 MATRIZES COM 2% CANAIS BF16 + EMBEDDINGS BF16) ---")
for k, v in ppl_pure_backbone.items():
    print(f"  {k:<14}: {v:.2f}")

print("\n--- GERAÇÃO NO BACKBONE COM 2% CANAIS BF16 ---")
for k in ["Capital", "Fibonacci", "Newton"]:
    print(f"  [{k}]: {gen_pure_backbone[k]['text']}")

# -----------------------------------------------------------------------------
# 6. AVALIAÇÃO TAMBÉM COM EMBEDDINGS HÍBRIDOS (TOP 32K BF16 / RESTO 3.05)
# -----------------------------------------------------------------------------
print("\n" + "=" * 90)
print("AVALIANDO TAMBÉM COM EMBEDDINGS HÍBRIDOS (ARQUIVO < 850 MB)...")
print("=" * 90)
t0_emb = time.time()
w_emb = model.model.embed_tokens.weight.data
total_tokens = w_emb.shape[0]
chunk_size = 50000
for start in range(32768, total_tokens, chunk_size):
    end = min(start + chunk_size, total_tokens)
    rec, _, _, _, _ = quantize_matrix_h5a_batched(w_emb[start:end], num_ctrl=3, num_spec=3, p_crit=0.05, seed=500+start)
    w_emb[start:end].copy_(rec)
t_emb = time.time() - t0_emb
print(f"-> Embeddings híbridos quantizados em {t_emb:.1f}s.")

ppl_full_850mb = eval_ppl(model)
gen_full_850mb = eval_gen(model)

print("\n--- PPL (MODELO 100% COMPRIMIDO < 850 MB COM 2% CANAIS BF16) ---")
for k, v in ppl_full_850mb.items():
    print(f"  {k:<14}: {v:.2f}")

print("\n--- GERAÇÃO (MODELO 100% COMPRIMIDO < 850 MB) ---")
for k in ["Capital", "Fibonacci", "Newton"]:
    print(f"  [{k}]: {gen_full_850mb[k]['text']}")

# -----------------------------------------------------------------------------
# CÁLCULO EXATO DE BYTES
# -----------------------------------------------------------------------------
ffn_params = 905969664
attn_params = 88080384
dnet_params = 377487360
crit_params = 1624896
norm_params = 100352
embed_bf16_tokens = 32768
embed_h5a_tokens = 248320 - 32768

# Taxa efetiva com 2% canais BF16:
bpw_prot = 0.98 * 3.05 + 0.02 * 16.0 # 3.309 BPW

b_ffn = int(ffn_params * bpw_prot / 8) # 374,8 MB (357,5 MiB)
b_attn = int(attn_params * bpw_prot / 8) # 36,4 MB (34,7 MiB)
b_dnet = int(dnet_params * bpw_prot / 8) # 156,1 MB (148,9 MiB)
b_crit = crit_params * 2 # 3,25 MB (3,10 MiB)
b_norm = norm_params * 2 # 0,20 MB (0,19 MiB)
b_embed_bf16 = embed_bf16_tokens * 2048 * 2 # 134,2 MB (128,0 MiB)
b_embed_h5a = int(embed_h5a_tokens * 2048 * 3.05 / 8) # 168,3 MB (160,5 MiB)
b_embed_total = b_embed_bf16 + b_embed_h5a # 302,5 MB (288,5 MiB)

total_bytes = b_ffn + b_attn + b_dnet + b_crit + b_norm + b_embed_total
size_mb = total_bytes / (1024**2)

print("\n" + "=" * 115)
print("QUADRO FINAL: IMPACTO DOS 2% DE CANAIS CRÍTICOS EM BF16")
print("=" * 115)
print(f"Componente                      | Parâmetros   | Taxa / Formato      | Tamanho em MB | % do Total")
print("-" * 115)
print(f"FFN (24 camadas, 72 matrizes)   | {ffn_params:<12,d} | H5-A (3.31 BPW)     | {b_ffn/(1024**2):<11.1f} MB | {b_ffn/total_bytes*100:.1f}%")
print(f"Self-Attention (6 cam, 24 mat)  | {attn_params:<12,d} | H5-A (3.31 BPW)     | {b_attn/(1024**2):<11.1f} MB | {b_attn/total_bytes*100:.1f}%")
print(f"DeltaNet Grandes (18 cam, 54)   | {dnet_params:<12,d} | H5-A (3.31 BPW)     | {b_dnet/(1024**2):<11.1f} MB | {b_dnet/total_bytes*100:.1f}%")
print(f"DeltaNet Críticos (decay/conv)  | {crit_params:<12,d} | BF16 (16.0 BPW)     | {b_crit/(1024**2):<11.2f} MB | {b_crit/total_bytes*100:.1f}%")
print(f"Embeddings Top 32k tokens       | {embed_bf16_tokens*2048:<12,d} | BF16 (16.0 BPW)     | {b_embed_bf16/(1024**2):<11.1f} MB | {b_embed_bf16/total_bytes*100:.1f}%")
print(f"Embeddings Resto (215k tokens)  | {embed_h5a_tokens*2048:<12,d} | H5-A (3.05 BPW)     | {b_embed_h5a/(1024**2):<11.1f} MB | {b_embed_h5a/total_bytes*100:.1f}%")
print(f"RMSNorms (LayerNorms)           | {norm_params:<12,d} | BF16 (16.0 BPW)     | {b_norm/(1024**2):<11.2f} MB | {b_norm/total_bytes*100:.1f}%")
print("-" * 115)
print(f"TOTAL DO MODELO FINAL           | 1,881,822,016| ~3,64 BPW Médio     | {size_mb:<11.1f} MB | 100.0%")
print(f"-> Tamanho Final: {size_mb:.1f} MB (Ficou abaixo de 850 MB? {'SIM' if size_mb < 850 else 'NAO'})")
print("=" * 115)

final_data = {
    "size_mb": size_mb,
    "size_bytes": total_bytes,
    "ppl_backbone": ppl_pure_backbone,
    "gen_backbone": gen_pure_backbone,
    "ppl_850mb": ppl_full_850mb,
    "gen_850mb": gen_full_850mb,
    "times": {
        "ffn": t_ffn,
        "attn": t_attn,
        "dnet": t_dnet,
        "emb": t_emb
    }
}

with open("e:/agente/qwen_h5a_channel_protected_results.json", "w") as f:
    json.dump(final_data, f, indent=2)

print("\nResultados gravados em e:/agente/qwen_h5a_channel_protected_results.json")
