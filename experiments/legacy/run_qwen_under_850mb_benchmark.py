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

print("=" * 115)
print("BENCHMARK H5-A < 850 MB: ALTA FIDELIDADE PRÓXIMA DO Q4_K_M")
print("Estratégia: FFN 3.05 + Attn 3.05 + DeltaNet 4.05 + Embeddings Híbridos (Top 32k BF16 / Resto 3.05)")
print("Alvo de Tamanho: < 850 MB | Alvo de Qualidade: PPL próxima de 5-8")
print("=" * 115)

device = "cpu"
torch.set_num_threads(os.cpu_count() or 8)

# -----------------------------------------------------------------------------
# FUNÇÕES H5-A 4.05 BPW
# -----------------------------------------------------------------------------
def train_kmeans(vectors, K=26, n_iter=6, seed=42):
    torch.manual_seed(seed)
    N, D = vectors.shape
    if N <= K:
        if N == 0: return torch.zeros(K, D)
        rep = (K // N) + 1
        return vectors.repeat(rep, 1)[:K]
    
    centers = [vectors[torch.randint(0, N, (1,))[0]]]
    for _ in range(1, K):
        sub_idx = torch.randint(0, N, (min(N, 2000),))
        sub_vecs = vectors[sub_idx]
        cur_centers = torch.stack(centers)
        dists = torch.cdist(sub_vecs, cur_centers)
        min_dists, _ = torch.min(dists, dim=1)
        probs = min_dists ** 2
        probs_sum = probs.sum()
        if probs_sum > 0:
            probs = probs / probs_sum
            next_idx = sub_idx[torch.multinomial(probs, 1)[0]]
            centers.append(vectors[next_idx])
        else:
            centers.append(vectors[torch.randint(0, N, (1,))[0]])
            
    c_core = torch.stack(centers)
    sub_sample = vectors[:min(N, 20000)]
    for _ in range(n_iter):
        d = torch.cdist(sub_sample, c_core)
        lbl = torch.argmin(d, dim=1)
        for k in range(K):
            m = (lbl == k)
            if m.sum() > 0:
                c_core[k] = sub_sample[m].mean(dim=0)
    return c_core

def quantize_matrix_h5a_4bpw(weight_tensor, num_ctrl=3, num_spec=3, p_crit=0.05, seed=42, block_size=8):
    torch.manual_seed(seed)
    orig_shape = weight_tensor.shape
    orig_dtype = weight_tensor.dtype
    w_flat = weight_tensor.float().reshape(-1, block_size)
    N = w_flat.shape[0]
    batch_size = 200000
    
    sample_n = min(N, 50000)
    perm_sample = torch.randperm(N)[:sample_n]
    w_sample = w_flat[perm_sample]
    
    # 1. Macro Famílias
    perm_fam = torch.randperm(sample_n)[:8]
    fam_centers = w_sample[perm_fam].clone()
    for _ in range(5):
        d = torch.cdist(w_sample[:10000], fam_centers)
        lbl = torch.argmin(d, dim=1)
        for k in range(8):
            m = (lbl == k)
            if m.sum() > 0: fam_centers[k] = w_sample[:10000][m].mean(dim=0)
                
    fam_ids = torch.empty(N, dtype=torch.long)
    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        fam_ids[i:end_i] = torch.argmin(torch.cdist(w_flat[i:end_i], fam_centers), dim=1)
        
    sample_fam_ids = fam_ids[perm_sample]
    libs_macro = [train_kmeans(w_sample[sample_fam_ids == k], K=26, seed=seed) for k in range(8)]
    
    libs_spec = []
    for k in range(8):
        m_k = (sample_fam_ids == k)
        vecs_k = w_sample[m_k]
        if vecs_k.shape[0] < num_spec:
            libs_spec.append(fam_centers[k].unsqueeze(0).repeat(num_spec, 1))
            continue
            
        d_base = torch.cdist(vecs_k, libs_macro[k])
        min_err, _ = torch.min(d_base, dim=1)
        norms = torch.norm(vecs_k, dim=1)
        crit_score = (norms + 1e-5) * (min_err ** 2)
        top_k = max(num_spec, int(vecs_k.shape[0] * 0.35))
        crit_idx = torch.topk(crit_score, k=top_k).indices
        spec_k = train_kmeans(vecs_k[crit_idx], K=num_spec, seed=seed)
        libs_spec.append(spec_k)
        
    macro_recon = torch.empty_like(w_flat)
    for k in range(8):
        m_full = (fam_ids == k)
        count_k = m_full.sum().item()
        if count_k == 0: continue
        w_k = w_flat[m_full]
        lib_base = libs_macro[k]
        lib_spec = libs_spec[k]
        
        recon_k = torch.empty_like(w_k)
        sub_b = 50000
        for sb_start in range(0, count_k, sub_b):
            sb_end = min(sb_start + sub_b, count_k)
            w_sub = w_k[sb_start:sb_end]
            
            d_base = torch.cdist(w_sub, lib_base)
            min_base, idx_base = torch.min(d_base, dim=1)
            
            d_c0 = torch.cdist(w_sub, lib_base * 1.5)
            d_c1 = torch.cdist(w_sub, lib_base * 0.5)
            d_c2 = torch.cdist(w_sub, -lib_base)
            min_c0, idx_c0 = torch.min(d_c0, dim=1)
            min_c1, idx_c1 = torch.min(d_c1, dim=1)
            min_c2, idx_c2 = torch.min(d_c2, dim=1)
            best_ctrl_dists = torch.stack([min_c0, min_c1, min_c2], dim=1)
            ctrl_indices = [idx_c0, idx_c1, idx_c2]
            min_ctrl_val, choice_ctrl = torch.min(best_ctrl_dists, dim=1)
            
            d_spec = torch.cdist(w_sub, lib_spec)
            min_spec_val, idx_spec = torch.min(d_spec, dim=1)
            
            sub_recon = lib_base[idx_base].clone()
            win_spec = (min_spec_val < min_base) & (min_spec_val < min_ctrl_val)
            win_ctrl = (min_ctrl_val < min_base) & (min_ctrl_val <= min_spec_val)
            sub_recon[win_spec] = lib_spec[idx_spec[win_spec]]
            
            mask_c0 = win_ctrl & (choice_ctrl == 0)
            mask_c1 = win_ctrl & (choice_ctrl == 1)
            mask_c2 = win_ctrl & (choice_ctrl == 2)
            sub_recon[mask_c0] = lib_base[ctrl_indices[0][mask_c0]] * 1.5
            sub_recon[mask_c1] = lib_base[ctrl_indices[1][mask_c1]] * 0.5
            sub_recon[mask_c2] = -lib_base[ctrl_indices[2][mask_c2]]
            recon_k[sb_start:sb_end] = sub_recon
            
        macro_recon[m_full] = recon_k
        
    # 2. Resíduo 1
    r1 = w_flat - macro_recon
    r1_sample = r1[perm_sample]
    perm_r1 = torch.randperm(sample_n)[:8]
    r1_centers = r1_sample[perm_r1].clone()
    for _ in range(5):
        d = torch.cdist(r1_sample[:10000], r1_centers)
        lbl = torch.argmin(d, dim=1)
        for k in range(8):
            m = (lbl == k)
            if m.sum() > 0: r1_centers[k] = r1_sample[:10000][m].mean(dim=0)
            
    r1_fam_ids = torch.empty(N, dtype=torch.long)
    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        r1_fam_ids[i:end_i] = torch.argmin(torch.cdist(r1[i:end_i], r1_centers), dim=1)
        
    sample_r1_fam_ids = r1_fam_ids[perm_sample]
    libs_r1 = [train_kmeans(r1_sample[sample_r1_fam_ids == k], K=26, seed=seed) for k in range(8)]
    
    recon_base = macro_recon.clone()
    for k in range(8):
        m_k = (r1_fam_ids == k)
        count_rk = m_k.sum().item()
        if count_rk == 0: continue
        r1_k = r1[m_k]
        recon_rk = torch.empty_like(r1_k)
        for sb_start in range(0, count_rk, 50000):
            sb_end = min(sb_start + 50000, count_rk)
            d_rk = torch.cdist(r1_k[sb_start:sb_end], libs_r1[k])
            recon_rk[sb_start:sb_end] = libs_r1[k][torch.argmin(d_rk, dim=1)]
        recon_base[m_k] += recon_rk
        
    # 3. Resíduo 2
    res_2 = w_flat - recon_base
    lib_r2 = train_kmeans(res_2[perm_sample], K=26, seed=seed)
    recon_3 = recon_base.clone()
    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        d_r2 = torch.cdist(res_2[i:end_i], lib_r2)
        recon_3[i:end_i] += lib_r2[torch.argmin(d_r2, dim=1)]
        
    # 4. Resíduo 3
    res_3 = w_flat - recon_3
    lib_r3 = train_kmeans(res_3[perm_sample], K=26, seed=seed)
    recon_4 = recon_3.clone()
    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        d_r3 = torch.cdist(res_3[i:end_i], lib_r3)
        recon_4[i:end_i] += lib_r3[torch.argmin(d_r3, dim=1)]
        
    # 5. Proteção Seletiva Top P_crit%
    if p_crit > 0:
        sens_list = []
        for i in range(0, N, batch_size):
            end_i = min(i + batch_size, N)
            sub_w = w_flat[i:end_i]
            sub_res = sub_w - recon_4[i:end_i]
            block_norm = torch.norm(sub_w, dim=1)
            sens = (block_norm + 1e-5) * torch.mean(sub_res ** 2, dim=1)
            sens_list.append(sens)
        sensitivity = torch.cat(sens_list)
        
        num_crit = int(p_crit * N)
        crit_idx = torch.topk(sensitivity, k=num_crit).indices
        r4_vecs = (w_flat[crit_idx] - recon_4[crit_idx])
        lib_r4 = train_kmeans(r4_vecs, K=26, seed=seed)
        for sb_start in range(0, num_crit, 50000):
            sb_end = min(sb_start + 50000, num_crit)
            sub_r4 = r4_vecs[sb_start:sb_end]
            d_r4 = torch.cdist(sub_r4, lib_r4)
            recon_4[crit_idx[sb_start:sb_end]] += lib_r4[torch.argmin(d_r4, dim=1)]
            
    mse = torch.mean((weight_tensor.float() - recon_4.view(orig_shape)) ** 2).item()
    cos = torch.nn.functional.cosine_similarity(weight_tensor.float().view(1, -1), recon_4.view(1, -1)).item()
    recon_reshaped = recon_4.view(orig_shape).to(orig_dtype)
    return recon_reshaped, mse, cos

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
# CARREGAR MODELO BF16 ORIGINAL
# -----------------------------------------------------------------------------
print("\n[1/5] Carregando modelo base Qwen3.5-2B em BF16 (model.safetensors)...")
t0_load = time.time()
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
model.eval()
print(f"-> Carregado em {time.time()-t0_load:.2f}s!")

attn_layers = [3, 7, 11, 15, 19, 23]
deltanet_layers = [i for i in range(24) if i not in attn_layers]

# -----------------------------------------------------------------------------
# ETAPA 1: FFN (72 MATRIZES) -> 3.05 BPW
# -----------------------------------------------------------------------------
print("\n[2/5] Quantizando 72 matrizes FFN com H5-A a 3.05 BPW...")
t0_ffn = time.time()
for i in range(24):
    l = model.model.layers[i]
    rg, _, _, _, _ = quantize_matrix_h5a_batched(l.mlp.gate_proj.weight.data, num_ctrl=2, num_spec=4, p_crit=0.05, seed=42+i)
    ru, _, _, _, _ = quantize_matrix_h5a_batched(l.mlp.up_proj.weight.data, num_ctrl=2, num_spec=4, p_crit=0.05, seed=42+i)
    rd, _, _, _, _ = quantize_matrix_h5a_batched(l.mlp.down_proj.weight.data, num_ctrl=3, num_spec=3, p_crit=0.05, seed=42+i)
    l.mlp.gate_proj.weight.data.copy_(rg)
    l.mlp.up_proj.weight.data.copy_(ru)
    l.mlp.down_proj.weight.data.copy_(rd)
t_ffn = time.time() - t0_ffn
print(f"-> 72 matrizes FFN quantizadas em {t_ffn:.1f}s.")

# -----------------------------------------------------------------------------
# ETAPA 2: SELF-ATTENTION (24 MATRIZES) -> 3.05 BPW
# -----------------------------------------------------------------------------
print("\n[3/5] Quantizando 24 matrizes Self-Attention com H5-A a 3.05 BPW...")
t0_attn = time.time()
for i in attn_layers:
    l = model.model.layers[i]
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

# -----------------------------------------------------------------------------
# ETAPA 3: DELTANET GRANDES PROJEÇÕES (54 MATRIZES) -> 4.05 BPW
# -----------------------------------------------------------------------------
print("\n[4/5] Quantizando 54 matrizes DeltaNet com H5-A a 4.05 BPW...")
t0_dnet = time.time()
for i in deltanet_layers:
    l = model.model.layers[i]
    rqkv, _, _ = quantize_matrix_h5a_4bpw(l.linear_attn.in_proj_qkv.weight.data, num_ctrl=3, num_spec=3, p_crit=0.05, seed=200+i)
    rz, _, _ = quantize_matrix_h5a_4bpw(l.linear_attn.in_proj_z.weight.data, num_ctrl=3, num_spec=3, p_crit=0.05, seed=200+i)
    rout, _, _ = quantize_matrix_h5a_4bpw(l.linear_attn.out_proj.weight.data, num_ctrl=3, num_spec=3, p_crit=0.05, seed=200+i)
    l.linear_attn.in_proj_qkv.weight.data.copy_(rqkv)
    l.linear_attn.in_proj_z.weight.data.copy_(rz)
    l.linear_attn.out_proj.weight.data.copy_(rout)
t_dnet = time.time() - t0_dnet
print(f"-> 54 matrizes DeltaNet quantizadas a 4.05 BPW em {t_dnet:.1f}s.")

# -----------------------------------------------------------------------------
# ETAPA 4: EMBEDDINGS HÍBRIDOS (TOP 32.768 EM BF16 + RESTO EM 3.05 BPW)
# -----------------------------------------------------------------------------
print("\n[5/5] Quantizando Embeddings Híbridos: Top 32.768 tokens em BF16, tokens 32.768..248.320 a 3.05 BPW...")
t0_emb = time.time()
w_emb = model.model.embed_tokens.weight.data
total_tokens = w_emb.shape[0]
chunk_size = 50000
for start in range(32768, total_tokens, chunk_size):
    end = min(start + chunk_size, total_tokens)
    rec, _, _, _, _ = quantize_matrix_h5a_batched(w_emb[start:end], num_ctrl=3, num_spec=3, p_crit=0.05, seed=500+start)
    w_emb[start:end].copy_(rec)
t_emb = time.time() - t0_emb
print(f"-> Embeddings híbridos concluídos em {t_emb:.1f}s!")

# -----------------------------------------------------------------------------
# AVALIAÇÃO FINAL DO MODELO
# -----------------------------------------------------------------------------
print("\n" + "=" * 90)
print("AVALIANDO PERPLEXIDADE E GERAÇÃO DO MODELO H5-A (< 850 MB)...")
print("=" * 90)
ppl_res = eval_ppl(model)
gen_res = eval_gen(model)

print("\n--- PERPLEXIDADE (PPL) OBTIDA ---")
for k, v in ppl_res.items():
    print(f"  {k:<14}: {v:.2f}")

print("\n--- GERAÇÃO COMPARATIVA ---")
for k in ["Capital", "Fibonacci", "Newton"]:
    print(f"  [{k}]: {gen_res[k]['text']} ({gen_res[k]['tok_per_sec']:.2f} tok/s)")

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

b_ffn = int(ffn_params * 3.05 / 8) # 345,4 MB (329,4 MiB)
b_attn = int(attn_params * 3.05 / 8) # 33,6 MB (32,0 MiB)
b_dnet = int(dnet_params * 4.05 / 8) # 191,1 MB (182,2 MiB)
b_crit = crit_params * 2 # 3,25 MB (3,10 MiB)
b_norm = norm_params * 2 # 0,20 MB (0,19 MiB)
b_embed_bf16 = embed_bf16_tokens * 2048 * 2 # 134,2 MB (128,0 MiB)
b_embed_h5a = int(embed_h5a_tokens * 2048 * 3.05 / 8) # 168,3 MB (160,5 MiB)
b_embed_total = b_embed_bf16 + b_embed_h5a # 302,5 MB (288,5 MiB)

total_bytes = b_ffn + b_attn + b_dnet + b_crit + b_norm + b_embed_total
size_mb = total_bytes / (1024**2)

size_q4_mb = 1221.5 # 1.28 GB decimal
size_bf16_mb = 4337.5 # 4.55 GB decimal

print("\n" + "=" * 115)
print("QUADRO FINAL: TAMANHO EXATO E COMPARATIVO COM Q4_K_M")
print("=" * 115)
print(f"Componente                      | Parâmetros   | Taxa / Formato      | Tamanho em MB | % do Total")
print("-" * 115)
print(f"FFN (24 camadas, 72 matrizes)   | {ffn_params:<12,d} | H5-A (3.05 BPW)     | {b_ffn/(1024**2):<11.1f} MB | {b_ffn/total_bytes*100:.1f}%")
print(f"Self-Attention (6 cam, 24 mat)  | {attn_params:<12,d} | H5-A (3.05 BPW)     | {b_attn/(1024**2):<11.1f} MB | {b_attn/total_bytes*100:.1f}%")
print(f"DeltaNet Grandes (18 cam, 54)   | {dnet_params:<12,d} | H5-A (4.05 BPW)     | {b_dnet/(1024**2):<11.1f} MB | {b_dnet/total_bytes*100:.1f}%")
print(f"DeltaNet Críticos (decay/conv)  | {crit_params:<12,d} | BF16 (16.0 BPW)     | {b_crit/(1024**2):<11.2f} MB | {b_crit/total_bytes*100:.1f}%")
print(f"Embeddings Top 32.768 tokens    | {embed_bf16_tokens*2048:<12,d} | BF16 (16.0 BPW)     | {b_embed_bf16/(1024**2):<11.1f} MB | {b_embed_bf16/total_bytes*100:.1f}%")
print(f"Embeddings Resto (215.552 toks) | {embed_h5a_tokens*2048:<12,d} | H5-A (3.05 BPW)     | {b_embed_h5a/(1024**2):<11.1f} MB | {b_embed_h5a/total_bytes*100:.1f}%")
print(f"RMSNorms (LayerNorms)           | {norm_params:<12,d} | BF16 (16.0 BPW)     | {b_norm/(1024**2):<11.2f} MB | {b_norm/total_bytes*100:.1f}%")
print("-" * 115)
print(f"TOTAL DO MODELO                 | 1,881,822,016| ~3,78 BPW Médio     | {size_mb:<11.1f} MB | 100.0%")
print(f"-> Tamanho Final: {size_mb:.1f} MB (Ficou abaixo de 850 MB? {'SIM' if size_mb < 850 else 'NAO'})")
print(f"-> Redução em relação ao Q4_K_M (1.221,5 MB / 1,28 GB): {((1 - size_mb/size_q4_mb)*100):.1f}% menor!")
print(f"-> Redução em relação ao BF16 Original (4.337,5 MB / 4,55 GB): {((1 - size_mb/size_bf16_mb)*100):.1f}% menor!")
print("=" * 115)

final_data = {
    "size_mb": size_mb,
    "size_bytes": total_bytes,
    "ppl": ppl_res,
    "gen": gen_res,
    "times": {
        "ffn": t_ffn,
        "attn": t_attn,
        "dnet": t_dnet,
        "emb": t_emb
    }
}

with open("e:/agente/qwen_under_850mb_results.json", "w") as f:
    json.dump(final_data, f, indent=2)

print("\nResultados gravados em e:/agente/qwen_under_850mb_results.json")
