import os
import sys
import time
import json
import gc
import psutil
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "E:/agente/modelo"
GGUF_FILE = "Qwen3.5-2B-Q4_K_M.gguf"
print("=" * 100)
print("EXPERIMENTO DE VARREDURA RATE-DISTORTION: H5-A (3.05 vs 4.05 BPW) vs Q4_K_M")
print("=" * 100)

device = "cpu"
torch.set_num_threads(os.cpu_count() or 8)

# 1. K-Means e H5-A a 4.05 bpp (4 Estágios Residuais em Cascata)
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

def quantize_matrix_h5a_4bpw(weight_tensor, num_ctrl=3, num_spec=3, spec_mode='sensitivity', p_crit=0.05, seed=42, block_size=8):
    """H5-A com 4 estágios residuais em cascata = 4.05 bpp com proteção p_crit."""
    torch.manual_seed(seed)
    orig_shape = weight_tensor.shape
    orig_dtype = weight_tensor.dtype
    w_flat = weight_tensor.float().reshape(-1, block_size)
    N = w_flat.shape[0]
    batch_size = 200000
    
    sample_n = min(N, 50000)
    perm_sample = torch.randperm(N)[:sample_n]
    w_sample = w_flat[perm_sample]
    
    # 1. Macro Famílias (1.0 bpp)
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
        
    # 2. Resíduo 1 (Base 2.0 bpp)
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
        
    # 3. Resíduo 2 (Base 3.0 bpp)
    res_2 = w_flat - recon_base
    lib_r2 = train_kmeans(res_2[perm_sample], K=26, seed=seed)
    recon_3 = recon_base.clone()
    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        d_r2 = torch.cdist(res_2[i:end_i], lib_r2)
        recon_3[i:end_i] += lib_r2[torch.argmin(d_r2, dim=1)]
        
    # 4. Resíduo 3 (Estágio Adicional -> Base 4.0 bpp)
    res_3 = w_flat - recon_3
    lib_r3 = train_kmeans(res_3[perm_sample], K=26, seed=seed)
    recon_4 = recon_3.clone()
    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        d_r3 = torch.cdist(res_3[i:end_i], lib_r3)
        recon_4[i:end_i] += lib_r3[torch.argmin(d_r3, dim=1)]
        
    # 5. Proteção Seletiva Top P_crit% (+0.05 bpp -> 4.05 bpp)
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

# 2. Pipeline de Avaliação
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

# 3. Carregar modelo base BF16
print("\nCarregando modelo base BF16...")
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
model.eval()

# Armazenar pesos FFN originais
orig_weights = {}
for i in range(24):
    orig_weights[f"l{i}.gate"] = model.model.layers[i].mlp.gate_proj.weight.data.clone()
    orig_weights[f"l{i}.up"] = model.model.layers[i].mlp.up_proj.weight.data.clone()
    orig_weights[f"l{i}.down"] = model.model.layers[i].mlp.down_proj.weight.data.clone()

# 4. TESTE A: Q4_K_M (SOMENTE FFN NO MESMO BACKBONE BF16)
print("\n" + "=" * 90)
print("TESTE A: Q4_K_M ISOLADO NO FFN (MESMO BACKBONE BF16 DO H5-A)")
print("=" * 90)
print("Carregando pesos FFN do GGUF...")
m_gguf = AutoModelForCausalLM.from_pretrained(MODEL_DIR, gguf_file=GGUF_FILE, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)

# Transferir SOMENTE as 72 matrizes FFN do GGUF para o modelo BF16
for i in range(24):
    model.model.layers[i].mlp.gate_proj.weight.data.copy_(m_gguf.model.layers[i].mlp.gate_proj.weight.data)
    model.model.layers[i].mlp.up_proj.weight.data.copy_(m_gguf.model.layers[i].mlp.up_proj.weight.data)
    model.model.layers[i].mlp.down_proj.weight.data.copy_(m_gguf.model.layers[i].mlp.down_proj.weight.data)

del m_gguf
gc.collect()

print("Avaliando PPL e Geração para Q4_K_M (FFN-Only)...")
ppl_q4_ffn = eval_ppl(model)
gen_q4_ffn = eval_gen(model)

print("PPL Q4_K_M (FFN-Only):")
for k, v in ppl_q4_ffn.items():
    print(f"   {k:<12}: {v:.2f}")

# 5. TESTE B: H5-A @ 4.05 BPW (4 ESTÁGIOS RESIDUAIS NO FFN)
print("\n" + "=" * 90)
print("TESTE B: H5-A @ 4.05 BPW EM TODAS AS 72 MATRIZES FFN")
print("=" * 90)

t0_4bpw = time.time()
cos_g_list, cos_u_list, cos_d_list = [], [], []

for i in range(24):
    t_l0 = time.time()
    l = model.model.layers[i]
    
    # Gate proj
    rec_gate, mse_g, cos_g = quantize_matrix_h5a_4bpw(orig_weights[f"l{i}.gate"], num_ctrl=3, num_spec=3, p_crit=0.05, seed=42+i)
    l.mlp.gate_proj.weight.data.copy_(rec_gate)
    cos_g_list.append(cos_g)
    
    # Up proj
    rec_up, mse_u, cos_u = quantize_matrix_h5a_4bpw(orig_weights[f"l{i}.up"], num_ctrl=3, num_spec=3, p_crit=0.05, seed=42+i)
    l.mlp.up_proj.weight.data.copy_(rec_up)
    cos_u_list.append(cos_u)
    
    # Down proj
    rec_down, mse_d, cos_d = quantize_matrix_h5a_4bpw(orig_weights[f"l{i}.down"], num_ctrl=3, num_spec=3, p_crit=0.05, seed=42+i)
    l.mlp.down_proj.weight.data.copy_(rec_down)
    cos_d_list.append(cos_d)
    
    if (i + 1) % 6 == 0 or i == 0 or i == 23:
        print(f"Camada {i:>2}/23 ({time.time()-t_l0:.1f}s) | Gate Cos: {cos_g:.4f} | Up Cos: {cos_u:.4f} | Down Cos: {cos_d:.4f}")

total_quant_4bpw = time.time() - t0_4bpw
print(f"Quantização H5-A a 4.05 bpp concluída em {total_quant_4bpw:.1f}s!")
print(f"Cosine Médio 4.05 bpp: Gate = {sum(cos_g_list)/24:.4f} | Up = {sum(cos_u_list)/24:.4f} | Down = {sum(cos_d_list)/24:.4f}")

print("\nAvaliando PPL e Geração para H5-A @ 4.05 BPW...")
ppl_h5a_4bpw = eval_ppl(model)
gen_h5a_4bpw = eval_gen(model)

print("PPL H5-A @ 4.05 BPW:")
for k, v in ppl_h5a_4bpw.items():
    print(f"   {k:<12}: {v:.2f}")

# 6. Carregar resultados anteriores de 3.05 bpp e BF16
with open("e:/agente/qwen_ffn_h5a_results.json") as f:
    prev_results = json.load(f)

ppl_bf16 = prev_results["ppl_bf16"]
ppl_h5a_305 = prev_results["ppl_h5a"]

# 7. TABELA CONSOLIDADA RATE-DISTORTION CURVE
print("\n" + "=" * 115)
print("CURVA RATE-DISTORTION: COMPARAÇÃO CONTROLADA (MESMO BACKBONE, MESMO PIPELINE)")
print("=" * 115)
print(f"{'Configuração':<24} | {'BPW FFN':<10} | {'PPL Ciência':<12} | {'PPL Código':<12} | {'PPL História':<13} | {'PPL Média':<10}")
print("-" * 115)
print(f"{'BF16 Original':<24} | {'16.00':<10} | {ppl_bf16['Ciencia']:<12.2f} | {ppl_bf16['Codigo']:<12.2f} | {ppl_bf16['Historia']:<13.2f} | {ppl_bf16['Media']:<10.2f}")
print(f"{'Q4_K_M (FFN-Only)':<24} | {'~4.5-5.0':<10} | {ppl_q4_ffn['Ciencia']:<12.2f} | {ppl_q4_ffn['Codigo']:<12.2f} | {ppl_q4_ffn['Historia']:<13.2f} | {ppl_q4_ffn['Media']:<10.2f}")
print(f"{'H5-A @ 4.05 BPW':<24} | {'4.05':<10} | {ppl_h5a_4bpw['Ciencia']:<12.2f} | {ppl_h5a_4bpw['Codigo']:<12.2f} | {ppl_h5a_4bpw['Historia']:<13.2f} | {ppl_h5a_4bpw['Media']:<10.2f}")
print(f"{'H5-A @ 3.05 BPW':<24} | {'3.05':<10} | {ppl_h5a_305['Ciencia']:<12.2f} | {ppl_h5a_305['Codigo']:<12.2f} | {ppl_h5a_305['Historia']:<13.2f} | {ppl_h5a_305['Media']:<10.2f}")
print("=" * 115)

print("\n--- GERAÇÃO COMPARATIVA NOS 3 TESTES ---")
for k in ["Capital", "Fibonacci", "Newton"]:
    print(f"\n[{k}]")
    print(f"  Q4_K_M (FFN-Only): {gen_q4_ffn[k]['text']}")
    print(f"  H5-A @ 4.05 BPW  : {gen_h5a_4bpw[k]['text']}")

sweep_results = {
    "ppl_bf16": ppl_bf16,
    "ppl_q4_ffn": ppl_q4_ffn,
    "ppl_h5a_305": ppl_h5a_305,
    "ppl_h5a_405": ppl_h5a_4bpw,
    "gen_q4_ffn": gen_q4_ffn,
    "gen_h5a_405": gen_h5a_4bpw,
    "quant_time_405_sec": total_quant_4bpw
}

with open("e:/agente/qwen_rate_distortion_sweep_results.json", "w") as f:
    json.dump(sweep_results, f, indent=2)

print("\nResultados salvos em e:/agente/qwen_rate_distortion_sweep_results.json")
print("=" * 100)
