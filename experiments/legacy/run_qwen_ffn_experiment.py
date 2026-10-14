import os
import sys
import time
import json
import gc
from collections import Counter
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "E:/agente/modelo"
print("=" * 100)
print("BENCHMARK H5-A EM ESCALA 2B: QWEN3.5-2B (72 MATRIZES FFN)")
print("=" * 100)

device = "cpu"
torch.set_num_threads(os.cpu_count() or 8)

# 1. K-Means e H5-A Batched
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

def quantize_matrix_h5a_fast(weight_tensor, num_ctrl=2, num_spec=4, spec_mode='sensitivity', p_crit=0.05, seed=42, block_size=8):
    torch.manual_seed(seed)
    orig_shape = weight_tensor.shape
    orig_dtype = weight_tensor.dtype
    w_flat = weight_tensor.float().reshape(-1, block_size)
    N = w_flat.shape[0]
    batch_size = 200000
    
    sample_n = min(N, 50000)
    perm_sample = torch.randperm(N)[:sample_n]
    w_sample = w_flat[perm_sample]
    
    # 1. 8 Famílias Macro
    perm_fam = torch.randperm(sample_n)[:8]
    fam_centers = w_sample[perm_fam].clone()
    for _ in range(5):
        d = torch.cdist(w_sample[:10000], fam_centers)
        lbl = torch.argmin(d, dim=1)
        for k in range(8):
            m = (lbl == k)
            if m.sum() > 0:
                fam_centers[k] = w_sample[:10000][m].mean(dim=0)
                
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
        
        if spec_mode == 'sensitivity':
            norms = torch.norm(vecs_k, dim=1)
            crit_score = (norms + 1e-5) * (min_err ** 2)
            top_k = max(num_spec, int(vecs_k.shape[0] * 0.35))
            crit_idx = torch.topk(crit_score, k=top_k).indices
            spec_k = train_kmeans(vecs_k[crit_idx], K=num_spec, seed=seed)
        else:
            top_k = max(num_spec, int(vecs_k.shape[0] * 0.35))
            hard_idx = torch.topk(min_err, k=top_k).indices
            spec_k = train_kmeans(vecs_k[hard_idx], K=num_spec, seed=seed)
            
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
            
            if num_ctrl == 3:
                d_c0 = torch.cdist(w_sub, lib_base * 1.5)
                d_c1 = torch.cdist(w_sub, lib_base * 0.5)
                d_c2 = torch.cdist(w_sub, -lib_base)
                min_c0, idx_c0 = torch.min(d_c0, dim=1)
                min_c1, idx_c1 = torch.min(d_c1, dim=1)
                min_c2, idx_c2 = torch.min(d_c2, dim=1)
                best_ctrl_dists = torch.stack([min_c0, min_c1, min_c2], dim=1)
                ctrl_indices = [idx_c0, idx_c1, idx_c2]
            else:
                d_c0 = torch.cdist(w_sub, lib_base * 1.5)
                d_c1 = torch.cdist(w_sub, -lib_base)
                min_c0, idx_c0 = torch.min(d_c0, dim=1)
                min_c1, idx_c1 = torch.min(d_c1, dim=1)
                best_ctrl_dists = torch.stack([min_c0, min_c1, min_c2 if False else idx_c1], dim=1)[:, :2]
                ctrl_indices = [idx_c0, idx_c1]
                
            min_ctrl_val, choice_ctrl = torch.min(best_ctrl_dists, dim=1)
            
            d_spec = torch.cdist(w_sub, lib_spec)
            min_spec_val, idx_spec = torch.min(d_spec, dim=1)
            
            sub_recon = lib_base[idx_base].clone()
            win_spec = (min_spec_val < min_base) & (min_spec_val < min_ctrl_val)
            win_ctrl = (min_ctrl_val < min_base) & (min_ctrl_val <= min_spec_val)
            
            sub_recon[win_spec] = lib_spec[idx_spec[win_spec]]
            
            if num_ctrl == 3:
                mask_c0 = win_ctrl & (choice_ctrl == 0)
                mask_c1 = win_ctrl & (choice_ctrl == 1)
                mask_c2 = win_ctrl & (choice_ctrl == 2)
                sub_recon[mask_c0] = lib_base[ctrl_indices[0][mask_c0]] * 1.5
                sub_recon[mask_c1] = lib_base[ctrl_indices[1][mask_c1]] * 0.5
                sub_recon[mask_c2] = -lib_base[ctrl_indices[2][mask_c2]]
            else:
                mask_c0 = win_ctrl & (choice_ctrl == 0)
                mask_c1 = win_ctrl & (choice_ctrl == 1)
                sub_recon[mask_c0] = lib_base[ctrl_indices[0][mask_c0]] * 1.5
                sub_recon[mask_c1] = -lib_base[ctrl_indices[1][mask_c1]]
                
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
    res_base = w_flat - recon_base
    lib_r2 = train_kmeans(res_base[perm_sample], K=26, seed=seed)
    recon_final = recon_base.clone()
    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        d_r2 = torch.cdist(res_base[i:end_i], lib_r2)
        recon_final[i:end_i] += lib_r2[torch.argmin(d_r2, dim=1)]
        
    # 4. Proteção Seletiva Top P_crit%
    if p_crit > 0:
        sens_list = []
        for i in range(0, N, batch_size):
            end_i = min(i + batch_size, N)
            sub_w = w_flat[i:end_i]
            sub_res = sub_w - recon_final[i:end_i]
            block_norm = torch.norm(sub_w, dim=1)
            sens = (block_norm + 1e-5) * torch.mean(sub_res ** 2, dim=1)
            sens_list.append(sens)
        sensitivity = torch.cat(sens_list)
        
        num_crit = int(p_crit * N)
        crit_idx = torch.topk(sensitivity, k=num_crit).indices
        
        r3_vecs = (w_flat[crit_idx] - recon_final[crit_idx])
        lib_r3 = train_kmeans(r3_vecs, K=26, seed=seed)
        
        for sb_start in range(0, num_crit, 50000):
            sb_end = min(sb_start + 50000, num_crit)
            sub_r3 = r3_vecs[sb_start:sb_end]
            d_r3 = torch.cdist(sub_r3, lib_r3)
            recon_final[crit_idx[sb_start:sb_end]] += lib_r3[torch.argmin(d_r3, dim=1)]
            
    mse = torch.mean((weight_tensor.float() - recon_final.view(orig_shape)) ** 2).item()
    cos = torch.nn.functional.cosine_similarity(weight_tensor.float().view(1, -1), recon_final.view(1, -1)).item()
    recon_reshaped = recon_final.view(orig_shape).to(orig_dtype)
    return recon_reshaped, mse, cos

# 2. Inicialização do Modelo e Tokenizer
print("Carregando tokenizer e modelo original BF16...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
model.eval()

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

# 3. Fase 1: Avaliação do BF16 Original
print("\n" + "=" * 80)
print("FASE 1: BASELINE ORIGINAL BF16 (16.0 bpp)")
print("=" * 80)
ppl_bf16 = eval_ppl(model)
gen_bf16 = eval_gen(model)
print("PPL Original BF16:")
for k, v in ppl_bf16.items():
    print(f"   {k:<12}: {v:.2f}")
print("Gerações Original BF16:")
for k, v in gen_bf16.items():
    print(f"   {k:<10}: {v['text']} (Div: {v['uniq']*100:.1f}%)")

# Guardar cópia dos pesos FFN originais para restaurar depois
print("\nArmazenando referências para quantização controlada...")
orig_weights = {}
for i in range(24):
    orig_weights[f"l{i}.gate"] = model.model.layers[i].mlp.gate_proj.weight.data.clone()
    orig_weights[f"l{i}.up"] = model.model.layers[i].mlp.up_proj.weight.data.clone()
    orig_weights[f"l{i}.down"] = model.model.layers[i].mlp.down_proj.weight.data.clone()

# 4. Fase 2 e 3: Quantização H5-A + RB no FFN Completo (24 camadas, 72 matrizes)
# gate_proj e up_proj: 2+4-D (Expansão)
# down_proj: 3+3 (Redução)
print("\n" + "=" * 80)
print("FASE 2 & 3: QUANTIZAÇÃO H5-A (3.05 bpp) EM TODAS AS 72 MATRIZES FFN")
print("Configuração Asimétrica: gate_proj (2+4-D), up_proj (2+4-D), down_proj (3+3)")
print("Proteção seletiva p_crit = 5% por matriz")
print("=" * 80)

total_t0 = time.time()
layer_stats = []

for i in range(24):
    t_layer0 = time.time()
    l = model.model.layers[i]
    
    # Gate proj (2+4-D)
    rec_gate, mse_g, cos_g = quantize_matrix_h5a_fast(orig_weights[f"l{i}.gate"], num_ctrl=2, num_spec=4, spec_mode='sensitivity', p_crit=0.05, seed=42+i)
    l.mlp.gate_proj.weight.data.copy_(rec_gate)
    
    # Up proj (2+4-D)
    rec_up, mse_u, cos_u = quantize_matrix_h5a_fast(orig_weights[f"l{i}.up"], num_ctrl=2, num_spec=4, spec_mode='sensitivity', p_crit=0.05, seed=42+i)
    l.mlp.up_proj.weight.data.copy_(rec_up)
    
    # Down proj (3+3)
    rec_down, mse_d, cos_d = quantize_matrix_h5a_fast(orig_weights[f"l{i}.down"], num_ctrl=3, num_spec=3, spec_mode='sensitivity', p_crit=0.05, seed=42+i)
    l.mlp.down_proj.weight.data.copy_(rec_down)
    
    t_layer = time.time() - t_layer0
    layer_stats.append({
        "layer": i,
        "mse_g": mse_g, "cos_g": cos_g,
        "mse_u": mse_u, "cos_u": cos_u,
        "mse_d": mse_d, "cos_d": cos_d,
        "time": t_layer
    })
    
    if (i + 1) % 4 == 0 or i == 0 or i == 23:
        print(f"Camada {i:>2}/23 concluída ({t_layer:.1f}s) | Gate Cos: {cos_g:.4f} | Up Cos: {cos_u:.4f} | Down Cos: {cos_d:.4f}")

total_quant_time = time.time() - total_t0
print(f"\nQuantização de 72 matrizes FFN (906M parâmetros) finalizada em {total_quant_time:.1f}s ({total_quant_time/60:.2f} min)!")

# Médias de fidelidade matricial
avg_cos_g = sum(s["cos_g"] for s in layer_stats) / 24
avg_cos_u = sum(s["cos_u"] for s in layer_stats) / 24
avg_cos_d = sum(s["cos_d"] for s in layer_stats) / 24
avg_mse_g = sum(s["mse_g"] for s in layer_stats) / 24
avg_mse_u = sum(s["mse_u"] for s in layer_stats) / 24
avg_mse_d = sum(s["mse_d"] for s in layer_stats) / 24

print("\n--- FIDELIDADE MATRICIAL MÉDIA NAS 24 CAMADAS FFN ---")
print(f"Gate Proj (2+4-D): Cosine = {avg_cos_g:.6f} | MSE = {avg_mse_g:.8f}")
print(f"Up Proj   (2+4-D): Cosine = {avg_cos_u:.6f} | MSE = {avg_mse_u:.8f}")
print(f"Down Proj (3+3)  : Cosine = {avg_cos_d:.6f} | MSE = {avg_mse_d:.8f}")

# Avaliar Modelo com FFN Comprimido a 3.05 bpp
print("\nAvaliando PPL e Geração com FFN 100% Comprimido a 3.05 bpp...")
ppl_h5a = eval_ppl(model)
gen_h5a = eval_gen(model)

print("\n--- RESULTADOS FINAIS: COMPARATIVO BF16 vs H5-A FFN (3.05 bpp) ---")
print(f"{'Domínio':<15} | {'BF16 Original':<15} | {'H5-A FFN (3.05 bpp)':<22} | {'Diferença PPL':<15}")
print("-" * 75)
for d in ["Ciencia", "Codigo", "Historia", "Enciclopedia", "Media"]:
    v_orig = ppl_bf16[d]
    v_h5a = ppl_h5a[d]
    diff = v_h5a - v_orig
    print(f"{d:<15} | {v_orig:<15.2f} | {v_h5a:<22.2f} | {diff:>+14.2f}")

print("\n--- GERAÇÃO DE TEXTO ---")
for k in ["Capital", "Fibonacci", "Newton"]:
    print(f"[{k}]")
    print(f"  BF16: {gen_bf16[k]['text']} (Uniq: {gen_bf16[k]['uniq']*100:.1f}%)")
    print(f"  H5-A: {gen_h5a[k]['text']} (Uniq: {gen_h5a[k]['uniq']*100:.1f}%)")

# Salvar relatório JSON
res_summary = {
    "model": "Qwen3.5-2B",
    "params": 2274069824,
    "ffn_params": 905969664,
    "quant_time_sec": total_quant_time,
    "matrix_stats": {
        "avg_cos_gate": avg_cos_g, "avg_mse_gate": avg_mse_g,
        "avg_cos_up": avg_cos_u, "avg_mse_up": avg_mse_u,
        "avg_cos_down": avg_cos_d, "avg_mse_down": avg_mse_d,
    },
    "ppl_bf16": ppl_bf16,
    "ppl_h5a": ppl_h5a,
    "gen_bf16": gen_bf16,
    "gen_h5a": gen_h5a,
    "layers": layer_stats
}

with open("e:/agente/qwen_ffn_h5a_results.json", "w") as f:
    json.dump(res_summary, f, indent=2)

print("\nResultados salvos em e:/agente/qwen_ffn_h5a_results.json")
print("=" * 100)
