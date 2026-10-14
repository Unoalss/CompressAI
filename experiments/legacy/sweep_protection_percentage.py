import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import time
import copy

def train_cb_optimized(vectors, K=26, n_iter=8):
    """K-means++ com 6 átomos reservados para outliers extremos."""
    N, D = vectors.shape
    if N == 0: return torch.zeros(K, D)
    if N < K:
        reps = (K // N) + 1
        return vectors.repeat(reps, 1)[:K]
        
    norms = torch.norm(vectors, dim=1)
    K_outliers = 6
    K_core = K - K_outliers
    
    top_outlier_idx = torch.topk(norms, k=K_outliers).indices
    outlier_atoms = vectors[top_outlier_idx].clone()
    
    core_mask = torch.ones(N, dtype=torch.bool)
    core_mask[top_outlier_idx] = False
    core_vectors = vectors[core_mask]
    
    centers = [core_vectors[torch.randint(0, core_vectors.shape[0], (1,))]]
    for _ in range(1, K_core):
        dists = torch.cdist(core_vectors, torch.stack(centers).squeeze(1))
        min_dists = torch.min(dists, dim=1).values
        probs = min_dists ** 2
        probs_sum = probs.sum()
        if probs_sum > 0:
            probs = probs / probs_sum
            chosen_idx = torch.multinomial(probs, 1)
            centers.append(core_vectors[chosen_idx])
        else:
            centers.append(core_vectors[torch.randint(0, core_vectors.shape[0], (1,))])
            
    c_core = torch.stack(centers).squeeze(1)
    for _ in range(n_iter):
        d = torch.cdist(core_vectors, c_core)
        lbl = torch.argmin(d, dim=1)
        for k in range(K_core):
            m = (lbl == k)
            if m.sum() > 0: c_core[k] = core_vectors[m].mean(dim=0)
            
    return torch.cat([c_core, outlier_atoms], dim=0)

def quantize_layer_with_p_crit(weight_tensor, p_crit=0.05, num_families=8, atoms_per_family=26, block_size=8):
    orig_shape = weight_tensor.shape
    w_flat = weight_tensor.float().reshape(-1, block_size)
    N = w_flat.shape[0]
    sample_n = min(N, 35000)
    w_sample = w_flat[:sample_n]
    
    # 1. 8 Famílias Macro
    perm = torch.randperm(sample_n)[:num_families]
    fam_centers = w_sample[perm].clone()
    for _ in range(5):
        d = torch.cdist(w_sample, fam_centers)
        lbl = torch.argmin(d, dim=1)
        for k in range(num_families):
            m = (lbl == k)
            if m.sum() > 0: fam_centers[k] = w_sample[m].mean(dim=0)
    fam_ids = torch.argmin(torch.cdist(w_flat, fam_centers), dim=1)
    
    libs_macro = [train_cb_optimized(w_sample[fam_ids[:sample_n] == k], K=atoms_per_family) for k in range(num_families)]
    macro_recon = torch.zeros_like(w_flat)
    for k in range(num_families):
        m = (fam_ids == k)
        if m.sum() > 0:
            d_k = torch.cdist(w_flat[m], libs_macro[k])
            macro_recon[m] = libs_macro[k][torch.argmin(d_k, dim=1)]
            
    # 2. Resíduo 1 (Base 2.0 bpp)
    r1 = w_flat - macro_recon
    r1_sample = r1[:sample_n]
    perm_r1 = torch.randperm(sample_n)[:num_families]
    r1_centers = r1_sample[perm_r1].clone()
    for _ in range(5):
        d = torch.cdist(r1_sample, r1_centers)
        lbl = torch.argmin(d, dim=1)
        for k in range(num_families):
            m = (lbl == k)
            if m.sum() > 0: r1_centers[k] = r1_sample[m].mean(dim=0)
    r1_fam_ids = torch.argmin(torch.cdist(r1, r1_centers), dim=1)
    
    libs_r1 = [train_cb_optimized(r1_sample[r1_fam_ids[:sample_n] == k], K=atoms_per_family) for k in range(num_families)]
    recon_base = macro_recon.clone()
    for k in range(num_families):
        m = (r1_fam_ids == k)
        if m.sum() > 0:
            d_rk = torch.cdist(r1[m], libs_r1[k])
            recon_base[m] += libs_r1[k][torch.argmin(d_rk, dim=1)]
            
    # 3. Resíduo 2 (Base 3.0 bpp)
    res_base = w_flat - recon_base
    lib_r2 = train_cb_optimized(res_base[:sample_n], K=atoms_per_family)
    d_r2 = torch.cdist(res_base, lib_r2)
    recon_final = recon_base + lib_r2[torch.argmin(d_r2, dim=1)]
    
    # 4. Proteção dos top P_crit% blocos com Resíduo 3 (4.0 bpp)
    if p_crit > 0:
        block_norm = torch.norm(w_flat, dim=1)
        res_3b = w_flat - recon_final
        # Métrica de sensibilidade: Energia do Bloco * Erro Residual Remanescente
        sensitivity = (block_norm + 1e-5) * torch.mean(res_3b ** 2, dim=1)
        num_crit = int(p_crit * N)
        crit_idx = torch.topk(sensitivity, k=num_crit).indices
        
        r3_vecs = res_3b[crit_idx]
        lib_r3 = train_cb_optimized(r3_vecs, K=atoms_per_family)
        d_r3 = torch.cdist(r3_vecs, lib_r3)
        recon_final[crit_idx] += lib_r3[torch.argmin(d_r3, dim=1)]
        actual_bpp = 3.0 + p_crit
    else:
        actual_bpp = 3.0
        
    return recon_final.view(orig_shape), actual_bpp

def evaluate_perplexity(model, tokenizer, text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        loss = model(**inputs, labels=inputs.input_ids).loss
        return torch.exp(loss).item()

def main():
    print("=" * 95)
    print("VARREDURA DA PROTEÇÃO CRÍTICA: DE 0% ATÉ 15%")
    print("Pergunta: 'Conseguimos a mesma qualidade protegendo apenas 2%, 5% ou 10% dos blocos?'")
    print("=" * 95)
    
    model_name = "distilbert/distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    base_model.eval()
    
    domains = {
        "Ciência / Física": (
            "Quantum mechanics describes nature at the smallest scales of energy levels of atoms and subatomic particles. "
            "Classical physics cannot explain blackbody radiation or the photoelectric effect without quantization."
        ),
        "Código / Lógica": (
            "def binary_search(array, target): "
            "    low = 0 "
            "    high = len(array) - 1 "
            "    while low <= high: "
            "        mid = (low + high) // 2 "
            "        if array[mid] == target: return mid "
            "    return -1"
        ),
        "História / Conhecimento": (
            "The ancient library of Alexandria was once the largest repository of knowledge in the Hellenistic world, "
            "attracting scholars from across the Mediterranean to study mathematics, astronomy, and literature."
        ),
        "Enciclopédia / Fatos": (
            "The solar system formed approximately 4.6 billion years ago from the gravitational collapse of a giant interstellar "
            "molecular cloud containing mostly hydrogen and helium gas."
        )
    }
    
    # 0. Referência Original
    ppl_orig = {d: evaluate_perplexity(base_model, tokenizer, text) for d, text in domains.items()}
    
    protection_rates = [0.00, 0.02, 0.05, 0.10, 0.15]
    sweep_results = []
    
    for p in protection_rates:
        pct_label = f"{int(p*100)}%"
        print(f"\n-> Testando Modelo Inteiro com Proteção de {pct_label} dos blocos...")
        model_p = copy.deepcopy(base_model)
        t0 = time.time()
        
        bpps = []
        for block in model_p.transformer.h:
            with torch.no_grad():
                w_fc, b_fc = quantize_layer_with_p_crit(block.mlp.c_fc.weight.data, p_crit=p)
                block.mlp.c_fc.weight.data = w_fc
                bpps.append(b_fc)
                
                w_proj, b_proj = quantize_layer_with_p_crit(block.mlp.c_proj.weight.data, p_crit=p)
                block.mlp.c_proj.weight.data = w_proj
                bpps.append(b_proj)
                
        elapsed = time.time() - t0
        avg_bpp = sum(bpps) / len(bpps)
        ppls = {d: evaluate_perplexity(model_p, tokenizer, text) for d, text in domains.items()}
        print(f"   [12 Camadas Quantizadas em {elapsed:.2f}s | BPP: {avg_bpp:.2f}]")
        print(f"   -> Física: {ppls['Ciência / Física']:.2f} | Enciclopédia: {ppls['Enciclopédia / Fatos']:.2f} | História: {ppls['História / Conhecimento']:.2f} | Código: {ppls['Código / Lógica']:.2f}")
        
        sweep_results.append({
            'pct_label': pct_label,
            'p': p,
            'bpp': avg_bpp,
            'ppls': ppls
        })
        
    print("\n" + "=" * 105)
    print("A CURVA DE RETORNO DA PROTEÇÃO CRÍTICA (0% até 15%)")
    print("=" * 105)
    print(f"{'Proteção (%)':<14} | {'BPP Médio':<10} | {'Ciência/Física':<16} | {'Enciclopédia':<15} | {'História':<12} | {'Código':<10}")
    print("-" * 105)
    print(f"{'Original FP16':<14} | {'16.00 b':<10} | {ppl_orig['Ciência / Física']:>14.2f}  | {ppl_orig['Enciclopédia / Fatos']:>13.2f}  | {ppl_orig['História / Conhecimento']:>10.2f} | {ppl_orig['Código / Lógica']:>8.2f}")
    print("-" * 105)
    for r in sweep_results:
        p_res = r['ppls']
        print(f"{r['pct_label']:<14} | {r['bpp']:>7.2f} b  | {p_res['Ciência / Física']:>14.2f}  | {p_res['Enciclopédia / Fatos']:>13.2f}  | {p_res['História / Conhecimento']:>10.2f} | {p_res['Código / Lógica']:>8.2f}")
    print("=" * 105)

if __name__ == '__main__':
    main()
