import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import time
import copy

def train_cb_optimized(vectors, K=26, n_iter=6, seed=42):
    """K-means++ com 6 átomos de outliers com seed controlada."""
    torch.manual_seed(seed)
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

def quantize_layer_with_seed(weight_tensor, p_crit=0.05, seed=42, num_families=8, atoms_per_family=26, block_size=8):
    torch.manual_seed(seed)
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
    
    libs_macro = [train_cb_optimized(w_sample[fam_ids[:sample_n] == k], K=atoms_per_family, seed=seed) for k in range(num_families)]
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
    
    libs_r1 = [train_cb_optimized(r1_sample[r1_fam_ids[:sample_n] == k], K=atoms_per_family, seed=seed) for k in range(num_families)]
    recon_base = macro_recon.clone()
    for k in range(num_families):
        m = (r1_fam_ids == k)
        if m.sum() > 0:
            d_rk = torch.cdist(r1[m], libs_r1[k])
            recon_base[m] += libs_r1[k][torch.argmin(d_rk, dim=1)]
            
    # 3. Resíduo 2 (Base 3.0 bpp)
    res_base = w_flat - recon_base
    lib_r2 = train_cb_optimized(res_base[:sample_n], K=atoms_per_family, seed=seed)
    d_r2 = torch.cdist(res_base, lib_r2)
    recon_final = recon_base + lib_r2[torch.argmin(d_r2, dim=1)]
    
    # 4. Proteção dos top P_crit% blocos
    if p_crit > 0:
        block_norm = torch.norm(w_flat, dim=1)
        res_3b = w_flat - recon_final
        sensitivity = (block_norm + 1e-5) * torch.mean(res_3b ** 2, dim=1)
        num_crit = int(p_crit * N)
        crit_idx = torch.topk(sensitivity, k=num_crit).indices
        
        r3_vecs = res_3b[crit_idx]
        lib_r3 = train_cb_optimized(r3_vecs, K=atoms_per_family, seed=seed)
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
    print("=" * 105)
    print("MATRIZ DE ROBUSTEZ: SEEDS (1, 2, 3) x TAXAS DE PROTEÇÃO (0% a 10%)")
    print("Objetivo: Testar se o 'Sweet Spot' de ~5% se mantém estável sob variação estocástica")
    print("=" * 105)
    
    model_name = "distilbert/distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    base_model.eval()
    
    domains = {
        "Ciência": "Quantum mechanics describes nature at the smallest scales of energy levels of atoms and subatomic particles.",
        "Código": "def binary_search(array, target): low = 0; high = len(array) - 1; while low <= high: mid = (low + high) // 2",
        "História": "The ancient library of Alexandria was once the largest repository of knowledge in the Hellenistic world.",
        "Enciclopédia": "The solar system formed approximately 4.6 billion years ago from the gravitational collapse of a molecular cloud."
    }
    
    # 0. Referência Original
    ppl_orig = {d: evaluate_perplexity(base_model, tokenizer, text) for d, text in domains.items()}
    avg_orig = np.mean(list(ppl_orig.values()))
    print(f"\n[0] MODELO ORIGINAL (16.0 bpp): Média PPL = {avg_orig:.2f}")
    
    rates = [0.00, 0.02, 0.04, 0.05, 0.06, 0.08, 0.10]
    seeds = [42, 123, 999]
    
    # Dicionário de resultados: rate -> list of seed results
    results_by_rate = {r: [] for r in rates}
    
    t_start = time.time()
    
    for r in rates:
        for s in seeds:
            m_test = copy.deepcopy(base_model)
            for block in m_test.transformer.h:
                with torch.no_grad():
                    w_fc, _ = quantize_layer_with_seed(block.mlp.c_fc.weight.data, p_crit=r, seed=s)
                    block.mlp.c_fc.weight.data = w_fc
                    w_proj, _ = quantize_layer_with_seed(block.mlp.c_proj.weight.data, p_crit=r, seed=s)
                    block.mlp.c_proj.weight.data = w_proj
                    
            ppls = [evaluate_perplexity(m_test, tokenizer, text) for text in domains.values()]
            mean_ppl = np.mean(ppls)
            max_ppl = np.max(ppls)
            results_by_rate[r].append((mean_ppl, max_ppl, ppls))
            
        print(f"-> Taxa {int(r*100)}% concluída nas 3 seeds (Média PPL: {np.mean([x[0] for x in results_by_rate[r]]):.2f})")
        
    total_elapsed = time.time() - t_start
    print(f"\nVarredura completa executada em {total_elapsed:.2f}s ({len(rates)*len(seeds)} modelos completos testados).")
    
    # Save to JSON
    import json
    save_data = {
        "original_fp16": {"avg": avg_orig, "domains": ppl_orig},
        "rates": rates,
        "seeds": seeds,
        "results": {
            str(r): [
                {"seed": seeds[i], "mean_ppl": results_by_rate[r][i][0], "max_ppl": results_by_rate[r][i][1], "domain_ppls": results_by_rate[r][i][2]}
                for i in range(len(seeds))
            ]
            for r in rates
        }
    }
    with open("robustness_results.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print("Resultados salvos em robustness_results.json")

    print("\n" + "=" * 105)
    print("MATRIZ DE ROBUSTEZ: MÉDIA, DESVIO PADRÃO E PIOR CASO ENTRE SEEDS")
    print("=" * 105)
    print(f"{'Proteção (%)':<14} | {'BPP Médio':<10} | {'PPL Médio (3 seeds)':<22} | {'Desvio (std)':<12} | {'Pior Caso (Max)':<16} | {'Delta vs FP16'}")
    print("-" * 105)
    print(f"{'Original FP16':<14} | {'16.00 b':<10} | {avg_orig:>18.2f}    | {'-':<12} | {'-':<16} | {'0.00 (Ref)'}")
    print("-" * 105)
    
    for r in rates:
        seed_means = [x[0] for x in results_by_rate[r]]
        seed_maxs = [x[1] for x in results_by_rate[r]]
        mean_all = np.mean(seed_means)
        std_all = np.std(seed_means)
        worst_all = np.max(seed_maxs)
        delta_fp16 = mean_all - avg_orig
        bpp_str = f"{3.0 + r:.2f} b"
        pct_str = f"{int(r*100)}%"
        print(f"{pct_str:<14} | {bpp_str:>7}    | {mean_all:>18.2f}    | {std_all:>10.2f}  | {worst_all:>14.2f}  | {delta_fp16:>+11.2f}")
    print("=" * 105)

if __name__ == '__main__':
    main()
