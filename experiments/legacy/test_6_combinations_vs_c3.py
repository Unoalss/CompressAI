import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import time
import copy
from itertools import combinations

def train_kmeans(vectors, K=26, n_iter=6, seed=42):
    """K-means++ rápido com átomos de outliers e controle de seed."""
    torch.manual_seed(seed)
    N, D = vectors.shape
    if N == 0: return torch.zeros(K, D)
    if N < K:
        reps = (K // N) + 1
        return vectors.repeat(reps, 1)[:K]
        
    norms = torch.norm(vectors, dim=1)
    K_outliers = min(4, K // 4)
    K_core = K - K_outliers
    
    top_outlier_idx = torch.topk(norms, k=K_outliers).indices
    outlier_atoms = vectors[top_outlier_idx].clone()
    
    core_mask = torch.ones(N, dtype=torch.bool)
    core_mask[top_outlier_idx] = False
    core_vectors = vectors[core_mask]
    if core_vectors.shape[0] == 0:
        return vectors[:K]
        
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

def find_error_guided_combinations(atoms, vectors, strategy='sparse_pairs'):
    """
    Constrói 6 combinações dos 26 átomos especificamente focadas
    nos blocos que os 26 átomos representam pior.
    """
    K, D = atoms.shape # 26, 8
    N = vectors.shape[0]
    if N == 0:
        return atoms[:6].clone()
        
    # 1. Identificar os 35% blocos de maior erro (hard blocks)
    d_base = torch.cdist(vectors, atoms)
    min_err, _ = torch.min(d_base, dim=1)
    top_k = max(6, int(N * 0.35))
    hard_idx = torch.topk(min_err, k=top_k).indices
    hard_vecs = vectors[hard_idx]
    
    if strategy == 'sparse_pairs':
        # Gera pares com pesos: (A_a + A_b)/2, (A_a - A_b)/2, e combinações ponderadas A_a + 0.5*A_b
        pairs = list(combinations(range(K), 2))
        cands = []
        for a, b in pairs:
            cands.append(0.5 * (atoms[a] + atoms[b]))
            cands.append(0.5 * (atoms[a] - atoms[b]))
            cands.append(atoms[a] + 0.5 * atoms[b])
            cands.append(atoms[a] - 0.5 * atoms[b])
        cand_mat = torch.stack(cands) # (~1300 candidatos em R^8)
        
        # Avaliar ganho contra os blocos difíceis
        sample_hard = hard_vecs[:min(hard_vecs.shape[0], 2500)]
        d_hard_base = torch.min(torch.cdist(sample_hard, atoms), dim=1).values
        d_hard_cands = torch.cdist(sample_hard, cand_mat)
        
        gains = torch.clamp(d_hard_base.unsqueeze(1) - d_hard_cands, min=0.0).sum(dim=0)
        best_6 = torch.topk(gains, k=6).indices
        return cand_mat[best_6]
        
    elif strategy == 'projected_specialists':
        # Treina 6 centróides especialistas nos blocos difíceis, e projeta cada um nos 2 átomos mais alinhados
        specialists = train_kmeans(hard_vecs, K=6)
        projected = []
        for j in range(6):
            spec = specialists[j]
            # Encontra os 2 átomos com maior correlação
            sims = torch.mv(atoms, spec) / (torch.norm(atoms, dim=1) * torch.norm(spec) + 1e-7)
            top2_atoms = torch.topk(sims, k=2).indices
            A1 = atoms[top2_atoms[0]]
            A2 = atoms[top2_atoms[1]]
            # Regressão linear simples: spec ≈ alpha * A1 + beta * A2
            M = torch.stack([A1, A2], dim=1) # (8, 2)
            coeffs = torch.linalg.lstsq(M, spec.unsqueeze(1)).solution.squeeze(1)
            comb = coeffs[0] * A1 + coeffs[1] * A2
            projected.append(comb)
        return torch.stack(projected)

    elif strategy == 'hybrid_atomic':
        # 3 combinações mono-atômicas (escalas/inversão dos átomos mais úteis) + 3 combinações de pares focadas em erro
        # 1. Três pares guiados por erro
        pairs = list(combinations(range(K), 2))
        cands = [0.5 * (atoms[a] + atoms[b]) for a, b in pairs] + [0.5 * (atoms[a] - atoms[b]) for a, b in pairs]
        cand_mat = torch.stack(cands)
        sample_hard = hard_vecs[:min(hard_vecs.shape[0], 2000)]
        d_hard_base = torch.min(torch.cdist(sample_hard, atoms), dim=1).values
        d_hard_cands = torch.cdist(sample_hard, cand_mat)
        gains = torch.clamp(d_hard_base.unsqueeze(1) - d_hard_cands, min=0.0).sum(dim=0)
        best_3_pairs = cand_mat[torch.topk(gains, k=3).indices]
        
        # 2. Três especialistas de erro reais
        spec_3 = train_kmeans(hard_vecs, K=3)
        return torch.cat([spec_3, best_3_pairs], dim=0)

def quantize_layer_with_combinations(weight_tensor, strategy='sparse_pairs', p_crit=0.05, seed=42, block_size=8):
    torch.manual_seed(seed)
    orig_shape = weight_tensor.shape
    w_flat = weight_tensor.float().reshape(-1, block_size)
    N = w_flat.shape[0]
    sample_n = min(N, 35000)
    w_sample = w_flat[:sample_n]
    
    # 1. 8 Famílias Macro
    perm = torch.randperm(sample_n)[:8]
    fam_centers = w_sample[perm].clone()
    for _ in range(5):
        d = torch.cdist(w_sample, fam_centers)
        lbl = torch.argmin(d, dim=1)
        for k in range(8):
            m = (lbl == k)
            if m.sum() > 0: fam_centers[k] = w_sample[m].mean(dim=0)
    fam_ids = torch.argmin(torch.cdist(w_flat, fam_centers), dim=1)
    
    libs_atoms = []
    libs_combs = []
    
    for k in range(8):
        m_k = (fam_ids[:sample_n] == k)
        vecs_k = w_sample[m_k]
        
        # 26 átomos base
        atoms_k = train_kmeans(vecs_k, K=26, seed=seed)
        libs_atoms.append(atoms_k)
        
        # 6 combinações especificamente focadas nos blocos difíceis da família k
        combs_k = find_error_guided_combinations(atoms_k, vecs_k, strategy=strategy)
        libs_combs.append(combs_k)
        
    macro_recon = torch.zeros_like(w_flat)
    stats = {'base_used': 0, 'comb_used': 0}
    
    for k in range(8):
        m = (fam_ids == k)
        if m.sum() == 0: continue
        w_k = w_flat[m]
        atoms_k = libs_atoms[k]
        combs_k = libs_combs[k]
        
        d_base = torch.cdist(w_k, atoms_k)
        min_base, idx_base = torch.min(d_base, dim=1)
        
        d_comb = torch.cdist(w_k, combs_k)
        min_comb, idx_comb = torch.min(d_comb, dim=1)
        
        recon_k = atoms_k[idx_base].clone()
        mask_comb = (min_comb < min_base)
        recon_k[mask_comb] = combs_k[idx_comb[mask_comb]]
        
        num_comb = mask_comb.sum().item()
        stats['comb_used'] += num_comb
        stats['base_used'] += (m.sum().item() - num_comb)
        
        macro_recon[m] = recon_k
        
    # 2. Resíduo 1 (Base 2.0 bpp)
    r1 = w_flat - macro_recon
    r1_sample = r1[:sample_n]
    perm_r1 = torch.randperm(sample_n)[:8]
    r1_centers = r1_sample[perm_r1].clone()
    for _ in range(5):
        d = torch.cdist(r1_sample, r1_centers)
        lbl = torch.argmin(d, dim=1)
        for k in range(8):
            m = (lbl == k)
            if m.sum() > 0: r1_centers[k] = r1_sample[m].mean(dim=0)
    r1_fam_ids = torch.argmin(torch.cdist(r1, r1_centers), dim=1)
    
    libs_r1 = [train_kmeans(r1_sample[r1_fam_ids[:sample_n] == k], K=26, seed=seed) for k in range(8)]
    recon_base = macro_recon.clone()
    for k in range(8):
        m = (r1_fam_ids == k)
        if m.sum() > 0:
            d_rk = torch.cdist(r1[m], libs_r1[k])
            recon_base[m] += libs_r1[k][torch.argmin(d_rk, dim=1)]
            
    # 3. Resíduo 2 (Base 3.0 bpp)
    res_base = w_flat - recon_base
    lib_r2 = train_kmeans(res_base[:sample_n], K=26, seed=seed)
    d_r2 = torch.cdist(res_base, lib_r2)
    recon_final = recon_base + lib_r2[torch.argmin(d_r2, dim=1)]
    
    # 4. Proteção dos top 5% blocos críticos (3.05 bpp)
    if p_crit > 0:
        block_norm = torch.norm(w_flat, dim=1)
        res_3b = w_flat - recon_final
        sensitivity = (block_norm + 1e-5) * torch.mean(res_3b ** 2, dim=1)
        num_crit = int(p_crit * N)
        crit_idx = torch.topk(sensitivity, k=num_crit).indices
        
        r3_vecs = res_3b[crit_idx]
        lib_r3 = train_kmeans(r3_vecs, K=26, seed=seed)
        d_r3 = torch.cdist(r3_vecs, lib_r3)
        recon_final[crit_idx] += lib_r3[torch.argmin(d_r3, dim=1)]
        
    actual_bpp = 3.0 + p_crit
    return recon_final.view(orig_shape), actual_bpp, stats

def evaluate_perplexity(model, tokenizer, text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        loss = model(**inputs, labels=inputs.input_ids).loss
        return torch.exp(loss).item()

def evaluate_generation(model, tokenizer, prompt):
    inp = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=22, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    tokens = tokenizer.encode(text)
    unique_ratio = len(set(tokens)) / len(tokens)
    return text, unique_ratio

def main():
    print("=" * 115)
    print("INVESTIGAÇÃO DE COMBINAÇÕES DOS 26 ÁTOMOS vs C.3 (3.05 bpp - 12 CAMADAS MLP)")
    print("Objetivo: Testar se 6 combinações dos 26 átomos focadas em resíduo mantêm o ganho do C.3")
    print("=" * 115)
    
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
    
    # 0. FP16 Original
    ppl_orig = {d: evaluate_perplexity(base_model, tokenizer, text) for d, text in domains.items()}
    avg_orig = np.mean(list(ppl_orig.values()))
    prompt = "The future of artificial intelligence is"
    gen_orig, div_orig = evaluate_generation(base_model, tokenizer, prompt)
    
    print(f"\n[0] FP16 ORIGINAL (16.0 bpp): Média PPL = {avg_orig:.2f}")
    print(f"    Geração: \"{gen_orig}\" (Diversidade: {div_orig*100:.1f}%)")
    
    from refine_model_c import quantize_layer_c_refined
    
    experiments = [
        ('Comb_Pares_FocoErro', 'Exp 1: 6 Pares Ponderados Focados no Top-35% Erro',
         lambda w: quantize_layer_with_combinations(w, strategy='sparse_pairs', p_crit=0.05, seed=42)),
        ('Comb_Proj_Especialistas', 'Exp 2: 6 Especialistas Projetados nos Átomos da Família',
         lambda w: quantize_layer_with_combinations(w, strategy='projected_specialists', p_crit=0.05, seed=42)),
        ('Comb_Hibrida_Atomica', 'Exp 3: 3 Especialistas Residuais + 3 Melhores Pares da Família',
         lambda w: quantize_layer_with_combinations(w, strategy='hybrid_atomic', p_crit=0.05, seed=42)),
        ('C3_Referencia', 'C.3 Referência: 26 Átomos + 3 Controles + 3 Subfamílias de Erro',
         lambda w: quantize_layer_c_refined(w, variant='C3_error_subfamilies', p_crit=0.05, seed=42))
    ]
    
    results = {}
    
    for code, name, q_fn in experiments:
        print(f"\n" + "-" * 115)
        print(f"Quantizando modelo completo: {name}...")
        print("-" * 115)
        
        m_test = copy.deepcopy(base_model)
        t0 = time.time()
        
        total_spec = 0
        total_base = 0
        
        for block in m_test.transformer.h:
            with torch.no_grad():
                w_fc, _, s_fc = q_fn(block.mlp.c_fc.weight.data)
                block.mlp.c_fc.weight.data = w_fc
                
                w_proj, _, s_proj = q_fn(block.mlp.c_proj.weight.data)
                block.mlp.c_proj.weight.data = w_proj
                
                if 'comb_used' in s_fc:
                    total_spec += (s_fc['comb_used'] + s_proj['comb_used'])
                    total_base += (s_fc['base_used'] + s_proj['base_used'])
                else:
                    total_spec += (s_fc['special_used'] + s_proj['special_used'])
                    total_base += (s_fc['base_used'] + s_proj['base_used'])
                    
        elapsed = time.time() - t0
        rate_spec = (total_spec / (total_spec + total_base)) * 100
        
        ppls = {d: evaluate_perplexity(m_test, tokenizer, text) for d, text in domains.items()}
        avg_ppl = np.mean(list(ppls.values()))
        gen_text, div_ratio = evaluate_generation(m_test, tokenizer, prompt)
        
        results[code] = {
            'name': name,
            'avg_ppl': avg_ppl,
            'ppls': ppls,
            'rate_spec': rate_spec,
            'gen_text': gen_text,
            'div_ratio': div_ratio,
            'time': elapsed
        }
        
        print(f"   -> Média PPL: {avg_ppl:.2f} (Ciência: {ppls['Ciência']:.1f} | Código: {ppls['Código']:.1f} | Hist: {ppls['História']:.1f} | Encic: {ppls['Enciclopédia']:.1f})")
        print(f"   -> Taxa de Uso das 6 Combinações: {rate_spec:.1f}%")
        print(f"   -> Diversidade Léxica: {div_ratio*100:.1f}%")
        print(f"   -> Geração: \"{gen_text}\"")

    print("\n" + "=" * 125)
    print("SCOREBOARD COMPARATIVO FINAL: 6 COMBINAÇÕES vs C.3 A 3.05 BPP")
    print("=" * 125)
    print(f"{'Arquitetura':<45} | {'BPP':<6} | {'PPL Médio':<10} | {'Ciência':<8} | {'Código':<8} | {'Uso 6 Estados':<13} | {'Diversidade'}")
    print("-" * 125)
    print(f"{'FP16 Original (16.0 bpp)':<45} | {'16.0':<6} | {avg_orig:>9.2f}  | {ppl_orig['Ciência']:>8.1f} | {ppl_orig['Código']:>8.1f} | {'-':<13} | {div_orig*100:>10.1f}%")
    print("-" * 125)
    
    for code, name, _ in experiments:
        r = results[code]
        print(f"{r['name'][:45]:<45} | {'3.05 b':<6} | {r['avg_ppl']:>9.2f}  | {r['ppls']['Ciência']:>8.1f} | {r['ppls']['Código']:>8.1f} | {r['rate_spec']:>11.1f}% | {r['div_ratio']*100:>10.1f}%")
    print("=" * 125)

if __name__ == '__main__':
    main()
