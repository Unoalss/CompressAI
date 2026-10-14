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

def find_top_6_substate_combinations(substates, vectors, mode='average'):
    """
    Dada uma família com 26 subestados (shape: 26, D),
    encontra as 6 melhores combinações de pares de subestados
    que mais reduzem o erro de reconstrução nos vetores da família.
    
    mode:
      - 'average': (S_a + S_b) / 2
      - 'diff_sum': (S_a + S_b) / 2 e (S_a - S_b) / 2
    """
    K, D = substates.shape
    pairs = list(combinations(range(K), 2)) # 325 pares
    
    cand_list = []
    pair_meta = []
    
    for a, b in pairs:
        # Combinação média
        cand_list.append((substates[a] + substates[b]) * 0.5)
        pair_meta.append((a, b, '+'))
        if mode == 'diff_sum':
            # Combinação diferencial
            cand_list.append((substates[a] - substates[b]) * 0.5)
            pair_meta.append((a, b, '-'))
            
    cand_matrix = torch.stack(cand_list) # (325 ou 650, D)
    
    # Avaliar contra os vetores da família
    N = vectors.shape[0]
    if N == 0:
        return cand_matrix[:6]
        
    # Erro base com subestados individuais
    d_single = torch.cdist(vectors, substates)
    min_single_err = torch.min(d_single, dim=1).values
    
    # Erro com os candidatos de pares
    sample_eval = vectors[:min(N, 5000)]
    sample_single_err = min_single_err[:min(N, 5000)]
    d_cands = torch.cdist(sample_eval, cand_matrix)
    
    # Ganho de cada candidato sobre o átomo individual
    cand_gain = torch.zeros(cand_matrix.shape[0])
    for j in range(cand_matrix.shape[0]):
        # Quanto este par reduz o erro quando é melhor que o átomo individual
        improvement = torch.clamp(sample_single_err - d_cands[:, j], min=0.0)
        cand_gain[j] = improvement.sum()
        
    # Top 6 combinações com maior ganho marginal
    top_6_idx = torch.topk(cand_gain, k=6).indices
    return cand_matrix[top_6_idx]

def quantize_substate_combinations(weight_tensor, comb_mode='average', p_crit=0.05, seed=42, block_size=8):
    """
    [3 bits] = 8 Famílias
    [5 bits] = 26 Subestados + 6 Combinações dos próprios subestados da família
    """
    torch.manual_seed(seed)
    orig_shape = weight_tensor.shape
    w_flat = weight_tensor.float().reshape(-1, block_size)
    N = w_flat.shape[0]
    sample_n = min(N, 35000)
    w_sample = w_flat[:sample_n]
    
    # 1. 8 Famílias Principais (3 bits)
    perm = torch.randperm(sample_n)[:8]
    fam_centers = w_sample[perm].clone()
    for _ in range(5):
        d = torch.cdist(w_sample, fam_centers)
        lbl = torch.argmin(d, dim=1)
        for k in range(8):
            m = (lbl == k)
            if m.sum() > 0: fam_centers[k] = w_sample[m].mean(dim=0)
    fam_ids = torch.argmin(torch.cdist(w_flat, fam_centers), dim=1)
    
    # 2. Para cada Família: 26 Subestados + 6 Combinações dos seus subestados
    libs_substates = []
    libs_combinations = []
    
    for k in range(8):
        m_k = (fam_ids[:sample_n] == k)
        vecs_k = w_sample[m_k]
        
        # 26 Subestados base da família
        substates_k = train_kmeans(vecs_k, K=26, seed=seed)
        libs_substates.append(substates_k)
        
        # 6 Combinações especiais formadas a partir dos 26 subestados da família k
        combs_k = find_top_6_substate_combinations(substates_k, vecs_k, mode=comb_mode)
        libs_combinations.append(combs_k)
        
    macro_recon = torch.zeros_like(w_flat)
    stats = {'substate_used': 0, 'combination_used': 0}
    
    # Codificação: 26 Subestados vs 6 Combinações
    for k in range(8):
        m = (fam_ids == k)
        if m.sum() == 0: continue
        w_k = w_flat[m]
        substates_k = libs_substates[k]     # 26 subestados
        combs_k = libs_combinations[k]       # 6 combinações
        
        d_sub = torch.cdist(w_k, substates_k)
        min_sub, idx_sub = torch.min(d_sub, dim=1)
        
        d_comb = torch.cdist(w_k, combs_k)
        min_comb, idx_comb = torch.min(d_comb, dim=1)
        
        recon_k = substates_k[idx_sub].clone()
        mask_comb = (min_comb < min_sub)
        recon_k[mask_comb] = combs_k[idx_comb[mask_comb]]
        
        num_comb = mask_comb.sum().item()
        stats['combination_used'] += num_comb
        stats['substate_used'] += (m.sum().item() - num_comb)
        
        macro_recon[m] = recon_k
        
    # 3. Resíduo 1 (Base 2.0 bpp)
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
            
    # 4. Resíduo 2 (Base 3.0 bpp)
    res_base = w_flat - recon_base
    lib_r2 = train_kmeans(res_base[:sample_n], K=26, seed=seed)
    d_r2 = torch.cdist(res_base, lib_r2)
    recon_final = recon_base + lib_r2[torch.argmin(d_r2, dim=1)]
    
    # 5. Proteção dos top 5% blocos críticos (3.05 bpp)
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
    print("TESTE DAS COMBINAÇÕES INTERNAS DA FAMÍLIA: 8 FAMÍLIAS x [26 SUBESTADOS + 6 COMBINAÇÕES]")
    print("Comparação com C.3 (Controles + Erro) e FP16 Original | 12 Camadas MLP | 3.05 bpp")
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
    
    variants = [
        ('Comb_Pares_Media', 'Combinações 1: Média dos 6 Melhores Pares de Subestados (Sa + Sb)/2',
         lambda w: quantize_substate_combinations(w, comb_mode='average', p_crit=0.05, seed=42)),
        ('Comb_Pares_DiffSum', 'Combinações 2: Pares Média e Diferença (Sa +/- Sb)/2',
         lambda w: quantize_substate_combinations(w, comb_mode='diff_sum', p_crit=0.05, seed=42)),
        ('C3_Referencia', 'C.3 Referência: 26 Átomos + 3 Controles + 3 Subfamílias de Erro',
         lambda w: quantize_layer_c_refined(w, variant='C3_error_subfamilies', p_crit=0.05, seed=42))
    ]
    
    results = {}
    
    for code, name, q_fn in variants:
        print(f"\n" + "-" * 115)
        print(f"Executando {name}...")
        print("-" * 115)
        
        m_test = copy.deepcopy(base_model)
        t0 = time.time()
        
        total_spec_used = 0
        total_base_used = 0
        
        for block in m_test.transformer.h:
            with torch.no_grad():
                w_fc, _, s_fc = q_fn(block.mlp.c_fc.weight.data)
                block.mlp.c_fc.weight.data = w_fc
                
                w_proj, _, s_proj = q_fn(block.mlp.c_proj.weight.data)
                block.mlp.c_proj.weight.data = w_proj
                
                if 'combination_used' in s_fc:
                    total_spec_used += (s_fc['combination_used'] + s_proj['combination_used'])
                    total_base_used += (s_fc['substate_used'] + s_proj['substate_used'])
                else:
                    total_spec_used += (s_fc['special_used'] + s_proj['special_used'])
                    total_base_used += (s_fc['base_used'] + s_proj['base_used'])
                    
        elapsed = time.time() - t0
        rate_spec = (total_spec_used / (total_spec_used + total_base_used)) * 100
        
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
    print("SCOREBOARD COMPARATIVO FINAL: COMBINAÇÕES DE SUBESTADOS vs C.3 A 3.05 BPP")
    print("=" * 125)
    print(f"{'Arquitetura':<45} | {'BPP':<6} | {'PPL Médio':<10} | {'Ciência':<8} | {'Código':<8} | {'Uso 6 Estados':<13} | {'Diversidade'}")
    print("-" * 125)
    print(f"{'FP16 Original':<45} | {'16.0':<6} | {avg_orig:>9.2f}  | {ppl_orig['Ciência']:>8.1f} | {ppl_orig['Código']:>8.1f} | {'-':<13} | {div_orig*100:>10.1f}%")
    print("-" * 125)
    
    for code, name, _ in variants:
        r = results[code]
        print(f"{r['name'][:45]:<45} | {'3.05 b':<6} | {r['avg_ppl']:>9.2f}  | {r['ppls']['Ciência']:>8.1f} | {r['ppls']['Código']:>8.1f} | {r['rate_spec']:>11.1f}% | {r['div_ratio']*100:>10.1f}%")
    print("=" * 125)

if __name__ == '__main__':
    main()
