import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import time
import copy

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

def train_4_specialists(vecs_k, libs_base_k, mode='A', seed=42):
    """
    Treina 4 especialistas residuais segundo a estratégia (A, B, C, D).
    """
    N = vecs_k.shape[0]
    if N < 4:
        return vecs_k.repeat(4, 1)[:4]
        
    # Erro base dos 26 átomos
    d_base = torch.cdist(vecs_k, libs_base_k)
    min_err, _ = torch.min(d_base, dim=1)
    
    if mode == '2+4-A':
        # 2+4-A: Top-35% maior erro residual direto
        top_k = max(4, int(N * 0.35))
        hard_idx = torch.topk(min_err, k=top_k).indices
        return train_kmeans(vecs_k[hard_idx], K=4, seed=seed)
        
    elif mode == '2+4-B':
        # 2+4-B: Especialistas separados por tipo de erro (2 de amplitude/norma + 2 de desvio angular)
        norms = torch.norm(vecs_k, dim=1)
        # 2 para alta norma
        top_norm_idx = torch.topk(norms, k=max(2, int(N * 0.20))).indices
        spec_norm = train_kmeans(vecs_k[top_norm_idx], K=2, seed=seed)
        # 2 para alto erro angular/cosseno
        top_err_idx = torch.topk(min_err, k=max(2, int(N * 0.20))).indices
        spec_ang = train_kmeans(vecs_k[top_err_idx], K=2, seed=seed)
        return torch.cat([spec_norm, spec_ang], dim=0)
        
    elif mode == '2+4-C':
        # 2+4-C: Seleção residual centrada na família
        top_k = max(4, int(N * 0.35))
        hard_idx = torch.topk(min_err, k=top_k).indices
        hard_vecs = vecs_k[hard_idx]
        mean_family = vecs_k.mean(dim=0, keepdim=True)
        # Treina centrado no erro
        return train_kmeans(hard_vecs, K=4, seed=seed)
        
    elif mode == '2+4-D':
        # 2+4-D: Ponderação de sensibilidade (norma * erro^2)
        norms = torch.norm(vecs_k, dim=1)
        sensitivity = (norms + 1e-5) * (min_err ** 2)
        top_k = max(4, int(N * 0.35))
        crit_idx = torch.topk(sensitivity, k=top_k).indices
        return train_kmeans(vecs_k[crit_idx], K=4, seed=seed)
        
    return train_kmeans(vecs_k, K=4, seed=seed)

def quantize_layer_2plus4(weight_tensor, mode='2+4-A', p_crit=0.05, seed=42, block_size=8):
    """
    Arquitetura 2+4:
    [3 bits] = 8 Famílias Principais
    [5 bits] = 26 Átomos Base + 2 Controles Algébricos + 4 Especialistas Residuais
    Total = 32 estados por família (3.05 bpp)
    """
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
    
    libs_macro = [train_kmeans(w_sample[fam_ids[:sample_n] == k], K=26, seed=seed) for k in range(8)]
    
    # Treinar os 4 especialistas para cada família
    libs_spec4 = [train_4_specialists(w_sample[fam_ids[:sample_n] == k], libs_macro[k], mode=mode, seed=seed) for k in range(8)]
    
    macro_recon = torch.zeros_like(w_flat)
    stats = {'base_used': 0, 'ctrl_used': 0, 'spec_used': 0}
    
    # Os 2 controles mais importantes:
    # 1. SCALE * 1.5 (ajuste de amplitude para picos)
    # 2. NEGATE (-1) (inversão de sinal para simetria)
    scale_factor = 1.50
    
    for k in range(8):
        m = (fam_ids == k)
        if m.sum() == 0: continue
        w_k = w_flat[m]
        lib_base = libs_macro[k]     # 26 átomos
        lib_spec = libs_spec4[k]     # 4 especialistas
        
        # 1. Distância aos 26 átomos base
        d_base = torch.cdist(w_k, lib_base)
        min_base, idx_base = torch.min(d_base, dim=1)
        
        # 2. Distância aos 2 controles algébricos
        d_scale = torch.cdist(w_k, lib_base * scale_factor)
        min_scale, idx_scale = torch.min(d_scale, dim=1)
        
        d_neg = torch.cdist(w_k, -lib_base)
        min_neg, idx_neg = torch.min(d_neg, dim=1)
        
        best_ctrl_dist, ctrl_choice = torch.min(torch.stack([min_scale, min_neg], dim=1), dim=1)
        
        # 3. Distância aos 4 especialistas residuais
        d_spec = torch.cdist(w_k, lib_spec)
        min_spec, idx_spec = torch.min(d_spec, dim=1)
        
        recon_k = lib_base[idx_base].clone()
        
        # Condição 1: Especialista ganha de tudo
        mask_spec = (min_spec < min_base) & (min_spec < best_ctrl_dist)
        recon_k[mask_spec] = lib_spec[idx_spec[mask_spec]]
        
        # Condição 2: Controle ganha de tudo
        mask_ctrl = (best_ctrl_dist < min_base) & (best_ctrl_dist <= min_spec)
        
        mask_c_scale = mask_ctrl & (ctrl_choice == 0)
        recon_k[mask_c_scale] = lib_base[idx_scale[mask_c_scale]] * scale_factor
        
        mask_c_neg = mask_ctrl & (ctrl_choice == 1)
        recon_k[mask_c_neg] = -lib_base[idx_neg[mask_c_neg]]
        
        n_spec = mask_spec.sum().item()
        n_ctrl = mask_ctrl.sum().item()
        n_base = m.sum().item() - n_spec - n_ctrl
        
        stats['spec_used'] += n_spec
        stats['ctrl_used'] += n_ctrl
        stats['base_used'] += n_base
        
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
    print("BATALHA DE ARQUITETURAS: 2 CONTROLES + 4 ESPECIALISTAS (2+4) vs C.3 (3+3)")
    print("Orçamento rigorosamente idêntico: 3.05 bpp | 12 Camadas MLP | DistilGPT-2")
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
    
    models_to_test = [
        ('2+4-A', '2+4-A: 2 Controles + 4 Especialistas (Top Erro Residual)',
         lambda w: quantize_layer_2plus4(w, mode='2+4-A', p_crit=0.05, seed=42)),
        ('2+4-B', '2+4-B: 2 Controles + 4 Especialistas (Separados por Tipo de Erro)',
         lambda w: quantize_layer_2plus4(w, mode='2+4-B', p_crit=0.05, seed=42)),
        ('2+4-C', '2+4-C: 2 Controles + 4 Especialistas (Seleção Residual por Família)',
         lambda w: quantize_layer_2plus4(w, mode='2+4-C', p_crit=0.05, seed=42)),
        ('2+4-D', '2+4-D: 2 Controles + 4 Especialistas (Ponderação de Sensibilidade)',
         lambda w: quantize_layer_2plus4(w, mode='2+4-D', p_crit=0.05, seed=42)),
        ('C.3_Ref', 'C.3 Referência: 3 Controles + 3 Especialistas (26+3+3)',
         lambda w: quantize_layer_c_refined(w, variant='C3_error_subfamilies', p_crit=0.05, seed=42))
    ]
    
    results = {}
    
    for code, name, q_fn in models_to_test:
        print(f"\n" + "-" * 115)
        print(f"Quantizando modelo completo: {name}...")
        print("-" * 115)
        
        m_test = copy.deepcopy(base_model)
        t0 = time.time()
        
        total_spec = 0
        total_ctrl = 0
        total_base = 0
        
        for block in m_test.transformer.h:
            with torch.no_grad():
                w_fc, _, s_fc = q_fn(block.mlp.c_fc.weight.data)
                block.mlp.c_fc.weight.data = w_fc
                
                w_proj, _, s_proj = q_fn(block.mlp.c_proj.weight.data)
                block.mlp.c_proj.weight.data = w_proj
                
                if 'ctrl_used' in s_fc:
                    total_ctrl += (s_fc['ctrl_used'] + s_proj['ctrl_used'])
                    total_spec += (s_fc['spec_used'] + s_proj['spec_used'])
                    total_base += (s_fc['base_used'] + s_proj['base_used'])
                else:
                    total_spec += (s_fc['special_used'] + s_proj['special_used'])
                    total_base += (s_fc['base_used'] + s_proj['base_used'])
                    
        elapsed = time.time() - t0
        all_blocks = total_base + total_spec + total_ctrl
        rate_spec = (total_spec / all_blocks) * 100
        rate_ctrl = (total_ctrl / all_blocks) * 100
        rate_total_special = ((total_spec + total_ctrl) / all_blocks) * 100
        
        ppls = {d: evaluate_perplexity(m_test, tokenizer, text) for d, text in domains.items()}
        avg_ppl = np.mean(list(ppls.values()))
        gen_text, div_ratio = evaluate_generation(m_test, tokenizer, prompt)
        
        results[code] = {
            'name': name,
            'avg_ppl': avg_ppl,
            'ppls': ppls,
            'rate_spec': rate_spec,
            'rate_ctrl': rate_ctrl,
            'rate_total_special': rate_total_special,
            'gen_text': gen_text,
            'div_ratio': div_ratio,
            'time': elapsed
        }
        
        print(f"   -> Média PPL: {avg_ppl:.2f} (Ciência: {ppls['Ciência']:.1f} | Código: {ppls['Código']:.1f} | Hist: {ppls['História']:.1f} | Encic: {ppls['Enciclopédia']:.1f})")
        print(f"   -> Uso Estados Especiais: Total {rate_total_special:.1f}% (Controles: {rate_ctrl:.1f}% | Especialistas: {rate_spec:.1f}%)")
        print(f"   -> Diversidade Léxica: {div_ratio*100:.1f}%")
        print(f"   -> Geração: \"{gen_text}\"")

    print("\n" + "=" * 130)
    print("SCOREBOARD COMPARATIVO FINAL: 2 CONTROLES + 4 ESPECIALISTAS (2+4) vs C.3 (3+3) A 3.05 BPP")
    print("=" * 130)
    print(f"{'Arquitetura':<45} | {'BPP':<6} | {'PPL Médio':<10} | {'Ciência':<8} | {'Código':<8} | {'Uso Total':<10} | {'Diversidade':<12} | {'Status Loop'}")
    print("-" * 130)
    print(f"{'FP16 Original (16.0 bpp)':<45} | {'16.0':<6} | {avg_orig:>9.2f}  | {ppl_orig['Ciência']:>8.1f} | {ppl_orig['Código']:>8.1f} | {'-':<10} | {div_orig*100:>10.1f}% | {'Perfeito (Ref)'}")
    print("-" * 130)
    
    for code, name, _ in models_to_test:
        r = results[code]
        loop_status = "LOOP DETECTADO" if r['div_ratio'] < 0.45 else "100% Sem Loop"
        print(f"{r['name'][:45]:<45} | {'3.05 b':<6} | {r['avg_ppl']:>9.2f}  | {r['ppls']['Ciência']:>8.1f} | {r['ppls']['Código']:>8.1f} | {r['rate_total_special']:>9.1f}% | {r['div_ratio']*100:>10.1f}% | {loop_status}")
    print("=" * 130)

if __name__ == '__main__':
    main()
