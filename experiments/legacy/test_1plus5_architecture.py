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

def quantize_matrix_custom(weight_tensor, num_ctrl=1, ctrl_type='negate', num_spec=5, spec_mode='sensitivity', p_crit=0.05, seed=42, block_size=8):
    """
    Quantizador flexível para 1+5, 2+4 ou 3+3.
    ctrl_type:
      - 'negate': o único controle é -1 * átomo
      - 'scale': o único controle é 1.5 * átomo
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
    
    # Treinar num_spec especialistas
    libs_spec = []
    for k in range(8):
        m_k = (fam_ids[:sample_n] == k)
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
        else: # standard
            top_k = max(num_spec, int(vecs_k.shape[0] * 0.35))
            hard_idx = torch.topk(min_err, k=top_k).indices
            spec_k = train_kmeans(vecs_k[hard_idx], K=num_spec, seed=seed)
            
        libs_spec.append(spec_k)
        
    macro_recon = torch.zeros_like(w_flat)
    stats = {'base_used': 0, 'ctrl_used': 0, 'spec_used': 0}
    
    for k in range(8):
        m = (fam_ids == k)
        if m.sum() == 0: continue
        w_k = w_flat[m]
        lib_base = libs_macro[k]
        lib_spec = libs_spec[k]
        
        d_base = torch.cdist(w_k, lib_base)
        min_base, idx_base = torch.min(d_base, dim=1)
        
        # Avalia controles algébricos
        if num_ctrl == 1:
            if ctrl_type == 'negate':
                d_c = torch.cdist(w_k, -lib_base)
                min_ctrl, idx_c = torch.min(d_c, dim=1)
                best_ctrl_val = min_ctrl
                choice_ctrl = torch.zeros(w_k.shape[0], dtype=torch.long)
            else: # 'scale'
                d_c = torch.cdist(w_k, lib_base * 1.5)
                min_ctrl, idx_c = torch.min(d_c, dim=1)
                best_ctrl_val = min_ctrl
                choice_ctrl = torch.zeros(w_k.shape[0], dtype=torch.long)
        elif num_ctrl == 2:
            d_c0 = torch.cdist(w_k, lib_base * 1.5)
            d_c1 = torch.cdist(w_k, -lib_base)
            min_c0, idx_c0 = torch.min(d_c0, dim=1)
            min_c1, idx_c1 = torch.min(d_c1, dim=1)
            best_ctrl_val, choice_ctrl = torch.min(torch.stack([min_c0, min_c1], dim=1), dim=1)
            ctrl_indices = [idx_c0, idx_c1]
        else: # num_ctrl == 3
            d_c0 = torch.cdist(w_k, lib_base * 1.5)
            d_c1 = torch.cdist(w_k, lib_base * 0.5)
            d_c2 = torch.cdist(w_k, -lib_base)
            min_c0, idx_c0 = torch.min(d_c0, dim=1)
            min_c1, idx_c1 = torch.min(d_c1, dim=1)
            min_c2, idx_c2 = torch.min(d_c2, dim=1)
            best_ctrl_val, choice_ctrl = torch.min(torch.stack([min_c0, min_c1, min_c2], dim=1), dim=1)
            ctrl_indices = [idx_c0, idx_c1, idx_c2]
            
        # Avalia especialistas residuais
        d_spec = torch.cdist(w_k, lib_spec)
        min_spec_val, idx_spec = torch.min(d_spec, dim=1)
        
        recon_k = lib_base[idx_base].clone()
        win_spec = (min_spec_val < min_base) & (min_spec_val < best_ctrl_val)
        win_ctrl = (best_ctrl_val < min_base) & (best_ctrl_val <= min_spec_val)
        
        recon_k[win_spec] = lib_spec[idx_spec[win_spec]]
        
        if num_ctrl == 1:
            if ctrl_type == 'negate':
                recon_k[win_ctrl] = -lib_base[idx_c[win_ctrl]]
            else:
                recon_k[win_ctrl] = lib_base[idx_c[win_ctrl]] * 1.5
        elif num_ctrl == 2:
            mask_c0 = win_ctrl & (choice_ctrl == 0)
            mask_c1 = win_ctrl & (choice_ctrl == 1)
            recon_k[mask_c0] = lib_base[ctrl_indices[0][mask_c0]] * 1.5
            recon_k[mask_c1] = -lib_base[ctrl_indices[1][mask_c1]]
        else:
            mask_c0 = win_ctrl & (choice_ctrl == 0)
            mask_c1 = win_ctrl & (choice_ctrl == 1)
            mask_c2 = win_ctrl & (choice_ctrl == 2)
            recon_k[mask_c0] = lib_base[ctrl_indices[0][mask_c0]] * 1.5
            recon_k[mask_c1] = lib_base[ctrl_indices[1][mask_c1]] * 0.5
            recon_k[mask_c2] = -lib_base[ctrl_indices[2][mask_c2]]
            
        stats['spec_used'] += win_spec.sum().item()
        stats['ctrl_used'] += win_ctrl.sum().item()
        stats['base_used'] += ((~win_spec) & (~win_ctrl)).sum().item()
        
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
    
    # 4. Proteção Seletiva Top P_crit%
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
    recon_reshaped = recon_final.view(orig_shape)
    
    mse = torch.mean((weight_tensor.float() - recon_reshaped) ** 2).item()
    cos = torch.nn.functional.cosine_similarity(weight_tensor.float().view(1, -1), recon_reshaped.view(1, -1)).item()
    
    return recon_reshaped, actual_bpp, stats, mse, cos

def evaluate_perplexity(model, tokenizer, text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        loss = model(**inputs, labels=inputs.input_ids).loss
        return torch.exp(loss).item()

def evaluate_generation(model, tokenizer, prompt, max_tokens=25):
    inp = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    tokens = tokenizer.encode(text)
    unique_ratio = len(set(tokens)) / len(tokens)
    return text, unique_ratio

def main():
    print("=" * 125)
    print("BATALHA DE ARQUITETURAS: 1 CONTROLE + 5 ESPECIALISTAS (1+5) vs H5-A (2+4 / 3+3)")
    print("Orçamento rigorosamente fixo: ~3.05 bpp (Redistribuição H5-A) | 12 Camadas MLP | DistilGPT-2")
    print("=" * 125)
    
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
    
    test_prompt = "The future of artificial intelligence is"
    
    ppl_orig = {d: evaluate_perplexity(base_model, tokenizer, text) for d, text in domains.items()}
    avg_orig = np.mean(list(ppl_orig.values()))
    gen_orig, div_orig = evaluate_generation(base_model, tokenizer, test_prompt)
    print(f"\n[0] FP16 ORIGINAL (16.0 bpp): Média PPL = {avg_orig:.2f}")
    print(f"    Geração: \"{gen_orig}\" (Diversidade: {div_orig*100:.1f}%)")
    
    # Configurações do experimento
    experiments = [
        ('H5-A Campeão', 'H5-A Atual: c_fc=(2+4-D) | c_proj=(3+3) [Referência: 77.34 PPL]',
         {'c_fc': (2, 'negate', 4, 'sensitivity'), 'c_proj': (3, 'negate', 3, 'standard')}),
        ('1+5 Puro (Neg)', '1+5 Puro (Negate): c_fc=(1+5) | c_proj=(1+5) com Negate',
         {'c_fc': (1, 'negate', 5, 'sensitivity'), 'c_proj': (1, 'negate', 5, 'standard')}),
        ('1+5 Puro (Scale)', '1+5 Puro (Scale): c_fc=(1+5) | c_proj=(1+5) com Scale 1.5',
         {'c_fc': (1, 'scale', 5, 'sensitivity'), 'c_proj': (1, 'scale', 5, 'standard')}),
        ('Híbrido 1+5 / 3+3', 'Híbrido: c_fc=(1+5 Negate) | c_proj=(3+3)',
         {'c_fc': (1, 'negate', 5, 'sensitivity'), 'c_proj': (3, 'negate', 3, 'standard')}),
        ('Híbrido 1+5 / 2+4', 'Híbrido: c_fc=(1+5 Negate) | c_proj=(2+4-D)',
         {'c_fc': (1, 'negate', 5, 'sensitivity'), 'c_proj': (2, 'negate', 4, 'sensitivity')})
    ]
    
    results = {}
    
    for code, name, cfg in experiments:
        print(f"\n" + "-" * 120)
        print(f"Quantizando modelo completo: {name}...")
        print("-" * 120)
        
        m_test = copy.deepcopy(base_model)
        t0 = time.time()
        
        bpp_list = []
        tot_spec = 0
        tot_ctrl = 0
        tot_base = 0
        
        for l_idx, block in enumerate(m_test.transformer.h):
            # Preserva a redistribuição H5-A: h0=3%, h5.c_proj=10%, resto=5%
            p_fc = 0.03 if l_idx == 0 else 0.05
            p_proj = 0.03 if l_idx == 0 else (0.10 if l_idx == 5 else 0.05)
            
            with torch.no_grad():
                # c_fc
                n_c, c_t, n_s, s_m = cfg['c_fc']
                w_fc, bpp_fc, s_fc, _, _ = quantize_matrix_custom(
                    block.mlp.c_fc.weight.data, num_ctrl=n_c, ctrl_type=c_t, num_spec=n_s, spec_mode=s_m, p_crit=p_fc, seed=42
                )
                block.mlp.c_fc.weight.data = w_fc
                bpp_list.append(bpp_fc)
                tot_spec += s_fc['spec_used']
                tot_ctrl += s_fc['ctrl_used']
                tot_base += s_fc['base_used']
                
                # c_proj
                n_cp, c_tp, n_sp, s_mp = cfg['c_proj']
                w_proj, bpp_proj, s_proj, _, _ = quantize_matrix_custom(
                    block.mlp.c_proj.weight.data, num_ctrl=n_cp, ctrl_type=c_tp, num_spec=n_sp, spec_mode=s_mp, p_crit=p_proj, seed=42
                )
                block.mlp.c_proj.weight.data = w_proj
                bpp_list.append(bpp_proj)
                tot_spec += s_proj['spec_used']
                tot_ctrl += s_proj['ctrl_used']
                tot_base += s_proj['base_used']
                
        elapsed = time.time() - t0
        actual_model_bpp = np.mean(bpp_list)
        all_blocks = tot_base + tot_ctrl + tot_spec
        rate_ctrl = (tot_ctrl / all_blocks) * 100
        rate_spec = (tot_spec / all_blocks) * 100
        rate_total_special = ((tot_ctrl + tot_spec) / all_blocks) * 100
        
        ppls = {d: evaluate_perplexity(m_test, tokenizer, text) for d, text in domains.items()}
        avg_ppl = np.mean(list(ppls.values()))
        gen_text, div_ratio = evaluate_generation(m_test, tokenizer, test_prompt)
        
        results[code] = {
            'name': name,
            'bpp': actual_model_bpp,
            'avg_ppl': avg_ppl,
            'ppls': ppls,
            'rate_ctrl': rate_ctrl,
            'rate_spec': rate_spec,
            'rate_total_special': rate_total_special,
            'gen_text': gen_text,
            'div_ratio': div_ratio,
            'time': elapsed
        }
        
        print(f"   -> BPP: {actual_model_bpp:.4f} b | PPL Médio: {avg_ppl:.2f} (Ciência: {ppls['Ciência']:.1f} | Código: {ppls['Código']:.1f} | Hist: {ppls['História']:.1f} | Encic: {ppls['Enciclopédia']:.1f})")
        print(f"   -> Uso Estados Especiais: Total {rate_total_special:.1f}% (Ctrl: {rate_ctrl:.1f}% | Spec: {rate_spec:.1f}%)")
        print(f"   -> Diversidade Léxica: {div_ratio*100:.1f}%")
        clean_g = gen_text.replace('\n', ' ')
        print(f"   -> Geração: \"{clean_g[:80]}...\"")

    print("\n" + "=" * 135)
    print("SCOREBOARD COMPARATIVO FINAL: 1+5 PURO & HÍBRIDOS vs H5-A (3.05 BPP)")
    print("=" * 135)
    print(f"{'Configuração':<35} | {'BPP':<7} | {'PPL Médio':<10} | {'Ciência':<8} | {'Código':<8} | {'História':<8} | {'Encic':<8} | {'Uso Total':<10} | {'Diversidade'}")
    print("-" * 135)
    print(f"{'FP16 Original (16.0 bpp)':<35} | {'16.000':<7} | {avg_orig:>9.2f}  | {ppl_orig['Ciência']:>8.1f} | {ppl_orig['Código']:>8.1f} | {ppl_orig['História']:>8.1f} | {ppl_orig['Enciclopédia']:>8.1f} | {'-':<10} | {div_orig*100:>10.1f}%")
    print("-" * 135)
    
    for code, name, _ in experiments:
        r = results[code]
        print(f"{name[:35]:<35} | {r['bpp']:>6.4f}b | {r['avg_ppl']:>9.2f}  | {r['ppls']['Ciência']:>8.1f} | {r['ppls']['Código']:>8.1f} | {r['ppls']['História']:>8.1f} | {r['ppls']['Enciclopédia']:>8.1f} | {r['rate_total_special']:>9.1f}% | {r['div_ratio']*100:>10.1f}%")
    print("=" * 135)

    print("\n" + "=" * 135)
    print("COMPARAÇÃO QUALITATIVA DE GERAÇÃO AUTOREGRESSIVA")
    print("=" * 135)
    for code, name, _ in experiments:
        g = results[code]['gen_text'].replace('\n', ' ')
        print(f"[{code:<18}]: \"{g[:95]}...\"")
    print("=" * 135)

if __name__ == '__main__':
    main()
