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

def quantize_flexible(weight_tensor, num_ctrl=2, num_spec=4, spec_mode='sensitivity', p_crit=0.05, seed=42, block_size=8):
    """
    Quantizador flexível para permitir qualquer combinação:
    num_ctrl: 2 ou 3 controles
    num_spec: 3 ou 4 especialistas
    spec_mode: 'sensitivity' (fórmula ponderada), 'standard' (apenas resíduo de erro), 'shuffled' (aleatório)
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
    
    # Treinar especialistas por família
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
        elif spec_mode == 'standard':
            top_k = max(num_spec, int(vecs_k.shape[0] * 0.35))
            hard_idx = torch.topk(min_err, k=top_k).indices
            spec_k = train_kmeans(vecs_k[hard_idx], K=num_spec, seed=seed)
        elif spec_mode == 'shuffled':
            # Embaralha / centróides aleatórios
            perm_rand = torch.randperm(vecs_k.shape[0])[:num_spec]
            spec_k = vecs_k[perm_rand].clone()
        else:
            spec_k = train_kmeans(vecs_k, K=num_spec, seed=seed)
            
        libs_spec.append(spec_k)
        
    macro_recon = torch.zeros_like(w_flat)
    
    # Rastreamento individual de uso por estado
    stats = {
        'atom_usage': np.zeros(26, dtype=int),
        'ctrl_usage': np.zeros(num_ctrl, dtype=int),
        'spec_usage': np.zeros(num_spec, dtype=int)
    }
    
    for k in range(8):
        m = (fam_ids == k)
        if m.sum() == 0: continue
        w_k = w_flat[m]
        lib_base = libs_macro[k]     # 26 átomos
        lib_spec = libs_spec[k]      # num_spec especialistas
        
        # 1. Distância átomos base
        d_base = torch.cdist(w_k, lib_base)
        min_base, idx_base = torch.min(d_base, dim=1)
        
        # 2. Controles algébricos
        if num_ctrl == 3:
            # 3 controles: SCALE 1.5, SCALE 0.5, NEGATE
            d_c0 = torch.cdist(w_k, lib_base * 1.5)
            d_c1 = torch.cdist(w_k, lib_base * 0.5)
            d_c2 = torch.cdist(w_k, -lib_base)
            min_c0, idx_c0 = torch.min(d_c0, dim=1)
            min_c1, idx_c1 = torch.min(d_c1, dim=1)
            min_c2, idx_c2 = torch.min(d_c2, dim=1)
            best_ctrl_dists = torch.stack([min_c0, min_c1, min_c2], dim=1)
            ctrl_indices = [idx_c0, idx_c1, idx_c2]
        else:
            # 2 controles: SCALE 1.5, NEGATE
            d_c0 = torch.cdist(w_k, lib_base * 1.5)
            d_c1 = torch.cdist(w_k, -lib_base)
            min_c0, idx_c0 = torch.min(d_c0, dim=1)
            min_c1, idx_c1 = torch.min(d_c1, dim=1)
            best_ctrl_dists = torch.stack([min_c0, min_c1], dim=1)
            ctrl_indices = [idx_c0, idx_c1]
            
        min_ctrl_val, choice_ctrl = torch.min(best_ctrl_dists, dim=1)
        
        # 3. Especialistas residuais
        d_spec = torch.cdist(w_k, lib_spec)
        min_spec_val, idx_spec = torch.min(d_spec, dim=1)
        
        # Decisão global de reconstrução
        recon_k = lib_base[idx_base].clone()
        
        # Máscaras de vitória
        win_spec = (min_spec_val < min_base) & (min_spec_val < min_ctrl_val)
        win_ctrl = (min_ctrl_val < min_base) & (min_ctrl_val <= min_spec_val)
        win_base = (~win_spec) & (~win_ctrl)
        
        # Aplica especialistas
        recon_k[win_spec] = lib_spec[idx_spec[win_spec]]
        
        # Aplica controles
        if num_ctrl == 3:
            mask_c0 = win_ctrl & (choice_ctrl == 0)
            mask_c1 = win_ctrl & (choice_ctrl == 1)
            mask_c2 = win_ctrl & (choice_ctrl == 2)
            recon_k[mask_c0] = lib_base[ctrl_indices[0][mask_c0]] * 1.5
            recon_k[mask_c1] = lib_base[ctrl_indices[1][mask_c1]] * 0.5
            recon_k[mask_c2] = -lib_base[ctrl_indices[2][mask_c2]]
            for c_i in range(3):
                stats['ctrl_usage'][c_i] += (win_ctrl & (choice_ctrl == c_i)).sum().item()
        else:
            mask_c0 = win_ctrl & (choice_ctrl == 0)
            mask_c1 = win_ctrl & (choice_ctrl == 1)
            recon_k[mask_c0] = lib_base[ctrl_indices[0][mask_c0]] * 1.5
            recon_k[mask_c1] = -lib_base[ctrl_indices[1][mask_c1]]
            for c_i in range(2):
                stats['ctrl_usage'][c_i] += (win_ctrl & (choice_ctrl == c_i)).sum().item()
                
        # Registra uso dos especialistas individuais
        for s_i in range(num_spec):
            stats['spec_usage'][s_i] += (win_spec & (idx_spec == s_i)).sum().item()
            
        # Registra uso dos átomos base
        for a_i in range(26):
            stats['atom_usage'][a_i] += (win_base & (idx_base == a_i)).sum().item()
            
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
    recon_reshaped = recon_final.view(orig_shape)
    
    mse = torch.mean((weight_tensor.float() - recon_reshaped) ** 2).item()
    cos = torch.nn.functional.cosine_similarity(weight_tensor.float().view(1, -1), recon_reshaped.view(1, -1)).item()
    
    return recon_reshaped, actual_bpp, stats, mse, cos

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
    print("=" * 120)
    print("ABLACAO DO M.2 & AUDITORIA FINA POR CAMADA: POR QUE O M.2 CHEGOU A 79.99?")
    print("Bateria de 4 Prompts | 12 Camadas MLP | 3.05 bpp | DistilGPT-2")
    print("=" * 120)
    
    model_name = "distilbert/distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    base_model.eval()
    
    domains = {
        "Ciencia": "Quantum mechanics describes nature at the smallest scales of energy levels of atoms and subatomic particles.",
        "Codigo": "def binary_search(array, target): low = 0; high = len(array) - 1; while low <= high: mid = (low + high) // 2",
        "Historia": "The ancient library of Alexandria was once the largest repository of knowledge in the Hellenistic world.",
        "Enciclopedia": "The solar system formed approximately 4.6 billion years ago from the gravitational collapse of a molecular cloud."
    }
    
    prompts = [
        ("Prompt 1 (IA)", "The future of artificial intelligence is"),
        ("Prompt 2 (Logica)", "To solve a complex problem, one must first"),
        ("Prompt 3 (Fisica)", "Quantum entanglement implies that two particles"),
        ("Prompt 4 (Codigo)", "def fibonacci(n):")
    ]
    
    # 0. FP16 Original
    ppl_orig = {d: evaluate_perplexity(base_model, tokenizer, text) for d, text in domains.items()}
    avg_orig = np.mean(list(ppl_orig.values()))
    
    print(f"\n[0] FP16 ORIGINAL (16.0 bpp): Media PPL = {avg_orig:.2f}")
    for p_id, p_text in prompts:
        gen, div = evaluate_generation(base_model, tokenizer, p_text)
        print(f"    {p_id}: \"{gen[:70]}...\" (Div: {div*100:.1f}%)")
        
    ablation_tests = [
        ('C.3', 'C.3 Referencia: c_fc=(3+3) | c_proj=(3+3)',
         {'c_fc': (3, 3, 'standard'), 'c_proj': (3, 3, 'standard')}),
        ('2+4 Puro', '2+4 Puro: c_fc=(2+4-D) | c_proj=(2+4-D)',
         {'c_fc': (2, 4, 'sensitivity'), 'c_proj': (2, 4, 'sensitivity')}),
        ('M.2', 'M.2 Campeao: c_fc=(2+4-D) | c_proj=(3+3)',
         {'c_fc': (2, 4, 'sensitivity'), 'c_proj': (3, 3, 'standard')}),
        ('M.2-A', 'M.2-A: c_fc sem 4o especialista (2+3-D) | c_proj=(3+3)',
         {'c_fc': (2, 3, 'sensitivity'), 'c_proj': (3, 3, 'standard')}),
        ('M.2-B', 'M.2-B: c_fc com residuo normal (2+4 standard) | c_proj=(3+3)',
         {'c_fc': (2, 4, 'standard'), 'c_proj': (3, 3, 'standard')}),
        ('M.2-C', 'M.2-C: c_fc especialistas embaralhados | c_proj=(3+3)',
         {'c_fc': (2, 4, 'shuffled'), 'c_proj': (3, 3, 'standard')})
    ]
    
    results = {}
    layer_audit_m2 = []
    
    for code, name, cfg in ablation_tests:
        print(f"\n" + "-" * 115)
        print(f"Executando Ablacao: {name}...")
        print("-" * 115)
        
        m_test = copy.deepcopy(base_model)
        t0 = time.time()
        
        is_m2 = (code == 'M.2')
        
        for l_idx, block in enumerate(m_test.transformer.h):
            with torch.no_grad():
                # c_fc
                n_c, n_s, s_m = cfg['c_fc']
                w_fc, _, s_fc, mse_fc, cos_fc = quantize_flexible(block.mlp.c_fc.weight.data, num_ctrl=n_c, num_spec=n_s, spec_mode=s_m, p_crit=0.05, seed=42)
                block.mlp.c_fc.weight.data = w_fc
                
                # c_proj
                n_cp, n_sp, s_mp = cfg['c_proj']
                w_proj, _, s_proj, mse_proj, cos_proj = quantize_flexible(block.mlp.c_proj.weight.data, num_ctrl=n_cp, num_spec=n_sp, spec_mode=s_mp, p_crit=0.05, seed=42)
                block.mlp.c_proj.weight.data = w_proj
                
                if is_m2:
                    layer_audit_m2.append({
                        'layer': f"h{l_idx}.c_fc",
                        'mse': mse_fc,
                        'cos': cos_fc,
                        'spec_usage': s_fc['spec_usage'],
                        'ctrl_usage': s_fc['ctrl_usage'],
                        'base_usage': s_fc['atom_usage'].sum(),
                        'total_blocks': s_fc['atom_usage'].sum() + s_fc['ctrl_usage'].sum() + s_fc['spec_usage'].sum()
                    })
                    layer_audit_m2.append({
                        'layer': f"h{l_idx}.c_proj",
                        'mse': mse_proj,
                        'cos': cos_proj,
                        'spec_usage': s_proj['spec_usage'],
                        'ctrl_usage': s_proj['ctrl_usage'],
                        'base_usage': s_proj['atom_usage'].sum(),
                        'total_blocks': s_proj['atom_usage'].sum() + s_proj['ctrl_usage'].sum() + s_proj['spec_usage'].sum()
                    })
                    
        elapsed = time.time() - t0
        
        ppls = {d: evaluate_perplexity(m_test, tokenizer, text) for d, text in domains.items()}
        avg_ppl = np.mean(list(ppls.values()))
        
        # Testar conjunto de 4 prompts
        gen_results = []
        div_list = []
        has_loop = False
        
        for p_id, p_text in prompts:
            g_text, div_r = evaluate_generation(m_test, tokenizer, p_text)
            gen_results.append((p_id, g_text))
            div_list.append(div_r)
            if div_r < 0.45 or ("complex and complex" in g_text) or ("world of artificial" in g_text):
                has_loop = True
                
        avg_div = np.mean(div_list)
        
        results[code] = {
            'name': name,
            'avg_ppl': avg_ppl,
            'ppls': ppls,
            'avg_div': avg_div,
            'has_loop': has_loop,
            'gen_results': gen_results,
            'time': elapsed
        }
        
        status_str = "LOOP DETECTADO" if has_loop else "100% Livre de Loops"
        print(f"   -> PPL Medio: {avg_ppl:.2f} (Ciencia: {ppls['Ciencia']:.1f} | Codigo: {ppls['Codigo']:.1f} | Hist: {ppls['Historia']:.1f} | Encic: {ppls['Enciclopedia']:.1f})")
        print(f"   -> Diversidade Media: {avg_div*100:.1f}% | Status: {status_str}")
        for p_id, g_text in gen_results:
            clean_g = g_text.replace('\n', ' ')
            print(f"      {p_id}: \"{clean_g[:65]}...\"")

    # 1. SCOREBOARD COMPARATIVO DE ABLAÇÃO
    print("\n" + "=" * 135)
    print("SCOREBOARD COMPARATIVO DA ABLAÇÃO DO M.2 A 3.05 BPP (12 CAMADAS MLP)")
    print("=" * 135)
    print(f"{'Teste':<28} | {'PPL Medio':<10} | {'Ciencia':<8} | {'Codigo':<8} | {'Historia':<8} | {'Encic':<8} | {'Diversidade':<12} | {'Loops (4 prompts)'}")
    print("-" * 135)
    print(f"{'FP16 Original (16.0 bpp)':<28} | {avg_orig:>9.2f}  | {ppl_orig['Ciencia']:>8.1f} | {ppl_orig['Codigo']:>8.1f} | {ppl_orig['Historia']:>8.1f} | {ppl_orig['Enciclopedia']:>8.1f} | {np.mean([d for _, d in [evaluate_generation(base_model, tokenizer, p[1]) for p in prompts]])*100:>10.1f}% | {'0/4 (Perfeito)'}")
    print("-" * 135)
    
    for code, name, _ in ablation_tests:
        r = results[code]
        loop_str = "LOOP (Colapso)" if r['has_loop'] else "0/4 (100% Livre)"
        print(f"{name[:28]:<28} | {r['avg_ppl']:>9.2f}  | {r['ppls']['Ciencia']:>8.1f} | {r['ppls']['Codigo']:>8.1f} | {r['ppls']['Historia']:>8.1f} | {r['ppls']['Enciclopedia']:>8.1f} | {r['avg_div']*100:>10.1f}% | {loop_str}")
    print("=" * 135)

    # 2. AUDITORIA FINA POR CAMADA DO M.2
    print("\n" + "=" * 135)
    print("AUDITORIA FINA POR CAMADA DO M.2: ERRO E DISTRIBUIÇÃO DOS 32 ESTADOS")
    print("=" * 135)
    print(f"{'Camada':<12} | {'MSE':<10} | {'Cosine':<8} | {'Base (26)':<10} | {'Controles':<12} | {'Spec 1':<8} | {'Spec 2':<8} | {'Spec 3':<8} | {'Spec 4':<8} | {'Total Spec'}")
    print("-" * 135)
    
    for row in layer_audit_m2:
        tot = row['total_blocks']
        base_pct = (row['base_usage'] / tot) * 100
        ctrl_pct = (row['ctrl_usage'].sum() / tot) * 100
        s_pcts = [(s / tot) * 100 for s in row['spec_usage']]
        tot_spec_pct = sum(s_pcts)
        
        s1 = f"{s_pcts[0]:.1f}%" if len(s_pcts) > 0 else "-"
        s2 = f"{s_pcts[1]:.1f}%" if len(s_pcts) > 1 else "-"
        s3 = f"{s_pcts[2]:.1f}%" if len(s_pcts) > 2 else "-"
        s4 = f"{s_pcts[3]:.1f}%" if len(s_pcts) > 3 else "-"
        
        print(f"{row['layer']:<12} | {row['mse']:>10.6f} | {row['cos']*100:>7.2f}% | {base_pct:>8.1f}% | {ctrl_pct:>10.1f}% | {s1:>8} | {s2:>8} | {s3:>8} | {s4:>8} | {tot_spec_pct:>9.1f}%")
    print("=" * 135)

if __name__ == '__main__':
    main()
