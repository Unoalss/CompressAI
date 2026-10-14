import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import time
import copy

def train_kmeans(vectors, K=26, n_iter=6, seed=42):
    """K-means++ com atoms de outliers e controle de seed."""
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

def quantize_layer_c_refined(weight_tensor, variant='C1_baseline', p_crit=0.05, seed=42, block_size=8):
    """
    Variantes de refinamento do Modelo C:
      - 'C1_baseline': O Modelo C original (Escalas 1.5, Neg, Zero + 3 subfamílias genéricas)
      - 'C2_soft_scales': Escalas suaves (1.2 e 0.8) + Preservação de Norma Global (evita drift de ativação)
      - 'C3_error_subfamilies': As 3 subfamílias são treinadas nos 25% blocos com maior erro residual dos 26 átomos
      - 'C4_champion': Escalas suaves (1.2, 0.8) + Subfamílias focadas em erro + Norm Matching
    """
    torch.manual_seed(seed)
    orig_shape = weight_tensor.shape
    w_flat = weight_tensor.float().reshape(-1, block_size)
    N = w_flat.shape[0]
    sample_n = min(N, 35000)
    w_sample = w_flat[:sample_n]
    
    orig_norm = torch.norm(weight_tensor.float())
    
    # 1. 8 Famílias Macro Base
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
    
    # Treinar as 3 subfamílias dependendo da variante
    subfam_centers = []
    for k in range(8):
        m_k = (fam_ids[:sample_n] == k)
        vecs_k = w_sample[m_k]
        if vecs_k.shape[0] < 3:
            subfam_centers.append(fam_centers[k].unsqueeze(0).repeat(3, 1))
            continue
            
        if variant in ['C3_error_subfamilies', 'C4_champion']:
            # Treina subfamílias nos blocos onde os 26 átomos base tiveram maior erro
            d_base = torch.cdist(vecs_k, libs_macro[k])
            min_err = torch.min(d_base, dim=1).values
            top_k_err = max(3, int(vecs_k.shape[0] * 0.35))
            hard_idx = torch.topk(min_err, k=top_k_err).indices
            sf = train_kmeans(vecs_k[hard_idx], K=3, seed=seed)
        else:
            # Subfamílias genéricas (K-means direto)
            sf = train_kmeans(vecs_k, K=3, seed=seed)
        subfam_centers.append(sf)
        
    macro_recon = torch.zeros_like(w_flat)
    stats = {'special_used': 0, 'base_used': 0}
    
    # Parâmetros de escala por variante
    scale_up = 1.20 if variant in ['C2_soft_scales', 'C4_champion'] else 1.50
    scale_down = 0.80 if variant in ['C2_soft_scales', 'C4_champion'] else 0.50
    
    for k in range(8):
        m = (fam_ids == k)
        if m.sum() == 0: continue
        w_k = w_flat[m]
        lib_base = libs_macro[k]
        lib_sub = subfam_centers[k]
        
        # 1. Distância para 26 átomos base
        d_base = torch.cdist(w_k, lib_base)
        min_base, idx_base = torch.min(d_base, dim=1)
        
        # 2. Controles algébricos
        d_up = torch.cdist(w_k, lib_base * scale_up)
        min_up, idx_up = torch.min(d_up, dim=1)
        
        d_down = torch.cdist(w_k, lib_base * scale_down)
        min_down, idx_down = torch.min(d_down, dim=1)
        
        d_neg = torch.cdist(w_k, -lib_base)
        min_neg, idx_neg = torch.min(d_neg, dim=1)
        
        # 3. Subfamílias (3 estados)
        d_sub = torch.cdist(w_k, lib_sub)
        min_sub, idx_sub = torch.min(d_sub, dim=1)
        
        recon_k = lib_base[idx_base].clone()
        
        # Comparação justa entre todos os candidatos
        # Melhor controle vs melhor subfamília vs átomo base
        best_ctrl_dist, ctrl_choice = torch.min(torch.stack([min_up, min_down, min_neg], dim=1), dim=1)
        
        # Condição 1: Subfamília ganha de tudo
        mask_sub = (min_sub < min_base) & (min_sub < best_ctrl_dist)
        recon_k[mask_sub] = lib_sub[idx_sub[mask_sub]]
        
        # Condição 2: Controle ganha de tudo
        mask_ctrl = (best_ctrl_dist < min_base) & (best_ctrl_dist <= min_sub)
        
        # Aplica o controle vencedor
        mask_c_up = mask_ctrl & (ctrl_choice == 0)
        recon_k[mask_c_up] = lib_base[idx_up[mask_c_up]] * scale_up
        
        mask_c_down = mask_ctrl & (ctrl_choice == 1)
        recon_k[mask_c_down] = lib_base[idx_down[mask_c_down]] * scale_down
        
        mask_c_neg = mask_ctrl & (ctrl_choice == 2)
        recon_k[mask_c_neg] = -lib_base[idx_neg[mask_c_neg]]
        
        num_spec = (mask_sub | mask_ctrl).sum().item()
        stats['special_used'] += num_spec
        stats['base_used'] += (m.sum().item() - num_spec)
        
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
        
    recon_reshaped = recon_final.view(orig_shape)
    
    # Preservação de Norma Global (Norm Matching)
    if variant in ['C2_soft_scales', 'C4_champion']:
        curr_norm = torch.norm(recon_reshaped)
        if curr_norm > 0:
            recon_reshaped = recon_reshaped * (orig_norm / curr_norm)
            
    actual_bpp = 3.0 + p_crit
    return recon_reshaped, actual_bpp, stats

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
    # Medir diversidade de vocabulário (razão de tokens únicos)
    tokens = tokenizer.encode(text)
    unique_ratio = len(set(tokens)) / len(tokens)
    return text, unique_ratio

def main():
    print("=" * 105)
    print("REFINAMENTO DO MODELO C (HÍBRIDO): CALIBRAÇÃO DE ESCALAS, SUBFAMÍLIAS POR ERRO E NORM MATCHING")
    print("Objetivo: Romper a barreira de 75-80 PPL com 0% de Repetição Autoregressiva")
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
    
    ppl_orig = {d: evaluate_perplexity(base_model, tokenizer, text) for d, text in domains.items()}
    avg_orig = np.mean(list(ppl_orig.values()))
    print(f"\n[0] FP16 ORIGINAL (16.0 bpp): Média PPL = {avg_orig:.2f}")
    
    test_prompt = "The future of artificial intelligence is"
    gen_orig, div_orig = evaluate_generation(base_model, tokenizer, test_prompt)
    print(f"    Geração Original: \"{gen_orig}\" (Diversidade: {div_orig*100:.1f}%)")
    
    variants = [
        ('C1_baseline', 'C.1 — Modelo C Anterior (Escala 1.5 + Subfamílias K-means)'),
        ('C2_soft_scales', 'C.2 — Escalas Suaves (1.2/0.8) + Norm Matching'),
        ('C3_error_subfamilies', 'C.3 — Subfamílias Treinadas nos Resíduos de Maior Erro'),
        ('C4_champion', 'C.4 — Campeão Combinado (Suave + Erro-Guided + Norm Matching)')
    ]
    
    results = {}
    
    for v_code, v_name in variants:
        print(f"\n" + "-" * 105)
        print(f"Executando {v_name}...")
        print("-" * 105)
        
        m_test = copy.deepcopy(base_model)
        t0 = time.time()
        
        special_usage_sum = 0
        total_blocks_sum = 0
        
        for block in m_test.transformer.h:
            with torch.no_grad():
                w_fc, _, s_fc = quantize_layer_c_refined(block.mlp.c_fc.weight.data, variant=v_code, p_crit=0.05, seed=42)
                block.mlp.c_fc.weight.data = w_fc
                
                w_proj, _, s_proj = quantize_layer_c_refined(block.mlp.c_proj.weight.data, variant=v_code, p_crit=0.05, seed=42)
                block.mlp.c_proj.weight.data = w_proj
                
                special_usage_sum += (s_fc['special_used'] + s_proj['special_used'])
                total_blocks_sum += (s_fc['base_used'] + s_fc['special_used'] + s_proj['base_used'] + s_proj['special_used'])
                
        enc_time = time.time() - t0
        rate_special = (special_usage_sum / total_blocks_sum) * 100
        
        # Perplexidade nos 4 domínios
        ppls = {d: evaluate_perplexity(m_test, tokenizer, text) for d, text in domains.items()}
        avg_ppl = np.mean(list(ppls.values()))
        
        # Avaliação de Geração
        gen_text, div_ratio = evaluate_generation(m_test, tokenizer, test_prompt)
        
        results[v_code] = {
            'name': v_name,
            'avg_ppl': avg_ppl,
            'ppls': ppls,
            'gen_text': gen_text,
            'div_ratio': div_ratio,
            'rate_special': rate_special,
            'time': enc_time
        }
        
        print(f"   -> Média PPL: {avg_ppl:.2f} (Ciência: {ppls['Ciência']:.1f} | Código: {ppls['Código']:.1f} | Hist: {ppls['História']:.1f} | Encic: {ppls['Enciclopédia']:.1f})")
        print(f"   -> Uso Estados Especiais: {rate_special:.1f}%")
        print(f"   -> Diversidade Léxica: {div_ratio*100:.1f}%")
        print(f"   -> Geração: \"{gen_text}\"")

    print("\n" + "=" * 115)
    print("SCOREBOARD COMPARATIVO: TODAS AS VARIANTES DE REFINAMENTO DO MODELO C (3.05 bpp)")
    print("=" * 115)
    print(f"{'Variante':<38} | {'PPL Médio':<10} | {'Ciência':<8} | {'Código':<8} | {'Diversidade':<12} | {'Status Loop'}")
    print("-" * 115)
    print(f"{'FP16 Original (16.0 bpp)':<38} | {avg_orig:>9.2f}  | {ppl_orig['Ciência']:>8.1f} | {ppl_orig['Código']:>8.1f} | {div_orig*100:>10.1f}% | {'Perfeito (Ref)'}")
    print("-" * 115)
    
    for v_code, v_name in variants:
        res = results[v_code]
        loop_status = "LOOP DETECTADO" if res['div_ratio'] < 0.45 else "100% Livre de Loop"
        print(f"{res['name'][:38]:<38} | {res['avg_ppl']:>9.2f}  | {res['ppls']['Ciência']:>8.1f} | {res['ppls']['Código']:>8.1f} | {res['div_ratio']*100:>10.1f}% | {loop_status}")
    print("=" * 115)

if __name__ == '__main__':
    main()
