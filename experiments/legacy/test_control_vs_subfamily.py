import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import time
import copy

def train_kmeans(vectors, K=26, n_iter=6, seed=42):
    """K-means++ rápido com controle de seed."""
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

def quantize_layer_variant(weight_tensor, mode='A', p_crit=0.05, seed=42, block_size=8):
    """
    mode:
      'A': 26 átomos + 6 controles VM (REF, SCALE 1.5, SCALE 0.5, NEGATE, ZERO, INTERP)
      'B': 26 átomos + 6 subfamílias secundárias (8x6 = 48 contextos especializados)
      'C': 26 átomos + 3 controles (REF, SCALE, ZERO) + 3 subfamílias secundárias (24 contextos)
    """
    torch.manual_seed(seed)
    orig_shape = weight_tensor.shape
    w_flat = weight_tensor.float().reshape(-1, block_size)
    N = w_flat.shape[0]
    sample_n = min(N, 35000)
    w_sample = w_flat[:sample_n]
    
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
    
    macro_recon = torch.zeros_like(w_flat)
    stats = {'special_state_usage': 0, 'base_atom_usage': 0}
    
    if mode == 'A':
        # Modelo A: 26 átomos + 6 controles
        # Controles: 26=REF, 27=SCALE*1.5, 28=SCALE*0.5, 29=NEG, 30=ZERO, 31=INTERP
        libs_macro = [train_kmeans(w_sample[fam_ids[:sample_n] == k], K=26, seed=seed) for k in range(8)]
        
        for k in range(8):
            m = (fam_ids == k)
            if m.sum() == 0: continue
            w_k = w_flat[m]
            lib = libs_macro[k]
            
            # Opções: 26 átomos base
            d_base = torch.cdist(w_k, lib)
            min_base, idx_base = torch.min(d_base, dim=1)
            
            # Opção de Escala 1.5 e 0.5
            d_s1 = torch.cdist(w_k, lib * 1.5)
            min_s1, idx_s1 = torch.min(d_s1, dim=1)
            
            d_s2 = torch.cdist(w_k, lib * 0.5)
            min_s2, idx_s2 = torch.min(d_s2, dim=1)
            
            # Opção Negate
            d_neg = torch.cdist(w_k, -lib)
            min_neg, idx_neg = torch.min(d_neg, dim=1)
            
            # Opção Zero
            norm_zero = torch.norm(w_k, dim=1)
            
            # Decisão
            recon_k = lib[idx_base].clone()
            
            # Checa se algum controle ganha do átomo base
            mask_s1 = (min_s1 < min_base) & (min_s1 < min_s2) & (min_s1 < min_neg) & (min_s1 < norm_zero)
            recon_k[mask_s1] = lib[idx_s1[mask_s1]] * 1.5
            
            mask_s2 = (min_s2 < min_base) & (min_s2 < min_s1) & (min_s2 < min_neg) & (min_s2 < norm_zero)
            recon_k[mask_s2] = lib[idx_s2[mask_s2]] * 0.5
            
            mask_neg = (min_neg < min_base) & (min_neg < min_s1) & (min_neg < min_s2) & (min_neg < norm_zero)
            recon_k[mask_neg] = -lib[idx_neg[mask_neg]]
            
            mask_zero = (norm_zero < min_base) & (norm_zero < min_s1) & (norm_zero < min_s2) & (norm_zero < min_neg)
            recon_k[mask_zero] = 0.0
            
            num_spec = (mask_s1 | mask_s2 | mask_neg | mask_zero).sum().item()
            stats['special_state_usage'] += num_spec
            stats['base_atom_usage'] += (m.sum().item() - num_spec)
            
            macro_recon[m] = recon_k

    elif mode == 'B':
        # Modelo B: 26 átomos + 6 subfamílias secundárias (8 x 6 = 48 contextos)
        # Cada família k treina 26 átomos base + 6 subfamílias
        libs_macro = [train_kmeans(w_sample[fam_ids[:sample_n] == k], K=26, seed=seed) for k in range(8)]
        
        # Centróides das 6 subfamílias por família
        subfam_centers = []
        for k in range(8):
            m_k = (fam_ids[:sample_n] == k)
            vecs_k = w_sample[m_k]
            if vecs_k.shape[0] >= 6:
                sf = train_kmeans(vecs_k, K=6, seed=seed)
            else:
                sf = fam_centers[k].unsqueeze(0).repeat(6, 1)
            subfam_centers.append(sf)
            
        for k in range(8):
            m = (fam_ids == k)
            if m.sum() == 0: continue
            w_k = w_flat[m]
            lib_base = libs_macro[k]
            lib_sub = subfam_centers[k]
            
            d_base = torch.cdist(w_k, lib_base)
            min_base, idx_base = torch.min(d_base, dim=1)
            
            d_sub = torch.cdist(w_k, lib_sub)
            min_sub, idx_sub = torch.min(d_sub, dim=1)
            
            recon_k = lib_base[idx_base].clone()
            mask_sub = (min_sub < min_base)
            recon_k[mask_sub] = lib_sub[idx_sub[mask_sub]]
            
            num_spec = mask_sub.sum().item()
            stats['special_state_usage'] += num_spec
            stats['base_atom_usage'] += (m.sum().item() - num_spec)
            
            macro_recon[m] = recon_k

    elif mode == 'C':
        # Modelo C (Híbrido): 26 átomos + 3 controles (SCALE, NEG, ZERO) + 3 subfamílias
        libs_macro = [train_kmeans(w_sample[fam_ids[:sample_n] == k], K=26, seed=seed) for k in range(8)]
        subfam_centers = []
        for k in range(8):
            m_k = (fam_ids[:sample_n] == k)
            vecs_k = w_sample[m_k]
            if vecs_k.shape[0] >= 3:
                sf = train_kmeans(vecs_k, K=3, seed=seed)
            else:
                sf = fam_centers[k].unsqueeze(0).repeat(3, 1)
            subfam_centers.append(sf)
            
        for k in range(8):
            m = (fam_ids == k)
            if m.sum() == 0: continue
            w_k = w_flat[m]
            lib_base = libs_macro[k]
            lib_sub = subfam_centers[k]
            
            d_base = torch.cdist(w_k, lib_base)
            min_base, idx_base = torch.min(d_base, dim=1)
            
            # Controles
            d_s1 = torch.cdist(w_k, lib_base * 1.5)
            min_s1, idx_s1 = torch.min(d_s1, dim=1)
            
            d_neg = torch.cdist(w_k, -lib_base)
            min_neg, idx_neg = torch.min(d_neg, dim=1)
            
            # Subfamílias
            d_sub = torch.cdist(w_k, lib_sub)
            min_sub, idx_sub = torch.min(d_sub, dim=1)
            
            recon_k = lib_base[idx_base].clone()
            
            # Subfamília ganha
            mask_sub = (min_sub < min_base) & (min_sub < min_s1) & (min_sub < min_neg)
            recon_k[mask_sub] = lib_sub[idx_sub[mask_sub]]
            
            # Controles ganham
            mask_s1 = (min_s1 < min_base) & (min_s1 < min_sub) & (min_s1 < min_neg)
            recon_k[mask_s1] = lib_base[idx_s1[mask_s1]] * 1.5
            
            mask_neg = (min_neg < min_base) & (min_neg < min_sub) & (min_neg < min_s1)
            recon_k[mask_neg] = -lib_base[idx_neg[mask_neg]]
            
            num_spec = (mask_sub | mask_s1 | mask_neg).sum().item()
            stats['special_state_usage'] += num_spec
            stats['base_atom_usage'] += (m.sum().item() - num_spec)
            
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

def main():
    print("=" * 95)
    print("EXPERIMENTO COMPARATIVO: CONTROLES vs SUBFAMÍLIAS SECUNDÁRIAS")
    print("Avaliação nos 6 estados especiais dos 5 bits [3+5] (3.05 bpp)")
    print("=" * 95)
    
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
    
    # Referência Original
    ppl_orig = {d: evaluate_perplexity(base_model, tokenizer, text) for d, text in domains.items()}
    avg_orig = np.mean(list(ppl_orig.values()))
    print(f"\n[0] MODELO ORIGINAL FP16 (16.0 bpp): Média PPL = {avg_orig:.2f}")
    
    # 1. TESTE ISOLADO NA CAMADA h[0].mlp.c_fc (Métricas Geométricas: MSE e Cosine)
    print("\n" + "-" * 95)
    print("FASE 1: COMPARAÇÃO GEOMÉTRICA NA CAMADA REAL (h[0].mlp.c_fc - 50.000 blocos = 400k pesos)")
    print("-" * 95)
    layer_orig = base_model.transformer.h[0].mlp.c_fc.weight.data.clone()
    
    modes = [
        ('A', 'Modelo A (26 átomos + 6 controles VM)'),
        ('B', 'Modelo B (26 átomos + 6 subfamílias - 48 contextos)'),
        ('C', 'Modelo C (Híbrido: 3 controles + 3 subfamílias)')
    ]
    
    layer_metrics = {}
    for m_code, m_name in modes:
        t0 = time.time()
        w_recon, bpp, stats = quantize_layer_variant(layer_orig, mode=m_code, p_crit=0.05, seed=42)
        elapsed = time.time() - t0
        
        mse = torch.mean((layer_orig - w_recon) ** 2).item()
        cos = torch.nn.functional.cosine_similarity(layer_orig.view(1, -1), w_recon.view(1, -1)).item()
        spec_ratio = (stats['special_state_usage'] / (stats['base_atom_usage'] + stats['special_state_usage'])) * 100
        
        layer_metrics[m_code] = {
            'mse': mse,
            'cos': cos,
            'spec_ratio': spec_ratio,
            'time': elapsed,
            'bpp': bpp
        }
        print(f"[{m_code}] {m_name}:")
        print(f"    - MSE da Camada: {mse:.6f}")
        print(f"    - Cosine Similarity: {cos*100:.2f}%")
        print(f"    - Taxa de Ativação dos 6 Estados Especiais: {spec_ratio:.1f}%")
        print(f"    - Tempo de Processamento: {elapsed*1000:.1f} ms")
        
    # 2. TESTE NO MODELO INTEIRO (12 CAMADAS MLP QUANTIZADAS)
    print("\n" + "-" * 95)
    print("FASE 2: COMPARAÇÃO COGNITIVA NO DISTILGPT-2 INTEIRO (Todas as 12 Camadas MLP)")
    print("-" * 95)
    
    full_results = {}
    for m_code, m_name in modes:
        print(f"\n-> Quantizando modelo completo com {m_name}...")
        m_test = copy.deepcopy(base_model)
        t0 = time.time()
        for block in m_test.transformer.h:
            with torch.no_grad():
                w_fc, _, _ = quantize_layer_variant(block.mlp.c_fc.weight.data, mode=m_code, p_crit=0.05, seed=42)
                block.mlp.c_fc.weight.data = w_fc
                w_proj, _, _ = quantize_layer_variant(block.mlp.c_proj.weight.data, mode=m_code, p_crit=0.05, seed=42)
                block.mlp.c_proj.weight.data = w_proj
                
        enc_time = time.time() - t0
        
        # Perplexidade nos 4 domínios
        ppls = {d: evaluate_perplexity(m_test, tokenizer, text) for d, text in domains.items()}
        avg_ppl = np.mean(list(ppls.values()))
        
        # Teste de Geração de Texto
        prompt = "The future of artificial intelligence is"
        inp = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = m_test.generate(**inp, max_new_tokens=22, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        gen_text = tokenizer.decode(out[0], skip_special_tokens=True)
        
        full_results[m_code] = {
            'avg_ppl': avg_ppl,
            'ppls': ppls,
            'gen_text': gen_text,
            'enc_time': enc_time
        }
        print(f"   Média PPL: {avg_ppl:.2f} (Ciência: {ppls['Ciência']:.1f} | Código: {ppls['Código']:.1f} | Hist: {ppls['História']:.1f} | Encic: {ppls['Enciclopédia']:.1f})")
        print(f"   Geração: \"{gen_text}\"")

    # 3. TABELA COMPARATIVA FINAL
    print("\n" + "=" * 105)
    print("SCOREBOARD COMPARATIVO FINAL: 6 CONTROLES (A) vs 6 FAMÍLIAS (B) vs HÍBRIDO (C)")
    print("=" * 105)
    print(f"{'Modelo':<25} | {'BPP':<6} | {'MSE Camada':<10} | {'Cosine':<8} | {'Uso 6 Estados':<13} | {'PPL Médio':<10} | {'Delta FP16'}")
    print("-" * 105)
    print(f"{'FP16 Original':<25} | {'16.0':<6} | {'0.000000':<10} | {'100.0%':<8} | {'-':<13} | {avg_orig:>9.2f}  | {'0.00 (Ref)'}")
    print("-" * 105)
    
    for m_code, m_name in modes:
        lm = layer_metrics[m_code]
        fr = full_results[m_code]
        delta = fr['avg_ppl'] - avg_orig
        print(f"{m_name[:25]:<25} | {lm['bpp']:>4.2f} b | {lm['mse']:>10.6f} | {lm['cos']*100:>7.2f}% | {lm['spec_ratio']:>11.1f}% | {fr['avg_ppl']:>9.2f}  | {delta:>+10.2f}")
    print("=" * 105)

if __name__ == '__main__':
    main()
