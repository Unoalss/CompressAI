import torch
from transformers import AutoModelForCausalLM
import numpy as np
import time

def train_kmeans_1d(values, K=16, n_iter=10):
    """Treina K valores de escala unidimensionais."""
    perm = torch.randperm(values.shape[0])[:K]
    centroids = values[perm].clone()
    for _ in range(n_iter):
        dists = torch.abs(values.unsqueeze(1) - centroids.unsqueeze(0))
        labels = torch.argmin(dists, dim=1)
        for k in range(K):
            mask = (labels == k)
            if mask.sum() > 0:
                centroids[k] = values[mask].mean()
    return torch.sort(centroids)[0]

def train_codebook_vectors(vectors, K=26, n_iter=8):
    """Treina K centroides multidimensionais."""
    N, D = vectors.shape
    perm = torch.randperm(N)[:K]
    centroids = vectors[perm].clone()
    for _ in range(n_iter):
        dists = torch.cdist(vectors, centroids)
        labels = torch.argmin(dists, dim=1)
        for k in range(K):
            mask = (labels == k)
            if mask.sum() > 0:
                centroids[k] = vectors[mask].mean(dim=0)
    return centroids

def test_calibrated_shape_scale_rdo():
    print("=" * 80)
    print("RDO CALIBRADO: FORMA (26) + ESCALA (16) + RESÍDUO (26) vs REF")
    print("Testando em Pesos Reais do DistilGPT-2")
    print("=" * 80)
    
    model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    weight = model.transformer.h[0].mlp.c_fc.weight.data.float() # (768, 3072)
    block_size = 8
    weights_flat = weight.reshape(-1, block_size)
    
    sample_size = 30000
    w_sample = weights_flat[:sample_size].clone()
    
    # 1. Normas e Escalas Reais
    norms = torch.norm(w_sample, dim=1)
    lib_scales = train_kmeans_1d(norms, K=16) # 16 escalas reais (4 bits)
    print(f"Escalas Calibradas (min/méd/max): {lib_scales.min():.3f} / {lib_scales.mean():.3f} / {lib_scales.max():.3f}")
    
    # 2. Formas Unitárias (26 átomos)
    w_unit = w_sample / (norms.unsqueeze(1) + 1e-7)
    lib_shape = train_codebook_vectors(w_unit, K=26, n_iter=8)
    lib_shape = lib_shape / (torch.norm(lib_shape, dim=1, keepdim=True) + 1e-7)
    
    # 3. Resíduos Calibrados após Forma * Escala
    # Achar melhor combinação base
    d_shape = torch.cdist(w_unit, lib_shape)
    best_shape_idx = torch.argmin(d_shape, dim=1)
    d_scale = torch.abs(norms.unsqueeze(1) - lib_scales.unsqueeze(0))
    best_scale_idx = torch.argmin(d_scale, dim=1)
    
    base_recon = lib_shape[best_shape_idx] * lib_scales[best_scale_idx].unsqueeze(1)
    residuals_real = w_sample - base_recon
    
    lib_residual = train_codebook_vectors(residuals_real, K=26, n_iter=8)
    
    print("\nExecutando Rate-Distortion Optimization (RDO) em 30.000 blocos...")
    
    # Tolerância equilibrada para manter similaridade > 90%
    TARGET_MSE = 0.0040
    
    mode_counts = {
        'REF (10b)': 0,
        'FORMA (5b)': 0,
        'FORMA + ESCALA (15b)': 0,
        'FORMA + RESÍDUO (15b)': 0,
        'FORMA + ESCALA + RESÍDUO (25b)': 0
    }
    
    reconstructed_blocks = []
    total_tokens = 0
    cache = []
    cache_size = 26
    
    t0 = time.time()
    for i in range(sample_size):
        b = w_sample[i]
        b_norm = norms[i]
        b_unit = w_unit[i]
        
        best_recon = None
        best_bits = 999
        best_mode = None
        
        # Modo 1: REF (Cache - 10 bits)
        if len(cache) > 0:
            c_mat = torch.stack(cache)
            d_c = torch.mean((c_mat - b) ** 2, dim=1)
            min_c, c_idx = torch.min(d_c, dim=0)
            if min_c.item() <= TARGET_MSE:
                best_recon = cache[c_idx.item()]
                best_bits = 10
                best_mode = 'REF (10b)'
                
        # Modo 2: FORMA pura (5 bits)
        if best_bits > 5:
            d_s = torch.sum((lib_shape - b) ** 2, dim=1)
            s_idx = torch.argmin(d_s).item()
            cand_shape = lib_shape[s_idx]
            err_shape = torch.mean((cand_shape - b) ** 2).item()
            if err_shape <= TARGET_MSE:
                best_recon = cand_shape
                best_bits = 5
                best_mode = 'FORMA (5b)'
                
        # Modo 3: FORMA + ESCALA (15 bits)
        if best_bits > 15:
            d_s_unit = torch.sum((lib_shape - b_unit) ** 2, dim=1)
            s_best = torch.argmin(d_s_unit).item()
            # Encontrar melhor escala
            cand_s_vec = lib_shape[s_best]
            scale_diff = torch.abs(lib_scales - b_norm)
            scale_best = torch.argmin(scale_diff).item()
            cand_scaled = cand_s_vec * lib_scales[scale_best]
            err_scaled = torch.mean((cand_scaled - b) ** 2).item()
            if err_scaled <= TARGET_MSE:
                best_recon = cand_scaled
                best_bits = 15
                best_mode = 'FORMA + ESCALA (15b)'
                
        # Modo 4: FORMA + ESCALA + RESÍDUO (25 bits)
        if best_bits > 25 or best_recon is None:
            d_s_unit = torch.sum((lib_shape - b_unit) ** 2, dim=1)
            s_best = torch.argmin(d_s_unit).item()
            scale_diff = torch.abs(lib_scales - b_norm)
            scale_best = torch.argmin(scale_diff).item()
            base_vec = lib_shape[s_best] * lib_scales[scale_best]
            
            res_vec = b - base_vec
            d_res = torch.sum((lib_residual - res_vec) ** 2, dim=1)
            r_best = torch.argmin(d_res).item()
            
            cand_full = base_vec + lib_residual[r_best]
            best_recon = cand_full
            best_bits = 25
            best_mode = 'FORMA + ESCALA + RESÍDUO (25b)'
            
        reconstructed_blocks.append(best_recon)
        mode_counts[best_mode] += 1
        total_tokens += (best_bits // 5)
        
        cache.append(best_recon)
        if len(cache) > cache_size: cache.pop(0)
        
    recon_mat = torch.stack(reconstructed_blocks)
    final_mse = torch.mean((w_sample - recon_mat) ** 2).item()
    cos_sim = torch.nn.functional.cosine_similarity(w_sample.view(1, -1), recon_mat.view(1, -1)).item()
    
    total_bits = total_tokens * 5
    bpp = total_bits / (sample_size * block_size)
    orig_kb = (sample_size * block_size * 2) / 1024
    dict_kb = (26 * 8 * 2 + 16 * 2 + 26 * 8 * 2) / 1024 # ~0.86 KB
    stream_kb = (total_bits / 8.0) / 1024
    total_compressed_kb = dict_kb + stream_kb
    reduction = 100 * (1 - total_compressed_kb / orig_kb)
    
    print("\n" + "=" * 80)
    print("DISTRIBUIÇÃO REAL DAS INSTRUÇÕES ESCOLHIDAS (RDO)")
    print("=" * 80)
    for mode, count in mode_counts.items():
        pct = (count / sample_size) * 100
        print(f"-> {mode:<32}: {count:>6,} blocos ({pct:>5.1f}%)")
        
    print("\n" + "=" * 80)
    print("MÉTRICAS FINAIS DO SISTEMA ADAPTATIVO")
    print("=" * 80)
    print(f"Média de Bits por Peso:          {bpp:.2f} bits/peso (Variável!)")
    print(f"Tamanho Original (FP16):          {orig_kb:.1f} KB")
    print(f"Dicionário (Formas+Escalas+Res):  {dict_kb:.2f} KB (menos de 1 KB!)")
    print(f"Total Comprimido:                 {total_compressed_kb:.1f} KB")
    print(f"Taxa Real de Redução:             {reduction:.1f}%")
    print(f"Erro Quadrático Médio (MSE):      {final_mse:.6f}")
    print(f"Similaridade de Cosseno:          {cos_sim * 100:.2f}%")
    print("=" * 80)

if __name__ == '__main__':
    test_calibrated_shape_scale_rdo()
