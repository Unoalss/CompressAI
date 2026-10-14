import os
import time
import torch
from safetensors import safe_open

MODEL_DIR = "E:/agente/modelo"
SAFETENSORS_PATH = os.path.join(MODEL_DIR, "model.safetensors-00001-of-00001.safetensors")

if __name__ == "__main__":
    print("=" * 80)
    print("TESTE DE QUANTIZAÇÃO H5-A EM MATRIZ FFN REAL DO QWEN3.5-2B")
    print("=" * 80)

    # Carregar uma matriz do layer 12: gate_proj, up_proj, down_proj
    with safe_open(SAFETENSORS_PATH, framework="pt", device="cpu") as f:
        w_gate = f.get_tensor("model.language_model.layers.12.mlp.gate_proj.weight")
        w_up = f.get_tensor("model.language_model.layers.12.mlp.up_proj.weight")
        w_down = f.get_tensor("model.language_model.layers.12.mlp.down_proj.weight")

    print(f"w_gate shape: {w_gate.shape}, dtype: {w_gate.dtype}, elements: {w_gate.numel():,d}")
    print(f"w_up   shape: {w_up.shape}, dtype: {w_up.dtype}, elements: {w_up.numel():,d}")
    print(f"w_down shape: {w_down.shape}, dtype: {w_down.dtype}, elements: {w_down.numel():,d}")

# Implementação otimizada do H5-A em batches para tensores grandes (1.57M blocos)
def train_kmeans(vectors, K=26, n_iter=6, seed=42):
    torch.manual_seed(seed)
    N, D = vectors.shape
    if N <= K:
        if N == 0: return torch.zeros(K, D)
        rep = (K // N) + 1
        return vectors.repeat(rep, 1)[:K]
    
    # K-means++ rápido
    centers = [vectors[torch.randint(0, N, (1,))[0]]]
    for _ in range(1, K):
        # Amostra para acelerar
        sub_idx = torch.randint(0, N, (min(N, 2000),))
        sub_vecs = vectors[sub_idx]
        cur_centers = torch.stack(centers)
        dists = torch.cdist(sub_vecs, cur_centers)
        min_dists, _ = torch.min(dists, dim=1)
        probs = min_dists ** 2
        probs_sum = probs.sum()
        if probs_sum > 0:
            probs = probs / probs_sum
            next_idx = sub_idx[torch.multinomial(probs, 1)[0]]
            centers.append(vectors[next_idx])
        else:
            centers.append(vectors[torch.randint(0, N, (1,))[0]])
            
    c_core = torch.stack(centers)
    sub_sample = vectors[:min(N, 20000)]
    for _ in range(n_iter):
        d = torch.cdist(sub_sample, c_core)
        lbl = torch.argmin(d, dim=1)
        for k in range(K):
            m = (lbl == k)
            if m.sum() > 0:
                c_core[k] = sub_sample[m].mean(dim=0)
    return c_core

def quantize_matrix_h5a_batched(weight_tensor, num_ctrl=2, num_spec=4, spec_mode='sensitivity', p_crit=0.05, seed=42, block_size=8, batch_size=200000):
    t0 = time.time()
    torch.manual_seed(seed)
    orig_shape = weight_tensor.shape
    w_flat = weight_tensor.float().reshape(-1, block_size)
    N = w_flat.shape[0]
    
    # Amostra de calibração para treinar os dicionários
    sample_n = min(N, 50000)
    perm_sample = torch.randperm(N)[:sample_n]
    w_sample = w_flat[perm_sample]
    
    # 1. 8 Famílias Macro
    perm_fam = torch.randperm(sample_n)[:8]
    fam_centers = w_sample[perm_fam].clone()
    for _ in range(5):
        d = torch.cdist(w_sample[:10000], fam_centers)
        lbl = torch.argmin(d, dim=1)
        for k in range(8):
            m = (lbl == k)
            if m.sum() > 0:
                fam_centers[k] = w_sample[:10000][m].mean(dim=0)
                
    # Atribuição das famílias em batches
    fam_ids = torch.empty(N, dtype=torch.long)
    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        fam_ids[i:end_i] = torch.argmin(torch.cdist(w_flat[i:end_i], fam_centers), dim=1)
        
    sample_fam_ids = fam_ids[perm_sample]
    libs_macro = [train_kmeans(w_sample[sample_fam_ids == k], K=26, seed=seed) for k in range(8)]
    
    libs_spec = []
    for k in range(8):
        m_k = (sample_fam_ids == k)
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
        else:
            top_k = max(num_spec, int(vecs_k.shape[0] * 0.35))
            hard_idx = torch.topk(min_err, k=top_k).indices
            spec_k = train_kmeans(vecs_k[hard_idx], K=num_spec, seed=seed)
            
        libs_spec.append(spec_k)
        
    # Reconstrução Macro por Batches
    macro_recon = torch.empty_like(w_flat)
    
    for k in range(8):
        m_full = (fam_ids == k)
        count_k = m_full.sum().item()
        if count_k == 0: continue
        
        w_k = w_flat[m_full]
        lib_base = libs_macro[k]
        lib_spec = libs_spec[k]
        
        # Processar w_k em sub-batches se for muito grande
        recon_k = torch.empty_like(w_k)
        sub_b = 50000
        for sb_start in range(0, count_k, sub_b):
            sb_end = min(sb_start + sub_b, count_k)
            w_sub = w_k[sb_start:sb_end]
            
            d_base = torch.cdist(w_sub, lib_base)
            min_base, idx_base = torch.min(d_base, dim=1)
            
            if num_ctrl == 3:
                d_c0 = torch.cdist(w_sub, lib_base * 1.5)
                d_c1 = torch.cdist(w_sub, lib_base * 0.5)
                d_c2 = torch.cdist(w_sub, -lib_base)
                min_c0, idx_c0 = torch.min(d_c0, dim=1)
                min_c1, idx_c1 = torch.min(d_c1, dim=1)
                min_c2, idx_c2 = torch.min(d_c2, dim=1)
                best_ctrl_dists = torch.stack([min_c0, min_c1, min_c2], dim=1)
                ctrl_indices = [idx_c0, idx_c1, idx_c2]
            else:
                d_c0 = torch.cdist(w_sub, lib_base * 1.5)
                d_c1 = torch.cdist(w_sub, -lib_base)
                min_c0, idx_c0 = torch.min(d_c0, dim=1)
                min_c1, idx_c1 = torch.min(d_c1, dim=1)
                best_ctrl_dists = torch.stack([min_c0, min_c1], dim=1)
                ctrl_indices = [idx_c0, idx_c1]
                
            min_ctrl_val, choice_ctrl = torch.min(best_ctrl_dists, dim=1)
            
            d_spec = torch.cdist(w_sub, lib_spec)
            min_spec_val, idx_spec = torch.min(d_spec, dim=1)
            
            sub_recon = lib_base[idx_base].clone()
            win_spec = (min_spec_val < min_base) & (min_spec_val < min_ctrl_val)
            win_ctrl = (min_ctrl_val < min_base) & (min_ctrl_val <= min_spec_val)
            
            sub_recon[win_spec] = lib_spec[idx_spec[win_spec]]
            
            if num_ctrl == 3:
                mask_c0 = win_ctrl & (choice_ctrl == 0)
                mask_c1 = win_ctrl & (choice_ctrl == 1)
                mask_c2 = win_ctrl & (choice_ctrl == 2)
                sub_recon[mask_c0] = lib_base[ctrl_indices[0][mask_c0]] * 1.5
                sub_recon[mask_c1] = lib_base[ctrl_indices[1][mask_c1]] * 0.5
                sub_recon[mask_c2] = -lib_base[ctrl_indices[2][mask_c2]]
            else:
                mask_c0 = win_ctrl & (choice_ctrl == 0)
                mask_c1 = win_ctrl & (choice_ctrl == 1)
                sub_recon[mask_c0] = lib_base[ctrl_indices[0][mask_c0]] * 1.5
                sub_recon[mask_c1] = -lib_base[ctrl_indices[1][mask_c1]]
                
            recon_k[sb_start:sb_end] = sub_recon
            
        macro_recon[m_full] = recon_k
        
    # 2. Resíduo 1 (Base 2.0 bpp)
    r1 = w_flat - macro_recon
    r1_sample = r1[perm_sample]
    perm_r1 = torch.randperm(sample_n)[:8]
    r1_centers = r1_sample[perm_r1].clone()
    for _ in range(5):
        d = torch.cdist(r1_sample[:10000], r1_centers)
        lbl = torch.argmin(d, dim=1)
        for k in range(8):
            m = (lbl == k)
            if m.sum() > 0: r1_centers[k] = r1_sample[:10000][m].mean(dim=0)
            
    r1_fam_ids = torch.empty(N, dtype=torch.long)
    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        r1_fam_ids[i:end_i] = torch.argmin(torch.cdist(r1[i:end_i], r1_centers), dim=1)
        
    sample_r1_fam_ids = r1_fam_ids[perm_sample]
    libs_r1 = [train_kmeans(r1_sample[sample_r1_fam_ids == k], K=26, seed=seed) for k in range(8)]
    
    recon_base = macro_recon.clone()
    for k in range(8):
        m_k = (r1_fam_ids == k)
        count_rk = m_k.sum().item()
        if count_rk == 0: continue
        r1_k = r1[m_k]
        
        recon_rk = torch.empty_like(r1_k)
        for sb_start in range(0, count_rk, 50000):
            sb_end = min(sb_start + 50000, count_rk)
            d_rk = torch.cdist(r1_k[sb_start:sb_end], libs_r1[k])
            recon_rk[sb_start:sb_end] = libs_r1[k][torch.argmin(d_rk, dim=1)]
        recon_base[m_k] += recon_rk
        
    # 3. Resíduo 2 (Base 3.0 bpp)
    res_base = w_flat - recon_base
    lib_r2 = train_kmeans(res_base[perm_sample], K=26, seed=seed)
    
    recon_final = recon_base.clone()
    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        d_r2 = torch.cdist(res_base[i:end_i], lib_r2)
        recon_final[i:end_i] += lib_r2[torch.argmin(d_r2, dim=1)]
        
    # 4. Proteção Seletiva Top P_crit%
    if p_crit > 0:
        # Calcular sensibilidade por batches para poupar memória
        sens_list = []
        for i in range(0, N, batch_size):
            end_i = min(i + batch_size, N)
            sub_w = w_flat[i:end_i]
            sub_res = sub_w - recon_final[i:end_i]
            block_norm = torch.norm(sub_w, dim=1)
            sens = (block_norm + 1e-5) * torch.mean(sub_res ** 2, dim=1)
            sens_list.append(sens)
        sensitivity = torch.cat(sens_list)
        
        num_crit = int(p_crit * N)
        crit_idx = torch.topk(sensitivity, k=num_crit).indices
        
        r3_vecs = (w_flat[crit_idx] - recon_final[crit_idx])
        lib_r3 = train_kmeans(r3_vecs, K=26, seed=seed)
        
        for sb_start in range(0, num_crit, 50000):
            sb_end = min(sb_start + 50000, num_crit)
            sub_r3 = r3_vecs[sb_start:sb_end]
            d_r3 = torch.cdist(sub_r3, lib_r3)
            recon_final[crit_idx[sb_start:sb_end]] += lib_r3[torch.argmin(d_r3, dim=1)]
            
    actual_bpp = 3.0 + p_crit
    recon_reshaped = recon_final.view(orig_shape).to(weight_tensor.dtype)
    
    elapsed = time.time() - t0
    mse = torch.mean((weight_tensor.float() - recon_final.view(orig_shape)) ** 2).item()
    cos = torch.nn.functional.cosine_similarity(weight_tensor.float().view(1, -1), recon_final.view(1, -1)).item()
    
    return recon_reshaped, actual_bpp, mse, cos, elapsed

if __name__ == "__main__":
    print("\n--- Testando Quantização em gate_proj (Expansão 2+4-D, p=5%) ---")
    rec_gate, bpp_g, mse_g, cos_g, t_g = quantize_matrix_h5a_batched(w_gate, num_ctrl=2, num_spec=4, spec_mode='sensitivity', p_crit=0.05)
    print(f"gate_proj: BPP={bpp_g:.2f} | MSE={mse_g:.6f} | Cosine={cos_g:.6f} | Tempo={t_g:.1f}s")

    print("\n--- Testando Quantização em up_proj (Expansão 2+4-D, p=5%) ---")
    rec_up, bpp_u, mse_u, cos_u, t_u = quantize_matrix_h5a_batched(w_up, num_ctrl=2, num_spec=4, spec_mode='sensitivity', p_crit=0.05)
    print(f"up_proj:   BPP={bpp_u:.2f} | MSE={mse_u:.6f} | Cosine={cos_u:.6f} | Tempo={t_u:.1f}s")

    print("\n--- Testando Quantização em down_proj (Redução 3+3, p=5%) ---")
    rec_down, bpp_d, mse_d, cos_d, t_d = quantize_matrix_h5a_batched(w_down, num_ctrl=3, num_spec=3, spec_mode='sensitivity', p_crit=0.05)
    print(f"down_proj: BPP={bpp_d:.2f} | MSE={mse_d:.6f} | Cosine={cos_d:.6f} | Tempo={t_d:.1f}s")

    print("\n" + "=" * 80)
    print("TESTE CONCLUÍDO COM SUCESSO!")
    print("=" * 80)
