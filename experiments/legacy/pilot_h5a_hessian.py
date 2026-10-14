"""
pilot_h5a_hessian.py - Teste de sanidade do H5-A 4.05 com Compensacao Hessiana em matriz piloto.
Verifica:
1. Correspondencia e integridade de data/hess_ffn.pt para todas as 72 matrizes FFN.
2. Reconstrucao de matriz piloto (l0.gate e l0.down) com:
   - BF16 Original
   - H5-A 4.05 Normal (Controle)
   - H5-A 4.05 + Hessiana (Diferentes formulacoes matematicas)
3. Mede Tr(E H E^T) / Tr(W H W^T) para garantir que a versao com Hessiana REDUZ comprovadamente o erro funcional.
"""
import os, sys, time, math
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "modelo")
DATA = os.path.join(ROOT, "data")
HESS_PATH = os.path.join(DATA, "hess_ffn.pt")

print("=" * 80)
print("TESTE DE SANIDADE: H5-A 4.05 + COMPENSACAO HESSIANA")
print("=" * 80)

# 1. Verificar Hessianas
hess_dict = torch.load(HESS_PATH, map_location="cpu")
print(f"Hessianas carregadas de {HESS_PATH}: {len(hess_dict)} tensores.")

# Verificar cobertura das 72 FFNs
ffn_names = []
for i in range(24):
    for kind in ["gate", "up", "down"]:
        ffn_names.append(f"l{i}.{kind}")

missing = []
for name in ffn_names:
    layer, kind = name.split(".")
    h_key = f"{layer}.mlp_in" if kind in ("gate", "up") else f"{layer}.down_in"
    if h_key not in hess_dict:
        missing.append((name, h_key))

print(f"Total de matrizes FFN alvo: {len(ffn_names)}")
if missing:
    print(f"ERRO: Faltam Hessianas para: {missing}")
else:
    print("Sucesso: Todas as 72 matrizes FFN possuem Hessiana correspondente exata!")

# Carregar camada 0 do modelo para teste piloto
from transformers import AutoModelForCausalLM
print("\nCarregando camada 0 de Qwen3.5-2B...")
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
W_gate = model.model.layers[0].mlp.gate_proj.weight.data.clone().float() # [6144, 2048]
W_down = model.model.layers[0].mlp.down_proj.weight.data.clone().float() # [2048, 6144]
H_gate = hess_dict["l0.mlp_in"].float() # [2048, 2048]
H_down = hess_dict["l0.down_in"].float() # [6144, 6144]

print(f"W_gate: shape={W_gate.shape}, H_gate: shape={H_gate.shape}")
print(f"W_down: shape={W_down.shape}, H_down: shape={H_down.shape}")

# Funcao de erro funcional exato
def calc_metrics(W, What, H):
    D = (W - What)
    fro_W = W.pow(2).sum()
    fro_D = D.pow(2).sum()
    rel_fro = (fro_D / fro_W).item()
    mse = F.mse_loss(What, W).item()
    cos_sim = F.cosine_similarity(W.flatten(), What.flatten(), dim=0).item()
    
    # Tr(D H D^T) e Tr(W H W^T)
    # D: [out, in], H: [in, in] -> D @ H @ D.T tem traco = sum((D @ H) * D)
    DH = D @ H
    WH = W @ H
    tr_dhd = (DH * D).sum().item()
    tr_whw = (WH * W).sum().item()
    rel_func = tr_dhd / tr_whw if tr_whw > 0 else 0.0
    return {
        "mse": mse,
        "rel_fro": rel_fro,
        "cos_sim": cos_sim,
        "tr_dhd": tr_dhd,
        "rel_func": rel_func
    }

print("\nMetricas base calculadas com sucesso.")

def train_kmeans(vectors, K=26, n_iter=6, seed=42):
    torch.manual_seed(seed)
    N, D = vectors.shape
    if N <= K:
        if N == 0: return torch.zeros(K, D)
        rep = (K // N) + 1
        return vectors.repeat(rep, 1)[:K]
    
    centers = [vectors[torch.randint(0, N, (1,))[0]]]
    for _ in range(1, K):
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

def quantize_matrix_h5a_4bpw(weight_tensor, num_ctrl=3, num_spec=3, p_crit=0.05, seed=42, block_size=8):
    torch.manual_seed(seed)
    orig_shape = weight_tensor.shape
    orig_dtype = weight_tensor.dtype
    w_flat = weight_tensor.float().reshape(-1, block_size)
    N = w_flat.shape[0]
    batch_size = 200000
    
    sample_n = min(N, 50000)
    perm_sample = torch.randperm(N)[:sample_n]
    w_sample = w_flat[perm_sample]
    
    # 1. Macro Familias (1.0 bpp)
    perm_fam = torch.randperm(sample_n)[:8]
    fam_centers = w_sample[perm_fam].clone()
    for _ in range(5):
        d = torch.cdist(w_sample[:10000], fam_centers)
        lbl = torch.argmin(d, dim=1)
        for k in range(8):
            m = (lbl == k)
            if m.sum() > 0: fam_centers[k] = w_sample[:10000][m].mean(dim=0)
                
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
        norms = torch.norm(vecs_k, dim=1)
        crit_score = (norms + 1e-5) * (min_err ** 2)
        top_k = max(num_spec, int(vecs_k.shape[0] * 0.35))
        crit_idx = torch.topk(crit_score, k=top_k).indices
        spec_k = train_kmeans(vecs_k[crit_idx], K=num_spec, seed=seed)
        libs_spec.append(spec_k)
        
    macro_recon = torch.empty_like(w_flat)
    for k in range(8):
        m_full = (fam_ids == k)
        count_k = m_full.sum().item()
        if count_k == 0: continue
        w_k = w_flat[m_full]
        lib_base = libs_macro[k]
        lib_spec = libs_spec[k]
        recon_k = torch.empty_like(w_k)
        for sb_start in range(0, count_k, 50000):
            sb_end = min(sb_start + 50000, count_k)
            w_sub = w_k[sb_start:sb_end]
            d_base = torch.cdist(w_sub, lib_base)
            min_base, idx_base = torch.min(d_base, dim=1)
            d_c0 = torch.cdist(w_sub, lib_base * 1.5)
            d_c1 = torch.cdist(w_sub, lib_base * 0.5)
            d_c2 = torch.cdist(w_sub, -lib_base)
            min_c0, idx_c0 = torch.min(d_c0, dim=1)
            min_c1, idx_c1 = torch.min(d_c1, dim=1)
            min_c2, idx_c2 = torch.min(d_c2, dim=1)
            best_ctrl_dists = torch.stack([min_c0, min_c1, min_c2], dim=1)
            ctrl_indices = [idx_c0, idx_c1, idx_c2]
            min_ctrl_val, choice_ctrl = torch.min(best_ctrl_dists, dim=1)
            d_spec = torch.cdist(w_sub, lib_spec)
            min_spec_val, idx_spec = torch.min(d_spec, dim=1)
            
            sub_recon = lib_base[idx_base].clone()
            win_spec = (min_spec_val < min_base) & (min_spec_val < min_ctrl_val)
            win_ctrl = (min_ctrl_val < min_base) & (min_ctrl_val <= min_spec_val)
            sub_recon[win_spec] = lib_spec[idx_spec[win_spec]]
            mask_c0 = win_ctrl & (choice_ctrl == 0)
            mask_c1 = win_ctrl & (choice_ctrl == 1)
            mask_c2 = win_ctrl & (choice_ctrl == 2)
            sub_recon[mask_c0] = lib_base[ctrl_indices[0][mask_c0]] * 1.5
            sub_recon[mask_c1] = lib_base[ctrl_indices[1][mask_c1]] * 0.5
            sub_recon[mask_c2] = -lib_base[ctrl_indices[2][mask_c2]]
            recon_k[sb_start:sb_end] = sub_recon
        macro_recon[m_full] = recon_k
        
    # 2. Residuo 1 (Base 2.0 bpp)
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
        
    # 3. Residuo 2 (Base 3.0 bpp)
    res_2 = w_flat - recon_base
    lib_r2 = train_kmeans(res_2[perm_sample], K=26, seed=seed)
    recon_3 = recon_base.clone()
    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        d_r2 = torch.cdist(res_2[i:end_i], lib_r2)
        recon_3[i:end_i] += lib_r2[torch.argmin(d_r2, dim=1)]
        
    # 4. Residuo 3 (Estagio Adicional -> Base 4.0 bpp)
    res_3 = w_flat - recon_3
    lib_r3 = train_kmeans(res_3[perm_sample], K=26, seed=seed)
    recon_4 = recon_3.clone()
    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        d_r3 = torch.cdist(res_3[i:end_i], lib_r3)
        recon_4[i:end_i] += lib_r3[torch.argmin(d_r3, dim=1)]
        
    # 5. Protecao Seletiva Top P_crit% (+0.05 bpp -> 4.05 bpp)
    if p_crit > 0:
        sens_list = []
        for i in range(0, N, batch_size):
            end_i = min(i + batch_size, N)
            sub_w = w_flat[i:end_i]
            sub_res = sub_w - recon_4[i:end_i]
            block_norm = torch.norm(sub_w, dim=1)
            sens = (block_norm + 1e-5) * torch.mean(sub_res ** 2, dim=1)
            sens_list.append(sens)
        sensitivity = torch.cat(sens_list)
        num_crit = int(p_crit * N)
        crit_idx = torch.topk(sensitivity, k=num_crit).indices
        r4_vecs = (w_flat[crit_idx] - recon_4[crit_idx])
        lib_r4 = train_kmeans(r4_vecs, K=26, seed=seed)
        for sb_start in range(0, num_crit, 50000):
            sb_end = min(sb_start + 50000, num_crit)
            sub_r4 = r4_vecs[sb_start:sb_end]
            d_r4 = torch.cdist(sub_r4, lib_r4)
            recon_4[crit_idx[sb_start:sb_end]] += lib_r4[torch.argmin(d_r4, dim=1)]
            
    return recon_4.view(orig_shape).to(orig_dtype)

# =============================================================================
# H5-A 4.05 com Compensacao Hessiana Integrada
# =============================================================================
def gptq_prep(H, damp=0.01):
    H = H.clone()
    d = torch.diag(H)
    dead = d == 0
    H[dead, dead] = 1.0
    H.diagonal().add_(damp * torch.mean(torch.diag(H)))
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    U = torch.linalg.cholesky(Hinv, upper=True)
    return U, dead

def quantize_matrix_h5a_4bpw_hessian(weight_tensor, H, num_ctrl=3, num_spec=3, p_crit=0.05, seed=42, block_size=8, BS=128):
    torch.manual_seed(seed)
    orig_shape = weight_tensor.shape
    orig_dtype = weight_tensor.dtype
    out_dim, in_dim = orig_shape
    
    # 1. Preparar Codebooks H5-A a partir de W original
    w_flat = weight_tensor.float().reshape(-1, block_size)
    N = w_flat.shape[0]
    
    sample_n = min(N, 50000)
    perm_sample = torch.randperm(N)[:sample_n]
    w_sample = w_flat[perm_sample]
    
    # Macro Familias
    perm_fam = torch.randperm(sample_n)[:8]
    fam_centers = w_sample[perm_fam].clone()
    for _ in range(5):
        d = torch.cdist(w_sample[:10000], fam_centers)
        lbl = torch.argmin(d, dim=1)
        for k in range(8):
            m = (lbl == k)
            if m.sum() > 0: fam_centers[k] = w_sample[:10000][m].mean(dim=0)
                
    fam_sample = torch.argmin(torch.cdist(w_sample, fam_centers), dim=1)
    libs_macro = [train_kmeans(w_sample[fam_sample == k], K=26, seed=seed) for k in range(8)]
    
    libs_spec = []
    for k in range(8):
        m_k = (fam_sample == k)
        vecs_k = w_sample[m_k]
        if vecs_k.shape[0] < num_spec:
            libs_spec.append(fam_centers[k].unsqueeze(0).repeat(num_spec, 1))
            continue
        d_base = torch.cdist(vecs_k, libs_macro[k])
        min_err, _ = torch.min(d_base, dim=1)
        crit_score = (torch.norm(vecs_k, dim=1) + 1e-5) * (min_err ** 2)
        top_k = max(num_spec, int(vecs_k.shape[0] * 0.35))
        crit_idx = torch.topk(crit_score, k=top_k).indices
        spec_k = train_kmeans(vecs_k[crit_idx], K=num_spec, seed=seed)
        libs_spec.append(spec_k)
        
    # Codebooks residuais treinados na amostra
    sample_rec_1 = torch.empty_like(w_sample)
    for k in range(8):
        m_k = (fam_sample == k)
        if m_k.sum() == 0: continue
        ws = w_sample[m_k]
        db = torch.cdist(ws, libs_macro[k])
        mb, ib = torch.min(db, dim=1)
        d0, d1, d2 = torch.cdist(ws, libs_macro[k]*1.5), torch.cdist(ws, libs_macro[k]*0.5), torch.cdist(ws, -libs_macro[k])
        mc0, ic0 = torch.min(d0, dim=1)
        mc1, ic1 = torch.min(d1, dim=1)
        mc2, ic2 = torch.min(d2, dim=1)
        best_c, ch_c = torch.min(torch.stack([mc0, mc1, mc2], dim=1), dim=1)
        ds = torch.cdist(ws, libs_spec[k])
        ms, is_ = torch.min(ds, dim=1)
        sr = libs_macro[k][ib].clone()
        ws_mask = (ms < mb) & (ms < best_c)
        wc_mask = (best_c < mb) & (best_c <= ms)
        sr[ws_mask] = libs_spec[k][is_[ws_mask]]
        m0 = wc_mask & (ch_c == 0)
        m1 = wc_mask & (ch_c == 1)
        m2 = wc_mask & (ch_c == 2)
        sr[m0] = libs_macro[k][ic0[m0]] * 1.5
        sr[m1] = libs_macro[k][ic1[m1]] * 0.5
        sr[m2] = -libs_macro[k][ic2[m2]]
        sample_rec_1[m_k] = sr
        
    r1_sample = w_sample - sample_rec_1
    perm_r1 = torch.randperm(sample_n)[:8]
    r1_centers = r1_sample[perm_r1].clone()
    for _ in range(5):
        d = torch.cdist(r1_sample[:10000], r1_centers)
        lbl = torch.argmin(d, dim=1)
        for k in range(8):
            m = (lbl == k)
            if m.sum() > 0: r1_centers[k] = r1_sample[:10000][m].mean(dim=0)
    r1_sample_fam = torch.argmin(torch.cdist(r1_sample, r1_centers), dim=1)
    libs_r1 = [train_kmeans(r1_sample[r1_sample_fam == k], K=26, seed=seed) for k in range(8)]
    
    sample_rec_2 = sample_rec_1.clone()
    for k in range(8):
        m_k = (r1_sample_fam == k)
        if m_k.sum() == 0: continue
        rk = r1_sample[m_k]
        sample_rec_2[m_k] += libs_r1[k][torch.argmin(torch.cdist(rk, libs_r1[k]), dim=1)]
        
    r2_sample = w_sample - sample_rec_2
    lib_r2 = train_kmeans(r2_sample, K=26, seed=seed)
    sample_rec_3 = sample_rec_2 + lib_r2[torch.argmin(torch.cdist(r2_sample, lib_r2), dim=1)]
    
    r3_sample = w_sample - sample_rec_3
    lib_r3 = train_kmeans(r3_sample, K=26, seed=seed)
    sample_rec_4 = sample_rec_3 + lib_r3[torch.argmin(torch.cdist(r3_sample, lib_r3), dim=1)]
    
    r4_sample = w_sample - sample_rec_4
    sens_sample = (torch.norm(w_sample, dim=1) + 1e-5) * torch.mean(r4_sample ** 2, dim=1)
    top_crit = int(p_crit * sample_n)
    crit_sample_idx = torch.topk(sens_sample, k=top_crit).indices
    lib_r4 = train_kmeans(r4_sample[crit_sample_idx], K=26, seed=seed)
    
    # Codificador por escada H5-A para um lote de vetores [M, 8]
    def encode_h5a_ladder(v_in):
        M = v_in.shape[0]
        f_ids = torch.argmin(torch.cdist(v_in, fam_centers), dim=1)
        r_stage1 = torch.empty_like(v_in)
        for k in range(8):
            m = (f_ids == k)
            if m.sum() == 0: continue
            vs = v_in[m]
            db = torch.cdist(vs, libs_macro[k])
            mb, ib = torch.min(db, dim=1)
            d0, d1, d2 = torch.cdist(vs, libs_macro[k]*1.5), torch.cdist(vs, libs_macro[k]*0.5), torch.cdist(vs, -libs_macro[k])
            mc0, ic0 = torch.min(d0, dim=1)
            mc1, ic1 = torch.min(d1, dim=1)
            mc2, ic2 = torch.min(d2, dim=1)
            best_c, ch_c = torch.min(torch.stack([mc0, mc1, mc2], dim=1), dim=1)
            ds = torch.cdist(vs, libs_spec[k])
            ms, is_ = torch.min(ds, dim=1)
            sr = libs_macro[k][ib].clone()
            ws_mask = (ms < mb) & (ms < best_c)
            wc_mask = (best_c < mb) & (best_c <= ms)
            sr[ws_mask] = libs_spec[k][is_[ws_mask]]
            m0 = wc_mask & (ch_c == 0)
            m1 = wc_mask & (ch_c == 1)
            m2 = wc_mask & (ch_c == 2)
            sr[m0] = libs_macro[k][ic0[m0]] * 1.5
            sr[m1] = libs_macro[k][ic1[m1]] * 0.5
            sr[m2] = -libs_macro[k][ic2[m2]]
            r_stage1[m] = sr
            
        res1 = v_in - r_stage1
        r1_ids = torch.argmin(torch.cdist(res1, r1_centers), dim=1)
        r_stage2 = r_stage1.clone()
        for k in range(8):
            m = (r1_ids == k)
            if m.sum() == 0: continue
            r_stage2[m] += libs_r1[k][torch.argmin(torch.cdist(res1[m], libs_r1[k]), dim=1)]
            
        res2 = v_in - r_stage2
        r_stage3 = r_stage2 + lib_r2[torch.argmin(torch.cdist(res2, lib_r2), dim=1)]
        
        res3 = v_in - r_stage3
        r_stage4 = r_stage3 + lib_r3[torch.argmin(torch.cdist(res3, lib_r3), dim=1)]
        
        if p_crit > 0:
            res4 = v_in - r_stage4
            crit_score = (torch.norm(v_in, dim=1) + 1e-5) * torch.mean(res4 ** 2, dim=1)
            n_top = int(p_crit * M)
            if n_top > 0:
                top_idx = torch.topk(crit_score, k=n_top).indices
                r4_v = res4[top_idx]
                r_stage4[top_idx] += lib_r4[torch.argmin(torch.cdist(r4_v, lib_r4), dim=1)]
                
        return r_stage4

    # 2. Reconstrucao com Compensacao Hessiana
    U, dead = gptq_prep(H)
    W_curr = weight_tensor.clone().float()
    W_curr[:, dead] = 0
    Q = torch.zeros_like(W_curr)
    
    for i1 in range(0, in_dim, BS):
        i2 = min(i1 + BS, in_dim)
        W1 = W_curr[:, i1:i2].clone()
        Err = torch.zeros_like(W1)
        for j in range(0, i2 - i1, block_size):
            blk = W1[:, j:j + block_size]
            q_blk = encode_h5a_ladder(blk)
            Q[:, i1 + j:i1 + j + block_size] = q_blk
            
            Ubb = U[i1 + j:i1 + j + block_size, i1 + j:i1 + j + block_size]
            E = torch.linalg.solve_triangular(Ubb, (blk - q_blk), upper=True, left=False)
            Err[:, j:j + block_size] = E
            if j + block_size < i2 - i1:
                W1[:, j + block_size:] -= E @ U[i1 + j:i1 + j + block_size, i1 + j + block_size:i2]
        if i2 < in_dim:
            W_curr[:, i2:] -= Err @ U[i1:i2, i2:]
            
    return Q.to(orig_dtype)

# =============================================================================
# EXECUTAR TESTE PILOTO EM W_gate
# =============================================================================
print("\n" + "=" * 80)
print("EXECUTANDO QUANTIZACAO PILOTO EM l0.gate [6144, 2048]...")
print("=" * 80)

# 1. H5-A 4.05 Normal (Controle)
print("1. Executando H5-A 4.05 Controle...")
t0 = time.time()
What_ctrl = quantize_matrix_h5a_4bpw(W_gate, seed=42)
t_ctrl = time.time() - t0
met_ctrl = calc_metrics(W_gate, What_ctrl, H_gate)
print(f"H5-A 4.05 Controle concluido em {t_ctrl:.2f}s:")
print(f"  MSE:        {met_ctrl['mse']:.6e}")
print(f"  Rel Fro:    {met_ctrl['rel_fro']:.6f}")
print(f"  Cos Sim:    {met_ctrl['cos_sim']:.6f}")
print(f"  Tr(D H D^T): {met_ctrl['tr_dhd']:.4e}")
print(f"  Rel Func:   {met_ctrl['rel_func']:.6f}")

# 2. H5-A 4.05 + Hessiana
print("\n2. Executando H5-A 4.05 + Hessiana (GPTQ Block-wise OBS)...")
t0 = time.time()
What_hess = quantize_matrix_h5a_4bpw_hessian(W_gate, H_gate, seed=42)
t_hess = time.time() - t0
met_hess = calc_metrics(W_gate, What_hess, H_gate)
print(f"H5-A 4.05 + Hessiana concluido em {t_hess:.2f}s:")
print(f"  MSE:        {met_hess['mse']:.6e}")
print(f"  Rel Fro:    {met_hess['rel_fro']:.6f}")
print(f"  Cos Sim:    {met_hess['cos_sim']:.6f}")
print(f"  Tr(D H D^T): {met_hess['tr_dhd']:.4e}")
print(f"  Rel Func:   {met_hess['rel_func']:.6f}")

# Comparacao l0.gate
print("\n" + "=" * 80)
print("COMPARATIVO DE SANIDADE (l0.gate):")
print(f"  Erro Funcional Relativo (Rel Func):")
print(f"    Controle: {met_ctrl['rel_func']:.6f}")
print(f"    Hessiana: {met_hess['rel_func']:.6f}")
delta_func = (met_hess['rel_func'] - met_ctrl['rel_func']) / met_ctrl['rel_func'] * 100
print(f"    Variacao no Erro Funcional: {delta_func:+.2f}%")
if met_hess['rel_func'] < met_ctrl['rel_func']:
    print("  -> SUCESSO: A compensacao Hessiana REDUZIU expressivamente o erro funcional em l0.gate!")
else:
    print("  -> ALERTA: A compensacao Hessiana nao reduziu o erro funcional em l0.gate.")
print("=" * 80)

# Teste piloto em l0.down [2048, 6144]
print("\n" + "=" * 80)
print("EXECUTANDO QUANTIZACAO PILOTO EM l0.down [2048, 6144]...")
print("=" * 80)
t0 = time.time()
What_down_ctrl = quantize_matrix_h5a_4bpw(W_down, seed=42)
t_down_ctrl = time.time() - t0
met_down_ctrl = calc_metrics(W_down, What_down_ctrl, H_down)
print(f"l0.down Controle concluido em {t_down_ctrl:.2f}s:")
print(f"  MSE:        {met_down_ctrl['mse']:.6e}")
print(f"  Tr(D H D^T): {met_down_ctrl['tr_dhd']:.4e}")
print(f"  Rel Func:   {met_down_ctrl['rel_func']:.6f}")

t0 = time.time()
What_down_hess = quantize_matrix_h5a_4bpw_hessian(W_down, H_down, seed=42)
t_down_hess = time.time() - t0
met_down_hess = calc_metrics(W_down, What_down_hess, H_down)
print(f"l0.down + Hessiana concluido em {t_down_hess:.2f}s:")
print(f"  MSE:        {met_down_hess['mse']:.6e}")
print(f"  Tr(D H D^T): {met_down_hess['tr_dhd']:.4e}")
print(f"  Rel Func:   {met_down_hess['rel_func']:.6f}")

delta_down = (met_down_hess['rel_func'] - met_down_ctrl['rel_func']) / met_down_ctrl['rel_func'] * 100
print(f"\nVariacao no Erro Funcional em l0.down: {delta_down:+.2f}%")
if met_down_hess['rel_func'] < met_down_ctrl['rel_func']:
    print("  -> SUCESSO: A compensacao Hessiana REDUZIU expressivamente o erro funcional em l0.down!")
print("=" * 80)


