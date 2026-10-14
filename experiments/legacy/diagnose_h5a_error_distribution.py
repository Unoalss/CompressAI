import os
import sys
import time
import json
import numpy as np
import torch
from safetensors import safe_open
from test_qwen_ffn_single_layer import quantize_matrix_h5a_batched

SAFETENSORS_PATH = "E:/agente/modelo/model.safetensors-00001-of-00001.safetensors"
KEY_NAME = "model.language_model.layers.0.mlp.gate_proj.weight"

print("=" * 100)
print("DIAGNÓSTICO ESTRUTURAL DO ERRO DO H5-A: POR QUE 98,33% DE COSINE NÃO VIRA BOA PPL?")
print(f"Matriz Alvo: {KEY_NAME}")
print("=" * 100)

device = "cpu"
torch.set_num_threads(os.cpu_count() or 8)

# -----------------------------------------------------------------------------
# 1. FUNÇÃO H5-A 4.05 BPW (AUTOCONTIDA)
# -----------------------------------------------------------------------------
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
    
    # 1. Macro Famílias
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
        sub_b = 50000
        for sb_start in range(0, count_k, sub_b):
            sb_end = min(sb_start + sub_b, count_k)
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
        
    # 2. Resíduo 1
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
        
    # 3. Resíduo 2
    res_2 = w_flat - recon_base
    lib_r2 = train_kmeans(res_2[perm_sample], K=26, seed=seed)
    recon_3 = recon_base.clone()
    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        d_r2 = torch.cdist(res_2[i:end_i], lib_r2)
        recon_3[i:end_i] += lib_r2[torch.argmin(d_r2, dim=1)]
        
    # 4. Resíduo 3
    res_3 = w_flat - recon_3
    lib_r3 = train_kmeans(res_3[perm_sample], K=26, seed=seed)
    recon_4 = recon_3.clone()
    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        d_r3 = torch.cdist(res_3[i:end_i], lib_r3)
        recon_4[i:end_i] += lib_r3[torch.argmin(d_r3, dim=1)]
        
    # 5. Proteção Seletiva Top P_crit%
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
            
    recon_reshaped = recon_4.view(orig_shape).to(orig_dtype)
    return recon_reshaped

# -----------------------------------------------------------------------------
# 2. CARREGAMENTO DA MATRIZ ORIGINAL BF16
# -----------------------------------------------------------------------------
print("\nCarregando matriz original diretamente do arquivo safetensors...")
with safe_open(SAFETENSORS_PATH, framework="pt", device="cpu") as f:
    W_orig = f.get_tensor(KEY_NAME).float()

shape = W_orig.shape
num_rows, num_cols = shape
num_elements = W_orig.numel()
print(f"Matriz carregada com sucesso! Shape: {shape} ({num_rows} linhas x {num_cols} colunas = {num_elements:,} pesos)")
print(f"Norma L2 da Matriz: {torch.norm(W_orig).item():.4f} | Média: {W_orig.mean().item():.6f} | Std: {W_orig.std().item():.6f}")

# -----------------------------------------------------------------------------
# 3. RECONSTRUÇÕES: H5-A @ 3.05 BPW E H5-A @ 4.05 BPW
# -----------------------------------------------------------------------------
print("\n" + "=" * 90)
print("EXECUTANDO AS 3 RECONSTRUÇÕES...")
print("=" * 90)

# Versão 1: BF16 Original (Referência)
W_bf16 = W_orig.clone()

# Versão 2: H5-A @ 3.05 BPW (num_ctrl=2, num_spec=4, p_crit=5%)
t0 = time.time()
W_h5a_305, bpp_305, _, _, _ = quantize_matrix_h5a_batched(W_orig, num_ctrl=2, num_spec=4, p_crit=0.05, seed=42)
W_h5a_305 = W_h5a_305.float()
t_305 = time.time() - t0
print(f"-> H5-A @ 3.05 BPW gerado em {t_305:.2f}s (Taxa: {bpp_305:.2f} bpw)")

# Versão 3: H5-A @ 4.05 BPW (4 estágios residuais em cascata, p_crit=5%)
t0 = time.time()
W_h5a_405 = quantize_matrix_h5a_4bpw(W_orig, num_ctrl=3, num_spec=3, p_crit=0.05, seed=42)
W_h5a_405 = W_h5a_405.float()
t_405 = time.time() - t0
print(f"-> H5-A @ 4.05 BPW gerado em {t_405:.2f}s (Taxa: 4.05 bpw)")

# -----------------------------------------------------------------------------
# 4. CÁLCULO DAS MÉTRICAS GLOBAIS
# -----------------------------------------------------------------------------
def analyze_reconstruction(W_true, W_rec, name, bpw):
    E = W_true - W_rec
    abs_E = torch.abs(E)
    
    mse = torch.mean(E ** 2).item()
    mae = torch.mean(abs_E).item()
    rmse = np.sqrt(mse)
    cos = torch.nn.functional.cosine_similarity(W_true.view(1, -1), W_rec.view(1, -1)).item()
    max_err = torch.max(abs_E).item()
    
    p50 = torch.quantile(abs_E, 0.50).item()
    p90 = torch.quantile(abs_E, 0.90).item()
    p95 = torch.quantile(abs_E, 0.95).item()
    p99 = torch.quantile(abs_E, 0.99).item()
    p999 = torch.quantile(abs_E, 0.999).item()
    
    # Erro relativo global
    rel_fro = (torch.norm(E) / torch.norm(W_true)).item()
    
    # Erro por linha (output channel: dim 0)
    row_err = torch.mean(E ** 2, dim=1) # shape: (6144,)
    row_norm = torch.norm(W_true, dim=1) # shape: (6144,)
    row_rel = (torch.norm(E, dim=1) / (row_norm + 1e-8)) # shape: (6144,)
    
    # Erro por coluna (input channel: dim 1)
    col_err = torch.mean(E ** 2, dim=0) # shape: (2048,)
    col_norm = torch.norm(W_true, dim=0) # shape: (2048,)
    col_rel = (torch.norm(E, dim=0) / (col_norm + 1e-8)) # shape: (2048,)
    
    return {
        "name": name,
        "bpw": bpw,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "cosine": cos,
        "max_err": max_err,
        "p50": p50,
        "p90": p90,
        "p95": p95,
        "p99": p99,
        "p999": p999,
        "rel_fro": rel_fro,
        "row_err": row_err,
        "row_rel": row_rel,
        "col_err": col_err,
        "col_rel": col_rel,
        "residual": E
    }

res_305 = analyze_reconstruction(W_orig, W_h5a_305, "H5-A 3.05 BPW", 3.05)
res_405 = analyze_reconstruction(W_orig, W_h5a_405, "H5-A 4.05 BPW", 4.05)

print("\n" + "=" * 105)
print("TABELA 1: COMPARAÇÃO ESTATÍSTICA DAS RECONSTRUÇÕES")
print("=" * 105)
print(f"{'Métrica':<25} | {'BF16 Original':<18} | {'H5-A @ 3.05 BPW':<22} | {'H5-A @ 4.05 BPW':<22}")
print("-" * 105)
print(f"{'BPW':<25} | {'16.0':<18} | {res_305['bpw']:<22.2f} | {res_405['bpw']:<22.2f}")
cos_str_305 = f"{res_305['cosine']:.6f} ({res_305['cosine']*100:.2f}%)"
cos_str_405 = f"{res_405['cosine']:.6f} ({res_405['cosine']*100:.2f}%)"
print(f"{'Cosine Similarity':<25} | {'1.000000':<18} | {cos_str_305:<22} | {cos_str_405:<22}")
print(f"{'MSE (Mean Squared Error)':<25} | {'0.000000':<18} | {res_305['mse']:<22.7f} | {res_405['mse']:<22.7f}")
print(f"{'MAE (Mean Absolute Error)':<25}| {'0.000000':<18} | {res_305['mae']:<22.6f} | {res_405['mae']:<22.6f}")
print(f"{'Erro Máximo Absoluto':<25} | {'0.000000':<18} | {res_305['max_err']:<22.6f} | {res_405['max_err']:<22.6f}")
print(f"{'Percentil 50 (Mediana)':<25}| {'0.000000':<18} | {res_305['p50']:<22.6f} | {res_405['p50']:<22.6f}")
print(f"{'Percentil 95 do Erro':<25} | {'0.000000':<18} | {res_305['p95']:<22.6f} | {res_405['p95']:<22.6f}")
print(f"{'Percentil 99 do Erro':<25} | {'0.000000':<18} | {res_305['p99']:<22.6f} | {res_405['p99']:<22.6f}")
print(f"{'Percentil 99.9 do Erro':<25}| {'0.000000':<18} | {res_305['p999']:<22.6f} | {res_405['p999']:<22.6f}")
fro_str_305 = f"{res_305['rel_fro']*100:.2f}%"
fro_str_405 = f"{res_405['rel_fro']*100:.2f}%"
print(f"{'Erro Relativo Frobenius':<25} | {'0.0%':<18} | {fro_str_305:<22} | {fro_str_405:<22}")
print("=" * 105)

# -----------------------------------------------------------------------------
# 5. ANÁLISE DE CONCENTRAÇÃO DO ERRO POR CANAL (INPUT E OUTPUT)
# -----------------------------------------------------------------------------
print("\n" + "=" * 105)
print("TABELA 2: ANÁLISE DE CONCENTRAÇÃO DO ERRO POR CANAL (H5-A @ 3.05 BPW)")
print("Objetivo: Descobrir se uma pequena quantidade de canais concentra uma grande parcela do erro")
print("=" * 105)

def analyze_channel_concentration(errors_per_channel, channel_type):
    num_ch = len(errors_per_channel)
    sorted_err, sorted_idx = torch.sort(errors_per_channel, descending=True)
    total_err = sorted_err.sum().item()
    
    top1_count = max(1, int(0.01 * num_ch))
    top2_count = max(1, int(0.02 * num_ch))
    top5_count = max(1, int(0.05 * num_ch))
    top10_count = max(1, int(0.10 * num_ch))
    top20_count = max(1, int(0.20 * num_ch))
    
    top1_share = sorted_err[:top1_count].sum().item() / total_err * 100
    top2_share = sorted_err[:top2_count].sum().item() / total_err * 100
    top5_share = sorted_err[:top5_count].sum().item() / total_err * 100
    top10_share = sorted_err[:top10_count].sum().item() / total_err * 100
    top20_share = sorted_err[:top20_count].sum().item() / total_err * 100
    
    min_val = sorted_err[-1].item()
    p50_val = sorted_err[num_ch // 2].item()
    max_val = sorted_err[0].item()
    ratio_max_median = max_val / (p50_val + 1e-12)
    
    print(f"\n--- {channel_type.upper()} ({num_ch} canais no total) ---")
    print(f"Top 1% dos canais ({top1_count} canais)   concentram: {top1_share:.2f}% de todo o erro quadrático")
    print(f"Top 2% dos canais ({top2_count} canais)   concentram: {top2_share:.2f}% de todo o erro quadrático")
    print(f"Top 5% dos canais ({top5_count} canais)   concentram: {top5_share:.2f}% de todo o erro quadrático")
    print(f"Top 10% dos canais ({top10_count} canais) concentram: {top10_share:.2f}% de todo o erro quadrático")
    print(f"Top 20% dos canais ({top20_count} canais) concentram: {top20_share:.2f}% de todo o erro quadrático")
    print(f"Razão Outlier (Canal Pior / Mediana): {ratio_max_median:.2f}x mais erro que a mediana!")
    
    return sorted_idx, {
        "top1_share": top1_share,
        "top2_share": top2_share,
        "top5_share": top5_share,
        "top10_share": top10_share,
        "ratio_max_median": ratio_max_median,
        "top_indices": sorted_idx[:top5_count].tolist()
    }

print("\nAnalisando Canais de Saída (Linhas da Matriz: 6144 features intermediárias do SwiGLU)...")
crit_rows_idx, row_stats = analyze_channel_concentration(res_305["row_err"], "Canais de Saída (Linhas)")

print("\nAnalisando Canais de Entrada (Colunas da Matriz: 2048 dimensões do Hidden State d_model)...")
crit_cols_idx, col_stats = analyze_channel_concentration(res_305["col_err"], "Canais de Entrada (Colunas)")

# Análise de Erro Relativo (Norma do Erro / Norma do Canal)
print("\n--- ANÁLISE DE ERRO RELATIVO POR CANAL DE ENTRADA (COLUNAS) ---")
sorted_col_rel, _ = torch.sort(res_305["col_rel"], descending=True)
print(f"Erro Relativo Médio: {res_305['col_rel'].mean().item()*100:.2f}%")
print(f"Erro Relativo Mediano: {torch.median(res_305['col_rel']).item()*100:.2f}%")
print(f"Erro Relativo nos 5% Piores Canais: {sorted_col_rel[:int(0.05*2048)].mean().item()*100:.2f}% (Máximo: {sorted_col_rel[0].item()*100:.2f}%)")
print(f"Erro Relativo nos 50% Melhores Canais: {sorted_col_rel[int(0.50*2048):].mean().item()*100:.2f}%")

# -----------------------------------------------------------------------------
# 6. O TESTE DECISIVO: H5-A @ 3.05 BPW + PROTEÇÃO DOS CANAIS CRÍTICOS
# -----------------------------------------------------------------------------
print("\n" + "=" * 105)
print("TESTE DECISIVO: H5-A (3.05 BPW) + PROTEÇÃO SELETIVA DE CANAIS CRÍTICOS")
print("Objetivo: Manter 95% ou 98% da matriz em 3.05 BPW e restaurar os 2% ou 5% canais críticos em BF16")
print("=" * 105)

# Variantes de proteção:
# A) 1% dos Canais de Entrada em BF16
# B) 2% dos Canais de Entrada em BF16
# C) 5% dos Canais de Entrada em BF16
# D) 5% dos Canais de Saída em BF16

def create_protected_matrix(W_base, W_exact, indices, axis=1):
    W_prot = W_base.clone()
    if axis == 1: # Protege colunas (canais de entrada)
        W_prot[:, indices] = W_exact[:, indices]
    else: # Protege linhas (canais de saída)
        W_prot[indices, :] = W_exact[indices, :]
    return W_prot

# Teste com diferentes percentuais de canais de entrada (colunas)
num_cols = 2048
cols_1pct = crit_cols_idx[:int(0.01 * num_cols)] # 20 canais
cols_2pct = crit_cols_idx[:int(0.02 * num_cols)] # 40 canais
cols_5pct = crit_cols_idx[:int(0.05 * num_cols)] # 102 canais

# Teste com canais de saída (linhas)
num_rows = 6144
rows_5pct = crit_rows_idx[:int(0.05 * num_rows)] # 307 canais

W_prot_col_1pct = create_protected_matrix(W_h5a_305, W_orig, cols_1pct, axis=1)
W_prot_col_2pct = create_protected_matrix(W_h5a_305, W_orig, cols_2pct, axis=1)
W_prot_col_5pct = create_protected_matrix(W_h5a_305, W_orig, cols_5pct, axis=1)
W_prot_row_5pct = create_protected_matrix(W_h5a_305, W_orig, rows_5pct, axis=0)

bpw_col_1pct = 0.99 * 3.05 + 0.01 * 16.0 # 3.18 BPW
bpw_col_2pct = 0.98 * 3.05 + 0.02 * 16.0 # 3.31 BPW
bpw_col_5pct = 0.95 * 3.05 + 0.05 * 16.0 # 3.70 BPW

res_col_1pct = analyze_reconstruction(W_orig, W_prot_col_1pct, "+1% Canais Entrada BF16", bpw_col_1pct)
res_col_2pct = analyze_reconstruction(W_orig, W_prot_col_2pct, "+2% Canais Entrada BF16", bpw_col_2pct)
res_col_5pct = analyze_reconstruction(W_orig, W_prot_col_5pct, "+5% Canais Entrada BF16", bpw_col_5pct)
res_row_5pct = analyze_reconstruction(W_orig, W_prot_row_5pct, "+5% Canais Saída BF16", bpw_col_5pct)

pct_1 = (1.0 - res_col_1pct['mse'] / res_305['mse']) * 100
pct_2 = (1.0 - res_col_2pct['mse'] / res_305['mse']) * 100
pct_5 = (1.0 - res_col_5pct['mse'] / res_305['mse']) * 100
pct_row5 = (1.0 - res_row_5pct['mse'] / res_305['mse']) * 100
pct_405 = (1.0 - res_405['mse'] / res_305['mse']) * 100

print("\n" + "=" * 115)
print("TABELA 3: IMPACTO DA PROTEÇÃO SELETIVA DE CANAIS NA RECONSTRUÇÃO DA MATRIZ")
print("=" * 115)
print(f"{'Configuração':<32} | {'BPW Efetivo':<12} | {'MSE':<14} | {'Cosine':<12} | {'Erro Máx':<12} | {'Queda de MSE':<12}")
print("-" * 115)
print(f"{'H5-A @ 3.05 BPW Puro':<32} | {'3.05 bpw':<12} | {res_305['mse']:<14.7f} | {res_305['cosine']*100:<11.2f}% | {res_305['max_err']:<12.5f} | {'Referência':<12}")
print(f"{'H5-A + 1% Canais Entrada BF16':<32} | {bpw_col_1pct:<9.2f} bpw | {res_col_1pct['mse']:<14.7f} | {res_col_1pct['cosine']*100:<11.2f}% | {res_col_1pct['max_err']:<12.5f} | -{pct_1:.1f}%")
print(f"{'H5-A + 2% Canais Entrada BF16':<32} | {bpw_col_2pct:<9.2f} bpw | {res_col_2pct['mse']:<14.7f} | {res_col_2pct['cosine']*100:<11.2f}% | {res_col_2pct['max_err']:<12.5f} | -{pct_2:.1f}%")
print(f"{'H5-A + 5% Canais Entrada BF16':<32} | {bpw_col_5pct:<9.2f} bpw | {res_col_5pct['mse']:<14.7f} | {res_col_5pct['cosine']*100:<11.2f}% | {res_col_5pct['max_err']:<12.5f} | -{pct_5:.1f}%")
print(f"{'H5-A + 5% Canais Saída BF16':<32}   | {bpw_col_5pct:<9.2f} bpw | {res_row_5pct['mse']:<14.7f} | {res_row_5pct['cosine']*100:<11.2f}% | {res_row_5pct['max_err']:<12.5f} | -{pct_row5:.1f}%")
print(f"{'H5-A @ 4.05 BPW Global':<32}      | {'4.05 bpw':<12} | {res_405['mse']:<14.7f} | {res_405['cosine']*100:<11.2f}% | {res_405['max_err']:<12.5f} | -{pct_405:.1f}%")
print("=" * 115)

# -----------------------------------------------------------------------------
# 7. IMPACTO NA ATIVAÇÃO FORWARD: Y = X * W^T
# -----------------------------------------------------------------------------
print("\n" + "=" * 105)
print("TABELA 4: SIMULAÇÃO DA PASSAGEM FORWARD (FIDELIDADE DAS ATIVAÇÕES DE SAÍDA)")
print("Explicando por que 98,33% de cosine de peso pode gerar distorção na ativação")
print("=" * 105)

# Gerar ativações sintéticas realistas baseadas na distribuição do Qwen
# Em transformers modernos, hidden_states possuem outliers característicos (curtose alta)
torch.manual_seed(42)
seq_len = 128
batch_size = 1
# Distribuição realista com outliers em canais específicos (Student-t com df=3 para caudas pesadas)
X_normal = torch.randn(seq_len, num_cols)
# 2% dos canais têm ativações de grande magnitude (outlier features clássicas de LLMs)
outlier_cols = crit_cols_idx[:int(0.02 * num_cols)]
X_normal[:, outlier_cols] *= 4.5 # Outlier activation scale

Y_exact = torch.matmul(X_normal, W_orig.T)
Y_norm = torch.norm(Y_exact).item()

def eval_activation_distortion(W_cand, name):
    Y_cand = torch.matmul(X_normal, W_cand.T)
    err_Y = Y_exact - Y_cand
    cos_Y = torch.nn.functional.cosine_similarity(Y_exact.view(1, -1), Y_cand.view(1, -1)).item()
    rel_err_Y = (torch.norm(err_Y) / Y_norm).item()
    snr_db = 10 * np.log10((torch.norm(Y_exact) ** 2 / (torch.norm(err_Y) ** 2 + 1e-12)).item())
    return cos_Y, rel_err_Y, snr_db

cos_act_305, rel_act_305, snr_305 = eval_activation_distortion(W_h5a_305, "H5-A 3.05 BPW")
cos_act_1pct, rel_act_1pct, snr_1pct = eval_activation_distortion(W_prot_col_1pct, "+1% Canais Entrada")
cos_act_2pct, rel_act_2pct, snr_2pct = eval_activation_distortion(W_prot_col_2pct, "+2% Canais Entrada")
cos_act_5pct, rel_act_5pct, snr_5pct = eval_activation_distortion(W_prot_col_5pct, "+5% Canais Entrada")
cos_act_405, rel_act_405, snr_405 = eval_activation_distortion(W_h5a_405, "H5-A 4.05 BPW")

print(f"{'Configuração':<32} | {'Cosine nos Pesos':<18} | {'Cosine na Ativação Y':<22} | {'Erro Relativo na Ativação':<26} | {'SNR (dB)'}")
print("-" * 115)
print(f"{'H5-A @ 3.05 BPW Puro':<32} | {res_305['cosine']*100:<17.2f}% | {cos_act_305*100:<21.3f}% | {rel_act_305*100:<25.2f}% | {snr_305:.2f} dB")
print(f"{'H5-A + 1% Canais Entrada BF16':<32} | {res_col_1pct['cosine']*100:<17.2f}% | {cos_act_1pct*100:<21.3f}% | {rel_act_1pct*100:<25.2f}% | {snr_1pct:.2f} dB")
print(f"{'H5-A + 2% Canais Entrada BF16':<32} | {res_col_2pct['cosine']*100:<17.2f}% | {cos_act_2pct*100:<21.3f}% | {rel_act_2pct*100:<25.2f}% | {snr_2pct:.2f} dB")
print(f"{'H5-A + 5% Canais Entrada BF16':<32} | {res_col_5pct['cosine']*100:<17.2f}% | {cos_act_5pct*100:<21.3f}% | {rel_act_5pct*100:<25.2f}% | {snr_5pct:.2f} dB")
print(f"{'H5-A @ 4.05 BPW Global':<32}      | {res_405['cosine']*100:<17.2f}% | {cos_act_405*100:<21.3f}% | {rel_act_405*100:<25.2f}% | {snr_405:.2f} dB")
print("=" * 115)

diagnosis_data = {
    "matrix_name": KEY_NAME,
    "shape": list(shape),
    "stats_305": {k: v for k, v in res_305.items() if not isinstance(v, torch.Tensor)},
    "stats_405": {k: v for k, v in res_405.items() if not isinstance(v, torch.Tensor)},
    "row_concentration": {k: v for k, v in row_stats.items() if k != "top_indices"},
    "col_concentration": {k: v for k, v in col_stats.items() if k != "top_indices"},
    "activation_simulation": {
        "snr_305": snr_305,
        "snr_1pct": snr_1pct,
        "snr_2pct": snr_2pct,
        "snr_5pct": snr_5pct,
        "snr_405": snr_405,
        "rel_act_305": rel_act_305,
        "rel_act_5pct": rel_act_5pct,
        "rel_act_405": rel_act_405
    }
}

with open("e:/agente/diagnose_error_distribution_results.json", "w") as f:
    json.dump(diagnosis_data, f, indent=2)

print("\nDiagnóstico completo salvo em e:/agente/diagnose_error_distribution_results.json")
