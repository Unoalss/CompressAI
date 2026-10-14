"""
test_h5a405_hessian.py - Experimento 2: H5-A 4.05 + Compensacao Hessiana (Isolado)

Objetivo:
Avaliar o ganho puro da compensacao Hessiana (Optimal Brain Surgeon / GPTQ Block-wise)
integrada na reconstrucao vetorial do H5-A 4.05 nas 72 matrizes FFN de Qwen3.5-2B.
Sem combinar com adaptadores R8/R16/R32 ou outras tecnicas.

Protocolo cientifico identico:
- Benchmark Original (Ciencia, Codigo, Historia, Enciclopedia)
- Holdout Real (Wikitext-2 + Code, 8 seqs x 512 tokens, bs=1)
- Mesmos tokens, mesma seed, mesmo modelo, mesmo calculo de PPL.
"""
import os, sys, time, math, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "modelo")
DATA = os.path.join(ROOT, "data")
HESS_PATH = os.path.join(DATA, "hess_ffn.pt")
OUT_FILE = os.path.join(ROOT, "h5a_405_hessian_results.json")

torch.set_num_threads(os.cpu_count() or 8)

def log(*args):
    print(*args, flush=True)

# -----------------------------------------------------------------------------
# 1. Pipeline H5-A 4.05 Base
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

def quantize_matrix_h5a_4bpw_hessian(weight_tensor, U_factor=None, dead_mask=None, H=None,
                                    num_ctrl=3, num_spec=3, p_crit=0.05, seed=42, block_size=8, BS=128):
    """
    Quantizacao H5-A 4.05 com Compensacao Hessiana Integrada.
    """
    torch.manual_seed(seed)
    orig_shape = weight_tensor.shape
    orig_dtype = weight_tensor.dtype
    out_dim, in_dim = orig_shape
    
    if U_factor is None or dead_mask is None:
        U, dead = gptq_prep(H)
    else:
        U, dead = U_factor, dead_mask
        
    w_flat = weight_tensor.float().reshape(-1, block_size)
    N = w_flat.shape[0]
    
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
        
    # Reconstrucao de estagio 1 na amostra para calcular residuos
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
        
    # 2. Residuo 1 (Base 2.0 bpp)
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
        
    # 3. Residuo 2 (Base 3.0 bpp)
    r2_sample = w_sample - sample_rec_2
    lib_r2 = train_kmeans(r2_sample, K=26, seed=seed)
    sample_rec_3 = sample_rec_2 + lib_r2[torch.argmin(torch.cdist(r2_sample, lib_r2), dim=1)]
    
    # 4. Residuo 3 (Base 4.0 bpp)
    r3_sample = w_sample - sample_rec_3
    lib_r3 = train_kmeans(r3_sample, K=26, seed=seed)
    sample_rec_4 = sample_rec_3 + lib_r3[torch.argmin(torch.cdist(r3_sample, lib_r3), dim=1)]
    
    # 5. Residuo 4 (Top P_crit% sensibilidade funcional -> 4.05 bpp)
    r4_sample = w_sample - sample_rec_4
    sens_sample = (torch.norm(w_sample, dim=1) + 1e-5) * torch.mean(r4_sample ** 2, dim=1)
    top_crit = int(p_crit * sample_n)
    crit_sample_idx = torch.topk(sens_sample, k=top_crit).indices
    lib_r4 = train_kmeans(r4_sample[crit_sample_idx], K=26, seed=seed)
    
    # Decodificador em escada H5-A para vetor de blocos [M, 8]
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

    # Compensacao Hessiana via Fator Cholesky Inverso
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

# -----------------------------------------------------------------------------
# 2. Metricas e Avaliacao
# -----------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

original_domains = {
    "Ciencia": "Quantum mechanics describes nature at the smallest scales of energy levels of atoms and subatomic particles.",
    "Codigo": "def binary_search(array, target): low = 0; high = len(array) - 1; while low <= high: mid = (low + high) // 2",
    "Historia": "The ancient library of Alexandria was once the largest repository of knowledge in the Hellenistic world.",
    "Enciclopedia": "The solar system formed approximately 4.6 billion years ago from the gravitational collapse of a molecular cloud."
}

def eval_original(m):
    scores = {}
    with torch.no_grad():
        for d, txt in original_domains.items():
            inputs = tokenizer(txt, return_tensors="pt")
            loss = m(**inputs, labels=inputs.input_ids).loss
            scores[d] = round(torch.exp(loss).item(), 2)
    scores["Media"] = round(sum(scores.values()) / len(scores), 2)
    return scores

def eval_holdout_fast(m, tokens_dict, n_seqs=4):
    scores = {}
    with torch.no_grad():
        for name, key in [("Wiki", "eval_wiki"), ("Code", "eval_code")]:
            toks = tokens_dict[key][:n_seqs]
            total_nll = 0.0
            total_tokens = 0
            for i in range(len(toks)):
                x = toks[i:i+1]
                logits = m(input_ids=x).logits
                loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                       x[:, 1:].reshape(-1), reduction="sum")
                total_nll += loss.item()
                total_tokens += x[:, 1:].numel()
            scores[name] = round(math.exp(total_nll / total_tokens), 2)
    scores["Media"] = round((scores["Wiki"] + scores["Code"]) / 2, 2)
    return scores

def calc_matrix_metrics(W, What, H):
    D = (W - What).float()
    W_f = W.float()
    fro_W = W_f.pow(2).sum()
    fro_D = D.pow(2).sum()
    rel_fro = (fro_D / fro_W).item() if fro_W > 0 else 0.0
    mse = F.mse_loss(What.float(), W_f).item()
    cos_sim = F.cosine_similarity(W_f.flatten(), What.float().flatten(), dim=0).item()
    
    DH = D @ H
    WH = W_f @ H
    tr_dhd = (DH * D).sum().item()
    tr_whw = (WH * W_f).sum().item()
    rel_func = tr_dhd / tr_whw if tr_whw > 0 else 0.0
    
    norm_w = torch.norm(W_f).item()
    norm_res = torch.norm(D).item()
    
    return {
        "mse": mse,
        "rel_fro": rel_fro,
        "cos_sim": cos_sim,
        "tr_dhd": tr_dhd,
        "rel_func": rel_func,
        "norm_w": norm_w,
        "norm_res": norm_res,
        "bpw": 4.05
    }

def save_data(res):
    with open(OUT_FILE, "w") as f:
        json.dump(res, f, indent=2)

# -----------------------------------------------------------------------------
# 3. Execucao Principal
# -----------------------------------------------------------------------------
def main():
    log("=" * 90)
    log("EXPERIMENTO 2: H5-A 4.05 + COMPENSACAO HESSIANA (ISOLADO)")
    log("=" * 90)
    
    tokens = torch.load(os.path.join(DATA, "tokens.pt"))
    hess_dict = torch.load(HESS_PATH, map_location="cpu")
    log(f"Carregados {len(hess_dict)} tensores de Hessiana de: {HESS_PATH}")
    
    results = {}
    
    # 1. Carregar Modelo BF16
    log("\n[1/3] Carregando modelo Qwen3.5-2B base...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()
    
    # Referencias conhecidas
    results["BF16"] = {
        "original": {"Ciencia": 5.05, "Codigo": 2.65, "Historia": 8.18, "Enciclopedia": 3.70, "Media": 4.89},
        "holdout": {"Wiki": 11.56, "Code": 3.76, "Media": 7.66},
        "bpw": 16.0
    }
    results["H5A_405_Controle"] = {
        "original": {"Ciencia": 10.11, "Codigo": 2.83, "Historia": 10.52, "Enciclopedia": 4.39, "Media": 6.96},
        "holdout": {"Wiki": 14.60, "Code": 4.86, "Media": 9.73},
        "bpw": 4.05
    }
    results["Q4_K_M"] = {
        "original": {"Media": 5.04},
        "holdout": {"Wiki": 15.63, "Code": 2.63, "Media": 9.13},
        "bpw": 5.45
    }
    results["GPTQ4_Hessian_g128"] = {
        "holdout": {"Wiki": 15.58, "Code": 2.62, "Media": 9.10},
        "bpw": 4.25
    }
    save_data(results)
    
    # 2. Quantizacao das 72 Matrizes FFN com H5-A 4.05 + Hessiana
    log("\n[2/3] Quantizando 72 matrizes FFN com H5-A 4.05 + Compensacao Hessiana...")
    matrix_metrics = {}
    t0_quant = time.time()
    
    for i in range(24):
        layer = model.model.layers[i]
        t_layer0 = time.time()
        
        # 1. Preparar Hessianas da camada i
        H_mlp = hess_dict[f"l{i}.mlp_in"].float()
        H_down = hess_dict[f"l{i}.down_in"].float()
        
        # Fator Cholesky Inverso compartilhado para gate e up
        U_mlp, dead_mlp = gptq_prep(H_mlp)
        U_down, dead_down = gptq_prep(H_down)
        
        # Quantizar gate_proj
        W_gate = layer.mlp.gate_proj.weight.data.clone()
        Q_gate = quantize_matrix_h5a_4bpw_hessian(W_gate, U_factor=U_mlp, dead_mask=dead_mlp, seed=42+i)
        matrix_metrics[f"l{i}.gate"] = calc_matrix_metrics(W_gate, Q_gate, H_mlp)
        layer.mlp.gate_proj.weight.data.copy_(Q_gate)
        
        # Quantizar up_proj
        W_up = layer.mlp.up_proj.weight.data.clone()
        Q_up = quantize_matrix_h5a_4bpw_hessian(W_up, U_factor=U_mlp, dead_mask=dead_mlp, seed=142+i)
        matrix_metrics[f"l{i}.up"] = calc_matrix_metrics(W_up, Q_up, H_mlp)
        layer.mlp.up_proj.weight.data.copy_(Q_up)
        
        # Quantizar down_proj
        W_down = layer.mlp.down_proj.weight.data.clone()
        Q_down = quantize_matrix_h5a_4bpw_hessian(W_down, U_factor=U_down, dead_mask=dead_down, seed=242+i)
        matrix_metrics[f"l{i}.down"] = calc_matrix_metrics(W_down, Q_down, H_down)
        layer.mlp.down_proj.weight.data.copy_(Q_down)
        
        t_l = time.time() - t_layer0
        rel_func_avg = (matrix_metrics[f"l{i}.gate"]["rel_func"] +
                        matrix_metrics[f"l{i}.up"]["rel_func"] +
                        matrix_metrics[f"l{i}.down"]["rel_func"]) / 3.0
        log(f"  Camada FFN {i:>2}/23 concluida ({t_l:.1f}s | RelFunc medio: {rel_func_avg:.5f} | Total: {time.time()-t0_quant:.1f}s)")
        
    t_quant_total = time.time() - t0_quant
    log(f"\nQuantizacao das 72 matrizes FFN com Hessiana concluida em {t_quant_total:.1f}s!")
    
    # Salvar metricas das matrizes
    results["matrix_metrics"] = matrix_metrics
    avg_rel_func = sum(m["rel_func"] for m in matrix_metrics.values()) / len(matrix_metrics)
    avg_rel_fro = sum(m["rel_fro"] for m in matrix_metrics.values()) / len(matrix_metrics)
    avg_mse = sum(m["mse"] for m in matrix_metrics.values()) / len(matrix_metrics)
    results["matrix_aggregates"] = {
        "avg_rel_func": avg_rel_func,
        "avg_rel_fro": avg_rel_fro,
        "avg_mse": avg_mse,
        "quant_time_s": t_quant_total
    }
    save_data(results)
    
    # 3. Avaliacao Completa
    log("\n[3/3] Avaliando modelo quantizado H5-A 4.05 + Hessiana...")
    log("Avaliando Benchmark Original (4 dominios)...")
    t0_eval_orig = time.time()
    ppl_orig = eval_original(model)
    log(f"-> H5-A 4.05 + Hessiana Original: {ppl_orig} ({time.time()-t0_eval_orig:.1f}s)")
    
    log("\nAvaliando Holdout Real (Wiki + Code, bs=1)...")
    t0_eval_hold = time.time()
    ppl_hold = eval_holdout_fast(model, tokens)
    log(f"-> H5-A 4.05 + Hessiana Holdout:  {ppl_hold} ({time.time()-t0_eval_hold:.1f}s)")
    
    results["H5A_405_Hessian"] = {
        "original": ppl_orig,
        "holdout": ppl_hold,
        "bpw": 4.05
    }
    save_data(results)
    
    log("\n" + "=" * 90)
    log("EXPERIMENTO 2 CONCLUIDO COM SUCESSO!")
    log(f"Resultados finais gravados em: {OUT_FILE}")
    log("=" * 90)

if __name__ == "__main__":
    main()
