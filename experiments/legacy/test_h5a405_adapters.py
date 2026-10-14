"""
test_h5a405_adapters.py - Teste Direto e Rapido dos Adaptadores Low-Rank no H5-A 4.05

Foco Exclusivo na Prioridade #1:
Base: H5-A 4.05 (72 matrizes FFN quantizadas a 4.05 BPW com o algoritmo que deu 6.96)
Comparacao:
- H5-A 4.05 Controle (esperado 6.96 no benchmark original)
- H5-A 4.05 + R8
- H5-A 4.05 + R16
- H5-A 4.05 + R32

Mede nos 2 protocolos:
1. Benchmark Original (Ciencia, Codigo, Historia, Enciclopedia) -> compara direto com 6.96, 5.04 e 4.90
2. Holdout Real (Wikitext-2 + Codigo disjunto) -> prova se ha ganho real e generalizacao

Salva os resultados a cada rank concluido em h5a_405_adapters_results.json.
"""
import os, sys, time, math, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "modelo")
DATA = os.path.join(ROOT, "data")
OUT_FILE = os.path.join(ROOT, "h5a_405_adapters_results.json")

torch.set_num_threads(os.cpu_count() or 8)

def log(*args):
    print(*args, flush=True)

# -----------------------------------------------------------------------------
# 1. H5-A 4.05 BPW Oficial
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

# -----------------------------------------------------------------------------
# 2. Avaliacao Otimizada (bs=1, rapida e sem gargalo)
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
    """Executa com batch=1 para maxima velocidade no CPU sem gargalo da DeltaNet."""
    scores = {}
    with torch.no_grad():
        for name, key in [("Wiki", "eval_wiki"), ("Code", "eval_code")]:
            toks = tokens_dict[key][:n_seqs]
            total_nll = 0.0
            total_tokens = 0
            for i in range(len(toks)):
                x = toks[i:i+1] # [1, 512]
                logits = m(input_ids=x).logits
                loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                       x[:, 1:].reshape(-1), reduction="sum")
                total_nll += loss.item()
                total_tokens += x[:, 1:].numel()
            scores[name] = round(math.exp(total_nll / total_tokens), 2)
    scores["Media"] = round((scores["Wiki"] + scores["Code"]) / 2, 2)
    return scores

def save_data(res):
    with open(OUT_FILE, "w") as f:
        json.dump(res, f, indent=2)

# -----------------------------------------------------------------------------
# 3. Main
# -----------------------------------------------------------------------------
def main():
    log("=" * 90)
    log("EXPERIMENTO 1: ADAPTADORES LOW-RANK (R8, R16, R32) NO H5-A 4.05 BPW")
    log("=" * 90)
    
    tokens = torch.load(os.path.join(DATA, "tokens.pt"))
    results = {}
    
    log("\n[1/4] Carregando modelo BF16 base...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()
    
    # Avaliar BF16 Base
    log("Avaliando BF16 Base...")
    t0 = time.time()
    ppl_bf16_orig = eval_original(model)
    ppl_bf16_hold = eval_holdout_fast(model, tokens)
    log(f"-> BF16 Original Benchmark: {ppl_bf16_orig} ({time.time()-t0:.1f}s)")
    log(f"-> BF16 Holdout Real:       {ppl_bf16_hold}")
    results["BF16"] = {"original": ppl_bf16_orig, "holdout": ppl_bf16_hold}
    save_data(results)
    
    # Salvar pesos originais das 72 matrizes FFN
    orig_ffn = {}
    for i in range(24):
        l = model.model.layers[i]
        orig_ffn[f"l{i}.gate"] = l.mlp.gate_proj.weight.data.clone()
        orig_ffn[f"l{i}.up"] = l.mlp.up_proj.weight.data.clone()
        orig_ffn[f"l{i}.down"] = l.mlp.down_proj.weight.data.clone()
        
    # Coletar representacoes BF16 de calibracao
    log("\n[2/4] Coletando representacoes BF16 de calibracao (16 seqs x 512)...")
    calib_seqs = tokens["calib_wiki"][:16]
    h_bf16_list = []
    with torch.no_grad():
        for i in range(len(calib_seqs)):
            out = model(calib_seqs[i:i+1], output_hidden_states=True)
            h_bf16_list.append(out.hidden_states[-1][0].float())
    H_BF16 = torch.cat(h_bf16_list, dim=0) # [N, 2048]
    log(f"-> H_BF16 coletado: {H_BF16.shape}")
    
    # Quantizar as 72 matrizes FFN com H5-A 4.05 BPW
    log("\n[3/4] Quantizando 72 matrizes FFN com H5-A 4.05 BPW...")
    t0_quant = time.time()
    for i in range(24):
        l = model.model.layers[i]
        l.mlp.gate_proj.weight.data.copy_(quantize_matrix_h5a_4bpw(orig_ffn[f"l{i}.gate"], seed=42+i))
        l.mlp.up_proj.weight.data.copy_(quantize_matrix_h5a_4bpw(orig_ffn[f"l{i}.up"], seed=42+i))
        l.mlp.down_proj.weight.data.copy_(quantize_matrix_h5a_4bpw(orig_ffn[f"l{i}.down"], seed=42+i))
        if i % 6 == 0 or i == 23:
            log(f"  Camada FFN {i:>2}/23 concluida ({time.time()-t0_quant:.1f}s)")
            
    log(f"Quantizacao H5-A 4.05 concluida em {time.time()-t0_quant:.1f}s!")
    
    # Avaliar H5-A 4.05 Controle (Sem Adaptador)
    log("\nAvaliando H5-A 4.05 Controle (sem adaptador)...")
    ppl_h5a_orig = eval_original(model)
    ppl_h5a_hold = eval_holdout_fast(model, tokens)
    log(f"-> H5-A 4.05 Controle Original: {ppl_h5a_orig} (Esperado: 6.96)")
    log(f"-> H5-A 4.05 Controle Holdout:  {ppl_h5a_hold}")
    results["H5A_405_Controle"] = {"original": ppl_h5a_orig, "holdout": ppl_h5a_hold}
    save_data(results)
    
    # Coletar representacoes H_H5A na mesma base de calibracao
    log("\nColetando representacoes H_H5A...")
    h_h5a_list = []
    with torch.no_grad():
        for i in range(len(calib_seqs)):
            out = model(calib_seqs[i:i+1], output_hidden_states=True)
            h_h5a_list.append(out.hidden_states[-1][0].float())
    H_H5A = torch.cat(h_h5a_list, dim=0) # [N, 2048]
    
    # =========================================================================
    # Treinar e Avaliar Adaptadores Low-Rank R8, R16, R32
    # =========================================================================
    log("\n[4/4] Treinando e avaliando adaptadores Low-Rank R8, R16, R32...")
    N_tokens, D_hidden = H_BF16.shape
    
    for r in [8, 16, 32]:
        log(f"\n==================================================")
        log(f"--- Treinando Adaptador Low-Rank R{r} no H5-A 4.05 ---")
        log(f"==================================================")
        torch.manual_seed(42)
        B = nn.Parameter(torch.randn(D_hidden, r) * (0.01 / math.sqrt(D_hidden)))
        A = nn.Parameter(torch.zeros(D_hidden, r))
        
        optimizer = torch.optim.AdamW([A, B], lr=2e-3, weight_decay=1e-4)
        
        t0_train = time.time()
        batch_size = 512
        for epoch in range(12):
            perm = torch.randperm(N_tokens)
            for idx in range(0, N_tokens, batch_size):
                b_idx = perm[idx:idx+batch_size]
                h_c = H_H5A[b_idx]
                h_t = H_BF16[b_idx]
                h_p = h_c + (h_c @ B) @ A.T
                loss = F.mse_loss(h_p, h_t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        log(f"R{r} treinado em {time.time()-t0_train:.1f}s | MSE Final: {loss.item():.6f}")
        
        # Instalar via hook no lm_head
        A_fixed = A.detach().clone().to(torch.bfloat16)
        B_fixed = B.detach().clone().to(torch.bfloat16)
        
        def make_hook(A_mat, B_mat):
            def hook_fn(module, args):
                h = args[0]
                delta = (h @ B_mat) @ A_mat.T
                return (h + delta,)
            return hook_fn
            
        handle = model.lm_head.register_forward_pre_hook(make_hook(A_fixed, B_fixed))
        
        # Avaliacao completa
        t0_ev = time.time()
        ppl_r_orig = eval_original(model)
        ppl_r_hold = eval_holdout_fast(model, tokens)
        log(f"-> H5-A 4.05 + R{r} Original: {ppl_r_orig} ({time.time()-t0_ev:.1f}s)")
        log(f"-> H5-A 4.05 + R{r} Holdout:  {ppl_r_hold}")
        
        results[f"H5A_405_R{r}"] = {"original": ppl_r_orig, "holdout": ppl_r_hold}
        save_data(results)
        
        handle.remove()
        
    log("\n" + "=" * 90)
    log("TODOS OS TESTES CONCLUIDOS COM SUCESSO!")
    log(f"Resultados finais salvos em: {OUT_FILE}")
    log("=" * 90)

if __name__ == "__main__":
    main()
