"""
exp1_h5a405_investigation.py - Investigacao Profunda do H5-A 4.05 BPW.

Executa de forma controlada e sem misturar protocolos:
1. Ablacao por Componente a 4.05 BPW:
   - BF16 Base (Controle)
   - Apenas FFN a 4.05 BPW (o teste de 6.96)
   - Apenas Self-Attention a 4.05 BPW
   - Apenas DeltaNet a 4.05 BPW
   - Apenas Embeddings a 4.05 BPW
2. Calibracao do H5-A 4.05 com Adaptadores Low-Rank:
   - H5-A 4.05 Base (Controle: 6.96)
   - H5-A 4.05 + R8
   - H5-A 4.05 + R16
   - H5-A 4.05 + R32
3. Metricas simultaneas:
   - Benchmark Original (Ciencia, Codigo, Historia, Enciclopedia) -> compara direto com 6.96, 5.04 e 4.90
   - Holdout Real (Wikitext-2 + Codigo disjunto) -> prova se ha generalizacao real
"""
import os, sys, time, math, json, gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "modelo")
DATA = os.path.join(ROOT, "data")
OUT_JSON = os.path.join(ROOT, "h5a_405_investigation_results.json")

torch.set_num_threads(os.cpu_count() or 8)

# -----------------------------------------------------------------------------
# 1. H5-A 4.05 BPW (Implementacao Oficial que gerou 6.96)
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
            
    recon_reshaped = recon_4.view(orig_shape).to(orig_dtype)
    return recon_reshaped

# -----------------------------------------------------------------------------
# 2. Avaliacao em Protocolos Rigorosos
# -----------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

original_benchmark_domains = {
    "Ciencia": "Quantum mechanics describes nature at the smallest scales of energy levels of atoms and subatomic particles.",
    "Codigo": "def binary_search(array, target): low = 0; high = len(array) - 1; while low <= high: mid = (low + high) // 2",
    "Historia": "The ancient library of Alexandria was once the largest repository of knowledge in the Hellenistic world.",
    "Enciclopedia": "The solar system formed approximately 4.6 billion years ago from the gravitational collapse of a molecular cloud."
}

def eval_original_benchmark(m):
    scores = {}
    with torch.no_grad():
        for d, txt in original_benchmark_domains.items():
            inputs = tokenizer(txt, return_tensors="pt")
            loss = m(**inputs, labels=inputs.input_ids).loss
            scores[d] = round(torch.exp(loss).item(), 2)
    scores["Media"] = round(sum(scores.values()) / len(scores), 2)
    return scores

def eval_holdout(m, tokens_dict, n_wiki=8, n_code=6):
    scores = {}
    with torch.no_grad():
        for name, key, n in [("Wiki_Holdout", "eval_wiki", n_wiki), ("Code_Holdout", "eval_code", n_code)]:
            x = tokens_dict[key][:n]
            logits = m(input_ids=x).logits
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                   x[:, 1:].reshape(-1), reduction="mean")
            scores[name] = round(math.exp(loss.item()), 2)
    scores["Media_Holdout"] = round((scores["Wiki_Holdout"] + scores["Code_Holdout"]) / 2, 2)
    return scores

# -----------------------------------------------------------------------------
# 3. Execucao Estruturada
# -----------------------------------------------------------------------------
def main():
    print("=" * 100)
    print("INVESTIGACAO PROFUNDA DO H5-A 4.05 BPW: ABLACAO + ADAPTADORES LOW-RANK")
    print("=" * 100)
    
    tokens = torch.load(os.path.join(DATA, "tokens.pt"))
    results = {}
    
    # Carregar modelo base BF16
    print("\n[Passo 1/4] Carregando modelo base BF16...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()
    
    # 1. Medir BF16 Original (Controle Absoluto)
    print("\n-> Avaliando BF16 Base...")
    ppl_bf16_orig = eval_original_benchmark(model)
    ppl_bf16_hold = eval_holdout(model, tokens)
    print(f"BF16 Original Benchmark: {ppl_bf16_orig}")
    print(f"BF16 Holdout Real:       {ppl_bf16_hold}")
    results["BF16"] = {"original": ppl_bf16_orig, "holdout": ppl_bf16_hold}
    
    # Guardar pesos originais de todos os componentes
    orig_ffn = {}
    orig_attn = {}
    orig_dnet = {}
    
    full_attn_layers = [3, 7, 11, 15, 19, 23]
    dnet_layers = [i for i in range(24) if i not in full_attn_layers]
    
    for i in range(24):
        l = model.model.layers[i]
        orig_ffn[f"l{i}.gate"] = l.mlp.gate_proj.weight.data.clone()
        orig_ffn[f"l{i}.up"] = l.mlp.up_proj.weight.data.clone()
        orig_ffn[f"l{i}.down"] = l.mlp.down_proj.weight.data.clone()
        
        if i in full_attn_layers:
            orig_attn[f"l{i}.q"] = l.self_attn.q_proj.weight.data.clone()
            orig_attn[f"l{i}.k"] = l.self_attn.k_proj.weight.data.clone()
            orig_attn[f"l{i}.v"] = l.self_attn.v_proj.weight.data.clone()
            orig_attn[f"l{i}.o"] = l.self_attn.o_proj.weight.data.clone()
        else:
            orig_dnet[f"l{i}.qkv"] = l.linear_attn.in_proj_qkv.weight.data.clone()
            orig_dnet[f"l{i}.z"] = l.linear_attn.in_proj_z.weight.data.clone()
            orig_dnet[f"l{i}.out"] = l.linear_attn.out_proj.weight.data.clone()
            
    orig_emb = model.model.embed_tokens.weight.data.clone()
    
    # =========================================================================
    # ABLACAO POR COMPONENTES A 4.05 BPW
    # =========================================================================
    print("\n" + "=" * 90)
    print("[Passo 2/4] ABLACAO POR COMPONENTE A 4.05 BPW")
    print("=" * 90)
    
    # A. Apenas FFN a 4.05 BPW (o teste oficial de 6.96)
    print("\n--- [Ablacao A] Quantizando 100% das FFNs com H5-A 4.05 ---")
    t0 = time.time()
    for i in range(24):
        l = model.model.layers[i]
        l.mlp.gate_proj.weight.data.copy_(quantize_matrix_h5a_4bpw(orig_ffn[f"l{i}.gate"], seed=42+i))
        l.mlp.up_proj.weight.data.copy_(quantize_matrix_h5a_4bpw(orig_ffn[f"l{i}.up"], seed=42+i))
        l.mlp.down_proj.weight.data.copy_(quantize_matrix_h5a_4bpw(orig_ffn[f"l{i}.down"], seed=42+i))
        if i % 6 == 0 or i == 23:
            print(f"  Camada FFN {i}/23 pronta ({time.time()-t0:.1f}s)")
            
    ppl_ffn_orig = eval_original_benchmark(model)
    ppl_ffn_hold = eval_holdout(model, tokens)
    print(f"-> H5-A 4.05 (FFN-Only) Original: {ppl_ffn_orig} (Esperado ~6.96)")
    print(f"-> H5-A 4.05 (FFN-Only) Holdout:  {ppl_ffn_hold}")
    results["H5A_405_FFN_Only"] = {"original": ppl_ffn_orig, "holdout": ppl_ffn_hold}
    
    # =========================================================================
    # EXPERIMENTO 1: CALIBRACAO DE ADAPTADORES R8, R16, R32 NO H5-A 4.05 FFN
    # =========================================================================
    print("\n" + "=" * 90)
    print("[Passo 3/4] EXPERIMENTO 1: ADAPTADORES LOW-RANK NO H5-A 4.05 FFN")
    print("Objetivo: Testar se R8, R16, R32 reduzem a PPL de 6.96 -> 5.x com generalizacao real")
    print("=" * 90)
    
    # Coletar representacoes BF16 vs H5-A 4.05 para calibracao
    # Usar calib_wiki e calib_code (amostras reais) + sequencias originais
    print("Coletando pares de calibracao...")
    calib_seqs = tokens["calib_wiki"][:16] # 16 sequencias de 512 tokens = 8192 tokens reais
    
    # Extrair H_H5A na saida
    h_h5a_list = []
    with torch.no_grad():
        for b in range(0, len(calib_seqs), 2):
            batch = calib_seqs[b:b+2]
            out = model(batch, output_hidden_states=True)
            h_h5a_list.append(out.hidden_states[-1].float().detach())
    H_H5A = torch.cat(h_h5a_list, dim=0).reshape(-1, 2048) # [N, 2048]
    
    # Restaurar FFN original temporariamente para capturar H_BF16 e Logits_BF16 perfeitos
    for i in range(24):
        model.model.layers[i].mlp.gate_proj.weight.data.copy_(orig_ffn[f"l{i}.gate"])
        model.model.layers[i].mlp.up_proj.weight.data.copy_(orig_ffn[f"l{i}.up"])
        model.model.layers[i].mlp.down_proj.weight.data.copy_(orig_ffn[f"l{i}.down"])
        
    h_bf16_list = []
    logits_bf16_list = []
    with torch.no_grad():
        for b in range(0, len(calib_seqs), 2):
            batch = calib_seqs[b:b+2]
            out = model(batch, output_hidden_states=True)
            h_bf16_list.append(out.hidden_states[-1].float().detach())
            logits_bf16_list.append(out.logits.float().detach())
    H_BF16 = torch.cat(h_bf16_list, dim=0).reshape(-1, 2048)
    
    # Recolocar H5-A 4.05 nas FFNs
    for i in range(24):
        l = model.model.layers[i]
        l.mlp.gate_proj.weight.data.copy_(quantize_matrix_h5a_4bpw(orig_ffn[f"l{i}.gate"], seed=42+i))
        l.mlp.up_proj.weight.data.copy_(quantize_matrix_h5a_4bpw(orig_ffn[f"l{i}.up"], seed=42+i))
        l.mlp.down_proj.weight.data.copy_(quantize_matrix_h5a_4bpw(orig_ffn[f"l{i}.down"], seed=42+i))
        
    W_head = model.lm_head.weight.data.clone().float() # [V, 2048]
    
    N_tokens, D_hidden = H_BF16.shape
    print(f"Calibracao: {N_tokens} tokens x {D_hidden} dim.")
    
    ranks = [8, 16, 32]
    for r in ranks:
        print(f"\n--- Treinando Adaptador Low-Rank R{r} no H5-A 4.05 ---")
        torch.manual_seed(42)
        B = nn.Parameter(torch.randn(D_hidden, r) * (0.01 / math.sqrt(D_hidden)))
        A = nn.Parameter(torch.zeros(D_hidden, r))
        
        optimizer = torch.optim.AdamW([A, B], lr=2e-3, weight_decay=1e-4)
        
        # Otimizar minimizando erro residual de hidden states ponderado:
        # min || (H_H5A + (H_H5A @ B) @ A.T) - H_BF16 ||^2
        t0_r = time.time()
        batch_size = 512
        for epoch in range(10):
            perm = torch.randperm(N_tokens)
            for idx in range(0, N_tokens, batch_size):
                b_idx = perm[idx:idx+batch_size]
                h_curr = H_H5A[b_idx]
                h_tgt = H_BF16[b_idx]
                
                delta = (h_curr @ B) @ A.T
                h_pred = h_curr + delta
                
                loss = F.mse_loss(h_pred, h_tgt)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        print(f"Adaptador R{r} treinado em {time.time()-t0_r:.1f}s | Loss final: {loss.item():.6f}")
        
        # Instalar adaptador via Hook no lm_head do modelo
        A_fixed = A.detach().clone().to(torch.bfloat16)
        B_fixed = B.detach().clone().to(torch.bfloat16)
        
        def make_hook(A_mat, B_mat):
            def hook_fn(module, args):
                h = args[0]
                delta = (h @ B_mat) @ A_mat.T
                return (h + delta,)
            return hook_fn
            
        handle = model.lm_head.register_forward_pre_hook(make_hook(A_fixed, B_fixed))
        
        ppl_r_orig = eval_original_benchmark(model)
        ppl_r_hold = eval_holdout(model, tokens)
        print(f"-> H5-A 4.05 + R{r} Original: {ppl_r_orig}")
        print(f"-> H5-A 4.05 + R{r} Holdout:  {ppl_r_hold}")
        results[f"H5A_405_R{r}"] = {"original": ppl_r_orig, "holdout": ppl_r_hold}
        
        handle.remove()
        
    # Restaurar FFN ao original para as proximas ablacoes
    for i in range(24):
        model.model.layers[i].mlp.gate_proj.weight.data.copy_(orig_ffn[f"l{i}.gate"])
        model.model.layers[i].mlp.up_proj.weight.data.copy_(orig_ffn[f"l{i}.up"])
        model.model.layers[i].mlp.down_proj.weight.data.copy_(orig_ffn[f"l{i}.down"])
        
    # =========================================================================
    # ABLACOES B, C, D: AVALIAR GARGALOS (ATTN, DNET, EMBEDDINGS)
    # =========================================================================
    print("\n" + "=" * 90)
    print("[Passo 4/4] ABLACAO DOS OUTROS COMPONENTES A 4.05 BPW")
    print("=" * 90)
    
    # B. Apenas Self-Attention a 4.05 BPW (camadas 3, 7, 11, 15, 19, 23)
    print("\n--- [Ablacao B] Apenas Self-Attention (Full Attn) a 4.05 BPW ---")
    for i in full_attn_layers:
        l = model.model.layers[i]
        l.self_attn.q_proj.weight.data.copy_(quantize_matrix_h5a_4bpw(orig_attn[f"l{i}.q"], seed=100+i))
        l.self_attn.k_proj.weight.data.copy_(quantize_matrix_h5a_4bpw(orig_attn[f"l{i}.k"], seed=100+i))
        l.self_attn.v_proj.weight.data.copy_(quantize_matrix_h5a_4bpw(orig_attn[f"l{i}.v"], seed=100+i))
        l.self_attn.o_proj.weight.data.copy_(quantize_matrix_h5a_4bpw(orig_attn[f"l{i}.o"], seed=100+i))
        
    ppl_attn_orig = eval_original_benchmark(model)
    ppl_attn_hold = eval_holdout(model, tokens)
    print(f"-> Apenas Self-Attention 4.05 Original: {ppl_attn_orig}")
    print(f"-> Apenas Self-Attention 4.05 Holdout:  {ppl_attn_hold}")
    results["Ablation_SelfAttn_405"] = {"original": ppl_attn_orig, "holdout": ppl_attn_hold}
    
    # Restaurar Self-Attention
    for i in full_attn_layers:
        l = model.model.layers[i]
        l.self_attn.q_proj.weight.data.copy_(orig_attn[f"l{i}.q"])
        l.self_attn.k_proj.weight.data.copy_(orig_attn[f"l{i}.k"])
        l.self_attn.v_proj.weight.data.copy_(orig_attn[f"l{i}.v"])
        l.self_attn.o_proj.weight.data.copy_(orig_attn[f"l{i}.o"])
        
    # C. Apenas DeltaNet (Linear Attention) a 4.05 BPW (18 camadas)
    print("\n--- [Ablacao C] Apenas DeltaNet (18 camadas) a 4.05 BPW ---")
    for i in dnet_layers:
        l = model.model.layers[i]
        l.linear_attn.in_proj_qkv.weight.data.copy_(quantize_matrix_h5a_4bpw(orig_dnet[f"l{i}.qkv"], seed=200+i))
        l.linear_attn.in_proj_z.weight.data.copy_(quantize_matrix_h5a_4bpw(orig_dnet[f"l{i}.z"], seed=200+i))
        l.linear_attn.out_proj.weight.data.copy_(quantize_matrix_h5a_4bpw(orig_dnet[f"l{i}.out"], seed=200+i))
        
    ppl_dnet_orig = eval_original_benchmark(model)
    ppl_dnet_hold = eval_holdout(model, tokens)
    print(f"-> Apenas DeltaNet 4.05 Original: {ppl_dnet_orig}")
    print(f"-> Apenas DeltaNet 4.05 Holdout:  {ppl_dnet_hold}")
    results["Ablation_DeltaNet_405"] = {"original": ppl_dnet_orig, "holdout": ppl_dnet_hold}
    
    # Restaurar DeltaNet
    for i in dnet_layers:
        l = model.model.layers[i]
        l.linear_attn.in_proj_qkv.weight.data.copy_(orig_dnet[f"l{i}.qkv"])
        l.linear_attn.in_proj_z.weight.data.copy_(orig_dnet[f"l{i}.z"])
        l.linear_attn.out_proj.weight.data.copy_(orig_dnet[f"l{i}.out"])
        
    # D. Apenas Embeddings a 4.05 BPW (corte sensivel 32k tokens BF16 + resto 4.05)
    print("\n--- [Ablacao D] Apenas Embeddings a 4.05 BPW ---")
    w_emb = model.model.embed_tokens.weight.data
    # Quantizar tokens acima de 32768
    total_tok = w_emb.shape[0]
    for st in range(32768, total_tok, 50000):
        en = min(st + 50000, total_tok)
        w_emb[st:en].copy_(quantize_matrix_h5a_4bpw(orig_emb[st:en], seed=300+st))
        
    ppl_emb_orig = eval_original_benchmark(model)
    ppl_emb_hold = eval_holdout(model, tokens)
    print(f"-> Apenas Embeddings 4.05 Original: {ppl_emb_orig}")
    print(f"-> Apenas Embeddings 4.05 Holdout:  {ppl_emb_hold}")
    results["Ablation_Embeddings_405"] = {"original": ppl_emb_orig, "holdout": ppl_emb_hold}
    
    # Salvar resultados consolidados
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados gravados em {OUT_JSON}")
    print("=" * 100)

if __name__ == "__main__":
    main()
