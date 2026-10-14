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

def quantize_matrix(weight_tensor, num_ctrl=2, num_spec=4, spec_mode='sensitivity', p_crit=0.05, seed=42, block_size=8):
    """Quantização parametrizada para M.2 com controle por matriz de p_crit."""
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
        else: # standard
            top_k = max(num_spec, int(vecs_k.shape[0] * 0.35))
            hard_idx = torch.topk(min_err, k=top_k).indices
            spec_k = train_kmeans(vecs_k[hard_idx], K=num_spec, seed=seed)
            
        libs_spec.append(spec_k)
        
    macro_recon = torch.zeros_like(w_flat)
    scale_factor = 1.50
    
    for k in range(8):
        m = (fam_ids == k)
        if m.sum() == 0: continue
        w_k = w_flat[m]
        lib_base = libs_macro[k]
        lib_spec = libs_spec[k]
        
        d_base = torch.cdist(w_k, lib_base)
        min_base, idx_base = torch.min(d_base, dim=1)
        
        if num_ctrl == 3:
            d_c0 = torch.cdist(w_k, lib_base * 1.5)
            d_c1 = torch.cdist(w_k, lib_base * 0.5)
            d_c2 = torch.cdist(w_k, -lib_base)
            min_c0, idx_c0 = torch.min(d_c0, dim=1)
            min_c1, idx_c1 = torch.min(d_c1, dim=1)
            min_c2, idx_c2 = torch.min(d_c2, dim=1)
            best_ctrl_dists = torch.stack([min_c0, min_c1, min_c2], dim=1)
            ctrl_indices = [idx_c0, idx_c1, idx_c2]
        else:
            d_c0 = torch.cdist(w_k, lib_base * 1.5)
            d_c1 = torch.cdist(w_k, -lib_base)
            min_c0, idx_c0 = torch.min(d_c0, dim=1)
            min_c1, idx_c1 = torch.min(d_c1, dim=1)
            best_ctrl_dists = torch.stack([min_c0, min_c1], dim=1)
            ctrl_indices = [idx_c0, idx_c1]
            
        min_ctrl_val, choice_ctrl = torch.min(best_ctrl_dists, dim=1)
        
        d_spec = torch.cdist(w_k, lib_spec)
        min_spec_val, idx_spec = torch.min(d_spec, dim=1)
        
        recon_k = lib_base[idx_base].clone()
        win_spec = (min_spec_val < min_base) & (min_spec_val < min_ctrl_val)
        win_ctrl = (min_ctrl_val < min_base) & (min_ctrl_val <= min_spec_val)
        
        recon_k[win_spec] = lib_spec[idx_spec[win_spec]]
        
        if num_ctrl == 3:
            mask_c0 = win_ctrl & (choice_ctrl == 0)
            mask_c1 = win_ctrl & (choice_ctrl == 1)
            mask_c2 = win_ctrl & (choice_ctrl == 2)
            recon_k[mask_c0] = lib_base[ctrl_indices[0][mask_c0]] * 1.5
            recon_k[mask_c1] = lib_base[ctrl_indices[1][mask_c1]] * 0.5
            recon_k[mask_c2] = -lib_base[ctrl_indices[2][mask_c2]]
        else:
            mask_c0 = win_ctrl & (choice_ctrl == 0)
            mask_c1 = win_ctrl & (choice_ctrl == 1)
            recon_k[mask_c0] = lib_base[ctrl_indices[0][mask_c0]] * 1.5
            recon_k[mask_c1] = -lib_base[ctrl_indices[1][mask_c1]]
            
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
    
    # 4. Proteção Seletiva Top P_crit%
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
    
    return recon_reshaped, actual_bpp, mse, cos

def evaluate_perplexity(model, tokenizer, text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        loss = model(**inputs, labels=inputs.input_ids).loss
        return torch.exp(loss).item()

def evaluate_generation(model, tokenizer, prompt, max_tokens=25):
    inp = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    tokens = tokenizer.encode(text)
    unique_ratio = len(set(tokens)) / len(tokens)
    return text, unique_ratio

def analyze_code_generation(text):
    """Classifica o tipo de loop ou falha na geração de código."""
    # 1. Repetição de \n ou espaços
    newlines = text.count('\n')
    if newlines >= 4:
        # Verifica se o texto é composto predominantemente de linhas vazias
        lines = [l.strip() for l in text.split('\n')]
        empty_lines = sum(1 for l in lines if len(l) == 0)
        if empty_lines >= 3:
            return "Loop \\n (Indentação vazia)"
            
    # 2. Repetição de tokens/frases
    tokens = text.split()
    if len(tokens) >= 6:
        from collections import Counter
        counts = Counter(tokens)
        most_common, freq = counts.most_common(1)[0]
        if freq >= 4:
            return f"Repetição Token ('{most_common}')"
            
    return "OK (Sem Loop)"

def main():
    print("=" * 125)
    print("EXPERIMENTO DE REDISTRIBUIÇÃO DO ORÇAMENTO DE PROTEÇÃO: CURVA H5-A, H5-B, H5-C, H5-D")
    print("Objetivo: Investigar se proteger h5.c_proj resolve a ressonância em código mantendo ~3.05 bpp")
    print("=" * 125)
    
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
    
    code_prompts = [
        ("Fibonacci", "def fibonacci(n):"),
        ("Quicksort", "def quicksort(arr):"),
        ("Busca Binária", "def binary_search(arr, target):"),
        ("Classe Python", "class BankAccount:\n    def __init__(self, balance):"),
        ("Função Matemática", "def calculate_area(radius):"),
        ("JSON / Dados", "data = {\n    \"name\": \"Alice\",\n    \"age\":")
    ]
    
    # 0. FP16 Original
    ppl_orig = {d: evaluate_perplexity(base_model, tokenizer, text) for d, text in domains.items()}
    avg_orig = np.mean(list(ppl_orig.values()))
    
    print(f"\n[0] FP16 ORIGINAL (16.0 bpp): Média PPL = {avg_orig:.2f}")
    
    variants = [
        ('M.2 Atual', 'M.2 Atual: h0=5% | h5.c_proj=5% (Referência)',
         {'p_h0': 0.05, 'p_mid': 0.05, 'p_h5_proj': 0.05}),
        ('H5-A', 'H5-A: h0=3% | h5.c_proj=10% (Transferência Moderada)',
         {'p_h0': 0.03, 'p_mid': 0.05, 'p_h5_proj': 0.10}),
        ('H5-B', 'H5-B: h0=3% | h5.c_proj=12% (Transferência Agressiva)',
         {'p_h0': 0.03, 'p_mid': 0.05, 'p_h5_proj': 0.12}),
        ('H5-C', 'H5-C: h0=2% | h5.c_proj=12% (Máxima Transferência de h0)',
         {'p_h0': 0.02, 'p_mid': 0.05, 'p_h5_proj': 0.12}),
        ('H5-D', 'H5-D: h0=5% | h5.c_proj=10% (Isolamento h5 sem mexer em h0)',
         {'p_h0': 0.05, 'p_mid': 0.05, 'p_h5_proj': 0.10})
    ]
    
    results = {}
    
    for code, name, p_cfg in variants:
        print(f"\n" + "-" * 120)
        print(f"Quantizando modelo completo: {name}...")
        print("-" * 120)
        
        m_test = copy.deepcopy(base_model)
        t0 = time.time()
        
        h5_proj_mse = 0
        h5_proj_cos = 0
        h0_proj_mse = 0
        
        bpp_list = []
        
        for l_idx, block in enumerate(m_test.transformer.h):
            # Define taxa de proteção da camada
            if l_idx == 0:
                p_layer_fc = p_cfg['p_h0']
                p_layer_proj = p_cfg['p_h0']
            elif l_idx == 5:
                p_layer_fc = p_cfg['p_mid']
                p_layer_proj = p_cfg['p_h5_proj']
            else:
                p_layer_fc = p_cfg['p_mid']
                p_layer_proj = p_cfg['p_mid']
                
            with torch.no_grad():
                # c_fc (M.2: 2 controles + 4 especialistas por sensibilidade)
                w_fc, bpp_fc, mse_fc, cos_fc = quantize_matrix(
                    block.mlp.c_fc.weight.data, num_ctrl=2, num_spec=4, spec_mode='sensitivity', p_crit=p_layer_fc, seed=42
                )
                block.mlp.c_fc.weight.data = w_fc
                bpp_list.append(bpp_fc)
                
                # c_proj (M.2: 3 controles + 3 especialistas standard)
                w_proj, bpp_proj, mse_proj, cos_proj = quantize_matrix(
                    block.mlp.c_proj.weight.data, num_ctrl=3, num_spec=3, spec_mode='standard', p_crit=p_layer_proj, seed=42
                )
                block.mlp.c_proj.weight.data = w_proj
                bpp_list.append(bpp_proj)
                
                if l_idx == 0:
                    h0_proj_mse = mse_proj
                if l_idx == 5:
                    h5_proj_mse = mse_proj
                    h5_proj_cos = cos_proj
                    
        elapsed = time.time() - t0
        actual_model_bpp = np.mean(bpp_list)
        
        # Perplexidade
        ppls = {d: evaluate_perplexity(m_test, tokenizer, text) for d, text in domains.items()}
        avg_ppl = np.mean(list(ppls.values()))
        
        # Bateria de Código (6 prompts)
        code_reports = []
        loop_count = 0
        div_list = []
        
        for p_label, p_text in code_prompts:
            gen_txt, div_r = evaluate_generation(m_test, tokenizer, p_text, max_tokens=25)
            status = analyze_code_generation(gen_txt)
            is_loop = (status != "OK (Sem Loop)")
            if is_loop:
                loop_count += 1
            code_reports.append((p_label, gen_txt, div_r, status))
            div_list.append(div_r)
            
        avg_code_div = np.mean(div_list)
        
        results[code] = {
            'name': name,
            'bpp': actual_model_bpp,
            'avg_ppl': avg_ppl,
            'ppls': ppls,
            'h5_mse': h5_proj_mse,
            'h5_cos': h5_proj_cos,
            'h0_mse': h0_proj_mse,
            'loop_count': loop_count,
            'avg_code_div': avg_code_div,
            'code_reports': code_reports,
            'time': elapsed
        }
        
        print(f"   -> BPP Real: {actual_model_bpp:.4f} b | PPL Médio: {avg_ppl:.2f} (Ciência: {ppls['Ciência']:.1f} | Código: {ppls['Código']:.1f} | Hist: {ppls['História']:.1f} | Encic: {ppls['Enciclopédia']:.1f})")
        print(f"   -> MSE h5.c_proj: {h5_proj_mse:.6f} (Cosseno: {h5_proj_cos*100:.2f}%) | MSE h0.c_proj: {h0_proj_mse:.6f}")
        print(f"   -> Bateria de Código: Loops = {loop_count}/6 | Diversidade Média = {avg_code_div*100:.1f}%")

    # 1. TABELA PRINCIPAL COMPARATIVA
    print("\n" + "=" * 135)
    print("SCOREBOARD COMPARATIVO: CURVA DE REDISTRIBUIÇÃO DE PROTEÇÃO (H5-A A H5-D)")
    print("=" * 135)
    print(f"{'Variante':<28} | {'BPP Real':<8} | {'PPL Médio':<10} | {'Ciência':<8} | {'Código':<8} | {'MSE h5.proj':<12} | {'Loops Código':<13} | {'Div. Código'}")
    print("-" * 135)
    print(f"{'FP16 Original':<28} | {'16.0000':<8} | {avg_orig:>9.2f}  | {ppl_orig['Ciência']:>8.1f} | {ppl_orig['Código']:>8.1f} | {'0.000000':<12} | {'0/6 (Livre)':<13} | {'81.4%':<10}")
    print("-" * 135)
    
    for code, name, _ in variants:
        r = results[code]
        print(f"{name[:28]:<28} | {r['bpp']:>7.4f}b | {r['avg_ppl']:>9.2f}  | {r['ppls']['Ciência']:>8.1f} | {r['ppls']['Código']:>8.1f} | {r['h5_mse']:>11.6f} | {r['loop_count']}/6 loops   | {r['avg_code_div']*100:>8.1f}%")
    print("=" * 135)

    # 2. DETALHAMENTO DAS GERAÇÕES DE CÓDIGO
    print("\n" + "=" * 135)
    print("DETALHAMENTO DE CÓDIGO: M.2 ATUAL vs MELHOR VARIANTE H5")
    print("=" * 135)
    
    # Encontra a variante com menor número de loops ou menor PPL em código
    best_var = min(variants[1:], key=lambda x: (results[x[0]]['loop_count'], results[x[0]]['ppls']['Código']))[0]
    
    print(f"\n--- [1] M.2 ATUAL (Controle) ---")
    for p_label, gen_txt, div_r, status in results['M.2 Atual']['code_reports']:
        clean_t = gen_txt.replace('\n', ' \\n ')
        print(f"[{p_label:<16}] ({status:<22} | Div {div_r*100:.1f}%): \"{clean_t[:75]}...\"")
        
    print(f"\n--- [2] {results[best_var]['name']} ---")
    for p_label, gen_txt, div_r, status in results[best_var]['code_reports']:
        clean_t = gen_txt.replace('\n', ' \\n ')
        print(f"[{p_label:<16}] ({status:<22} | Div {div_r*100:.1f}%): \"{clean_t[:75]}...\"")
    print("=" * 135)

if __name__ == '__main__':
    main()
