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

def quantize_fc_spec4_experiment(weight_tensor, mode='M2_current', p_crit=0.05, seed=42, block_size=8):
    """
    mode:
      - 'M2_current': 2 controles + K=4 especialistas (usa todos os 4 no encode)
      - 'M2_D_forced': 2 controles + K=4 especialistas, onde o 4o especialista é re-ancorado nos resíduos não capturados
      - 'M2_E_k3': 2 controles + K=3 especialistas treinados diretamente (estado 4 vago)
      - 'M2_F_k4_drop': K=4 treinado, mas remove o 4o especialista após o treino (encode usa apenas 1, 2, 3)
    """
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
        d_base = torch.cdist(vecs_k, libs_macro[k])
        min_err, _ = torch.min(d_base, dim=1)
        norms = torch.norm(vecs_k, dim=1)
        crit_score = (norms + 1e-5) * (min_err ** 2)
        top_k = max(4, int(vecs_k.shape[0] * 0.35))
        crit_idx = torch.topk(crit_score, k=top_k).indices
        crit_vecs = vecs_k[crit_idx]
        
        if mode == 'M2_E_k3':
            # Treina exatamente K=3
            spec_k = train_kmeans(crit_vecs, K=3, seed=seed)
        elif mode == 'M2_F_k4_drop':
            # Treina K=4, mas descarta o 4o centróide!
            spec_k_all = train_kmeans(crit_vecs, K=4, seed=seed)
            spec_k = spec_k_all[:3] # mantém apenas os 3 primeiros
        elif mode == 'M2_D_forced':
            # Treina K=3, acha resíduos dos blocos críticos não bem cobertos e treina o 4o centroide neles
            spec_k_3 = train_kmeans(crit_vecs, K=3, seed=seed)
            d_spec3 = torch.cdist(crit_vecs, spec_k_3)
            err_spec3 = torch.min(d_spec3, dim=1).values
            top_hard_idx = torch.topk(err_spec3, k=max(1, crit_vecs.shape[0] // 5)).indices
            spec_4th = crit_vecs[top_hard_idx].mean(dim=0, keepdim=True)
            spec_k = torch.cat([spec_k_3, spec_4th], dim=0)
        else: # 'M2_current'
            spec_k = train_kmeans(crit_vecs, K=4, seed=seed)
            
        libs_spec.append(spec_k)
        
    macro_recon = torch.zeros_like(w_flat)
    num_spec = libs_spec[0].shape[0]
    stats = {
        'atom_usage': np.zeros(26, dtype=int),
        'ctrl_usage': np.zeros(2, dtype=int),
        'spec_usage': np.zeros(num_spec, dtype=int)
    }
    
    scale_factor = 1.50
    for k in range(8):
        m = (fam_ids == k)
        if m.sum() == 0: continue
        w_k = w_flat[m]
        lib_base = libs_macro[k]
        lib_spec = libs_spec[k]
        
        d_base = torch.cdist(w_k, lib_base)
        min_base, idx_base = torch.min(d_base, dim=1)
        
        d_scale = torch.cdist(w_k, lib_base * scale_factor)
        min_scale, idx_scale = torch.min(d_scale, dim=1)
        
        d_neg = torch.cdist(w_k, -lib_base)
        min_neg, idx_neg = torch.min(d_neg, dim=1)
        
        best_ctrl_dist, ctrl_choice = torch.min(torch.stack([min_scale, min_neg], dim=1), dim=1)
        
        d_spec = torch.cdist(w_k, lib_spec)
        min_spec_val, idx_spec = torch.min(d_spec, dim=1)
        
        recon_k = lib_base[idx_base].clone()
        win_spec = (min_spec_val < min_base) & (min_spec_val < best_ctrl_dist)
        win_ctrl = (best_ctrl_dist < min_base) & (best_ctrl_dist <= min_spec_val)
        win_base = (~win_spec) & (~win_ctrl)
        
        recon_k[win_spec] = lib_spec[idx_spec[win_spec]]
        
        mask_c0 = win_ctrl & (ctrl_choice == 0)
        mask_c1 = win_ctrl & (ctrl_choice == 1)
        recon_k[mask_c0] = lib_base[idx_scale[mask_c0]] * scale_factor
        recon_k[mask_c1] = -lib_base[idx_neg[mask_c1]]
        
        for c_i in range(2):
            stats['ctrl_usage'][c_i] += (win_ctrl & (ctrl_choice == c_i)).sum().item()
        for s_i in range(num_spec):
            stats['spec_usage'][s_i] += (win_spec & (idx_spec == s_i)).sum().item()
        for a_i in range(26):
            stats['atom_usage'][a_i] += (win_base & (idx_base == a_i)).sum().item()
            
        macro_recon[m] = recon_k
        
    # Resíduo 1 (Base 2.0 bpp)
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
            
    # Resíduo 2 (Base 3.0 bpp)
    res_base = w_flat - recon_base
    lib_r2 = train_kmeans(res_base[:sample_n], K=26, seed=seed)
    d_r2 = torch.cdist(res_base, lib_r2)
    recon_final = recon_base + lib_r2[torch.argmin(d_r2, dim=1)]
    
    # Proteção dos top 5% blocos críticos (3.05 bpp)
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
    
    return recon_reshaped, actual_bpp, stats, mse, cos

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

def main():
    print("=" * 125)
    print("EXPERIMENTO PRIORITÁRIO: INVESTIGAÇÃO MECANÍSTICA DO SPEC 4 & BATERIA COMPLETA DE CÓDIGO")
    print("Testando Hipóteses A e B | 12 Camadas MLP | 3.05 bpp | DistilGPT-2")
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
    
    from refine_model_c import quantize_layer_c_refined
    
    experiments = [
        ('M.2 Atual', 'M.2 Atual: c_fc=(2+4-D) | c_proj=(3+3) [Referência]',
         'M2_current'),
        ('M.2-D', 'M.2-D: c_fc=(2+4-D forçado em resíduos livres) | c_proj=(3+3)',
         'M2_D_forced'),
        ('M.2-E', 'M.2-E: c_fc=(2+3-D direto, 4o estado livre) | c_proj=(3+3)',
         'M2_E_k3'),
        ('M.2-F', 'M.2-F: c_fc=(treina K=4, remove Spec 4 após treino) | c_proj=(3+3)',
         'M2_F_k4_drop')
    ]
    
    results = {}
    
    for code, name, mode_fc in experiments:
        print(f"\n" + "-" * 120)
        print(f"Executando {name}...")
        print("-" * 120)
        
        m_test = copy.deepcopy(base_model)
        t0 = time.time()
        
        spec_uses = []
        
        for l_idx, block in enumerate(m_test.transformer.h):
            with torch.no_grad():
                # c_fc com a variante
                w_fc, _, s_fc, _, _ = quantize_fc_spec4_experiment(block.mlp.c_fc.weight.data, mode=mode_fc, p_crit=0.05, seed=42)
                block.mlp.c_fc.weight.data = w_fc
                spec_uses.append(s_fc['spec_usage'])
                
                # c_proj padrão (3+3)
                w_proj, _, _ = quantize_layer_c_refined(block.mlp.c_proj.weight.data, variant='C3_error_subfamilies', p_crit=0.05, seed=42)
                block.mlp.c_proj.weight.data = w_proj
                
        elapsed = time.time() - t0
        
        # Perplexidade
        ppls = {d: evaluate_perplexity(m_test, tokenizer, text) for d, text in domains.items()}
        avg_ppl = np.mean(list(ppls.values()))
        
        # Bateria de Código (6 prompts)
        code_results = []
        code_divs = []
        loop_count = 0
        
        for c_name, c_prompt in code_prompts:
            gen_c, div_c = evaluate_generation(m_test, tokenizer, c_prompt, max_tokens=22)
            code_results.append((c_name, gen_c, div_c))
            code_divs.append(div_c)
            # Detecta repetição óbvia
            if div_c < 0.45 or ("fibonacci(n):" in gen_c and gen_c.count("fibonacci(n):") > 2):
                loop_count += 1
                
        avg_code_div = np.mean(code_divs)
        
        # Distribuição agregada de uso dos especialistas no c_fc
        agg_spec = np.sum(spec_uses, axis=0)
        tot_spec_calls = np.sum(agg_spec)
        spec_dist_pct = [(s / tot_spec_calls) * 100 if tot_spec_calls > 0 else 0 for s in agg_spec]
        
        results[code] = {
            'name': name,
            'avg_ppl': avg_ppl,
            'ppls': ppls,
            'spec_dist': spec_dist_pct,
            'avg_code_div': avg_code_div,
            'loop_count': loop_count,
            'code_results': code_results,
            'time': elapsed
        }
        
        dist_str = " | ".join([f"S{i+1}: {pct:.1f}%" for i, pct in enumerate(spec_dist_pct)])
        print(f"   -> Média PPL: {avg_ppl:.2f} (Ciência: {ppls['Ciência']:.1f} | Código: {ppls['Código']:.1f} | Hist: {ppls['História']:.1f} | Encic: {ppls['Enciclopédia']:.1f})")
        print(f"   -> Uso Relativo dos Especialistas no c_fc: [{dist_str}]")
        print(f"   -> Bateria de Código: Diversidade Média = {avg_code_div*100:.1f}% | Loops Detectados = {loop_count}/6")

    print("\n" + "=" * 135)
    print("SCOREBOARD COMPARATIVO FINAL: ABLAÇÃO DO SPEC 4 (HIPÓTESES A vs B) A 3.05 BPP")
    print("=" * 135)
    print(f"{'Variante':<32} | {'PPL Médio':<10} | {'Ciência':<8} | {'Código':<8} | {'Uso Spec 4':<12} | {'Div. Código':<12} | {'Loops Código'}")
    print("-" * 135)
    print(f"{'FP16 Original (16.0 bpp)':<32} | {avg_orig:>9.2f}  | {ppl_orig['Ciência']:>8.1f} | {ppl_orig['Código']:>8.1f} | {'-':<12} | {'81.4%':<12} | {'0/6 (Livre)'}")
    print("-" * 135)
    
    for code, name, _ in experiments:
        r = results[code]
        s4_str = f"{r['spec_dist'][3]:.1f}%" if len(r['spec_dist']) > 3 else "N/A (K=3)"
        print(f"{name[:32]:<32} | {r['avg_ppl']:>9.2f}  | {r['ppls']['Ciência']:>8.1f} | {r['ppls']['Código']:>8.1f} | {s4_str:>10}  | {r['avg_code_div']*100:>10.1f}% | {r['loop_count']}/6 loops")
    print("=" * 135)

    # Detalhar as gerações de código para o M.2-F e M.2 Atual
    print("\n" + "=" * 135)
    print("DETALHAMENTO DA BATERIA DE CÓDIGO (6 PROMPTS)")
    print("=" * 135)
    for c_name, gen_c, div_c in results['M.2-F']['code_results']:
        clean_text = gen_c.replace('\n', ' \\n ')
        print(f"[{c_name:<16}] (Div: {div_c*100:.1f}%): \"{clean_text[:85]}...\"")
    print("=" * 135)

if __name__ == '__main__':
    main()
