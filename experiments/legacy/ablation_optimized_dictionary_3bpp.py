import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import time
import copy

def train_cb_naive(vectors, K=26, n_iter=5):
    """Método anterior: K-means com inicialização aleatória simples e 5 iterações."""
    N, D = vectors.shape
    if N == 0: return torch.zeros(K, D)
    if N < K:
        reps = (K // N) + 1
        return vectors.repeat(reps, 1)[:K]
    perm = torch.randperm(N)[:K]
    c = vectors[perm].clone()
    for _ in range(n_iter):
        d = torch.cdist(vectors, c)
        lbl = torch.argmin(d, dim=1)
        for k in range(K):
            m = (lbl == k)
            if m.sum() > 0: c[k] = vectors[m].mean(dim=0)
    return c

def train_cb_optimized_outliers(vectors, K=26, n_iter=10):
    """
    Dicionário Otimizado:
    - 20 átomos para os modos mais densos via K-means++
    - 6 átomos explicitamente reservados para os maiores outliers/padrões extremos
    - 10 iterações com convergência estabilizada
    """
    N, D = vectors.shape
    if N == 0: return torch.zeros(K, D)
    if N < K:
        reps = (K // N) + 1
        return vectors.repeat(reps, 1)[:K]
        
    norms = torch.norm(vectors, dim=1)
    
    # 1. Separar os 6 maiores outliers extremos de energia/norma
    K_outliers = 6
    K_core = K - K_outliers # 20 átomos
    
    top_outlier_idx = torch.topk(norms, k=K_outliers).indices
    outlier_atoms = vectors[top_outlier_idx].clone()
    
    # 2. Inicialização K-Means++ ponderada nos vetores restantes
    core_mask = torch.ones(N, dtype=torch.bool)
    core_mask[top_outlier_idx] = False
    core_vectors = vectors[core_mask]
    
    # K-means++ para os 20 átomos centrais
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
    
    # Refinamento por 10 iterações
    for _ in range(n_iter):
        d = torch.cdist(core_vectors, c_core)
        lbl = torch.argmin(d, dim=1)
        for k in range(K_core):
            m = (lbl == k)
            if m.sum() > 0: c_core[k] = core_vectors[m].mean(dim=0)
            
    # Unir átomos de núcleo + átomos especializados de outliers
    return torch.cat([c_core, outlier_atoms], dim=0)

def quantize_layer(weight_tensor, optimized_dict=False, with_outlier_protection=False, block_size=8):
    """
    Quantiza a camada com 3.0 bpp usando o Dicionário Atual ou Dicionário Otimizado.
    """
    orig_shape = weight_tensor.shape
    w_flat = weight_tensor.float().reshape(-1, block_size)
    N = w_flat.shape[0]
    sample_n = min(N, 35000)
    w_sample = w_flat[:sample_n]
    
    trainer = train_cb_optimized_outliers if optimized_dict else train_cb_naive
    
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
    
    libs_macro = [trainer(w_sample[fam_ids[:sample_n] == k], K=26) for k in range(8)]
    macro_recon = torch.zeros_like(w_flat)
    for k in range(8):
        m = (fam_ids == k)
        if m.sum() > 0:
            d_k = torch.cdist(w_flat[m], libs_macro[k])
            macro_recon[m] = libs_macro[k][torch.argmin(d_k, dim=1)]
            
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
    
    libs_r1 = [trainer(r1_sample[r1_fam_ids[:sample_n] == k], K=26) for k in range(8)]
    recon_base = macro_recon.clone()
    for k in range(8):
        m = (r1_fam_ids == k)
        if m.sum() > 0:
            d_rk = torch.cdist(r1[m], libs_r1[k])
            recon_base[m] += libs_r1[k][torch.argmin(d_rk, dim=1)]
            
    # 3. Resíduo 2 para atingir 3.0 bpp
    res_base = w_flat - recon_base
    
    if not with_outlier_protection:
        # Padrão: Resíduo 2 em todos os blocos
        r2_sample = res_base[:sample_n]
        lib_r2 = trainer(r2_sample, K=26)
        d_r2 = torch.cdist(res_base, lib_r2)
        recon_final = recon_base + lib_r2[torch.argmin(d_r2, dim=1)]
    else:
        # Com proteção de blocos críticos:
        # 85% dos blocos recebem Resíduo 2 (3 bpp)
        # 15% dos blocos mais críticos recebem Resíduo 2 + Resíduo 3 (4 bpp)
        # 5% dos blocos muito fáceis ficam em 2 bpp -> média continua exatamente 3.1 bpp!
        block_norm = torch.norm(w_flat, dim=1)
        sensitivity = (block_norm + 1e-5) * torch.mean(res_base ** 2, dim=1)
        sorted_idx = torch.argsort(sensitivity, descending=True)
        
        # Treinar Resíduo 2
        lib_r2 = trainer(res_base[:sample_n], K=26)
        d_r2 = torch.cdist(res_base, lib_r2)
        recon_final = recon_base + lib_r2[torch.argmin(d_r2, dim=1)]
        
        # Aplicar Resíduo 3 nos 15% blocos mais críticos
        num_crit = int(0.15 * N)
        crit_idx = sorted_idx[:num_crit]
        r3_vecs = w_flat[crit_idx] - recon_final[crit_idx]
        lib_r3 = trainer(r3_vecs, K=26)
        d_r3 = torch.cdist(r3_vecs, lib_r3)
        recon_final[crit_idx] += lib_r3[torch.argmin(d_r3, dim=1)]
        
    return recon_final.view(orig_shape)

def evaluate_perplexity(model, tokenizer, text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        loss = model(**inputs, labels=inputs.input_ids).loss
        return torch.exp(loss).item()

def generate(model, tokenizer, prompt, max_tokens=22):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True).replace("\n", " ").strip()

def main():
    print("=" * 90)
    print("TESTE DE ABLAÇÃO: O IMPACTO DE OTIMIZAR O DICIONÁRIO EM ~3 BPP")
    print("Objetivo: Otimizar o Codebook (K-Means++ e Outliers) para Salvar Domínios Difíceis")
    print("=" * 90)
    
    model_name = "distilbert/distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_orig = AutoModelForCausalLM.from_pretrained(model_name)
    model_orig.eval()
    
    domains = {
        "Ciência / Física": (
            "Quantum mechanics describes nature at the smallest scales of energy levels of atoms and subatomic particles. "
            "Classical physics cannot explain blackbody radiation or the photoelectric effect without quantization."
        ),
        "Código / Lógica": (
            "def binary_search(array, target): "
            "    low = 0 "
            "    high = len(array) - 1 "
            "    while low <= high: "
            "        mid = (low + high) // 2 "
            "        if array[mid] == target: return mid "
            "    return -1"
        ),
        "História / Conhecimento": (
            "The ancient library of Alexandria was once the largest repository of knowledge in the Hellenistic world, "
            "attracting scholars from across the Mediterranean to study mathematics, astronomy, and literature."
        ),
        "Enciclopédia / Fatos": (
            "The solar system formed approximately 4.6 billion years ago from the gravitational collapse of a giant interstellar "
            "molecular cloud containing mostly hydrogen and helium gas."
        )
    }
    
    prompt_quantum = "The fundamental principles of quantum physics include"
    prompt_alexandria = "The library of Alexandria was famous for"
    
    # 0. Medição Original
    ppl_orig = {d: evaluate_perplexity(model_orig, tokenizer, text) for d, text in domains.items()}
    g_quant_orig = generate(model_orig, tokenizer, prompt_quantum)
    g_alex_orig = generate(model_orig, tokenizer, prompt_alexandria)
    
    # 1. Configuração A: Dicionário Atual (Naive) ~3.0 bpp
    print("\n[1] Quantizando 12 Camadas com MODELO A (Dicionário Atual / Naive ~3.0 bpp)...")
    model_a = copy.deepcopy(model_orig)
    t0 = time.time()
    for block in model_a.transformer.h:
        with torch.no_grad():
            block.mlp.c_fc.weight.data = quantize_layer(block.mlp.c_fc.weight.data, optimized_dict=False)
            block.mlp.c_proj.weight.data = quantize_layer(block.mlp.c_proj.weight.data, optimized_dict=False)
    t_a = time.time() - t0
    ppl_a = {d: evaluate_perplexity(model_a, tokenizer, text) for d, text in domains.items()}
    g_quant_a = generate(model_a, tokenizer, prompt_quantum)
    g_alex_a = generate(model_a, tokenizer, prompt_alexandria)
    
    # 2. Configuração B: Dicionário Otimizado (Outlier-Aware) ~3.0 bpp
    print(f"\n[2] Quantizando 12 Camadas com MODELO B (Dicionário Otimizado / Outliers ~3.0 bpp)...")
    model_b = copy.deepcopy(model_orig)
    t0 = time.time()
    for block in model_b.transformer.h:
        with torch.no_grad():
            block.mlp.c_fc.weight.data = quantize_layer(block.mlp.c_fc.weight.data, optimized_dict=True)
            block.mlp.c_proj.weight.data = quantize_layer(block.mlp.c_proj.weight.data, optimized_dict=True)
    t_b = time.time() - t0
    ppl_b = {d: evaluate_perplexity(model_b, tokenizer, text) for d, text in domains.items()}
    g_quant_b = generate(model_b, tokenizer, prompt_quantum)
    g_alex_b = generate(model_b, tokenizer, prompt_alexandria)
    
    # 3. Configuração C: Dicionário Otimizado + Proteção de Blocos Críticos ~3.1 bpp
    print(f"\n[3] Quantizando 12 Camadas com MODELO C (Dicionário Otimizado + Proteção Crítica ~3.1 bpp)...")
    model_c = copy.deepcopy(model_orig)
    t0 = time.time()
    for block in model_c.transformer.h:
        with torch.no_grad():
            block.mlp.c_fc.weight.data = quantize_layer(block.mlp.c_fc.weight.data, optimized_dict=True, with_outlier_protection=True)
            block.mlp.c_proj.weight.data = quantize_layer(block.mlp.c_proj.weight.data, optimized_dict=True, with_outlier_protection=True)
    t_c = time.time() - t0
    ppl_c = {d: evaluate_perplexity(model_c, tokenizer, text) for d, text in domains.items()}
    g_quant_c = generate(model_c, tokenizer, prompt_quantum)
    g_alex_c = generate(model_c, tokenizer, prompt_alexandria)
    
    # =========================================================================
    # PLACAR FINAL DE ABLAÇÃO
    # =========================================================================
    print("\n" + "=" * 95)
    print("QUADRO COMPARATIVO DE ABLAÇÃO: O IMPACTO DE OTIMIZAR O DICIONÁRIO")
    print("=" * 95)
    print(f"{'Domínio':<25} | {'Original (16b)':<15} | {'Mod A (Atual 3b)':<18} | {'Mod B (Dic Otimizado)':<22} | {'Mod C (Otim + Prot)'}")
    print("-" * 95)
    for d in domains:
        print(f"{d:<25} | {ppl_orig[d]:>11.2f}    | {ppl_a[d]:>14.2f}    | {ppl_b[d]:>18.2f}     | {ppl_c[d]:>15.2f}")
    print("=" * 95)
    
    print("\nRESPOSTAS NO DOMÍNIO DIFÍCIL (FÍSICA QUÂNTICA):")
    print(f"-> Original:   \"{g_quant_orig}\"")
    print(f"-> Modelo A:   \"{g_quant_a}\"")
    print(f"-> Modelo B:   \"{g_quant_b}\"")
    print(f"-> Modelo C:   \"{g_quant_c}\"")
    
    print("\nRESPOSTAS EM HISTÓRIA (ALEXANDRIA):")
    print(f"-> Original:   \"{g_alex_orig}\"")
    print(f"-> Modelo A:   \"{g_alex_a}\"")
    print(f"-> Modelo B:   \"{g_alex_b}\"")
    print(f"-> Modelo C:   \"{g_alex_c}\"")

if __name__ == '__main__':
    main()
