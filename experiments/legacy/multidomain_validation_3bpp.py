import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import time
import copy

def train_cb(vectors, K=26, n_iter=5):
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
            mask = (lbl == k)
            if mask.sum() > 0: c[k] = vectors[mask].mean(dim=0)
    return c

def quantize_layer_3bpp_adaptive(weight_tensor, num_families=8, atoms_per_family=26, block_size=8):
    orig_shape = weight_tensor.shape
    w_flat = weight_tensor.float().reshape(-1, block_size)
    N = w_flat.shape[0]
    sample_n = min(N, 35000)
    w_sample = w_flat[:sample_n]
    
    # 1. 8 Famílias Macro
    perm = torch.randperm(sample_n)[:num_families]
    fam_centers = w_sample[perm].clone()
    for _ in range(5):
        d = torch.cdist(w_sample, fam_centers)
        lbl = torch.argmin(d, dim=1)
        for k in range(num_families):
            m = (lbl == k)
            if m.sum() > 0: fam_centers[k] = w_sample[m].mean(dim=0)
    fam_ids = torch.argmin(torch.cdist(w_flat, fam_centers), dim=1)
    
    libs_macro = [train_cb(w_sample[fam_ids[:sample_n] == k], K=atoms_per_family, n_iter=5) for k in range(num_families)]
    macro_recon = torch.zeros_like(w_flat)
    for k in range(num_families):
        m = (fam_ids == k)
        if m.sum() > 0:
            d_k = torch.cdist(w_flat[m], libs_macro[k])
            macro_recon[m] = libs_macro[k][torch.argmin(d_k, dim=1)]
            
    # 2. Resíduo 1 (Base 2.0 bpp para todos)
    r1 = w_flat - macro_recon
    r1_sample = r1[:sample_n]
    perm_r1 = torch.randperm(sample_n)[:num_families]
    r1_centers = r1_sample[perm_r1].clone()
    for _ in range(5):
        d = torch.cdist(r1_sample, r1_centers)
        lbl = torch.argmin(d, dim=1)
        for k in range(num_families):
            m = (lbl == k)
            if m.sum() > 0: r1_centers[k] = r1_sample[m].mean(dim=0)
    r1_fam_ids = torch.argmin(torch.cdist(r1, r1_centers), dim=1)
    
    libs_r1 = [train_cb(r1_sample[r1_fam_ids[:sample_n] == k], K=atoms_per_family, n_iter=5) for k in range(num_families)]
    recon_base = macro_recon.clone()
    for k in range(num_families):
        m = (r1_fam_ids == k)
        if m.sum() > 0:
            d_rk = torch.cdist(r1[m], libs_r1[k])
            recon_base[m] += libs_r1[k][torch.argmin(d_rk, dim=1)]
            
    # 3. Detectar os 100% blocos para atingir 3.0 bpp (Resíduo 2 para todos os blocos)
    # Ou selecionar top sensíveis para Resíduo 2 e Resíduo 3
    res_base = w_flat - recon_base
    block_norm = torch.norm(w_flat, dim=1)
    sensitivity = (block_norm + 1e-5) * torch.mean(res_base ** 2, dim=1)
    
    sorted_idx = torch.argsort(sensitivity, descending=True)
    # 1.0 byte extra médio = todos recebem Resíduo 2
    r2_sample = res_base[sorted_idx[:sample_n]]
    lib_r2 = train_cb(r2_sample, K=atoms_per_family, n_iter=5)
    d_r2 = torch.cdist(res_base, lib_r2)
    recon_final = recon_base + lib_r2[torch.argmin(d_r2, dim=1)]
    
    return recon_final.view(orig_shape)

def evaluate_perplexity(model, tokenizer, text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        loss = model(**inputs, labels=inputs.input_ids).loss
        return torch.exp(loss).item()

def generate(model, tokenizer, prompt, max_tokens=25):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True).replace("\n", " ").strip()

def main():
    print("=" * 90)
    print("VALIDAÇÃO MULTIDOMÍNIO: MODELO ORIGINAL (16b) vs ADAPTATIVO 3.0 BPP")
    print("Testando Generalização em 4 Domínios Distintos e Prompts Diversos")
    print("=" * 90)
    
    model_name = "distilbert/distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_orig = AutoModelForCausalLM.from_pretrained(model_name)
    model_orig.eval()
    
    # 4 Domínios Textuais para Teste de Perplexidade
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
    
    prompts = [
        ("Ciência", "The fundamental principles of quantum physics include"),
        ("Código", "A binary search algorithm operates by"),
        ("História", "The library of Alexandria was famous for"),
        ("Lógica", "To solve a complex engineering problem, an engineer must")
    ]
    
    # Medições no Modelo Original
    ppl_orig_domains = {d: evaluate_perplexity(model_orig, tokenizer, text) for d, text in domains.items()}
    gen_orig_prompts = [(cat, p, generate(model_orig, tokenizer, p)) for cat, p in prompts]
    
    # Quantizar todas as 12 Camadas Transformer MLP com 3.0 bpp Adaptativo
    print("\nQuantizando 12 Camadas MLP do DistilGPT-2 com Sistema 3.0 bpp...")
    model_3b = copy.deepcopy(model_orig)
    model_3b.eval()
    
    t0 = time.time()
    for block in model_3b.transformer.h:
        with torch.no_grad():
            block.mlp.c_fc.weight.data = quantize_layer_3bpp_adaptive(block.mlp.c_fc.weight.data)
            block.mlp.c_proj.weight.data = quantize_layer_3bpp_adaptive(block.mlp.c_proj.weight.data)
    elapsed = time.time() - t0
    print(f"Quantização concluída em {elapsed:.2f}s!")
    
    # Medições no Modelo Comprimido 3.0 bpp
    ppl_3b_domains = {d: evaluate_perplexity(model_3b, tokenizer, text) for d, text in domains.items()}
    gen_3b_prompts = [(cat, p, generate(model_3b, tokenizer, p)) for cat, p in prompts]
    
    # Comparativo de Perplexidade Multidomínio
    print("\n" + "=" * 90)
    print("PERPLEXIDADE EM 4 DOMÍNIOS DIFERENTES (ORIGINAL 16b vs ADAPTATIVO 3.0b)")
    print("=" * 90)
    print(f"{'Domínio':<25} | {'Original (16 bpp)':<18} | {'Adaptativo (3.0 bpp)':<20} | {'Delta PPL'}")
    print("-" * 90)
    for d in domains:
        po = ppl_orig_domains[d]
        p3 = ppl_3b_domains[d]
        delta = p3 - po
        print(f"{d:<25} | {po:>14.2f}    | {p3:>16.2f}     | {delta:>+10.2f}")
    print("=" * 90)
    
    # Comparativo de Texto Gerado
    print("\n" + "=" * 90)
    print("AVALIAÇÃO DE COERÊNCIA E ZERO-REPETIÇÃO EM MÚLTIPLOS PROMPTS")
    print("=" * 90)
    for (cat, p, g_orig), (_, _, g_3b) in zip(gen_orig_prompts, gen_3b_prompts):
        print(f"\n[{cat}] Prompt: \"{p}\"")
        print(f"  -> Original (16b):    \"{g_orig}\"")
        print(f"  -> Adaptativo (3.0b): \"{g_3b}\"")
    print("=" * 90)

if __name__ == '__main__':
    main()
