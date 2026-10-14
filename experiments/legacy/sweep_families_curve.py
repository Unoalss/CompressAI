import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import time
import copy

def train_cb(vectors, K=26, n_iter=5):
    """Treina K centroides em vetores 8D de forma robusta."""
    N, D = vectors.shape
    if N == 0:
        return torch.zeros(K, D)
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

def evaluate_reconstruction(w_sample, num_families, atoms_per_family=26, block_size=8):
    """
    Treina a arquitetura com F famílias aprendidas (Macro + Resíduo)
    Mantendo 2 bytes por bloco = 16 bits = 2.0 bits/peso.
    """
    N = w_sample.shape[0]
    
    # 1. Clusterizar em F famílias macro
    perm = torch.randperm(N)[:num_families]
    fam_centers = w_sample[perm].clone()
    for _ in range(5):
        d = torch.cdist(w_sample, fam_centers)
        lbl = torch.argmin(d, dim=1)
        for k in range(num_families):
            m = (lbl == k)
            if m.sum() > 0: fam_centers[k] = w_sample[m].mean(dim=0)
            
    fam_ids = torch.argmin(torch.cdist(w_sample, fam_centers), dim=1)
    
    # 2. Treinar átomos especializados para cada família macro
    libs_macro = []
    for k in range(num_families):
        m = (fam_ids == k)
        libs_macro.append(train_cb(w_sample[m], K=atoms_per_family, n_iter=5))
        
    macro_recon = torch.zeros_like(w_sample)
    for k in range(num_families):
        m = (fam_ids == k)
        if m.sum() > 0:
            d_k = torch.cdist(w_sample[m], libs_macro[k])
            macro_recon[m] = libs_macro[k][torch.argmin(d_k, dim=1)]
            
    # 3. Resíduos: Clusterizar em F famílias de resíduo
    res = w_sample - macro_recon
    perm_r = torch.randperm(N)[:num_families]
    res_fam_centers = res[perm_r].clone()
    for _ in range(5):
        d = torch.cdist(res, res_fam_centers)
        lbl = torch.argmin(d, dim=1)
        for k in range(num_families):
            m = (lbl == k)
            if m.sum() > 0: res_fam_centers[k] = res[m].mean(dim=0)
            
    res_fam_ids = torch.argmin(torch.cdist(res, res_fam_centers), dim=1)
    
    libs_res = []
    for k in range(num_families):
        m = (res_fam_ids == k)
        libs_res.append(train_cb(res[m], K=atoms_per_family, n_iter=5))
        
    res_recon = torch.zeros_like(w_sample)
    for k in range(num_families):
        m = (res_fam_ids == k)
        if m.sum() > 0:
            d_rk = torch.cdist(res[m], libs_res[k])
            res_recon[m] = libs_res[k][torch.argmin(d_rk, dim=1)]
            
    # Tempo de Decode de alta precisão
    t0_dec = time.perf_counter()
    recon_final = macro_recon + res_recon
    t_dec = (time.perf_counter() - t0_dec) * 1000 # ms
    
    mse = torch.mean((w_sample - recon_final) ** 2).item()
    cos_sim = torch.nn.functional.cosine_similarity(w_sample.view(1, -1), recon_final.view(1, -1)).item()
    
    # Dicionário: 2 estágios * F famílias * atoms * block_size * 2 bytes
    dict_bytes = 2 * (num_families * atoms_per_family * block_size * 2)
    dict_kb = dict_bytes / 1024
    
    return recon_final, mse, cos_sim, dict_kb, t_dec

def main():
    print("=" * 90)
    print("CURVA EXPERIMENTAL: QUANTAS FAMÍLIAS SÃO NECESSÁRIAS? (F = 2 até 16)")
    print("Mantendo Rigorosamente 2.0 Bits por Peso (2 Bytes por Bloco de 8 Pesos)")
    print("Camada: transformer.h[0].mlp.c_fc (50.000 blocos = 400.000 pesos reais)")
    print("=" * 90)
    
    model_name = "distilbert/distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    base_model.eval()
    
    test_text = (
        "Artificial intelligence is transforming science, industry, and daily life. "
        "Researchers are developing models that can reason, write code, and assist humans."
    )
    prompt = "The future of artificial intelligence is"
    
    with torch.no_grad():
        inputs = tokenizer(test_text, return_tensors="pt")
        ppl_orig = torch.exp(base_model(**inputs, labels=inputs.input_ids).loss).item()
        
    orig_weight = base_model.transformer.h[0].mlp.c_fc.weight.data.clone()
    block_size = 8
    w_flat = orig_weight.float().reshape(-1, block_size)
    sample_size = 50000
    w_sample = w_flat[:sample_size].clone()
    
    family_counts = [2, 3, 4, 5, 6, 7, 8, 16]
    results = []
    
    for F in family_counts:
        # Se F == 16, usamos 16 átomos por família para caber em 8 bits [4+4 = 256 estados]
        atoms = 16 if F == 16 else 26
        
        recon, mse, cos_sim, dict_kb, t_dec = evaluate_reconstruction(
            w_sample, num_families=F, atoms_per_family=atoms, block_size=block_size
        )
        
        # Testar PPL na LLM
        model_test = copy.deepcopy(base_model)
        full_w = orig_weight.clone()
        full_w.view(-1, block_size)[:sample_size] = recon
        model_test.transformer.h[0].mlp.c_fc.weight.data = full_w
        
        with torch.no_grad():
            inp_tok = tokenizer(test_text, return_tensors="pt")
            ppl = torch.exp(model_test(**inp_tok, labels=inp_tok.input_ids).loss).item()
            
            inp_gen = tokenizer(prompt, return_tensors="pt")
            out = model_test.generate(**inp_gen, max_new_tokens=18, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            gen_text = tokenizer.decode(out[0], skip_special_tokens=True).replace("\n", " ").strip()
            
        results.append({
            'F': F,
            'atoms': atoms,
            'format': f"[{int(np.log2(F))}+{int(np.log2(atoms))}]" if F == 16 else f"[3+5]",
            'mse': mse,
            'cos_sim': cos_sim,
            'dict_kb': dict_kb,
            't_dec': t_dec,
            'ppl': ppl,
            'gen': gen_text
        })
        
    print("\n" + "=" * 90)
    print("A CURVA DE FAMÍLIAS: MSE, FIDELIDADE, TAMANHO E PERPLEXIDADE")
    print("=" * 90)
    print(f"{'Famílias':<10} | {'Formato':<8} | {'Dicionário':<11} | {'MSE da Camada':<14} | {'Cosseno':<9} | {'Decode':<8} | {'PPL na LLM'}")
    print("-" * 90)
    print(f"{'Original':<10} | {'16b FP':<8} | {'-':<11} | {'0.000000':<14} | {'100.0%':<9} | {'-':<8} | {ppl_orig:<10.2f}")
    
    for r in results:
        f_label = f"{r['F']} fam ({r['atoms']}at)"
        print(f"{f_label:<10} | {r['format']:<8} | {r['dict_kb']:>8.2f} KB | {r['mse']:<14.6f} | {r['cos_sim']*100:>6.2f}% | {r['t_dec']:>5.2f}ms | {r['ppl']:<10.2f}")
    print("=" * 90)
    
    print("\nAMOSTRAS DE RESPOSTA NA LLM:")
    for r in results:
        print(f"-> {r['F']} Famílias (PPL {r['ppl']:.2f}): \"{r['gen']}\"")

if __name__ == '__main__':
    main()
