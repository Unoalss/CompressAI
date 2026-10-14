import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import time
import copy

def train_cb(vectors, K=26, n_iter=6):
    """Treina K centroides em vetores 8D de forma robusta."""
    N, D = vectors.shape
    if N == 0:
        return torch.zeros(K, D)
    if N < K:
        # Se houver menos amostras que K, preenche repetindo amostras
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

def evaluate_llm_quality(model, tokenizer, test_text, prompts):
    """Mede PPL e gera respostas com tempo de inferência."""
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(test_text, return_tensors="pt")
        loss = model(**inputs, labels=inputs.input_ids).loss
        ppl = torch.exp(loss).item()
        
    answers = []
    for p in prompts:
        inp = tokenizer(p, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=22, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        ans = tokenizer.decode(out[0], skip_special_tokens=True)
        answers.append(ans)
        
    return ppl, answers

def run_experiment_8_families():
    print("=" * 85)
    print("BENCHMARK DECISIVO: 8 BITS PLANO vs 8 BITS [3+5] COM 8 FAMÍLIAS APRENDIDAS")
    print("Testando Hipótese: 'Roteamento de 3 bits + 26 Dados por Família'")
    print("Camada: transformer.h[0].mlp.c_fc (50.000 blocos = 400.000 pesos reais)")
    print("=" * 85)
    
    model_name = "distilbert/distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    
    test_text = (
        "Artificial intelligence is transforming science, industry, and daily life. "
        "Researchers are developing models that can reason, write code, and assist humans."
    )
    prompts = [
        "The future of artificial intelligence is",
        "To solve a complex problem, one must first"
    ]
    
    # 0. Modelo Original
    ppl_orig, gen_orig = evaluate_llm_quality(model, tokenizer, test_text, prompts)
    print(f"\n[0] MODELO ORIGINAL (16.0 bpp):")
    print(f"    - PPL: {ppl_orig:.2f}")
    print(f"    - Prompt 1: \"{gen_orig[0]}\"")
    
    target_layer = model.transformer.h[0].mlp.c_fc
    orig_weight = target_layer.weight.data.clone()
    block_size = 8
    
    w_flat = orig_weight.float().reshape(-1, block_size)
    sample_size = 50000
    w_sample = w_flat[:sample_size].clone()
    orig_bytes = sample_size * block_size * 2
    
    # =========================================================================
    # VERSÃO B: 8 BITS SIMPLES PLANO (256 Centroides Globais / 2 Estágios)
    # =========================================================================
    print("\n" + "-" * 85)
    print("[1] VERSÃO B: 8 BITS SIMPLES PLANO (256 centroides globais / 2 estágios)")
    print("-" * 85)
    t0_enc = time.perf_counter()
    
    K_256 = 256
    perm = torch.randperm(sample_size)[:K_256]
    cb_b1 = w_sample[perm].clone()
    for _ in range(6):
        d = torch.cdist(w_sample, cb_b1)
        lbl = torch.argmin(d, dim=1)
        for k in range(K_256):
            m = (lbl == k)
            if m.sum() > 0: cb_b1[k] = w_sample[m].mean(dim=0)
            
    lbls_b1 = torch.argmin(torch.cdist(w_sample, cb_b1), dim=1)
    res_b1 = w_sample - cb_b1[lbls_b1]
    
    perm = torch.randperm(sample_size)[:K_256]
    cb_b2 = res_b1[perm].clone()
    for _ in range(6):
        d = torch.cdist(res_b1, cb_b2)
        lbl = torch.argmin(d, dim=1)
        for k in range(K_256):
            m = (lbl == k)
            if m.sum() > 0: cb_b2[k] = res_b1[m].mean(dim=0)
            
    lbls_b2 = torch.argmin(torch.cdist(res_b1, cb_b2), dim=1)
    t_enc_b = time.perf_counter() - t0_enc
    
    t0_dec = time.perf_counter()
    recon_b = cb_b1[lbls_b1] + cb_b2[lbls_b2]
    t_dec_b = (time.perf_counter() - t0_dec) * 1000 # ms
    
    mse_b = torch.mean((w_sample - recon_b) ** 2).item()
    cos_b = torch.nn.functional.cosine_similarity(w_sample.view(1, -1), recon_b.view(1, -1)).item()
    dict_b = (2 * 256 * block_size * 2)
    bytes_b = (sample_size * 2) + dict_b
    bpp_b = 2.00
    
    # =========================================================================
    # VERSÃO C1: 8 BITS [3+5] COM 3 FAMÍLIAS (Hardcoded / Teste anterior)
    # =========================================================================
    print("\n" + "-" * 85)
    print("[2] VERSÃO C1: 8 BITS [3+5] COM 3 FAMÍLIAS (78 átomos)")
    print("-" * 85)
    t0_enc = time.perf_counter()
    
    # 3 famílias
    perm3 = torch.randperm(sample_size)[:3]
    fam_c1 = w_sample[perm3].clone()
    for _ in range(5):
        d = torch.cdist(w_sample, fam_c1)
        lbl = torch.argmin(d, dim=1)
        for k in range(3):
            m = (lbl == k)
            if m.sum() > 0: fam_c1[k] = w_sample[m].mean(dim=0)
    fam_ids3 = torch.argmin(torch.cdist(w_sample, fam_c1), dim=1)
    
    libs_c1 = [train_cb(w_sample[fam_ids3 == k], K=26, n_iter=5) for k in range(3)]
    macro_c1 = torch.zeros_like(w_sample)
    for k in range(3):
        m = (fam_ids3 == k)
        if m.sum() > 0:
            d = torch.cdist(w_sample[m], libs_c1[k])
            macro_c1[m] = libs_c1[k][torch.argmin(d, dim=1)]
            
    res_c1 = w_sample - macro_c1
    lib_res_c1 = train_cb(res_c1, K=26, n_iter=5)
    d_res3 = torch.cdist(res_c1, lib_res_c1)
    recon_c1 = macro_c1 + lib_res_c1[torch.argmin(d_res3, dim=1)]
    t_enc_c1 = time.perf_counter() - t0_enc
    
    t0_dec = time.perf_counter()
    _ = macro_c1 + lib_res_c1[torch.argmin(d_res3, dim=1)]
    t_dec_c1 = (time.perf_counter() - t0_dec) * 1000
    
    mse_c1 = torch.mean((w_sample - recon_c1) ** 2).item()
    cos_c1 = torch.nn.functional.cosine_similarity(w_sample.view(1, -1), recon_c1.view(1, -1)).item()
    dict_c1 = (3 * 26 * block_size * 2) + (26 * block_size * 2)
    bytes_c1 = (sample_size * 2) + dict_c1
    bpp_c1 = 2.00
    
    # =========================================================================
    # VERSÃO C2: 8 BITS [3+5] COM 8 FAMÍLIAS APRENDIDAS (208 Átomos Especializados!)
    # Os 3 bits indexam as 8 famílias naturais (0 a 7).
    # Os 5 bits indexam os 26 átomos de dados dentro da família selecionada!
    # =========================================================================
    print("\n" + "-" * 85)
    print("[3] VERSÃO C2: 8 BITS [3+5] COM 8 FAMÍLIAS APRENDIDAS (8x26 = 208 átomos macro!)")
    print("-" * 85)
    t0_enc = time.perf_counter()
    
    # 1. Aprender as 8 Famílias Naturais da Camada via K-Means
    perm8 = torch.randperm(sample_size)[:8]
    fam_centers_8 = w_sample[perm8].clone()
    for _ in range(6):
        d = torch.cdist(w_sample, fam_centers_8)
        lbl = torch.argmin(d, dim=1)
        for k in range(8):
            m = (lbl == k)
            if m.sum() > 0: fam_centers_8[k] = w_sample[m].mean(dim=0)
            
    fam_ids_8 = torch.argmin(torch.cdist(w_sample, fam_centers_8), dim=1)
    
    # 2. Treinar 26 átomos especializados para CADA UMA das 8 famílias macro
    # Total de átomos = 8 * 26 = 208 átomos macro
    libs_macro_8 = []
    for k in range(8):
        mask_k = (fam_ids_8 == k)
        vectors_k = w_sample[mask_k]
        libs_macro_8.append(train_cb(vectors_k, K=26, n_iter=6))
        
    # Calcular reconstrução macro do Nível 1
    macro_recon_8 = torch.zeros_like(w_sample)
    macro_atom_indices = torch.zeros(sample_size, dtype=torch.long)
    for k in range(8):
        m = (fam_ids_8 == k)
        if m.sum() > 0:
            d_k = torch.cdist(w_sample[m], libs_macro_8[k])
            best_idx = torch.argmin(d_k, dim=1)
            macro_recon_8[m] = libs_macro_8[k][best_idx]
            macro_atom_indices[m] = best_idx
            
    # 3. Resíduos: Aprender 8 Famílias Especializadas de Resíduo
    res_8 = w_sample - macro_recon_8
    perm_r8 = torch.randperm(sample_size)[:8]
    res_fam_centers = res_8[perm_r8].clone()
    for _ in range(6):
        d = torch.cdist(res_8, res_fam_centers)
        lbl = torch.argmin(d, dim=1)
        for k in range(8):
            m = (lbl == k)
            if m.sum() > 0: res_fam_centers[k] = res_8[m].mean(dim=0)
            
    res_fam_ids = torch.argmin(torch.cdist(res_8, res_fam_centers), dim=1)
    
    libs_res_8 = []
    for k in range(8):
        mask_rk = (res_fam_ids == k)
        vecs_rk = res_8[mask_rk]
        libs_res_8.append(train_cb(vecs_rk, K=26, n_iter=6))
        
    res_recon_8 = torch.zeros_like(w_sample)
    for k in range(8):
        m = (res_fam_ids == k)
        if m.sum() > 0:
            d_rk = torch.cdist(res_8[m], libs_res_8[k])
            res_recon_8[m] = libs_res_8[k][torch.argmin(d_rk, dim=1)]
            
    # Reconstrução Final de 2 Bytes:
    # Byte 1: (fam_ids_8[i] << 5) | macro_atom_indices[i]
    # Byte 2: (res_fam_ids[i] << 5) | res_atom_indices[i]
    recon_c2 = macro_recon_8 + res_recon_8
    t_enc_c2 = time.perf_counter() - t0_enc
    
    # Decodificação de alta velocidade
    t0_dec = time.perf_counter()
    _ = macro_recon_8 + res_recon_8
    t_dec_c2 = (time.perf_counter() - t0_dec) * 1000
    
    mse_c2 = torch.mean((w_sample - recon_c2) ** 2).item()
    cos_c2 = torch.nn.functional.cosine_similarity(w_sample.view(1, -1), recon_c2.view(1, -1)).item()
    
    # Dicionário: 8 famílias macro * 26 + 8 famílias resíduo * 26 = 416 vetores de 8 floats FP16
    dict_bytes_c2 = 2 * (8 * 26 * block_size * 2) # 6.65 KB
    bytes_c2 = (sample_size * 2) + dict_bytes_c2
    bpp_c2 = 2.00
    
    # =========================================================================
    # AVALIAÇÃO NA LLM COMPLETA
    # =========================================================================
    print("\n" + "=" * 85)
    print("AVALIAÇÃO DE PERPLEXIDADE E GERAÇÃO NA LLM (DISTILGPT-2)")
    print("=" * 85)
    
    # Versão B na LLM
    model_b = copy.deepcopy(model)
    fw_b = orig_weight.clone()
    fw_b.view(-1, block_size)[:sample_size] = recon_b
    model_b.transformer.h[0].mlp.c_fc.weight.data = fw_b
    ppl_b, gen_b = evaluate_llm_quality(model_b, tokenizer, test_text, prompts)
    
    # Versão C1 na LLM
    model_c1 = copy.deepcopy(model)
    fw_c1 = orig_weight.clone()
    fw_c1.view(-1, block_size)[:sample_size] = recon_c1
    model_c1.transformer.h[0].mlp.c_fc.weight.data = fw_c1
    ppl_c1, gen_c1 = evaluate_llm_quality(model_c1, tokenizer, test_text, prompts)
    
    # Versão C2 (8 Famílias) na LLM
    model_c2 = copy.deepcopy(model)
    fw_c2 = orig_weight.clone()
    fw_c2.view(-1, block_size)[:sample_size] = recon_c2
    model_c2.transformer.h[0].mlp.c_fc.weight.data = fw_c2
    ppl_c2, gen_c2 = evaluate_llm_quality(model_c2, tokenizer, test_text, prompts)
    
    # =========================================================================
    # PLACAR COMPARATIVO FINAL
    # =========================================================================
    print("\n" + "=" * 90)
    print("PLACAR COMPARATIVO DEFINITIVO: O IMPACTO DAS 8 FAMÍLIAS APRENDIDAS")
    print("=" * 90)
    print(f"{'Métrica':<25} | {'B: 8b Simples Plano':<20} | {'C1: [3+5] 3 Famílias':<20} | {'C2: [3+5] 8 Famílias'}")
    print("-" * 90)
    print(f"{'Capacidade do Codebook':<25} | {'256 globais':<20} | {'78 átomos (3x26)':<20} | {'208 átomos (8x26)'}")
    print(f"{'Tamanho Total (KB)':<25} | {bytes_b/1024:>17.1f} KB | {bytes_c1/1024:>17.1f} KB | {bytes_c2/1024:>17.1f} KB")
    print(f"{'Dicionário (KB)':<25} | {dict_b/1024:>17.2f} KB | {dict_c1/1024:>17.2f} KB | {dict_bytes_c2/1024:>17.2f} KB")
    print(f"{'Bits / Peso (BPP)':<25} | {bpp_b:>17.2f} b | {bpp_c1:>17.2f} b | {bpp_c2:>17.2f} b")
    print(f"{'MSE da Camada':<25} | {mse_b:>20.6f} | {mse_c1:>20.6f} | {mse_c2:>20.6f}")
    print(f"{'Fidelidade (Cosseno)':<25} | {cos_b*100:>19.2f}% | {cos_c1*100:>19.2f}% | {cos_c2*100:>19.2f}%")
    print(f"{'Tempo de Decode (ms)':<25} | {t_dec_b:>17.2f} ms | {t_dec_c1:>17.2f} ms | {t_dec_c2:>17.2f} ms")
    print(f"{'Perplexidade (PPL)':<25} | {ppl_b:>20.2f} | {ppl_c1:>20.2f} | {ppl_c2:>20.2f}")
    print("=" * 90)
    
    print("\nRESPOSTAS GERADAS NA LLM:")
    print(f"-> Original:            \"{gen_orig[0]}\"")
    print(f"-> B (8b Plano):        \"{gen_b[0]}\"")
    print(f"-> C1 ([3+5] 3 Fam):    \"{gen_c1[0]}\"")
    print(f"-> C2 ([3+5] 8 Fam):    \"{gen_c2[0]}\"")

if __name__ == '__main__':
    run_experiment_8_families()
