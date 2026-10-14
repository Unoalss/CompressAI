import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import time
import copy

def train_cb(vectors, K=26, n_iter=6):
    """Treina K centroides em vetores 8D."""
    N, D = vectors.shape
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
    """Mede PPL e gera respostas para múltiplos prompts."""
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

def run_experiment():
    print("=" * 80)
    print("BATALHA DE ARQUITETURAS: 5 BITS vs 8 BITS SIMPLES vs 8 BITS [3+5] HIERÁRQUICO")
    print("Camada Real: transformer.h[0].mlp.c_fc do DistilGPT-2 (50.000 blocos = 400k pesos)")
    print("=" * 80)
    
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
    
    # 0. Referência do Modelo Original
    ppl_orig, gen_orig = evaluate_llm_quality(model, tokenizer, test_text, prompts)
    print(f"\n[0] MODELO ORIGINAL (FP16 - 16.0 bpp):")
    print(f"    - PPL: {ppl_orig:.2f}")
    print(f"    - Prompt 1: \"{gen_orig[0]}\"")
    print(f"    - Prompt 2: \"{gen_orig[1]}\"")
    
    # Extrair camada
    target_layer = model.transformer.h[0].mlp.c_fc
    orig_weight = target_layer.weight.data.clone()
    orig_shape = orig_weight.shape # (768, 3072)
    block_size = 8
    
    w_flat = orig_weight.float().reshape(-1, block_size)
    sample_size = 50000
    w_sample = w_flat[:sample_size].clone()
    orig_bytes = sample_size * block_size * 2 # FP16 = 800 KB
    
    x_test = torch.randn(block_size, 1)
    
    # =========================================================================
    # VERSÃO A: 5 BITS PURO (Stream de Tokens de 5 bits)
    # Token 1 (5b): Forma macro (26 átomos)
    # Token 2 (5b): Escala ou Resíduo
    # Token 3 (5b): Resíduo fino
    # =========================================================================
    print("\n" + "-" * 80)
    print("[1] EXECUTANDO VERSÃO A: SISTEMA 5 BITS PURO")
    print("-" * 80)
    t0_enc = time.time()
    
    lib_a_macro = train_cb(w_sample, K=26, n_iter=6)
    dists_a = torch.cdist(w_sample, lib_a_macro)
    lbls_a1 = torch.argmin(dists_a, dim=1)
    res_a1 = w_sample - lib_a_macro[lbls_a1]
    
    lib_a_res1 = train_cb(res_a1, K=26, n_iter=6)
    dists_a2 = torch.cdist(res_a1, lib_a_res1)
    lbls_a2 = torch.argmin(dists_a2, dim=1)
    res_a2 = res_a1 - lib_a_res1[lbls_a2]
    
    lib_a_res2 = train_cb(res_a2, K=26, n_iter=6)
    dists_a3 = torch.cdist(res_a2, lib_a_res2)
    lbls_a3 = torch.argmin(dists_a3, dim=1)
    
    # Cada bloco usa 3 tokens de 5 bits = 15 bits por bloco = 1.875 bpp
    tokens_a = sample_size * 3
    bits_a = tokens_a * 5
    bpp_a = bits_a / (sample_size * block_size)
    bytes_a = int(np.ceil(bits_a / 8.0)) + (3 * 26 * block_size * 2)
    t_enc_a = time.time() - t0_enc
    
    t0_dec = time.time()
    recon_a = lib_a_macro[lbls_a1] + lib_a_res1[lbls_a2] + lib_a_res2[lbls_a3]
    t_dec_a = time.time() - t0_dec
    
    mse_a = torch.mean((w_sample - recon_a) ** 2).item()
    cos_a = torch.nn.functional.cosine_similarity(w_sample.view(1, -1), recon_a.view(1, -1)).item()
    
    # =========================================================================
    # VERSÃO B: 8 BITS SIMPLES (Flat RVQ - 2 Estágios de 8 bits = 16 bits/bloco)
    # Tabela 1 (256 centroides = 8 bits) + Tabela 2 (256 centroides = 8 bits)
    # Custo: exatamente 2 bytes por bloco = 16 bits = 2.0 bpp
    # =========================================================================
    print("\n" + "-" * 80)
    print("[2] EXECUTANDO VERSÃO B: 8 BITS SIMPLES (RVQ Plano - 2 Bytes / 16 bits)")
    print("-" * 80)
    t0_enc = time.time()
    
    # K-means 256 centroides
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
    
    bits_b = sample_size * 16 # 2 bytes por bloco
    bpp_b = 16.0 / block_size # 2.0 bpp
    bytes_b = (sample_size * 2) + (2 * 256 * block_size * 2)
    t_enc_b = time.time() - t0_enc
    
    t0_dec = time.time()
    recon_b = cb_b1[lbls_b1] + cb_b2[lbls_b2]
    t_dec_b = time.time() - t0_dec
    
    mse_b = torch.mean((w_sample - recon_b) ** 2).item()
    cos_b = torch.nn.functional.cosine_similarity(w_sample.view(1, -1), recon_b.view(1, -1)).item()
    
    # =========================================================================
    # VERSÃO C: 8 BITS [3+5] HIERÁRQUICO
    # Cada símbolo tem 8 bits:
    # 3 bits (0-7): CHAVE PRINCIPAL (Seleciona Família / Modo)
    #   Key 0 (000): Família A (Padrões Macro Suaves - 26 átomos)
    #   Key 1 (001): Família B (Padrões Macro Picos - 26 átomos)
    #   Key 2 (010): Família C (Padrões Macro Oscilatórios - 26 átomos)
    #   Key 3 (011): REF (Cache LRU de 26 posições) -> CUSTA APENAS 1 BYTE (8 bits)!
    #   Key 4 (100): ESCALA
    #   Key 5 (101): ADD (Resíduo fino - 26 átomos)
    # 5 bits (0-31): Subestado (0-25 dados, 26-31 comandos)
    # =========================================================================
    print("\n" + "-" * 80)
    print("[3] EXECUTANDO VERSÃO C: 8 BITS [3+5] HIERÁRQUICO")
    print("-" * 80)
    t0_enc = time.time()
    
    # Classificar blocos em 3 famílias geométricas usando KMeans inicial
    # 3 clusters de famílias
    perm = torch.randperm(sample_size)[:3]
    fam_centers = w_sample[perm].clone()
    for _ in range(6):
        d = torch.cdist(w_sample, fam_centers)
        l_fam = torch.argmin(d, dim=1)
        for k in range(3):
            m = (l_fam == k)
            if m.sum() > 0: fam_centers[k] = w_sample[m].mean(dim=0)
            
    d_fam = torch.cdist(w_sample, fam_centers)
    family_assign = torch.argmin(d_fam, dim=1)
    
    # Treinar 26 átomos especializados para cada uma das 3 famílias
    lib_fam0 = train_cb(w_sample[family_assign == 0], K=26, n_iter=6)
    lib_fam1 = train_cb(w_sample[family_assign == 1], K=26, n_iter=6)
    lib_fam2 = train_cb(w_sample[family_assign == 2], K=26, n_iter=6)
    
    # Calcular aproximação macro com a família especializada
    macro_recon_c = torch.zeros_like(w_sample)
    d0 = torch.cdist(w_sample, lib_fam0)
    d1 = torch.cdist(w_sample, lib_fam1)
    d2 = torch.cdist(w_sample, lib_fam2)
    
    min_d0, idx0 = torch.min(d0, dim=1)
    min_d1, idx1 = torch.min(d1, dim=1)
    min_d2, idx2 = torch.min(d2, dim=1)
    
    all_d = torch.stack([min_d0, min_d1, min_d2], dim=1)
    best_fam = torch.argmin(all_d, dim=1)
    
    best_atom_c = torch.zeros(sample_size, dtype=torch.long)
    for i in range(sample_size):
        f = best_fam[i].item()
        if f == 0:
            best_atom_c[i] = idx0[i]
            macro_recon_c[i] = lib_fam0[idx0[i]]
        elif f == 1:
            best_atom_c[i] = idx1[i]
            macro_recon_c[i] = lib_fam1[idx1[i]]
        else:
            best_atom_c[i] = idx2[i]
            macro_recon_c[i] = lib_fam2[idx2[i]]
            
    # Resíduos após a família especializada
    res_c = w_sample - macro_recon_c
    lib_res_c = train_cb(res_c, K=26, n_iter=6)
    d_resc = torch.cdist(res_c, lib_res_c)
    best_res_c = torch.argmin(d_resc, dim=1)
    
    # RDO com seleção de 8 bits:
    # REF = 1 Byte (Key 3 + Cache ID)
    # MACRO puro = 1 Byte (Key fam + Atom ID)
    # MACRO + ADD = 2 Bytes (Byte 1: Key fam + Atom ID | Byte 2: Key 5 + Res ID)
    TOL_C = 0.0003
    recon_c_list = []
    total_bytes_c = 0
    cache_c = []
    
    for i in range(sample_size):
        b = w_sample[i]
        reused = False
        
        # 1. Checa REF (Custa apenas 1 Byte de 8 bits!)
        if len(cache_c) > 0:
            c_mat = torch.stack(cache_c)
            d_cache = torch.mean((c_mat - b) ** 2, dim=1)
            min_c, c_id = torch.min(d_cache, dim=0)
            if min_c.item() <= TOL_C:
                chosen_b = cache_c[c_id.item()]
                recon_c_list.append(chosen_b)
                total_bytes_c += 1 # 1 Byte apenas!
                cache_c.append(chosen_b)
                if len(cache_c) > 26: cache_c.pop(0)
                reused = True
                continue
                
        if not reused:
            # 2. Avalia se Macro puro basta (1 Byte) ou se precisa de ADD (2 Bytes)
            m_vec = macro_recon_c[i]
            err_m = torch.mean((m_vec - b) ** 2).item()
            if err_m <= TOL_C:
                recon_c_list.append(m_vec)
                total_bytes_c += 1 # 1 Byte
                cache_c.append(m_vec)
            else:
                full_b = m_vec + lib_res_c[best_res_c[i]]
                recon_c_list.append(full_b)
                total_bytes_c += 2 # 2 Bytes (16 bits)
                cache_c.append(full_b)
                
            if len(cache_c) > 26: cache_c.pop(0)
            
    t_enc_c = time.time() - t0_enc
    
    t0_dec = time.time()
    recon_c = torch.stack(recon_c_list)
    t_dec_c = time.time() - t0_dec
    
    mse_c = torch.mean((w_sample - recon_c) ** 2).item()
    cos_c = torch.nn.functional.cosine_similarity(w_sample.view(1, -1), recon_c.view(1, -1)).item()
    
    bpp_c = (total_bytes_c * 8) / (sample_size * block_size)
    dict_bytes_c = (3 * 26 * block_size * 2) + (26 * block_size * 2) # 4 libs de 26 vetores = 1.66 KB
    bytes_c = total_bytes_c + dict_bytes_c
    
    # =========================================================================
    # TESTE NA LLM COMPLETA PARA AS 3 VERSÕES
    # =========================================================================
    print("\n" + "=" * 80)
    print("AVALIAÇÃO NA LLM COMPLETA (DISTILGPT-2)")
    print("=" * 80)
    
    # Testar Versão A na LLM
    model_a = copy.deepcopy(model)
    full_w_a = orig_weight.clone()
    full_w_a.view(-1, block_size)[:sample_size] = recon_a
    model_a.transformer.h[0].mlp.c_fc.weight.data = full_w_a
    ppl_a, gen_a = evaluate_llm_quality(model_a, tokenizer, test_text, prompts)
    
    # Testar Versão B na LLM
    model_b = copy.deepcopy(model)
    full_w_b = orig_weight.clone()
    full_w_b.view(-1, block_size)[:sample_size] = recon_b
    model_b.transformer.h[0].mlp.c_fc.weight.data = full_w_b
    ppl_b, gen_b = evaluate_llm_quality(model_b, tokenizer, test_text, prompts)
    
    # Testar Versão C na LLM
    model_c = copy.deepcopy(model)
    full_w_c = orig_weight.clone()
    full_w_c.view(-1, block_size)[:sample_size] = recon_c
    model_c.transformer.h[0].mlp.c_fc.weight.data = full_w_c
    ppl_c, gen_c = evaluate_llm_quality(model_c, tokenizer, test_text, prompts)
    
    # =========================================================================
    # PLACAR FINAL
    # =========================================================================
    print("\n" + "=" * 85)
    print("PLACAR COMPARATIVO DEFINITIVO: 5 BITS vs 8 BITS SIMPLES vs 8 BITS [3+5] HIERÁRQUICO")
    print("=" * 85)
    print(f"{'Métrica':<25} | {'Versão A (5 bits)':<18} | {'Versão B (8b simples)':<18} | {'Versão C ([3+5] Hierárq)'}")
    print("-" * 85)
    print(f"{'Tamanho Total (KB)':<25} | {bytes_a/1024:>14.1f} KB | {bytes_b/1024:>14.1f} KB | {bytes_c/1024:>14.1f} KB")
    print(f"{'Redução vs FP16':<25} | {100*(1-bytes_a/orig_bytes):>14.1f} % | {100*(1-bytes_b/orig_bytes):>14.1f} % | {100*(1-bytes_c/orig_bytes):>14.1f} %")
    print(f"{'Bits / Peso (BPP)':<25} | {bpp_a:>14.2f} b | {bpp_b:>14.2f} b | {bpp_c:>14.2f} b")
    print(f"{'MSE da Camada':<25} | {mse_a:>18.6f} | {mse_b:>18.6f} | {mse_c:>18.6f}")
    print(f"{'Fidelidade (Cosseno)':<25} | {cos_a*100:>17.2f}% | {cos_b*100:>17.2f}% | {cos_c*100:>17.2f}%")
    print(f"{'Tempo de Encode':<25} | {t_enc_a:>15.2f} s | {t_enc_b:>15.2f} s | {t_enc_c:>15.2f} s")
    print(f"{'Tempo de Decode':<25} | {t_dec_a*1000:>13.2f} ms | {t_dec_b*1000:>13.2f} ms | {t_dec_c*1000:>13.2f} ms")
    print(f"{'Perplexidade (PPL)':<25} | {ppl_a:>18.2f} | {ppl_b:>18.2f} | {ppl_c:>18.2f}")
    print("=" * 85)
    
    print("\nRESPOSTAS GERADAS NA LLM:")
    print(f"-> Original:   \"{gen_orig[0]}\"")
    print(f"-> Versão A:   \"{gen_a[0]}\"")
    print(f"-> Versão B:   \"{gen_b[0]}\"")
    print(f"-> Versão C:   \"{gen_c[0]}\"")

if __name__ == '__main__':
    run_experiment()
