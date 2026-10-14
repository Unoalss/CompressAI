import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import time
import copy

def train_cb(vectors, K=26, n_iter=5):
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

def quantize_layer_sensitivity_adaptive(weight_tensor, target_bpp=2.25, num_families=8, atoms_per_family=26, block_size=8):
    """
    Quantiza a camada com precisão adaptativa baseada em sensibilidade:
    - Base: 2.0 bpp (2 bytes [3+5]: Macro + Resíduo 1)
    - Blocos mais sensíveis (alto resíduo * magnitude) recebem Byte 3 (+Resíduo 2 = 3.0 bpp)
    - Blocos hiper-sensíveis recebem Byte 4 (+Resíduo 3 = 4.0 bpp)
    Garante a taxa média exata target_bpp (ex: 2.25, 2.50, 3.00 bpp).
    """
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
    
    # Treinar átomos macro
    libs_macro = []
    for k in range(num_families):
        mask_k = (fam_ids[:sample_n] == k)
        libs_macro.append(train_cb(w_sample[mask_k], K=atoms_per_family, n_iter=5))
        
    macro_recon = torch.zeros_like(w_flat)
    for k in range(num_families):
        m = (fam_ids == k)
        if m.sum() > 0:
            d_k = torch.cdist(w_flat[m], libs_macro[k])
            macro_recon[m] = libs_macro[k][torch.argmin(d_k, dim=1)]
            
    # 2. Resíduo 1 (para todos os blocos -> Base de 2.0 bpp)
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
    libs_r1 = []
    for k in range(num_families):
        mask_rk = (r1_fam_ids[:sample_n] == k)
        libs_r1.append(train_cb(r1_sample[mask_rk], K=atoms_per_family, n_iter=5))
        
    recon_base = macro_recon.clone()
    for k in range(num_families):
        m = (r1_fam_ids == k)
        if m.sum() > 0:
            d_rk = torch.cdist(r1[m], libs_r1[k])
            recon_base[m] += libs_r1[k][torch.argmin(d_rk, dim=1)]
            
    if target_bpp <= 2.0:
        return recon_base.view(orig_shape), 2.0
        
    # 3. DETECTAR BLOCOS SENSÍVEIS
    # Sensibilidade = Magnitude do Bloco * Erro Residual ao quadrado
    res_base = w_flat - recon_base
    block_norm = torch.norm(w_flat, dim=1)
    mse_block = torch.mean(res_base ** 2, dim=1)
    sensitivity = (block_norm + 1e-5) * mse_block
    
    # Orçamento extra de bits acima de 2.0 bpp
    extra_bits_per_weight = target_bpp - 2.0
    extra_bytes_total = int(extra_bits_per_weight * N) # cada byte extra adiciona 1.0 bpp por bloco
    
    # Alocar bits extras aos mais sensíveis:
    # Top blocos recebem Byte 3 (+Resíduo 2)
    # Desses, os mais extremos ainda recebem Byte 4 (+Resíduo 3)
    p_extra1 = min(1.0, extra_bits_per_weight) # fração que recebe Byte 3
    num_extra1 = int(p_extra1 * N)
    
    sorted_indices = torch.argsort(sensitivity, descending=True)
    idx_extra1 = sorted_indices[:num_extra1]
    
    # Treinar e aplicar Resíduo 2 nos blocos sensíveis
    r2_vecs = res_base[idx_extra1]
    lib_r2 = train_cb(r2_vecs, K=atoms_per_family, n_iter=5)
    d_r2 = torch.cdist(r2_vecs, lib_r2)
    recon_base[idx_extra1] += lib_r2[torch.argmin(d_r2, dim=1)]
    
    # Se target_bpp > 3.0, os top mais extremos recebem Byte 4
    if target_bpp > 3.0:
        p_extra2 = min(1.0, target_bpp - 3.0)
        num_extra2 = int(p_extra2 * N)
        idx_extra2 = sorted_indices[:num_extra2]
        res_r2 = w_flat[idx_extra2] - recon_base[idx_extra2]
        lib_r3 = train_cb(res_r2, K=atoms_per_family, n_iter=5)
        d_r3 = torch.cdist(res_r2, lib_r3)
        recon_base[idx_extra2] += lib_r3[torch.argmin(d_r3, dim=1)]
        
    actual_bpp = 2.0 + (num_extra1 / N) + (num_extra2 / N if target_bpp > 3.0 else 0.0)
    return recon_base.view(orig_shape), actual_bpp

def evaluate_llm_quality(model, tokenizer, test_text, prompts):
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
        ans = tokenizer.decode(out[0], skip_special_tokens=True).replace("\n", " ").strip()
        answers.append(ans)
        
    return ppl, answers

def main():
    print("=" * 90)
    print("EXPERIMENTO DE PRECISÃO ADAPTATIVA GUIADA POR SENSIBILIDADE")
    print("Quantizando TODAS AS 12 CAMADAS MLP do DistilGPT-2 Inteiro")
    print("Testando Estratégias: 2.00, 2.25, 2.50, 3.00 e 3.50 bpp")
    print("=" * 90)
    
    model_name = "distilbert/distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    base_model.eval()
    
    test_text = (
        "Artificial intelligence is transforming science, industry, and daily life. "
        "Researchers are developing models that can reason, write code, and assist humans."
    )
    prompts = [
        "The future of artificial intelligence is",
        "To solve a complex problem, one must first"
    ]
    
    # 0. Medição Original
    ppl_orig, gen_orig = evaluate_llm_quality(base_model, tokenizer, test_text, prompts)
    print(f"\n[0] MODELO ORIGINAL (16.0 bpp - Referência):")
    print(f"    - PPL: {ppl_orig:.2f}")
    print(f"    - Prompt 1: \"{gen_orig[0]}\"")
    print(f"    - Prompt 2: \"{gen_orig[1]}\"")
    
    target_bpps = [2.00, 2.25, 2.50, 3.00, 3.50]
    results = []
    
    for tbpp in target_bpps:
        print("\n" + "-" * 90)
        mode_desc = "Uniforme" if tbpp == 2.0 else f"Adaptativo (Protegendo Sensíveis)"
        print(f"-> EXECUTANDO MODELO INTEIRO COM {tbpp:.2f} BITS/PESO [{mode_desc}]...")
        print("-" * 90)
        
        m_test = copy.deepcopy(base_model)
        m_test.eval()
        
        t0 = time.time()
        actual_bpps = []
        for i, block in enumerate(m_test.transformer.h):
            with torch.no_grad():
                w_fc, b_fc = quantize_layer_sensitivity_adaptive(block.mlp.c_fc.weight.data, target_bpp=tbpp)
                block.mlp.c_fc.weight.data = w_fc
                actual_bpps.append(b_fc)
                
                w_proj, b_proj = quantize_layer_sensitivity_adaptive(block.mlp.c_proj.weight.data, target_bpp=tbpp)
                block.mlp.c_proj.weight.data = w_proj
                actual_bpps.append(b_proj)
                
        elapsed = time.time() - t0
        avg_bpp = sum(actual_bpps) / len(actual_bpps)
        
        ppl, answers = evaluate_llm_quality(m_test, tokenizer, test_text, prompts)
        print(f"   [12 Camadas Quantizadas em {elapsed:.2f}s | BPP Real: {avg_bpp:.2f}]")
        print(f"   -> Perplexidade: {ppl:.2f} (Original era {ppl_orig:.2f})")
        print(f"   -> Resposta: \"{answers[0]}\"")
        
        results.append({
            'target_bpp': tbpp,
            'actual_bpp': avg_bpp,
            'ppl': ppl,
            'elapsed': elapsed,
            'gen1': answers[0],
            'gen2': answers[1]
        })
        
    print("\n" + "=" * 90)
    print("QUADRO RESUMO: O IMPACTO DA PROTEÇÃO POR SENSIBILIDADE NA LLM INTEIRA")
    print("=" * 90)
    print(f"{'Estratégia':<20} | {'Bits / Peso':<12} | {'Perplexidade (PPL)':<20} | {'Delta vs Original'}")
    print("-" * 90)
    print(f"{'Original FP16':<20} | {'16.00 bpp':<12} | {ppl_orig:<20.2f} | {'0.00 (Referência)'}")
    for r in results:
        delta = r['ppl'] - ppl_orig
        lbl = f"{r['target_bpp']:.2f} bpp {'Uniforme' if r['target_bpp']==2.0 else 'Adaptativo'}"
        print(f"{lbl:<20} | {r['actual_bpp']:>7.2f} bpp  | {r['ppl']:<20.2f} | {delta:>+10.2f}")
    print("=" * 90)
    
    print("\nAMOSTRAS DE TEXTO GERADO:")
    for r in results:
        print(f"\n[{r['target_bpp']:.2f} bpp] -> PPL {r['ppl']:.2f}:")
        print(f"  Prompt 1: \"{r['gen1']}\"")
        print(f"  Prompt 2: \"{r['gen2']}\"")

if __name__ == '__main__':
    main()
