import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import copy
import time

def quantize_hierarchical_keys_4stages(weight_tensor, block_size=8, K=32, n_iter=6):
    """
    Quantiza o tensor de pesos usando Chaves Hierárquicas em 4 Estágios (2.5 bits/peso):
    - Chave 1: Macro (32 clusters)
    - Chave 2: Meso Condicional (32 sub-clusters por ramo da Chave 1)
    - Chave 3 e 4: Micro correções
    """
    orig_shape = weight_tensor.shape
    weights_flat = weight_tensor.float().reshape(-1, block_size)
    N = weights_flat.shape[0]
    
    # 1. Chave 1 (Macro)
    perm = torch.randperm(N)[:K]
    cb1 = weights_flat[perm].clone()
    for _ in range(n_iter):
        dists = torch.cdist(weights_flat, cb1)
        l1 = torch.argmin(dists, dim=1)
        for k in range(K):
            m = (l1 == k)
            if m.sum() > 0: cb1[k] = weights_flat[m].mean(dim=0)
    dists = torch.cdist(weights_flat, cb1)
    l1 = torch.argmin(dists, dim=1)
    recon = cb1[l1].clone()
    res = weights_flat - recon
    
    # 2. Chave 2 (Meso Condicional para cada cluster da Chave 1)
    cb2_cond = torch.zeros(K, K, block_size)
    for k1 in range(K):
        m = (l1 == k1)
        sub_res = res[m]
        if sub_res.shape[0] >= K:
            perm = torch.randperm(sub_res.shape[0])[:K]
            c_loc = sub_res[perm].clone()
            for _ in range(n_iter):
                dists = torch.cdist(sub_res, c_loc)
                lbls = torch.argmin(dists, dim=1)
                for k2 in range(K):
                    sm = (lbls == k2)
                    if sm.sum() > 0: c_loc[k2] = sub_res[sm].mean(dim=0)
            dists = torch.cdist(sub_res, c_loc)
            cb2_cond[k1] = c_loc
            sub_lbls = torch.argmin(dists, dim=1)
            recon[m] += c_loc[sub_lbls]
        elif sub_res.shape[0] > 0:
            recon[m] += sub_res.mean(dim=0)
            
    res = weights_flat - recon
    
    # 3 e 4. Chaves Micro
    for _ in range(2):
        perm = torch.randperm(N)[:K]
        c_micro = res[perm].clone()
        for _ in range(n_iter):
            dists = torch.cdist(res, c_micro)
            lbls = torch.argmin(dists, dim=1)
            for k in range(K):
                m = (lbls == k)
                if m.sum() > 0: c_micro[k] = res[m].mean(dim=0)
        dists = torch.cdist(res, c_micro)
        lbls = torch.argmin(dists, dim=1)
        recon += c_micro[lbls]
        res = weights_flat - recon
        
    return recon.view(orig_shape)

def evaluate(model, tokenizer, text, prompt):
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    with torch.no_grad():
        loss = model(input_ids, labels=input_ids).loss
        ppl = torch.exp(loss).item()
        
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=25, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    gen = tokenizer.decode(out[0], skip_special_tokens=True)
    return ppl, gen

def main():
    print("=" * 75)
    print("PROVA NA LLM: CHAVES HIERÁRQUICAS vs TABELA PLANA EM [5]x4")
    print("Modelo: DistilGPT-2 (82M) | Orçamento: 20 bits/bloco = 2.5 bits/peso")
    print("=" * 75)
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    model.eval()
    
    test_text = (
        "Artificial intelligence is transforming science, industry, and daily life. "
        "Researchers are developing models that can reason, write code, and assist humans."
    )
    test_prompt = "The future of artificial intelligence is"
    
    print("\nQuantizando 12 camadas Transformer com Chaves Hierárquicas [5]x4...")
    start_t = time.time()
    for block in model.transformer.h:
        with torch.no_grad():
            block.mlp.c_fc.weight.data = quantize_hierarchical_keys_4stages(block.mlp.c_fc.weight.data)
            block.mlp.c_proj.weight.data = quantize_hierarchical_keys_4stages(block.mlp.c_proj.weight.data)
    elapsed = time.time() - start_t
    print(f"Quantização concluída em {elapsed:.1f}s.")
    
    ppl, gen = evaluate(model, tokenizer, test_text, test_prompt)
    print(f"\n-> RESULTADO COM CHAVES HIERÁRQUICAS [5]x4:")
    print(f"   - Perplexidade: {ppl:.2f} (Tabela Plana era 131.03)")
    print(f"   - Geração: \"{gen}\"")

if __name__ == '__main__':
    main()
