import torch
from transformers import AutoModelForCausalLM
import numpy as np

def audit_cache_and_reuse():
    print("=" * 80)
    print("AUDITORIA MATEMÁTICA ESTRITA: REUTILIZAÇÃO, CACHE E ERRO REAL")
    print("Camada: transformer.h[0].mlp.c_fc (DistilGPT-2)")
    print("=" * 80)
    
    # 1. Carregar pesos reais da camada
    model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    weight = model.transformer.h[0].mlp.c_fc.weight.data.float() # (768, 3072)
    block_size = 8
    weights_flat = weight.reshape(-1, block_size)
    
    # Amostra de 50.000 blocos (400.000 pesos)
    sample_size = 50000
    w_sample = weights_flat[:sample_size].clone()
    
    # Entrada de teste para medir cosseno
    x_test = torch.randn(block_size, 1)
    
    # -------------------------------------------------------------
    # PERGUNTA 1: QUANTOS SÃO REPETIÇÃO EXATA (BIT A BIT)?
    # -------------------------------------------------------------
    cache_26 = []
    exact_reused = 0
    
    for i in range(sample_size):
        b = w_sample[i]
        if len(cache_26) > 0:
            c_mat = torch.stack(cache_26)
            # Checa se todos os 8 números são exatamente iguais
            is_exact = torch.all(c_mat == b, dim=1)
            if torch.any(is_exact):
                exact_reused += 1
        cache_26.append(b)
        if len(cache_26) > 26:
            cache_26.pop(0)
            
    # -------------------------------------------------------------
    # PERGUNTA 2, 3, 4 e 5: AUDITORIA DE TOLERÂNCIA, TAMANHO DO CACHE E MSE
    # Vamos testar 3 regimes de tolerância:
    # A) RÍGIDO (Tolerância baixa: MSE < 0.0005)
    # B) MODERADO (Tolerância média: MSE < 0.002)
    # C) FROUXO (O que o teste anterior usou: MSE < 0.015)
    # -------------------------------------------------------------
    tolerances = [0.0005, 0.002, 0.015]
    
    # Treinar bibliotecas de 26 átomos
    def train_cb(vecs, K=26):
        perm = torch.randperm(vecs.shape[0])[:K]
        c = vecs[perm].clone()
        for _ in range(6):
            d = torch.cdist(vecs, c)
            lbl = torch.argmin(d, dim=1)
            for k in range(K):
                m = (lbl == k)
                if m.sum() > 0: c[k] = vecs[m].mean(dim=0)
        return c
        
    lib_macro = train_cb(w_sample, K=26)
    d = torch.cdist(w_sample, lib_macro)
    res_train = w_sample - lib_macro[torch.argmin(d, dim=1)]
    lib_res = train_cb(res_train, K=26)
    
    print("\n[RESULTADOS DA AUDITORIA]")
    print(f"1. Repetições Exatas Bit-a-Bit (Tolerância = 0.0):")
    print(f"   -> {exact_reused} blocos de {sample_size:,} ({exact_reused/sample_size*100:.4f}%)")
    print(f"   -> Conclusão: Em pesos contínuos, repetição bit-a-bit é virtualmente ZERO.")
    
    print("\n" + "=" * 80)
    print("ANÁLISE DE REUTILIZAÇÃO POR TOLERÂNCIA (Cache LRU = 26 blocos / 5 bits)")
    print("=" * 80)
    print(f"{'Regime':<12} | {'Tolerância MSE':<15} | {'% KEY_REF':<12} | {'Bits/Peso':<12} | {'MSE Final':<12} | {'Cosseno'}")
    print("-" * 80)
    
    for tol in tolerances:
        cache = []
        recon_blocks = []
        reused_count = 0
        composite_count = 0
        total_tokens = 0
        
        for i in range(sample_size):
            b = w_sample[i]
            reused = False
            
            if len(cache) > 0:
                c_mat = torch.stack(cache)
                dists_c = torch.mean((c_mat - b) ** 2, dim=1)
                min_d, best_idx = torch.min(dists_c, dim=0)
                if min_d.item() < tol:
                    reused_count += 1
                    reused = True
                    total_tokens += 2 # [KEY_REF, ID_CACHE] = 10 bits
                    chosen_b = cache[best_idx.item()]
                    recon_blocks.append(chosen_b)
                    cache.append(chosen_b)
                    if len(cache) > 26: cache.pop(0)
                    continue
                    
            if not reused:
                # Composição Macro + Resíduo: [Macro, KEY_ADD, Res, KEY_END] = 4 tokens (20 bits)
                d_m = torch.mean((lib_macro - b) ** 2, dim=1)
                best_m = torch.argmin(d_m).item()
                macro_v = lib_macro[best_m]
                r = b - macro_v
                d_r = torch.mean((lib_res - r) ** 2, dim=1)
                best_r = torch.argmin(d_r).item()
                
                composite_count += 1
                total_tokens += 4
                recon_b = macro_v + lib_res[best_r]
                recon_blocks.append(recon_b)
                cache.append(recon_b)
                if len(cache) > 26: cache.pop(0)
                
        recon_mat = torch.stack(recon_blocks)
        mse_final = torch.mean((w_sample - recon_mat) ** 2).item()
        
        # Cosseno da ativação
        cos_sim = torch.nn.functional.cosine_similarity(
            w_sample.view(1, -1), recon_mat.view(1, -1)
        ).item()
        
        bpp = (total_tokens * 5) / (sample_size * block_size)
        regime_name = "Rígido" if tol < 0.001 else ("Moderado" if tol < 0.005 else "Frouxo")
        print(f"{regime_name:<12} | {tol:<15.4f} | {reused_count/sample_size*100:>10.1f}% | {bpp:>10.2f} b | {mse_final:<12.6f} | {cos_sim*100:.2f}%")
        
    print("=" * 80)

if __name__ == '__main__':
    audit_cache_and_reuse()
