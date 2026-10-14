import torch
from transformers import AutoModelForCausalLM
import numpy as np

KEY_REF   = 26  # Reutilização de bloco recente (Ponteiro)
KEY_ADD   = 27  # Composição Aditiva
KEY_SHORT = 30  # Bloco Simples (1 átomo é suficiente)
KEY_END   = 31  # Fim de Bloco

def train_codebook(vectors, K=26, n_iter=8):
    N, D = vectors.shape
    perm = torch.randperm(N)[:K]
    centroids = vectors[perm].clone()
    for _ in range(n_iter):
        dists = torch.cdist(vectors, centroids)
        labels = torch.argmin(dists, dim=1)
        for k in range(K):
            mask = (labels == k)
            if mask.sum() > 0:
                centroids[k] = vectors[mask].mean(dim=0)
    return centroids

def test_vm_on_llm_layer():
    print("=" * 80)
    print("TESTE DA MÁQUINA DE 26 DADOS + 6 CHAVES EM PESOS REAIS DO DISTILGPT-2")
    print("Camada: transformer.h[0].mlp.c_fc (2.359.296 pesos reais)")
    print("=" * 80)
    
    # Carregar modelo
    model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    weight = model.transformer.h[0].mlp.c_fc.weight.data.float() # (768, 3072)
    block_size = 8
    weights_flat = weight.reshape(-1, block_size)
    N = weights_flat.shape[0] # 294.912 blocos
    
    # Amostrar 50.000 blocos para teste ágil
    sample_size = 50000
    w_sample = weights_flat[:sample_size].clone()
    
    # 1. Treinar 26 átomos macro e 26 de resíduo
    print(f"Treinando 26 átomos macro em {sample_size:,} blocos reais...")
    lib_macro = train_codebook(w_sample, K=26, n_iter=6)
    
    dists = torch.cdist(w_sample, lib_macro)
    lbls = torch.argmin(dists, dim=1)
    res_sample = w_sample - lib_macro[lbls]
    
    print("Treinando 26 átomos de resíduo...")
    lib_residual = train_codebook(res_sample, K=26, n_iter=6)
    
    # Testar com tolerância adaptativa calibrada pelo MSE médio
    avg_macro_mse = torch.mean(res_sample ** 2).item()
    print(f"MSE médio da aproximação macro: {avg_macro_mse:.6f}")
    
    # Tolerância de reutilização: se a distância for compatível com o erro aceitável da quantização
    tol_ref = avg_macro_mse * 1.5
    tol_short = avg_macro_mse * 0.5
    
    bytecode = []
    cache_blocks = []
    cache_size = 26
    
    stats = {'reused': 0, 'short': 0, 'composite': 0}
    
    for i in range(sample_size):
        b = w_sample[i]
        reused = False
        
        # 1. Checa Cache de Reutilização
        if len(cache_blocks) > 0:
            c_mat = torch.stack(cache_blocks)
            dists_c = torch.mean((c_mat - b) ** 2, dim=1)
            min_d, best_idx = torch.min(dists_c, dim=0)
            if min_d.item() < tol_ref:
                bytecode.extend([KEY_REF, best_idx.item()])
                stats['reused'] += 1
                reused = True
                cache_blocks.append(cache_blocks[best_idx.item()])
                if len(cache_blocks) > cache_size: cache_blocks.pop(0)
                continue
                
        if not reused:
            dists_m = torch.mean((lib_macro - b) ** 2, dim=1)
            best_m = torch.argmin(dists_m).item()
            macro_v = lib_macro[best_m]
            r = b - macro_v
            r_mse = torch.mean(r ** 2).item()
            
            if r_mse < tol_short:
                bytecode.extend([best_m, KEY_SHORT])
                stats['short'] += 1
                cache_blocks.append(macro_v)
            else:
                dists_r = torch.mean((lib_residual - r) ** 2, dim=1)
                best_r = torch.argmin(dists_r).item()
                bytecode.extend([best_m, KEY_ADD, best_r, KEY_END])
                stats['composite'] += 1
                cache_blocks.append(macro_v + lib_residual[best_r])
                
            if len(cache_blocks) > cache_size: cache_blocks.pop(0)
            
    total_tokens = len(bytecode)
    total_bits = total_tokens * 5
    bpp = total_bits / (sample_size * block_size)
    
    print("\n" + "=" * 80)
    print("COMPORTAMENTO DAS CHAVES EM PESOS REAIS DO TRANSFORMER")
    print("=" * 80)
    print(f"Blocos Analisados:               {sample_size:,}")
    print(f"-> Blocos REUTILIZADOS (KEY_REF): {stats['reused']:,} ({stats['reused']/sample_size*100:.2f}%)")
    print(f"-> Blocos SIMPLES (KEY_SHORT):    {stats['short']:,} ({stats['short']/sample_size*100:.2f}%)")
    print(f"-> Blocos COMPOSTOS (KEY_ADD):    {stats['composite']:,} ({stats['composite']/sample_size*100:.2f}%)")
    print("-" * 80)
    print(f"Total de Tokens de 5 bits:       {total_tokens:,}")
    print(f"Média Efetiva de Bits por Peso:  {bpp:.2f} bits/peso (vs 5.0 bits do sistema fixo!)")
    reducao_vs_fixo = (1 - bpp / 5.0) * 100
    print(f"Economia vs Sistema Fixo [5]x8:   {reducao_vs_fixo:+.1f}% menos bits!")
    print("=" * 80)

if __name__ == '__main__':
    test_vm_on_llm_layer()
