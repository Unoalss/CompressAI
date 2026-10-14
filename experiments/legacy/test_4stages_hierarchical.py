import torch
import numpy as np

def run_multi_stage_hierarchical():
    print("=" * 80)
    print("TESTE AVANÇADO: CHAVES HIERÁRQUICAS EM 4 ESTÁGIOS [5]x4 (2.5 bits/peso)")
    print("Comparando Tabela Plana vs Chaves Hierárquicas")
    print("=" * 80)
    
    torch.manual_seed(42)
    rows, cols = 1024, 1024
    total_params = rows * cols
    block_size = 8
    
    std = np.sqrt(2.0 / cols)
    weights = torch.randn(rows, cols) * std
    weights_fp16 = weights.half()
    weights_flat = weights_fp16.view(-1, block_size).float()
    N = weights_flat.shape[0]
    
    x_test = torch.randn(cols, 1).half()
    y_real = torch.matmul(weights_fp16, x_test)
    
    # -------------------------------------------------------------
    # 1. FLAT RVQ - 4 ESTÁGIOS DE 5 BITS
    # -------------------------------------------------------------
    res_flat = weights_flat.clone()
    recon_flat = torch.zeros_like(weights_flat)
    K = 32
    
    for stage in range(4):
        perm = torch.randperm(N)[:K]
        centroids = res_flat[perm].clone()
        for _ in range(8):
            dists = torch.cdist(res_flat, centroids)
            labels = torch.argmin(dists, dim=1)
            for k in range(K):
                m = (labels == k)
                if m.sum() > 0:
                    centroids[k] = res_flat[m].mean(dim=0)
        dists = torch.cdist(res_flat, centroids)
        labels = torch.argmin(dists, dim=1)
        recon_flat += centroids[labels]
        res_flat = weights_flat - recon_flat
        
    mse_flat_4 = torch.mean(res_flat ** 2).item()
    mat_flat = recon_flat.half().view(rows, cols)
    cos_flat_4 = torch.nn.functional.cosine_similarity(y_real.view(1, -1).float(), torch.matmul(mat_flat, x_test).view(1, -1).float()).item()
    
    # -------------------------------------------------------------
    # 2. CHAVES HIERÁRQUICAS (2 Níveis Macro/Meso Condicionais + 2 Níveis Micro)
    # Chave 1: Macro (32 clusters)
    # Chave 2: Meso Condicional (32 sub-clusters para cada um dos 32 ramos)
    # Chave 3 e 4: Resíduos Micro
    # -------------------------------------------------------------
    # Nível 1: Chave 1
    perm = torch.randperm(N)[:K]
    cb1 = weights_flat[perm].clone()
    for _ in range(8):
        dists = torch.cdist(weights_flat, cb1)
        l1 = torch.argmin(dists, dim=1)
        for k in range(K):
            m = (l1 == k)
            if m.sum() > 0: cb1[k] = weights_flat[m].mean(dim=0)
    dists = torch.cdist(weights_flat, cb1)
    l1 = torch.argmin(dists, dim=1)
    recon_hier = cb1[l1].clone()
    res_hier = weights_flat - recon_hier
    
    # Nível 2: Chave 2 Condicional por Chave 1
    cb2_cond = torch.zeros(K, K, block_size)
    l2 = torch.zeros(N, dtype=torch.long)
    for k1 in range(K):
        m = (l1 == k1)
        sub_res = res_hier[m]
        if sub_res.shape[0] >= K:
            perm = torch.randperm(sub_res.shape[0])[:K]
            c_loc = sub_res[perm].clone()
            for _ in range(8):
                dists = torch.cdist(sub_res, c_loc)
                lbls = torch.argmin(dists, dim=1)
                for k2 in range(K):
                    sm = (lbls == k2)
                    if sm.sum() > 0: c_loc[k2] = sub_res[sm].mean(dim=0)
            dists = torch.cdist(sub_res, c_loc)
            cb2_cond[k1] = c_loc
            l2[m] = torch.argmin(dists, dim=1)
            recon_hier[m] += c_loc[l2[m]]
        elif sub_res.shape[0] > 0:
            recon_hier[m] += sub_res.mean(dim=0)
            
    res_hier = weights_flat - recon_hier
    
    # Níveis 3 e 4: Correções Micro Aditivas
    for stage in [3, 4]:
        perm = torch.randperm(N)[:K]
        c_micro = res_hier[perm].clone()
        for _ in range(8):
            dists = torch.cdist(res_hier, c_micro)
            lbls = torch.argmin(dists, dim=1)
            for k in range(K):
                m = (lbls == k)
                if m.sum() > 0: c_micro[k] = res_hier[m].mean(dim=0)
        dists = torch.cdist(res_hier, c_micro)
        lbls = torch.argmin(dists, dim=1)
        recon_hier += c_micro[lbls]
        res_hier = weights_flat - recon_hier
        
    mse_hier_4 = torch.mean(res_hier ** 2).item()
    mat_hier = recon_hier.half().view(rows, cols)
    cos_hier_4 = torch.nn.functional.cosine_similarity(y_real.view(1, -1).float(), torch.matmul(mat_hier, x_test).view(1, -1).float()).item()
    
    print("\n" + "=" * 80)
    print("RESULTADO DO TESTE DE 4 CHAVES [5]x4 (20 BITS = 2.5 BITS/PESO)")
    print("=" * 80)
    print(f"{'Arquitetura':<25} | {'Bits/Peso':<12} | {'MSE':<12} | {'Fidelidade (Cosseno)'}")
    print("-" * 75)
    print(f"{'Tabela Plana [5]x4':<25} | {'2.50 bits':<12} | {mse_flat_4:<12.6f} | {cos_flat_4*100:.2f}%")
    print(f"{'Chaves Hierárquicas':<25} | {'2.50 bits':<12} | {mse_hier_4:<12.6f} | {cos_hier_4*100:.2f}%")
    print("=" * 75)
    ganho_mse = (1 - mse_hier_4/mse_flat_4) * 100
    print(f"Redução adicional no MSE graças à estrutura de chaves: {ganho_mse:+.1f}%!")

if __name__ == '__main__':
    run_multi_stage_hierarchical()
