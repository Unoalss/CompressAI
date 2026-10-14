import torch
import numpy as np
import time

def train_flat_rvq_stage(residuals, K=32, n_iter=8):
    """Treina um estágio de RVQ plano com uma única tabela global de K centroides."""
    N, D = residuals.shape
    perm = torch.randperm(N)[:K]
    centroids = residuals[perm].clone()
    
    for _ in range(n_iter):
        dists = torch.cdist(residuals, centroids)
        labels = torch.argmin(dists, dim=1)
        for k in range(K):
            mask = (labels == k)
            if mask.sum() > 0:
                centroids[k] = residuals[mask].mean(dim=0)
                
    dists = torch.cdist(residuals, centroids)
    labels = torch.argmin(dists, dim=1)
    return centroids, labels

def train_conditional_keys_stage2(residuals, labels_stage1, K1=32, K2=32, n_iter=8):
    """
    Treina o Sistema de Chaves Hierárquicas:
    Para cada Chave 1 (k1 in 0..K1-1), cria uma sub-biblioteca especializada
    com K2 opções de resíduos específicas para aquele contexto macro.
    """
    N, D = residuals.shape
    conditional_codebooks = torch.zeros(K1, K2, D)
    labels_stage2 = torch.zeros(N, dtype=torch.long)
    
    for k1 in range(K1):
        mask = (labels_stage1 == k1)
        cluster_residuals = residuals[mask]
        num_in_cluster = cluster_residuals.shape[0]
        
        if num_in_cluster == 0:
            continue
        elif num_in_cluster < K2:
            # Se houver menos amostras que K2, preenche com as amostras existentes
            conditional_codebooks[k1, :num_in_cluster] = cluster_residuals
            # Atribui labels locais simples
            local_labels = torch.arange(num_in_cluster)
            labels_stage2[mask] = local_labels
        else:
            # K-means especializado para a sub-árvore do k1
            perm = torch.randperm(num_in_cluster)[:K2]
            c_local = cluster_residuals[perm].clone()
            for _ in range(n_iter):
                dists = torch.cdist(cluster_residuals, c_local)
                local_lbls = torch.argmin(dists, dim=1)
                for k2 in range(K2):
                    sub_mask = (local_lbls == k2)
                    if sub_mask.sum() > 0:
                        c_local[k2] = cluster_residuals[sub_mask].mean(dim=0)
            
            dists = torch.cdist(cluster_residuals, c_local)
            local_lbls = torch.argmin(dists, dim=1)
            conditional_codebooks[k1] = c_local
            labels_stage2[mask] = local_lbls
            
    return conditional_codebooks, labels_stage2

def main():
    print("=" * 80)
    print("BENCHMARK COMPARATIVO: TABELAS PLANAS vs CHAVES HIERÁRQUICAS")
    print("Mesmo Orçamento de Bits: Blocos de 8 pesos | 5 bits por Chave")
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
    
    print(f"\nCamada de Teste: {rows}x{cols} ({total_params:,} pesos FP16 = {total_params*2/1024:.1f} KB)")
    print(f"Total de Blocos: {N:,} blocos de {block_size} números")
    print(f"Cada Chave/Código tem 5 bits (32 opções)\n")
    
    # -------------------------------------------------------------
    # NÍVEL 1 (CHAVE MACRO / PADRÃO FUNDAMENTAL) - Comum aos dois
    # -------------------------------------------------------------
    print("[1] Treinando Chave 1 (Macro - 5 bits / 32 opções)...")
    cb1, labels1 = train_flat_rvq_stage(weights_flat, K=32, n_iter=10)
    recon_lvl1 = cb1[labels1]
    res1 = weights_flat - recon_lvl1
    
    mse_lvl1 = torch.mean(res1 ** 2).item()
    y_p1 = torch.matmul(recon_lvl1.half().view(rows, cols), x_test)
    cos_lvl1 = torch.nn.functional.cosine_similarity(y_real.view(1, -1).float(), y_p1.view(1, -1).float()).item()
    
    print(f"    - Nível 1 apenas (5 bits/bloco = 0.62 bpp): MSE = {mse_lvl1:.6f} | Cosseno = {cos_lvl1*100:.2f}%\n")
    
    # -------------------------------------------------------------
    # NÍVEL 2 - OPÇÃO A: TABELA PLANA (Flat RVQ)
    # A segunda tabela é uma única tabela de 32 vetores para todo mundo
    # -------------------------------------------------------------
    print("[2A] Treinando Nível 2 via TABELA PLANA (Flat RVQ)...")
    t0 = time.time()
    cb2_flat, labels2_flat = train_flat_rvq_stage(res1, K=32, n_iter=10)
    time_flat = time.time() - t0
    
    recon_flat = (recon_lvl1 + cb2_flat[labels2_flat]).half().view(rows, cols)
    res_flat = weights_flat - recon_flat.float().view(-1, block_size)
    mse_flat = torch.mean(res_flat ** 2).item()
    
    y_p_flat = torch.matmul(recon_flat, x_test)
    cos_flat = torch.nn.functional.cosine_similarity(y_real.view(1, -1).float(), y_p_flat.view(1, -1).float()).item()
    
    # -------------------------------------------------------------
    # NÍVEL 2 - OPÇÃO B: CHAVES HIERÁRQUICAS CONDICIONAIS
    # A Chave 1 seleciona uma sub-tabela especializada para a Chave 2
    # -------------------------------------------------------------
    print("[2B] Treinando Nível 2 via CHAVES HIERÁRQUICAS (Tree / Conditional)...")
    t0 = time.time()
    cb2_cond, labels2_cond = train_conditional_keys_stage2(res1, labels1, K1=32, K2=32, n_iter=10)
    time_cond = time.time() - t0
    
    # Reconstrução com Chaves Hierárquicas:
    # Para cada bloco i: Bloco = CB1[labels1[i]] + CB2[labels1[i], labels2_cond[i]]
    recon_cond_blocks = torch.zeros_like(weights_flat)
    for k1 in range(32):
        mask = (labels1 == k1)
        if mask.sum() > 0:
            sub_labels = labels2_cond[mask]
            sub_cb = cb2_cond[k1]
            recon_cond_blocks[mask] = cb1[k1] + sub_cb[sub_labels]
            
    recon_cond = recon_cond_blocks.half().view(rows, cols)
    res_cond = weights_flat - recon_cond.float().view(-1, block_size)
    mse_cond = torch.mean(res_cond ** 2).item()
    
    y_p_cond = torch.matmul(recon_cond, x_test)
    cos_cond = torch.nn.functional.cosine_similarity(y_real.view(1, -1).float(), y_p_cond.view(1, -1).float()).item()
    
    # -------------------------------------------------------------
    # COMPARATIVO DE ARMAZENAMENTO E DESEMPENHO
    # -------------------------------------------------------------
    # Armazenamento dos Índices nos dois casos:
    # 131.072 blocos * 10 bits = 1.310.720 bits = 160 KB (EXATAMENTE IGUAL!)
    indices_bytes = int(np.ceil(N * 10 / 8.0))
    
    # Dicionários:
    # Flat: CB1 (32*8*2 = 512B) + CB2 (32*8*2 = 512B) = 1.0 KB
    dict_flat_bytes = 2 * (32 * block_size * 2)
    # Hierárquico: CB1 (512B) + 32 sub-tabelas CB2 (32 * 512B = 16 KB) = 16.5 KB
    dict_cond_bytes = (32 * block_size * 2) + (32 * 32 * block_size * 2)
    
    total_flat_kb = (indices_bytes + dict_flat_bytes) / 1024
    total_cond_kb = (indices_bytes + dict_cond_bytes) / 1024
    
    print("\n" + "=" * 80)
    print("QUADRO COMPARATIVO: TABELA PLANA vs CHAVES HIERÁRQUICAS (10 bits = 1.25 bpp)")
    print("=" * 80)
    print(f"{'Métrica':<30} | {'Tabela Plana (Flat)':<20} | {'Chaves Hierárquicas':<20}")
    print("-" * 80)
    print(f"{'Bits de Índice por Bloco':<30} | {'10 bits (5 + 5)':<20} | {'10 bits (5 + 5)':<20}")
    print(f"{'Bits Efetivos por Peso':<30} | {'1.25 bits':<20} | {'1.25 bits':<20}")
    print(f"{'Armazenamento de Índices':<30} | {indices_bytes/1024:>16.1f} KB  | {indices_bytes/1024:>16.1f} KB")
    print(f"{'Tamanho do Dicionário':<30} | {dict_flat_bytes/1024:>16.2f} KB  | {dict_cond_bytes/1024:>16.2f} KB")
    print(f"{'Tamanho Total Comprimido':<30} | {total_flat_kb:>16.1f} KB  | {total_cond_kb:>16.1f} KB")
    print("-" * 80)
    print(f"{'Erro Quadrático Médio (MSE)':<30} | {mse_flat:>20.6f} | {mse_cond:>20.6f}")
    
    reducao_mse = (1.0 - (mse_cond / mse_flat)) * 100
    print(f"{'Queda no Erro MSE':<30} | {'-':<20} | {f'{reducao_mse:+.1f}% menor erro!':>20}")
    print(f"{'Fidelidade da Saída (Cosseno)':<30} | {cos_flat*100:>19.2f}% | {cos_cond*100:>19.2f}%")
    print("=" * 80)
    
    print("\n-> CONCLUSÃO TÉCNICA:")
    if mse_cond < mse_flat:
        print(f"O Sistema de Chaves Hierárquicas superou a Tabela Plana!")
        print(f"Com os mesmos 10 bits de índice, o erro caiu {reducao_mse:.1f}% e o cosseno subiu de {cos_flat*100:.2f}% para {cos_cond*100:.2f}%.")
        print(f"Custo adicional de dicionário: apenas {dict_cond_bytes/1024:.1f} KB (ainda minúsculo, cabendo com folga na cache L2/L3).")

if __name__ == '__main__':
    main()
