import numpy as np
import torch
import time

def run_benchmark():
    print("=" * 70)
    print("PROTÓTIPO: COMPRESSÃO DE PESOS POR CODEBOOK & QUANTIZAÇÃO RESIDUAL")
    print("=" * 70)
    
    # 1. GERAR PESOS REALISTAS DE UMA CAMADA DE REDE NEURAL (ex: Linear Layer)
    # Matriz 1024 x 1024 = 1.048.576 parâmetros contínuos
    torch.manual_seed(42)
    rows, cols = 1024, 1024
    total_params = rows * cols
    
    # Distribuição típica de pesos (Kaiming / Xavier normal)
    std = np.sqrt(2.0 / cols)
    weights_fp32 = torch.randn(rows, cols) * std
    weights_fp16 = weights_fp32.half() # Representação padrão de inferência moderna
    
    # Entrada de teste para medir o erro na ativação real (Y = W * X)
    x_test = torch.randn(cols, 1).half()
    y_real = torch.matmul(weights_fp16, x_test)
    
    tamanho_original_bytes = total_params * 2 # 2 bytes por float16
    print(f"\n[1] CAMADA ORIGINAL:")
    print(f"    - Parâmetros: {total_params:,}")
    print(f"    - Precisão: Float16 (2 bytes / peso)")
    print(f"    - Tamanho na Memória: {tamanho_original_bytes / 1024:.2f} KB ({tamanho_original_bytes / (1024*1024):.2f} MB)")

    # -------------------------------------------------------------
    # 2. VERSÃO A: TENTATIVA DE BLOCOS IDÊNTICOS (SEM PERDA)
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("[2] VERSÃO A: BUSCA POR BLOCOS IDÊNTICOS (Lossless)")
    print("-" * 70)
    
    # Vamos dividir em blocos de tamanho 8
    block_size = 8
    num_blocks = total_params // block_size
    weights_flat = weights_fp16.view(-1, block_size)
    
    # Converter para tuplas de bits/valores para checar duplicatas exatas
    blocks_bytes = [weights_flat[i].numpy().tobytes() for i in range(num_blocks)]
    unique_blocks = set(blocks_bytes)
    
    print(f"    - Tamanho do bloco: {block_size} pesos")
    print(f"    - Total de blocos: {num_blocks:,}")
    print(f"    - Blocos únicos encontrados: {len(unique_blocks):,}")
    
    taxa_unicos = (len(unique_blocks) / num_blocks) * 100
    print(f"    - % de blocos únicos: {taxa_unicos:.2f}%")
    print("    -> CONCLUSÃO DA VERSÃO A:")
    print("       Em pesos contínuos reais, praticamente 100% dos blocos são únicos.")
    print("       Compressão exata por repetição resulta em 0% de economia (e até overhead).")

    # -------------------------------------------------------------
    # 3. VERSÃO B1: QUANTIZAÇÃO VETORIAL SIMPLES (1 CODEBOOK DE 256 CENTROIDES)
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("[3] VERSÃO B1: QUANTIZAÇÃO VETORIAL COM 1 CODEBOOK (256 Centroides / 8 bits)")
    print("-" * 70)
    
    # Codebook com 256 vetores de tamanho 8 (1 byte por índice de bloco)
    K = 256
    X_train = weights_flat.float()
    
    # K-means simples para encontrar os 256 centroides representativos
    # Inicialização aleatória com dados
    indices = torch.randperm(num_blocks)[:K]
    centroids_1 = X_train[indices].clone()
    
    # 10 iterações de Lloyd / K-Means
    for _ in range(10):
        # Distâncias euclidianas quadradas
        dists = torch.cdist(X_train, centroids_1)
        labels_1 = torch.argmin(dists, dim=1)
        for k in range(K):
            mask = (labels_1 == k)
            if mask.sum() > 0:
                centroids_1[k] = X_train[mask].mean(dim=0)
                
    # Reconstrução B1
    reconstructed_b1 = centroids_1[labels_1].half().view(rows, cols)
    
    # Cálculo de Tamanho B1:
    # Codebook: 256 vetores * 8 floats * 2 bytes = 4.096 bytes (4 KB)
    # IDs: 131.072 blocos * 1 byte (uint8) = 131.072 bytes (128 KB)
    size_codebook_b1 = K * block_size * 2
    size_indices_b1 = num_blocks * 1 # 1 byte porque 256 cabe em uint8
    total_b1 = size_codebook_b1 + size_indices_b1
    reducao_b1 = 100 * (1 - total_b1 / tamanho_original_bytes)
    
    # Erro B1
    mse_b1 = torch.mean((weights_fp16.float() - reconstructed_b1.float()) ** 2).item()
    y_pred_b1 = torch.matmul(reconstructed_b1, x_test)
    cos_sim_b1 = torch.nn.functional.cosine_similarity(y_real.view(1, -1).float(), y_pred_b1.view(1, -1).float()).item()
    
    print(f"    - Tamanho Original:    {tamanho_original_bytes / 1024:.2f} KB")
    print(f"    - Codebook (Dicionário): {size_codebook_b1 / 1024:.2f} KB")
    print(f"    - Índices (IDs 1 byte):  {size_indices_b1 / 1024:.2f} KB")
    print(f"    - Total Comprimido:      {total_b1 / 1024:.2f} KB")
    print(f"    - Redução de Tamanho:    {reducao_b1:.2f}%")
    print(f"    - Erro Médio Quadrático (MSE): {mse_b1:.6f}")
    print(f"    - Similaridade de Saída (Cosseno): {cos_sim_b1 * 100:.2f}%")

    # -------------------------------------------------------------
    # 4. VERSÃO B2: QUANTIZAÇÃO RESIDUAL ADITIVA (ESTILO AQLM / RVQ - 2 CODEBOOKS)
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("[4] VERSÃO B2: QUANTIZAÇÃO ADITIVA (2 CODEBOOKS SOMADOS - Base + Resíduo)")
    print("-" * 70)
    
    # Resíduo é o que o primeiro codebook não conseguiu capturar
    residuals = X_train - centroids_1[labels_1]
    
    # Treinar segundo Codebook de 256 centroides APENAS para os erros residuais
    indices_r = torch.randperm(num_blocks)[:K]
    centroids_2 = residuals[indices_r].clone()
    
    for _ in range(10):
        dists = torch.cdist(residuals, centroids_2)
        labels_2 = torch.argmin(dists, dim=1)
        for k in range(K):
            mask = (labels_2 == k)
            if mask.sum() > 0:
                centroids_2[k] = residuals[mask].mean(dim=0)
                
    # Reconstrução Aditiva: Bloco = Codebook1[ID1] + Codebook2[ID2]
    reconstructed_b2 = (centroids_1[labels_1] + centroids_2[labels_2]).half().view(rows, cols)
    
    # Cálculo de Tamanho B2:
    # 2 Codebooks: 2 * (256 * 8 * 2) = 8.192 bytes (8 KB)
    # 2 IDs por bloco: 131.072 blocos * 2 bytes = 262.144 bytes (256 KB)
    # Ou seja: 16 bits (2 bytes) representam 8 pesos -> média de 2 bits por peso!
    size_codebooks_b2 = 2 * (K * block_size * 2)
    size_indices_b2 = num_blocks * 2 # 2 bytes (1 byte para cada codebook)
    total_b2 = size_codebooks_b2 + size_indices_b2
    reducao_b2 = 100 * (1 - total_b2 / tamanho_original_bytes)
    
    # Erro B2
    mse_b2 = torch.mean((weights_fp16.float() - reconstructed_b2.float()) ** 2).item()
    y_pred_b2 = torch.matmul(reconstructed_b2, x_test)
    cos_sim_b2 = torch.nn.functional.cosine_similarity(y_real.view(1, -1).float(), y_pred_b2.view(1, -1).float()).item()
    
    print(f"    - Tamanho Original:    {tamanho_original_bytes / 1024:.2f} KB")
    print(f"    - 2 Codebooks (Dics):    {size_codebooks_b2 / 1024:.2f} KB")
    print(f"    - Índices (IDs 2 bytes): {size_indices_b2 / 1024:.2f} KB")
    print(f"    - Total Comprimido:      {total_b2 / 1024:.2f} KB")
    print(f"    - Redução de Tamanho:    {reducao_b2:.2f}%")
    print(f"    - Erro Médio Quadrático (MSE): {mse_b2:.6f}  (Reduziu o erro em {(1 - mse_b2/mse_b1)*100:.1f}% vs B1!)")
    print(f"    - Similaridade de Saída (Cosseno): {cos_sim_b2 * 100:.2f}%")

    print("\n" + "=" * 70)
    print("QUADRO COMPARATIVO FINAL")
    print("=" * 70)
    print(f"{'Método':<25} | {'Tamanho':<10} | {'Redução':<10} | {'MSE':<12} | {'Fidelidade (Cosseno)'}")
    print("-" * 75)
    print(f"{'Original (FP16)':<25} | {tamanho_original_bytes/1024:>7.1f} KB | {'0.0%':>10} | {'0.000000':<12} | {'100.00%'}")
    print(f"{'Versão A (Exato)':<25} | {tamanho_original_bytes/1024:>7.1f} KB | {'0.0%':>10} | {'0.000000':<12} | {'Inviável (100% únicos)'}")
    print(f"{'Versão B1 (1 Codebook)':<25} | {total_b1/1024:>7.1f} KB | {reducao_b1:>9.1f}% | {mse_b1:<12.6f} | {cos_sim_b1*100:.2f}%")
    print(f"{'Versão B2 (2 Additive Cb)':<25} | {total_b2/1024:>7.1f} KB | {reducao_b2:>9.1f}% | {mse_b2:<12.6f} | {cos_sim_b2*100:.2f}%")
    print("=" * 70)

if __name__ == '__main__':
    run_benchmark()
