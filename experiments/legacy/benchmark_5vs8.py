import torch
import numpy as np

def train_codebook(residuals, K, n_iter=8):
    """
    Treina um codebook de K vetores usando k-means nos resíduos atuais.
    """
    N, D = residuals.shape
    # Inicialização aleatória com amostras dos próprios resíduos
    perm = torch.randperm(N)[:K]
    centroids = residuals[perm].clone()
    
    for _ in range(n_iter):
        dists = torch.cdist(residuals, centroids)
        labels = torch.argmin(dists, dim=1)
        
        # Atualização dos centroides
        for k in range(K):
            mask = (labels == k)
            if mask.sum() > 0:
                centroids[k] = residuals[mask].mean(dim=0)
                
    dists = torch.cdist(residuals, centroids)
    labels = torch.argmin(dists, dim=1)
    return centroids, labels

def evaluate_rvq(weights_flat, block_size, bits_per_code, stages, weights_orig_matrix, x_test, y_real):
    """
    Executa Quantização Vetorial Residual (RVQ) para 'stages' etapas.
    bits_per_code = 5 (K=32) ou 8 (K=256).
    """
    K = 2 ** bits_per_code
    N = weights_flat.shape[0]
    total_params = N * block_size
    tamanho_orig_bytes = total_params * 2 # FP16
    
    current_residual = weights_flat.clone()
    reconstructed_blocks = torch.zeros_like(weights_flat)
    
    codebooks = []
    all_labels = []
    
    results_per_stage = []
    
    for m in range(1, stages + 1):
        cb, labels = train_codebook(current_residual, K=K, n_iter=8)
        codebooks.append(cb)
        all_labels.append(labels)
        
        # Atualiza a aproximação somando o novo codebook
        reconstructed_blocks += cb[labels]
        current_residual = weights_flat - reconstructed_blocks
        
        # Medições da etapa atual
        reconstructed_matrix = reconstructed_blocks.half().view(weights_orig_matrix.shape)
        mse = torch.mean((weights_orig_matrix.float() - reconstructed_matrix.float()) ** 2).item()
        
        y_pred = torch.matmul(reconstructed_matrix, x_test)
        cos_sim = torch.nn.functional.cosine_similarity(y_real.view(1, -1).float(), y_pred.view(1, -1).float()).item()
        
        # Cálculo exato de bits e bytes
        # Tamanho dos codebooks (dicionários): m * K * block_size * 2 bytes (FP16)
        size_cb_bytes = m * K * block_size * 2
        # Tamanho dos índices: N blocos * (m * bits_per_code) bits
        total_index_bits = N * m * bits_per_code
        size_indices_bytes = int(np.ceil(total_index_bits / 8.0))
        
        total_compressed_bytes = size_cb_bytes + size_indices_bytes
        reducao = 100 * (1 - total_compressed_bytes / tamanho_orig_bytes)
        bits_per_weight = (total_index_bits) / total_params
        
        results_per_stage.append({
            'stage': m,
            'bits_pw': bits_per_weight,
            'size_kb': total_compressed_bytes / 1024,
            'cb_kb': size_cb_bytes / 1024,
            'reducao': reducao,
            'mse': mse,
            'cos_sim': cos_sim
        })
        
    return results_per_stage

def main():
    print("=" * 80)
    print("BATALHA DE ARQUITETURAS: SISTEMA [5] vs SISTEMA [8]")
    print("Quantização Residual Aditiva com Blocos de 8 Pesos")
    print("=" * 80)
    
    torch.manual_seed(42)
    rows, cols = 1024, 1024
    total_params = rows * cols
    block_size = 8
    
    # Criar camada com distribuição realista
    std = np.sqrt(2.0 / cols)
    weights = torch.randn(rows, cols) * std
    weights_fp16 = weights.half()
    weights_flat = weights_fp16.view(-1, block_size).float()
    
    # Ativação de teste
    x_test = torch.randn(cols, 1).half()
    y_real = torch.matmul(weights_fp16, x_test)
    
    orig_kb = (total_params * 2) / 1024
    print(f"\nMatriz: {rows}x{cols} ({total_params:,} pesos FP16 = {orig_kb:.1f} KB)")
    print(f"Bloco elementar: {block_size} pesos")
    print(f"Total de blocos: {weights_flat.shape[0]:,}\n")
    
    print("[Executando RVQ para Sistema [5] (K=32 centroides por estágio)...]")
    res_5 = evaluate_rvq(weights_flat, block_size, bits_per_code=5, stages=8, 
                         weights_orig_matrix=weights_fp16, x_test=x_test, y_real=y_real)
    
    print("[Executando RVQ para Sistema [8] (K=256 centroides por estágio)...]")
    res_8 = evaluate_rvq(weights_flat, block_size, bits_per_code=8, stages=8, 
                         weights_orig_matrix=weights_fp16, x_test=x_test, y_real=y_real)
    
    print("\n" + "=" * 80)
    print("TABELA COMPARATIVA POR NÍVEL (ESTÁGIOS 1 a 8)")
    print("=" * 80)
    print(f"{'Config':<10} | {'Bits/Peso':<10} | {'Tamanho':<10} | {'Dicionário':<11} | {'Redução':<8} | {'MSE':<10} | {'Fidelidade'}")
    print("-" * 80)
    
    # Exibir estágios chave: 1, 2, 4, 8
    stages_to_show = [1, 2, 3, 4, 6, 8]
    
    for s in stages_to_show:
        r5 = res_5[s-1]
        name5 = f"[5]x{s}"
        print(f"{name5:<10} | {r5['bits_pw']:>7.2f} b | {r5['size_kb']:>7.1f} KB | {r5['cb_kb']:>7.2f} KB  | {r5['reducao']:>6.1f}% | {r5['mse']:<10.6f} | {r5['cos_sim']*100:.2f}%")
        
        r8 = res_8[s-1]
        name8 = f"[8]x{s}"
        print(f"{name8:<10} | {r8['bits_pw']:>7.2f} b | {r8['size_kb']:>7.1f} KB | {r8['cb_kb']:>7.2f} KB  | {r8['reducao']:>6.1f}% | {r8['mse']:<10.6f} | {r8['cos_sim']*100:.2f}%")
        print("-" * 80)
        
    print("\n" + "=" * 80)
    print("ANÁLISE DO MARCO: 5 BYTES vs 8 BYTES (ESTÁGIO 8)")
    print("=" * 80)
    r5_final = res_5[7]
    r8_final = res_8[7]
    print(f"-> [5]x8 (40 bits = exatamente 5 BYTES por bloco):")
    print(f"   - Combinações: 32^8 = 1.099 trilhões")
    print(f"   - Dicionário Total: apenas {r5_final['cb_kb']:.2f} KB (extremamente leve na cache L1/L2!)")
    print(f"   - Tamanho Total: {r5_final['size_kb']:.1f} KB (Redução de {r5_final['reducao']:.1f}%)")
    print(f"   - MSE: {r5_final['mse']:.8f}")
    print(f"   - Fidelidade da Ativação: {r5_final['cos_sim']*100:.4f}%")
    
    print(f"\n-> [8]x8 (64 bits = exatamente 8 BYTES por bloco):")
    print(f"   - Combinações: 256^8 = 18.45 quintilhões")
    print(f"   - Dicionário Total: {r8_final['cb_kb']:.2f} KB")
    print(f"   - Tamanho Total: {r8_final['size_kb']:.1f} KB (Redução de {r8_final['reducao']:.1f}%)")
    print(f"   - MSE: {r8_final['mse']:.8f}")
    print(f"   - Fidelidade da Ativação: {r8_final['cos_sim']*100:.4f}%")

if __name__ == '__main__':
    main()
