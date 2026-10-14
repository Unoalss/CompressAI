import torch
import numpy as np
import time

# =====================================================================
# DEFINIÇÃO DO INSTRUCTION SET ARCHITECTURE (ISA) DE 5 BITS
# 32 Códigos: 26 Dados (0-25) + 6 Chaves de Controle (26-31)
# =====================================================================
KEY_REF   = 26  # Instrução 11010: Reutiliza bloco do cache (Ponteiro)
KEY_ADD   = 27  # Instrução 11011: Composição Aditiva (Soma próximo átomo como resíduo)
KEY_SCALE = 28  # Instrução 11100: Modificador de Escala (multiplica por fator)
KEY_LIB   = 29  # Instrução 11101: Troca de Biblioteca de Átomos
KEY_SHORT = 30  # Instrução 11110: Bloco Simples (Termina imediatamente após 1 átomo)
KEY_END   = 31  # Instrução 11111: Fim de Bloco (Terminador de instrução variável)

SCALE_FACTORS = [0.25, 0.5, 1.0, 2.0] # 4 fatores selecionáveis via dados

def train_codebook(vectors, K=26, n_iter=8):
    """Treina uma biblioteca de K=26 átomos fundamentais."""
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

class CompressionVM:
    """
    Máquina Virtual que decodifica o fluxo de tokens de 5 bits
    executando as 6 chaves de controle e montando a matriz de pesos.
    """
    def __init__(self, lib_macro, lib_residual, block_size=8, cache_size=26):
        self.lib_macro = lib_macro          # 26 átomos base (0-25)
        self.lib_residual = lib_residual    # 26 átomos de resíduo (0-25)
        self.block_size = block_size
        self.cache_size = cache_size
        
    def decode(self, bytecode):
        reconstructed_blocks = []
        cache = [] # Cache de blocos recentes para reutilização
        
        idx = 0
        total_tokens = len(bytecode)
        
        while idx < total_tokens:
            token = bytecode[idx]
            idx += 1
            
            # CASO 1: CHAVE DE REUTILIZAÇÃO (KEY_REF)
            if token == KEY_REF:
                cache_idx = bytecode[idx]
                idx += 1
                block = cache[cache_idx].clone()
                reconstructed_blocks.append(block)
                # Atualiza cache (move para o topo)
                cache.append(block)
                if len(cache) > self.cache_size:
                    cache.pop(0)
                continue
                
            # CASO 2: ÁTOMO BASE
            current_block = self.lib_macro[token].clone()
            current_scale = 1.0
            
            # Loop de processamento de modificadores e resíduos até KEY_END ou KEY_SHORT
            while idx < total_tokens:
                next_token = bytecode[idx]
                idx += 1
                
                if next_token == KEY_SHORT or next_token == KEY_END:
                    break
                elif next_token == KEY_ADD:
                    # Composição aditiva
                    res_token = bytecode[idx]
                    idx += 1
                    current_block += self.lib_residual[res_token] * current_scale
                elif next_token == KEY_SCALE:
                    scale_idx = bytecode[idx] % len(SCALE_FACTORS)
                    idx += 1
                    current_scale = SCALE_FACTORS[scale_idx]
                    current_block = current_block * current_scale
                    
            reconstructed_blocks.append(current_block)
            cache.append(current_block)
            if len(cache) > self.cache_size:
                cache.pop(0)
                
        return torch.stack(reconstructed_blocks)

def encode_layer_with_keys(weights_flat, lib_macro, lib_residual, tolerance_ref=0.0003, tolerance_short=0.00015):
    """
    Codificador inteligente:
    Decide quando emitir REUTILIZAÇÃO (KEY_REF), BLOCO SIMPLES (KEY_SHORT) ou COMPOSIÇÃO (KEY_ADD).
    """
    bytecode = []
    N, D = weights_flat.shape
    cache_blocks = []
    cache_size = 26
    
    stats = {
        'total_blocks': N,
        'reused_blocks': 0,
        'short_blocks': 0,
        'composite_blocks': 0
    }
    
    for i in range(N):
        b = weights_flat[i]
        
        # 1. Tentar REUTILIZAÇÃO via Cache
        reused = False
        if len(cache_blocks) > 0:
            cache_tensor = torch.stack(cache_blocks)
            dists_cache = torch.sum((cache_tensor - b) ** 2, dim=1)
            min_dist, best_c_idx = torch.min(dists_cache, dim=0)
            
            if min_dist.item() < tolerance_ref:
                # Emite: [KEY_REF, cache_index] -> Apenas 2 tokens (10 bits!)
                bytecode.extend([KEY_REF, best_c_idx.item()])
                stats['reused_blocks'] += 1
                reused = True
                cache_blocks.append(cache_blocks[best_c_idx.item()])
                if len(cache_blocks) > cache_size:
                    cache_blocks.pop(0)
                continue
                
        if not reused:
            # 2. Encontrar melhor átomo macro
            dists_macro = torch.sum((lib_macro - b) ** 2, dim=1)
            best_macro_idx = torch.argmin(dists_macro).item()
            macro_vec = lib_macro[best_macro_idx]
            residual = b - macro_vec
            res_norm = torch.sum(residual ** 2).item()
            
            # Se o erro for minúsculo, emite apenas 2 tokens: [macro_idx, KEY_SHORT] (10 bits)
            if res_norm < tolerance_short:
                bytecode.extend([best_macro_idx, KEY_SHORT])
                stats['short_blocks'] += 1
                cache_blocks.append(macro_vec)
                if len(cache_blocks) > cache_size:
                    cache_blocks.pop(0)
            else:
                # 3. Composição Aditiva: [macro_idx, KEY_ADD, res_idx, KEY_END] (20 bits)
                dists_res = torch.sum((lib_residual - residual) ** 2, dim=1)
                best_res_idx = torch.argmin(dists_res).item()
                
                bytecode.extend([best_macro_idx, KEY_ADD, best_res_idx, KEY_END])
                stats['composite_blocks'] += 1
                recon_b = macro_vec + lib_residual[best_res_idx]
                cache_blocks.append(recon_b)
                if len(cache_blocks) > cache_size:
                    cache_blocks.pop(0)
                    
    return bytecode, stats

def main():
    print("=" * 80)
    print("MÁQUINA VIRTUAL DE COMPRESSÃO: ARQUITETURA 26 DADOS + 6 CHAVES")
    print("Testando Bytecode Dinâmico em Matriz Real de Rede Neural (1M Pesos)")
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
    
    orig_kb = (total_params * 2) / 1024
    print(f"\nCamada Original: {rows}x{cols} ({total_params:,} pesos FP16 = {orig_kb:.1f} KB)")
    print(f"Total de Blocos: {N:,} blocos de {block_size} números\n")
    
    # Treinar as duas bibliotecas de 26 átomos cada
    print("[1] Treinando Biblioteca 1: 26 Átomos Macro...")
    lib_macro = train_codebook(weights_flat, K=26, n_iter=10)
    
    # Calcular resíduos iniciais para treinar a biblioteca de resíduos
    dists = torch.cdist(weights_flat, lib_macro)
    lbls = torch.argmin(dists, dim=1)
    res_train = weights_flat - lib_macro[lbls]
    
    print("[2] Treinando Biblioteca 2: 26 Átomos de Resíduo...")
    lib_residual = train_codebook(res_train, K=26, n_iter=10)
    
    # Executar a compilação/codificação via Máquina de Chaves
    print("\n[3] Codificando fluxo com Chaves de Controle (KEY_REF, KEY_ADD, KEY_SHORT)...")
    t0 = time.time()
    bytecode, stats = encode_layer_with_keys(weights_flat, lib_macro, lib_residual)
    enc_time = time.time() - t0
    
    # Decodificar via Máquina Virtual
    print(f"[4] Executando Decodificação na Máquina Virtual ({enc_time:.2f}s para codificar)...")
    vm = CompressionVM(lib_macro, lib_residual, block_size=block_size)
    recon_blocks = vm.decode(bytecode)
    recon_matrix = recon_blocks.half().view(rows, cols)
    
    # Medir fidelidade e erro
    mse = torch.mean((weights_fp16.float() - recon_matrix.float()) ** 2).item()
    y_pred = torch.matmul(recon_matrix, x_test)
    cos_sim = torch.nn.functional.cosine_similarity(y_real.view(1, -1).float(), y_pred.view(1, -1).float()).item()
    
    # Medir Tamanhos e Bits
    total_tokens = len(bytecode)
    total_bits = total_tokens * 5 # Cada token é estritamente 5 bits
    bits_per_weight = total_bits / total_params
    size_stream_bytes = int(np.ceil(total_bits / 8.0))
    
    # Tamanho do dicionário: 2 bibliotecas * 26 vetores * 8 floats * 2 bytes = 832 bytes (~0.8 KB!)
    size_dict_bytes = 2 * (26 * block_size * 2)
    total_compressed_bytes = size_stream_bytes + size_dict_bytes
    total_compressed_kb = total_compressed_bytes / 1024
    reducao = 100 * (1 - total_compressed_bytes / (total_params * 2))
    
    print("\n" + "=" * 80)
    print("RELATÓRIO DE EXECUÇÃO DA MÁQUINA DE CHAVES")
    print("=" * 80)
    print(f"Blocos Totais Processados:            {stats['total_blocks']:,}")
    print(f"-> Blocos Reutilizados (KEY_REF):     {stats['reused_blocks']:,} ({stats['reused_blocks']/N*100:.1f}%) [Custo: 10 bits]")
    print(f"-> Blocos Simples (KEY_SHORT):        {stats['short_blocks']:,} ({stats['short_blocks']/N*100:.1f}%) [Custo: 10 bits]")
    print(f"-> Blocos Compostos (KEY_ADD):        {stats['composite_blocks']:,} ({stats['composite_blocks']/N*100:.1f}%) [Custo: 20 bits]")
    
    print("\n" + "-" * 80)
    print("EFICIÊNCIA DE ARMAZENAMENTO")
    print("-" * 80)
    print(f"Total de Tokens de 5 bits:            {total_tokens:,}")
    print(f"Bits Efetivos por Peso (BPP):         {bits_per_weight:.2f} bits/peso")
    print(f"Tamanho do Dicionário (2x26 átomos):  {size_dict_bytes / 1024:.2f} KB (menos de 1 KB!)")
    print(f"Tamanho Original (FP16):              {orig_kb:.1f} KB")
    print(f"Tamanho Comprimido Total:             {total_compressed_kb:.1f} KB")
    print(f"Taxa de Redução de Tamanho:           {reducao:.2f}%")
    print(f"Erro Quadrático Médio (MSE):          {mse:.6f}")
    print(f"Similaridade de Saída (Cosseno):      {cos_sim * 100:.2f}%")
    print("=" * 80)

if __name__ == '__main__':
    main()
