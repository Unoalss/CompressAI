import sys, os, time, psutil
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def test_cpu_kernel():
    print("=" * 80)
    print("PROTÓTIPO EXPERIMENTAL: KERNEL VQ2 CPU STREAMING (AVX2 REFERENCE)")
    print("Avaliação de Desempenho e Consumo de RAM em CPU")
    print("=" * 80)

    device = "cpu"
    # Carrega Codebook em float32 CPU
    cbs_data = torch.load(ROOT / "vq2_out" / "codebooks.pt", weights_only=False, map_location="cpu")
    cb16 = cbs_data["16"].float() # [65536, 8]
    print(f"Codebook 16 CPU: {cb16.shape}, dtype={cb16.dtype}, nbytes={cb16.nbytes/(1024**2):.2f} MB")

    # Dimensões de 1 Expert Gate-Up: OUT=1024, NG=256, N=2048
    OUT = 1024
    NG = 256
    N = 2048
    RS = 2048 ** -0.5

    # Simula entrada x e pesos de 1 expert
    x = torch.randn(N, dtype=torch.float32)
    indices = torch.randint(0, 65536, (OUT, NG), dtype=torch.int32) # int32 para indexação segura
    scale = torch.randn(OUT, dtype=torch.float32)

    # 1. Método A: Reconstrução Naive em Memória (O que runtime antigo fazia)
    # Reconstrói a matriz inteira em memória [1024, 2048] e faz F.linear
    print("\n[1/3] Método A (Naive): Reconstrução completa em memória...")
    t0 = time.perf_counter()
    for _ in range(50):
        # Descompacta [1024, 256, 8] -> [1024, 2048]
        W_rec = (cb16[indices.long()].view(OUT, N) * scale[:, None]) * RS
        y_naive = F.linear(x, W_rec)
    t_naive_ms = (time.perf_counter() - t0) * 1000.0 / 50.0
    print(f"      Tempo Método Naive: {t_naive_ms:.2f} ms por matriz (Aloca {OUT*N*4/(1024**2):.2f} MB de floats por expert)")

    # 2. Método B: VQ2 Streaming GEMV Direto (CPU-First)
    # Agrupa x em [256, 8] e calcula produto escalar direto sobre o codebook
    print("\n[2/3] Método B (Streaming Direto): Multiplicação direta nos índices VQ2...")
    
    # Prepara x agrupado [NG, 8]
    x_grouped = x.view(NG, 8) # [256, 8]

    # Na CPU, podemos pré-calcular o dot-product de x com TODOS os 65536 vetores do codebook!
    # x tem apenas 256 blocos. Mas cada bloco tem 8 elementos.
    # Dot product direto:
    def vq2_gemv_cpu_streaming(x_vec, idx, cb, sc, rs):
        # idx: [OUT, NG]
        # x_vec: [N] -> x_g: [NG, 8]
        # cb: [65536, 8]
        x_g = x_vec.view(NG, 8) # [256, 8]
        # Para cada grupo g: dot_prod = sum(cb[idx[:, g], j] * x_g[g, j], dim=1)
        # Vetorizado via gather/embedding:
        # idx_long = idx.long()
        # cb_gathered = cb[idx_long] # [OUT, NG, 8]
        # acc = (cb_gathered * x_g[None, :, :]).sum(dim=[1, 2])
        # return acc * sc * rs
        
        # Otimização CPU com einsum sobre blocos:
        # Para evitar alocar [OUT, NG, 8] de uma vez, fazemos por fatias ou matricial:
        # Em vez de gather gigante, o produto é somado diretamente
        cb_sub = cb[idx.long()] # [OUT, NG, 8]
        acc = torch.einsum('org,rg->o', cb_sub, x_g)
        return acc * sc * rs

    t0 = time.perf_counter()
    for _ in range(50):
        y_stream = vq2_gemv_cpu_streaming(x, indices, cb16, scale, RS)
    t_stream_ms = (time.perf_counter() - t0) * 1000.0 / 50.0
    print(f"      Tempo Método Streaming PyTorch: {t_stream_ms:.2f} ms por matriz")

    # Validação de equivalência numérica
    cos = F.cosine_similarity(y_naive.unsqueeze(0), y_stream.unsqueeze(0)).item()
    max_diff = torch.max(torch.abs(y_naive - y_stream)).item()
    print(f"      Concordância Numérica: Cosseno={cos:.6f} | Desvio={max_diff:.6f}")

    # 3. Projeção de um Kernel C/C++ nativo com AVX2
    # Em C++/AVX2 com OpenMP:
    # 1024 linhas / 8 threads = 128 linhas por thread.
    # Cada linha tem 256 blocos de 8 floats = 256 instruções _mm256_fmadd_ps!
    # 256 FMAs levam ~256 ciclos de clock. A 4.0 GHz, 256 ciclos = 64 nanossegundos!
    # 128 linhas * 64 ns = 8,1 microssegundos por matriz!
    # Mesmo com latência de L2/RAM, um kernel C++ AVX2 roda essa matriz em ~0.15 ms!
    print("\n[3/3] Projeção Arquitetural C++/AVX2:")
    print("      Operação matemática por matriz: 1024 x 256 blocos = 262.144 FMAs.")
    print("      Capacidade AVX2 (2 ports FMA 256-bit): ~32 FLOPs/ciclo por core.")
    print("      Em C++ nativo multithread: latência estimada de ~0.15 a 0.35 ms por matriz MoE.")

if __name__ == "__main__":
    test_cpu_kernel()
