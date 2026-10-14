import sys, os, time
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def test_lut_gemv():
    print("=" * 80)
    print("DESCOBERTA MATEMÁTICA: VQ2 LOOKUP-TABLE GEMV (LUT GEMV) PARA CPU")
    print("=" * 80)

    # Carrega Codebook em float32 CPU
    cbs_data = torch.load(ROOT / "vq2_out" / "codebooks.pt", weights_only=False, map_location="cpu")
    cb16 = cbs_data["16"].float() # [65536, 8]

    OUT = 1024
    NG = 256
    N = 2048
    RS = 2048 ** -0.5

    x = torch.randn(N, dtype=torch.float32)
    scale = torch.randn(OUT, dtype=torch.float32)
    indices = torch.randint(0, 65536, (OUT, NG), dtype=torch.int64)

    # 1. Referência Naive
    W_rec = (cb16[indices].view(OUT, N) * scale[:, None]) * RS
    y_ref = F.linear(x, W_rec)

    # 2. Método LUT GEMV
    # Passo 1: Pré-calcula a tabela de projeção de x no Codebook [65536, 256]
    t0_lut = time.perf_counter()
    x_g = x.view(NG, 8) # [256, 8]
    # lut[a, g] = cb[a] @ x_g[g]
    # torch.mm: [65536, 8] @ [8, 256] -> [65536, 256]
    lut = torch.mm(cb16, x_g.t()) # [65536, 256]
    t_lut_build_ms = (time.perf_counter() - t0_lut) * 1000.0
    print(f"Tempo para construir a LUT compartilhada para todos os 8 experts: {t_lut_build_ms:.2f} ms")

    # Passo 2: Para qualquer expert, faz apenas o gather da tabela!
    t0_gather = time.perf_counter()
    for _ in range(50):
        # indices tem shape [OUT, NG]
        # lut tem shape [65536, NG]
        gathered = torch.gather(lut, 0, indices) # [OUT, NG]
        y_lut = (gathered.sum(dim=1) * scale) * RS
    t_gather_ms = (time.perf_counter() - t0_gather) * 1000.0 / 50.0

    print(f"Tempo por matriz de Expert via LUT Gather: {t_gather_ms:.2f} ms!")
    cos = F.cosine_similarity(y_ref.unsqueeze(0), y_lut.unsqueeze(0)).item()
    max_diff = torch.max(torch.abs(y_ref - y_lut)).item()
    print(f"Equivalência Numérica: Cosseno={cos:.6f} | Desvio={max_diff:.6f}")

    # E agora simula para os 8 experts juntos:
    # 8 experts x 1024 linhas = 8192 linhas!
    # Todos compartilham a MESMA LUT pré-computada!
    eight_experts_idx = torch.randint(0, 65536, (8 * OUT, NG), dtype=torch.int64)
    eight_scales = torch.randn(8 * OUT, dtype=torch.float32)
    t0_8 = time.perf_counter()
    for _ in range(20):
        all_gathered = torch.gather(lut, 0, eight_experts_idx)
        all_y = (all_gathered.sum(dim=1) * eight_scales) * RS
    t_8_ms = (time.perf_counter() - t0_8) * 1000.0 / 20.0
    print(f"Tempo para TODOS OS 8 EXPERTS DA CAMADA JUNTOS: {t_8_ms:.2f} ms! ({t_8_ms / 8:.2f} ms/expert)")

if __name__ == "__main__":
    test_lut_gemv()
