import sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def test_fused_in_proj_isolated():
    print("=" * 80)
    print("TESTE MICRO-OPERACIONAL ISOLADO: FUSÃO DAS PROJEÇÕES DE ENTRADA DELTANET")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "CUDA é obrigatório!"

    # Dimensões exatas da DeltaNet no Qwen 3.5 35B
    hidden_size = 2048
    d_qkv = 8192
    d_z = 4096
    d_b = 32
    d_a = 32
    d_total = d_qkv + d_z + d_b + d_a # 12352

    print(f"Dimensões:")
    print(f"  hidden_size: 2048")
    print(f"  in_proj_qkv: [{d_qkv}, {hidden_size}]")
    print(f"  in_proj_z:   [{d_z}, {hidden_size}]")
    print(f"  in_proj_b:   [{d_b}, {hidden_size}]")
    print(f"  in_proj_a:   [{d_a}, {hidden_size}]")
    print(f"  Total Linhas: {d_total} x {hidden_size}")

    torch.manual_seed(42)
    # 1. Simula 30 camadas de pesos reais em bfloat16
    N_LAYERS = 30
    w_qkv_list = [torch.randn((d_qkv, hidden_size), device=device, dtype=torch.bfloat16) for _ in range(N_LAYERS)]
    w_z_list = [torch.randn((d_z, hidden_size), device=device, dtype=torch.bfloat16) for _ in range(N_LAYERS)]
    w_b_list = [torch.randn((d_b, hidden_size), device=device, dtype=torch.bfloat16) for _ in range(N_LAYERS)]
    w_a_list = [torch.randn((d_a, hidden_size), device=device, dtype=torch.bfloat16) for _ in range(N_LAYERS)]

    # Concatenação das matrizes W_in
    w_fused_list = [
        torch.cat([w_qkv_list[i], w_z_list[i], w_b_list[i], w_a_list[i]], dim=0).contiguous()
        for i in range(N_LAYERS)
    ]

    # Vetor de entrada representativo (batch=1, seq=1, dim=2048)
    x = torch.randn((1, 1, hidden_size), device=device, dtype=torch.bfloat16)

    # 2. Teste de equivalência numérica saída por saída
    print("\n[1/3] Verificando equivalência numérica saída por saída em todas as 30 camadas...")
    max_diff_qkv = 0.0
    max_diff_z = 0.0
    max_diff_b = 0.0
    max_diff_a = 0.0

    for l in range(N_LAYERS):
        # Original: 4 GEMVs separados
        out_qkv = F.linear(x, w_qkv_list[l])
        out_z = F.linear(x, w_z_list[l])
        out_b = F.linear(x, w_b_list[l])
        out_a = F.linear(x, w_a_list[l])

        # Fundido: 1 GEMV + slicing
        out_fused = F.linear(x, w_fused_list[l])
        f_qkv = out_fused[:, :, :d_qkv]
        f_z = out_fused[:, :, d_qkv : d_qkv + d_z]
        f_b = out_fused[:, :, d_qkv + d_z : d_qkv + d_z + d_b]
        f_a = out_fused[:, :, d_qkv + d_z + d_b :]

        diff_qkv = (out_qkv.float() - f_qkv.float()).abs().max().item()
        diff_z = (out_z.float() - f_z.float()).abs().max().item()
        diff_b = (out_b.float() - f_b.float()).abs().max().item()
        diff_a = (out_a.float() - f_a.float()).abs().max().item()

        max_diff_qkv = max(max_diff_qkv, diff_qkv)
        max_diff_z = max(max_diff_z, diff_z)
        max_diff_b = max(max_diff_b, diff_b)
        max_diff_a = max(max_diff_a, diff_a)

    print(f"  Erro absoluto máximo QKV (8192 linhas): {max_diff_qkv:.6f}")
    print(f"  Erro absoluto máximo Z   (4096 linhas): {max_diff_z:.6f}")
    print(f"  Erro absoluto máximo B   (32 linhas):   {max_diff_b:.6f}")
    print(f"  Erro absoluto máximo A   (32 linhas):   {max_diff_a:.6f}")
    assert max(max_diff_qkv, max_diff_z, max_diff_b, max_diff_a) == 0.0, "Divergência numérica detectada!"
    print("  [OK] Equivalência numérica 100% BIT A BIT em todas as 4 saídas!")

    # 3. Benchmark de latência na GPU (simulando 1 token através das 30 camadas)
    print("\n[2/3] Executando benchmark de latência na GPU (1.000 iterações)...")
    N_ITERS = 1000

    # Warmup
    for _ in range(100):
        for l in range(N_LAYERS):
            _ = F.linear(x, w_qkv_list[l])
            _ = F.linear(x, w_z_list[l])
            _ = F.linear(x, w_b_list[l])
            _ = F.linear(x, w_a_list[l])
            _ = F.linear(x, w_fused_list[l])
    torch.cuda.synchronize()

    # Mede 4 GEMVs separados (120 lançamentos por token)
    ev_s = torch.cuda.Event(enable_timing=True)
    ev_e = torch.cuda.Event(enable_timing=True)
    ev_s.record()
    for _ in range(N_ITERS):
        for l in range(N_LAYERS):
            qkv = F.linear(x, w_qkv_list[l])
            z = F.linear(x, w_z_list[l])
            b = F.linear(x, w_b_list[l])
            a = F.linear(x, w_a_list[l])
    ev_e.record()
    torch.cuda.synchronize()
    time_sep_ms = ev_s.elapsed_time(ev_e) / N_ITERS

    # Mede 1 GEMV fundido + slicing (30 lançamentos por token)
    ev_s.record()
    for _ in range(N_ITERS):
        for l in range(N_LAYERS):
            out_fused = F.linear(x, w_fused_list[l])
            qkv = out_fused[:, :, :d_qkv]
            z = out_fused[:, :, d_qkv : d_qkv + d_z]
            b = out_fused[:, :, d_qkv + d_z : d_qkv + d_z + d_b]
            a = out_fused[:, :, d_qkv + d_z + d_b :]
    ev_e.record()
    torch.cuda.synchronize()
    time_fused_ms = ev_s.elapsed_time(ev_e) / N_ITERS

    print(f"\nResultados Micro-operacionais (30 camadas acumuladas / token):")
    print(f"  4 GEMVs Separados (120 kernels/tok):  {time_sep_ms:.3f} ms/token ({time_sep_ms / 30 * 1000:.1f} us/camada)")
    print(f"  1 GEMV Fundido    (30 kernels/tok):   {time_fused_ms:.3f} ms/token ({time_fused_ms / 30 * 1000:.1f} us/camada)")
    print(f"  Ganho Micro-operacional:              -{time_sep_ms - time_fused_ms:.3f} ms/token ({((time_sep_ms - time_fused_ms)/time_sep_ms)*100:.1f}% mais rápido!)")

    # 4. Avaliação de Memória e Estratégia de Liberação
    bytes_per_layer = d_total * hidden_size * 2 # bfloat16
    total_mb = (bytes_per_layer * N_LAYERS) / (1024 * 1024)
    print("\n[3/3] Orçamento de VRAM:")
    print(f"  Tamanho das 4 matrizes separadas por camada: {bytes_per_layer / (1024*1024):.2f} MB")
    print(f"  Tamanho da matriz fundida por camada:        {bytes_per_layer / (1024*1024):.2f} MB")
    print(f"  Total acumulado nas 30 camadas:              {total_mb:.2f} MB")
    print(f"  Estratégia de Memória Zero-Overhead:")
    print(f"    Ao fundir W_in, os pesos originais (w_qkv, w_z, w_b, w_a) DEVEM ser deletados")
    print(f"    para liberar os {total_mb:.2f} MB. Com isso, o acréscimo líquido de VRAM é RIGOROSAMENTE 0,00 MB!")
    print("=" * 80)

if __name__ == "__main__":
    test_fused_in_proj_isolated()
