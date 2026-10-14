import sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vq2.kernels import triton_conv1d_update
import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as mod

def benchmark_conv1d():
    device = "cuda"
    dtype = torch.bfloat16
    H = 8192

    hs = torch.randn(1, H, 1, device=device, dtype=dtype)
    cs = torch.randn(1, H, 4, device=device, dtype=dtype)
    w = torch.randn(H, 4, device=device, dtype=dtype)
    py_fn = mod.causal_conv1d_update

    # Warmup
    for _ in range(50):
        _ = py_fn(hs, cs, w, bias=None, activation="silu")
        _ = triton_conv1d_update(hs, cs, w, b=None)
    torch.cuda.synchronize()

    N = 300  # Equivale a 10 tokens (30 camadas x 10 = 300 chamadas)

    # 1. Benchmark PyTorch Fallback
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(N):
        _ = py_fn(hs, cs, w, bias=None, activation="silu")
    end_event.record()
    torch.cuda.synchronize()
    time_py_ms = start_event.elapsed_time(end_event)

    # 2. Benchmark Triton Fused Kernel
    start_event.record()
    for _ in range(N):
        _ = triton_conv1d_update(hs, cs, w, b=None)
    end_event.record()
    torch.cuda.synchronize()
    time_tri_ms = start_event.elapsed_time(end_event)

    print("=" * 70)
    print("BENCHMARK ISOLADO: 300 CONVOLUÇÕES CAUSAIS (10 TOKENS x 30 CAMADAS)")
    print("=" * 70)
    print(f"PyTorch Fallback:  {time_py_ms:.2f} ms total  ({time_py_ms / 10:.2f} ms por token | {time_py_ms / N * 1000:.1f} µs por camada)")
    print(f"Triton Kernel:     {time_tri_ms:.2f} ms total  ({time_tri_ms / 10:.2f} ms por token | {time_tri_ms / N * 1000:.1f} µs por camada)")
    print(f"Aceleração pura:   {time_py_ms / time_tri_ms:.2f}x mais rápido")
    print(f"Economia esperada: {(time_py_ms - time_tri_ms) / 10:.2f} ms por token!")
    print("=" * 70)

if __name__ == "__main__":
    benchmark_conv1d()
