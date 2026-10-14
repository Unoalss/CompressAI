import sys
sys.path.insert(0, r"e:\compressAI")
import torch, time

def fast_walsh_hadamard(x):
    """
    Fast Walsh-Hadamard Transform (FWHT) in-place/out-of-place.
    Complexidade: O(N log N) em vez de O(N^2).
    Para N=2048: 22.528 operacoes em vez de 4.194.304 operacoes!
    """
    h = 1
    y = x.clone()
    N = y.shape[-1]
    while h < N:
        y = y.view(-1, N // (2 * h), 2, h)
        a = y[:, :, 0, :]
        b = y[:, :, 1, :]
        y = torch.cat([a + b, a - b], dim=2)
        h *= 2
    return y.view_as(x)

# Vamos comparar com _hmat
from vq2 import vq
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
hmat = vq._hmat(DEVICE, torch.float32, 2048)

x = torch.randn(1, 2048, device=DEVICE, dtype=torch.float32)

# Referência GEMM
y_ref = x @ hmat

# Fast Walsh-Hadamard
y_fwht = fast_walsh_hadamard(x)

diff = (y_ref - y_fwht).abs().max().item()
print(f"Diff máximo entre GEMM e Fast Walsh-Hadamard: {diff:.6f}")

# Benchmark de velocidade
N_TEST = 1000
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N_TEST):
    _ = x @ hmat
torch.cuda.synchronize()
t_gemm = (time.perf_counter() - t0) / N_TEST * 1000

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N_TEST):
    _ = fast_walsh_hadamard(x)
torch.cuda.synchronize()
t_fwht = (time.perf_counter() - t0) / N_TEST * 1000

print(f"Tempo GEMM (_hmat denso): {t_gemm:.4f} ms")
print(f"Tempo FWHT (PyTorch view/cat): {t_fwht:.4f} ms")
