import sys, time
sys.path.insert(0, r"e:\compressAI")
import torch
from vq2 import vq

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
rot = vq.get_rot("moe_expert_2048", 2048, DEVICE)
x = torch.randn(2048, device=DEVICE, dtype=torch.bfloat16)

# Teste 1: rot.fwd normal
torch.cuda.synchronize()
t0 = time.perf_counter()
N = 1000
for _ in range(N):
    y = rot.fwd(x)
torch.cuda.synchronize()
dt1 = (time.perf_counter() - t0) / N * 1000
print(f"rot.fwd(x) puro: {dt1:.4f} ms ({dt1*1000:.1f} us)")

# Teste 2: decompõe cada linha de rot.fwd
p = rot.perm
s = rot.sign.to(x.dtype)
hmat = vq._hmat(DEVICE, x.dtype, 2048)
scale = rot.scale

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    x_perm = x[p] * s
    y = (x_perm.unsqueeze(0) @ hmat) * scale
torch.cuda.synchronize()
dt2 = (time.perf_counter() - t0) / N * 1000
print(f"Decomposto: {dt2:.4f} ms ({dt2*1000:.1f} us)")
