import time, torch
from vq2.kernels import vq_moe_parallel_gate_up, vq_moe_parallel_down

device = "cuda"
C = 96 # 96 experts in cache
loaded_gu_idx = torch.randint(0, 1000, (C, 1024, 256), dtype=torch.int16, device=device)
loaded_gu_scale = torch.randn(C, 1024, dtype=torch.bfloat16, device=device)
loaded_dn_idx = torch.randint(0, 1000, (C, 2048, 64), dtype=torch.int16, device=device)
loaded_dn_scale = torch.randn(C, 2048, dtype=torch.bfloat16, device=device)

cb16 = torch.randn(65536, 8, dtype=torch.bfloat16, device=device)
xr = torch.randn(2048, dtype=torch.bfloat16, device=device)
y_gu = torch.empty(8, 1024, dtype=torch.bfloat16, device=device)
act_r = torch.empty(8, 512, dtype=torch.bfloat16, device=device)
y_down = torch.empty(8, 2048, dtype=torch.bfloat16, device=device)

# Suppose the 8 active experts are at slots [5, 12, 23, 40, 55, 67, 80, 91]
slot_ids = torch.tensor([5, 12, 23, 40, 55, 67, 80, 91], dtype=torch.int32, device=device)

# Warmup
for _ in range(5):
    vq_moe_parallel_gate_up(xr, slot_ids, loaded_gu_idx, cb16, loaded_gu_scale, y_gu, OUT=1024, NG=256)
    act_r.copy_(y_gu[:, :512])
    vq_moe_parallel_down(act_r, slot_ids, loaded_dn_idx, cb16, loaded_dn_scale, y_down, OUT=2048, NG=64)
torch.cuda.synchronize()

N = 200
t0 = time.perf_counter()
for _ in range(N):
    vq_moe_parallel_gate_up(xr, slot_ids, loaded_gu_idx, cb16, loaded_gu_scale, y_gu, OUT=1024, NG=256)
    act_r.copy_(y_gu[:, :512])
    vq_moe_parallel_down(act_r, slot_ids, loaded_dn_idx, cb16, loaded_dn_scale, y_down, OUT=2048, NG=64)
torch.cuda.synchronize()
t_per_layer = (time.perf_counter() - t0) / N * 1000
print(f"Tempo MoE por camada (ZERO-COPY!): {t_per_layer:.3f} ms")
print(f"Tempo MoE em 40 camadas: {t_per_layer * 40:.2f} ms")
print(f"Taxa teorica de tokens/s (MoE puro): {1000.0 / (t_per_layer * 40):.1f} tok/s!")
