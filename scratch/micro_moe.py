import sys, time
sys.path.insert(0, r"e:\compressAI")
import torch
import torch.nn.functional as F
from vq2 import vq
from vq2.kernels import vq_moe_parallel_gate_up, vq_moe_parallel_down

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
s1 = torch.load("e:/compressAI/vq2_35b_out/quant_model-00001-of-00021.safetensors.pt", weights_only=False, map_location="cpu")["quant"]
cbs_data = torch.load("e:/compressAI/vq2_out/codebooks.pt", weights_only=False)
cb16 = cbs_data["16"].to(device=DEVICE, dtype=torch.bfloat16)

gate_up_idx = s1["model.language_model.layers.0.mlp.experts.gate_up_proj"]["idx"][0].view(256, 1024, 256).to(DEVICE)
gate_up_scale = s1["model.language_model.layers.0.mlp.experts.gate_up_proj"]["scale"].view(256, 1024).to(device=DEVICE, dtype=torch.bfloat16)
down_idx = s1["model.language_model.layers.0.mlp.experts.down_proj"]["idx"][0].view(256, 2048, 64).to(DEVICE)
down_scale = s1["model.language_model.layers.0.mlp.experts.down_proj"]["scale"].view(256, 2048).to(device=DEVICE, dtype=torch.bfloat16)

rot_2048 = vq.get_rot("moe_expert_2048", 2048, DEVICE)
rot_512 = vq.get_rot("moe_expert_512", 512, DEVICE)

y_gu = torch.empty((8, 1024), device=DEVICE, dtype=torch.bfloat16)
y_down = torch.empty((8, 2048), device=DEVICE, dtype=torch.bfloat16)
act_r = torch.empty((8, 512), device=DEVICE, dtype=torch.bfloat16)

h = torch.randn(2048, device=DEVICE, dtype=torch.bfloat16)
topk_ids = torch.tensor([5, 12, 47, 88, 120, 155, 201, 240], device=DEVICE, dtype=torch.int32)
topk_w = torch.tensor([0.25, 0.20, 0.15, 0.10, 0.10, 0.08, 0.07, 0.05], device=DEVICE, dtype=torch.bfloat16)

N = 200

# 1. rot_2048.fwd
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    xr = rot_2048.fwd(h)
torch.cuda.synchronize()
t1 = (time.perf_counter() - t0) / N * 1000
print(f"1. rot_2048.fwd: {t1:.4f} ms")

# 2. vq_moe_parallel_gate_up
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    vq_moe_parallel_gate_up(xr, topk_ids, gate_up_idx, cb16, gate_up_scale, y_gu, OUT=1024, NG=256, rs=1.0)
torch.cuda.synchronize()
t2 = (time.perf_counter() - t0) / N * 1000
print(f"2. vq_moe_parallel_gate_up: {t2:.4f} ms")

# 3. silu
gate = y_gu[:, :512]
up = y_gu[:, 512:]
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    act = (F.silu(gate.float()) * up.float()).bfloat16()
torch.cuda.synchronize()
t3 = (time.perf_counter() - t0) / N * 1000
print(f"3. SiLU activation: {t3:.4f} ms")

# 4. rot_512.fwd(act)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    act_rot = rot_512.fwd(act)
torch.cuda.synchronize()
t4 = (time.perf_counter() - t0) / N * 1000
print(f"4. rot_512.fwd (batch 8): {t4:.4f} ms")

# 5. vq_moe_parallel_down
act_r.copy_(act_rot)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    vq_moe_parallel_down(act_r, topk_ids, down_idx, cb16, down_scale, y_down, OUT=2048, NG=64, rs=1.0)
torch.cuda.synchronize()
t5 = (time.perf_counter() - t0) / N * 1000
print(f"5. vq_moe_parallel_down: {t5:.4f} ms")

# 6. reduce
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    out = (y_down * topk_w[:, None]).sum(dim=0, keepdim=True)
torch.cuda.synchronize()
t6 = (time.perf_counter() - t0) / N * 1000
print(f"6. Reduce sum: {t6:.4f} ms")

print(f"\nTotal soma: {t1+t2+t3+t4+t5+t6:.4f} ms")
