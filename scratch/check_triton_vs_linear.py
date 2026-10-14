import sys
sys.path.insert(0, r"e:\compressAI")
import torch
import torch.nn.functional as F
from vq2 import vq
from vq2.kernels import vq_moe_parallel_gate_up, vq_moe_parallel_down

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
s1 = torch.load("e:/compressAI/vq2_35b_out/quant_model-00001-of-00021.safetensors.pt", weights_only=False, map_location="cpu")["quant"]
cbs_data = torch.load("e:/compressAI/vq2_out/codebooks.pt", weights_only=False)
cb16 = cbs_data["16"].to(device=DEVICE, dtype=torch.bfloat16)

q_gu = s1["model.language_model.layers.0.mlp.experts.gate_up_proj"]
q_down = s1["model.language_model.layers.0.mlp.experts.down_proj"]

gate_up_idx = q_gu["idx"][0].view(256, 1024, 256).to(DEVICE)
gate_up_scale = q_gu["scale"].view(256, 1024).to(device=DEVICE, dtype=torch.bfloat16)
down_idx = q_down["idx"][0].view(256, 2048, 64).to(DEVICE)
down_scale = q_down["scale"].view(256, 2048).to(device=DEVICE, dtype=torch.bfloat16)

rot_2048 = vq.get_rot("moe_expert_2048", 2048, DEVICE)
rot_512 = vq.get_rot("moe_expert_512", 512, DEVICE)

x = torch.randn(1, 2048, device=DEVICE, dtype=torch.bfloat16)
xr = rot_2048.fwd(x)

# Teste com Triton kernel:
topk_ids = torch.tensor([5, 12, 47, 88, 120, 155, 201, 240], device=DEVICE, dtype=torch.int32)
y_gu = torch.empty((8, 1024), device=DEVICE, dtype=torch.bfloat16)
y_down = torch.empty((8, 2048), device=DEVICE, dtype=torch.bfloat16)
act_r = torch.empty((8, 512), device=DEVICE, dtype=torch.bfloat16)

# Teste 1: rs = 1.0
vq_moe_parallel_gate_up(xr.view(-1), topk_ids, gate_up_idx, cb16, gate_up_scale, y_gu, OUT=1024, NG=256, rs=1.0)
gate = y_gu[:, :512]
up = y_gu[:, 512:]
act = (F.silu(gate.float()) * up.float()).bfloat16()
act_r.copy_(rot_512.fwd(act))
vq_moe_parallel_down(act_r, topk_ids, down_idx, cb16, down_scale, y_down, OUT=2048, NG=64, rs=1.0)

print("Triton (rs=1.0) y_gu[0] norm:", y_gu[0].norm().item())
print("Triton (rs=1.0) y_down[0] norm:", y_down[0].norm().item())

# Teste com F.linear de referência para o expert 5 (topk_ids[0]):
idx_gu = q_gu["idx"][0].to(DEVICE).long()
sc_gu = q_gu["scale"].to(DEVICE).float()
Q_gu = (cb16[idx_gu].reshape(-1, 2048) * sc_gu[:, None]).to(torch.bfloat16)
Q_gu_5 = Q_gu[5 * 1024 : 6 * 1024]
y_gu_ref = F.linear(xr, Q_gu_5)[0]

print("Ref F.linear y_gu_5 norm:", y_gu_ref.norm().item())
print("Diff Triton vs Ref y_gu:", (y_gu[0] - y_gu_ref).abs().max().item())

idx_down = q_down["idx"][0].to(DEVICE).long()
sc_down = q_down["scale"].to(DEVICE).float()
Q_down = (cb16[idx_down].reshape(-1, 512) * sc_down[:, None]).to(torch.bfloat16)
Q_down_5 = Q_down[5 * 2048 : 6 * 2048]
y_down_ref = F.linear(act_r[0:1], Q_down_5)[0]

print("Ref F.linear y_down_5 norm:", y_down_ref.norm().item())
print("Diff Triton vs Ref y_down:", (y_down[0] - y_down_ref).abs().max().item())
