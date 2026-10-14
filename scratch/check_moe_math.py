import sys
sys.path.insert(0, r"e:\compressAI")
import torch
import torch.nn.functional as F
from vq2 import vq
from vq2.load_and_test_full_35b import dequant_matrix

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
s1 = torch.load("e:/compressAI/vq2_35b_out/quant_model-00001-of-00021.safetensors.pt", weights_only=False, map_location="cpu")["quant"]
cbs_data = torch.load("e:/compressAI/vq2_out/codebooks.pt", weights_only=False)

q_gu = s1["model.language_model.layers.0.mlp.experts.gate_up_proj"]
q_down = s1["model.language_model.layers.0.mlp.experts.down_proj"]

print("q_gu shape:", q_gu["shape"], "tag:", q_gu["tag"])
print("q_down shape:", q_down["shape"], "tag:", q_down["tag"])

# 1. Dequantize via dequant_matrix
W_gu_all = dequant_matrix(q_gu, cbs_data, device=DEVICE) # [256*1024, 2048]
W_down_all = dequant_matrix(q_down, cbs_data, device=DEVICE) # [256*2048, 512]

print("W_gu_all shape:", W_gu_all.shape)
print("W_down_all shape:", W_down_all.shape)

# Escolhe expert 5
exp_id = 5
W_gu_5 = W_gu_all[exp_id * 1024 : (exp_id + 1) * 1024] # [1024, 2048]
W_down_5 = W_down_all[exp_id * 2048 : (exp_id + 1) * 2048] # [2048, 512]

x = torch.randn(1, 2048, device=DEVICE, dtype=torch.bfloat16)

# Referência exata dequantizada:
y_gu_ref = F.linear(x, W_gu_5) # [1, 1024]
gate_ref = y_gu_ref[:, :512]
up_ref = y_gu_ref[:, 512:]
act_ref = F.silu(gate_ref.float()) * up_ref.float()
act_ref = act_ref.bfloat16()
out_ref = F.linear(act_ref, W_down_5) # [1, 2048]

print("Ref y_gu norm:", y_gu_ref.norm().item())
print("Ref act norm:", act_ref.norm().item())
print("Ref out norm:", out_ref.norm().item())

# Agora vamos testar a decomposição da rotação:
# Para gate_up: W_gu = rot_2048.inv(Q_gu)
# x @ W_gu.T = x @ (rot_2048.inv(Q_gu)).T
# Como rot_2048 é ortogonal: x @ (R_inv @ Q)^T = x @ Q^T @ (R_inv)^T = (x @ R_fwd) @ Q^T
rot_2048 = vq.get_rot("moe_expert_2048", 2048, DEVICE)
rot_512 = vq.get_rot("moe_expert_512", 512, DEVICE)

# Q_gu puro reconstruído sem rotação inversa:
cb16 = cbs_data["16"].to(DEVICE)
idx_gu = q_gu["idx"][0].to(DEVICE).long()
sc_gu = q_gu["scale"].to(DEVICE).float()
Q_gu = (cb16[idx_gu].reshape(-1, 2048) * sc_gu[:, None]).to(torch.bfloat16)
Q_gu_5 = Q_gu[exp_id * 1024 : (exp_id + 1) * 1024]

xr = rot_2048.fwd(x)
y_gu_rot = F.linear(xr, Q_gu_5)
print("Rotated y_gu norm:", y_gu_rot.norm().item())
print("Diff y_gu vs ref:", (y_gu_rot - y_gu_ref).abs().max().item())

# Agora para down_proj: W_down [2048, 512]
# W_down = rot_512.inv(Q_down)
# act [1, 512] @ W_down.T [512, 2048] = act @ (rot_512.inv(Q_down)).T = (rot_512.fwd(act)) @ Q_down.T
idx_down = q_down["idx"][0].to(DEVICE).long()
sc_down = q_down["scale"].to(DEVICE).float()
Q_down = (cb16[idx_down].reshape(-1, 512) * sc_down[:, None]).to(torch.bfloat16)
Q_down_5 = Q_down[exp_id * 2048 : (exp_id + 1) * 2048]

act_r = rot_512.fwd(act_ref)
out_rot = F.linear(act_r, Q_down_5) # [1, 2048]
print("Rotated out norm (SEM rot_2048.inv):", out_rot.norm().item())
print("Diff out vs ref:", (out_rot - out_ref).abs().max().item())

out_scrambled = rot_2048.inv(out_rot)
print("Scrambled out norm (COM rot_2048.inv):", out_scrambled.norm().item())
print("Diff scrambled vs ref:", (out_scrambled - out_ref).abs().max().item())
