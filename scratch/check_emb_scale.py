import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vq2 import vq, common

# Load original model embed_tokens
m = common.load_model(device="cpu")
W0 = m.model.embed_tokens.weight.data.float().cuda()  # [248320, 2048]

# Load quantized qmats
qm_data = torch.load(common.OUT / "qmats_rdo_560.pt", weights_only=False)
qm_emb = qm_data["qmats"]["model.embed_tokens"]
cbs = torch.load(common.OUT / "codebooks.pt", weights_only=False)

rot = vq.get_rot(qm_emb["tag"], 2048, "cuda")
Wr = rot.fwd(W0)
s0 = qm_emb["scale"].float().cuda() # [248320]

# Compute unscaled reconstruction
cb16 = cbs["16"].float().cuda()
idx = qm_emb["idx"][0].long().cuda() # [248320, 256]
rec = cb16[idx].view(248320, 2048)

# Current MSE
curr_W = rec * s0[:, None]
mse_curr = ((Wr - curr_W)**2).mean().item()

# Closed-form optimal scale: (Wr * rec).sum(1) / (rec**2).sum(1)
s_opt = (Wr * rec).sum(dim=1) / rec.pow(2).sum(dim=1).clamp_min(1e-8)
opt_W = rec * s_opt[:, None]
mse_opt = ((Wr - opt_W)**2).mean().item()

print(f"Embedding MSE atual: {mse_curr:.6f}")
print(f"Embedding MSE com escala ótima de mínimos quadrados: {mse_opt:.6f}")
print(f"Redução de erro no embedding: {100 * (1 - mse_opt / mse_curr):.2f}%")
