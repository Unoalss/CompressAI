import time, torch
from vq2.common import *
from vq2 import vq
from vq2.calib import list_targets

model = load_model()
T = list_targets(model)
log(len(T), "alvos;", "tied:", model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr())
from collections import Counter
log(Counter(t["kind"] for t in T))
log("dtype", next(model.parameters()).dtype, "GPU mem GB", torch.cuda.memory_allocated() / 2**30)
log([ (n, tuple(p.shape)) for n, p in list(model.named_parameters())[:6]])

# rotacao ortogonal?
rot = vq.get_rot("test", 6144)
x = torch.randn(5, 6144, device=DEV)
log("rot norm ratio", (rot.fwd(x).norm(dim=-1) / x.norm(dim=-1)).tolist(), "inv err", (rot.inv(rot.fwd(x)) - x).abs().max().item())

# quantizacao de uma matriz com codebook pequeno e H = I (sanidade)
t = [t for t in T if t["name"].endswith("layers.5.mlp.down_proj")][0]
W = t["module"].weight.data
n = W.shape[1]
rotm = vq.get_rot(t["tag"], n)
Wr = rotm.fwd(W.float()); s = Wr.pow(2).mean(1).sqrt()
u = (Wr / s[:, None]).reshape(-1, 8)
log("u std", u.std().item(), "kurtosis", ((u - u.mean()) ** 4).mean().item() / u.var().item() ** 2)
idx = torch.randperm(u.shape[0], device=DEV)[:2_000_000]
t0 = time.time()
cbs = vq.train_codebooks(u[idx], [(12,), (12, 8)], iters=8)
log("train time", time.time() - t0)
H = torch.eye(n, device=DEV)
t0 = time.time()
for cfg in [(12,), (12, 8)]:
    qm, Wd, rel = vq.quantize_matrix(W, H, t["tag"], cfg, cbs)
    err = ((Wd - W.float()) ** 2).sum() / (W.float() ** 2).sum()
    dq = qm.dequant(cbs)
    log(cfg, "rel_err(weight)", err.item(), "rel_func", rel, "dequant match", (dq - Wd).abs().max().item(), "time", time.time() - t0)
