"""
vq2/kd.py - ponto 4: fine-tuning por destilacao (KL contra o teacher BF16) de TUDO que e continuo:
  escalas por linha (em log), codebooks compartilhados (estagios 1/2), RMSNorms.
Os indices (codigos) ficam fixos. Dados de treino != dados de avaliacao.
O teacher e usado via cache top-K (+ massa da cauda), entao so o aluno fica na GPU (~3 GB).
"""
import os, time, math, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import *
from . import vq
from .calib import list_targets


class QLinear(nn.Module):
    def __init__(self, qm, cbstore, device=DEV):
        super().__init__()
        self.out, self.n = qm.shape
        self.cfg, self.tag = qm.cfg, qm.tag
        for i, a in enumerate(qm.idx):
            self.register_buffer(f"idx{i}", a.to(device), persistent=False)
        self.log_scale = nn.Parameter(torch.log(qm.scale.float().clamp_min(1e-6)).to(device))
        self.keys = [vq.cfg_key(self.cfg, i) for i in range(len(self.cfg))]
        self._store = [cbstore]
        self.rot = vq.get_rot(self.tag, self.n, device)

    def weight(self):
        rec = 0
        for i, k in enumerate(self.keys):
            rec = rec + self._store[0][k][getattr(self, f"idx{i}").long()]
        return rec.reshape(self.out, self.n) * self.log_scale.exp()[:, None]

    def forward(self, x):
        return F.linear(self.rot.fwd(x), self.weight().to(x.dtype))


def _set(model, name, new):
    parent = model
    parts = name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


def build_student(model, qfile, cbs_cpu):
    """Substitui as matrizes alvo (exceto embedding) por QLinear; embedding vira tabela densa congelada."""
    Q = torch.load(qfile)
    qmats = {k: vq.QMat(v["shape"], v["cfg"], v["tag"], v["idx"], v["scale"]) for k, v in Q["qmats"].items()}
    cbs = vq.Codebooks({k: v.half().float().to(DEV) for k, v in cbs_cpu.items()})
    store = nn.ParameterDict({k: nn.Parameter(v.clone().float()) for k, v in cbs.cb.items()})
    qlins = {}
    for t in list_targets(model):
        qm = qmats[t["name"]]
        if t["kind"] == "embed":
            t["module"].weight.data.copy_(qm.dequant(cbs, DEV, torch.bfloat16))   # congelado (codebook proprio)
            continue
        ql = QLinear(qm, store)
        _set(model, t["name"], ql)
        qlins[t["name"]] = ql
    return qmats, store, qlins


def trainable_params(model, store, qlins):
    groups = {"scale": [], "cb": [], "norm": []}
    for ql in qlins.values():
        groups["scale"].append(ql.log_scale)
    used = set()
    for ql in qlins.values():
        used.update(ql.keys)
    for k in used:
        store[k].requires_grad_(True)
        groups["cb"].append(store[k])
    for n, p in model.named_parameters():
        if "norm" in n.lower() and p.ndim == 1:
            p.data = p.data.float()
            p.requires_grad_(True)
            groups["norm"].append(p)
    return groups


def ensure_train_cache(n_wiki=300, n_code=150, path=None, K=64):
    path = path or (OUT / "kd_cache.pt")
    toks = mixed_train_set(n_wiki, n_code, offset_wiki=100, offset_code=100, seed=11)
    if os.path.exists(path):
        c = torch.load(path)
        return toks, c
    m = load_model()
    t0 = time.time()
    c = cache_teacher(m, toks, K=K)
    torch.save(c, path)
    log(f"[kd] cache do teacher: {tuple(c['idx'].shape)} em {time.time()-t0:.0f}s")
    del m
    torch.cuda.empty_cache()
    return toks, c


def run_kd(name, steps=100, mb=2, accum=2, lr_scale=2e-3, lr_cb=1e-4, lr_norm=2e-5, eval_every=50):
    qfile = OUT / f"qmats_{name}.pt"
    toks, cache = ensure_train_cache()
    cbs_cpu = torch.load(OUT / "codebooks.pt")
    model = load_model()
    qmats, store, qlins = build_student(model, qfile, cbs_cpu)
    store.to(DEV)
    groups = trainable_params(model, store, qlins)
    for g, ps in groups.items():
        log(f"[kd] grupo {g}: {sum(p.numel() for p in ps):,} parametros")
    opt = torch.optim.AdamW([
        dict(params=groups["scale"], lr=lr_scale, weight_decay=0.0),
        dict(params=groups["cb"], lr=lr_cb, weight_decay=0.0),
        dict(params=groups["norm"], lr=lr_norm, weight_decay=0.0)], betas=(0.9, 0.99))
    base_lrs = [g["lr"] for g in opt.param_groups]
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    ev = Evaluator()
    model.eval()
    r0 = ev.run(model, full=False)
    log(f"[kd] antes: {r0}")
    N = toks.shape[0]
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(5))
    t0, ptr, hist = time.time(), 0, []
    for step in range(steps):
        lr_mult = min(1.0, (step + 1) / 20) * 0.5 * (1 + math.cos(math.pi * step / steps))
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * max(lr_mult, 0.02)
        model.train()
        tot = 0.0
        for a in range(accum):
            ids = perm[(ptr + torch.arange(mb)) % N]
            ptr += mb
            x = toks[ids].to(DEV)
            lg = model(input_ids=x).logits
            kl, _ = kd_loss_topk(lg, cache["idx"][ids].to(DEV), cache["lp"][ids].to(DEV))
            loss = kl.mean() / accum
            loss.backward()
            tot += loss.item()
            del lg, kl, loss
        torch.nn.utils.clip_grad_norm_([p for ps in groups.values() for p in ps], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        hist.append(tot)
        if step % 10 == 0:
            log(f"[kd] step {step}/{steps} loss(KL) {tot:.4f}  ({time.time()-t0:.0f}s)")
        if (step + 1) % eval_every == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                log(f"[kd] eval @{step+1}: {ev.run(model, full=False)}")
    model.gradient_checkpointing_disable()
    # congela em fp16 (como sera armazenado) e materializa para avaliacao final
    final = materialize(model, qmats, store, qlins, name)
    return final


@torch.no_grad()
def materialize(model, qmats, store, qlins, name):
    """Converte QLinear em nn.Linear densos (bf16) usando escalas/codebooks arredondados em fp16 (= armazenamento)."""
    cb16 = {k: v.detach().half().float() for k, v in store.items()}
    cbs = vq.Codebooks(cb16)
    for nm, ql in qlins.items():
        qm = qmats[nm]
        qm.scale = ql.log_scale.detach().exp().half().cpu()
        Wd = qm.dequant(cbs, DEV, torch.bfloat16)
        lin = nn.Linear(ql.n, ql.out, bias=False, device=DEV, dtype=torch.bfloat16)
        lin.weight.data.copy_(Wd)
        lin.weight.requires_grad_(False)
        _set(model, nm, lin)
    for n, p in model.named_parameters():
        if p.dtype == torch.float32:
            p.data = p.data.to(torch.bfloat16)
            p.requires_grad_(False)
    torch.save({"qmats": {k: dict(shape=v.shape, cfg=v.cfg, tag=v.tag, idx=v.idx, scale=v.scale) for k, v in qmats.items()},
                "codebooks": {k: v.half().cpu() for k, v in cb16.items()},
                "norms": {n: p.detach().cpu() for n, p in model.named_parameters() if "norm" in n.lower() and p.ndim == 1}},
               OUT / f"final_{name}.pt")
    ev = Evaluator()
    r = ev.run(model, full=True)
    log(f"[kd] FINAL {name}: {r}")
    save_json(r, OUT / f"result_{name}_kd.json")
    return r
