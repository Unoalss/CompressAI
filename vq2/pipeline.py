"""
vq2/pipeline.py - orquestracao dos 6 pontos.

  python -m vq2.pipeline prepare      # baseline BF16, Hessianas, codebooks universais   (pontos 0,1,2)
  python -m vq2.pipeline scan         # sensibilidade por matriz (KL)                    (ponto 3)
  python -m vq2.pipeline quantize B   # alocacao por orcamento B (bpw medio) + LDLQ      (pontos 1,2,3)
  python -m vq2.pipeline kd NAME      # fine-tuning por destilacao                       (ponto 4)
  python -m vq2.pipeline export NAME  # entropia + container so-texto                    (pontos 5,6)
"""
import os, sys, time, json, math
import torch
import torch.nn as nn
from .common import *
from .calib import list_targets, collect_hessians
from . import vq

SCAN_CFGS = [(12,), (16,), (16, 8), (16, 16)]
ALL_CFGS = [(12,), (16,), (16, 4), (16, 8), (16, 12), (16, 16)]


def target_weight(t):
    return t["module"].weight


def setup(load_hess=True):
    model = load_model()
    targets = list_targets(model)
    emb = model.model.embed_tokens.weight
    log(f"[setup] {len(targets)} matrizes alvo | lm_head tied: {model.lm_head.weight.data_ptr() == emb.data_ptr()}")
    return model, targets


# ----------------------------------------------------------------------------------------
# prepare
# ----------------------------------------------------------------------------------------
@torch.no_grad()
def collect_samples(model, targets, per_matrix=100000, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    S = []
    for t in targets:
        W = t["module"].weight.data
        out, n = W.shape
        rows = torch.randint(0, out, (min(out, 4096),), device=DEV, generator=g).unique()
        Wr = vq.get_rot(t["tag"], n).fwd(W[rows].float())
        s = Wr.pow(2).mean(1).sqrt().clamp_min(1e-6)
        u = (Wr / s[:, None]).reshape(rows.numel(), n // vq.D, vq.D)
        r = torch.randint(0, u.shape[0], (per_matrix,), device=DEV, generator=g)
        c = torch.randint(0, u.shape[1], (per_matrix,), device=DEV, generator=g)
        S.append(u[r, c])
    return torch.cat(S, 0)


def step_prepare():
    t0 = time.time()
    model, targets = setup()
    build_train_tokens()
    ev = Evaluator(teacher_model=model)
    base = ev.run(model)
    n_params = sum(p.numel() for p in model.parameters())
    base["params_text_only"] = n_params
    base["size_mb_bf16"] = round(n_params * 2 / 2 ** 20, 1)
    log("[BF16 baseline]", base)
    save_json(base, OUT / "bf16_baseline.json")
    collect_hessians(model, calib_tokens())
    cbp = OUT / "codebooks.pt"
    if not cbp.exists():
        log("[prepare] coletando amostras e treinando codebooks universais...")
        S = collect_samples(model, targets)
        log(f"  amostras {tuple(S.shape)} std={S.std().item():.4f}")
        cbs = vq.train_codebooks(S, ALL_CFGS)
        torch.save(cbs.state(), cbp)
    log(f"[prepare] ok em {time.time()-t0:.0f}s")


def load_codebooks():
    d = torch.load(OUT / "codebooks.pt")
    return vq.Codebooks({k: v.half().float() for k, v in d.items()}).to(DEV)


# ----------------------------------------------------------------------------------------
# scan (ponto 3): sensibilidade medida por KL ao quantizar UMA matriz por vez
# ----------------------------------------------------------------------------------------
def step_scan(n_scan_seqs=16):
    model, targets = setup()
    Hs = collect_hessians(model, calib_tokens())
    cbs = load_codebooks()
    scan = mixed_train_set(n_scan_seqs // 2, n_scan_seqs // 2, offset_wiki=900, offset_code=450, seed=3)
    cache_path = OUT / "scan_cache.pt"
    if cache_path.exists():
        cache = torch.load(cache_path)
    else:
        cache = cache_teacher(model, scan)
        torch.save(cache, cache_path)
    res_path = OUT / "sensitivity.json"
    res = json.load(open(res_path)) if res_path.exists() else {}
    t0 = time.time()
    for ti, t in enumerate(targets):
        name = t["name"]
        W0 = t["module"].weight.data.clone()
        for cfg in SCAN_CFGS:
            key = f"{name}|{'-'.join(map(str, cfg))}"
            if key in res:
                continue
            qm, Wd, rel = vq.quantize_matrix(W0, Hs[t["tag"]], t["tag"], cfg, cbs)
            t["module"].weight.data.copy_(Wd.to(torch.bfloat16))
            kl, t1 = eval_kl(model, scan, cache)
            t["module"].weight.data.copy_(W0)
            res[key] = dict(kl=kl, top1=t1, rel=rel, n=int(W0.numel()))
            del qm, Wd
        save_json(res, res_path)
        log(f"[scan] {ti+1}/{len(targets)} {name}  " +
            " ".join(f"{'-'.join(map(str,c))}:{res[name+'|'+'-'.join(map(str,c))]['kl']:.4f}" for c in SCAN_CFGS) +
            f"  ({time.time()-t0:.0f}s)")


# ----------------------------------------------------------------------------------------
# alocacao por orcamento (ponto 3)
# ----------------------------------------------------------------------------------------
def level_bits(shape, cfg):
    out, n = shape
    return out * n / vq.D * sum(cfg) + out * 16


def allocate(targets, sens, budget_bpw, cfgs=SCAN_CFGS):
    """Lagrangiana sobre a envoltoria convexa de (bits, KL) por matriz. budget_bpw = bits medios por peso
    (incluindo escalas) sobre todos os pesos quantizados. Retorna {name: cfg}."""
    names = [t["name"] for t in targets]
    shapes = {t["name"]: tuple(t["module"].weight.shape) for t in targets}
    total_w = sum(a * b for a, b in shapes.values())
    B = budget_bpw * total_w
    opts = {}
    for nm in names:
        o = []
        for cfg in cfgs:
            r = sens[f"{nm}|{'-'.join(map(str, cfg))}"]
            o.append((level_bits(shapes[nm], cfg), r["kl"], cfg))
        o.sort(key=lambda z: z[0])
        best = float("inf")   # KL monotono nao-crescente com bits (remove ruido)
        o2 = []
        for b, k, c in o:
            best = min(best, k)
            o2.append((b, best, c))
        opts[nm] = o2

    def pick(lam):
        ch, tot = {}, 0.0
        for nm in names:
            b, k, c = min(opts[nm], key=lambda z: z[1] + lam * z[0])
            ch[nm] = c
            tot += b
        return ch, tot

    lo, hi = 0.0, 1e-3
    while pick(hi)[1] > B and hi < 1e12:
        hi *= 4
    # lo: lambda=0 -> maximo de bits
    for _ in range(60):
        mid = (lo * hi) ** 0.5 if lo > 0 else hi / 1e6
        ch, tot = pick(mid)
        if tot > B:
            lo = mid
        else:
            hi = mid
    ch, tot = pick(hi)
    pred_kl = sum(min((k for b, k, c in opts[nm] if c == ch[nm])) for nm in names)
    return ch, tot / total_w, pred_kl


def step_quantize(budget, name=None):
    budget = float(budget)
    name = name or f"b{budget:.2f}"
    model, targets = setup()
    sens = json.load(open(OUT / "sensitivity.json"))
    choice, eff_bpw, pred = allocate(targets, sens, budget)
    from collections import Counter
    log(f"[alloc] orcamento {budget} bpw -> efetivo {eff_bpw:.3f} | KL previsto (aditivo) {pred:.4f}")
    log("  niveis:", dict(Counter("-".join(map(str, c)) for c in choice.values())))
    quantize_choice(model, targets, choice, eff_bpw, budget, name)


def step_uniform(cfg_str, name=None):
    cfg = tuple(int(x) for x in cfg_str.split("-"))
    name = name or f"u{cfg_str}"
    model, targets = setup()
    choice = {t["name"]: cfg for t in targets}
    quantize_choice(model, targets, choice, vq.cfg_bpw(cfg), vq.cfg_bpw(cfg), name)


def quantize_choice(model, targets, choice, eff_bpw, budget, name):
    Hs = collect_hessians(model, calib_tokens())
    cbs = load_codebooks()
    ev = Evaluator()
    qmats, t0 = {}, time.time()
    for i, t in enumerate(targets):
        cfg = choice[t["name"]]
        qm, Wd, rel = vq.quantize_matrix(t["module"].weight.data, Hs[t["tag"]], t["tag"], cfg, cbs)
        t["module"].weight.data.copy_(Wd.to(torch.bfloat16))
        qmats[t["name"]] = qm
        if i % 10 == 0:
            log(f"  [quant] {i+1}/{len(targets)} {t['name']} cfg={cfg} rel_func={rel:.4f} ({time.time()-t0:.0f}s)")
    torch.save({"qmats": {k: dict(shape=v.shape, cfg=v.cfg, tag=v.tag, idx=v.idx, scale=v.scale) for k, v in qmats.items()},
                "choice": choice}, OUT / f"qmats_{name}.pt")
    r = ev.run(model)
    n_q = sum(v.n_weights() for v in qmats.values())
    n_dense = sum(p.numel() for p in model.parameters()) - n_q
    bits = sum(v.bits_fixed() for v in qmats.values()) + n_dense * 16
    r["size_mb_fixed"] = round(bits / 8 / 2 ** 20, 1)
    r["bpw_quantized"] = round(eff_bpw, 3)
    r["budget"] = budget
    log(f"[RESULT {name}]", r)
    save_json(r, OUT / f"result_{name}_ptq.json")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prepare":
        step_prepare()
    elif cmd == "scan":
        step_scan()
    elif cmd == "quantize":
        step_quantize(*sys.argv[2:])
    elif cmd == "uniform":
        step_uniform(*sys.argv[2:])
    elif cmd == "kd":
        from .kd import run_kd
        kw = {}
        for a in sys.argv[3:]:
            k, v = a.split("=")
            kw[k] = float(v) if "." in v or "e-" in v else int(v)
        run_kd(sys.argv[2], **kw)
    elif cmd == "export":
        from .export import step_export
        step_export(sys.argv[2], entropy=(len(sys.argv) < 4 or sys.argv[3] != "fixed"))
    else:
        raise SystemExit(f"comando desconhecido: {cmd}")
