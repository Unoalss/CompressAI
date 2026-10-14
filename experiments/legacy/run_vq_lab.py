"""
run_vq_lab.py - driver dos experimentos (FFN-only, mesmo protocolo dos testes anteriores,
mas com avaliacao em corpora reais e calibracao disjunta da avaliacao).

Uso:
  python run_vq_lab.py smoke
  python run_vq_lab.py hess              # coleta Hessianas FFN (salva em data/hess_ffn.pt)
  python run_vq_lab.py run exp1 exp2 ... # roda experimentos e grava em vq_lab_results.json
"""
import os, sys, time, json, math
import torch
import vq_lab as L

RESULTS = os.path.join(L.ROOT, "vq_lab_results.json")
HESS = os.path.join(L.DATA, "hess_ffn.pt")


def log(*a):
    print(*a, flush=True)


_R = {}


def rot(inn, i):
    if (inn, i) not in _R:
        _R[(inn, i)] = L.make_rotation(inn, seed=1000 + i)
    return _R[(inn, i)]


def make_exps():
    E = {}
    E["rtn4_g128"] = lambda W, H, i: L.rtn_scalar(W, 4, 128)
    E["rtn4_g32"] = lambda W, H, i: L.rtn_scalar(W, 4, 32)
    E["gptq4_g128"] = lambda W, H, i: L.gptq_scalar(W, H, 4, 128)
    E["h5a_orig"] = lambda W, H, i: L.h5a_original(W, seed=42 + i)
    for tag, Ks in (("2", [256] * 2), ("3", [256] * 3), ("35", [256, 256, 256, 16]),
                    ("4", [256] * 4)):
        E[f"rvq{tag}"] = (lambda Ks: lambda W, H, i: L.quant_rvq(W, Ks, seed=i))(Ks)
        E[f"had_rvq{tag}"] = (lambda Ks: lambda W, H, i: L.quant_rvq(W, Ks, R=rot(W.shape[1], i), seed=i))(Ks)
        E[f"had_gptqvq{tag}"] = (lambda Ks: lambda W, H, i: L.quant_rvq(
            W, Ks, R=rot(W.shape[1], i), H=H, hessian_aware=True, seed=i))(Ks)
    E["gptqvq4"] = lambda W, H, i: L.quant_rvq(W, [256] * 4, H=H, hessian_aware=True, seed=i)
    E["gptq4_g64"] = lambda W, H, i: L.gptq_scalar(W, H, 4, 64)
    E["gptq3_g128"] = lambda W, H, i: L.gptq_scalar(W, H, 3, 128)
    E["gptq3_g64"] = lambda W, H, i: L.gptq_scalar(W, H, 3, 64)
    # VQ de dimensao menor com codebook unico grande (GPTVQ-style)
    for tag, dim, Ks in (("2d4", 2, [256]), ("2d3", 2, [64]), ("4d3", 4, [4096]),
                         ("4d35", 4, [16384]), ("4d25", 4, [1024])):
        E[f"gv{tag}"] = (lambda dim, Ks: lambda W, H, i: L.quant_rvq(
            W, Ks, H=H, hessian_aware=True, seed=i, dim=dim))(dim, Ks)
        E[f"had_gv{tag}"] = (lambda dim, Ks: lambda W, H, i: L.quant_rvq(
            W, Ks, R=rot(W.shape[1], i), H=H, hessian_aware=True, seed=i, dim=dim))(dim, Ks)
        E[f"vq{tag}_nohess"] = (lambda dim, Ks: lambda W, H, i: L.quant_rvq(
            W, Ks, seed=i, dim=dim))(dim, Ks)
    return E


def rel_errs(W, What, H):
    D = (W - What)
    w = (D.pow(2).sum() / W.pow(2).sum()).item()
    o = (torch.einsum("ij,jk,ik->", D, H, D) / torch.einsum("ij,jk,ik->", W, H, W)).item()
    return w, o


def smoke():
    m = L.load_model()
    tok = L.load_tokens()
    log("modelo carregado; camadas:", len(m.model.layers))
    log(m.model.layers[0])
    t = time.time()
    with torch.no_grad():
        m(input_ids=tok["eval_wiki"][:2])
    log(f"forward bs=2 x 512: {time.time() - t:.1f}s")
    t = time.time()
    Hs = L.collect_hessians(m, tok["calib_wiki"][:8], bs=4, log=log)
    log(f"hessianas 8 seqs: {time.time() - t:.1f}s")
    i = 5
    mods = L.ffn_modules(m)
    E = make_exps()
    for kind in ("gate", "down"):
        W = mods[f"l{i}.{kind}"].weight.data.float()
        H = Hs[L.hess_key(f"l{i}.{kind}")]
        for name in ("rtn4_g128", "gptq4_g128", "rvq4", "had_rvq4", "had_gptqvq4", "had_gptqvq3", "h5a_orig"):
            t = time.time()
            What, bpw = E[name](W, H, i)
            w, o = rel_errs(W, What, H)
            log(f"l{i}.{kind} {name:<14} bpw={bpw:.3f}  relW={w:.5f}  relOut={o:.5f}  {time.time() - t:.1f}s")


def do_hess():
    m = L.load_model()
    tok = L.load_tokens()
    ids = torch.cat([tok["calib_wiki"], tok["calib_code"]])
    Hs = L.collect_hessians(m, ids, bs=4, log=log)
    torch.save(Hs, HESS)
    log("salvo", HESS)


def load_results():
    return json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}


def save_result(name, r):
    d = load_results()
    d[name] = r
    json.dump(d, open(RESULTS, "w"), indent=2)


def run(names):
    E = make_exps()
    m = L.load_model()
    tok = L.load_tokens()
    mods = L.ffn_modules(m)
    orig = {k: v.weight.data.clone() for k, v in mods.items()}
    Hs = torch.load(HESS) if os.path.exists(HESS) else None
    ndone = load_results()
    if "bf16" not in ndone:
        r = L.eval_ppl(m, tok)
        log("bf16 (fp32 run):", r)
        save_result("bf16", r)
    for name in names:
        if name in ndone and "--force" not in sys.argv:
            log("pulando (ja feito):", name)
            continue
        log("=" * 80, f"\nEXPERIMENTO {name}")
        t0 = time.time()
        sw, so, wsum, bits_sum, n_sum = 0.0, 0.0, 0, 0.0, 0
        for k, mod in mods.items():
            W = orig[k].float()
            i = int(k.split(".")[0][1:])
            H = Hs[L.hess_key(k)] if Hs is not None else None
            What, bpw = E[name](W, H, i)
            mod.weight.data.copy_(What)
            n = W.numel()
            bits_sum += bpw * n
            n_sum += n
            if H is not None:
                w, o = rel_errs(W, What, H)
                sw += w
                so += o
                wsum += 1
            if k.endswith(".down") and (i % 4 == 3 or i == 0):
                log(f"  camada {i:>2} ok ({time.time() - t0:.0f}s) bpw={bpw:.3f}")
        qt = time.time() - t0
        r = L.eval_ppl(m, tok)
        r.update(bpw=bits_sum / n_sum, quant_s=qt,
                 mean_rel_w=sw / max(wsum, 1), mean_rel_out=so / max(wsum, 1))
        log(f"RESULT {name}: {r}")
        save_result(name, r)
        for k, mod in mods.items():
            mod.weight.data.copy_(orig[k])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "smoke":
        smoke()
    elif cmd == "hess":
        do_hess()
    elif cmd == "run":
        run([a for a in sys.argv[2:] if not a.startswith("--")])
