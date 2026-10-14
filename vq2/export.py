"""
vq2/export.py - pontos 5 e 6:
  * codificacao de entropia (ANS estatico, i.i.d. por estagio) dos indices dos codebooks
  * container so-texto (sem vision tower / MTP): indices + escalas fp16 + codebooks fp16 + parametros densos bf16
  * loader que reconstroi o modelo SO a partir do container (round-trip verificado por PPL)
"""
import os, time, io
import numpy as np
import torch
from .common import *
from . import vq

try:
    import constriction
    HAVE_ANS = True
except Exception:  # pragma: no cover
    HAVE_ANS = False


def _probs(counts):
    c = counts.astype(np.float64)
    p = (c + 0.25) / (c.sum() + 0.25 * len(c))
    return p.astype(np.float32)


def ans_encode(sym, counts):
    model = constriction.stream.model.Categorical(_probs(counts), perfect=False)
    coder = constriction.stream.stack.AnsCoder()
    coder.encode_reverse(sym.astype(np.int32), model)
    return coder.get_compressed()


def ans_decode(comp, counts, n):
    model = constriction.stream.model.Categorical(_probs(counts), perfect=False)
    coder = constriction.stream.stack.AnsCoder(comp)
    return coder.decode(model, n)


def pack_fixed(sym, bits):
    sym = sym.astype(np.uint32)
    if bits == 16:
        return sym.astype(np.uint16).view(np.uint8)
    if bits == 8:
        return sym.astype(np.uint8)
    if bits == 4:
        if len(sym) % 2: sym = np.append(sym, 0)
        return (sym[0::2] | (sym[1::2] << 4)).astype(np.uint8)
    if bits == 12:
        if len(sym) % 2: sym = np.append(sym, 0)
        a, b = sym[0::2], sym[1::2]
        w = a | (b << 12)    # 24 bits
        return np.stack([w & 255, (w >> 8) & 255, (w >> 16) & 255], 1).astype(np.uint8).reshape(-1)
    raise ValueError(bits)


def unpack_fixed(buf, bits, n):
    if bits == 16:
        return buf.view(np.uint16)[:n].astype(np.int64)
    if bits == 8:
        return buf[:n].astype(np.int64)
    if bits == 4:
        b = buf.astype(np.uint32)
        out = np.empty(len(b) * 2, dtype=np.int64); out[0::2] = b & 15; out[1::2] = b >> 4
        return out[:n]
    if bits == 12:
        b = buf.reshape(-1, 3).astype(np.uint32)
        w = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        out = np.empty(len(w) * 2, dtype=np.int64); out[0::2] = w & 4095; out[1::2] = w >> 12
        return out[:n]
    raise ValueError(bits)


def step_export(name, entropy=True, verify=True):
    fpath = OUT / f"final_{name}.pt"
    if fpath.exists():
        F = torch.load(fpath)
        Q = F["qmats"]
        codebooks = F["codebooks"]
        norms = F.get("norms", {})
    else:
        qpath = OUT / f"qmats_{name}.pt"
        Q_data = torch.load(qpath)
        Q = Q_data["qmats"]
        codebooks = torch.load(OUT / "codebooks.pt")
        norms = {}
    names = list(Q.keys())
    # --- fluxos de indices por chave de estagio
    streams = {}
    for nm in names:
        q = Q[nm]
        for i, a in enumerate(q["idx"]):
            streams.setdefault(vq.cfg_key(q["cfg"], i), []).append((nm, a.numpy().reshape(-1)))
    container = {"names": names, "meta": {nm: dict(shape=Q[nm]["shape"], cfg=Q[nm]["cfg"], tag=Q[nm]["tag"]) for nm in names},
                 "streams": {}, "scales": {nm: Q[nm]["scale"].numpy() for nm in names}}
    stats = {}
    for key, lst in streams.items():
        bits = int(key.split("_")[-1])
        allsym = np.concatenate([a for _, a in lst])
        counts = np.bincount(allsym, minlength=2 ** bits).astype(np.uint32)
        p = counts[counts > 0] / counts.sum()
        H = float(-(p * np.log2(p)).sum())
        stats[key] = dict(n=int(len(allsym)), bits_fixed=bits, entropy_bits=H)
        entry = dict(sizes=[len(a) for _, a in lst], counts=counts, bits=bits)
        if entropy and HAVE_ANS:
            t0 = time.time()
            comp = ans_encode(allsym, counts)
            entry["ans"] = torch.from_numpy(np.ascontiguousarray(comp))
            stats[key]["ans_bits_per_sym"] = comp.nbytes * 8 / len(allsym)
            log(f"[export] ANS '{key}': {bits} -> {stats[key]['ans_bits_per_sym']:.4f} bits/simbolo (entropia {H:.4f}) em {time.time()-t0:.0f}s")
            entry["mode"] = "ans"
        else:
            entry["fixed"] = torch.from_numpy(np.ascontiguousarray(pack_fixed(allsym, bits)))
            entry["mode"] = "fixed"
        if entry["mode"] == "ans":
            entry["counts"] = torch.from_numpy(counts)
        container["streams"][key] = entry
    # codebooks fp16 (aluno) + codebook congelado da embedding (inicial)
    container["codebooks"] = {k: v.half() for k, v in codebooks.items()}
    emb_name = [n for n in names if n.endswith("embed_tokens")][0]
    init = torch.load(OUT / "codebooks.pt")
    container["codebooks_embed"] = {vq.cfg_key(Q[emb_name]["cfg"], i): init[vq.cfg_key(Q[emb_name]["cfg"], i)].half().float().half()
                                    for i in range(len(Q[emb_name]["cfg"]))}
    # parametros densos (tudo que nao e quantizado) - ponto 6: SO modelo de texto
    qnames = set(names)
    model = load_model(device="cpu")
    dense = {}
    for n, p in model.named_parameters():
        if n.rsplit(".weight", 1)[0] in qnames:
            continue
        dense[n] = norms[n].to(torch.bfloat16) if n in norms else p.detach().to(torch.bfloat16)
    container["dense"] = dense
    path = OUT / f"model_{name}.vq2"
    torch.save(container, path)
    size = os.path.getsize(path)
    n_q = sum(int(np.prod(Q[nm]["shape"])) for nm in names)
    n_dense = sum(v.numel() for v in dense.values())
    fixed_bits = sum(vq.QMat(Q[nm]["shape"], Q[nm]["cfg"], Q[nm]["tag"], Q[nm]["idx"], Q[nm]["scale"]).bits_fixed() for nm in names) + n_dense * 16
    ent_bits = sum(s["entropy_bits"] * s["n"] for s in stats.values()) + sum(Q[nm]["shape"][0] * 16 for nm in names) + n_dense * 16
    info = dict(file_mb=round(size / 2 ** 20, 1), fixed_mb=round(fixed_bits / 8 / 2 ** 20, 1),
                entropy_ideal_mb=round(ent_bits / 8 / 2 ** 20, 1),
                bpw_file=round(size * 8 / (n_q + n_dense), 3), n_quant=n_q, n_dense=n_dense, streams=stats,
                mode="ans" if (entropy and HAVE_ANS) else "fixed")
    log(f"[export] {path.name}: {info['file_mb']} MB (fixo {info['fixed_mb']} MB | entropia ideal {info['entropy_ideal_mb']} MB) | "
        f"Q4_K_M = 1221.5 MB | BF16 so-texto = {(n_q+n_dense)*2/2**20:.0f} MB")
    save_json(info, OUT / f"export_{name}.json")
    if verify:
        r = verify_container(path)
        info["verified"] = r
        save_json(info, OUT / f"export_{name}.json")
    return info


@torch.no_grad()
def load_compressed(path, device=DEV):
    from transformers import AutoConfig, AutoModelForCausalLM
    C = torch.load(path, weights_only=False)
    cfg = AutoConfig.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).to(device).eval()
    # indices
    idx = {nm: [None] * len(C["meta"][nm]["cfg"]) for nm in C["names"]}
    for key, e in C["streams"].items():
        n = int(sum(e["sizes"]))
        if e["mode"] == "ans":
            ans_data = e["ans"].numpy() if isinstance(e["ans"], torch.Tensor) else e["ans"]
            cnt_data = e["counts"].numpy() if isinstance(e["counts"], torch.Tensor) else e["counts"]
            sym = ans_decode(ans_data, cnt_data, n)
        else:
            fix_data = e["fixed"].numpy() if isinstance(e["fixed"], torch.Tensor) else e["fixed"]
            sym = unpack_fixed(fix_data, e["bits"], n)
        off = 0
        for nm, sz in zip([nm for nm in C["names"] if key in [vq.cfg_key(C["meta"][nm]["cfg"], i) for i in range(len(C["meta"][nm]["cfg"]))]], e["sizes"]):
            cfg_ = C["meta"][nm]["cfg"]
            i = [vq.cfg_key(cfg_, j) for j in range(len(cfg_))].index(key)
            idx[nm][i] = torch.from_numpy(np.asarray(sym[off:off + sz]).astype(np.int64)).view(C["meta"][nm]["shape"][0], -1)
            off += sz
    sd = dict(model.named_parameters())
    for n, v in C["dense"].items():
        sd[n].data.copy_(v.to(device))
    for nm in C["names"]:
        m = C["meta"][nm]
        is_emb = nm.endswith("embed_tokens")
        cbs = vq.Codebooks({k: v.float() for k, v in (C["codebooks_embed"] if is_emb else C["codebooks"]).items()}).to(device)
        sc = C["scales"][nm]
        sc_t = sc if isinstance(sc, torch.Tensor) else torch.from_numpy(sc)
        qm = vq.QMat(m["shape"], m["cfg"], m["tag"], idx[nm], sc_t)
        Wd = qm.dequant(cbs, device, torch.bfloat16)
        sd[nm + ".weight"].data.copy_(Wd)
    model.tie_weights()
    return model


def verify_container(path):
    model = load_compressed(path)
    r = Evaluator().run(model, full=True)
    log(f"[export] verificacao a partir do container: {r}")
    return r
