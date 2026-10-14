"""
vq2/calib.py - enumera as matrizes alvo e coleta as Hessianas H = E[x x^T] (ponto 1) por grupo de entrada.
"""
import os, re, time
import torch
import torch.nn as nn
from .common import DEV, OUT, log

GROUP = {
    "gate_proj": "mlp_in", "up_proj": "mlp_in", "down_proj": "down_in",
    "q_proj": "attn_in", "k_proj": "attn_in", "v_proj": "attn_in", "o_proj": "o_in",
    "in_proj_qkv": "dnet_in", "in_proj_z": "dnet_in", "out_proj": "dnet_out_in",
}


def list_targets(model):
    """Retorna lista de dicts: name, module, tag (grupo de Hessiana/rotacao), layer, kind."""
    T = []
    for name, mod in model.named_modules():
        if name.endswith("embed_tokens"):
            T.append(dict(name=name, module=mod, tag="final", layer=-1, kind="embed"))
            continue
        m = re.search(r"layers\.(\d+)\.", name)
        if not m or not isinstance(mod, nn.Linear):
            continue
        leaf = name.split(".")[-1]
        if leaf in GROUP:
            L = int(m.group(1))
            T.append(dict(name=name, module=mod, tag=f"L{L}.{GROUP[leaf]}", layer=L, kind=leaf))
    return T


@torch.no_grad()
def collect_hessians(model, tokens, layer_ranges=((0, 12), (12, 24)), bs=4, path=None):
    """Coleta H por tag. Duas passadas (metade das camadas cada) para caber na GPU de 12 GB."""
    path = path or (OUT / "hessians.pt")
    if os.path.exists(path):
        log(f"[calib] carregando Hessianas de {path}")
        return torch.load(path)
    targets = list_targets(model)
    final_norm = model.model.norm
    Hs = {}
    for (a, b) in layer_ranges:
        t0 = time.time()
        acc, cnt, hooks = {}, {}, []
        first_of = {}
        for t in targets:
            if t["tag"] == "final" or not (a <= t["layer"] < b):
                continue
            first_of.setdefault(t["tag"], t["module"])
        for tag, mod in first_of.items():
            def mk(tag):
                def hook(m, args):
                    x = args[0].reshape(-1, args[0].shape[-1])
                    for i in range(0, x.shape[0], 4096):
                        xc = x[i:i + 4096].float()
                        g = xc.T @ xc
                        if tag in acc:
                            acc[tag] += g
                        else:
                            acc[tag] = g
                    cnt[tag] = cnt.get(tag, 0) + x.shape[0]
                return hook
            hooks.append(mod.register_forward_pre_hook(mk(tag)))
        if a == layer_ranges[0][0]:
            def fhook(m, args, out):
                x = out.reshape(-1, out.shape[-1])
                for i in range(0, x.shape[0], 4096):
                    xc = x[i:i + 4096].float()
                    acc["final"] = acc.get("final", 0) + xc.T @ xc
                cnt["final"] = cnt.get("final", 0) + x.shape[0]
            hooks.append(final_norm.register_forward_hook(fhook))
        for i in range(0, tokens.shape[0], bs):
            model(input_ids=tokens[i:i + bs].to(DEV))
        for h in hooks:
            h.remove()
        for k in acc:
            Hs[k] = (acc[k] / cnt[k]).cpu()
        del acc
        torch.cuda.empty_cache()
        log(f"[calib] camadas {a}-{b-1}: {len(first_of)} Hessianas em {time.time()-t0:.0f}s")
    torch.save(Hs, path)
    return Hs
