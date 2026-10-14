"""
vq_lab.py - Laboratorio de quantizacao para fechar o gap contra o Q4_K_M.

Implementa, sobre os MESMOS dados/avaliacao (ver vq_data.py):
  - RTN escalar por grupo                      (referencia)
  - GPTQ escalar por grupo (Hessiana)          (referencia estado-da-arte simples)
  - RVQ 8-dim (k-means residual) sem/com rotacao de Hadamard
  - GPTQ-VQ: VQ residual 8-dim + compensacao de erro por Hessiana (+ Hadamard)
  - H5-A original (importado de run_qwen_rate_distortion_sweep.py)

Contabilidade de bits HONESTA (codigos de comprimento fixo + escalas + codebooks).
"""
import os, sys, time, math, json, gc
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "modelo")
GGUF_FILE = "Qwen3.5-2B-Q4_K_M.gguf"
DATA = os.path.join(ROOT, "data")
torch.set_num_threads(os.cpu_count() or 8)

# ----------------------------------------------------------------------------
# Modelo / dados / avaliacao
# ----------------------------------------------------------------------------

def load_model(dtype=torch.float32, gguf=False):
    from transformers import AutoModelForCausalLM
    kw = dict(torch_dtype=dtype, low_cpu_mem_usage=True)
    if gguf:
        kw["gguf_file"] = GGUF_FILE
    m = AutoModelForCausalLM.from_pretrained(MODEL_DIR, **kw)
    m.eval()
    return m


def load_tokens():
    return torch.load(os.path.join(DATA, "tokens.pt"))


@torch.no_grad()
def nll_sum(model, toks, bs=2):
    tot, cnt = 0.0, 0
    for i in range(0, len(toks), bs):
        x = toks[i:i + bs]
        logits = model(input_ids=x).logits
        l = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                            x[:, 1:].reshape(-1), reduction="sum")
        tot += l.item()
        cnt += x[:, 1:].numel()
    return tot, cnt


def eval_ppl(model, tokens, n_wiki=12, n_code=8):
    out = {}
    t0 = time.time()
    for name, key, n in (("wiki", "eval_wiki", n_wiki), ("code", "eval_code", n_code)):
        s, c = nll_sum(model, tokens[key][:n])
        out[name] = math.exp(s / c)
    out["avg"] = (out["wiki"] + out["code"]) / 2
    out["eval_s"] = time.time() - t0
    return out


# ----------------------------------------------------------------------------
# Hessianas
# ----------------------------------------------------------------------------

def ffn_modules(model):
    """Devolve dict nome -> modulo Linear das FFN (gate, up, down) por camada."""
    d = {}
    for i, layer in enumerate(model.model.layers):
        d[f"l{i}.gate"] = layer.mlp.gate_proj
        d[f"l{i}.up"] = layer.mlp.up_proj
        d[f"l{i}.down"] = layer.mlp.down_proj
    return d


@torch.no_grad()
def collect_hessians(model, calib_ids, bs=4, log=print):
    """H = sum x x^T por entrada. gate/up compartilham a entrada (chave 'lI.mlp_in')."""
    layers = model.model.layers
    H, N, hooks = {}, {}, []

    def mk(key):
        def hook(mod, inp):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            if key not in H:
                H[key] = torch.zeros(x.shape[1], x.shape[1])
                N[key] = 0
            H[key].addmm_(x.T, x)
            N[key] += x.shape[0]
        return hook

    for i, layer in enumerate(layers):
        hooks.append(layer.mlp.gate_proj.register_forward_pre_hook(mk(f"l{i}.mlp_in")))
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(mk(f"l{i}.down_in")))
    t0 = time.time()
    for b in range(0, len(calib_ids), bs):
        model(input_ids=calib_ids[b:b + bs])
        log(f"  hessian calib {b + bs}/{len(calib_ids)} ({time.time() - t0:.0f}s)")
    for h in hooks:
        h.remove()
    for k in H:
        H[k] /= N[k]
    return H


def hess_key(name):
    layer, kind = name.split(".")
    return f"{layer}.mlp_in" if kind in ("gate", "up") else f"{layer}.down_in"


# ----------------------------------------------------------------------------
# Rotacao de Hadamard em blocos (incoerencia)
# ----------------------------------------------------------------------------

_HAD = {}


def hadamard(n):
    if n not in _HAD:
        h = torch.ones(1, 1)
        while h.shape[0] < n:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        _HAD[n] = h / math.sqrt(n)
    return _HAD[n]


def make_rotation(n_in, seed, block=2048):
    """R ortogonal [n_in,n_in] = blockdiag(diag(s_b) H_block)."""
    g = torch.Generator().manual_seed(seed)
    bs = block if n_in % block == 0 else n_in
    nb = n_in // bs
    R = torch.zeros(n_in, n_in)
    hb = hadamard(bs)
    for b in range(nb):
        s = (torch.randint(0, 2, (bs,), generator=g) * 2 - 1).float()
        R[b * bs:(b + 1) * bs, b * bs:(b + 1) * bs] = s.unsqueeze(1) * hb
    return R  # W' = W @ R ; W = W' @ R.T


# ----------------------------------------------------------------------------
# Escalar: RTN e GPTQ
# ----------------------------------------------------------------------------

def _asym_quant(w, bits):
    """w: [rows, g] -> (q dequantizado, scale, zero) por linha."""
    mx, mn = w.max(1, keepdim=True).values, w.min(1, keepdim=True).values
    qmax = 2 ** bits - 1
    scale = ((mx - mn) / qmax).clamp_min(1e-8).half().float()
    zero = (-mn / scale).round().clamp(0, qmax).half().float()
    q = (torch.round(w / scale) + zero).clamp(0, qmax)
    return (q - zero) * scale, scale, zero


def rtn_scalar(W, bits=4, group=128):
    out, inn = W.shape
    Wg = W.float().reshape(out, inn // group, group).reshape(-1, group)
    q, _, _ = _asym_quant(Wg, bits)
    return q.reshape(out, inn), bits + 32.0 / group


def gptq_prep(H, damp=0.01):
    H = H.clone()
    d = torch.diag(H)
    dead = d == 0
    H[dead, dead] = 1.0
    H.diagonal().add_(damp * torch.mean(torch.diag(H)))
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    U = torch.linalg.cholesky(Hinv, upper=True)
    return U, dead


def gptq_scalar(W, H, bits=4, group=128, damp=0.01):
    W = W.clone().float()
    out, inn = W.shape
    U, dead = gptq_prep(H, damp)
    W[:, dead] = 0
    Q = torch.zeros_like(W)
    for i1 in range(0, inn, group):
        i2 = i1 + group
        W1 = W[:, i1:i2].clone()
        Err = torch.zeros_like(W1)
        Hb = U[i1:i2, i1:i2]
        _, scale, zero = _asym_quant(W1, bits)
        qmax = 2 ** bits - 1
        for i in range(group):
            w = W1[:, i]
            q = ((torch.round(w.unsqueeze(1) / scale) + zero).clamp(0, qmax) - zero) * scale
            q = q.squeeze(1)
            Q[:, i1 + i] = q
            err = (w - q) / Hb[i, i]
            W1[:, i:] -= err.unsqueeze(1) * Hb[i, i:].unsqueeze(0)
            Err[:, i] = err
        W[:, i2:] -= Err @ U[i1:i2, i2:]
    return Q, bits + 32.0 / group


# ----------------------------------------------------------------------------
# VQ residual 8-dim
# ----------------------------------------------------------------------------

def kmeans(x, K, iters=12, gen=None, chunk=None):
    N = x.shape[0]
    chunk = chunk or max(1024, (1 << 26) // K)
    idx = torch.randperm(N, generator=gen)[:K]
    c = x[idx].clone()
    x2 = (x * x).sum(1, keepdim=True)
    for _ in range(iters):
        asg = torch.empty(N, dtype=torch.long)
        for s in range(0, N, chunk):
            xs = x[s:s + chunk]
            d = (xs * xs).sum(1, keepdim=True) - 2 * xs @ c.T + (c * c).sum(1).unsqueeze(0)
            asg[s:s + chunk] = d.argmin(1)
        cnt = torch.bincount(asg, minlength=K).float()
        new = torch.zeros_like(c).index_add_(0, asg, x)
        nz = cnt > 0
        c[nz] = new[nz] / cnt[nz].unsqueeze(1)
        if (~nz).any():  # reinicia clusters vazios
            c[~nz] = x[torch.randperm(N, generator=gen)[:int((~nz).sum())]]
    return c


def nearest(x, c, chunk=None):
    chunk = chunk or max(1024, (1 << 26) // c.shape[0])
    out = torch.empty(x.shape[0], dtype=torch.long)
    c2 = (c * c).sum(1).unsqueeze(0)
    for s in range(0, x.shape[0], chunk):
        xs = x[s:s + chunk]
        out[s:s + chunk] = (c2 - 2 * xs @ c.T).argmin(1)
    return out


def train_rvq(x, Ks, n_train=262144, iters=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    sub = x[torch.randperm(x.shape[0], generator=g)[:n_train]]
    cbs, res = [], sub
    for K in Ks:
        c = kmeans(res, K, iters, g)
        cbs.append(c)
        res = res - c[nearest(res, c)]
    return cbs


def encode_rvq(x, cbs):
    rec = torch.zeros_like(x)
    for c in cbs:
        rec_i = c[nearest(x - rec, c)]
        rec = rec + rec_i
    return rec


def rvq_bits(Ks, n_weights, n_rows, dim=8):
    cb_bits = sum(K * dim * 16 for K in Ks)
    return sum(math.log2(K) for K in Ks) / dim + (n_rows * 16 + cb_bits) / n_weights


def quant_rvq(W, Ks, R=None, H=None, hessian_aware=False, seed=0, dim=8):
    """
    W [out,in] fp32. R: rotacao opcional (in x in). H: Hessiana (ja na base original).
    hessian_aware=True -> GPTQ-VQ (propagacao de erro por blocos de `dim` colunas).
    Retorna (W_hat na base original, bpw).
    """
    out, inn = W.shape
    Wr = W @ R if R is not None else W.clone()
    scale = Wr.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-8).half().float()
    Wn = Wr / scale
    cbs = train_rvq(Wn.reshape(-1, dim), Ks, seed=seed)
    cbs = [c.half().float() for c in cbs]
    if not hessian_aware:
        rec = encode_rvq(Wn.reshape(-1, dim), cbs).reshape(out, inn)
    else:
        Hr = R.T @ H @ R if R is not None else H
        U, dead = gptq_prep(Hr)
        Wn = Wn.clone()
        Wn[:, dead] = 0
        rec = torch.zeros_like(Wn)
        BS = 128
        for i1 in range(0, inn, BS):
            i2 = min(i1 + BS, inn)
            W1 = Wn[:, i1:i2].clone()
            Err = torch.zeros_like(W1)
            for j in range(0, i2 - i1, dim):
                blk = W1[:, j:j + dim]
                q = encode_rvq(blk, cbs)
                rec[:, i1 + j:i1 + j + dim] = q
                Ubb = U[i1 + j:i1 + j + dim, i1 + j:i1 + j + dim]
                E = torch.linalg.solve_triangular(Ubb, (blk - q), upper=True, left=False)
                Err[:, j:j + dim] = E
                if j + dim < i2 - i1:
                    W1[:, j + dim:] -= E @ U[i1 + j:i1 + j + dim, i1 + j + dim:i2]
            if i2 < inn:
                Wn[:, i2:] -= Err @ U[i1:i2, i2:]
    What = rec * scale
    if R is not None:
        What = What @ R.T
    return What, rvq_bits(Ks, out * inn, out, dim)


# ----------------------------------------------------------------------------
# H5-A original (reaproveita o codigo existente sem executar o experimento)
# ----------------------------------------------------------------------------

_h5a_ns = {}


def h5a_original(W, seed=42, p_crit=0.05):
    if "fn" not in _h5a_ns:
        src = open(os.path.join(ROOT, "run_qwen_rate_distortion_sweep.py"), encoding="utf-8").read().splitlines()
        code = "import torch\n" + "\n".join(src[19:234])
        exec(code, _h5a_ns)
        _h5a_ns["fn"] = _h5a_ns["quantize_matrix_h5a_4bpw"]
    rec, _, _ = _h5a_ns["fn"](W, num_ctrl=3, num_spec=3, p_crit=p_crit, seed=seed)
    # bits reais com codigos de comprimento fixo:
    # familia(3)+idx em 26 base+78 ctrl+3 spec; r1: 3+log2 26; r2,r3: log2 26; r4: 5% * log2 26
    bits_vec = 3 + math.log2(107) + 3 + math.log2(26) * 3 + 0.05 * math.log2(26)
    return rec.float(), bits_vec / 8.0
