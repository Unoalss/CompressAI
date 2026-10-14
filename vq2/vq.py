"""
vq2/vq.py - nucleo do quantizador (pontos 1, 2, 3, 5).

  * Ponto 2: rotacao de incoerencia (permutacao + sinais aleatorios + Hadamard em blocos de 2048) na dim de
    entrada, escala por linha (fp16), codebook UNIVERSAL de 8-D (k-means em pesos rotacionados/normalizados)
    e quantizacao vetorial residual em estagios (RVQ): cfg=(16,) => 2 bpw, (16,8) => 3 bpw, (16,16) => 4 bpw.
  * Ponto 1: LDLQ / GPTQ em blocos de 8 colunas com a Hessiana H = E[x x^T] (erro de SAIDA, nao de peso).
  * Ponto 5: contagem de bits com entropia (Shannon == tamanho de um codificador aritmetico/ANS ideal).
"""
import math, time, hashlib
import torch
import torch.nn as nn
from .common import DEV, log

D = 8          # dimensao do vetor quantizado
HB = 2048      # bloco de Hadamard


# ----------------------------------------------------------------------------------------
# Rotacao de incoerencia
# ----------------------------------------------------------------------------------------
def hadamard(n):
    assert n & (n - 1) == 0
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H


_HCACHE = {}


def _hmat(device, dtype, hb=HB):
    k = (str(device), dtype, hb)
    if k not in _HCACHE:
        _HCACHE[k] = hadamard(hb).to(device=device, dtype=dtype)
    return _HCACHE[k]


def tag_seed(tag):
    return int(hashlib.md5(tag.encode()).hexdigest()[:8], 16)


class Rot:
    """x' = R^T x  na ultima dimensao (n multiplo de 512 ou 2048). Ortogonal => preserva normas e produtos."""

    def __init__(self, n, seed, device=DEV):
        self.hb = min(HB, n) if (n % min(HB, n) == 0 and (min(HB, n) & (min(HB, n) - 1) == 0)) else HB
        assert n % self.hb == 0, f"n={n} nao e multiplo de hb={self.hb}"
        self.n, self.nb = n, n // self.hb
        g = torch.Generator().manual_seed(seed)
        self.perm = torch.randperm(n, generator=g).to(device)
        self.inv_perm = torch.argsort(self.perm)
        self.sign_bf16 = (torch.randint(0, 2, (n,), generator=g).float() * 2 - 1).to(device=device, dtype=torch.bfloat16)
        self.sign = self.sign_bf16
        self.scale = self.hb ** -0.5
        self._M_fwd_bf16 = None

    def get_fused_fwd_matrix(self):
        """Pré-computa M = scale * (P @ S @ H) para transformar 4 kernels (gather+mul+gemm+scale) em 1 único cuBLAS GEMM."""
        if self._M_fwd_bf16 is None:
            I = torch.eye(self.n, device=self.perm.device, dtype=torch.bfloat16)
            self._M_fwd_bf16 = self._orig_fwd(I).contiguous()
        return self._M_fwd_bf16

    def _blk(self, x):
        H = _hmat(x.device, x.dtype, self.hb)
        if x.dim() == 1 and self.nb == 1:
            return (x.unsqueeze(0) @ H).squeeze(0) * self.scale
        sh = x.shape
        y = x.reshape(*sh[:-1], self.nb, self.hb) @ H
        return (y * self.scale).reshape(sh)

    def _orig_fwd(self, x):
        s = self.sign_bf16 if x.dtype == torch.bfloat16 else self.sign.to(x.dtype)
        return self._blk(x[..., self.perm] * s)

    def fwd(self, x):
        if x.is_cuda and x.dtype == torch.bfloat16 and self.nb == 1:
            M = self.get_fused_fwd_matrix()
            return torch.matmul(x, M)
        return self._orig_fwd(x)

    def inv(self, y):
        s = self.sign_bf16 if y.dtype == torch.bfloat16 else self.sign.to(y.dtype)
        z = self._blk(y) * s
        return z[..., self.inv_perm]

    def rot_hessian(self, H):
        """H' = R^T H R (H [n,n] na base original)."""
        Hr = self.fwd(H)           # aplica na ultima dim
        Hr = self.fwd(Hr.T).T      # aplica na outra
        return Hr


_ROTS = {}


def get_rot(tag, n, device=DEV):
    k = (tag, n, str(device))
    if k not in _ROTS:
        _ROTS[k] = Rot(n, tag_seed(tag), device)
    return _ROTS[k]


# ----------------------------------------------------------------------------------------
# Codebooks (universais) e k-means em GPU
# ----------------------------------------------------------------------------------------
def cfg_key(cfg, i):
    return "_".join(str(b) for b in cfg[: i + 1])


def cfg_bpw(cfg):
    return sum(cfg) / D


@torch.no_grad()
def assign(x, cb, chunk=None):
    """argmin_k ||x - cb_k||^2 para x [n,8] (fp32); retorna idx int64 [n]."""
    K = cb.shape[0]
    if chunk is None:
        chunk = max(1024, int(2.5e8 // K))
    cb16 = cb.half()
    c2 = (cb.float() ** 2).sum(1).half()
    out = torch.empty(x.shape[0], dtype=torch.long, device=x.device)
    x16 = x.half()
    for i in range(0, x.shape[0], chunk):
        s = torch.addmm(c2, x16[i:i + chunk], cb16.T, alpha=-2.0)   # [c,K] fp16
        out[i:i + chunk] = s.argmin(1)
    return out


@torch.no_grad()
def kmeans(x, K, iters=15, seed=0, verbose=True):
    g = torch.Generator(device=x.device).manual_seed(seed)
    N = x.shape[0]
    c = x[torch.randperm(N, device=x.device, generator=g)[:K]].clone()
    for it in range(iters):
        a = assign(x, c)
        sums = torch.zeros_like(c).index_add_(0, a, x)
        cnt = torch.bincount(a, minlength=K).float()
        empty = cnt == 0
        newc = sums / cnt.clamp_min(1)[:, None]
        if empty.any():
            newc[empty] = x[torch.randint(0, N, (int(empty.sum()),), device=x.device, generator=g)]
        c = newc
        if verbose and (it % 5 == 0 or it == iters - 1):
            mse = ((x[:200000] - c[assign(x[:200000], c)]) ** 2).sum(1).mean().item() / D
            log(f"    kmeans K={K} it={it} mse/dim={mse:.5f} empty={int(empty.sum())}")
    return c


class Codebooks:
    """Dicionario {chave: tensor[K,8]} para todos os estagios usados."""

    def __init__(self, d=None):
        self.cb = d or {}

    def get(self, cfg, i):
        return self.cb[cfg_key(cfg, i)]

    def to(self, dev):
        self.cb = {k: v.to(dev) for k, v in self.cb.items()}
        return self

    def state(self):
        return {k: v.cpu() for k, v in self.cb.items()}


@torch.no_grad()
def train_codebooks(samples, cfgs, iters=15):
    """samples: [N,8] normalizados (std~1). Treina estagio 1 (cada K distinto) e os residuais."""
    cbs = Codebooks()
    # estagios 1
    first = sorted({c[0] for c in cfgs})
    for b in first:
        log(f"  [codebook] estagio1 K=2^{b}")
        cbs.cb[str(b)] = kmeans(samples, 2 ** b, iters=iters if b > 8 else 25)
    # estagios seguintes
    for cfg in cfgs:
        for i in range(1, len(cfg)):
            key = cfg_key(cfg, i)
            if key in cbs.cb:
                continue
            res = samples.clone()
            for j in range(i):
                cb = cbs.cb[cfg_key(cfg, j)]
                res -= cb[assign(res, cb)]
            log(f"  [codebook] residual {key} (std res={res.std().item():.4f})")
            cbs.cb[key] = kmeans(res, 2 ** cfg[i], iters=iters if cfg[i] > 8 else 25)
    return cbs


@torch.no_grad()
def encode(u, cfg, cbs):
    """u [n,8] -> (lista de idx por estagio, reconstrucao)."""
    rec = torch.zeros_like(u)
    idxs = []
    res = u
    for i in range(len(cfg)):
        cb = cbs.cb[cfg_key(cfg, i)]
        a = assign(res, cb)
        q = cb[a]
        rec = rec + q
        res = res - q
        idxs.append(a)
    return idxs, rec


# ----------------------------------------------------------------------------------------
# LDLQ / GPTQ com Hessiana
# ----------------------------------------------------------------------------------------
@torch.no_grad()
def chol_inv_upper(Hr, damp=0.01):
    """U tal que H^-1 = U^T U (U triangular superior), com amortecimento."""
    n = Hr.shape[0]
    H = Hr.clone().double()
    mean = torch.diag(H).mean()
    for attempt in range(6):
        Hd = H.clone()
        Hd.diagonal().add_(damp * mean * (10 ** attempt))
        try:
            L = torch.linalg.cholesky(Hd)
            Hinv = torch.cholesky_inverse(L)
            U = torch.linalg.cholesky(Hinv, upper=True)
            return U.float()
        except Exception:
            continue
    raise RuntimeError("Cholesky falhou")


@torch.no_grad()
def ldlq(Wr, s, U, cfg, cbs, bs=128, row_chunk=65536):
    """Quantiza Wr [out,n] (ja rotacionado) com escala por linha s [out] usando feedback de erro via U.
    Retorna (idx[list] cada [out,n/8] int32, Q [out,n] reconstrucao rotacionada)."""
    out, n = Wr.shape
    ng = n // D
    # inversas dos blocos diagonais 8x8 de U
    Ublk = torch.stack([U[g * D:(g + 1) * D, g * D:(g + 1) * D] for g in range(ng)])
    Uinv = torch.linalg.inv(Ublk)
    idx = [torch.empty(out, ng, dtype=torch.int32, device=Wr.device) for _ in cfg]
    Q = torch.empty_like(Wr)
    for r0 in range(0, out, row_chunk):
        r1 = min(r0 + row_chunk, out)
        W = Wr[r0:r1].clone()
        sc = s[r0:r1]
        inv_sc = (1.0 / sc)[:, None]
        for i1 in range(0, n, bs):
            i2 = min(i1 + bs, n)
            W1 = W[:, i1:i2].clone()
            Err = torch.zeros_like(W1)
            for j in range(0, i2 - i1, D):
                g = (i1 + j) // D
                blk = W1[:, j:j + D]
                ids, rec = encode(blk * inv_sc, cfg, cbs)
                q = rec * sc[:, None]
                Q[r0:r1, i1 + j:i1 + j + D] = q
                for k, a in enumerate(ids):
                    idx[k][r0:r1, g] = a.int()
                E = (blk - q) @ Uinv[g]
                Err[:, j:j + D] = E
                if j + D < i2 - i1:
                    W1[:, j + D:] -= E @ U[i1 + j:i1 + j + D, i1 + j + D:i2]
            if i2 < n:
                W[:, i2:] -= Err @ U[i1:i2, i2:]
    return idx, Q


class QMat:
    """Matriz quantizada: indices por estagio (int32 CPU), escala fp16 por linha, cfg e tag da rotacao."""

    def __init__(self, shape, cfg, tag, idx, scale):
        self.shape, self.cfg, self.tag = shape, tuple(cfg), tag
        self.idx = idx            # lista de tensores [out, n/8] int32
        self.scale = scale        # [out] fp16

    def n_weights(self):
        return self.shape[0] * self.shape[1]

    def bits_fixed(self):
        return self.n_weights() / D * sum(self.cfg) + self.shape[0] * 16

    @torch.no_grad()
    def dequant(self, cbs, device=DEV, dtype=torch.float32):
        out, n = self.shape
        rec = 0
        for i in range(len(self.cfg)):
            cb = cbs.cb[cfg_key(self.cfg, i)].to(device)
            rec = rec + cb[self.idx[i].to(device).long()]
        Q = rec.reshape(out, n) * self.scale.to(device).float()[:, None]
        W = get_rot(self.tag, n, device).inv(Q)
        return W.to(dtype)


@torch.no_grad()
def quantize_matrix(W, H, tag, cfg, cbs, damp=0.01, return_Q=False):
    """W [out,n] (qualquer dtype/dev), H [n,n] fp32 (base original). Retorna (QMat, W_deq bf16, metrica rel_func)."""
    W = W.to(DEV).float()
    out, n = W.shape
    rot = get_rot(tag, n)
    Wr = rot.fwd(W)
    Hr = rot.rot_hessian(H.to(DEV).float())
    s = Wr.pow(2).mean(1).sqrt().clamp_min(1e-6).half().float()
    U = chol_inv_upper(Hr, damp)
    idx, Q = ldlq(Wr, s, U, cfg, cbs)
    Dq = Wr - Q
    rel = ((Dq @ Hr) * Dq).sum() / ((Wr @ Hr) * Wr).sum().clamp_min(1e-12)
    qm = QMat((out, n), cfg, tag, [i.cpu() for i in idx], s.half().cpu())
    Wd = rot.inv(Q)
    return qm, Wd, rel.item()


# ----------------------------------------------------------------------------------------
# Estatisticas de entropia (ponto 5)
# ----------------------------------------------------------------------------------------
def entropy_bits(idx_list_per_stage_counts):
    """Dado {stage_key: contagens[K]} retorna bits totais ideais (Shannon)."""
    tot = 0.0
    for cnt in idx_list_per_stage_counts.values():
        c = cnt.double()
        n = c.sum()
        p = c[c > 0] / n
        tot += float(-(c[c > 0] * torch.log2(p)).sum())
    return tot
