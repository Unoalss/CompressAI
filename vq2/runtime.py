"""
vq2/runtime.py - inferencia com os pesos COMPRIMIDOS residentes na GPU/RAM (nao so no disco).

O container `.vq2` e decodificado (ANS) para os indices crus dos codebooks e estes ficam como
  * uint16 (estagio 16 bits)  -> guardado como int16 (mesmos bits)
  * uint8  (estagio 8 bits)
Cada nn.Linear vira um `PackedLinear`: a cada forward, apenas ESSA matriz e reconstruida
(lookup nos codebooks + escala por linha) em bf16 temporario, usada no matmul e descartada.
A rotacao de incoerencia e aplicada na ENTRADA (x' = R^T x), entao nunca se rotaciona o peso.
A embedding (tied com o lm_head) tambem fica comprimida: lookup so das linhas dos tokens usados
e, no lm_head, reconstrucao em blocos de linhas do vocabulario.

Memoria na GPU ~= tamanho do arquivo (~0.7 GB) + 1 matriz temporaria, em vez de ~3.6 GB (BF16).
"""
import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import *
from . import vq
from .export import ans_decode, unpack_fixed
from .kernels import HAVE_TRITON, vq_gemv_16_8, vq_gemv_16, patch_transformers_kernels

USE_TRITON = HAVE_TRITON and os.environ.get("VQ2_NO_TRITON", "0") != "1"
if USE_TRITON:
    patch_transformers_kernels()

CHUNK_ELEMS = 1 << 25      # ~33M elementos (~64 MB bf16) por bloco de reconstrucao


def _as_index(a, bits):
    """buffer comprimido -> int32 pronto para F.embedding."""
    a = a.to(torch.int32)
    return a & 0xFFFF if bits > 8 else a

class FastRot:
    """Rotacao de decodificacao (1 token): permuta + sinais + Hadamard NAO normalizada (+-1 exato em bf16).
    O fator 1/sqrt(2048) vai para o kernel. O resultado e reaproveitado entre matrizes que recebem o
    MESMO tensor de entrada (q/k/v, gate/up, in_proj_qkv/z), validado por identidade + versao."""
    _ALL = {}

    def __init__(self, rot, device):
        self.perm = rot.perm
        self.sign = rot.sign.to(torch.bfloat16)
        self.nb = rot.nb
        self.H = vq._hmat(device, torch.bfloat16)
        self._x, self._ver, self._y = None, -1, None

    @classmethod
    def get(cls, tag, n, device, rot):
        k = (tag, n, str(device))
        if k not in cls._ALL:
            cls._ALL[k] = cls(rot, device)
        return cls._ALL[k]

    def __call__(self, x):
        if self._x is x and self._ver == x._version:
            return self._y
        y = torch.index_select(x.reshape(-1), 0, self.perm) * self.sign
        y = (y.view(self.nb, vq.HB) @ self.H).reshape(-1)
        self._x, self._ver, self._y = x, x._version, y
        return y


RS = vq.HB ** -0.5



class PackedLinear(nn.Module):
    def __init__(self, shape, cfg, tag, idx, scale, cbs, device):
        super().__init__()
        self.out, self.n = shape
        self.cfg, self.tag = tuple(cfg), tag
        self.keys = [vq.cfg_key(self.cfg, i) for i in range(len(self.cfg))]
        for i, a in enumerate(idx):
            self.register_buffer(f"idx{i}", a.to(device), persistent=False)
        self.register_buffer("scale", scale.to(device=device, dtype=torch.bfloat16), persistent=False)
        self._cbs = [cbs]                      # dict compartilhado (nao registra como estado do modulo)
        self.rot = vq.get_rot(tag, self.n, device)

    def weight_rows(self, r0, r1):
        """Reconstroi as linhas [r0, r1) no espaco rotacionado (bf16)."""
        rec = None
        for i, k in enumerate(self.keys):
            ix = _as_index(getattr(self, f"idx{i}")[r0:r1], self.cfg[i])
            g = F.embedding(ix, self._cbs[0][k])                 # [rows, n/8, 8]
            rec = g if rec is None else rec + g
        return rec.reshape(r1 - r0, self.n) * self.scale[r0:r1, None]

    def forward(self, x):
        if (USE_TRITON and x.is_cuda and x.dtype == torch.bfloat16 and x.numel() == self.n):
            if self.cfg == (16, 8):
                xr = FastRot.get(self.tag, self.n, x.device, self.rot)(x)
                y = vq_gemv_16_8(xr, self.idx0, self.idx1,
                                 self._cbs[0][self.keys[0]], self._cbs[0][self.keys[1]],
                                 self.scale, self.out, self.n, RS)
                return y.view(*x.shape[:-1], self.out)
            elif self.cfg == (16,):
                xr = FastRot.get(self.tag, self.n, x.device, self.rot)(x)
                y = vq_gemv_16(xr, self.idx0,
                               self._cbs[0][self.keys[0]],
                               self.scale, self.out, self.n, RS)
                return y.view(*x.shape[:-1], self.out)
        xr = self.rot.fwd(x)
        ch = max(1, CHUNK_ELEMS // self.n)
        if self.out <= ch:
            return F.linear(xr, self.weight_rows(0, self.out).to(x.dtype))
        outs = [F.linear(xr, self.weight_rows(r, min(r + ch, self.out)).to(x.dtype)) for r in range(0, self.out, ch)]
        return torch.cat(outs, -1)

    def nbytes(self):
        return sum(b.numel() * b.element_size() for b in self.buffers())


class PackedEmbedding(nn.Module):
    """Lookup de linhas comprimidas (fp32 para reproduzir exatamente a reconstrucao original)."""

    def __init__(self, lin, cbs32):
        super().__init__()
        self.lin = lin
        self._cb32 = [cbs32]

    def forward(self, ids):
        flat = ids.reshape(-1)
        rec = None
        for i, k in enumerate(self.lin.keys):
            ix = _as_index(getattr(self.lin, f"idx{i}")[flat], self.lin.cfg[i])
            g = F.embedding(ix, self._cb32[0][k])
            rec = g if rec is None else rec + g
        q = rec.reshape(flat.numel(), self.lin.n) * self.lin.scale[flat].float()[:, None]
        w = self.lin.rot.inv(q)
        return w.to(torch.bfloat16).reshape(*ids.shape, self.lin.n)


class PackedLMHead(nn.Module):
    """lm_head tied com a embedding: compartilha os MESMOS indices comprimidos."""

    def __init__(self, lin):
        super().__init__()
        self.lin = lin

    def forward(self, x):
        return self.lin(x)


def _set(model, name, new):
    parent = model
    parts = name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


@torch.no_grad()
def decode_indices(C):
    """Container -> {nome: [tensor comprimido por estagio]} (CPU, uint8/int16, NAO int64)."""
    idx = {nm: [None] * len(C["meta"][nm]["cfg"]) for nm in C["names"]}
    for key, e in C["streams"].items():
        bits = e["bits"]
        n = int(sum(e["sizes"]))
        if e["mode"] == "ans":
            ans = e["ans"].numpy() if isinstance(e["ans"], torch.Tensor) else e["ans"]
            cnt = e["counts"].numpy() if isinstance(e["counts"], torch.Tensor) else e["counts"]
            sym = ans_decode(ans, cnt, n)
        else:
            fx = e["fixed"].numpy() if isinstance(e["fixed"], torch.Tensor) else e["fixed"]
            sym = unpack_fixed(fx, bits, n)
        assert bits <= 16
        sym = np.asarray(sym)
        arr = sym.astype(np.uint8) if bits <= 8 else sym.astype(np.uint16).view(np.int16)
        t_all = torch.from_numpy(np.ascontiguousarray(arr))
        owners = [nm for nm in C["names"]
                  if key in [vq.cfg_key(C["meta"][nm]["cfg"], i) for i in range(len(C["meta"][nm]["cfg"]))]]
        off = 0
        for nm, sz in zip(owners, e["sizes"]):
            cfg_ = C["meta"][nm]["cfg"]
            i = [vq.cfg_key(cfg_, j) for j in range(len(cfg_))].index(key)
            idx[nm][i] = t_all[off:off + sz].view(C["meta"][nm]["shape"][0], -1).clone()
            off += sz
        del sym, arr, t_all
    return idx


def read_container(path):
    """Le .vq2 (ANS, distribuicao) ou .vq2r (indices crus, carga rapida). Retorna dict pronto para load_packed."""
    C = torch.load(path, weights_only=False)
    if C.get("format") == "vq2r":
        return C
    idx = decode_indices(C)
    scales = {nm: (s if isinstance(s, torch.Tensor) else torch.from_numpy(s)) for nm, s in C["scales"].items()}
    return dict(format="vq2r", names=C["names"], meta=C["meta"], idx=idx, scales=scales,
                codebooks=C["codebooks"], codebooks_embed=C["codebooks_embed"], dense=C["dense"])


def convert_to_runtime(src, dst):
    """Decodifica a ANS uma vez e grava os indices crus (carga em segundos; ~+1-2 MB vs .vq2)."""
    R = read_container(src)
    torch.save(R, dst)
    log(f"[runtime] {dst}: {os.path.getsize(dst)/2**20:.1f} MiB")


def _build_cpu_model(cfg):
    from transformers import AutoModelForCausalLM
    try:
        from transformers.initialization import no_init_weights
    except Exception:  # pragma: no cover
        import contextlib
        no_init_weights = contextlib.nullcontext
    with no_init_weights():
        return AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).eval()   # CPU (temporario)


@torch.no_grad()
def load_packed(path, device=DEV):
    """Carrega o .vq2/.vq2r mantendo os pesos comprimidos na GPU. Retorna o modelo HF com PackedLinear."""
    from transformers import AutoConfig
    C = read_container(path)
    idx = C["idx"]
    cfg = AutoConfig.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    model = _build_cpu_model(cfg)
    for p in model.parameters():
        p.requires_grad_(False)
    sd = dict(model.named_parameters())
    for n, v in C["dense"].items():
        sd[n].data.copy_(v)
    cb_lin = {k: v.to(device=device, dtype=torch.bfloat16) for k, v in C["codebooks"].items()}
    cb_emb16 = {k: v.to(device=device, dtype=torch.bfloat16) for k, v in C["codebooks_embed"].items()}
    cb_emb32 = {k: v.to(device=device, dtype=torch.float32) for k, v in C["codebooks_embed"].items()}
    emb_lin = None
    for nm in C["names"]:
        m = C["meta"][nm]
        sc = C["scales"][nm]
        is_emb = nm.endswith("embed_tokens")
        pl = PackedLinear(m["shape"], m["cfg"], m["tag"], idx[nm], sc, cb_emb16 if is_emb else cb_lin, device)
        idx[nm] = None
        if is_emb:
            emb_lin = pl
            _set(model, nm, PackedEmbedding(pl, cb_emb32))
        else:
            _set(model, nm, pl)
    if emb_lin is not None and getattr(cfg.get_text_config(), "tie_word_embeddings", True):
        model.lm_head = PackedLMHead(emb_lin)
    model.to(device)
    torch.cuda.empty_cache() if str(device).startswith("cuda") else None
    return model


def packed_bytes(model):
    seen, tot = set(), 0
    for mod in model.modules():
        if isinstance(mod, PackedLinear) and id(mod) not in seen:
            seen.add(id(mod))
            tot += mod.nbytes()
    return tot


def report_memory(model, tag=""):
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    log(f"[mem{tag}] GPU alocada: {torch.cuda.memory_allocated()/2**20:.0f} MiB | pico: {torch.cuda.max_memory_allocated()/2**20:.0f} MiB "
        f"| pesos comprimidos: {packed_bytes(model)/2**20:.0f} MiB")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 4 and sys.argv[1] == "convert":
        convert_to_runtime(sys.argv[2], sys.argv[3])
    else:
        print("uso: python -m vq2.runtime convert entrada.vq2 saida.vq2r")
