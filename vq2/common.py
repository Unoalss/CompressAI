"""
vq2/common.py - infraestrutura compartilhada (Ponto 0: harness de avaliacao honesto).

- carga do modelo (texto apenas = ponto 6) e dos dados
- separacao estrita: CALIBRACAO/KD (train_tokens) x AVALIACAO (wiki_test completo + eval_wiki/eval_code)
- metricas: PPL (wikitext-2 test completo, eval_wiki, eval_code) e KL / top1 contra o teacher BF16
  usando logits do teacher em cache (top-K + massa da cauda) para nao precisar de 2 modelos na GPU.
"""
import os, sys, math, time, json, glob
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "modelo"
DATA = ROOT / "data"
OUT = ROOT / "vq2_out"
OUT.mkdir(exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEQ = 512


def log(*a):
    print(*a, flush=True)


def get_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)


def load_model(dtype=torch.bfloat16, device=DEV):
    """Carrega SOMENTE o modelo de texto (sem vision tower / MTP) - ponto 6."""
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), torch_dtype=dtype, low_cpu_mem_usage=True)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m.to(device)


# ----------------------------------------------------------------------------------------
# Dados
# ----------------------------------------------------------------------------------------
def build_train_tokens(n_wiki=1400, n_code=700, force=False):
    """Janelas de 512 tokens para calibracao (Hessianas) e KD.
    wiki_train.parquet + codigo Python (site-packages do venv). Nada disso e usado na avaliacao."""
    path = DATA / "train_tokens.pt"
    if path.exists() and not force:
        return torch.load(path)
    import pandas as pd
    tok = get_tokenizer()
    df = pd.read_parquet(DATA / "wiki_train.parquet")
    text = "".join(df["text"].tolist())
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    nw = min(n_wiki, ids.numel() // SEQ)
    wiki = ids[: nw * SEQ].view(nw, SEQ)
    # codigo: arquivos .py de bibliotecas instaladas (determinístico: ordem alfabetica)
    import site
    roots = [p for p in site.getsitepackages() if os.path.isdir(p)]
    files = []
    for r in roots:
        for sub in ("torch", "transformers", "numpy", "pandas", "accelerate", "safetensors"):
            files += sorted(glob.glob(os.path.join(r, sub, "**", "*.py"), recursive=True))
    chunks, total = [], 0
    for f in files:
        try:
            t = open(f, "r", encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        if len(t) < 2000:
            continue
        chunks.append(t)
        total += len(t)
        if total > n_code * SEQ * 4.2:
            break
    code_ids = tok("\n\n".join(chunks), return_tensors="pt", add_special_tokens=False).input_ids[0]
    nc = min(n_code, code_ids.numel() // SEQ)
    code = code_ids[: nc * SEQ].view(nc, SEQ)
    out = {"wiki": wiki, "code": code}
    torch.save(out, path)
    log(f"[data] train_tokens: wiki={tuple(wiki.shape)} code={tuple(code.shape)}")
    return out


def mixed_train_set(n_wiki, n_code, offset_wiki=0, offset_code=0, seed=0):
    d = build_train_tokens()
    w = d["wiki"][offset_wiki: offset_wiki + n_wiki]
    c = d["code"][offset_code: offset_code + n_code]
    x = torch.cat([w, c], 0)
    g = torch.Generator().manual_seed(seed)
    return x[torch.randperm(x.shape[0], generator=g)]


def calib_tokens():
    """Calibracao dos Hessianos: tokens.pt (calib_wiki/code) + janelas extras de train_tokens."""
    t = torch.load(DATA / "tokens.pt")
    extra = mixed_train_set(64, 64, offset_wiki=0, offset_code=0, seed=1)
    return torch.cat([t["calib_wiki"], t["calib_code"], extra], 0)


def eval_sets():
    t = torch.load(DATA / "tokens.pt")
    return {"eval_wiki": t["eval_wiki"], "eval_code": t["eval_code"]}


def wikitext_test_windows(ctx=1024, max_windows=None):
    import pandas as pd
    path = DATA / f"wiki_test_ctx{ctx}.pt"
    if path.exists():
        w = torch.load(path)
    else:
        tok = get_tokenizer()
        df = pd.read_parquet(DATA / "wiki_test.parquet")
        text = "\n\n".join(df["text"].tolist())
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        n = ids.numel() // ctx
        w = ids[: n * ctx].view(n, ctx)
        torch.save(w, path)
    return w if max_windows is None else w[:max_windows]


# ----------------------------------------------------------------------------------------
# Avaliacao
# ----------------------------------------------------------------------------------------
@torch.no_grad()
def nll_tokens(model, seqs, bs=4):
    """Retorna (sum_nll, n_tokens)."""
    tot, n = 0.0, 0
    for i in range(0, seqs.shape[0], bs):
        x = seqs[i:i + bs].to(DEV)
        logits = model(input_ids=x).logits
        l = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(), x[:, 1:].reshape(-1), reduction="sum")
        tot += l.item()
        n += x[:, 1:].numel()
        del logits
    return tot, n


@torch.no_grad()
def eval_ppl(model, seqs, bs=4):
    s, n = nll_tokens(model, seqs, bs)
    return math.exp(s / n)


@torch.no_grad()
def cache_teacher(model, seqs, K=64, bs=4):
    """Guarda top-K logprobs do teacher (+ indices). A massa da cauda e implicita (1 - sum p)."""
    idxs, lps = [], []
    for i in range(0, seqs.shape[0], bs):
        x = seqs[i:i + bs].to(DEV)
        logits = model(input_ids=x).logits.float()
        lp = torch.log_softmax(logits, -1)
        v, ix = torch.topk(lp, K, dim=-1)
        idxs.append(ix.int().cpu())
        lps.append(v.half().cpu())
        del logits, lp
    return {"idx": torch.cat(idxs, 0), "lp": torch.cat(lps, 0)}


def kd_loss_topk(student_logits, t_idx, t_lp):
    """KL(teacher || student) restrito ao top-K do teacher + balde unico para a cauda.
    student_logits [B,L,V]; t_idx/t_lp [B,L,K]. Retorna (kl_medio_por_token, top1_agree)."""
    sl = student_logits.float()
    slp = torch.log_softmax(sl, -1)
    s_at = torch.gather(slp, -1, t_idx.long())
    t_lp = t_lp.float()
    pt = t_lp.exp()
    kl_top = (pt * (t_lp - s_at)).sum(-1)
    tail_t = (1.0 - pt.sum(-1)).clamp_min(1e-8)
    tail_s = (1.0 - s_at.exp().sum(-1)).clamp_min(1e-8)
    kl_tail = tail_t * (tail_t.log() - tail_s.log())
    top1 = (sl.argmax(-1) == t_idx[..., 0].long()).float()
    return (kl_top + kl_tail), top1


@torch.no_grad()
def eval_kl(model, seqs, cache, bs=4):
    kls, t1s = [], []
    for i in range(0, seqs.shape[0], bs):
        x = seqs[i:i + bs].to(DEV)
        lg = model(input_ids=x).logits
        kl, t1 = kd_loss_topk(lg, cache["idx"][i:i + bs].to(DEV), cache["lp"][i:i + bs].to(DEV))
        kls.append(kl.flatten().cpu())
        t1s.append(t1.flatten().cpu())
        del lg
    return torch.cat(kls).mean().item(), torch.cat(t1s).mean().item() * 100


class Evaluator:
    """Avaliacao completa: wikitext-2 test (inteiro), eval_wiki, eval_code, KL/top1 vs teacher."""

    def __init__(self, teacher_model=None, wiki_ctx=1024, cache_path=None):
        self.wiki = wikitext_test_windows(wiki_ctx)
        es = eval_sets()
        self.ew, self.ec = es["eval_wiki"], es["eval_code"]
        self.kl_set = torch.cat([self.ew, self.ec], 0)
        self.cache = None
        cache_path = cache_path or (OUT / "teacher_eval_cache.pt")
        if os.path.exists(cache_path):
            self.cache = torch.load(cache_path)
        elif teacher_model is not None:
            self.cache = cache_teacher(teacher_model, self.kl_set)
            torch.save(self.cache, cache_path)

    @torch.no_grad()
    def run(self, model, full=True):
        r = {}
        r["ppl_eval_wiki"] = eval_ppl(model, self.ew)
        r["ppl_eval_code"] = eval_ppl(model, self.ec)
        r["ppl_holdout_media"] = (r["ppl_eval_wiki"] + r["ppl_eval_code"]) / 2
        if full:
            r["ppl_wikitext2_test"] = eval_ppl(model, self.wiki, bs=2)
        if self.cache is not None:
            r["kl"], r["top1"] = eval_kl(model, self.kl_set, self.cache)
        return {k: round(v, 4) for k, v in r.items()}


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
