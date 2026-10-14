"""
tools/prepare_data.py - Prepara corpora reais (wikitext-2 + codigo Python da stdlib) com split
calibracao / avaliacao disjuntos, para calibracao de Hessianas, treino KD e avaliacao honesta.

Saida: data/tokens.pt com
  calib_wiki, calib_code : tensores [N, SEQ] (calibracao: Hessianas / treino)
  eval_wiki,  eval_code  : tensores [M, SEQ] (avaliacao: nunca vista na calibracao)
"""
import os, sys, glob, hashlib, urllib.request, argparse
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data"
DEFAULT_MODEL_DIR = ROOT / "modelo"
SEQ = 512

BASE_URL = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/"
FILES = {"train": "train-00000-of-00001.parquet", "test": "test-00000-of-00001.parquet"}


def fetch_wiki(split: str, data_dir: Path):
    path = data_dir / f"wiki_{split}.parquet"
    if not path.exists():
        print(f"Baixando wikitext-2 {split}...")
        urllib.request.urlretrieve(BASE_URL + FILES[split], str(path))
    import pandas as pd
    df = pd.read_parquet(path)
    return "".join(t for t in df["text"].tolist())


def code_corpus():
    import sysconfig
    lib = sysconfig.get_paths()["stdlib"]
    files = sorted(glob.glob(os.path.join(lib, "*.py")) + glob.glob(os.path.join(lib, "*", "*.py")))
    cal, ev = [], []
    for f in files:
        if "test" in f.lower() or "idlelib" in f.lower():
            continue
        try:
            txt = open(f, encoding="utf-8").read()
        except Exception:
            continue
        if len(txt) < 3000:
            continue
        h = int(hashlib.md5(os.path.basename(f).encode()).hexdigest(), 16) % 10
        (ev if h < 2 else cal).append(txt)
    return "\n\n".join(cal), "\n\n".join(ev)


def windows(tok, text, n, seq=SEQ):
    ids = tok(text, return_tensors="pt").input_ids[0]
    total = ids.numel() // seq
    n = min(n, total)
    idx = torch.linspace(0, total - 1, n).long()
    return torch.stack([ids[i * seq:(i + 1) * seq] for i in idx])


def prepare_dataset(model_dir: Path = DEFAULT_MODEL_DIR, data_dir: Path = DEFAULT_DATA):
    data_dir.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    print(f"Carregando tokenizer a partir de: {model_dir}")
    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    
    print("Processando Wikitext-2...")
    wt_train = fetch_wiki("train", data_dir)
    wt_test = fetch_wiki("test", data_dir)
    
    print("Processando corpus Python da biblioteca padrao...")
    code_cal, code_ev = code_corpus()
    
    out = {
        "calib_wiki": windows(tok, wt_train, 96),
        "calib_code": windows(tok, code_cal, 32),
        "eval_wiki": windows(tok, wt_test, 24),
        "eval_code": windows(tok, code_ev, 16),
    }
    for k, v in out.items():
        print(f"  - {k}: shape {tuple(v.shape)}")
        
    out_path = data_dir / "tokens.pt"
    torch.save(out, out_path)
    print(f"Dataset de calibracao/avaliacao salvo com sucesso em: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara datasets de calibracao e avaliacao")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Caminho do modelo para o tokenizer")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA), help="Diretorio de saida dos dados")
    args = parser.parse_args()
    prepare_dataset(model_dir=Path(args.model_dir), data_dir=Path(args.data_dir))
