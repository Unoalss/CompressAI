"""Baseline Q4_K_M medido com o MESMO harness (Ponto 0) para comparacao justa."""
import torch
from transformers import AutoModelForCausalLM
from .common import *

if __name__ == "__main__":
    m = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), gguf_file="Qwen3.5-2B-Q4_K_M.gguf", dtype=torch.bfloat16)
    m = m.to(DEV).eval()
    r = Evaluator().run(m, full=True)
    r["size_mb"] = round((MODEL_DIR / "Qwen3.5-2B-Q4_K_M.gguf").stat().st_size / 2 ** 20, 1)
    log("[Q4_K_M]", r)
    save_json(r, OUT / "q4km_baseline.json")
