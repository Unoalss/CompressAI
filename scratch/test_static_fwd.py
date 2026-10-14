import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, StaticCache

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vq2.runtime import load_packed

p = ROOT / "VQ2_MARCO_FINAL" / "qwen3.5_2b_vq2_560mb.vq2r"
model = load_packed(str(p), device="cuda")
tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "modelo"), trust_remote_code=True)
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

try:
    cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=128, device="cuda", dtype=torch.bfloat16)
    out = model(input_ids=inputs.input_ids, past_key_values=cache)
    print("Prefill with StaticCache worked! Logits shape:", out.logits.shape)
except Exception as e:
    print("StaticCache forward failed:", e)
