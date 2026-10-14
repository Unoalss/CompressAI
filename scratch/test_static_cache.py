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

try:
    cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=512, device="cuda", dtype=torch.bfloat16)
    print("StaticCache initialized successfully!")
except Exception as e:
    print("StaticCache failed:", e)
