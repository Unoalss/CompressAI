import sys, time
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

max_len = 64
cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=max_len, device="cuda", dtype=torch.bfloat16)

# Prefill
out = model(input_ids=inputs.input_ids, past_key_values=cache)
next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
tokens = [next_tok]

# Static tensors for graph
static_tok = torch.empty((1, 1), dtype=torch.long, device="cuda")
static_tok.copy_(next_tok)

# Warmup 3 steps
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        out = model(input_ids=static_tok, past_key_values=cache)
torch.cuda.current_stream().wait_stream(s)

# Capture graph
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_out = model(input_ids=static_tok, past_key_values=cache)
    static_next = static_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

# Replay
t0 = time.perf_counter()
N = 30
for _ in range(N):
    static_tok.copy_(static_next)
    g.replay()
    tokens.append(static_next.clone())
torch.cuda.synchronize()
dt = time.perf_counter() - t0

full = torch.cat([inputs.input_ids] + tokens, dim=-1)
text = tokenizer.decode(full[0], skip_special_tokens=True)
print("StaticCache + CUDA Graph text:\n", text)
print(f"Speed: {N} tokens in {dt:.3f}s -> {N/dt:.2f} tok/s!")
