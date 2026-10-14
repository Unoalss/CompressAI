import sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vq2.runtime import load_packed

p = ROOT / "VQ2_MARCO_FINAL" / "qwen3.5_2b_vq2_560mb.vq2r"
model = load_packed(str(p), device="cuda")

# Prefill
tok = torch.tensor([[100]], device="cuda")
out = model(input_ids=tok, use_cache=True)
past = out.past_key_values
next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

# Warmup 3 steps
for _ in range(3):
    out = model(input_ids=next_tok, past_key_values=past, use_cache=True)
    past = out.past_key_values
    next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

print("Attempting CUDA Graph capture...")
try:
    static_tok = next_tok.clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            out = model(input_ids=static_tok, past_key_values=past, use_cache=True)
            past = out.past_key_values
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_out = model(input_ids=static_tok, past_key_values=past, use_cache=True)
    print("CUDA Graph captured successfully!")
    
    # Benchmark graph replay!
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    N = 100
    for _ in range(N):
        g.replay()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"CUDA Graph replay: {N} tokens in {dt:.3f}s -> {N/dt:.2f} tok/s!")
except Exception as e:
    print("CUDA Graph capture failed:", e)
