import os, sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vq2.runtime import load_packed

def profile_one_step():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p = ROOT / "VQ2_MARCO_FINAL" / "qwen3.5_2b_vq2_560mb.vq2r"
    model = load_packed(str(p), device=device)
    
    # 1 input token with dummy past_key_values
    input_ids = torch.tensor([[100]], device=device)
    out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    
    # Warmup
    for _ in range(5):
        out = model(input_ids=next_tok, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    
    # Measure detailed CUDA events or PyTorch profiler
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(10):
            out = model(input_ids=next_tok, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize()
        
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))

if __name__ == "__main__":
    profile_one_step()
