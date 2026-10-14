import os, sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vq2.runtime import load_packed

def test_opt():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p = ROOT / "VQ2_MARCO_FINAL" / "qwen3.5_2b_vq2_560mb.vq2r"
    model = load_packed(str(p), device=device)
    
    # Check if torch.compile works on the model
    input_ids = torch.tensor([[100]], device=device)
    out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    
    print("Testing torch.compile on single step decode...")
    class StepWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, tok, past):
            return self.m(input_ids=tok, past_key_values=past, use_cache=True)
            
    # Test standard uncompiled speed for 50 steps
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    p_curr = past
    tok_curr = next_tok
    for _ in range(50):
        out = model(input_ids=tok_curr, past_key_values=p_curr, use_cache=True)
        p_curr = out.past_key_values
        tok_curr = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"Uncompiled direct loop: 50 tokens in {dt:.2f}s -> {50/dt:.2f} tok/s")

if __name__ == "__main__":
    test_opt()
