import sys, os
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legion.engine import LegionEngine
from scratch.test_teacher_forcing import make_fa_forward

engine = LegionEngine()
full_layers = [l.self_attn for l in engine.model.model.layers if hasattr(l, "self_attn")]

prompts = [
    "A capital da França é",
    "O maior planeta do sistema solar é",
    "Em ciência da computação, uma árvore binária é"
]

def generate(prompt: str, use_fused: bool, max_tokens: int = 20):
    full_layers[0].__class__.forward = make_fa_forward(use_fused)
    torch.cuda.synchronize()
    
    prompt_ids = engine.tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
    with torch.no_grad():
        out = engine.model(input_ids=prompt_ids, use_cache=True)
        past_kv = out.past_key_values
        next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()
        
    tokens = [next_tok]
    curr = torch.tensor([[next_tok]], device="cuda")
    
    for _ in range(max_tokens - 1):
        with torch.no_grad():
            out = engine.model(input_ids=curr, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()
            tokens.append(next_tok)
            curr = torch.tensor([[next_tok]], device="cuda")
            
    return engine.tokenizer.decode(tokens)

for p in prompts:
    txt_pt = generate(p, use_fused=False, max_tokens=15)
    txt_tri = generate(p, use_fused=True, max_tokens=15)
    same = (txt_pt == txt_tri)
    print("=" * 60)
    print(f"PROMPT: '{p}'")
    print(f"IDÊNTICO: {same}")
    print(f"  PyTorch: {txt_pt}")
    print(f"  Triton:  {txt_tri}")
