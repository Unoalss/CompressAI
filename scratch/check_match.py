import copy, sys
from pathlib import Path
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vq2.runtime import load_packed

p = ROOT / "VQ2_MARCO_FINAL" / "qwen3.5_2b_vq2_560mb.vq2r"
model = load_packed(str(p), device="cuda")
tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "modelo"), trust_remote_code=True)
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    out_hf = model.generate(**inputs, max_new_tokens=20, do_sample=False)
hf_text = tokenizer.decode(out_hf[0], skip_special_tokens=True)
print("HF text:\n", hf_text)

# Direct loop:
@torch.no_grad()
def direct_decode(m, in_ids, max_new):
    out = m(input_ids=in_ids, use_cache=True)
    past = out.past_key_values
    next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    tokens = [next_tok]
    for _ in range(max_new - 1):
        out = m(input_ids=next_tok, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens.append(next_tok)
    return torch.cat([in_ids] + tokens, dim=-1)

out_direct = direct_decode(model, inputs.input_ids, 20)
direct_text = tokenizer.decode(out_direct[0], skip_special_tokens=True)
print("Direct text:\n", direct_text)
print("HF == Direct:", (out_hf == out_direct).all().item())
