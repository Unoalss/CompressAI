import sys, time
sys.path.insert(0, r"e:\compressAI")
from pathlib import Path
import torch
import torch.nn.functional as F

from vq2.runtime import load_packed
from transformers import AutoTokenizer

ROOT = Path(r"e:\compressAI")
MODEL_2B_PATH = ROOT / "VQ2_MARCO_FINAL" / "qwen3.5_2b_vq2_560mb.vq2r"

print("Testando carregamento do modelo 2B...")
device_2b = "cpu"
t0 = time.time()
model_2b = load_packed(str(MODEL_2B_PATH), device=device_2b)
print(f"Modelo 2B carregado na {device_2b.upper()} em {time.time()-t0:.2f}s!")

tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "modelo"), trust_remote_code=True)
prompt = "O Brasil é um país localizado na América do Sul. Sua capital é"
inputs = tokenizer(prompt, return_tensors="pt").to(device_2b)

# Testa velocidade de geração do modelo 2B na CPU
with torch.no_grad():
    t_start = time.perf_counter()
    out = model_2b.generate(**inputs, max_new_tokens=10, do_sample=False)
    dt = time.perf_counter() - t_start
    toks = out.shape[1] - inputs.input_ids.shape[1]
    print(f"2B na CPU: {toks} tokens em {dt:.3f}s -> {toks/dt:.1f} tok/s")
    print("Texto gerado:", tokenizer.decode(out[0], skip_special_tokens=True))
