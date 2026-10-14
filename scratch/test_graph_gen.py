import sys, time
from pathlib import Path
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vq2.runtime import load_packed

class CUDAGraphGenerator:
    def __init__(self, model, max_seq_len=2048):
        self.model = model
        self.device = next(model.parameters()).device
        self.tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "modelo"), trust_remote_code=True)
        self.graph = None
        self.static_input_ids = None
        self.static_past = None
        self.static_out = None

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=50):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        prompt_len = input_ids.shape[1]
        
        # 1. Prefill
        out = self.model(input_ids=input_ids, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens = [next_tok]

        # 2. Setup or use CUDA Graph for 1-token decode
        static_tok = torch.empty((1, 1), dtype=torch.long, device=self.device)
        static_tok.copy_(next_tok)
        
        # Warmup stream
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                out = self.model(input_ids=static_tok, past_key_values=past, use_cache=True)
                past = out.past_key_values
        torch.cuda.current_stream().wait_stream(s)

        # Graph Capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_out = self.model(input_ids=static_tok, past_key_values=past, use_cache=True)
            static_logits = static_out.logits
            static_next = static_logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # 3. Decode loop using CUDA Graph!
        for _ in range(max_new_tokens - 1):
            static_tok.copy_(static_next)
            g.replay()
            tokens.append(static_next.clone())

        full_ids = torch.cat([input_ids] + tokens, dim=-1)
        return self.tokenizer.decode(full_ids[0], skip_special_tokens=True), len(tokens)

def test_generation():
    p = ROOT / "VQ2_MARCO_FINAL" / "qwen3.5_2b_vq2_560mb.vq2r"
    model = load_packed(str(p), device="cuda")
    gen = CUDAGraphGenerator(model)
    
    prompt = "def fibonacci(n):\n    \"\"\"Return nth Fibonacci number\"\"\"\n"
    print(f"Prompt:\n{prompt}")
    
    # Warmup
    _ = gen.generate(prompt, max_new_tokens=10)
    torch.cuda.synchronize()
    
    t0 = time.perf_counter()
    text, n_tok = gen.generate(prompt, max_new_tokens=50)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    
    print("\nGerado:")
    print(text)
    print(f"\nVelocidade: {n_tok} tokens em {dt:.3f}s -> {n_tok/dt:.2f} tok/s!")

if __name__ == "__main__":
    test_generation()
