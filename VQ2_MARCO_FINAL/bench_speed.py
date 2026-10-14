import os, sys, time
from pathlib import Path
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vq2.runtime import load_packed, report_memory

def run_speed_test(model_path=None, n_warmup=10, n_gen=50):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_path is None:
        p560 = ROOT / "VQ2_MARCO_FINAL" / "qwen3.5_2b_vq2_560mb.vq2r"
        p680 = ROOT / "VQ2_MARCO_FINAL" / "qwen3.5_2b_vq2_680mb.vq2r"
        model_path = p560 if p560.exists() else p680
    model_path = Path(model_path)
    
    print("=" * 80)
    print(f"BENCHMARK DE VELOCIDADE: {model_path.name} ({model_path.stat().st_size / 2**20:.1f} MiB)")
    print("=" * 80)
    
    model = load_packed(str(model_path), device=device)
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "modelo"), trust_remote_code=True)
    
    prompt = "def fibonacci(n):\n    \"\"\"Return nth Fibonacci number\"\"\"\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    print("\n--- MODO 1: HuggingFace model.generate (com GenerationMixin overhead) ---")
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=n_warmup, do_sample=False)
    torch.cuda.synchronize()
    
    t0 = time.perf_counter()
    with torch.no_grad():
        out1 = model.generate(**inputs, max_new_tokens=n_gen, do_sample=False)
    torch.cuda.synchronize()
    dt1 = time.perf_counter() - t0
    toks1 = out1.shape[1] - inputs.input_ids.shape[1]
    speed1 = toks1 / dt1
    print(f"HF generate: {toks1} tokens em {dt1:.2f}s | Velocidade: {speed1:.2f} tok/s")
    
    print("\n--- MODO 2: Loop Direto Otimizado (sem overhead do GenerationMixin) ---")
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

    _ = direct_decode(model, inputs.input_ids, n_warmup)
    torch.cuda.synchronize()
    
    t0 = time.perf_counter()
    out2 = direct_decode(model, inputs.input_ids, n_gen)
    torch.cuda.synchronize()
    dt2 = time.perf_counter() - t0
    toks2 = out2.shape[1] - inputs.input_ids.shape[1]
    speed2 = toks2 / dt2
    print(f"Loop Direto:  {toks2} tokens em {dt2:.2f}s | Velocidade: {speed2:.2f} tok/s | Aceleração: {speed2/speed1:.2f}x")
    
    print("\n--- MODO 3: Decodificação Acelerada por CUDA Graphs (Zero CPU Overhead) ---")
    @torch.no_grad()
    def graph_decode(m, in_ids, max_new):
        out = m(input_ids=in_ids, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens = [next_tok]

        static_tok = torch.empty((1, 1), dtype=torch.long, device=device)
        static_tok.copy_(next_tok)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                out = m(input_ids=static_tok, past_key_values=past, use_cache=True)
                past = out.past_key_values
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_out = m(input_ids=static_tok, past_key_values=past, use_cache=True)
            static_next = static_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        for _ in range(max_new - 1):
            static_tok.copy_(static_next)
            g.replay()
            tokens.append(static_next.clone())
        return torch.cat([in_ids] + tokens, dim=-1)

    _ = graph_decode(model, inputs.input_ids, n_warmup)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    out3 = graph_decode(model, inputs.input_ids, n_gen)
    torch.cuda.synchronize()
    dt3 = time.perf_counter() - t0
    toks3 = out3.shape[1] - inputs.input_ids.shape[1]
    speed3 = toks3 / dt3
    print(f"CUDA Graph:   {toks3} tokens em {dt3:.2f}s | Velocidade: {speed3:.2f} tok/s | Aceleração: {speed3/speed1:.2f}x")

    match = (out1 == out3).all().item()
    print(f"\nVerificação de fidelidade de saída: {'IDÊNTICAS (100% EXATO)' if match else 'DIFERENTES'}")
    return speed1, speed2, speed3

if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else None
    run_speed_test(p)
