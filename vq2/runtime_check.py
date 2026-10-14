"""Verifica o runtime comprimido: memoria real na GPU, PPL/KL/Top-1 e velocidade de geracao.
Uso: .\\.venv\\Scripts\\python.exe -m vq2.runtime_check [caminho.vq2] [--quick]"""
import sys, time
import torch
from .common import *
from .runtime import load_packed, report_memory


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    quick = "--quick" in sys.argv
    path = args[0] if args else str(ROOT / "VQ2_MARCO_FINAL" / "qwen3.5_2b_vq2_678mb.vq2")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model = load_packed(path)
    log(f"[runtime] carregado em {time.time()-t0:.0f}s")
    report_memory(model, " apos load")

    tok = get_tokenizer()
    for p in ["def fibonacci(n):\n    \"\"\"Return nth Fibonacci number\"\"\"\n", "The capital of France is"]:
        ids = tok(p, return_tensors="pt").to(DEV)
        t1 = time.time()
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=40, do_sample=False)
        dt = time.time() - t1
        n = out.shape[1] - ids.input_ids.shape[1]
        log(f"[gen] {n/dt:.1f} tok/s | {tok.decode(out[0], skip_special_tokens=True)!r}")
    report_memory(model, " apos geracao")

    ev = Evaluator()
    r = ev.run(model, full=not quick)
    log(f"[runtime] metricas: {r}")
    report_memory(model, " apos avaliacao")
    save_json(r, OUT / "runtime_check.json")


if __name__ == "__main__":
    main()
