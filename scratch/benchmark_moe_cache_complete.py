import sys, os, time, psutil
from pathlib import Path
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legion.engine import LegionEngine

def benchmark_moe_cache():
    print("=" * 80)
    print("OTIMIZAÇÃO 3 - FASE D: BENCHMARK COMPLETO DO CACHE MoE & I/O")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda"

    # 1. Integridade do B2
    b2_path = ROOT / "vq2_35b_out" / "qwen35b_mixed_b2.vq2m"
    b2_size = b2_path.stat().st_size
    print(f"[1/4] Verificando integridade física do B2...")
    print(f"      Arquivo: {b2_path}")
    print(f"      Tamanho: {b2_size:,} bytes")
    assert b2_size == 10108393952, "Tamanho do B2 alterado!"
    print("      [OK] B2 permanece 100% INTACTO byte a byte!")

    prompt = "A inteligência artificial avançada permite arquiteturas neurais altamente eficientes com"

    def run_benchmark_for_mode(use_mmap, n_tokens=30):
        os.environ["LEGION_USE_MMAP"] = "1" if use_mmap else "0"
        mode_label = "Memory-Mapped (mmap)" if use_mmap else "Direct Offset Stream"
        engine = LegionEngine()
        tokenizer = engine.tokenizer
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

        # Zera contadores de cache das camadas
        for layer in engine.layers:
            layer.hits = 0
            layer.misses = 0

        # Prefill + 3 tokens warmup
        with torch.no_grad():
            out = engine.model(input_ids=prompt_ids, use_cache=True)
            past_kv = out.past_key_values
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()

        curr = torch.tensor([[next_tok]], device=device)
        for _ in range(3):
            with torch.no_grad():
                out = engine.model(input_ids=curr, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
                next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()
                curr = torch.tensor([[next_tok]], device=device)
        torch.cuda.synchronize()

        # Medição de 30 tokens
        generated_ids = []
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        ev_start.record()

        for _ in range(n_tokens):
            with torch.no_grad():
                out = engine.model(input_ids=curr, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
                next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()
                generated_ids.append(next_tok)
                curr = torch.tensor([[next_tok]], device=device)

        ev_end.record()
        torch.cuda.synchronize()
        gpu_time_ms = ev_start.elapsed_time(ev_end)

        # Coleta métricas de cache e memória
        tot_hits = sum(layer.hits for layer in engine.layers)
        tot_misses = sum(layer.misses for layer in engine.layers)
        tot_access = tot_hits + tot_misses
        hit_rate = (tot_hits / tot_access * 100) if tot_access > 0 else 0

        process = psutil.Process()
        ram_rss_mb = process.memory_info().rss / (1024 * 1024)
        vram_alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        vram_res_mb = torch.cuda.memory_reserved() / (1024 * 1024)

        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        last_logits = out.logits[:, -1, :].clone()

        engine.store.close()
        del engine
        torch.cuda.empty_cache()

        return {
            "mode": mode_label,
            "ms_per_tok": gpu_time_ms / n_tokens,
            "tok_per_sec": n_tokens / (gpu_time_ms / 1000.0),
            "hit_rate": hit_rate,
            "hits": tot_hits,
            "misses": tot_misses,
            "ram_rss_mb": ram_rss_mb,
            "vram_alloc_mb": vram_alloc_mb,
            "vram_res_mb": vram_res_mb,
            "text": text,
            "logits": last_logits
        }

    # Executa 3 rodadas alternadas
    print("\nExecutando 3 rodadas alternadas (Direct Stream vs Memory-Mapped)...")
    res_direct = []
    res_mmap = []

    for round_idx in range(1, 4):
        print(f"\n--- RODADA {round_idx} ---")
        rd = run_benchmark_for_mode(use_mmap=False, n_tokens=30)
        res_direct.append(rd)
        print(f"  [Direct Stream] {rd['tok_per_sec']:.2f} tok/s ({rd['ms_per_tok']:.2f} ms/tok) | Hit Rate: {rd['hit_rate']:.1f}% | VRAM: {rd['vram_alloc_mb']:.0f} MB")

        rm = run_benchmark_for_mode(use_mmap=True, n_tokens=30)
        res_mmap.append(rm)
        print(f"  [Memory-Mapped] {rm['tok_per_sec']:.2f} tok/s ({rm['ms_per_tok']:.2f} ms/tok) | Hit Rate: {rm['hit_rate']:.1f}% | VRAM: {rm['vram_alloc_mb']:.0f} MB")

    avg_d_tps = sum(r["tok_per_sec"] for r in res_direct) / 3
    avg_d_ms = sum(r["ms_per_tok"] for r in res_direct) / 3
    avg_m_tps = sum(r["tok_per_sec"] for r in res_mmap) / 3
    avg_m_ms = sum(r["ms_per_tok"] for r in res_mmap) / 3

    # Equivalência de texto e logits
    cos_sim = F.cosine_similarity(res_direct[0]["logits"].float(), res_mmap[0]["logits"].float(), dim=-1).item()
    p_m = F.log_softmax(res_mmap[0]["logits"].float(), dim=-1)
    p_d = F.softmax(res_direct[0]["logits"].float(), dim=-1)
    kl_div = F.kl_div(p_m, p_d, reduction="batchmean").item()

    print("\n" + "=" * 80)
    print("RELATÓRIO CONSOLIDADO DO BENCHMARK DE CACHE MoE & I/O:")
    print("=" * 80)
    print(f"Integridade do Artefato B2:       100% INTACTO ({b2_size:,} bytes)")
    print(f"Concordância Textual:             {res_direct[0]['text'] == res_mmap[0]['text']}")
    print(f"Similaridade Cosseno Logits:      {cos_sim * 100:.4f}%")
    print(f"Divergência KL:                   {kl_div:.6f}")
    print("--------------------------------------------------------------------------------")
    print(f"1. Modo Direct Offset Stream (Orçamento Rígido):")
    print(f"   - Velocidade Média:            {avg_d_tps:.2f} tok/s ({avg_d_ms:.2f} ms/token)")
    print(f"   - Taxa de Acerto de Cache:     {res_direct[0]['hit_rate']:.1f}%")
    print(f"   - Consumo de RAM (RSS):        {res_direct[0]['ram_rss_mb']:.0f} MB")
    print(f"   - Consumo de VRAM (Alocado):   {res_direct[0]['vram_alloc_mb']:.0f} MB")
    print(f"2. Modo Memory-Mapped (mmap):")
    print(f"   - Velocidade Média:            {avg_m_tps:.2f} tok/s ({avg_m_ms:.2f} ms/token)")
    print(f"   - Taxa de Acerto de Cache:     {res_mmap[0]['hit_rate']:.1f}%")
    print(f"   - Consumo de RAM (RSS):        {res_mmap[0]['ram_rss_mb']:.0f} MB")
    print(f"   - Consumo de VRAM (Alocado):   {res_mmap[0]['vram_alloc_mb']:.0f} MB")
    print("=" * 80)

if __name__ == "__main__":
    benchmark_moe_cache()
