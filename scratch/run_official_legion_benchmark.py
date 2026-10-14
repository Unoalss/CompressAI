import sys, os, time, json, hashlib
from pathlib import Path
import torch
import torch.nn.functional as F
import psutil

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legion.engine import LegionEngine
from vq2.kernels import set_fused_kernels_enabled

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(64 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def get_ram_gb() -> float:
    return psutil.Process().memory_info().rss / (1024**3)

def get_vram_gb():
    alloc = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    res = torch.cuda.memory_reserved() / (1024**3) if torch.cuda.is_available() else 0.0
    return alloc, res

def percentile(data, p):
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    d = k - f
    return sorted_data[f] * (1.0 - d) + sorted_data[c] * d

def run_official_benchmark():
    print("=" * 80)
    print("PROTOCOLO OFICIAL DE BENCHMARK DO MOTOR LEGION (RTX 3060 12GB)")
    print("Avaliação Comparativa em Regime Estacionário & Auditoria Criptográfica")
    print("=" * 80)

    b2_path = ROOT / "vq2_35b_out" / "qwen35b_mixed_b2.vq2m"
    b2_size_init = b2_path.stat().st_size
    print(f"\n[1/6] Computando SHA-256 inicial do Artefato B2 ({b2_size_init:,} bytes)...")
    t0_hash = time.time()
    sha256_before = compute_sha256(b2_path)
    print(f"      SHA-256 Inicial: {sha256_before} ({time.time() - t0_hash:.1f}s)")
    assert b2_size_init == 10108393952, "Tamanho violado!"

    # 2. Inicialização do motor LEGION
    print("\n[2/6] Inicializando Motor Neural LEGION na RTX 3060...")
    engine = LegionEngine()
    device = "cuda"

    ram_init = get_ram_gb()
    vram_alloc_init, vram_res_init = get_vram_gb()
    print(f"      RAM do Sistema (RSS): {ram_init:.2f} GB")
    print(f"      VRAM Alocada:        {vram_alloc_init:.2f} GB ({vram_res_init:.2f} GB reservada)")

    prompt = "A inteligência artificial avançada permite arquiteturas neurais altamente eficientes com"
    prompt_ids = engine.tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()

    def get_cache_stats():
        total_hits = sum(l.hits for l in engine.layers)
        total_misses = sum(l.misses for l in engine.layers)
        total = total_hits + total_misses
        rate = (total_hits / total * 100.0) if total > 0 else 0.0
        return total_hits, total_misses, rate

    # 3. Bateria 0: Desempenho com Cache Frio (Cold Cache)
    print("\n[3/6] Executando Bateria 0 (Desempenho com Cache Frio - 30 tokens)...")
    set_fused_kernels_enabled(True)
    torch.cuda.synchronize()
    
    with torch.no_grad():
        out_cold = engine.model(input_ids=prompt_ids, use_cache=True)
        past_kv = out_cold.past_key_values
        next_tok = torch.argmax(out_cold.logits[:, -1, :], dim=-1).item()

    curr = torch.tensor([[next_tok]], device=device)
    cold_token_times = []
    cold_tokens = [next_tok]

    for _ in range(30):
        ev_s = torch.cuda.Event(enable_timing=True)
        ev_e = torch.cuda.Event(enable_timing=True)
        ev_s.record()
        with torch.no_grad():
            out = engine.model(input_ids=curr, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()
            cold_tokens.append(next_tok)
            curr = torch.tensor([[next_tok]], device=device)
        ev_e.record()
        torch.cuda.synchronize()
        cold_token_times.append(ev_s.elapsed_time(ev_e))

    cold_total_ms = sum(cold_token_times)
    cold_ms_per_tok = cold_total_ms / 30.0
    cold_tps = 30.0 / (cold_total_ms / 1000.0)
    cold_hits, cold_misses, cold_hit_rate = get_cache_stats()
    print(f"      Cache Frio: {cold_tps:.2f} tok/s ({cold_ms_per_tok:.2f} ms/token) | Hit Rate: {cold_hit_rate:.1f}% ({cold_hits} hits / {cold_misses} misses)")

    # 4. Aquecimento de Estabilização (Warmup do Cache LRU)
    print("\n[4/6] Executando Aquecimento de Estabilização (Warmup do Cache LRU)...")
    with torch.no_grad():
        out = engine.model(input_ids=prompt_ids, use_cache=True)
        past_kv = out.past_key_values
        next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()
        curr = torch.tensor([[next_tok]], device=device)
        for _ in range(35):
            out = engine.model(input_ids=curr, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()
            curr = torch.tensor([[next_tok]], device=device)
    torch.cuda.synchronize()
    print("      Cache estabilizado para regime quente!")

    # 5. Execuções em Ordem Alternada entre Versões:
    # Versão A: Baseline (Sem fusões Triton / Padrão PyTorch)
    # Versão B: Otimizada (Kernels Fundidos Triton + DeltaNet W_in + Add/RMSNorm)
    # 5 rodadas alternadas (A, B, A, B, A, B, A, B, A, B) = 5 baterias de cada!
    print("\n[5/6] Executando 10 Baterias em Ordem Alternada (5x Baseline vs 5x Otimizado)...")
    print("      Protocolo: 30 tokens por bateria | Medição exata token a token via CUDA Events")

    N_TOKENS = 30
    NUM_ROUNDS = 5

    optimized_all_samples_ms = []
    baseline_all_samples_ms = []
    optimized_summaries = []
    baseline_summaries = []

    has_nan_any = False
    top1_matches = 0
    total_comparisons = 0
    cosine_sims = []
    max_logit_diffs = []

    sample_texts_opt = []
    sample_texts_base = []

    for round_idx in range(1, NUM_ROUNDS + 1):
        # Alternância estrita: Versão Baseline depois Versão Otimizada (ou vice-versa)
        for version_mode in ["baseline", "optimized"]:
            is_opt = (version_mode == "optimized")
            set_fused_kernels_enabled(is_opt)
            ver_label = "OTIMIZADO" if is_opt else "BASELINE "
            
            prompt_ids = engine.tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
            torch.cuda.synchronize()

            hits_before = sum(l.hits for l in engine.layers)
            misses_before = sum(l.misses for l in engine.layers)

            # Prefill inicial
            with torch.no_grad():
                out = engine.model(input_ids=prompt_ids, use_cache=True)
                past_kv = out.past_key_values
                next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()

            curr = torch.tensor([[next_tok]], device=device)
            tok_list = [next_tok]

            # 3 passos de estabilização pós-prefill
            for _ in range(3):
                with torch.no_grad():
                    out = engine.model(input_ids=curr, past_key_values=past_kv, use_cache=True)
                    past_kv = out.past_key_values
                    next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()
                    tok_list.append(next_tok)
                    curr = torch.tensor([[next_tok]], device=device)
            torch.cuda.synchronize()

            # Medição precisa dos 30 tokens gerados
            batch_times = []
            final_step_logits = None
            for _ in range(N_TOKENS):
                ev_start = torch.cuda.Event(enable_timing=True)
                ev_end = torch.cuda.Event(enable_timing=True)
                ev_start.record()
                with torch.no_grad():
                    out = engine.model(input_ids=curr, past_key_values=past_kv, use_cache=True)
                    past_kv = out.past_key_values
                    l = out.logits[:, -1, :]
                    if torch.isnan(l).any() or torch.isinf(l).any():
                        has_nan_any = True
                    next_tok = torch.argmax(l, dim=-1).item()
                    tok_list.append(next_tok)
                    curr = torch.tensor([[next_tok]], device=device)
                    final_step_logits = l
                ev_end.record()
                torch.cuda.synchronize()
                t_ms = ev_start.elapsed_time(ev_end)
                batch_times.append(t_ms)

            hits_after = sum(l.hits for l in engine.layers)
            misses_after = sum(l.misses for l in engine.layers)
            delta_hits = hits_after - hits_before
            delta_misses = misses_after - misses_before
            delta_total = delta_hits + delta_misses
            b_hit_rate = (delta_hits / delta_total * 100.0) if delta_total > 0 else 100.0

            b_total_ms = sum(batch_times)
            b_avg_ms = b_total_ms / N_TOKENS
            b_tps = N_TOKENS / (b_total_ms / 1000.0)
            b_p50 = percentile(batch_times, 50)
            b_min = min(batch_times)
            b_max = max(batch_times)

            gen_text = engine.tokenizer.decode(tok_list)

            summary_item = {
                "round": round_idx,
                "version": version_mode,
                "tps": b_tps,
                "avg_ms": b_avg_ms,
                "p50_ms": b_p50,
                "min_ms": b_min,
                "max_ms": b_max,
                "hit_rate": b_hit_rate
            }

            if is_opt:
                optimized_all_samples_ms.extend(batch_times)
                optimized_summaries.append(summary_item)
                sample_texts_opt.append(gen_text)
                last_opt_logits = final_step_logits.clone()
            else:
                baseline_all_samples_ms.extend(batch_times)
                baseline_summaries.append(summary_item)
                sample_texts_base.append(gen_text)
                last_base_logits = final_step_logits.clone()

            print(f"  [Rodada {round_idx} | {ver_label}] {b_tps:.2f} tok/s | P50: {b_p50:.2f} ms | Média: {b_avg_ms:.2f} ms | Min: {b_min:.2f} ms | Max: {b_max:.2f} ms | Hit Rate: {b_hit_rate:.1f}%")

        # Comparação cruzada de logits e top-1 no final de cada rodada
        total_comparisons += 1
        t1_base = torch.argmax(last_base_logits, dim=-1).item()
        t1_opt = torch.argmax(last_opt_logits, dim=-1).item()
        if t1_base == t1_opt:
            top1_matches += 1
        cos = F.cosine_similarity(last_base_logits.float(), last_opt_logits.float()).item()
        diff = torch.max(torch.abs(last_base_logits.float() - last_opt_logits.float())).item()
        cosine_sims.append(cos)
        max_logit_diffs.append(diff)

    # 6. Verificação de Integridade Criptográfica do B2 pós-testes
    print("\n[6/6] Computando SHA-256 pós-testes do Artefato B2...")
    t0_hash_post = time.time()
    sha256_after = compute_sha256(b2_path)
    b2_size_final = b2_path.stat().st_size
    print(f"      SHA-256 Final:   {sha256_after} ({time.time() - t0_hash_post:.1f}s)")
    assert sha256_before == sha256_after, f"ERRO CRÍTICO: SHA-256 alterado ({sha256_before} != {sha256_after})!"
    assert b2_size_final == 10108393952, "Tamanho violado!"
    print("      [CONFIRMADO] Integridade criptográfica 100% INTACTA byte a byte!")

    # Cálculo dos percentis consolidados para a Versão Otimizada (150 amostras)
    opt_latencies = sorted(optimized_all_samples_ms)
    opt_mean = sum(opt_latencies) / len(opt_latencies)
    opt_p50 = percentile(opt_latencies, 50)
    opt_p90 = percentile(opt_latencies, 90)
    opt_p95 = percentile(opt_latencies, 95)
    opt_p99 = percentile(opt_latencies, 99)
    opt_min = min(opt_latencies)
    opt_max = max(opt_latencies)
    opt_sustained_tps = 1000.0 / opt_mean
    opt_median_tps = 1000.0 / opt_p50
    opt_peak_tps = 1000.0 / opt_min

    # Cálculo dos percentis consolidados para a Versão Baseline (150 amostras)
    base_latencies = sorted(baseline_all_samples_ms)
    base_mean = sum(base_latencies) / len(base_latencies)
    base_p50 = percentile(base_latencies, 50)
    base_p90 = percentile(base_latencies, 90)
    base_p95 = percentile(base_latencies, 95)
    base_p99 = percentile(base_latencies, 99)
    base_min = min(base_latencies)
    base_max = max(base_latencies)
    base_sustained_tps = 1000.0 / base_mean
    base_median_tps = 1000.0 / base_p50
    base_peak_tps = 1000.0 / base_min

    # Métricas de Recursos
    vram_alloc_final, vram_res_final = get_vram_gb()
    ram_final = get_ram_gb()

    # Estatísticas de fidelidade
    top1_agreement_pct = (top1_matches / total_comparisons) * 100.0
    mean_cosine_sim = sum(cosine_sims) / len(cosine_sims)
    max_observed_diff = max(max_logit_diffs)
    opt_deterministic = all(t == sample_texts_opt[0] for t in sample_texts_opt)
    base_deterministic = all(t == sample_texts_base[0] for t in sample_texts_base)
    text_cross_match = (sample_texts_opt[0] == sample_texts_base[0])

    report = {
        "metadata": {
            "device": torch.cuda.get_device_name(0),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "artifact": str(b2_path.name),
            "size_bytes": b2_size_final,
            "sha256_before": sha256_before,
            "sha256_after": sha256_after,
            "sha256_intact": (sha256_before == sha256_after)
        },
        "resources": {
            "vram_allocated_gb": vram_alloc_final,
            "vram_reserved_gb": vram_res_final,
            "ram_rss_gb": ram_final,
        },
        "cold_cache": {
            "tps": cold_tps,
            "ms_per_token": cold_ms_per_tok,
            "hit_rate_pct": cold_hit_rate,
        },
        "optimized_version": {
            "n_samples": len(opt_latencies),
            "percentiles_ms": {
                "min": opt_min,
                "p50_median": opt_p50,
                "mean": opt_mean,
                "p90": opt_p90,
                "p95": opt_p95,
                "p99": opt_p99,
                "max": opt_max,
            },
            "throughput_tokens_per_sec": {
                "peak_instant": opt_peak_tps,
                "median_p50": opt_median_tps,
                "sustained_mean": opt_sustained_tps,
            },
            "batteries": optimized_summaries
        },
        "baseline_version": {
            "n_samples": len(base_latencies),
            "percentiles_ms": {
                "min": base_min,
                "p50_median": base_p50,
                "mean": base_mean,
                "p90": base_p90,
                "p95": base_p95,
                "p99": base_p99,
                "max": base_max,
            },
            "throughput_tokens_per_sec": {
                "peak_instant": base_peak_tps,
                "median_p50": base_median_tps,
                "sustained_mean": base_sustained_tps,
            },
            "batteries": baseline_summaries
        },
        "comparison_and_fidelity": {
            "has_nan_or_inf": has_nan_any,
            "top1_agreement_pct": top1_agreement_pct,
            "mean_cosine_similarity": mean_cosine_sim,
            "max_logit_abs_diff": max_observed_diff,
            "opt_internal_deterministic": opt_deterministic,
            "base_internal_deterministic": base_deterministic,
            "cross_version_text_identical": text_cross_match,
            "speedup_median_pct": ((opt_median_tps - base_median_tps) / base_median_tps) * 100.0,
            "speedup_mean_pct": ((opt_sustained_tps - base_sustained_tps) / base_sustained_tps) * 100.0,
            "sample_generated_text": sample_texts_opt[0][:150]
        }
    }

    report_json_path = ROOT / "scratch" / "official_benchmark_results.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print("QUADRO OFICIAL CONSOLIDADO DO BENCHMARK LEGION")
    print("=" * 80)
    print(f"Placa de Vídeo:                     {torch.cuda.get_device_name(0)} (12 GB GDDR6)")
    print(f"Artefato B2:                        {b2_path.name} ({b2_size_final:,} bytes)")
    print(f"SHA-256 Verificado:                 {sha256_after}")
    print(f"Integridade Byte a Byte:            100% INTACTO ({sha256_before == sha256_after})")
    print("--------------------------------------------------------------------------------")
    print(f"VRAM Alocada / Reservada:           {vram_alloc_final:.2f} GB / {vram_res_final:.2f} GB")
    print(f"RAM do Sistema (RSS):               {ram_final:.2f} GB")
    print(f"Estabilidade Numérica:              Ausência de NaNs/Infs: {not has_nan_any}")
    print(f"Concordância Top-1 (Argmax):        {top1_agreement_pct:.1f}%")
    print(f"Similaridade de Cosseno de Logits:  {mean_cosine_sim:.6f}")
    print(f"Desvio Máximo Absoluto de Logits:   {max_observed_diff:.6f}")
    print(f"Geração Textual Idêntica:           {text_cross_match}")
    print("--------------------------------------------------------------------------------")
    print(f"Desempenho com Cache Frio (Cold):   {cold_tps:.2f} tok/s ({cold_ms_per_tok:.2f} ms/token | Hit Rate: {cold_hit_rate:.1f}%)")
    print("--------------------------------------------------------------------------------")
    print(f"{'MÉTRICA':<28} | {'BASELINE (PYTORCH)':<22} | {'OTIMIZADO (TRITON)':<22} | {'GANHO':<10}")
    print("-" * 80)
    print(f"{'Pico Instantâneo (Mínimo)':<28} | {base_peak_tps:5.2f} tok/s ({base_min:6.2f} ms) | {opt_peak_tps:5.2f} tok/s ({opt_min:6.2f} ms) | +{(opt_peak_tps - base_peak_tps)/base_peak_tps*100:5.1f}%")
    print(f"{'Mediana (P50)':<28} | {base_median_tps:5.2f} tok/s ({base_p50:6.2f} ms) | {opt_median_tps:5.2f} tok/s ({opt_p50:6.2f} ms) | +{(opt_median_tps - base_median_tps)/base_median_tps*100:5.1f}%")
    print(f"{'Média Sustentada':<28} | {base_sustained_tps:5.2f} tok/s ({base_mean:6.2f} ms) | {opt_sustained_tps:5.2f} tok/s ({opt_mean:6.2f} ms) | +{(opt_sustained_tps - base_sustained_tps)/base_sustained_tps*100:5.1f}%")
    print(f"{'Percentil 90 (P90)':<28} | {1000.0/base_p90:5.2f} tok/s ({base_p90:6.2f} ms) | {1000.0/opt_p90:5.2f} tok/s ({opt_p90:6.2f} ms) | +{(base_p90 - opt_p90)/base_p90*100:5.1f}%")
    print(f"{'Percentil 95 (P95)':<28} | {1000.0/base_p95:5.2f} tok/s ({base_p95:6.2f} ms) | {1000.0/opt_p95:5.2f} tok/s ({opt_p95:6.2f} ms) | +{(base_p95 - opt_p95)/base_p95*100:5.1f}%")
    print(f"{'Percentil 99 (P99)':<28} | {1000.0/base_p99:5.2f} tok/s ({base_p99:6.2f} ms) | {1000.0/opt_p99:5.2f} tok/s ({opt_p99:6.2f} ms) | +{(base_p99 - opt_p99)/base_p99*100:5.1f}%")
    print(f"{'Pior Caso (Máximo)':<28} | {1000.0/base_max:5.2f} tok/s ({base_max:6.2f} ms) | {1000.0/opt_max:5.2f} tok/s ({opt_max:6.2f} ms) | +{(base_max - opt_max)/base_max*100:5.1f}%")
    print("=" * 80)
    print(f"Texto de Amostra Gerado:\n\"{sample_texts_opt[0][:200]}...\"")
    print("=" * 80)

if __name__ == "__main__":
    run_official_benchmark()
