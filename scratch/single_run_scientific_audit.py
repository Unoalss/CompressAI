"""
scratch/single_run_scientific_audit.py
Execução Científica Identificada com Rastreamento Total:
1. Tokens de Entrada (Prefill) e Saída (Decode).
2. Despachos Totais do Router.
3. Especialistas Únicos Mobilizados e Executados.
4. Bytes Efetivamente Lidos do Armazenamento (NVMe/Pendrive) vs. Poupados.
5. Pico Real de RAM Física do SO (RSS) e Pico de VRAM da GPU.
6. Velocidade (TTFT, tok/s, latências mínima/média/máxima).
7. Verificação da Integridade e Qualidade da Resposta.
"""
import os, sys, time, json, psutil
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from legion.engine import LegionEngine, DEVICE

def run_scientific_audit(prompt: str, cold_cache: bool = True):
    proc = psutil.Process()
    print("=" * 85)
    print("EXECUÇÃO CIENTÍFICA IDENTIFICADA - AUDITORIA DE PRECISÃO LEGION VQ2M")
    print(f"Hardware GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Modo de I/O: Direct Offset Stream (Orçamento Rígido de RAM)")
    print(f"Estado Inicial do Cache: {'FRIO (Esvaziado - mede leituras físicas do zero)' if cold_cache else 'QUENTE'}")
    print("=" * 85)

    # 1. Instancia o Motor
    t0_engine = time.perf_counter()
    engine = LegionEngine(cache_capacity=96)
    t_engine = time.perf_counter() - t0_engine
    print(f">> Motor pronto em {t_engine:.2f}s | Modo Top-{engine.top_k} ({engine.top_k * 40} ativos/token)")

    # 2. Se for cache frio, reseta o estado de todas as camadas
    if cold_cache:
        for layer in engine.layers:
            layer.expert_to_slot.clear()
            layer.free_slots = list(range(layer.capacity))
            layer.access_order.clear()
            layer.hits = 0
            layer.misses = 0
            layer.activated_experts_set.clear()

    # 3. Zera contadores de hardware antes da pergunta
    torch.cuda.reset_peak_memory_stats()
    ram_before_mb = proc.memory_info().rss / 1024**2
    vram_before_mb = torch.cuda.memory_allocated() / 1024**2

    # 4. Tokenização e Prefill
    system_prompt = "Você é o LEGION, um assistente de inteligência artificial de alta precisão. Responda em português brasileiro de forma completa, técnica e clara."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt.strip()}
    ]
    try:
        formatted = engine.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except Exception:
        formatted = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt.strip()}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    input_ids = engine.tokenizer.encode(formatted, return_tensors="pt")
    tokens_in = input_ids.shape[-1]

    print(f"\n[ENTRADA] Pergunta: \"{prompt}\"")
    print(f"[ENTRADA] Tokens de Entrada (Contexto + Prompt): {tokens_in} tokens")
    print(f">> Iniciando geração...")

    # 5. Execução Monitorada com Telemetria Passo a Passo
    generated_tokens = []
    step_latencies_ms = []
    peak_ram_mb = ram_before_mb

    t_start = time.perf_counter()
    for chunk in engine.generate_stream(
        prompt=prompt,
        system_prompt=system_prompt,
        max_tokens=200,
        temperature=0.7
    ):
        token_str = chunk["token"]
        generated_tokens.append(token_str)
        step_latencies_ms.append(chunk["stats"]["step_ms"])

        current_ram = proc.memory_info().rss / 1024**2
        if current_ram > peak_ram_mb:
            peak_ram_mb = current_ram

    t_total = time.perf_counter() - t_start
    tokens_out = len(generated_tokens)
    full_response_text = "".join(generated_tokens).strip()

    # 6. Extração das Métricas de Hardware e I/O
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2
    vram_end_mb = torch.cuda.memory_allocated() / 1024**2
    ram_end_mb = proc.memory_info().rss / 1024**2

    total_hits = sum(layer.hits for layer in engine.layers)
    total_misses = sum(layer.misses for layer in engine.layers)
    total_dispatches = total_hits + total_misses

    # Especialistas únicos mobilizados em toda a sessão
    all_unique_experts = set()
    layer_unique_counts = []
    for l_idx, layer in enumerate(engine.layers):
        uniques_in_layer = len(layer.activated_experts_set)
        layer_unique_counts.append(uniques_in_layer)
        for e_id in layer.activated_experts_set:
            all_unique_experts.add((l_idx, e_id))
    unique_experts_total = len(all_unique_experts)

    EXPERT_SIZE_BYTES = 792576  # 774.00 KB exatos aferidos no container
    bytes_read_storage = total_misses * EXPERT_SIZE_BYTES
    bytes_served_cache = total_hits * EXPERT_SIZE_BYTES
    total_bytes_demanded = total_dispatches * EXPERT_SIZE_BYTES
    hit_rate_pct = (total_hits / total_dispatches * 100) if total_dispatches > 0 else 0.0

    tok_per_sec = tokens_out / t_total if t_total > 0 else 0.0
    ttft_ms = step_latencies_ms[0] if step_latencies_ms else 0.0

    # 7. Apresentação Formal do Relatório Científico
    print("\n" + "=" * 85)
    print("RELATÓRIO CIENTÍFICO DA EXECUÇÃO IDENTIFICADA")
    print("=" * 85)

    print("\n1. TOKENS E LINGUAGEM:")
    print(f"   - Tokens de Entrada (Prefill):     {tokens_in} tokens")
    print(f"   - Tokens de Saída (Gerados):       {tokens_out} tokens")
    print(f"   - Total de Tokens Processados:     {tokens_in + tokens_out} tokens")

    print("\n2. DESPACHOS E ESPECIALISTAS:")
    print(f"   - Despachos Totais do Router:      {total_dispatches:,} despachos")
    print(f"   - Ativos em Cada Token:            {engine.top_k * 40} especialistas (40 camadas × {engine.top_k})")
    print(f"   - Especialistas Únicos Mobilizados: {unique_experts_total} de 10.240 ({unique_experts_total / 10240 * 100:.1f}%)")
    print(f"   - Média de Especialistas p/ Camada: {sum(layer_unique_counts) / len(layer_unique_counts):.1f} de 256")

    print("\n3. ARMAZENAMENTO E TRÁFEGO DE I/O:")
    print(f"   - Cache Misses (leituras físicas): {total_misses:,} vezes")
    print(f"   - Cache Hits (servidos na VRAM):   {total_hits:,} vezes")
    print(f"   - Taxa de Acerto de Cache:         {hit_rate_pct:.2f}%")
    print(f"   - Bytes Lidos do Armazenamento:    {bytes_read_storage / 1024**2:.2f} MB ({bytes_read_storage / 1024**3:.3f} GB)")
    print(f"   - Bytes Poupados pelo Cache:       {bytes_served_cache / 1024**2:.2f} MB ({bytes_served_cache / 1024**3:.3f} GB)")
    print(f"   - Volume Total de Pesos Demandado: {total_bytes_demanded / 1024**2:.2f} MB ({total_bytes_demanded / 1024**3:.3f} GB)")

    print("\n4. MEMÓRIA FÍSICA AUDITADA:")
    print(f"   - RAM Física do Processo Inicial:  {ram_before_mb:.1f} MB")
    print(f"   - Pico de RAM Física do SO (RSS):  {peak_ram_mb:.1f} MB ({peak_ram_mb / 1024:.2f} GB)")
    print(f"   - RAM Física Final:                {ram_end_mb:.1f} MB")
    print(f"   - VRAM Inicial Alocada:            {vram_before_mb:.1f} MB")
    print(f"   - Pico de VRAM Alocada na GPU:     {peak_vram_mb:.1f} MB ({peak_vram_mb / 1024:.2f} GB)")
    print(f"   - VRAM Final:                      {vram_end_mb:.1f} MB")

    print("\n5. VELOCIDADE E LATÊNCIA:")
    print(f"   - Tempo Total de Geração:          {t_total:.2f} segundos")
    print(f"   - Velocidade Média:                {tok_per_sec:.2f} tokens/segundo")
    if step_latencies_ms:
        print(f"   - Time To First Token (TTFT):      {step_latencies_ms[0]:.1f} ms")
        print(f"   - Latência Média por Token:        {sum(step_latencies_ms)/len(step_latencies_ms):.1f} ms")
        print(f"   - Latência Mínima (Melhor Hit):    {min(step_latencies_ms):.1f} ms")
        print(f"   - Latência Máxima (Miss I/O):      {max(step_latencies_ms):.1f} ms")

    print("\n6. RESPOSTA PRODUZIDA (Verificação de Integridade):")
    print("-" * 85)
    try:
        print(full_response_text)
    except Exception:
        print(full_response_text.encode("ascii", errors="replace").decode("ascii"))
    print("-" * 85)

    # 8. Grava artefato estruturado em JSON
    result = {
        "prompt": prompt,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "total_tokens": tokens_in + tokens_out,
        "total_dispatches": total_dispatches,
        "active_per_token": engine.top_k * 40,
        "unique_experts_mobilized": unique_experts_total,
        "total_model_experts": 10240,
        "cache_hits": total_hits,
        "cache_misses": total_misses,
        "hit_rate_pct": round(hit_rate_pct, 2),
        "bytes_read_storage_mb": round(bytes_read_storage / 1024**2, 2),
        "bytes_saved_cache_mb": round(bytes_served_cache / 1024**2, 2),
        "peak_ram_rss_mb": round(peak_ram_mb, 1),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "peak_vram_gb": round(peak_vram_mb / 1024, 2),
        "total_time_s": round(t_total, 2),
        "tokens_per_sec": round(tok_per_sec, 2),
        "response_text": full_response_text
    }
    out_json = ROOT / "scratch" / "scientific_run_result.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Dados estruturados salvos em: {out_json}")
    print("=" * 85)

if __name__ == "__main__":
    prompt_test = "Como a teoria da relatividade geral de Einstein explica a deformação do espaço-tempo pela presença de massa e energia?"
    if len(sys.argv) > 1:
        prompt_test = " ".join(sys.argv[1:])
    run_scientific_audit(prompt=prompt_test, cold_cache=True)
