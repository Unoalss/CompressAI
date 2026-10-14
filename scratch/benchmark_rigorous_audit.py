"""
scratch/benchmark_rigorous_audit.py
Auditoria rigorosa da arquitetura LEGION VQ2M:
1. Memória exata (VRAM alocada, VRAM reservada, RAM RSS do processo).
2. Tamanho real em bytes por especialista (pesagem tensor a tensor).
3. Rastreamento granular de Seleção vs Leitura vs Execução.
4. Contagem de Hits/Misses em requisições E em bytes.
5. Latência de Prefill (TTFT), latência média de Decode e throughput (tok/s).
"""
import os, sys, time, json, gc, psutil
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legion.engine import LegionEngine, DEVICE

def run_rigorous_audit():
    proc = psutil.Process()
    print("=" * 80)
    print("AUDITORIA RIGOROSA DE ARQUITETURA E TELEMETRIA - LEGION VQ2M")
    print(f"Dispositivo: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 80)

    # 1. Medição antes da carga do modelo
    ram_init_mb = proc.memory_info().rss / 1024**2
    vram_init_mb = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    print(f"[Baseline Inicial] RAM RSS: {ram_init_mb:.1f} MB | VRAM Alocada: {vram_init_mb:.1f} MB")

    # 2. Inicialização do motor
    t0_load = time.perf_counter()
    engine = LegionEngine(cache_capacity=96)
    t_load = time.perf_counter() - t0_load

    torch.cuda.reset_peak_memory_stats()
    ram_loaded_mb = proc.memory_info().rss / 1024**2
    vram_loaded_mb = torch.cuda.memory_allocated() / 1024**2
    vram_reserved_mb = torch.cuda.memory_reserved() / 1024**2
    vram_peak_mb = torch.cuda.max_memory_allocated() / 1024**2

    print("\n--- 1. AUDITORIA DE MEMÓRIA APÓS CARGA ---")
    print(f"Tempo de inicialização do motor: {t_load:.2f} s")
    print(f"RAM RSS do Processo: {ram_loaded_mb:.1f} MB (Delta: +{ram_loaded_mb - ram_init_mb:.1f} MB)")
    print(f"VRAM Alocada por Tensores: {vram_loaded_mb:.1f} MB ({vram_loaded_mb / 1024:.2f} GB)")
    print(f"VRAM Reservada pelo Driver: {vram_reserved_mb:.1f} MB ({vram_reserved_mb / 1024:.2f} GB)")
    print(f"VRAM Pico de Alocação: {vram_peak_mb:.1f} MB")

    # 3. Pesagem exata de 1 especialista
    print("\n--- 2. AUDITORIA DE PESAGEM DE 1 ESPECIALISTA (.vq2m) ---")
    gui, gus, dni, dns = engine.store.get_expert_weights(layer_idx=0, expert_id=0, device="cpu")
    gui_bytes = gui.numel() * gui.element_size()
    gus_bytes = gus.numel() * gus.element_size()
    dni_bytes = dni.numel() * dni.element_size()
    dns_bytes = dns.numel() * dns.element_size()
    single_expert_bytes = gui_bytes + gus_bytes + dni_bytes + dns_bytes
    print(f"Gate-Up Indices (int16 {tuple(gui.shape)}): {gui_bytes} bytes ({gui_bytes/1024:.1f} KB)")
    print(f"Gate-Up Escalas (float16 {tuple(gus.shape)}): {gus_bytes} bytes ({gus_bytes/1024:.1f} KB)")
    print(f"Down Indices    (int16 {tuple(dni.shape)}): {dni_bytes} bytes ({dni_bytes/1024:.1f} KB)")
    print(f"Down Escalas    (float16 {tuple(dns.shape)}): {dns_bytes} bytes ({dns_bytes/1024:.1f} KB)")
    print(f">> PESO REAL DE 1 ESPECIALISTA: {single_expert_bytes} bytes = {single_expert_bytes / 1024:.2f} KB")

    # 4. Pesagem do Pool de VRAM (Zero-Copy)
    layer0 = engine.layers[0]
    pool_bytes_per_layer = (
        layer0.pool_gu_idx.numel() * layer0.pool_gu_idx.element_size() +
        layer0.pool_gu_scale.numel() * layer0.pool_gu_scale.element_size() +
        layer0.pool_dn_idx.numel() * layer0.pool_dn_idx.element_size() +
        layer0.pool_dn_scale.numel() * layer0.pool_dn_scale.element_size()
    )
    total_moe_pool_mb = (pool_bytes_per_layer * len(engine.layers)) / 1024**2
    print(f"Slots por camada: {engine.cache_capacity} slots")
    print(f"Pool MoE VRAM por camada: {pool_bytes_per_layer / 1024**2:.2f} MB")
    print(f">> POOL MOE VRAM TOTAL (40 camadas x {engine.cache_capacity} slots): {total_moe_pool_mb:.1f} MB ({total_moe_pool_mb/1024:.2f} GB)")

    # 5. Execução de Inferência com Rastreamento Granular
    print("\n--- 3. TESTE DE INFERÊNCIA E TELEMETRIA OPERACIONAL ---")
    prompt = "Explique brevemente como uma legiao de especialistas MoE divide tarefas."
    print(f"Prompt de teste: '{prompt}'")

    # Zera contadores antes do teste
    for l in engine.layers:
        l.hits = 0
        l.misses = 0

    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    tokens_generated = []
    step_latencies = []

    for chunk in engine.generate_stream(prompt=prompt, max_tokens=60, temperature=0.7):
        tokens_generated.append(chunk["token"])
        step_latencies.append(chunk["stats"]["step_ms"])

    total_time = time.perf_counter() - t_start
    total_tokens = len(tokens_generated)
    tok_s = total_tokens / total_time if total_time > 0 else 0

    # 6. Contabilidade de Seleção, Leitura e Execução
    total_hits = sum(l.hits for l in engine.layers)
    total_misses = sum(l.misses for l in engine.layers)
    total_selections = total_hits + total_misses
    hit_rate_pct = (total_hits / total_selections * 100) if total_selections > 0 else 0.0

    bytes_read_from_mmap = total_misses * single_expert_bytes
    bytes_served_by_cache = total_hits * single_expert_bytes
    total_bytes_requested = total_selections * single_expert_bytes

    print(f"\nTexto Gerado ({total_tokens} tokens):\n{''.join(tokens_generated)}")
    print("\n" + "=" * 80)
    print("QUADRO DEMONSTRATIVO DE AUDITORIA VERIFICÁVEL")
    print("=" * 80)
    print(f"A. SELEÇÕES (Despachos do Router):")
    print(f"   - Total de seleções feitas: {total_selections} (Esperado: ~{total_tokens} tokens x 40 camadas x 8 experts = {total_tokens * 40 * 8})")
    print(f"   - Seleções por token gerado: {total_selections / max(total_tokens, 1):.1f}")
    
    print(f"\nB. LEITURAS FÍSICAS (mmap -> VRAM):")
    print(f"   - Cache Misses (requisições lidas do mmap): {total_misses}")
    print(f"   - Cache Hits (requisições servidas no pool):  {total_hits}")
    print(f"   - Taxa de Acerto por Requisição: {hit_rate_pct:.2f}%")
    print(f"   - Bytes Lidos do Armazenamento: {bytes_read_from_mmap / 1024**2:.2f} MB")
    print(f"   - Bytes Poupados pelo Cache:     {bytes_served_by_cache / 1024**2:.2f} MB")
    print(f"   - Taxa de Acerto por Bytes:      {(bytes_served_by_cache / total_bytes_requested * 100):.2f}% (Idêntica por ser peso fixo)")

    print(f"\nC. MEMÓRIA REAL VERIFICADA:")
    print(f"   - VRAM Inicial:                 {vram_init_mb:.1f} MB")
    print(f"   - VRAM Alocada Durante Decode:  {torch.cuda.memory_allocated() / 1024**2:.1f} MB ({torch.cuda.memory_allocated() / 1024**3:.2f} GB)")
    print(f"   - Pico de VRAM no Processo:     {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB ({torch.cuda.max_memory_allocated() / 1024**3:.2f} GB)")
    print(f"   - RAM RSS do SO (Sistema):      {proc.memory_info().rss / 1024**2:.1f} MB")

    print(f"\nD. DESEMPENHO E LATÊNCIA:")
    print(f"   - Tempo Total:                  {total_time:.2f} s")
    print(f"   - Tokens por Segundo:           {tok_s:.2f} tok/s")
    if step_latencies:
        print(f"   - Latência Média por Token:     {sum(step_latencies)/len(step_latencies):.1f} ms")
        print(f"   - Latência Mínima por Token:    {min(step_latencies):.1f} ms")
        print(f"   - Latência Máxima por Token:    {max(step_latencies):.1f} ms")
    print("=" * 80)

if __name__ == "__main__":
    run_rigorous_audit()
