import sys, time
from pathlib import Path
from collections import defaultdict
import torch

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legion.engine import LegionEngine

def run_cache_profiling():
    print("=" * 80)
    print("OTIMIZAÇÃO 3 - FASE A: PROFILING GRANULAR DO CACHE MoE E I/O")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda"

    engine = LegionEngine()
    tokenizer = engine.tokenizer

    prompt = "Explique a função do cache MoE em modelos de linguagem com mistura de especialistas:"
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    # 1. Warmup (prefill + 3 tokens para estabilizar o cache com seus especialistas quentes)
    print("\nExecutando prefill e warmup de 3 tokens...")
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

    # 2. Instrumentação fina das sub-etapas do cache em todas as 40 camadas
    timings = defaultdict(float)
    counts = defaultdict(int)

    orig_get_slot = engine.layers[0].__class__._get_slot

    def instrumented_get_slot(self, e_id: int) -> int:
        t0 = time.perf_counter()
        self.activated_experts_set.add(e_id)

        # Sub-etapa 1: Lookup e Hit
        if e_id in self.expert_to_slot:
            t_hit_start = time.perf_counter()
            self.hits += 1
            self.access_order.move_to_end(e_id)
            slot = self.expert_to_slot[e_id]
            t_hit_end = time.perf_counter()
            timings["hit_lookup_cpu"] += (t_hit_end - t_hit_start)
            counts["cache_hits"] += 1
            timings["total_get_slot_cpu"] += (t_hit_end - t0)
            return slot

        # Sub-etapa 2: Miss e Evicção LRU
        counts["cache_misses"] += 1
        t_evict_start = time.perf_counter()
        self.misses += 1
        if self.free_slots:
            slot = self.free_slots.pop()
        else:
            evict_id, _ = self.access_order.popitem(last=False)
            slot = self.expert_to_slot.pop(evict_id)
        t_evict_end = time.perf_counter()
        timings["lru_eviction_cpu"] += (t_evict_end - t_evict_start)

        # Sub-etapa 3 & 4: Leitura de dados e parsing
        if self.store.use_mmap:
            t_io_start = time.perf_counter()
            (off_gui, sz_gui, off_gus, sz_gus, off_dni, sz_dni, off_dns, sz_dns) = self.store.moe_offsets[self.layer_idx][e_id]
            gui = torch.frombuffer(self.store.mm, dtype=torch.int16, count=sz_gui // 2, offset=off_gui).view(self.store.gu_rows, self.store.gu_cols)
            gus = torch.frombuffer(self.store.mm, dtype=torch.float16, count=sz_gus // 2, offset=off_gus)
            dni = torch.frombuffer(self.store.mm, dtype=torch.int16, count=sz_dni // 2, offset=off_dni).view(self.store.dn_rows, self.store.dn_cols)
            dns = torch.frombuffer(self.store.mm, dtype=torch.float16, count=sz_dns // 2, offset=off_dns)
            t_io_end = time.perf_counter()
            timings["disk_read_io_cpu"] += (t_io_end - t_io_start)
            timings["parse_tensors_cpu"] += 0.0
        else:
            t_io_start = time.perf_counter()
            (off_gui, sz_gui, off_gus, sz_gus, off_dni, sz_dni, off_dns, sz_dns) = self.store.moe_offsets[self.layer_idx][e_id]
            with self.store.lock:
                self.store.file.seek(off_gui)
                self.store.file.readinto(self.store.expert_scratch)
            t_io_end = time.perf_counter()
            timings["disk_read_io_cpu"] += (t_io_end - t_io_start)

            t_parse_start = time.perf_counter()
            gui_raw = memoryview(self.store.expert_scratch)[0 : sz_gui]
            gus_raw = memoryview(self.store.expert_scratch)[sz_gui : sz_gui + sz_gus]
            dni_raw = memoryview(self.store.expert_scratch)[sz_gui + sz_gus : sz_gui + sz_gus + sz_dni]
            dns_raw = memoryview(self.store.expert_scratch)[sz_gui + sz_gus + sz_dni : sz_gui + sz_gus + sz_dni + sz_dns]

            gui = torch.frombuffer(gui_raw, dtype=torch.int16, count=sz_gui // 2).view(self.store.gu_rows, self.store.gu_cols)
            gus = torch.frombuffer(gus_raw, dtype=torch.float16, count=sz_gus // 2)
            dni = torch.frombuffer(dni_raw, dtype=torch.int16, count=sz_dni // 2).view(self.store.dn_rows, self.store.dn_cols)
            dns = torch.frombuffer(dns_raw, dtype=torch.float16, count=sz_dns // 2)
            t_parse_end = time.perf_counter()
            timings["parse_tensors_cpu"] += (t_parse_end - t_parse_start)

        # Sub-etapa 5: Cópia Host-to-Device para os buffers da GPU
        t_copy_start = time.perf_counter()
        self.pool_gu_idx[slot].copy_(gui)
        self.pool_gu_scale[slot].copy_(gus.to(dtype=torch.bfloat16))
        self.pool_dn_idx[slot].copy_(dni)
        self.pool_dn_scale[slot].copy_(dns.to(dtype=torch.bfloat16))
        t_copy_end = time.perf_counter()
        timings["htod_copy_cpu"] += (t_copy_end - t_copy_start)

        self.expert_to_slot[e_id] = slot
        self.access_order[e_id] = None
        t_end = time.perf_counter()
        timings["total_get_slot_cpu"] += (t_end - t0)
        return slot

    # Instrumenta a criação do tensor de slots
    orig_fwd = engine.layers[0].__class__.forward
    slots_tensor_creation_time = [0.0]

    # Aplica o monkey patch temporário
    engine.layers[0].__class__._get_slot = instrumented_get_slot

    N_TOKENS = 15
    print(f"\nMedindo {N_TOKENS} passos de decodificação com tracking detalhado...")

    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)
    ev_start.record()
    t_wall_start = time.perf_counter()

    for step in range(N_TOKENS):
        with torch.no_grad():
            out = engine.model(input_ids=curr, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()
            curr = torch.tensor([[next_tok]], device=device)

    ev_end.record()
    torch.cuda.synchronize()
    t_wall_end = time.perf_counter()

    # Restaura
    engine.layers[0].__class__._get_slot = orig_get_slot

    gpu_total_ms = ev_start.elapsed_time(ev_end)
    wall_total_ms = (t_wall_end - t_wall_start) * 1000.0

    total_calls = counts["cache_hits"] + counts["cache_misses"]
    hit_rate = (counts["cache_hits"] / total_calls * 100) if total_calls > 0 else 0

    print("\n" + "=" * 80)
    print("RELATÓRIO DA FASE A: DECOMPOSIÇÃO GRANULAR DO CACHE MoE E I/O")
    print("=" * 80)
    print(f"Tokens medidos:                {N_TOKENS}")
    print(f"Tempo total GPU:               {gpu_total_ms / N_TOKENS:.2f} ms/token ({N_TOKENS / (gpu_total_ms/1000):.2f} tok/s)")
    print(f"Tempo total Wall-Clock:        {wall_total_ms / N_TOKENS:.2f} ms/token")
    print(f"Chamadas _get_slot por token:  {total_calls / N_TOKENS:.1f} (8 experts x 40 camadas = 320 max)")
    print(f"Taxa de Acerto (Cache Hit):    {hit_rate:.2f}% ({counts['cache_hits']} hits / {counts['cache_misses']} misses)")
    print("--------------------------------------------------------------------------------")
    print("TEMPO MÉDIO POR TOKEN GASTO NAS SUB-ETAPAS:")
    print(f"  1. Cache Hit Lookup (Python dict):       {timings['hit_lookup_cpu'] / N_TOKENS * 1000:.2f} ms/tok")
    print(f"  2. Cache Miss - Evicção LRU:             {timings['lru_eviction_cpu'] / N_TOKENS * 1000:.2f} ms/tok")
    print(f"  3. Cache Miss - I/O Leitura de Disco:    {timings['disk_read_io_cpu'] / N_TOKENS * 1000:.2f} ms/tok")
    print(f"  4. Cache Miss - Parsing de Buffers:      {timings['parse_tensors_cpu'] / N_TOKENS * 1000:.2f} ms/tok")
    print(f"  5. Cache Miss - Cópias HtoD para GPU:    {timings['htod_copy_cpu'] / N_TOKENS * 1000:.2f} ms/tok")
    print(f"  ------------------------------------------------------------------------------")
    print(f"  TOTAL DE TEMPO DE GERENCIAMENTO DE SLOTS: {timings['total_get_slot_cpu'] / N_TOKENS * 1000:.2f} ms/tok")
    print("=" * 80)

if __name__ == "__main__":
    run_cache_profiling()
