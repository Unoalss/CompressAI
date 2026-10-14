import sys, time
from pathlib import Path
from collections import defaultdict
import torch

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legion.engine import LegionEngine
from vq2 import vq

def run_hadamard_profiling():
    print("=" * 80)
    print("OTIMIZAÇÃO 2 - FASE A: PROFILING ISOLADO DAS ROTAÇÕES HADAMARD")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "CUDA necessária para medição de GPU"

    # 1. Carrega as instâncias de rotação reais
    rot_2048 = vq.get_rot("moe_expert_2048", 2048, device)
    rot_512 = vq.get_rot("moe_expert_512", 512, device)

    # -------------------------------------------------------------
    # PARTE 1: MICRO-BENCHMARK ISOLADO (40 chamadas = 1 token equivalente)
    # -------------------------------------------------------------
    print("\n[PARTE 1] Micro-Benchmark Isolado (40 chamadas = 1 token equivalente):")
    
    # Entradas típicas
    x2048 = torch.randn(2048, device=device, dtype=torch.bfloat16)
    act512_k8 = torch.randn(8, 512, device=device, dtype=torch.bfloat16)

    # Warmup
    for _ in range(50):
        _ = rot_2048.fwd(x2048)
        _ = rot_512.fwd(act512_k8)
    torch.cuda.synchronize()

    N_REPS = 100 # 100 tokens equivalentes = 4000 chamadas
    CALLS_PER_REP = 40 # 40 camadas MoE

    # Medição rot_2048 (GPU puro via CUDA Events)
    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)

    t0_cpu = time.perf_counter()
    ev_start.record()
    for _ in range(N_REPS):
        for _ in range(CALLS_PER_REP):
            _ = rot_2048.fwd(x2048)
    ev_end.record()
    torch.cuda.synchronize()
    t1_cpu = time.perf_counter()

    gpu_ms_rot2048 = ev_start.elapsed_time(ev_end)
    cpu_ms_rot2048 = (t1_cpu - t0_cpu) * 1000.0

    # Medição rot_512 (GPU puro via CUDA Events)
    ev_start.record()
    t0_cpu = time.perf_counter()
    for _ in range(N_REPS):
        for _ in range(CALLS_PER_REP):
            _ = rot_512.fwd(act512_k8)
    ev_end.record()
    torch.cuda.synchronize()
    t1_cpu = time.perf_counter()

    gpu_ms_rot512 = ev_start.elapsed_time(ev_end)
    cpu_ms_rot512 = (t1_cpu - t0_cpu) * 1000.0

    print(f"  rot_2048 (40 camadas por token | entrada [2048]):")
    print(f"    - Tempo GPU puro:       {gpu_ms_rot2048 / N_REPS:.3f} ms/token ({gpu_ms_rot2048 / (N_REPS * CALLS_PER_REP) * 1000:.1f} us/chamada)")
    print(f"    - Tempo CPU (launch):   {cpu_ms_rot2048 / N_REPS:.3f} ms/token")
    print(f"  rot_512 (40 camadas por token | entrada [8, 512]):")
    print(f"    - Tempo GPU puro:       {gpu_ms_rot512 / N_REPS:.3f} ms/token ({gpu_ms_rot512 / (N_REPS * CALLS_PER_REP) * 1000:.1f} us/chamada)")
    print(f"    - Tempo CPU (launch):   {cpu_ms_rot512 / N_REPS:.3f} ms/token")
    print(f"  Soma Isolada das 2 rotações (40 camadas):")
    print(f"    - GPU Puro:             {(gpu_ms_rot2048 + gpu_ms_rot512) / N_REPS:.3f} ms/token")
    print(f"    - CPU Overhead:         {(cpu_ms_rot2048 + cpu_ms_rot512) / N_REPS:.3f} ms/token")

    # -------------------------------------------------------------
    # PARTE 2: PROFILING DENTRO DO MODELO REAL (com PagedMoEExpert)
    # -------------------------------------------------------------
    print("\n[PARTE 2] Profiling Dentro do Motor LEGION Completo...")
    engine = LegionEngine()
    tokenizer = engine.tokenizer

    prompt = "A inteligência artificial avançada permite arquiteturas neurais altamente eficientes com"
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    # Warmup no motor
    with torch.no_grad():
        out = engine.model(input_ids=inputs, use_cache=True)
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

    # Instrumenta rot_2048 e rot_512 no modelo real
    real_timings = defaultdict(list)
    call_counts = defaultdict(int)

    orig_fwd_2048 = rot_2048.fwd
    orig_fwd_512 = rot_512.fwd

    def timed_rot2048(x):
        call_counts["rot_2048"] += 1
        ev_s = torch.cuda.Event(enable_timing=True)
        ev_e = torch.cuda.Event(enable_timing=True)
        ev_s.record()
        res = orig_fwd_2048(x)
        ev_e.record()
        torch.cuda.synchronize()
        real_timings["rot_2048"].append(ev_s.elapsed_time(ev_e))
        return res

    def timed_rot512(act):
        call_counts["rot_512"] += 1
        ev_s = torch.cuda.Event(enable_timing=True)
        ev_e = torch.cuda.Event(enable_timing=True)
        ev_s.record()
        res = orig_fwd_512(act)
        ev_e.record()
        torch.cuda.synchronize()
        real_timings["rot_512"].append(ev_s.elapsed_time(ev_e))
        return res

    rot_2048.fwd = timed_rot2048
    rot_512.fwd = timed_rot512

    N_TOKENS = 10
    print(f"Executando {N_TOKENS} tokens de decodificação autoregressiva real...")
    for _ in range(N_TOKENS):
        with torch.no_grad():
            out = engine.model(input_ids=curr, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()
            curr = torch.tensor([[next_tok]], device=device)

    # Restaura
    rot_2048.fwd = orig_fwd_2048
    rot_512.fwd = orig_fwd_512

    tot_ms_2048 = sum(real_timings["rot_2048"]) / N_TOKENS
    tot_ms_512 = sum(real_timings["rot_512"]) / N_TOKENS
    calls_2048 = call_counts["rot_2048"] // N_TOKENS
    calls_512 = call_counts["rot_512"] // N_TOKENS

    print("\n" + "=" * 80)
    print("RELATÓRIO DE PROFILING REAL (DENTRO DA DECODIFICAÇÃO LEGION):")
    print("=" * 80)
    print(f"1. Rotação rot_2048:")
    print(f"   - Chamadas reais por token:  {calls_2048} (1 por camada MoE com especialistas comuns)")
    print(f"   - Tempo GPU sincronizado:    {tot_ms_2048:.2f} ms/token ({tot_ms_2048 / calls_2048 * 1000:.1f} us/camada)")
    print(f"2. Rotação rot_512:")
    print(f"   - Chamadas reais por token:  {calls_512} (1 por camada MoE com especialistas comuns)")
    print(f"   - Tempo GPU sincronizado:    {tot_ms_512:.2f} ms/token ({tot_ms_512 / calls_512 * 1000:.1f} us/camada)")
    print(f"--------------------------------------------------------------------------------")
    print(f"TEMPO TOTAL DAS ROTAÇÕES HADAMARD: {tot_ms_2048 + tot_ms_512:.2f} ms por token!")
    print("=" * 80)

if __name__ == "__main__":
    run_hadamard_profiling()
