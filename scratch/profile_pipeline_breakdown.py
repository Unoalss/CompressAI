import sys, os, time
from pathlib import Path
import torch

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legion.engine import LegionEngine

def profile_breakdown():
    print("=" * 80)
    print("PROFILING DE PIPELINE: DECOMPOSIÇÃO DE TEMPO GPU vs CPU NO MOTOR LEGION")
    print("=" * 80)

    engine = LegionEngine()
    prompt = "A inteligência artificial avançada permite arquiteturas neurais altamente eficientes com"
    prompt_ids = engine.tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()

    # Aquecimento de 30 tokens para regime estacionário de cache
    print("Aquecendo motor e cache MoE...")
    with torch.no_grad():
        out = engine.model(input_ids=prompt_ids, use_cache=True)
        past_kv = out.past_key_values
        next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()
        curr = torch.tensor([[next_tok]], device="cuda")
        for _ in range(30):
            out = engine.model(input_ids=curr, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()
            curr = torch.tensor([[next_tok]], device="cuda")
    torch.cuda.synchronize()
    print("Regime estacionário atingido!\n")

    # Agora instrumentamos as camadas para medir sub-etapas exatas por token
    timings = {
        "router_gpu_ms": 0.0,
        "router_sync_cpu_ms": 0.0,
        "cache_lookup_cpu_ms": 0.0,
        "moe_hadamard_gpu_ms": 0.0,
        "moe_triton_kernels_gpu_ms": 0.0,
        "moe_activation_gpu_ms": 0.0,
        "deltanet_gpu_ms": 0.0,
        "full_attn_gpu_ms": 0.0,
        "layernorms_gpu_ms": 0.0,
        "total_token_gpu_ms": 0.0,
        "total_token_wall_ms": 0.0,
    }

    # Mediremos 20 tokens em detalhe
    N_TEST_TOKENS = 20

    # Monkey-patch temporário nos métodos para instrumentação
    moe_layers = [l.mlp.experts for l in engine.model.model.layers if hasattr(l.mlp, "experts")]
    linear_layers = [l.linear_attn for l in engine.model.model.layers if hasattr(l, "linear_attn")]
    full_layers = [l.self_attn for l in engine.model.model.layers if hasattr(l, "self_attn")]

    # Instrumentação dos experts MoE
    orig_moe_forward = moe_layers[0].__class__.forward
    
    layer_stats = {
        "moe_total_ms": [],
        "moe_sync_ms": [],
        "moe_triton_ms": [],
        "deltanet_ms": [],
        "full_attn_ms": [],
    }

    # Medição ponta a ponta por token
    torch.cuda.synchronize()
    token_wall_times = []
    token_gpu_times = []

    for step in range(N_TEST_TOKENS):
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        
        t0_wall = time.perf_counter()
        ev_start.record()

        with torch.no_grad():
            out = engine.model(input_ids=curr, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()
            curr = torch.tensor([[next_tok]], device="cuda")

        ev_end.record()
        torch.cuda.synchronize()
        t1_wall = time.perf_counter()

        token_wall_times.append((t1_wall - t0_wall) * 1000.0)
        token_gpu_times.append(ev_start.elapsed_time(ev_end))

    avg_wall = sum(token_wall_times) / len(token_wall_times)
    avg_gpu = sum(token_gpu_times) / len(token_gpu_times)

    print(f"Tempo Total Médio por Token:")
    print(f"  GPU Time:  {avg_gpu:.2f} ms/token ({1000.0/avg_gpu:.2f} tok/s)")
    print(f"  Wall Time: {avg_wall:.2f} ms/token ({1000.0/avg_wall:.2f} tok/s)")
    print(f"  CPU Overhead / Discrepância: {avg_wall - avg_gpu:+.2f} ms/token")

    # Agora profiling isolado por subsistema: MoE vs Atenção
    # 1. Medir 40 camadas MoE isoladas
    x_test = torch.randn(1, 2048, device="cuda", dtype=torch.bfloat16)
    topk_idx = torch.tensor([[10, 25, 42, 88, 105, 140, 201, 230]], device="cuda", dtype=torch.int64)
    topk_w = torch.ones(1, 8, device="cuda", dtype=torch.bfloat16) / 8.0

    torch.cuda.synchronize()
    ev_m_start = torch.cuda.Event(enable_timing=True)
    ev_m_end = torch.cuda.Event(enable_timing=True)
    ev_m_start.record()
    for _ in range(50):
        for l in moe_layers:
            l(x_test, topk_idx, topk_w)
    ev_m_end.record()
    torch.cuda.synchronize()
    moe_isolated_ms = ev_m_start.elapsed_time(ev_m_end) / 50.0

    # 2. Medir 30 camadas DeltaNet isoladas
    class DummyLayer:
        conv_states = [torch.randn(1, 12352, 4, device="cuda", dtype=torch.bfloat16)]
        recurrent_states = [torch.randn(1, 32, 128, 128, device="cuda", dtype=torch.float32)]
        record_past = False

    class DummyCache:
        def __init__(self):
            self.layers = [DummyLayer() for _ in range(40)]
        def has_previous_state(self, idx, state_idx=0): return True
        def update_recurrent_state(self, s, idx): pass
        def update_conv_state(self, s, idx, **kw): return s

    x_test_3d = torch.randn(1, 1, 2048, device="cuda", dtype=torch.bfloat16)
    dummy_cache = DummyCache()
    torch.cuda.synchronize()
    ev_d_start = torch.cuda.Event(enable_timing=True)
    ev_d_end = torch.cuda.Event(enable_timing=True)
    ev_d_start.record()
    for _ in range(50):
        for l in linear_layers:
            l(x_test_3d, cache_params=dummy_cache)
    ev_d_end.record()
    torch.cuda.synchronize()
    deltanet_isolated_ms = ev_d_start.elapsed_time(ev_d_end) / 50.0

    # 3. Medir 10 camadas Full Attention isoladas
    pos_emb = (
        torch.randn(1, 1, 64, device="cuda", dtype=torch.bfloat16),
        torch.randn(1, 1, 64, device="cuda", dtype=torch.bfloat16)
    )
    torch.cuda.synchronize()
    ev_fa_start = torch.cuda.Event(enable_timing=True)
    ev_fa_end = torch.cuda.Event(enable_timing=True)
    ev_fa_start.record()
    for _ in range(50):
        for l in full_layers:
            l(x_test_3d, position_embeddings=pos_emb)
    ev_fa_end.record()
    torch.cuda.synchronize()
    full_attn_isolated_ms = ev_fa_start.elapsed_time(ev_fa_end) / 50.0

    # 4. Medir custo do MoE Router (40 routers)
    routers = [l.mlp.gate for l in engine.model.model.layers if hasattr(l.mlp, "gate")]
    torch.cuda.synchronize()
    ev_r_start = torch.cuda.Event(enable_timing=True)
    ev_r_end = torch.cuda.Event(enable_timing=True)
    ev_r_start.record()
    for _ in range(100):
        for r in routers:
            r(x_test)
    ev_r_end.record()
    torch.cuda.synchronize()
    router_ms = ev_r_start.elapsed_time(ev_r_end) / 100.0

    # 5. Medir custo de 40 sincronizações CPU-GPU com .tolist()
    torch.cuda.synchronize()
    t_sync_0 = time.perf_counter()
    for _ in range(100):
        for _ in range(40):
            # simula topk_idx.tolist()
            lst = topk_idx.view(-1).tolist()
    t_sync_ms = (time.perf_counter() - t_sync_0) * 1000.0 / 100.0

    print("\n" + "=" * 80)
    print("DECOMPOSIÇÃO GRANULAR DO CAMINHO CRÍTICO (POR TOKEN):")
    print("=" * 80)
    print(f"1. 40 Camadas MoE Experts:         {moe_isolated_ms:6.2f} ms ({moe_isolated_ms/avg_gpu*100:5.1f}%)")
    print(f"2. 30 Camadas DeltaNet (Lineares): {deltanet_isolated_ms:6.2f} ms ({deltanet_isolated_ms/avg_gpu*100:5.1f}%)")
    print(f"3. 10 Camadas Full Attention:      {full_attn_isolated_ms:6.2f} ms ({full_attn_isolated_ms/avg_gpu*100:5.1f}%)")
    print(f"4. 40 Routers MoE:                 {router_ms:6.2f} ms ({router_ms/avg_gpu*100:5.1f}%)")
    print(f"5. Sobrecarga de Sync (.tolist):   {t_sync_ms:6.2f} ms (CPU-bound)")
    sub_sum = moe_isolated_ms + deltanet_isolated_ms + full_attn_isolated_ms + router_ms
    print(f"--------------------------------------------------------------------------------")
    print(f"Soma dos Componentes Isolados:     {sub_sum:6.2f} ms/token")
    print(f"Tempo Total Integrado GPU:         {avg_gpu:6.2f} ms/token")
    print(f"Overhead de Despacho / Pipeline:   {avg_gpu - sub_sum:6.2f} ms/token")
    print("=" * 80)

if __name__ == "__main__":
    profile_breakdown()
