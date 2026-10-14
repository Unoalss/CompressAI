import sys, os, time
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legion.engine import LegionEngine

def test_deltanet_graph_patch():
    print("=" * 80)
    print("TESTE: ACELERAÇÃO DE TODAS AS 30 CAMADAS DELTANET VIA CUDA GRAPHS")
    print("=" * 80)

    engine = LegionEngine()
    device = "cuda"

    prompt = "A inteligência artificial avançada permite criar sistemas neurais altamente eficientes com"
    prompt_ids = engine.tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()

    with torch.no_grad():
        out = engine.model(input_ids=prompt_ids, use_cache=True)
        past_kv = out.past_key_values
        next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()

    curr = torch.tensor([[next_tok]], device=device)
    # Warmup de 10 passos
    for _ in range(10):
        with torch.no_grad():
            out = engine.model(input_ids=curr, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()
            curr = torch.tensor([[next_tok]], device=device)
    torch.cuda.synchronize()

    # Captura grafos para linear_attn em cada uma das 30 camadas DeltaNet
    print("\n[1/3] Capturando CUDA Graphs para linear_attn nas 30 camadas DeltaNet...")
    graph_data = {}
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())

    for l_idx, layer in enumerate(engine.model.model.layers):
        if layer.block_type != "linear_attention":
            continue

        static_in = torch.zeros(1, 1, 2048, dtype=torch.bfloat16, device=device)
        attn_fn = layer.linear_attn

        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                with torch.no_grad():
                    _ = attn_fn(static_in, cache_params=past_kv)
        torch.cuda.current_stream().wait_stream(capture_stream)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=capture_stream):
            static_out = attn_fn(static_in, cache_params=past_kv)

        graph_data[l_idx] = {
            "graph": g,
            "static_in": static_in,
            "static_out": static_out,
            "orig_fn": attn_fn
        }

        # Cria wrapper que usa replay
        def make_graph_linear_attn(info):
            def graph_fwd(hidden_states, cache_params=None, **kwargs):
                if hidden_states.shape[1] == 1 and hidden_states.is_cuda:
                    info["static_in"].copy_(hidden_states)
                    info["graph"].replay()
                    return info["static_out"]
                return info["orig_fn"](hidden_states, cache_params=cache_params, **kwargs)
            return graph_fwd

        layer.linear_attn.forward = make_graph_linear_attn(graph_data[l_idx])

    print("      [OK] 30 camadas DeltaNet empacotadas em CUDA Graphs!")

    # 2. Teste de Geração de 30 tokens
    print("\n[2/3] Gerando 30 tokens com as 30 camadas DeltaNet em CUDA Graphs...")
    curr_in = torch.tensor([[next_tok]], device=device)
    gen_tokens = [next_tok]
    latencies = []

    for _ in range(30):
        ev_s = torch.cuda.Event(enable_timing=True)
        ev_e = torch.cuda.Event(enable_timing=True)
        ev_s.record()
        with torch.no_grad():
            out = engine.model(input_ids=curr_in, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_t = torch.argmax(out.logits[:, -1, :], dim=-1).item()
            gen_tokens.append(next_t)
            curr_in = torch.tensor([[next_t]], device=device)
        ev_e.record()
        torch.cuda.synchronize()
        latencies.append(ev_s.elapsed_time(ev_e))

    text = engine.tokenizer.decode(gen_tokens)
    p50_ms = sorted(latencies)[15]
    avg_ms = sum(latencies) / len(latencies)
    min_ms = min(latencies)
    tps = 1000.0 / p50_ms
    peak_tps = 1000.0 / min_ms

    print("\n" + "=" * 80)
    print("RESULTADOS DO TESTE COM DELTANET CUDA GRAPHS:")
    print("=" * 80)
    print(f"Velocidade Mediana (P50): {tps:.2f} tok/s ({p50_ms:.2f} ms/token)")
    print(f"Velocidade Média:         {1000.0/avg_ms:.2f} tok/s ({avg_ms:.2f} ms/token)")
    print(f"Pico Instantâneo:         {peak_tps:.2f} tok/s ({min_ms:.2f} ms/token)")
    print("-" * 80)
    print(f"Texto Gerado:\n\"{text}\"")
    print("=" * 80)

if __name__ == "__main__":
    test_deltanet_graph_patch()
