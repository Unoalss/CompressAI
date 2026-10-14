import sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legion.engine import LegionEngine
from vq2.kernels import triton_conv1d_update
import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as mod

def repeated_benchmark():
    engine = LegionEngine()
    tokenizer = engine.tokenizer

    prompt = "A inteligência artificial avançada permite arquiteturas neurais altamente eficientes com"
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")

    def pytorch_fallback(hidden_states, conv_state, weight, bias=None, activation=None):
        _, hidden_size, seq_len = hidden_states.shape
        state_len = conv_state.shape[-1]
        hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
        conv_state.copy_(hidden_states_new[:, :, -state_len:])
        out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
        out = out[:, :, -seq_len:]
        if activation is not None:
            out = F.silu(out)
        return out.to(hidden_states.dtype)

    def triton_wrapper(hidden_states, conv_state, weight, bias=None, activation=None):
        if (hidden_states.is_cuda and hidden_states.shape[0] == 1 and hidden_states.shape[-1] == 1 and
            conv_state.shape[-1] == 4 and (activation == "silu" or activation is None)):
            return triton_conv1d_update(hidden_states, conv_state, weight, bias)
        return pytorch_fallback(hidden_states, conv_state, weight, bias, activation)

    def run_batch(fn, n_tokens=30):
        mod.causal_conv1d_update = fn

        torch.cuda.synchronize()
        with torch.no_grad():
            out = engine.model(input_ids=prompt_ids, use_cache=True)
            past_kv = out.past_key_values
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()

        curr = torch.tensor([[next_tok]], device="cuda")

        # 3 tokens warmup
        for _ in range(3):
            with torch.no_grad():
                out = engine.model(input_ids=curr, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
                next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()
                curr = torch.tensor([[next_tok]], device="cuda")
        torch.cuda.synchronize()

        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        ev_start.record()

        for _ in range(n_tokens):
            with torch.no_grad():
                out = engine.model(input_ids=curr, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
                next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()
                curr = torch.tensor([[next_tok]], device="cuda")

        ev_end.record()
        torch.cuda.synchronize()
        gpu_time_ms = ev_start.elapsed_time(ev_end)
        return gpu_time_ms / n_tokens, n_tokens / (gpu_time_ms / 1000.0)

    py_ms = []
    py_tps = []
    tri_ms = []
    tri_tps = []

    print("=" * 70)
    print("BENCHMARK REPETIDO RIGOROSO (3 RODADAS ALTERNADAS)")
    print("=" * 70)

    for round_idx in range(1, 4):
        ms_p, tps_p = run_batch(pytorch_fallback, n_tokens=30)
        py_ms.append(ms_p)
        py_tps.append(tps_p)

        ms_t, tps_t = run_batch(triton_wrapper, n_tokens=30)
        tri_ms.append(ms_t)
        tri_tps.append(tps_t)

        print(f"Rodada {round_idx}:")
        print(f"  PyTorch Fallback:      {ms_p:.2f} ms/tok  ({tps_p:.2f} tok/s)")
        print(f"  Triton Fused Kernel:   {ms_t:.2f} ms/tok  ({tps_t:.2f} tok/s)")
        print(f"  Ganho na Rodada:       +{tps_t - tps_p:.2f} tok/s (-{ms_p - ms_t:.2f} ms/tok)")

    avg_py_ms = sum(py_ms) / 3
    avg_py_tps = sum(py_tps) / 3
    avg_tri_ms = sum(tri_ms) / 3
    avg_tri_tps = sum(tri_tps) / 3

    print("=" * 70)
    print("MÉDIAS CONSOLIDADAS (3 RODADAS REAIS):")
    print(f"  PyTorch Fallback:      {avg_py_ms:.2f} ms/tok  ({avg_py_tps:.2f} tok/s)")
    print(f"  Triton Fused Kernel:   {avg_tri_ms:.2f} ms/tok  ({avg_tri_tps:.2f} tok/s)")
    print(f"  Economia Média:        -{avg_py_ms - avg_tri_ms:.2f} ms por token")
    print(f"  Ganho de Throughput:   +{avg_tri_tps - avg_py_tps:.2f} tok/s (+{(avg_tri_tps/avg_py_tps - 1)*100:.1f}%)")
    print("=" * 70)

    # Restaura permanentemente o Triton
    mod.causal_conv1d_update = triton_wrapper

if __name__ == "__main__":
    repeated_benchmark()
