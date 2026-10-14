import sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legion.engine import LegionEngine
from vq2 import vq

def repeated_hadamard_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda"

    # Pré-computa as matrizes fundidas M
    rot_2048 = vq.get_rot("moe_expert_2048", 2048, device)
    rot_512 = vq.get_rot("moe_expert_512", 512, device)

    I_2048 = torch.eye(2048, device=device, dtype=torch.bfloat16)
    M_2048 = rot_2048.fwd(I_2048).contiguous()

    I_512 = torch.eye(512, device=device, dtype=torch.bfloat16)
    M_512 = rot_512.fwd(I_512).contiguous()

    engine = LegionEngine()
    tokenizer = engine.tokenizer

    prompt = "A inteligência artificial avançada permite arquiteturas neurais altamente eficientes com"
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    orig_fwd_2048 = rot_2048.fwd
    orig_fwd_512 = rot_512.fwd

    def fused_fwd_2048(x):
        return torch.matmul(x, M_2048)

    def fused_fwd_512(act):
        return torch.matmul(act, M_512)

    def run_batch(use_fused, n_tokens=30):
        if use_fused:
            rot_2048.fwd = fused_fwd_2048
            rot_512.fwd = fused_fwd_512
        else:
            rot_2048.fwd = orig_fwd_2048
            rot_512.fwd = orig_fwd_512

        torch.cuda.synchronize()
        with torch.no_grad():
            out = engine.model(input_ids=prompt_ids, use_cache=True)
            past_kv = out.past_key_values
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()

        curr = torch.tensor([[next_tok]], device=device)

        # 3 tokens warmup
        for _ in range(3):
            with torch.no_grad():
                out = engine.model(input_ids=curr, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
                next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()
                curr = torch.tensor([[next_tok]], device=device)
        torch.cuda.synchronize()

        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        ev_start.record()

        for _ in range(n_tokens):
            with torch.no_grad():
                out = engine.model(input_ids=curr, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
                next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()
                curr = torch.tensor([[next_tok]], device=device)

        ev_end.record()
        torch.cuda.synchronize()
        gpu_time_ms = ev_start.elapsed_time(ev_end)
        return gpu_time_ms / n_tokens, n_tokens / (gpu_time_ms / 1000.0)

    orig_ms = []
    orig_tps = []
    fused_ms = []
    fused_tps = []

    print("=" * 70)
    print("BENCHMARK REPETIDO: 3 RODADAS ALTERNADAS (30 TOKENS CADA)")
    print("=" * 70)

    for round_idx in range(1, 4):
        ms_o, tps_o = run_batch(use_fused=False, n_tokens=30)
        orig_ms.append(ms_o)
        orig_tps.append(tps_o)

        ms_f, tps_f = run_batch(use_fused=True, n_tokens=30)
        fused_ms.append(ms_f)
        fused_tps.append(tps_f)

        print(f"Rodada {round_idx}:")
        print(f"  Rotação Original:     {ms_o:.2f} ms/tok  ({tps_o:.2f} tok/s)")
        print(f"  Rotação Fundida:      {ms_f:.2f} ms/tok  ({tps_f:.2f} tok/s)")
        print(f"  Ganho na Rodada:      +{tps_f - tps_o:.2f} tok/s (-{ms_o - ms_f:.2f} ms/tok)")

    avg_o_ms = sum(orig_ms) / 3
    avg_o_tps = sum(orig_tps) / 3
    avg_f_ms = sum(fused_ms) / 3
    avg_f_tps = sum(fused_tps) / 3

    print("=" * 70)
    print("MÉDIAS CONSOLIDADAS (3 RODADAS):")
    print(f"  Rotação Original:     {avg_o_ms:.2f} ms/tok  ({avg_o_tps:.2f} tok/s)")
    print(f"  Rotação Fundida:      {avg_f_ms:.2f} ms/tok  ({avg_f_tps:.2f} tok/s)")
    print(f"  Economia Média:       -{avg_o_ms - avg_f_ms:.2f} ms por token")
    print(f"  Ganho de Throughput:  +{avg_f_tps - avg_o_tps:.2f} tok/s (+{(avg_f_tps/avg_o_tps - 1)*100:.1f}%)")
    print("=" * 70)

if __name__ == "__main__":
    repeated_hadamard_benchmark()
