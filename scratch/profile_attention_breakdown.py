import sys, time
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legion.engine import LegionEngine

def profile_attention():
    print("=" * 80)
    print("PROFILING SUB-OPERACIONAL DE ATENÇÃO (DELTA-NET + FULL ATTENTION)")
    print("=" * 80)

    engine = LegionEngine()
    tokenizer = engine.tokenizer

    prompt = "A inteligência artificial avançada demonstra que a otimização de kernels"
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")

    # Warmup
    print("Executando prefill e warmup...")
    with torch.no_grad():
        out = engine.model(input_ids=inputs, use_cache=True)
        past_kv = out.past_key_values
        next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()

    curr = torch.tensor([[next_tok]], device="cuda")
    for _ in range(3):
        with torch.no_grad():
            out = engine.model(input_ids=curr, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()
            curr = torch.tensor([[next_tok]], device="cuda")
    torch.cuda.synchronize()

    # Vamos instrumentar as camadas de atenção do modelo
    timings = defaultdict(list)

    # Identificar quais são linear_attn e quais são self_attn (full attention)
    linear_layers = []
    full_layers = []

    for l_idx, layer in enumerate(engine.model.model.layers):
        if hasattr(layer, "linear_attn"):
            linear_layers.append((l_idx, layer.linear_attn))
        elif hasattr(layer, "self_attn"):
            full_layers.append((l_idx, layer.self_attn))

    print(f"Camadas identificadas: {len(linear_layers)} Linear Attention (DeltaNet) | {len(full_layers)} Full Attention")

    # Monkey patch no forward do linear_attn e self_attn para medir sub-etapas
    orig_linear_fwd = linear_layers[0][1].__class__.forward
    orig_full_fwd = full_layers[0][1].__class__.forward

    # Eventos CUDA para medição precisa
    def timed_linear_forward(self, hidden_states, *args, **kwargs):
        # Mede o bloco inteiro
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        ev_start.record()
        res = orig_linear_fwd(self, hidden_states, *args, **kwargs)
        ev_end.record()
        torch.cuda.synchronize()
        timings["linear_attn_total"].append(ev_start.elapsed_time(ev_end))
        return res

    def timed_full_forward(self, hidden_states, *args, **kwargs):
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        ev_start.record()
        res = orig_full_fwd(self, hidden_states, *args, **kwargs)
        ev_end.record()
        torch.cuda.synchronize()
        timings["full_attn_total"].append(ev_start.elapsed_time(ev_end))
        return res

    # Instrumenta causal_conv1d_update especificamente
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as mod
    orig_conv = mod.causal_conv1d_update
    def timed_conv(hidden_states, conv_state, weight, bias=None, activation=None):
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        ev_start.record()
        res = orig_conv(hidden_states, conv_state, weight, bias, activation)
        ev_end.record()
        torch.cuda.synchronize()
        timings["conv1d_update"].append(ev_start.elapsed_time(ev_end))
        return res

    linear_layers[0][1].__class__.forward = timed_linear_forward
    full_layers[0][1].__class__.forward = timed_full_forward
    mod.causal_conv1d_update = timed_conv

    # Executa 5 passos de decodificação
    print("\nMedindo 5 passos de decodificação...")
    N_STEPS = 5
    for s in range(N_STEPS):
        with torch.no_grad():
            out = engine.model(input_ids=curr, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()
            curr = torch.tensor([[next_tok]], device="cuda")

    # Restaura
    linear_layers[0][1].__class__.forward = orig_linear_fwd
    full_layers[0][1].__class__.forward = orig_full_fwd
    mod.causal_conv1d_update = orig_conv

    print("\n" + "=" * 80)
    print("RESULTADOS DO TEMPO MÉDIO POR TOKEN (CUDA SYNCHRONIZED):")
    print("=" * 80)
    
    linear_tot_per_tok = sum(timings["linear_attn_total"]) / N_STEPS
    full_tot_per_tok = sum(timings["full_attn_total"]) / N_STEPS
    conv_tot_per_tok = sum(timings["conv1d_update"]) / N_STEPS
    attn_grand_total = linear_tot_per_tok + full_tot_per_tok

    print(f"1. Linear Attention (30 camadas):   {linear_tot_per_tok:.2f} ms/tok  ({linear_tot_per_tok / 30 * 1000:.1f} us/camada)")
    print(f"   -> causal_conv1d_update:          {conv_tot_per_tok:.2f} ms/tok  ({conv_tot_per_tok / 30 * 1000:.1f} us/camada)")
    print(f"   -> DeltaNet / Projecoes / Norms:  {linear_tot_per_tok - conv_tot_per_tok:.2f} ms/tok")
    print(f"2. Full Attention (10 camadas):     {full_tot_per_tok:.2f} ms/tok  ({full_tot_per_tok / 10 * 1000:.1f} us/camada)")
    print(f"--------------------------------------------------------------------------------")
    print(f"TOTAL ATENÇÃO (40 CAMADAS):         {attn_grand_total:.2f} ms/tok")
    print("=" * 80)

if __name__ == "__main__":
    profile_attention()
