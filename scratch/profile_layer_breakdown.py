import sys, os, time
from pathlib import Path
import torch

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legion.engine import LegionEngine

engine = LegionEngine()
prompt = "A inteligência artificial avançada permite arquiteturas neurais altamente eficientes com"
prompt_ids = engine.tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()

with torch.no_grad():
    out = engine.model(input_ids=prompt_ids, use_cache=True)
    past_kv = out.past_key_values
    next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()
    curr = torch.tensor([[next_tok]], device="cuda")

# Warmup cache
for _ in range(30):
    with torch.no_grad():
        out = engine.model(input_ids=curr, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()
        curr = torch.tensor([[next_tok]], device="cuda")
torch.cuda.synchronize()

# Agora vamos interceptar exatamente o que acontece dentro de uma camada DecoderLayer
layer0 = engine.model.model.layers[0] # DeltaNet
layer3 = engine.model.model.layers[3] # Full Attention

# Medição de 100 execuções de Layer 0
N = 100
h = torch.randn(1, 1, 2048, device="cuda", dtype=torch.bfloat16)

# 1. Medição isolada de cada etapa da Layer 0
torch.cuda.synchronize()
ev_start = torch.cuda.Event(enable_timing=True)
ev_end = torch.cuda.Event(enable_timing=True)

# Etapa A: input_layernorm
ev_start.record()
for _ in range(N):
    h_norm = layer0.input_layernorm(h)
ev_end.record()
torch.cuda.synchronize()
t_ln1 = ev_start.elapsed_time(ev_end) / N

# Etapa B: linear_attn
ev_start.record()
for _ in range(N):
    attn_out = layer0.linear_attn(h_norm, cache_params=past_kv)
ev_end.record()
torch.cuda.synchronize()
t_attn = ev_start.elapsed_time(ev_end) / N

# Etapa C: post_attention_layernorm
h_res = h + attn_out
ev_start.record()
for _ in range(N):
    h_post = layer0.post_attention_layernorm(h_res)
ev_end.record()
torch.cuda.synchronize()
t_ln2 = ev_start.elapsed_time(ev_end) / N

# Etapa D: mlp (gate + experts)
ev_start.record()
for _ in range(N):
    mlp_out = layer0.mlp(h_post)
ev_end.record()
torch.cuda.synchronize()
t_mlp = ev_start.elapsed_time(ev_end) / N

# Etapa E: layer0 inteira (chamando forward)
# Atenção: para a layer0 inteira, passamos os argumentos normais
pos_emb = engine.model.model.rotary_emb(h, torch.full((3, 1, 1), 50, device="cuda", dtype=torch.long))
ev_start.record()
for _ in range(N):
    layer_out = layer0(h, position_embeddings=pos_emb, past_key_values=past_kv, use_cache=True)
ev_end.record()
torch.cuda.synchronize()
t_layer_full = ev_start.elapsed_time(ev_end) / N

print("=" * 80)
print("DECOMPOSIÇÃO MICRO-TEMPORAL DE 1 CAMADA DELTANET (LAYER 0):")
print("=" * 80)
print(f"1. input_layernorm:            {t_ln1:.4f} ms")
print(f"2. linear_attn:                {t_attn:.4f} ms")
print(f"3. post_attention_layernorm:   {t_ln2:.4f} ms")
print(f"4. mlp (router + experts):     {t_mlp:.4f} ms")
print(f"Soma das partes isoladas:      {t_ln1 + t_attn + t_ln2 + t_mlp:.4f} ms")
print(f"layer0(forward) completo:      {t_layer_full:.4f} ms")
print(f"Diferença / Overhead da camada:{t_layer_full - (t_ln1 + t_attn + t_ln2 + t_mlp):.4f} ms")
print(f"Projeção para 40 camadas:      {(t_layer_full - (t_ln1 + t_attn + t_ln2 + t_mlp)) * 40:.2f} ms")

# Agora decompomos o MLP em Router vs Experts
ev_start.record()
for _ in range(N):
    router_logits = layer0.mlp.gate(h_post)
    r_probs = torch.nn.functional.softmax(router_logits, dim=-1, dtype=torch.float32)
    topk_w, topk_idx = torch.topk(r_probs, 8, dim=-1)
    topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
ev_end.record()
torch.cuda.synchronize()
t_router = ev_start.elapsed_time(ev_end) / N

ev_start.record()
for _ in range(N):
    exp_out = layer0.mlp.experts(h_post.view(1, 2048), topk_idx, topk_w)
ev_end.record()
torch.cuda.synchronize()
t_exp = ev_start.elapsed_time(ev_end) / N

print("\n" + "=" * 80)
print("DECOMPOSIÇÃO DO BLOCO MLP (ROUTER vs EXPERTS):")
print("=" * 80)
print(f"Router + TopK selection:       {t_router:.4f} ms")
print(f"MoE Experts (Paged):           {t_exp:.4f} ms")
print(f"Soma Router + Experts:         {t_router + t_exp:.4f} ms")
print(f"mlp(...) completo:             {t_mlp:.4f} ms")
print(f"Diferença no MLP:              {t_mlp - (t_router + t_exp):.4f} ms")
print("=" * 80)
