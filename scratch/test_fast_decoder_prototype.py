import sys, os, time
from pathlib import Path
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legion.engine import LegionEngine
from vq2.kernels import (
    triton_conv1d_update,
    triton_rmsnorm_gated,
    triton_fused_rmsnorm_rope_qwen,
    fused_delta_gates
)
import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as qm

print("=" * 80)
print("PROTÓTIPO: LEGION FAST DECODER (BATCH 1 LINEARIZADO)")
print("=" * 80)

engine = LegionEngine()
device = "cuda"

model = engine.model.model
layers = model.layers

def rmsnorm_zero_centered(x, weight, eps=1e-6):
    var = x.float().pow(2).mean(-1, keepdim=True)
    norm = (x.float() * torch.rsqrt(var + eps)) * (1.0 + weight.float())
    return norm.to(x.dtype)

class FlatDecoder:
    def __init__(self, engine):
        self.engine = engine
        self.model = engine.model.model
        self.embed_tokens = self.model.embed_tokens
        self.final_norm_w = self.model.norm.weight
        self.final_norm_eps = self.model.norm.eps
        self.lm_head_w = engine.model.lm_head.weight
        self.rotary_emb = self.model.rotary_emb

        self.flat_layers = []
        for i, l in enumerate(self.model.layers):
            block_type = "linear_attention" if hasattr(l, "linear_attn") else "full_attention"
            fl = {
                "idx": i,
                "block_type": block_type,
                "input_ln_w": l.input_layernorm.weight,
                "input_ln_eps": l.input_layernorm.eps,
                "post_ln_w": l.post_attention_layernorm.weight,
                "post_ln_eps": l.post_attention_layernorm.eps,
                "router_w": l.mlp.gate.weight,
                "router_bias": getattr(l.mlp.gate, "bias", None),
                "top_k": l.mlp.gate.top_k,
                "experts": l.mlp.experts,
            }
            if block_type == "linear_attention":
                la = l.linear_attn
                fl["W_in_fused"] = la.W_in_fused
                fl["conv1d_w"] = la.conv1d.weight.squeeze(1)
                fl["conv1d_b"] = la.conv1d.bias
                fl["A_exp"] = la.A_exp
                fl["dt_bias"] = la.dt_bias
                fl["norm_w"] = la.norm.weight
                fl["norm_eps"] = la.norm.variance_epsilon
                fl["out_proj_w"] = la.out_proj.weight
            else:
                fa = l.self_attn
                fl["q_proj_w"] = fa.q_proj.weight
                fl["k_proj_w"] = fa.k_proj.weight
                fl["v_proj_w"] = fa.v_proj.weight
                fl["o_proj_w"] = fa.o_proj.weight
                fl["q_norm_w"] = fa.q_norm.weight
                fl["k_norm_w"] = fa.k_norm.weight
                fl["q_norm_eps"] = fa.q_norm.eps
                fl["k_norm_eps"] = fa.k_norm.eps
                fl["scaling"] = fa.scaling
            self.flat_layers.append(fl)

    def decode_token(self, token_id, past_kv):
        # 1. Embedding direto
        hidden_states = self.embed_tokens(token_id) # (1, 1, 2048)
        
        # 2. RoPE position embeddings (3, 1, 1) para MRoPE
        seq_pos = past_kv.get_seq_length()
        position_ids = torch.full((3, 1, 1), seq_pos, device="cuda", dtype=torch.long)
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        position_embeddings = (cos, sin)

        # 3. Loop linearizado monolítico sobre as 40 camadas
        for fl in self.flat_layers:
            residual = hidden_states
            idx = fl["idx"]

            # Input LayerNorm
            normed_h = rmsnorm_zero_centered(hidden_states, fl["input_ln_w"], fl["input_ln_eps"])

            # Token Mixer
            if fl["block_type"] == "linear_attention":
                fused_out = F.linear(normed_h, fl["W_in_fused"])
                mixed_qkv = fused_out[:, :, :8192].transpose(1, 2)
                z = fused_out[:, :, 8192:12288].reshape(1, 1, -1, 128)
                b = fused_out[:, :, 12288:12320]
                a = fused_out[:, :, 12320:12352]

                conv_state = past_kv.layers[idx].conv_states[0]
                mixed_qkv = triton_conv1d_update(mixed_qkv, conv_state, fl["conv1d_w"], fl["conv1d_b"])
                mixed_qkv = mixed_qkv.transpose(1, 2)

                q = mixed_qkv[:, :, :2048].reshape(1, 1, 16, 128)
                k = mixed_qkv[:, :, 2048:4096].reshape(1, 1, 16, 128)
                v = mixed_qkv[:, :, 4096:].reshape(1, 1, 32, 128)

                beta, g = fused_delta_gates(b, a, fl["A_exp"], fl["dt_bias"])
                q = q.repeat_interleave(2, dim=2)
                k = k.repeat_interleave(2, dim=2)

                recurrent_state = past_kv.layers[idx].recurrent_states[0]
                core_attn_out, last_recurrent_state = qm.torch_recurrent_gated_delta_rule(
                    q, k, v, g=g, beta=beta, initial_state=recurrent_state,
                    output_final_state=True, use_qk_l2norm_in_kernel=True
                )
                past_kv.update_recurrent_state(last_recurrent_state, idx)

                core_attn_out = triton_rmsnorm_gated(
                    core_attn_out.reshape(-1, 128), z.reshape(-1, 128),
                    fl["norm_w"], fl["norm_eps"]
                )
                attn_out = F.linear(core_attn_out.reshape(1, 1, -1), fl["out_proj_w"])
            else:
                # Full Attention
                q_raw, gate = torch.chunk(F.linear(normed_h, fl["q_proj_w"]).view(1, 1, -1, 512), 2, dim=-1)
                gate = gate.reshape(1, 1, -1)

                q = triton_fused_rmsnorm_rope_qwen(
                    q_raw.view(1, 1, 16, 256), fl["q_norm_w"], cos, sin, eps=fl["q_norm_eps"]
                )
                k_raw = F.linear(normed_h, fl["k_proj_w"]).view(1, 1, 2, 256)
                k = triton_fused_rmsnorm_rope_qwen(
                    k_raw, fl["k_norm_w"], cos, sin, eps=fl["k_norm_eps"]
                )
                v = F.linear(normed_h, fl["v_proj_w"]).view(1, 1, 2, 256).transpose(1, 2)

                k, v = past_kv.update(k, v, idx)
                k_rep = qm.repeat_kv(k, 8)
                v_rep = qm.repeat_kv(v, 8)

                scores = torch.matmul(q, k_rep.transpose(2, 3)) * fl["scaling"]
                probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
                attn_out = torch.matmul(probs, v_rep).transpose(1, 2).reshape(1, 1, -1)
                attn_out = attn_out * torch.sigmoid(gate)
                attn_out = F.linear(attn_out, fl["o_proj_w"])

            hidden_states = residual + attn_out

            # Post Attention LayerNorm
            residual = hidden_states
            normed_post = rmsnorm_zero_centered(hidden_states, fl["post_ln_w"], fl["post_ln_eps"])

            # MoE Block
            normed_2d = normed_post.view(1, -1)
            r_logits = F.linear(normed_2d, fl["router_w"], fl["router_bias"])
            r_probs = F.softmax(r_logits, dim=-1, dtype=torch.float32)
            topk_w, topk_idx = torch.topk(r_probs, fl["top_k"], dim=-1)
            topk_w = (topk_w / topk_w.sum(dim=-1, keepdim=True)).to(hidden_states.dtype)

            # MoE Experts
            moe_out = fl["experts"](normed_2d, topk_idx, topk_w).view(1, 1, -1)
            hidden_states = residual + moe_out

        # Final Norm + LM Head
        hidden_states = rmsnorm_zero_centered(hidden_states, self.final_norm_w, self.final_norm_eps)
        logits = F.linear(hidden_states, self.lm_head_w)
        return logits

flat_decoder = FlatDecoder(engine)

# Teste de equivalência numérica e velocidade
prompt = "A inteligência artificial avançada permite arquiteturas neurais altamente eficientes com"
prompt_ids = engine.tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()

with torch.no_grad():
    out = engine.model(input_ids=prompt_ids, use_cache=True)
    past_kv_hf = out.past_key_values
    next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()

curr_hf = torch.tensor([[next_tok]], device="cuda")
curr_flat = torch.tensor([[next_tok]], device="cuda")
pos_id = torch.tensor([[prompt_ids.shape[1]]], device="cuda")

# 1 passo com HuggingFace
torch.cuda.synchronize()
with torch.no_grad():
    out_hf = engine.model(input_ids=curr_hf, past_key_values=past_kv_hf, use_cache=True)
    l_hf = out_hf.logits[:, -1, :]

# 1 passo com FlatDecoder no mesmo estado
with torch.no_grad():
    l_flat = flat_decoder.decode_token(curr_flat, past_kv_hf)[:, -1, :]

diff = (l_hf - l_flat).abs().max().item()
cos = F.cosine_similarity(l_hf.float(), l_flat.float(), dim=-1).item()
print("\n--- COMPARAÇÃO NUMÉRICA: HUGGINGFACE vs FLAT DECODER ---")
print(f"Max Absolute Diff: {diff:.6e}")
print(f"Cosine Similarity: {cos * 100:.6f}%")
print(f"Next Token HF:     {torch.argmax(l_hf).item()} ('{engine.tokenizer.decode([torch.argmax(l_hf).item()])}')")
print(f"Next Token Flat:   {torch.argmax(l_flat).item()} ('{engine.tokenizer.decode([torch.argmax(l_flat).item()])}')")

# Benchmark comparativo de 30 tokens
print("\n--- BENCHMARK DE VELOCIDADE (30 TOKENS) ---")
N_RUNS = 30

# Warmup FlatDecoder
for _ in range(5):
    with torch.no_grad():
        tok = torch.tensor([[next_tok]], device="cuda")
        l = flat_decoder.decode_token(tok, past_kv_hf)
torch.cuda.synchronize()

ev_start = torch.cuda.Event(enable_timing=True)
ev_end = torch.cuda.Event(enable_timing=True)

t0_wall = time.perf_counter()
ev_start.record()
with torch.no_grad():
    for _ in range(N_RUNS):
        tok = torch.tensor([[next_tok]], device="cuda")
        l = flat_decoder.decode_token(tok, past_kv_hf)
        next_tok = torch.argmax(l[:, -1, :], dim=-1).item()
ev_end.record()
torch.cuda.synchronize()
t1_wall = time.perf_counter()

gpu_ms = ev_start.elapsed_time(ev_end)
wall_ms = (t1_wall - t0_wall) * 1000.0

print(f"FlatDecoder GPU Time:  {gpu_ms / N_RUNS:.2f} ms/token ({N_RUNS / (gpu_ms / 1000.0):.2f} tok/s)")
print(f"FlatDecoder Wall Time: {wall_ms / N_RUNS:.2f} ms/token ({N_RUNS / (wall_ms / 1000.0):.2f} tok/s)")
print(f"Referência HF Anterior: ~147.11 ms/token (6.80 tok/s)")
print(f"Ganho Líquido Direto:  {147.11 - (gpu_ms / N_RUNS):+.2f} ms/token")
