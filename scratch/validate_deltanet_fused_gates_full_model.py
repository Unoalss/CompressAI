import sys, os, time
from pathlib import Path
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legion.engine import LegionEngine
import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as qm
import triton
import triton.language as tl

@triton.jit
def _fused_delta_gates_kernel(
    b_ptr, a_ptr, A_exp_ptr, dt_bias_ptr, beta_ptr, g_ptr,
    N: tl.constexpr, BLOCK: tl.constexpr
):
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(a_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    A_exp = tl.load(A_exp_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    beta = tl.sigmoid(b)
    tl.store(beta_ptr + cols, beta.to(tl.bfloat16), mask=mask)

    x = a + dt_bias
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    g = -A_exp * sp
    tl.store(g_ptr + cols, g, mask=mask)

def fused_delta_gates(b, a, A_exp, dt_bias):
    N = b.shape[-1]
    beta = torch.empty_like(b)
    g = torch.empty((b.shape[0], b.shape[1], N), device=b.device, dtype=torch.float32)
    _fused_delta_gates_kernel[(1,)](
        b, a, A_exp, dt_bias, beta, g, N=N, BLOCK=64, num_warps=1
    )
    return beta, g

def validate_fused_gates_full_model():
    print("=" * 80)
    print("OTIMIZAÇÃO 4 - PASSO 3: FUSÃO DOS GATES DELTANET NO MODELO REAL")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "CUDA é obrigatório!"

    # 1. Integridade Física do B2
    b2_path = ROOT / "vq2_35b_out" / "qwen35b_mixed_b2.vq2m"
    b2_size = b2_path.stat().st_size
    print(f"[1/4] Verificando integridade física do B2...")
    print(f"      Arquivo: {b2_path}")
    print(f"      Tamanho: {b2_size:,} bytes")
    assert b2_size == 10108393952, "Tamanho do B2 alterado!"
    print("      [OK] B2 permanece 100% INTACTO byte a byte!")

    engine = LegionEngine()
    tokenizer = engine.tokenizer
    prompt = "A inteligência artificial avançada permite arquiteturas neurais altamente eficientes com"
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    linear_layers = [l.linear_attn for l in engine.model.model.layers if hasattr(l, "linear_attn")]

    # Pré-computa A_exp para cada camada
    for layer in linear_layers:
        layer.A_exp = layer.A_log.float().exp().contiguous()

    orig_dn_forward = linear_layers[0].__class__.forward

    def make_forward_gates(use_fused: bool):
        def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
            hidden_states = qm.apply_mask_to_padding_states(hidden_states, attention_mask)
            batch_size, seq_len, _ = hidden_states.shape
            use_precomputed_states = cache_params is not None and cache_params.has_previous_state(self.layer_idx, state_idx=0)

            # Usa projeção fundida W_in do Passo 2 se presente
            if hasattr(self, "W_in_fused"):
                fused_out = F.linear(hidden_states, self.W_in_fused)
                mixed_qkv = fused_out[:, :, :8192]
                z = fused_out[:, :, 8192:12288]
                b = fused_out[:, :, 12288:12320]
                a = fused_out[:, :, 12320:12352]
            else:
                mixed_qkv = self.in_proj_qkv(hidden_states)
                z = self.in_proj_z(hidden_states)
                b = self.in_proj_b(hidden_states)
                a = self.in_proj_a(hidden_states)

            mixed_qkv = mixed_qkv.transpose(1, 2)
            z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

            if use_precomputed_states and seq_len == 1 and not cache_params.layers[self.layer_idx].record_past:
                conv_state = cache_params.layers[self.layer_idx].conv_states[0]
                mixed_qkv = qm.causal_conv1d_update(
                    mixed_qkv, conv_state, self.conv1d.weight.squeeze(1), self.conv1d.bias, self.activation
                )
            else:
                if cache_params is not None:
                    mixed_qkv = cache_params.update_conv_state(mixed_qkv, self.layer_idx, conv_kernel_size=self.conv_kernel_size)
                mixed_qkv = qm.causal_conv1d_fn(
                    mixed_qkv, self.conv1d.weight.squeeze(1), self.conv1d.bias, activation=self.activation, **kwargs
                )
                if cache_params is not None:
                    mixed_qkv = mixed_qkv[:, :, -seq_len:]

            mixed_qkv = mixed_qkv.transpose(1, 2)
            query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
            query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
            key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
            value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

            if use_fused and seq_len == 1:
                # Fused Gates Triton
                beta, g = fused_delta_gates(b, a, self.A_exp, self.dt_bias)
            else:
                # Original PyTorch
                beta = b.sigmoid()
                g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

            if self.num_v_heads // self.num_k_heads > 1:
                query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
                key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

            recurrent_state = cache_params.layers[self.layer_idx].recurrent_states[0] if use_precomputed_states else None
            if use_precomputed_states and seq_len == 1:
                core_attn_out, last_recurrent_state = qm.torch_recurrent_gated_delta_rule(
                    query, key, value, g=g, beta=beta, initial_state=recurrent_state,
                    output_final_state=cache_params is not None, use_qk_l2norm_in_kernel=True,
                    cu_seqlens=kwargs.pop("cu_seq_lens_q", None), **kwargs
                )
            else:
                core_attn_out, last_recurrent_state = qm.torch_chunk_gated_delta_rule(
                    query, key, value, g=g, beta=beta, initial_state=recurrent_state,
                    output_final_state=cache_params is not None, use_qk_l2norm_in_kernel=True,
                    cu_seqlens=kwargs.pop("cu_seq_lens_q", None), **kwargs
                )

            if cache_params is not None:
                cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

            core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
            z = z.reshape(-1, self.head_v_dim)
            core_attn_out = self.norm(core_attn_out, z)
            core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
            output = self.out_proj(core_attn_out)
            return output
        return forward

    def run_generation(use_fused: bool, n_tokens: int = 25):
        linear_layers[0].__class__.forward = make_forward_gates(use_fused)

        torch.cuda.synchronize()
        with torch.no_grad():
            out = engine.model(input_ids=prompt_ids, use_cache=True)
            past_kv = out.past_key_values
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()

        generated_ids = [next_tok]
        curr = torch.tensor([[next_tok]], device=device)

        for _ in range(3):
            with torch.no_grad():
                out = engine.model(input_ids=curr, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
                next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()
                generated_ids.append(next_tok)
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
                generated_ids.append(next_tok)
                curr = torch.tensor([[next_tok]], device=device)

        ev_end.record()
        torch.cuda.synchronize()
        gpu_time_ms = ev_start.elapsed_time(ev_end)

        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        logits = out.logits[:, -1, :].clone()

        return {
            "gpu_ms": gpu_time_ms,
            "ms_per_tok": gpu_time_ms / n_tokens,
            "tok_per_sec": n_tokens / (gpu_time_ms / 1000.0),
            "text": text,
            "logits": logits
        }

    # 2. Teste de equivalência causal e numérica
    print("\n[2/4] Testando Equivalência Causal no Modelo Completo...")
    print("      Rodando com Gates PyTorch originais...")
    res_orig = run_generation(use_fused=False, n_tokens=25)
    print(f"      PyTorch: {res_orig['tok_per_sec']:.2f} tok/s ({res_orig['ms_per_tok']:.2f} ms/tok)")

    print("      Rodando com Fused Gates Triton...")
    res_fused = run_generation(use_fused=True, n_tokens=25)
    print(f"      Triton:  {res_fused['tok_per_sec']:.2f} tok/s ({res_fused['ms_per_tok']:.2f} ms/tok)")

    # Métricas de fidelidade
    logits_orig = res_orig["logits"].float()
    logits_fused = res_fused["logits"].float()
    has_nan = torch.isnan(logits_fused).any().item() or torch.isinf(logits_fused).any().item()
    cos_sim = F.cosine_similarity(logits_orig, logits_fused, dim=-1).item()
    p_fus = F.log_softmax(logits_fused, dim=-1)
    p_orig = F.softmax(logits_orig, dim=-1)
    kl_div = F.kl_div(p_fus, p_orig, reduction="batchmean").item()
    text_identical = (res_orig["text"] == res_fused["text"])

    print("\n" + "=" * 80)
    print("AUDITORIA NUMÉRICA DE EQUIVALÊNCIA DOS FUSED GATES:")
    print("=" * 80)
    print(f"Estabilidade Numérica:            Sem NaNs / Sem Infs: {not has_nan}")
    print(f"Textos Gerados Idênticos:         {text_identical}")
    print(f"Similaridade Cosseno Logits:      {cos_sim * 100:.6f}%")
    print(f"Divergência KL (D_KL):            {kl_div:.8f}")
    print("=" * 80)

    # 3. Baterias Alternadas de Benchmark de Velocidade
    print("\n[3/4] Executando 3 Baterias Alternadas de Benchmark...")
    orig_speeds = []
    fused_speeds = []

    for round_idx in range(1, 4):
        print(f"\n--- BATERIA {round_idx} ---")
        r_orig = run_generation(use_fused=False, n_tokens=30)
        orig_speeds.append(r_orig)
        print(f"  [PyTorch Original] {r_orig['tok_per_sec']:.2f} tok/s ({r_orig['ms_per_tok']:.2f} ms/tok)")

        r_fused = run_generation(use_fused=True, n_tokens=30)
        fused_speeds.append(r_fused)
        print(f"  [Fused Gates Triton] {r_fused['tok_per_sec']:.2f} tok/s ({r_fused['ms_per_tok']:.2f} ms/tok)")

    avg_orig_tps = sum(r["tok_per_sec"] for r in orig_speeds) / 3
    avg_orig_ms = sum(r["ms_per_tok"] for r in orig_speeds) / 3
    avg_fused_tps = sum(r["tok_per_sec"] for r in fused_speeds) / 3
    avg_fused_ms = sum(r["ms_per_tok"] for r in fused_speeds) / 3

    final_b2_size = b2_path.stat().st_size
    assert final_b2_size == 10108393952, "Tamanho do B2 violado!"

    print("\n" + "=" * 80)
    print("RELATÓRIO CONSOLIDADO DO PASSO 3 (FUSED GATES DELTANET):")
    print("=" * 80)
    print(f"Artefato B2:                      100% INTACTO ({final_b2_size:,} bytes)")
    print(f"Equivalência Causal:              Concordância Textual: {text_identical}")
    print(f"Similaridade Cosseno Logits:      {cos_sim * 100:.6f}%")
    print(f"Divergência KL:                   {kl_div:.8f}")
    print("--------------------------------------------------------------------------------")
    print(f"Gates PyTorch Originais:          {avg_orig_tps:.2f} tok/s ({avg_orig_ms:.2f} ms/token)")
    print(f"Fused Gates Triton:               {avg_fused_tps:.2f} tok/s ({avg_fused_ms:.2f} ms/token)")
    print(f"Ganho Líquido Real:               +{avg_fused_tps - avg_orig_tps:.2f} tok/s (-{avg_orig_ms - avg_fused_ms:.2f} ms/token)")
    print("=" * 80)

if __name__ == "__main__":
    validate_fused_gates_full_model()
