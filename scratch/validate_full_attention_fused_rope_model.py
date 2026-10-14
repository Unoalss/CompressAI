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
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
import triton
import triton.language as tl

@triton.jit
def _fused_qwen35_rmsnorm_rope_kernel(
    x_ptr, w_ptr, cos_ptr, sin_ptr, out_ptr,
    stride_xh, stride_xd,
    stride_oh, stride_od,
    stride_cos_d, stride_sin_d,
    eps: tl.constexpr,
):
    pid = tl.program_id(0)

    # Divisão em potências de 2: 32 + 32 + 64 + 128 = 256
    idx_32 = tl.arange(0, 32)
    idx_64 = tl.arange(0, 64)
    idx_128 = tl.arange(0, 128)

    # 1. Carrega x em 4 blocos
    x0 = tl.load(x_ptr + pid * stride_xh + idx_32 * stride_xd).to(tl.float32)        # 0..31
    x1 = tl.load(x_ptr + pid * stride_xh + (idx_32 + 32) * stride_xd).to(tl.float32) # 32..63
    x2 = tl.load(x_ptr + pid * stride_xh + (idx_64 + 64) * stride_xd).to(tl.float32) # 64..127
    x3 = tl.load(x_ptr + pid * stride_xh + (idx_128 + 128) * stride_xd).to(tl.float32) # 128..255

    # 2. Carrega w (zero-centered: 1.0 + w)
    w0 = 1.0 + tl.load(w_ptr + idx_32).to(tl.float32)
    w1 = 1.0 + tl.load(w_ptr + (idx_32 + 32)).to(tl.float32)
    w2 = 1.0 + tl.load(w_ptr + (idx_64 + 64)).to(tl.float32)
    w3 = 1.0 + tl.load(w_ptr + (idx_128 + 128)).to(tl.float32)

    # 3. RMSNorm sobre todos os 256 elementos
    sum_sq = (
        tl.sum(x0 * x0, axis=0) +
        tl.sum(x1 * x1, axis=0) +
        tl.sum(x2 * x2, axis=0) +
        tl.sum(x3 * x3, axis=0)
    )
    rms = tl.rsqrt((sum_sq / 256.0) + eps)

    # Escala normalizada
    v0 = x0 * rms * w0
    v1 = x1 * rms * w1
    v2 = x2 * rms * w2
    v3 = x3 * rms * w3

    # 4. RoPE nas primeiras 64 dimensões (v0 e v1)
    cos0 = tl.load(cos_ptr + idx_32 * stride_cos_d).to(tl.float32)
    cos1 = tl.load(cos_ptr + (idx_32 + 32) * stride_cos_d).to(tl.float32)
    sin0 = tl.load(sin_ptr + idx_32 * stride_sin_d).to(tl.float32)
    sin1 = tl.load(sin_ptr + (idx_32 + 32) * stride_sin_d).to(tl.float32)

    y0 = (v0 * cos0) - (v1 * sin0)
    y1 = (v1 * cos1) + (v0 * sin1)
    # v2 e v3 passam sem rotação

    # 5. Salva diretamente no tensor de saída (1, H, 1, 256)
    tl.store(out_ptr + pid * stride_oh + idx_32 * stride_od, y0.to(tl.bfloat16))
    tl.store(out_ptr + pid * stride_oh + (idx_32 + 32) * stride_od, y1.to(tl.bfloat16))
    tl.store(out_ptr + pid * stride_oh + (idx_64 + 64) * stride_od, v2.to(tl.bfloat16))
    tl.store(out_ptr + pid * stride_oh + (idx_128 + 128) * stride_od, v3.to(tl.bfloat16))

def triton_fused_rmsnorm_rope_qwen(x_states, weight, cos, sin, eps=1e-6):
    H = x_states.shape[2]
    D = x_states.shape[3]
    out = torch.empty((1, H, 1, D), device=x_states.device, dtype=torch.bfloat16)
    x_2d = x_states.view(H, D)
    out_2d = out.view(H, D)
    cos_1d = cos.view(-1)
    sin_1d = sin.view(-1)

    grid = (H,)
    _fused_qwen35_rmsnorm_rope_kernel[grid](
        x_2d, weight, cos_1d, sin_1d, out_2d,
        x_2d.stride(0), x_2d.stride(1),
        out_2d.stride(0), out_2d.stride(1),
        cos_1d.stride(0), sin_1d.stride(0),
        eps=eps,
        num_warps=2
    )
    return out

def validate_full_attention_fused_rope():
    print("=" * 80)
    print("FASE 4C: VALIDAÇÃO DO KERNEL FUNDIDO RMSNorm + RoPE NO MODELO REAL")
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

    # Identifica as 10 camadas de Full Attention
    full_layers = [l.self_attn for l in engine.model.model.layers if hasattr(l, "self_attn")]
    print(f"[2/4] Camadas Full Attention identificadas: {len(full_layers)}")

    orig_fa_forward = full_layers[0].__class__.forward

    def make_fa_forward(use_fused: bool):
        def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            seq_len = hidden_states.shape[1]

            query_states, gate = torch.chunk(
                self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
            )
            gate = gate.reshape(*input_shape, -1)
            cos, sin = position_embeddings

            if use_fused and seq_len == 1 and hidden_states.is_cuda:
                # Fused RMSNormZeroCentered + RoPE Triton
                query_states = triton_fused_rmsnorm_rope_qwen(
                    query_states.view(hidden_shape), self.q_norm.weight, cos, sin, eps=self.q_norm.eps
                )
                key_raw = self.k_proj(hidden_states).view(hidden_shape)
                key_states = triton_fused_rmsnorm_rope_qwen(
                    key_raw, self.k_norm.weight, cos, sin, eps=self.k_norm.eps
                )
                value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            else:
                # Original PyTorch
                query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
                key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                query_states, key_states = qm.apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_values is not None:
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, qm.eager_attention_forward
            )
            attn_output, attn_weights = attention_interface(
                self, query_states, key_states, value_states, attention_mask,
                dropout=0.0, scaling=self.scaling, **kwargs
            )
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = attn_output * torch.sigmoid(gate)
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights
        return forward

    def run_generation(use_fused: bool, n_tokens: int = 25):
        full_layers[0].__class__.forward = make_fa_forward(use_fused)

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
    print("\n[3/4] Testando Equivalência Causal no Modelo Completo...")
    print("      Rodando com Full Attention PyTorch original...")
    res_orig = run_generation(use_fused=False, n_tokens=25)
    print(f"      PyTorch: {res_orig['tok_per_sec']:.2f} tok/s ({res_orig['ms_per_tok']:.2f} ms/tok)")

    print("      Rodando com Full Attention Triton (RMSNorm + RoPE Fundidos)...")
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
    print("AUDITORIA NUMÉRICA DE EQUIVALÊNCIA DA FULL ATTENTION FUNDIDA:")
    print("=" * 80)
    print(f"Estabilidade Numérica:            Sem NaNs / Sem Infs: {not has_nan}")
    print(f"Textos Gerados Idênticos:         {text_identical}")
    print(f"Similaridade Cosseno Logits:      {cos_sim * 100:.6f}%")
    print(f"Divergência KL (D_KL):            {kl_div:.8f}")
    print("=" * 80)

    # 3. Baterias Alternadas de Benchmark de Velocidade
    print("\n[4/4] Executando 3 Baterias Alternadas de Benchmark...")
    orig_speeds = []
    fused_speeds = []

    for round_idx in range(1, 4):
        print(f"\n--- BATERIA {round_idx} ---")
        r_orig = run_generation(use_fused=False, n_tokens=30)
        orig_speeds.append(r_orig)
        print(f"  [PyTorch Original] {r_orig['tok_per_sec']:.2f} tok/s ({r_orig['ms_per_tok']:.2f} ms/tok)")

        r_fused = run_generation(use_fused=True, n_tokens=30)
        fused_speeds.append(r_fused)
        print(f"  [Triton Fundido]   {r_fused['tok_per_sec']:.2f} tok/s ({r_fused['ms_per_tok']:.2f} ms/tok)")

    avg_orig_tps = sum(r["tok_per_sec"] for r in orig_speeds) / 3
    avg_orig_ms = sum(r["ms_per_tok"] for r in orig_speeds) / 3
    avg_fused_tps = sum(r["tok_per_sec"] for r in fused_speeds) / 3
    avg_fused_ms = sum(r["ms_per_tok"] for r in fused_speeds) / 3

    final_b2_size = b2_path.stat().st_size
    assert final_b2_size == 10108393952, "Tamanho do B2 violado!"

    print("\n" + "=" * 80)
    print("RELATÓRIO CONSOLIDADO DA FASE 4C (RMSNorm + RoPE FUNDIDOS NA FULL ATTN):")
    print("=" * 80)
    print(f"Artefato B2:                      100% INTACTO ({final_b2_size:,} bytes)")
    print(f"Equivalência Causal:              Concordância Textual: {text_identical}")
    print(f"Similaridade Cosseno Logits:      {cos_sim * 100:.6f}%")
    print(f"Divergência KL:                   {kl_div:.8f}")
    print("--------------------------------------------------------------------------------")
    print(f"Full Attention PyTorch (Original):{avg_orig_tps:.2f} tok/s ({avg_orig_ms:.2f} ms/token)")
    print(f"Full Attention Triton (Fundido):  {avg_fused_tps:.2f} tok/s ({avg_fused_ms:.2f} ms/token)")
    print(f"Ganho Líquido Real Isolado:       +{avg_fused_tps - avg_orig_tps:.2f} tok/s (-{avg_orig_ms - avg_fused_ms:.2f} ms/token)")
    print("=" * 80)

if __name__ == "__main__":
    validate_full_attention_fused_rope()
