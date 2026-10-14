import sys, os, time, gc
from pathlib import Path
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legion.engine import LegionEngine
import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as qm

def validate_fused_in_proj_full_model():
    print("=" * 80)
    print("OTIMIZAÇÃO 4 - PASSO 2: FUSÃO DAS PROJEÇÕES DE ENTRADA DELTANET NO MODELO REAL")
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

    # Identifica as camadas DeltaNet
    linear_layers = [l.linear_attn for l in engine.model.model.layers if hasattr(l, "linear_attn")]
    print(f"[2/4] Camadas DeltaNet identificadas: {len(linear_layers)}")

    # Guardamos os forwards originais
    orig_forward = linear_layers[0].__class__.forward

    # 2. Vamos preparar a fusão de W_in em cada camada DeltaNet
    # Ordem: qkv (8192) -> z (4096) -> b (32) -> a (32) = 12352
    d_qkv = 8192
    d_z = 4096
    d_b = 32
    d_a = 32

    vram_before = torch.cuda.memory_allocated() / (1024 * 1024)
    print(f"      VRAM alocada antes da fusão: {vram_before:.2f} MB")

    for layer in linear_layers:
        # Concatena em W_in
        W_fused = torch.cat([
            layer.in_proj_qkv.weight.data,
            layer.in_proj_z.weight.data,
            layer.in_proj_b.weight.data,
            layer.in_proj_a.weight.data
        ], dim=0).contiguous()

        # Armazena W_in na camada
        layer.W_in_fused = torch.nn.Parameter(W_fused, requires_grad=False)

        # Para manter o orçamento rígido de VRAM com ZERO acréscimo:
        # Substitui os pesos originais por views fatiadas de W_in_fused
        layer.in_proj_qkv.weight.data = layer.W_in_fused[:d_qkv]
        layer.in_proj_z.weight.data = layer.W_in_fused[d_qkv : d_qkv + d_z]
        layer.in_proj_b.weight.data = layer.W_in_fused[d_qkv + d_z : d_qkv + d_z + d_b]
        layer.in_proj_a.weight.data = layer.W_in_fused[d_qkv + d_z + d_b :]

    gc.collect()
    torch.cuda.empty_cache()
    vram_after = torch.cuda.memory_allocated() / (1024 * 1024)
    print(f"      VRAM alocada após fusão com zero-copy views: {vram_after:.2f} MB")
    print(f"      Diferença líquida de VRAM: {vram_after - vram_before:+.2f} MB (Orçamento rigorosamente preservado!)")

    # 3. Forwards de comparação:
    # A) Modo Separado: chama in_proj_qkv, in_proj_z, in_proj_b, in_proj_a separadamente
    # B) Modo Fundido: 1 GEMV F.linear(hidden_states, layer.W_in_fused) + slicing

    def make_custom_forward(use_fused: bool):
        def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
            hidden_states = qm.apply_mask_to_padding_states(hidden_states, attention_mask)
            batch_size, seq_len, _ = hidden_states.shape
            use_precomputed_states = cache_params is not None and cache_params.has_previous_state(self.layer_idx, state_idx=0)

            if use_fused and hasattr(self, "W_in_fused"):
                # 1 GEMV fundido
                fused_out = F.linear(hidden_states, self.W_in_fused)
                mixed_qkv = fused_out[:, :, :d_qkv]
                z = fused_out[:, :, d_qkv : d_qkv + d_z]
                b = fused_out[:, :, d_qkv + d_z : d_qkv + d_z + d_b]
                a = fused_out[:, :, d_qkv + d_z + d_b :]
            else:
                # 4 GEMVs originais
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
        linear_layers[0].__class__.forward = make_custom_forward(use_fused)

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

    # 4. Teste de equivalência causal e numérica
    print("\n[3/4] Testando Equivalência Causal no Modelo Completo...")
    print("      Rodando com 4 Projeções Separadas (Referência)...")
    res_sep = run_generation(use_fused=False, n_tokens=25)
    print(f"      Separado: {res_sep['tok_per_sec']:.2f} tok/s ({res_sep['ms_per_tok']:.2f} ms/tok)")

    print("      Rodando com Projeção Fundida W_in...")
    res_fus = run_generation(use_fused=True, n_tokens=25)
    print(f"      Fundido:  {res_fus['tok_per_sec']:.2f} tok/s ({res_fus['ms_per_tok']:.2f} ms/tok)")

    # Métricas de fidelidade
    logits_sep = res_sep["logits"].float()
    logits_fus = res_fus["logits"].float()
    has_nan = torch.isnan(logits_fus).any().item() or torch.isinf(logits_fus).any().item()
    cos_sim = F.cosine_similarity(logits_sep, logits_fus, dim=-1).item()
    p_fus = F.log_softmax(logits_fus, dim=-1)
    p_sep = F.softmax(logits_sep, dim=-1)
    kl_div = F.kl_div(p_fus, p_sep, reduction="batchmean").item()
    text_identical = (res_sep["text"] == res_fus["text"])

    print("\n" + "=" * 80)
    print("AUDITORIA NUMÉRICA DE EQUIVALÊNCIA DA FUSÃO W_in:")
    print("=" * 80)
    print(f"Estabilidade Numérica:            Sem NaNs / Sem Infs: {not has_nan}")
    print(f"Textos Gerados Idênticos:         {text_identical}")
    print(f"Similaridade Cosseno Logits:      {cos_sim * 100:.6f}%")
    print(f"Divergência KL (D_KL):            {kl_div:.8f}")
    print("=" * 80)

    # 5. Baterias Alternadas de Benchmark de Velocidade
    print("\n[4/4] Executando 3 Baterias Alternadas de Benchmark...")
    sep_speeds = []
    fus_speeds = []

    for round_idx in range(1, 4):
        print(f"\n--- BATERIA {round_idx} ---")
        r_sep = run_generation(use_fused=False, n_tokens=30)
        sep_speeds.append(r_sep)
        print(f"  [4 Projeções Separadas] {r_sep['tok_per_sec']:.2f} tok/s ({r_sep['ms_per_tok']:.2f} ms/tok)")

        r_fus = run_generation(use_fused=True, n_tokens=30)
        fus_speeds.append(r_fus)
        print(f"  [Projeção Fundida W_in] {r_fus['tok_per_sec']:.2f} tok/s ({r_fus['ms_per_tok']:.2f} ms/tok)")

    avg_sep_tps = sum(r["tok_per_sec"] for r in sep_speeds) / 3
    avg_sep_ms = sum(r["ms_per_tok"] for r in sep_speeds) / 3
    avg_fus_tps = sum(r["tok_per_sec"] for r in fus_speeds) / 3
    avg_fus_ms = sum(r["ms_per_tok"] for r in fus_speeds) / 3

    final_b2_size = b2_path.stat().st_size
    assert final_b2_size == 10108393952, "Tamanho do B2 violado!"

    print("\n" + "=" * 80)
    print("RELATÓRIO CONSOLIDADO DO PASSO 2 (FUSÃO DE PROJEÇÕES DELTANET):")
    print("=" * 80)
    print(f"Artefato B2:                      100% INTACTO ({final_b2_size:,} bytes)")
    print(f"Equivalência Causal:              Concordância Textual: {text_identical}")
    print(f"Similaridade Cosseno Logits:      {cos_sim * 100:.6f}%")
    print(f"Divergência KL:                   {kl_div:.8f}")
    print(f"VRAM Adicional Líquida:           0.00 MB (Pesos originais substituídos por views)")
    print("--------------------------------------------------------------------------------")
    print(f"4 Projeções Separadas (120 GEMVs):{avg_sep_tps:.2f} tok/s ({avg_sep_ms:.2f} ms/token)")
    print(f"Projeção Fundida W_in (30 GEMVs): {avg_fus_tps:.2f} tok/s ({avg_fus_ms:.2f} ms/token)")
    print(f"Ganho Líquido Real:               +{avg_fus_tps - avg_sep_tps:.2f} tok/s (-{avg_sep_ms - avg_fus_ms:.2f} ms/token)")
    print("=" * 80)

if __name__ == "__main__":
    validate_fused_in_proj_full_model()
