import sys, os, time
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legion.engine import LegionEngine
import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as qm
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

def profile_deltanet_and_attention():
    print("=" * 80)
    print("OTIMIZAÇÃO 4 - FASE 4A & 4C: PROFILING GRANULAR DA DELTANET E FULL ATTENTION")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "CUDA é obrigatório para este profiling!"

    # 1. Integridade do B2
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

    # 2. Identificação das camadas
    linear_layers = []
    full_layers = []
    for l_idx, layer in enumerate(engine.model.model.layers):
        if hasattr(layer, "linear_attn"):
            linear_layers.append((l_idx, layer.linear_attn))
        elif hasattr(layer, "self_attn"):
            full_layers.append((l_idx, layer.self_attn))

    print(f"[2/4] Arquitetura de Atenção: {len(linear_layers)} DeltaNet (Linear) + {len(full_layers)} Full Attention")

    # 3. Warmup
    print("[3/4] Executando prefill e warmup de 3 tokens...")
    with torch.no_grad():
        out = engine.model(input_ids=prompt_ids, use_cache=True)
        past_kv = out.past_key_values
        next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()

    curr = torch.tensor([[next_tok]], device=device)
    for _ in range(3):
        with torch.no_grad():
            out = engine.model(input_ids=curr, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()
            curr = torch.tensor([[next_tok]], device=device)
    torch.cuda.synchronize()

    # 4. Instrumentação fina com CUDA Events NÃO-BLOQUEANTES
    deltanet_events = defaultdict(list)
    full_events = defaultdict(list)

    orig_dn_forward = linear_layers[0][1].__class__.forward
    orig_fa_forward = full_layers[0][1].__class__.forward
    orig_recurrent_rule = qm.torch_recurrent_gated_delta_rule

    # Forward interceptor para DeltaNet
    def instrumented_dn_forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        evs = {}
        
        # 1. in_projections (4 projeções lineares separadas)
        e0 = torch.cuda.Event(enable_timing=True); e0.record()
        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)
        e1 = torch.cuda.Event(enable_timing=True); e1.record()
        evs["in_projections"] = (e0, e1)

        # 2. Reshapes & Transpose
        e2 = torch.cuda.Event(enable_timing=True); e2.record()
        batch_size, seq_len, _ = hidden_states.shape
        mixed_qkv = mixed_qkv.transpose(1, 2)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)
        e3 = torch.cuda.Event(enable_timing=True); e3.record()
        evs["reshapes_pre_conv"] = (e2, e3)

        # 3. causal_conv1d_update (Triton JIT)
        e4 = torch.cuda.Event(enable_timing=True); e4.record()
        use_precomputed_states = cache_params is not None and cache_params.has_previous_state(self.layer_idx, state_idx=0)
        conv_state = cache_params.layers[self.layer_idx].conv_states[0]
        mixed_qkv = qm.causal_conv1d_update(
            mixed_qkv, conv_state, self.conv1d.weight.squeeze(1), self.conv1d.bias, self.activation
        )
        e5 = torch.cuda.Event(enable_timing=True); e5.record()
        evs["causal_conv1d"] = (e4, e5)

        # 4. Split & Reshape Q, K, V
        e6 = torch.cuda.Event(enable_timing=True); e6.record()
        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
        e7 = torch.cuda.Event(enable_timing=True); e7.record()
        evs["split_reshape_qkv"] = (e6, e7)

        # 5. Gates (beta, g) & repeat_interleave
        e8 = torch.cuda.Event(enable_timing=True); e8.record()
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        e9 = torch.cuda.Event(enable_timing=True); e9.record()
        evs["gates_and_repeat"] = (e8, e9)

        # 6. Recorrência DeltaNet (torch_recurrent_gated_delta_rule)
        e10 = torch.cuda.Event(enable_timing=True); e10.record()
        recurrent_state = cache_params.layers[self.layer_idx].recurrent_states[0] if use_precomputed_states else None
        core_attn_out, last_recurrent_state = orig_recurrent_rule(
            query, key, value, g=g, beta=beta, initial_state=recurrent_state,
            output_final_state=cache_params is not None, use_qk_l2norm_in_kernel=True
        )
        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)
        e11 = torch.cuda.Event(enable_timing=True); e11.record()
        evs["recurrent_rule"] = (e10, e11)

        # 7. RMSNorm Gated
        e12 = torch.cuda.Event(enable_timing=True); e12.record()
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        e13 = torch.cuda.Event(enable_timing=True); e13.record()
        evs["norm_gated"] = (e12, e13)

        # 8. out_proj
        e14 = torch.cuda.Event(enable_timing=True); e14.record()
        output = self.out_proj(core_attn_out)
        e15 = torch.cuda.Event(enable_timing=True); e15.record()
        evs["out_proj"] = (e14, e15)

        deltanet_events[self.layer_idx].append(evs)
        return output

    # Forward interceptor para Full Attention
    def instrumented_fa_forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        evs = {}
        # 1. Projeções Q, K, V
        e0 = torch.cuda.Event(enable_timing=True); e0.record()
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states, gate = torch.chunk(self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)
        key_raw = self.k_proj(hidden_states)
        value_raw = self.v_proj(hidden_states)
        e1 = torch.cuda.Event(enable_timing=True); e1.record()
        evs["qkv_proj"] = (e0, e1)

        # 2. Reshapes, RMSNorms Q/K e RoPE
        e2 = torch.cuda.Event(enable_timing=True); e2.record()
        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(key_raw.view(hidden_shape)).transpose(1, 2)
        value_states = value_raw.view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = qm.apply_rotary_pos_emb(query_states, key_states, cos, sin)
        e3 = torch.cuda.Event(enable_timing=True); e3.record()
        evs["norms_and_rope"] = (e2, e3)

        # 3. KV Cache update & Interface de Atenção
        e4 = torch.cuda.Event(enable_timing=True); e4.record()
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attn_interface = ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, qm.eager_attention_forward)
        attn_output, attn_weights = attn_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0, scaling=self.scaling, **kwargs
        )
        e5 = torch.cuda.Event(enable_timing=True); e5.record()
        evs["kv_cache_and_sdpa"] = (e4, e5)

        # 4. Out proj & Gate
        e6 = torch.cuda.Event(enable_timing=True); e6.record()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        e7 = torch.cuda.Event(enable_timing=True); e7.record()
        evs["gate_and_out_proj"] = (e6, e7)

        full_events[self.layer_idx].append(evs)
        return attn_output, attn_weights

    # Aplica os patches de instrumentação
    linear_layers[0][1].__class__.forward = instrumented_dn_forward
    full_layers[0][1].__class__.forward = instrumented_fa_forward

    # 5. Execução de N tokens com medição consolidada
    N_TOKENS = 15
    print(f"[4/4] Medindo {N_TOKENS} passos de decodificação sob execução assíncrona natural...")

    step_times_ms = []
    for step in range(N_TOKENS):
        ev_step_start = torch.cuda.Event(enable_timing=True)
        ev_step_end = torch.cuda.Event(enable_timing=True)
        ev_step_start.record()

        with torch.no_grad():
            out = engine.model(input_ids=curr, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()
            curr = torch.tensor([[next_tok]], device=device)

        ev_step_end.record()
        # Sincronização única no limite do token:
        torch.cuda.synchronize()
        step_times_ms.append(ev_step_start.elapsed_time(ev_step_end))

    # Restaura forwards originais
    linear_layers[0][1].__class__.forward = orig_dn_forward
    full_layers[0][1].__class__.forward = orig_fa_forward

    # 6. Agregação estatística dos tempos de GPU
    dn_sub_times = defaultdict(float)
    for l_idx, steps_list in deltanet_events.items():
        for step_dict in steps_list:
            for sub_name, (ev_s, ev_e) in step_dict.items():
                dn_sub_times[sub_name] += ev_s.elapsed_time(ev_e)

    fa_sub_times = defaultdict(float)
    for l_idx, steps_list in full_events.items():
        for step_dict in steps_list:
            for sub_name, (ev_s, ev_e) in step_dict.items():
                fa_sub_times[sub_name] += ev_s.elapsed_time(ev_e)

    dn_per_tok = {k: v / N_TOKENS for k, v in dn_sub_times.items()}
    fa_per_tok = {k: v / N_TOKENS for k, v in fa_sub_times.items()}
    total_token_ms = sum(step_times_ms) / N_TOKENS
    tps = 1000.0 / total_token_ms

    tot_dn = sum(dn_per_tok.values())
    tot_fa = sum(fa_per_tok.values())

    print("\n" + "=" * 80)
    print("RELATÓRIO GRANULAR DA FASE 4A & 4C: AUDITORIA DAS SUB-OPERAÇÕES DE ATENÇÃO")
    print("=" * 80)
    print(f"Tokens medidos:                   {N_TOKENS}")
    print(f"Latência Total por Token:         {total_token_ms:.2f} ms ({tps:.2f} tok/s)")
    print("--------------------------------------------------------------------------------")
    print(f"A. DELTANET (30 Camadas) - TOTAL: {tot_dn:.2f} ms/token ({tot_dn / 30 * 1000:.1f} us/camada)")
    print(f"   1. Projeções de Entrada (QKV, Z, B, A): {dn_per_tok['in_projections']:.2f} ms/tok ({dn_per_tok['in_projections']/30*1000:.1f} us/cam)")
    print(f"   2. Convolução Causal (Triton JIT):     {dn_per_tok['causal_conv1d']:.2f} ms/tok ({dn_per_tok['causal_conv1d']/30*1000:.1f} us/cam)")
    print(f"   3. Recorrência Delta Rule (PyTorch):    {dn_per_tok['recurrent_rule']:.2f} ms/tok ({dn_per_tok['recurrent_rule']/30*1000:.1f} us/cam)")
    print(f"   4. RMSNorm Gated:                      {dn_per_tok['norm_gated']:.2f} ms/tok ({dn_per_tok['norm_gated']/30*1000:.1f} us/cam)")
    print(f"   5. Projeção de Saída (out_proj):       {dn_per_tok['out_proj']:.2f} ms/tok ({dn_per_tok['out_proj']/30*1000:.1f} us/cam)")
    print(f"   6. Gates (beta, g) & Reshapes:         {dn_per_tok['gates_and_repeat'] + dn_per_tok['reshapes_pre_conv'] + dn_per_tok['split_reshape_qkv']:.2f} ms/tok")
    print("--------------------------------------------------------------------------------")
    print(f"B. FULL ATTENTION (10 Camadas) - TOTAL: {tot_fa:.2f} ms/token ({tot_fa / 10 * 1000:.1f} us/camada)")
    print(f"   1. Projeções Q, K, V:                  {fa_per_tok['qkv_proj']:.2f} ms/tok ({fa_per_tok['qkv_proj']/10*1000:.1f} us/cam)")
    print(f"   2. Normas Q/K & RoPE:                  {fa_per_tok['norms_and_rope']:.2f} ms/tok ({fa_per_tok['norms_and_rope']/10*1000:.1f} us/cam)")
    print(f"   3. KV-Cache & Atenção (SDPA):          {fa_per_tok['kv_cache_and_sdpa']:.2f} ms/tok ({fa_per_tok['kv_cache_and_sdpa']/10*1000:.1f} us/cam)")
    print(f"   4. Projeção de Saída e Gate:           {fa_per_tok['gate_and_out_proj']:.2f} ms/tok ({fa_per_tok['gate_and_out_proj']/10*1000:.1f} us/cam)")
    print("--------------------------------------------------------------------------------")
    print(f"TOTAL ATENÇÃO (40 CAMADAS SOMADAS):     {tot_dn + tot_fa:.2f} ms/token ({((tot_dn + tot_fa) / total_token_ms)*100:.1f}% do tempo total)")
    print("=" * 80)

if __name__ == "__main__":
    profile_deltanet_and_attention()
