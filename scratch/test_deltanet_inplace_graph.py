import sys, os, time
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legion.engine import LegionEngine

def test_inplace_deltanet():
    print("=" * 80)
    print("TESTE: DELTANET CUDA GRAPH COM ATUALIZAÇÃO IN-PLACE DO ESTADO RECORRENTE")
    print("=" * 80)

    engine = LegionEngine()
    device = "cuda"

    prompt = "A inteligência artificial avançada permite criar sistemas neurais altamente eficientes com"
    prompt_ids = engine.tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()

    with torch.no_grad():
        out = engine.model(input_ids=prompt_ids, use_cache=True)
        past_kv = out.past_key_values
        next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()

    # Vamos rodar 10 tokens normais (Eager) e guardar os tokens de referência
    curr = torch.tensor([[next_tok]], device=device)
    ref_tokens = [next_tok]
    print("\n[1/3] Gerando 15 tokens de referência (Modo Eager)...")
    for _ in range(15):
        with torch.no_grad():
            out = engine.model(input_ids=curr, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_t = torch.argmax(out.logits[:, -1, :], dim=-1).item()
            ref_tokens.append(next_t)
            curr = torch.tensor([[next_t]], device=device)

    ref_text = engine.tokenizer.decode(ref_tokens)
    print(f"      Texto Referência: \"{ref_text}\"")

    # Agora reinicia para testar o modo CUDA Graphs
    with torch.no_grad():
        out_g = engine.model(input_ids=prompt_ids, use_cache=True)
        past_kv_g = out_g.past_key_values
        next_tok_g = torch.argmax(out_g.logits[:, -1, :], dim=-1).item()

    # Modifica make_fused_deltanet_fwd para fazer copy_ in-place no recurrent_state
    print("\n[2/3] Configurando DeltaNet com in-place recurrent state copy...")
    for l_idx, layer in enumerate(engine.model.model.layers):
        if layer.block_type != "linear_attention":
            continue

        orig_fwd = layer.linear_attn.forward
        def make_inplace_fwd(orig_fn, l_idx_val):
            def fwd(hidden_states, cache_params=None, **kwargs):
                out = orig_fn(hidden_states, cache_params=cache_params, **kwargs)
                # O estado foi atualizado em cache_params.layers[l_idx_val].recurrent_states[0]
                # Para grafos, garantimos que seja copiado in-place no buffer original
                return out
            return fwd

    # Captura grafos para linear_attn com backup e restauração dos estados durante o warmup
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    graph_data = {}

    for l_idx, layer in enumerate(engine.model.model.layers):
        if layer.block_type != "linear_attention":
            continue

        # Faz backup do estado antes da captura
        cs_bak = past_kv_g.layers[l_idx].conv_states[0].clone()
        rs_bak = past_kv_g.layers[l_idx].recurrent_states[0].clone()

        static_in = torch.zeros(1, 1, 2048, dtype=torch.bfloat16, device=device)
        attn_layer = layer.linear_attn

        # Modifica linear_attn para fazer copy_ no próprio buffer gravado
        rs_target = past_kv_g.layers[l_idx].recurrent_states[0]

        def make_capture_fwd(target_rs, l_idx_val):
            import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as qm
            def fwd(hidden_states, cache_params=None, **kwargs):
                # Executa original
                batch_size, seq_len, _ = hidden_states.shape
                d_qkv = 8192
                d_z = 4096
                d_b = 32
                d_a = 32

                fused_out = F.linear(hidden_states, attn_layer.W_in_fused)
                mixed_qkv = fused_out[:, :, :d_qkv].transpose(1, 2)
                z = fused_out[:, :, d_qkv : d_qkv + d_z].reshape(batch_size, seq_len, -1, attn_layer.head_v_dim)
                b = fused_out[:, :, d_qkv + d_z : d_qkv + d_z + d_b]
                a = fused_out[:, :, d_qkv + d_z + d_b :]

                conv_state = cache_params.layers[l_idx_val].conv_states[0]
                mixed_qkv = qm.causal_conv1d_update(
                    mixed_qkv, conv_state, attn_layer.conv1d.weight.squeeze(1), attn_layer.conv1d.bias, attn_layer.activation
                )
                mixed_qkv = mixed_qkv.transpose(1, 2)
                query, key, value = torch.split(mixed_qkv, [attn_layer.key_dim, attn_layer.key_dim, attn_layer.value_dim], dim=-1)
                query = query.reshape(batch_size, seq_len, -1, attn_layer.head_k_dim)
                key = key.reshape(batch_size, seq_len, -1, attn_layer.head_k_dim)
                value = value.reshape(batch_size, seq_len, -1, attn_layer.head_v_dim)

                from vq2.kernels import fused_delta_gates
                beta, g = fused_delta_gates(b, a, attn_layer.A_exp, attn_layer.dt_bias)

                if attn_layer.num_v_heads // attn_layer.num_k_heads > 1:
                    query = query.repeat_interleave(attn_layer.num_v_heads // attn_layer.num_k_heads, dim=2)
                    key = key.repeat_interleave(attn_layer.num_v_heads // attn_layer.num_k_heads, dim=2)

                core_attn_out, last_rs = qm.torch_recurrent_gated_delta_rule(
                    query, key, value, g=g, beta=beta, initial_state=target_rs,
                    output_final_state=True, use_qk_l2norm_in_kernel=True
                )
                # ATUALIZAÇÃO IN-PLACE NO GRAFO!
                target_rs.copy_(last_rs)

                core_attn_out = core_attn_out.reshape(-1, attn_layer.head_v_dim)
                z = z.reshape(-1, attn_layer.head_v_dim)
                core_attn_out = attn_layer.norm(core_attn_out, z).reshape(batch_size, seq_len, -1)
                return attn_layer.out_proj(core_attn_out)
            return fwd

        captured_fn = make_capture_fwd(rs_target, l_idx)

        # Restaura os backups antes da captura real
        past_kv_g.layers[l_idx].conv_states[0].copy_(cs_bak)
        past_kv_g.layers[l_idx].recurrent_states[0].copy_(rs_bak)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=capture_stream):
            static_out = captured_fn(static_in, cache_params=past_kv_g)

        # Restaura os backups novamente para começar limpo
        past_kv_g.layers[l_idx].conv_states[0].copy_(cs_bak)
        past_kv_g.layers[l_idx].recurrent_states[0].copy_(rs_bak)

        graph_data[l_idx] = {
            "graph": g,
            "static_in": static_in,
            "static_out": static_out,
            "orig_fn": layer.linear_attn.forward
        }

        def make_fwd_wrapper(info):
            def fwd_wrapper(hidden_states, cache_params=None, **kwargs):
                if hidden_states.shape[1] == 1 and hidden_states.is_cuda:
                    info["static_in"].copy_(hidden_states)
                    info["graph"].replay()
                    return info["static_out"]
                return info["orig_fn"](hidden_states, cache_params=cache_params, **kwargs)
            return fwd_wrapper

        layer.linear_attn.forward = make_fwd_wrapper(graph_data[l_idx])

    print("      [OK] 30 camadas DeltaNet empacotadas com avanço de estado in-place!")

    # 3. Teste de Geração (15 tokens)
    print("\n[3/3] Gerando 15 tokens com DeltaNet CUDA Graphs...")
    curr_g = torch.tensor([[next_tok_g]], device=device)
    graph_tokens = [next_tok_g]
    graph_times = []

    for _ in range(15):
        ev_s = torch.cuda.Event(enable_timing=True)
        ev_e = torch.cuda.Event(enable_timing=True)
        ev_s.record()
        with torch.no_grad():
            out = engine.model(input_ids=curr_g, past_key_values=past_kv_g, use_cache=True)
            past_kv_g = out.past_key_values
            next_t = torch.argmax(out.logits[:, -1, :], dim=-1).item()
            graph_tokens.append(next_t)
            curr_g = torch.tensor([[next_t]], device=device)
        ev_e.record()
        torch.cuda.synchronize()
        graph_times.append(ev_s.elapsed_time(ev_e))

    graph_text = engine.tokenizer.decode(graph_tokens)
    p50_ms = sorted(graph_times)[len(graph_times)//2]
    tps = 1000.0 / p50_ms
    print(f"\nVelocidade Mediana: {tps:.2f} tok/s ({p50_ms:.2f} ms/token)")
    print(f"Texto Graphs:\n\"{graph_text}\"")
    print("-" * 80)
    print(f"Texto Referência:\n\"{ref_text}\"")
    print(f"Tokens Idênticos: {ref_tokens == graph_tokens}")

if __name__ == "__main__":
    test_inplace_deltanet()
