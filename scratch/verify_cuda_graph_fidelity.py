import sys, os, time
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legion.engine import LegionEngine
from vq2.kernels import vq_moe_parallel_gate_up, vq_moe_parallel_down, triton_fused_add_rmsnorm

def verify_fidelity():
    print("=" * 80)
    print("AUDITORIA RIGOROSA DE FIDELIDADE NUMÉRICA DOS CUDA GRAPHS")
    print("=" * 80)

    engine = LegionEngine()
    device = "cuda"

    prompt = "A inteligência artificial avançada permite"
    prompt_ids = engine.tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()

    # 1. Execução de Referência: Modo Eager (15 tokens)
    print("\n[1/3] Gerando 15 tokens em Modo Eager de Referência...")
    with torch.no_grad():
        out = engine.model(input_ids=prompt_ids, use_cache=True)
        past_kv_eager = out.past_key_values
        next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()

    curr_eager = torch.tensor([[next_tok]], device=device)
    eager_tokens = [next_tok]
    eager_logits_list = []

    for _ in range(15):
        with torch.no_grad():
            out = engine.model(input_ids=curr_eager, past_key_values=past_kv_eager, use_cache=True)
            past_kv_eager = out.past_key_values
            l = out.logits[:, -1, :]
            next_t = torch.argmax(l, dim=-1).item()
            eager_tokens.append(next_t)
            eager_logits_list.append(l.clone())
            curr_eager = torch.tensor([[next_t]], device=device)

    eager_text = engine.tokenizer.decode(eager_tokens)
    print(f"      Texto Eager: \"{eager_text}\"")

    # 2. Execução com CUDA Graphs
    print("\n[2/3] Gerando 15 tokens em Modo CUDA Graphs com mesmo prompt e contexto...")
    with torch.no_grad():
        out_g = engine.model(input_ids=prompt_ids, use_cache=True)
        past_kv_graph = out_g.past_key_values
        next_tok_g = torch.argmax(out_g.logits[:, -1, :], dim=-1).item()

    curr_graph = torch.tensor([[next_tok_g]], device=device)
    graph_tokens = [next_tok_g]
    graph_logits_list = []

    # Constrói grafos com past_kv_graph
    N_LAYERS = 40
    graphs_a = []
    graphs_b = []
    buffers = []

    pos_step = prompt_ids.shape[1]
    pos_ids_3d = torch.zeros(3, 1, 1, dtype=torch.long, device=device) + pos_step
    dummy_x = torch.zeros(1, 1, 2048, dtype=torch.bfloat16, device=device)
    static_cos, static_sin = engine.model.model.rotary_emb(dummy_x, pos_ids_3d)

    pinned_slots = torch.empty(8, dtype=torch.int32, pin_memory=True)
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())

    for l_idx in range(N_LAYERS):
        layer = engine.model.model.layers[l_idx]
        paged_expert = layer.mlp.experts

        buf = {
            "x_in": torch.zeros(1, 1, 2048, dtype=torch.bfloat16, device=device),
            "slots": torch.zeros(8, dtype=torch.int32, device=device),
            "topk_weights": torch.zeros(8, dtype=torch.bfloat16, device=device),
            "res": None,
            "post_norm": None,
            "sh_final": None,
            "rw": None,
            "ri": None,
            "x_out": None
        }

        def make_part_a(l, b):
            is_linear = (l.block_type == "linear_attention")
            def part_a():
                x = b["x_in"]
                residual = x
                normed = l.input_layernorm(x)
                if is_linear:
                    attn_out = l.linear_attn(normed, cache_params=past_kv_graph)
                else:
                    attn_out, _ = l.self_attn(normed, position_embeddings=(static_cos, static_sin), past_key_values=past_kv_graph)
                res, post_norm = triton_fused_add_rmsnorm(
                    residual, attn_out, l.post_attention_layernorm.weight, l.post_attention_layernorm.eps
                )
                sh_out = l.mlp.shared_expert(post_norm.view(-1, 2048))
                sh_gate = F.sigmoid(l.mlp.shared_expert_gate(post_norm.view(-1, 2048)))
                sh_final = (sh_gate * sh_out).view(1, 1, 2048)
                _, r_weights, r_indices = l.mlp.gate(post_norm.view(-1, 2048))
                return res, post_norm, sh_final, r_weights, r_indices
            return part_a

        def make_part_b(l, b, exp):
            def part_b():
                K = 8
                xt = b["post_norm"].view(-1).bfloat16()
                xr = exp.rot_2048.fwd(xt)
                y_gu = exp.y_gu[:K]
                vq_moe_parallel_gate_up(
                    xr, b["slots"], exp.pool_gu_idx, exp.cb16, exp.pool_gu_scale, y_gu,
                    OUT=1024, NG=256, rs=exp.RS_2048
                )
                gate = y_gu[:, :512]
                up = y_gu[:, 512:]
                act = (F.silu(gate.float()) * up.float()).bfloat16()
                act_r = exp.rot_512.fwd(act)
                y_down = exp.y_down[:K]
                vq_moe_parallel_down(
                    act_r, b["slots"], exp.pool_dn_idx, exp.cb16, exp.pool_dn_scale, y_down,
                    OUT=2048, NG=64, rs=exp.RS_512
                )
                expert_out = (y_down * b["topk_weights"][:, None]).sum(dim=0, keepdim=True).view(1, 1, 2048)
                return b["res"] + b["sh_final"] + expert_out
            return part_b

        fn_a = make_part_a(layer, buf)
        fn_b = make_part_b(layer, buf, paged_expert)

        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                with torch.no_grad():
                    res, pn, sh, rw, ri = fn_a()
        torch.cuda.current_stream().wait_stream(capture_stream)

        ga = torch.cuda.CUDAGraph()
        with torch.cuda.graph(ga, stream=capture_stream):
            buf["res"], buf["post_norm"], buf["sh_final"], buf["rw"], buf["ri"] = fn_a()

        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                with torch.no_grad():
                    out_b = fn_b()
        torch.cuda.current_stream().wait_stream(capture_stream)

        gb = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gb, stream=capture_stream):
            buf["x_out"] = fn_b()

        graphs_a.append(ga)
        graphs_b.append(gb)
        buffers.append(buf)

    lm_head_w = engine.model.lm_head.weight

    for step in range(15):
        with torch.no_grad():
            h = engine.model.model.embed_tokens(curr_graph)
            for l_idx in range(N_LAYERS):
                buf = buffers[l_idx]
                buf["x_in"].copy_(h)
                
                graphs_a[l_idx].replay()
                
                exp = engine.model.model.layers[l_idx].mlp.experts
                topk_ids = buf["ri"].view(-1).tolist()
                for i, e in enumerate(topk_ids):
                    pinned_slots[i] = exp._get_slot(e)
                buf["slots"].copy_(pinned_slots, non_blocking=True)
                buf["topk_weights"].copy_(buf["rw"].view(-1), non_blocking=True)
                
                graphs_b[l_idx].replay()
                h = buf["x_out"]

            normed = engine.model.model.norm(h)
            logits = F.linear(normed, lm_head_w).squeeze(0).squeeze(0).float()
            next_t = torch.argmax(logits, dim=-1).item()
            graph_tokens.append(next_t)
            graph_logits_list.append(logits.clone())
            curr_graph = torch.tensor([[next_t]], device=device)

    graph_text = engine.tokenizer.decode(graph_tokens)
    print(f"      Texto Graphs: \"{graph_text}\"")

    # 3. Auditoria Cruzada
    print("\n[3/3] Comparação Cruzada:")
    print(f"Tokens Eager:  {eager_tokens}")
    print(f"Tokens Graphs: {graph_tokens}")
    tokens_match = (eager_tokens == graph_tokens)
    print(f"Tokens 100% Idênticos: {tokens_match}")

    cos_sims = []
    max_diffs = []
    for step in range(15):
        l_e = eager_logits_list[step]
        l_g = graph_logits_list[step]
        cos = F.cosine_similarity(l_e.float().unsqueeze(0), l_g.float().unsqueeze(0)).item()
        diff = torch.max(torch.abs(l_e.float() - l_g.float())).item()
        cos_sims.append(cos)
        max_diffs.append(diff)
        print(f"  Passo {step+1:02d}: Top1 Eager={eager_tokens[step+1]} | Top1 Graph={graph_tokens[step+1]} | Cosine={cos:.6f} | MaxDiff={diff:.4f}")

    print("\n" + "=" * 80)
    print(f"Média Cosseno de Logits: {sum(cos_sims)/len(cos_sims):.6f}")
    print(f"Desvio Máximo de Logits: {max(max_diffs):.4f}")
    print("=" * 80)

if __name__ == "__main__":
    verify_fidelity()
