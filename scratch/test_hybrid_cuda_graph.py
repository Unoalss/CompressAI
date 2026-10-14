import sys, os, time
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legion.engine import LegionEngine
from vq2.kernels import vq_moe_parallel_gate_up, vq_moe_parallel_down, triton_fused_add_rmsnorm

def test_hybrid():
    print("=" * 80)
    print("TESTE DE PIPELINE HÍBRIDO: DELTANET CUDA GRAPHS + FULL ATTN EAGER")
    print("=" * 80)

    engine = LegionEngine()
    device = "cuda"

    prompt = "A inteligência artificial avançada permite"
    prompt_ids = engine.tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()

    with torch.no_grad():
        out = engine.model(input_ids=prompt_ids, use_cache=True)
        past_kv = out.past_key_values
        next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()

    curr = torch.tensor([[next_tok]], device=device)
    # Warmup do cache MoE
    for _ in range(5):
        with torch.no_grad():
            out = engine.model(input_ids=curr, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()
            curr = torch.tensor([[next_tok]], device=device)
    torch.cuda.synchronize()

    # Prepara grafos APENAS para as 30 camadas DeltaNet
    # As 10 camadas Full Attention executam no caminho nativo estável
    print("\n[1/3] Capturando grafos para as 30 camadas DeltaNet...")
    delta_graphs_a = {}
    delta_graphs_b = {}
    delta_buffers = {}

    pinned_slots = torch.empty(8, dtype=torch.int32, pin_memory=True)
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())

    t0_cap = time.time()
    for l_idx, layer in enumerate(engine.model.model.layers):
        if layer.block_type != "linear_attention":
            continue

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
            def part_a():
                x = b["x_in"]
                residual = x
                normed = l.input_layernorm(x)
                attn_out = l.linear_attn(normed, cache_params=past_kv)
                res, post_norm = triton_fused_add_rmsnorm(
                    residual, attn_out, l.post_attention_layernorm.weight, l.post_attention_layernorm.eps
                )
                sh_out = l.mlp.shared_expert(post_norm.view(-1, 2048))
                sh_gate = F.sigmoid(l.mlp.shared_expert_gate(post_norm.view(-1, 2048)))
                sh_final = (sh_gate * sh_out).view(1, 1, 2048)
                _, r_weights, r_indices = l.mlp.gate(post_norm.view(-1, 2048))
                return res, post_norm, sh_final, r_weights, r_indices
            return part_a

        def make_part_b(b, exp):
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
        fn_b = make_part_b(buf, paged_expert)

        # Warmup e Captura A
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                with torch.no_grad():
                    res, pn, sh, rw, ri = fn_a()
        torch.cuda.current_stream().wait_stream(capture_stream)

        ga = torch.cuda.CUDAGraph()
        with torch.cuda.graph(ga, stream=capture_stream):
            buf["res"], buf["post_norm"], buf["sh_final"], buf["rw"], buf["ri"] = fn_a()

        # Warmup e Captura B
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                with torch.no_grad():
                    out_b = fn_b()
        torch.cuda.current_stream().wait_stream(capture_stream)

        gb = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gb, stream=capture_stream):
            buf["x_out"] = fn_b()

        delta_graphs_a[l_idx] = ga
        delta_graphs_b[l_idx] = gb
        delta_buffers[l_idx] = buf

    print(f"      [OK] 30 camadas DeltaNet capturadas em {time.time() - t0_cap:.2f}s!")

    # 2. Teste de Geração Híbrida (15 tokens)
    print("\n[2/3] Gerando 15 tokens no modo Híbrido...")
    pos = prompt_ids.shape[1] + 5
    hybrid_tokens = [next_tok]
    hybrid_times = []
    curr_tok = next_tok
    lm_head_w = engine.model.lm_head.weight

    for _ in range(15):
        ev_s = torch.cuda.Event(enable_timing=True)
        ev_e = torch.cuda.Event(enable_timing=True)
        ev_s.record()
        with torch.no_grad():
            curr_in = torch.tensor([[curr_tok]], device=device)
            h = engine.model.model.embed_tokens(curr_in)
            
            # Posições para Full Attention
            pos_ids_3d = torch.zeros(3, 1, 1, dtype=torch.long, device=device) + pos
            pos_emb = engine.model.model.rotary_emb(h, pos_ids_3d)

            for l_idx, layer in enumerate(engine.model.model.layers):
                if layer.block_type == "linear_attention":
                    # Camada DeltaNet acelerada via CUDA Graphs
                    buf = delta_buffers[l_idx]
                    buf["x_in"].copy_(h)
                    delta_graphs_a[l_idx].replay()

                    exp = layer.mlp.experts
                    topk_ids = buf["ri"].view(-1).tolist()
                    for i, e in enumerate(topk_ids):
                        pinned_slots[i] = exp._get_slot(e)
                    buf["slots"].copy_(pinned_slots, non_blocking=True)
                    buf["topk_weights"].copy_(buf["rw"].view(-1), non_blocking=True)

                    delta_graphs_b[l_idx].replay()
                    h = buf["x_out"]
                else:
                    # Camada Full Attention executada nativamente em Eager
                    out_l = layer(h, position_embeddings=pos_emb, past_key_values=past_kv)
                    h = out_l[0] if isinstance(out_l, tuple) else out_l

            normed = engine.model.model.norm(h)
            logits = F.linear(normed, lm_head_w).squeeze(0).squeeze(0).float()
            curr_tok = torch.argmax(logits, dim=-1).item()
            hybrid_tokens.append(curr_tok)
            pos += 1
        ev_e.record()
        torch.cuda.synchronize()
        hybrid_times.append(ev_s.elapsed_time(ev_e))

    text_hybrid = engine.tokenizer.decode(hybrid_tokens)
    p50_ms = sorted(hybrid_times)[len(hybrid_times)//2]
    tps = 1000.0 / p50_ms
    print(f"\n[3/3] Desempenho Híbrido: {tps:.2f} tok/s ({p50_ms:.2f} ms/token)")
    print(f"      Texto Gerado: \"{text_hybrid}\"")

if __name__ == "__main__":
    test_hybrid()
