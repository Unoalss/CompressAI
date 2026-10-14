import sys, os, time
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legion.engine import LegionEngine
from vq2.kernels import vq_moe_parallel_gate_up, vq_moe_parallel_down, triton_fused_add_rmsnorm

class CUDAGraphDecoder:
    def __init__(self, engine: LegionEngine):
        self.engine = engine
        self.device = "cuda"
        self.model = engine.model
        self.layers = engine.layers
        self.num_layers = len(self.model.model.layers)
        self.is_captured = False

        self.graphs_a = []
        self.graphs_b = []
        self.buffers = []
        self.pinned_slots = torch.empty(8, dtype=torch.int32, pin_memory=True)

    def capture(self, past_kv, initial_pos=32):
        print(f"      Capturando grafos estáticos das {self.num_layers} camadas...")
        t0 = time.time()
        
        pos_ids_3d = torch.zeros(3, 1, 1, dtype=torch.long, device=self.device) + initial_pos
        dummy_x = torch.zeros(1, 1, 2048, dtype=torch.bfloat16, device=self.device)
        static_cos, static_sin = self.model.model.rotary_emb(dummy_x, pos_ids_3d)

        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())

        self.graphs_a.clear()
        self.graphs_b.clear()
        self.buffers.clear()

        for l_idx in range(self.num_layers):
            layer = self.model.model.layers[l_idx]
            paged_expert = layer.mlp.experts

            buf = {
                "x_in": torch.zeros(1, 1, 2048, dtype=torch.bfloat16, device=self.device),
                "slots": torch.zeros(8, dtype=torch.int32, device=self.device),
                "topk_weights": torch.zeros(8, dtype=torch.bfloat16, device=self.device),
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
                        attn_out = l.linear_attn(normed, cache_params=past_kv)
                    else:
                        attn_out, _ = l.self_attn(normed, position_embeddings=(static_cos, static_sin), past_key_values=past_kv)
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

            self.graphs_a.append(ga)
            self.graphs_b.append(gb)
            self.buffers.append(buf)

        self.is_captured = True
        print(f"      [OK] 40 camadas capturadas em {time.time() - t0:.2f}s!")

    def step(self, input_id: int):
        curr_in = torch.tensor([[input_id]], device=self.device)
        h = self.model.model.embed_tokens(curr_in)

        for l_idx in range(self.num_layers):
            buf = self.buffers[l_idx]
            buf["x_in"].copy_(h)

            self.graphs_a[l_idx].replay()

            exp = self.model.model.layers[l_idx].mlp.experts
            topk_ids = buf["ri"].view(-1).tolist()
            for i, e in enumerate(topk_ids):
                self.pinned_slots[i] = exp._get_slot(e)
            buf["slots"].copy_(self.pinned_slots, non_blocking=True)
            buf["topk_weights"].copy_(buf["rw"].view(-1), non_blocking=True)

            self.graphs_b[l_idx].replay()
            h = buf["x_out"]

        normed = self.model.model.norm(h)
        logits = F.linear(normed, self.model.lm_head.weight).squeeze(0).squeeze(0).float()
        next_tok = torch.argmax(logits, dim=-1).item()
        return next_tok, logits

def run_test():
    engine = LegionEngine()
    prompt = "A inteligência artificial avançada permite criar sistemas neurais altamente eficientes com"
    prompt_ids = engine.tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()

    with torch.no_grad():
        out = engine.model(input_ids=prompt_ids, use_cache=True)
        past_kv = out.past_key_values
        next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()

    # Aquecimento de 10 passos para o cache MoE
    curr = torch.tensor([[next_tok]], device="cuda")
    print("\nExecutando aquecimento inicial (10 tokens)...")
    for _ in range(10):
        with torch.no_grad():
            out = engine.model(input_ids=curr, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()
            curr = torch.tensor([[next_tok]], device="cuda")
    torch.cuda.synchronize()

    # Inicializa decodificador CUDA Graph
    decoder = CUDAGraphDecoder(engine)
    decoder.capture(past_kv, initial_pos=prompt_ids.shape[1] + 10)

    # Geração de 30 tokens medindo tempo
    print("\nGerando 30 tokens com CUDA Graphs...")
    times = []
    generated = []
    tok = next_tok

    for _ in range(30):
        ev_s = torch.cuda.Event(enable_timing=True)
        ev_e = torch.cuda.Event(enable_timing=True)
        ev_s.record()
        with torch.no_grad():
            tok, _ = decoder.step(tok)
            generated.append(tok)
        ev_e.record()
        torch.cuda.synchronize()
        times.append(ev_s.elapsed_time(ev_e))

    avg_ms = sum(times) / len(times)
    p50_ms = sorted(times)[15]
    tps = 1000.0 / p50_ms
    text = engine.tokenizer.decode(generated)
    print(f"\nVelocidade Mediana: {tps:.2f} tok/s ({p50_ms:.2f} ms/token | Média: {avg_ms:.2f} ms)")
    print(f"Texto Gerado:\n\"{text}\"")

if __name__ == "__main__":
    run_test()
