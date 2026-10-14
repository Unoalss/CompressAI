import sys, os, time
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legion.engine import LegionEngine
from vq2.kernels import vq_moe_parallel_gate_up, vq_moe_parallel_down

def test_moe_graph():
    print("=" * 70)
    print("TESTANDO CAPTURA DE MOE EXECUTION PURA EM CUDA GRAPH")
    print("=" * 70)

    engine = LegionEngine()
    device = "cuda"

    layer_0 = engine.model.model.layers[0]
    paged_expert = layer_0.mlp.experts

    # Prepara slots estáticos e pesos estáticos na GPU
    static_hidden = torch.randn(1, 2048, dtype=torch.bfloat16, device=device)
    static_slots = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32, device=device)
    static_weights = torch.ones(8, dtype=torch.bfloat16, device=device) / 8.0

    def moe_kernel_path(xt, slots, weights):
        K = 8
        xr = paged_expert.rot_2048.fwd(xt.view(-1).bfloat16())
        y_gu = paged_expert.y_gu[:K]
        vq_moe_parallel_gate_up(
            xr, slots, paged_expert.pool_gu_idx, paged_expert.cb16, paged_expert.pool_gu_scale, y_gu,
            OUT=1024, NG=256, rs=paged_expert.RS_2048
        )
        gate = y_gu[:, :512]
        up = y_gu[:, 512:]
        act = (F.silu(gate.float()) * up.float()).bfloat16()
        act_r = paged_expert.rot_512.fwd(act)
        y_down = paged_expert.y_down[:K]
        vq_moe_parallel_down(
            act_r, slots, paged_expert.pool_dn_idx, paged_expert.cb16, paged_expert.pool_dn_scale, y_down,
            OUT=2048, NG=64, rs=paged_expert.RS_512
        )
        out = (y_down * weights[:, None]).sum(dim=0, keepdim=True)
        return out

    # Warmup
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            with torch.no_grad():
                _ = moe_kernel_path(static_hidden, static_slots, static_weights)
    torch.cuda.current_stream().wait_stream(s)

    # Capture
    g_moe = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g_moe, stream=s):
            out_moe = moe_kernel_path(static_hidden, static_slots, static_weights)
        print("  [SUCESSO] MoE kernel execution capturada em CUDA Graph com sucesso!")
        
        # Test replay with different weights/slots
        static_slots.copy_(torch.tensor([7, 6, 5, 4, 3, 2, 1, 0], dtype=torch.int32, device=device))
        g_moe.replay()
        torch.cuda.synchronize()
        print(f"  [SUCESSO] Replay executado! Shape de saída: {out_moe.shape}")
    except Exception as e:
        print(f"  [FALHA] MoE kernel capture error: {e}")

if __name__ == "__main__":
    test_moe_graph()
