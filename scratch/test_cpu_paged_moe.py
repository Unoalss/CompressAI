import sys, os, time
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def vq_moe_parallel_gate_up_cpu(xr, slots, pool_idx, cb16, pool_scale, y_out, OUT=1024, NG=256, rs=1.0):
    K = slots.shape[0]
    xg = xr.view(NG, 8)
    for k in range(K):
        s = slots[k].item()
        idx = (pool_idx[s].to(torch.int32) & 0xFFFF).long()
        sc = pool_scale[s].float()
        rec = cb16[idx] # [1024, 256, 8]
        acc = torch.einsum('org,rg->o', rec, xg)
        y_out[k] = (acc * sc * rs).to(y_out.dtype)

def vq_moe_parallel_down_cpu(act_r, slots, pool_idx, cb16, pool_scale, y_out, OUT=2048, NG=64, rs=1.0):
    K = slots.shape[0]
    for k in range(K):
        s = slots[k].item()
        idx = (pool_idx[s].to(torch.int32) & 0xFFFF).long()
        sc = pool_scale[s].float()
        act_g = act_r[k].view(NG, 8)
        rec = cb16[idx] # [2048, 64, 8]
        acc = torch.einsum('org,rg->o', rec, act_g)
        y_out[k] = (acc * sc * rs).to(y_out.dtype)

def test_cpu_moe():
    print("=" * 80)
    print("TESTE: EXECUÇÃO COMPLETA DE 1 CAMADA MOE EM CPU (8 EXPERTS TOP-8)")
    print("=" * 80)

    cbs_data = torch.load(ROOT / "vq2_out" / "codebooks.pt", weights_only=False, map_location="cpu")
    cb16 = cbs_data["16"].float()

    K = 8
    capacity = 16
    device = "cpu"

    pool_gu_idx = torch.randint(0, 65536, (capacity, 1024, 256), dtype=torch.int32, device=device).to(torch.int16)
    pool_gu_scale = torch.randn(capacity, 1024, dtype=torch.float32, device=device)
    pool_dn_idx = torch.randint(0, 65536, (capacity, 2048, 64), dtype=torch.int32, device=device).to(torch.int16)
    pool_dn_scale = torch.randn(capacity, 2048, dtype=torch.float32, device=device)

    slots = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32, device=device)
    topk_w = torch.ones(8, dtype=torch.float32, device=device) / 8.0

    xr = torch.randn(2048, dtype=torch.float32, device=device)
    y_gu = torch.empty((8, 1024), device=device, dtype=torch.float32)
    y_down = torch.empty((8, 2048), device=device, dtype=torch.float32)

    # 1. Gate Up
    t0 = time.perf_counter()
    vq_moe_parallel_gate_up_cpu(xr, slots, pool_gu_idx, cb16, pool_gu_scale, y_gu, OUT=1024, NG=256, rs=2048**-0.5)
    t_gu = (time.perf_counter() - t0) * 1000.0

    # 2. Activation Silu * Up
    gate = y_gu[:, :512]
    up = y_gu[:, 512:]
    act = F.silu(gate) * up # [8, 512]

    # 3. Down
    t0 = time.perf_counter()
    vq_moe_parallel_down_cpu(act, slots, pool_dn_idx, cb16, pool_dn_scale, y_down, OUT=2048, NG=64, rs=512**-0.5)
    t_dn = (time.perf_counter() - t0) * 1000.0

    # 4. Reduction
    out = (y_down * topk_w[:, None]).sum(dim=0, keepdim=True)

    print(f"Tempo Gate-Up (8 experts em paralelo na CPU): {t_gu:.2f} ms")
    print(f"Tempo Down    (8 experts em paralelo na CPU): {t_dn:.2f} ms")
    print(f"Tempo Total MoE na camada (CPU):               {t_gu + t_dn:.2f} ms")
    print(f"Shape de saída final:                         {out.shape}")

if __name__ == "__main__":
    test_cpu_moe()
