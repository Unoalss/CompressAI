import sys, time
sys.path.insert(0, r"e:\compressAI")
import torch
import triton
import triton.language as tl
from vq2 import vq

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
s1 = torch.load("e:/compressAI/vq2_35b_out/quant_model-00001-of-00021.safetensors.pt", weights_only=False, map_location="cpu")["quant"]
cbs_data = torch.load("e:/compressAI/vq2_out/codebooks.pt", weights_only=False)
cb16 = cbs_data["16"].to(device=DEVICE, dtype=torch.bfloat16)

q_gu = s1["model.language_model.layers.0.mlp.experts.gate_up_proj"]
gate_up_idx = q_gu["idx"][0].view(256, 1024, 256).to(DEVICE)
gate_up_scale = q_gu["scale"].view(256, 1024).to(device=DEVICE, dtype=torch.bfloat16)

topk_ids = torch.tensor([5, 12, 47, 88, 120, 155, 201, 240], device=DEVICE, dtype=torch.int32)
xr = torch.randn(2048, device=DEVICE, dtype=torch.bfloat16)
y_gu = torch.empty((8, 1024), device=DEVICE, dtype=torch.bfloat16)

# Kernel atual:
from vq2.kernels import _parallel_moe_gate_up_kernel

torch.cuda.synchronize()
t0 = time.perf_counter()
N = 500
for _ in range(N):
    grid = (8, 1024 // 16)
    _parallel_moe_gate_up_kernel[grid](
        xr, topk_ids, gate_up_idx, cb16, gate_up_scale, y_gu,
        OUT=1024, NG=256, rs=1.0, BR=16, BG=64, num_warps=4
    )
torch.cuda.synchronize()
dt_orig = (time.perf_counter() - t0) / N * 1000
print(f"Kernel Original BR=16: {dt_orig:.4f} ms ({dt_orig*1000:.1f} us)")

# Kernel otimizado com BR=32 e xv sem broadcast redundante:
@triton.jit
def _opt_gate_up_v2(
    x_ptr, e_ids_ptr, i0_ptr, cb0_ptr, sc_ptr, y_ptr,
    OUT: tl.constexpr, NG: tl.constexpr, rs: tl.constexpr,
    BR: tl.constexpr, BG: tl.constexpr
):
    pid_e = tl.program_id(0)
    pid_r = tl.program_id(1)
    e_idx = tl.load(e_ids_ptr + pid_e).to(tl.int64)
    rows = pid_r * BR + tl.arange(0, BR)
    rmask = rows < OUT
    gl = tl.arange(0, BG)
    j = tl.arange(0, 8)
    acc = tl.zeros([BR], dtype=tl.float32)
    e_off = e_idx * OUT * NG
    e_sc_off = e_idx * OUT
    for g0 in range(0, NG, BG):
        g = g0 + gl
        gmask = g < NG
        # Carrega x apenas UMA vez para todo o bloco! [BG, 8]
        xv = tl.load(x_ptr + g[:, None] * 8 + j[None, :], mask=gmask[:, None], other=0.0).to(tl.float32)
        
        m2 = rmask[:, None] & gmask[None, :]
        off = e_off + rows[:, None].to(tl.int64) * NG + g[None, :]
        a0 = tl.load(i0_ptr + off, mask=m2, other=0).to(tl.int32) & 0xFFFF
        v0 = tl.load(cb0_ptr + a0[:, :, None] * 8 + j[None, None, :]).to(tl.float32)
        
        # Multiplica [BR, BG, 8] por [1, BG, 8]
        acc += tl.sum(tl.sum(v0 * xv[None, :, :], axis=2), axis=1)
        
    sc = tl.load(sc_ptr + e_sc_off + rows, mask=rmask, other=0.0).to(tl.float32)
    y_off = pid_e * OUT + rows
    tl.store(y_ptr + y_off, (acc * sc * rs).to(tl.bfloat16), mask=rmask)

# Testa com BR=32, BG=64, warps=4
y_gu2 = torch.empty((8, 1024), device=DEVICE, dtype=torch.bfloat16)
for BR in [16, 32, 64]:
    grid = (8, 1024 // BR)
    _opt_gate_up_v2[grid](
        xr, topk_ids, gate_up_idx, cb16, gate_up_scale, y_gu2,
        OUT=1024, NG=256, rs=1.0, BR=BR, BG=64, num_warps=4
    )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        _opt_gate_up_v2[grid](
            xr, topk_ids, gate_up_idx, cb16, gate_up_scale, y_gu2,
            OUT=1024, NG=256, rs=1.0, BR=BR, BG=64, num_warps=4
        )
    torch.cuda.synchronize()
    dt_v2 = (time.perf_counter() - t0) / N * 1000
    diff = (y_gu - y_gu2).abs().max().item()
    print(f"Kernel V2 (BR={BR}): {dt_v2:.4f} ms ({dt_v2*1000:.1f} us) | Diff: {diff:.6f}")
