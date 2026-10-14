import torch, triton, time
import triton.language as tl

@triton.jit
def vq_gemv_16_opt(x_ptr, i0_ptr, cb0_ptr, sc_ptr, y_ptr, OUT, NG, rs,
                   BR: tl.constexpr, BG: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    rmask = rows < OUT
    gl = tl.arange(0, BG)
    j = tl.arange(0, 8)
    acc = tl.zeros([BR], dtype=tl.float32)
    for g0 in range(0, NG, BG):
        g = g0 + gl
        gmask = g < NG
        m2 = rmask[:, None] & gmask[None, :]
        off = rows[:, None].to(tl.int64) * NG + g[None, :]
        a0 = tl.load(i0_ptr + off, mask=m2, other=0).to(tl.int32) & 0xFFFF
        v0 = tl.load(cb0_ptr + a0[:, :, None] * 8 + j[None, None, :]).to(tl.float32)
        xv = tl.load(x_ptr + g[None, :, None] * 8 + j[None, None, :],
                     mask=gmask[None, :, None], other=0.0).to(tl.float32)
        acc += tl.sum(tl.sum(v0 * xv, axis=2), axis=1)
    sc = tl.load(sc_ptr + rows, mask=rmask, other=0.0).to(tl.float32)
    tl.store(y_ptr + rows, (acc * sc * rs).to(tl.bfloat16), mask=rmask)

def bench_lm_head():
    OUT = 248320
    N = 2048
    NG = N // 8
    device = "cuda"
    xr = torch.randn(N, dtype=torch.bfloat16, device=device)
    i0 = torch.randint(-32768, 32767, (OUT, NG), dtype=torch.int16, device=device)
    cb0 = torch.randn(65536, 8, dtype=torch.bfloat16, device=device)
    scale = torch.randn(OUT, dtype=torch.bfloat16, device=device)
    y = torch.empty(OUT, dtype=torch.bfloat16, device=device)

    configs = [
        (4, 128, 2),
        (8, 64, 2),
        (16, 64, 4),
        (32, 64, 4),
        (32, 128, 4),
        (64, 64, 4),
        (64, 128, 8)
    ]
    print(f"Benchmarking lm_head: OUT={OUT}, N={N}")
    for BR, BG, warps in configs:
        grid = (triton.cdiv(OUT, BR),)
        # warmup
        for _ in range(5):
            vq_gemv_16_opt[grid](xr, i0, cb0, scale, y, OUT, NG, 1.0, BR=BR, BG=BG, num_warps=warps)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        N_RUNS = 20
        for _ in range(N_RUNS):
            vq_gemv_16_opt[grid](xr, i0, cb0, scale, y, OUT, NG, 1.0, BR=BR, BG=BG, num_warps=warps)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / N_RUNS * 1000
        print(f"BR={BR:2d}, BG={BG:3d}, warps={warps} : {dt:.3f} ms")

if __name__ == "__main__":
    bench_lm_head()
