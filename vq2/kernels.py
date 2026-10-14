"""
vq2/kernels.py - kernels Triton fundidos para decodificacao ultrarrapida:
1. `vq_gemv_16_8`: matriz a 3.0 bpw (estagio 16 + resíduo 8)
2. `vq_gemv_16`: matriz a 2.0 bpw (estagio 16 apenas, 33% menos dados lidos)
3. `triton_conv1d_update`: convolucao causal 1D fundida do DeltaNet (3.3x mais rapido que PyTorch fallback)
"""
import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # pragma: no cover
    HAVE_TRITON = False

FUSED_KERNELS_ENABLED = True

def set_fused_kernels_enabled(enabled: bool):
    global FUSED_KERNELS_ENABLED
    FUSED_KERNELS_ENABLED = bool(enabled)

if HAVE_TRITON:
    _CONFIGS = [
        triton.Config({"BR": 4, "BG": 128}, num_warps=2),
        triton.Config({"BR": 4, "BG": 64}, num_warps=2),
        triton.Config({"BR": 8, "BG": 64}, num_warps=2),
        triton.Config({"BR": 8, "BG": 64}, num_warps=4),
        triton.Config({"BR": 16, "BG": 64}, num_warps=4),
        triton.Config({"BR": 2, "BG": 128}, num_warps=1),
    ]

    @triton.autotune(configs=_CONFIGS, key=["OUT", "NG"])
    @triton.jit
    def _vq_gemv_16_8(x_ptr, i0_ptr, i1_ptr, cb0_ptr, cb1_ptr, sc_ptr, y_ptr, OUT, NG, rs,
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
            a1 = tl.load(i1_ptr + off, mask=m2, other=0).to(tl.int32)
            v0 = tl.load(cb0_ptr + a0[:, :, None] * 8 + j[None, None, :]).to(tl.float32)
            v1 = tl.load(cb1_ptr + a1[:, :, None] * 8 + j[None, None, :]).to(tl.float32)
            xv = tl.load(x_ptr + g[None, :, None] * 8 + j[None, None, :],
                         mask=gmask[None, :, None], other=0.0).to(tl.float32)
            acc += tl.sum(tl.sum((v0 + v1) * xv, axis=2), axis=1)
        sc = tl.load(sc_ptr + rows, mask=rmask, other=0.0).to(tl.float32)
        tl.store(y_ptr + rows, (acc * sc * rs).to(tl.bfloat16), mask=rmask)

    @triton.autotune(configs=_CONFIGS, key=["OUT", "NG"])
    @triton.jit
    def _vq_gemv_16(x_ptr, i0_ptr, cb0_ptr, sc_ptr, y_ptr, OUT, NG, rs,
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

    @triton.jit
    def _triton_conv1d_update(
        cs_ptr, hs_ptr, w_ptr, b_ptr, out_ptr,
        H: tl.constexpr, BLOCK: tl.constexpr
    ):
        pid = tl.program_id(0)
        ch = pid * BLOCK + tl.arange(0, BLOCK)
        mask = ch < H
        off_cs0 = ch * 4 + 0
        off_cs1 = ch * 4 + 1
        off_cs2 = ch * 4 + 2
        off_cs3 = ch * 4 + 3

        c1 = tl.load(cs_ptr + off_cs1, mask=mask, other=0.0).to(tl.float32)
        c2 = tl.load(cs_ptr + off_cs2, mask=mask, other=0.0).to(tl.float32)
        c3 = tl.load(cs_ptr + off_cs3, mask=mask, other=0.0).to(tl.float32)
        c4 = tl.load(hs_ptr + ch, mask=mask, other=0.0).to(tl.float32)

        tl.store(cs_ptr + off_cs0, c1.to(tl.bfloat16), mask=mask)
        tl.store(cs_ptr + off_cs1, c2.to(tl.bfloat16), mask=mask)
        tl.store(cs_ptr + off_cs2, c3.to(tl.bfloat16), mask=mask)
        tl.store(cs_ptr + off_cs3, c4.to(tl.bfloat16), mask=mask)

        w0 = tl.load(w_ptr + ch * 4 + 0, mask=mask, other=0.0).to(tl.float32)
        w1 = tl.load(w_ptr + ch * 4 + 1, mask=mask, other=0.0).to(tl.float32)
        w2 = tl.load(w_ptr + ch * 4 + 2, mask=mask, other=0.0).to(tl.float32)
        w3 = tl.load(w_ptr + ch * 4 + 3, mask=mask, other=0.0).to(tl.float32)

        acc = c1 * w0 + c2 * w1 + c3 * w2 + c4 * w3
        if b_ptr is not None:
            b = tl.load(b_ptr + ch, mask=mask, other=0.0).to(tl.float32)
            acc += b

        acc = acc * tl.sigmoid(acc)
        tl.store(out_ptr + ch, acc.to(tl.bfloat16), mask=mask)


    @triton.jit
    def _parallel_moe_gate_up_kernel(
        x_ptr, e_ids_ptr, i0_ptr, cb0_ptr, sc_ptr, y_ptr,
        OUT: tl.constexpr, NG: tl.constexpr, rs: tl.constexpr,
        BR: tl.constexpr, BG: tl.constexpr
    ):
        pid_e = tl.program_id(0) # 0..7 (cada expert ativo)
        pid_r = tl.program_id(1) # blocos de linhas
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
            acc += tl.sum(tl.sum(v0 * xv[None, :, :], axis=2), axis=1)
        sc = tl.load(sc_ptr + e_sc_off + rows, mask=rmask, other=0.0).to(tl.float32)
        y_off = pid_e * OUT + rows
        tl.store(y_ptr + y_off, (acc * sc * rs).to(tl.bfloat16), mask=rmask)

    @triton.jit
    def _parallel_moe_down_kernel(
        act_ptr, e_ids_ptr, i0_ptr, cb0_ptr, sc_ptr, y_ptr,
        OUT: tl.constexpr, NG: tl.constexpr, rs: tl.constexpr,
        BR: tl.constexpr, BG: tl.constexpr
    ):
        pid_e = tl.program_id(0) # 0..7
        pid_r = tl.program_id(1)
        e_idx = tl.load(e_ids_ptr + pid_e).to(tl.int64)
        rows = pid_r * BR + tl.arange(0, BR)
        rmask = rows < OUT
        gl = tl.arange(0, BG)
        j = tl.arange(0, 8)
        acc = tl.zeros([BR], dtype=tl.float32)
        e_off = e_idx * OUT * NG
        e_sc_off = e_idx * OUT
        act_e_off = pid_e * (NG * 8)
        for g0 in range(0, NG, BG):
            g = g0 + gl
            gmask = g < NG
            # Carrega ativação do expert pid_e apenas UMA vez para todo o bloco! [BG, 8]
            xv = tl.load(act_ptr + act_e_off + g[:, None] * 8 + j[None, :], mask=gmask[:, None], other=0.0).to(tl.float32)
            m2 = rmask[:, None] & gmask[None, :]
            off = e_off + rows[:, None].to(tl.int64) * NG + g[None, :]
            a0 = tl.load(i0_ptr + off, mask=m2, other=0).to(tl.int32) & 0xFFFF
            v0 = tl.load(cb0_ptr + a0[:, :, None] * 8 + j[None, None, :]).to(tl.float32)
            acc += tl.sum(tl.sum(v0 * xv[None, :, :], axis=2), axis=1)
        sc = tl.load(sc_ptr + e_sc_off + rows, mask=rmask, other=0.0).to(tl.float32)
        y_off = pid_e * OUT + rows
        tl.store(y_ptr + y_off, (acc * sc * rs).to(tl.bfloat16), mask=rmask)

    @triton.jit
    def _triton_rmsnorm_gated_kernel(
        x_ptr, gate_ptr, w_ptr, out_ptr,
        stride_xm, stride_xd,
        stride_gm, stride_gd,
        stride_om, stride_od,
        eps: tl.constexpr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        cols = tl.arange(0, BLOCK_D)
        mask = cols < D

        off_x = pid * stride_xm + cols * stride_xd
        off_g = pid * stride_gm + cols * stride_gd
        off_o = pid * stride_om + cols * stride_od

        x = tl.load(x_ptr + off_x, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.bfloat16)
        g = tl.load(gate_ptr + off_g, mask=mask, other=0.0).to(tl.float32)

        # 1. Variância em float32
        var = tl.sum(x * x, axis=0) / D
        rsqrt_val = tl.math.rsqrt(var + eps)

        # 2. Normalização e cast para bfloat16 conforme referência
        norm_x = (x * rsqrt_val).to(tl.bfloat16)
        weighted = norm_x * w

        # 3. SiLU do gate em float32: g * sigmoid(g)
        silu_g = g * tl.sigmoid(g)

        # 4. Produto final em float32 convertido para bfloat16
        out = (weighted.to(tl.float32) * silu_g).to(tl.bfloat16)
        tl.store(out_ptr + off_o, out, mask=mask)

def vq_gemv_16_8(xr, i0, i1, cb0, cb1, scale, out, n, rs=1.0):
    """xr [n] bf16 -> y [out] bf16 para matriz 3.0 bpw."""
    import torch
    y = torch.empty(out, dtype=torch.bfloat16, device=xr.device)
    grid = lambda meta: (triton.cdiv(out, meta["BR"]),)
    _vq_gemv_16_8[grid](xr, i0, i1, cb0, cb1, scale, y, out, n // 8, rs)
    return y


def vq_gemv_16(xr, i0, cb0, scale, out, n, rs=1.0):
    """xr [n] bf16 -> y [out] bf16 para matriz 2.0 bpw (sem estagio residual)."""
    import torch
    y = torch.empty(out, dtype=torch.bfloat16, device=xr.device)
    grid = lambda meta: (triton.cdiv(out, meta["BR"]),)
    _vq_gemv_16[grid](xr, i0, cb0, scale, y, out, n // 8, rs)
    return y


def triton_conv1d_update(hs, cs, w, b=None):
    """hs [1, H, 1], cs [1, H, 4], w [H, 4], b [H] -> out [1, H, 1]."""
    import torch
    H = hs.shape[1]
    out = torch.empty_like(hs)
    BLOCK = 64
    grid = (triton.cdiv(H, BLOCK),)
    _triton_conv1d_update[grid](
        cs, hs, w, b, out, H=H, BLOCK=BLOCK, num_warps=2
    )
    return out


def vq_moe_parallel_gate_up(xr, topk_ids, gate_up_idx, cb16, gate_up_scale, y_out, OUT=1024, NG=256, rs=1.0):
    """Executa o gate_up_proj dos experts ativos em paralelo com carregamento O(1) de x."""
    K = topk_ids.shape[0]
    BR, BG = 32, 64
    grid = (K, OUT // BR)
    _parallel_moe_gate_up_kernel[grid](
        xr, topk_ids, gate_up_idx, cb16, gate_up_scale, y_out,
        OUT=OUT, NG=NG, rs=rs, BR=BR, BG=BG, num_warps=4
    )

def vq_moe_parallel_down(act_r, topk_ids, down_idx, cb16, down_scale, y_out, OUT=2048, NG=64, rs=1.0):
    """Executa o down_proj dos experts ativos em paralelo com carregamento O(1) da ativação."""
    K = topk_ids.shape[0]
    BR, BG = 64, 64
    grid = (K, OUT // BR)
    _parallel_moe_down_kernel[grid](
        act_r, topk_ids, down_idx, cb16, down_scale, y_out,
        OUT=OUT, NG=NG, rs=rs, BR=BR, BG=BG, num_warps=4
    )


def triton_rmsnorm_gated(hidden_states, gate, weight, eps: float = 1e-6):
    """Executa RMSNorm + SiLU Gating fundidos em 1 único kernel Triton."""
    import torch
    D = hidden_states.shape[-1]
    orig_shape = hidden_states.shape
    x_2d = hidden_states.view(-1, D)
    g_2d = gate.view(-1, D)
    M = x_2d.shape[0]

    out = torch.empty_like(x_2d)
    grid = (M,)
    BLOCK_D = triton.next_power_of_2(D)

    _triton_rmsnorm_gated_kernel[grid](
        x_2d, g_2d, weight, out,
        x_2d.stride(0), x_2d.stride(1),
        g_2d.stride(0), g_2d.stride(1),
        out.stride(0), out.stride(1),
        eps=eps, D=D, BLOCK_D=BLOCK_D,
        num_warps=2
    )
    return out.view(orig_shape)


if HAVE_TRITON:
    @triton.jit
    def _fused_qwen35_rmsnorm_rope_kernel(
        x_ptr, w_ptr, cos_ptr, sin_ptr, out_ptr,
        stride_xh, stride_xd,
        stride_oh, stride_od,
        stride_cos_d, stride_sin_d,
        eps: tl.constexpr,
    ):
        """
        Fused RMSNorm (zero-centered: 1.0 + w) + Partial RoPE (primeiras 64 dims de 256).
        D = 256, ROTARY_DIM = 64, HALF_ROTARY = 32, PASS_DIM = 192.
        Grid: (H,) onde H é o número de cabeças de atenção.
        """
        pid = tl.program_id(0)

        # Divisão em potências de 2: 32 + 32 + 64 + 128 = 256
        idx_32 = tl.arange(0, 32)
        idx_64 = tl.arange(0, 64)
        idx_128 = tl.arange(0, 128)

        # 1. Carrega x em 4 blocos
        x0 = tl.load(x_ptr + pid * stride_xh + idx_32 * stride_xd).to(tl.float32)        # 0..31
        x1 = tl.load(x_ptr + pid * stride_xh + (idx_32 + 32) * stride_xd).to(tl.float32) # 32..63
        x2 = tl.load(x_ptr + pid * stride_xh + (idx_64 + 64) * stride_xd).to(tl.float32) # 64..127
        x3 = tl.load(x_ptr + pid * stride_xh + (idx_128 + 128) * stride_xd).to(tl.float32) # 128..255

        # 2. Carrega w (zero-centered: 1.0 + w)
        w0 = 1.0 + tl.load(w_ptr + idx_32).to(tl.float32)
        w1 = 1.0 + tl.load(w_ptr + (idx_32 + 32)).to(tl.float32)
        w2 = 1.0 + tl.load(w_ptr + (idx_64 + 64)).to(tl.float32)
        w3 = 1.0 + tl.load(w_ptr + (idx_128 + 128)).to(tl.float32)

        # 3. RMSNorm sobre todos os 256 elementos
        sum_sq = (
            tl.sum(x0 * x0, axis=0) +
            tl.sum(x1 * x1, axis=0) +
            tl.sum(x2 * x2, axis=0) +
            tl.sum(x3 * x3, axis=0)
        )
        rms = tl.rsqrt((sum_sq / 256.0) + eps)

        # Escala normalizada
        v0 = x0 * rms * w0
        v1 = x1 * rms * w1
        v2 = x2 * rms * w2
        v3 = x3 * rms * w3

        # 4. RoPE nas primeiras 64 dimensões (v0 e v1)
        cos0 = tl.load(cos_ptr + idx_32 * stride_cos_d).to(tl.float32)
        cos1 = tl.load(cos_ptr + (idx_32 + 32) * stride_cos_d).to(tl.float32)
        sin0 = tl.load(sin_ptr + idx_32 * stride_sin_d).to(tl.float32)
        sin1 = tl.load(sin_ptr + (idx_32 + 32) * stride_sin_d).to(tl.float32)

        y0 = (v0 * cos0) - (v1 * sin0)
        y1 = (v1 * cos1) + (v0 * sin1)
        # v2 e v3 passam sem rotação

        # 5. Salva diretamente no tensor de saída (1, H, 1, 256)
        tl.store(out_ptr + pid * stride_oh + idx_32 * stride_od, y0.to(tl.bfloat16))
        tl.store(out_ptr + pid * stride_oh + (idx_32 + 32) * stride_od, y1.to(tl.bfloat16))
        tl.store(out_ptr + pid * stride_oh + (idx_64 + 64) * stride_od, v2.to(tl.bfloat16))
        tl.store(out_ptr + pid * stride_oh + (idx_128 + 128) * stride_od, v3.to(tl.bfloat16))

    def triton_fused_rmsnorm_rope_qwen(x_states, weight, cos, sin, eps=1e-6):
        H = x_states.shape[2]
        D = x_states.shape[3]
        out = torch.empty((1, H, 1, D), device=x_states.device, dtype=torch.bfloat16)
        x_2d = x_states.view(H, D)
        out_2d = out.view(H, D)
        cos_1d = cos.view(-1)
        sin_1d = sin.view(-1)

        grid = (H,)
        _fused_qwen35_rmsnorm_rope_kernel[grid](
            x_2d, weight, cos_1d, sin_1d, out_2d,
            x_2d.stride(0), x_2d.stride(1),
            out_2d.stride(0), out_2d.stride(1),
            cos_1d.stride(0), sin_1d.stride(0),
            eps=eps,
            num_warps=2
        )
        return out

    @triton.jit
    def _fused_add_rmsnorm_zero_centered_kernel(
        x_ptr, delta_ptr, res_out_ptr, norm_out_ptr, w_ptr,
        eps: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr
    ):
        cols = tl.arange(0, BLOCK_D)
        mask = cols < D

        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        delta = tl.load(delta_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        # 1. Soma residual
        res = x + delta
        tl.store(res_out_ptr + cols, res.to(tl.bfloat16), mask=mask)

        # 2. RMSNorm zero-centered sobre res (1.0 + w)
        var = tl.sum(res * res, axis=0) / D
        rsqrt_val = tl.rsqrt(var + eps)
        normed = res * rsqrt_val * (1.0 + w)
        tl.store(norm_out_ptr + cols, normed.to(tl.bfloat16), mask=mask)

    def triton_fused_add_rmsnorm(x, delta, weight, eps=1e-6):
        """Funde a soma residual e a normalização zero-centered em uma única passagem de registradores."""
        orig_shape = x.shape
        D = x.shape[-1]
        x_flat = x.view(-1)
        delta_flat = delta.view(-1)
        res_out = torch.empty_like(x_flat)
        norm_out = torch.empty_like(x_flat)
        BLOCK_D = triton.next_power_of_2(D)

        _fused_add_rmsnorm_zero_centered_kernel[(1,)](
            x_flat, delta_flat, res_out, norm_out, weight,
            eps=eps, D=D, BLOCK_D=BLOCK_D, num_warps=2
        )
        return res_out.view(orig_shape), norm_out.view(orig_shape)

    @triton.jit
    def _fused_delta_gates_kernel(
        b_ptr, a_ptr, A_exp_ptr, dt_bias_ptr, beta_ptr, g_ptr,
        N: tl.constexpr, BLOCK: tl.constexpr
    ):
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        a = tl.load(a_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        A_exp = tl.load(A_exp_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        dt_bias = tl.load(dt_bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        beta = tl.sigmoid(b)
        tl.store(beta_ptr + cols, beta.to(tl.bfloat16), mask=mask)

        x = a + dt_bias
        sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
        g = -A_exp * sp
        tl.store(g_ptr + cols, g, mask=mask)

    def fused_delta_gates(b, a, A_exp, dt_bias):
        N = b.shape[-1]
        beta = torch.empty_like(b)
        g = torch.empty((b.shape[0], b.shape[1], N), device=b.device, dtype=torch.float32)
        _fused_delta_gates_kernel[(1,)](
            b, a, A_exp, dt_bias, beta, g, N=N, BLOCK=64, num_warps=1
        )
        return beta, g


def optimize_deltanet_layers(model):
    """
    Otimização 4 (Passos 2 e 3): Funde as matrizes de projeção W_in (QKV, Z, B, A)
    e pré-computa A_exp para as camadas DeltaNet com zero overhead de VRAM.
    """
    import torch
    import torch.nn.functional as F
    import gc

    d_qkv = 8192
    d_z = 4096
    d_b = 32
    d_a = 32

    linear_layers = [l.linear_attn for l in model.model.layers if hasattr(l, "linear_attn")]
    if not linear_layers:
        return

    for layer in linear_layers:
        if getattr(layer, "_is_deltanet_optimized", False):
            continue

        W_fused = torch.cat([
            layer.in_proj_qkv.weight.data,
            layer.in_proj_z.weight.data,
            layer.in_proj_b.weight.data,
            layer.in_proj_a.weight.data
        ], dim=0).contiguous()

        layer.W_in_fused = torch.nn.Parameter(W_fused, requires_grad=False)
        layer.in_proj_qkv.weight.data = layer.W_in_fused[:d_qkv]
        layer.in_proj_z.weight.data = layer.W_in_fused[d_qkv : d_qkv + d_z]
        layer.in_proj_b.weight.data = layer.W_in_fused[d_qkv + d_z : d_qkv + d_z + d_b]
        layer.in_proj_a.weight.data = layer.W_in_fused[d_qkv + d_z + d_b :]
        layer.A_exp = layer.A_log.float().exp().contiguous()

        orig_linear_fwd = layer.forward
        def make_fused_deltanet_fwd(layer_ref):
            import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as qm
            def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
                hidden_states = qm.apply_mask_to_padding_states(hidden_states, attention_mask)
                batch_size, seq_len, _ = hidden_states.shape
                use_precomputed_states = cache_params is not None and cache_params.has_previous_state(self.layer_idx, state_idx=0)

                # Projeção de entrada (Passo 2)
                if FUSED_KERNELS_ENABLED:
                    fused_out = F.linear(hidden_states, self.W_in_fused)
                    mixed_qkv = fused_out[:, :, :d_qkv]
                    z = fused_out[:, :, d_qkv : d_qkv + d_z]
                    b = fused_out[:, :, d_qkv + d_z : d_qkv + d_z + d_b]
                    a = fused_out[:, :, d_qkv + d_z + d_b :]
                else:
                    mixed_qkv = self.in_proj_qkv(hidden_states)
                    z = self.in_proj_z(hidden_states)
                    b = self.in_proj_b(hidden_states)
                    a = self.in_proj_a(hidden_states)

                mixed_qkv = mixed_qkv.transpose(1, 2)
                z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

                conv_state = cache_params.layers[self.layer_idx].conv_states[0] if (use_precomputed_states and seq_len == 1) else None
                if conv_state is not None:
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

                # Gates fundidos Triton (Passo 3)
                if seq_len == 1 and HAVE_TRITON and FUSED_KERNELS_ENABLED and hidden_states.is_cuda:
                    beta, g = fused_delta_gates(b, a, self.A_exp, self.dt_bias)
                else:
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

                # RMSNormGated Triton (Passo 1)
                core_attn_out = self.norm(core_attn_out, z)
                core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
                output = self.out_proj(core_attn_out)
                return output
            return forward

        layer.forward = make_fused_deltanet_fwd(layer).__get__(layer, layer.__class__)
        layer._is_deltanet_optimized = True

    gc.collect()
    torch.cuda.empty_cache()


import sys

_PATCHED = False

def patch_transformers_kernels():
    """Injeta automaticamente kernels Triton em causal_conv1d_update, RMSNormGated e Full Attention, e garante fallback CPU seguro para delta rule."""
    global _PATCHED
    if _PATCHED:
        return
    import torch
    for mod_name in [
        "transformers.models.qwen3_5.modeling_qwen3_5",
        "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"
    ]:
        try:
            import importlib
            mod = sys.modules.get(mod_name)
            if mod is None:
                mod = importlib.import_module(mod_name)

            # Fallback seguro CPU para Gated Delta Rule (evita que fla/Triton seja chamado na CPU)
            for fn_name in ["torch_recurrent_gated_delta_rule", "torch_chunk_gated_delta_rule"]:
                orig_rule = getattr(mod, fn_name, None)
                if orig_rule is not None and not getattr(orig_rule, "_is_cpu_fallback_patched", False):
                    wrapped_rule = getattr(orig_rule, "__wrapped__", orig_rule)
                    def make_patched_rule(old_r, fallback_r):
                        def patched_rule(*args, **kwargs):
                            q = args[0] if len(args) > 0 else kwargs.get("query")
                            if q is not None and not q.is_cuda:
                                return fallback_r(*args, **kwargs)
                            return old_r(*args, **kwargs)
                        patched_rule._is_cpu_fallback_patched = True
                        return patched_rule
                    setattr(mod, fn_name, make_patched_rule(orig_rule, wrapped_rule))

            orig_fn = getattr(mod, "causal_conv1d_update", None)
            if orig_fn is not None and not getattr(orig_fn, "_is_triton_patched", False):
                def make_patched(old_fn):
                    def patched(hidden_states, conv_state, weight, bias=None, activation=None):
                        if (HAVE_TRITON and FUSED_KERNELS_ENABLED and hidden_states.is_cuda and hidden_states.shape[0] == 1 and hidden_states.shape[-1] == 1 and
                            conv_state.shape[-1] == 4 and (activation == "silu" or activation is None)):
                            return triton_conv1d_update(hidden_states, conv_state, weight, bias)
                        return old_fn(hidden_states, conv_state, weight, bias, activation)
                    patched._is_triton_patched = True
                    return patched
                setattr(mod, "causal_conv1d_update", make_patched(orig_fn))

            for norm_cls_name in ["Qwen3_5MoeRMSNormGated", "Qwen3_5RMSNormGated"]:
                norm_cls = getattr(mod, norm_cls_name, None)
                if norm_cls is not None and not getattr(norm_cls.forward, "_is_triton_patched", False):
                    def make_patched_norm(old_fwd):
                        def patched_norm(self, hidden_states, gate):
                            if (FUSED_KERNELS_ENABLED and hidden_states.is_cuda and self.activation == "silu" and
                                hidden_states.shape[-1] <= 256 and
                                (hidden_states.dtype == torch.bfloat16 or hidden_states.dtype == torch.float16)):
                                return triton_rmsnorm_gated(hidden_states, gate, self.weight, self.variance_epsilon)
                            return old_fwd(self, hidden_states, gate)
                        patched_norm._is_triton_patched = True
                        return patched_norm
                    norm_cls.forward = make_patched_norm(norm_cls.forward)

            # Fase 4C: Patch Full Attention (RMSNorm + RoPE fundidos no Qwen 3.5)
            for attn_cls_name in ["Qwen3_5MoeAttention", "Qwen3_5Attention"]:
                attn_cls = getattr(mod, attn_cls_name, None)
                if attn_cls is not None and not getattr(attn_cls.forward, "_is_triton_fused_rope_patched", False):
                    def make_patched_fa(old_fa_fwd):
                        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
                        def patched_fa(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
                            input_shape = hidden_states.shape[:-1]
                            seq_len = hidden_states.shape[1]
                            if (FUSED_KERNELS_ENABLED and hidden_states.is_cuda and seq_len == 1 and self.head_dim == 256 and
                                hasattr(self, "q_norm") and hasattr(self, "k_norm")):
                                hidden_shape = (*input_shape, -1, self.head_dim)
                                query_states, gate = torch.chunk(
                                    self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
                                )
                                gate = gate.reshape(*input_shape, -1)
                                cos, sin = position_embeddings

                                query_states = triton_fused_rmsnorm_rope_qwen(
                                    query_states.view(hidden_shape), self.q_norm.weight, cos, sin, eps=self.q_norm.eps
                                )
                                key_raw = self.k_proj(hidden_states).view(hidden_shape)
                                key_states = triton_fused_rmsnorm_rope_qwen(
                                    key_raw, self.k_norm.weight, cos, sin, eps=self.k_norm.eps
                                )
                                value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                                if past_key_values is not None:
                                    key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

                                attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                                    self.config._attn_implementation, mod.eager_attention_forward
                                )
                                attn_output, attn_weights = attention_interface(
                                    self, query_states, key_states, value_states, attention_mask,
                                    dropout=0.0 if not self.training else self.attention_dropout,
                                    scaling=self.scaling, **kwargs
                                )
                                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                                attn_output = attn_output * torch.sigmoid(gate)
                                attn_output = self.o_proj(attn_output)
                                return attn_output, attn_weights

                            return old_fa_fwd(self, hidden_states, position_embeddings, attention_mask=attention_mask, past_key_values=past_key_values, **kwargs)
                        patched_fa._is_triton_fused_rope_patched = True
                        return patched_fa
                    attn_cls.forward = make_patched_fa(attn_cls.forward)

            # Pipeline Optimization: Fused Residual Add + Post Attention RMSNorm
            for layer_cls_name in ["Qwen3_5MoeDecoderLayer", "Qwen3_5DecoderLayer"]:
                layer_cls = getattr(mod, layer_cls_name, None)
                if layer_cls is not None and not getattr(layer_cls.forward, "_is_triton_fused_add_norm_patched", False):
                    def make_patched_layer(old_layer_fwd):
                        def patched_layer(self, hidden_states, position_embeddings, attention_mask=None,
                                          position_ids=None, past_key_values=None, **kwargs):
                            residual = hidden_states
                            hidden_states = self.input_layernorm(hidden_states)

                            if self.block_type == "linear_attention":
                                hidden_states = self.linear_attn(
                                    hidden_states=hidden_states,
                                    cache_params=past_key_values,
                                    attention_mask=attention_mask,
                                    **kwargs,
                                )
                            elif self.block_type == "full_attention":
                                hidden_states, _ = self.self_attn(
                                    hidden_states=hidden_states,
                                    attention_mask=attention_mask,
                                    position_ids=position_ids,
                                    past_key_values=past_key_values,
                                    position_embeddings=position_embeddings,
                                    **kwargs,
                                )

                            seq_len = hidden_states.shape[1]
                            if FUSED_KERNELS_ENABLED and hidden_states.is_cuda and seq_len == 1 and HAVE_TRITON:
                                residual, hidden_states = triton_fused_add_rmsnorm(
                                    residual, hidden_states, self.post_attention_layernorm.weight, self.post_attention_layernorm.eps
                                )
                            else:
                                hidden_states = residual + hidden_states
                                residual = hidden_states
                                hidden_states = self.post_attention_layernorm(hidden_states)

                            hidden_states = self.mlp(hidden_states)
                            if isinstance(hidden_states, tuple):
                                hidden_states, _ = hidden_states
                            hidden_states = residual + hidden_states
                            return hidden_states

                        patched_layer._is_triton_fused_add_norm_patched = True
                        return patched_layer
                    layer_cls.forward = make_patched_layer(layer_cls.forward)

            _PATCHED = True
        except Exception:
            pass

