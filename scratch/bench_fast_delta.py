import sys, time
sys.path.insert(0, r"e:\compressAI")
import torch
import torch.nn.functional as F

DEVICE = "cuda"

B, H, K_DIM, V_DIM = 1, 32, 128, 128
state = torch.randn(B, H, K_DIM, V_DIM, device=DEVICE, dtype=torch.float32)
q = torch.randn(B, 1, H, K_DIM, device=DEVICE, dtype=torch.bfloat16)
k = torch.randn(B, 1, H, K_DIM, device=DEVICE, dtype=torch.bfloat16)
v = torch.randn(B, 1, H, V_DIM, device=DEVICE, dtype=torch.bfloat16)
decay = torch.randn(B, H, 1, device=DEVICE, dtype=torch.float32)
beta = torch.randn(B, H, 1, device=DEVICE, dtype=torch.float32)

def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt(torch.sum(x**2, dim=dim, keepdim=True) + eps)

def original_step(query, key, value, g, beta, last_recurrent_state):
    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, g)
    ]
    query = l2norm(query, dim=-1, eps=1e-6)
    key = l2norm(key, dim=-1, eps=1e-6)
    query = query / (query.shape[-1] ** 0.5)
    
    q_t, k_t, v_t = query[:, :, 0], key[:, :, 0], value[:, :, 0]
    decay_t = decay[:, :, 0].exp()[..., None, None]
    last_recurrent_state = last_recurrent_state * decay_t
    beta_t = beta[:, :, 0].unsqueeze(-1)
    kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
    delta = (v_t - kv_mem) * beta_t
    last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
    core_attn_out = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)
    return core_attn_out.transpose(1, 2).contiguous().to(torch.bfloat16), last_recurrent_state

def fast_step(query, key, value, g, beta, state):
    # query: (1, 1, H, K)
    q_norm = l2norm(query.squeeze(1).float(), dim=-1) * (1.0 / (K_DIM ** 0.5)) # (1, H, K)
    k_norm = l2norm(key.squeeze(1).float(), dim=-1) # (1, H, K)
    v_f = value.squeeze(1).float() # (1, H, V)
    
    # 1. Decay state in-place: g is (1, H, 1) -> (1, H, 1, 1)
    state.mul_(g.exp().unsqueeze(-1))
    
    # 2. kv_mem = S^T @ k: S is (H, K, V), k_norm is (H, 1, K)
    S = state[0] # (H, K, V)
    k_h = k_norm[0].unsqueeze(1) # (H, 1, K)
    kv_mem = torch.bmm(k_h, S).squeeze(1) # (H, V)
    
    # 3. delta = (v - kv_mem) * beta
    delta = (v_f[0] - kv_mem) * beta[0] # (H, V)
    
    # 4. update state: S += k^T @ delta -> (H, K, 1) @ (H, 1, V) -> (H, K, V)
    torch.baddbmm(S, k_h.transpose(1, 2), delta.unsqueeze(1), out=S)
    
    # 5. core_attn_out = q @ S -> (H, 1, K) @ (H, K, V) -> (H, 1, V)
    q_h = q_norm[0].unsqueeze(1)
    out = torch.bmm(q_h, S).squeeze(1) # (H, V)
    return out.unsqueeze(0).unsqueeze(1).to(torch.bfloat16), state

s1 = state.clone()
s2 = state.clone()
o1, s1 = original_step(q, k, v, decay, beta, s1)
o2, s2 = fast_step(q, k, v, decay, beta, s2)
print("o1 shape:", o1.shape)
print("o2 shape:", o2.shape)
print("Diferença max saída:", (o1 - o2).abs().max().item())
print("Diferença max estado:", (s1 - s2).abs().max().item())

N = 2000
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    o1, s1 = original_step(q, k, v, decay, beta, s1)
torch.cuda.synchronize()
dt_orig = (time.perf_counter() - t0) / N * 1000

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    o2, s2 = fast_step(q, k, v, decay, beta, s2)
torch.cuda.synchronize()
dt_fast = (time.perf_counter() - t0) / N * 1000

print(f"Original: {dt_orig:.4f} ms")
print(f"Fast baddbmm: {dt_fast:.4f} ms | Aceleração: {dt_orig/dt_fast:.2f}x!")
