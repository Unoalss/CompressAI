import sys, time
from pathlib import Path
import torch
sys.path.insert(0, r"e:\compressAI")
from safetensors import safe_open
from vq2 import vq

# Test on 1 expert tensor from Shard 1
f = safe_open(r"e:\compressAI\modelo_35b\model-00001-of-00021.safetensors", framework="pt")
w = f.get_tensor("model.language_model.layers.0.mlp.experts.gate_up_proj") # [256, 1024, 2048]
print(f"Tensor shape: {w.shape}, dtype: {w.dtype}, numel: {w.numel():,}")

w_2d = w.view(-1, 2048).float().cuda()
print(f"2D shape: {w_2d.shape}")

# Load universal codebooks
cbs_data = torch.load(r"e:\compressAI\vq2_out\codebooks.pt", weights_only=False)
cbs = vq.Codebooks({k: v.float().cuda() for k, v in cbs_data.items()})

# Test Rotation
rot = vq.get_rot("test_moe_2048", 2048, "cuda")
t0 = time.time()
wr = rot.fwd(w_2d)
dt_rot = time.time() - t0
print(f"Rotacao fwd em {dt_rot:.2f}s")

# Test Quantization with cfg=(16,) (2.0 bpw)
s = wr.pow(2).mean(1).sqrt().clamp_min(1e-6).half().float()
# Reshape into 8-D sub-vectors: [262144 * 256, 8] = [67108864, 8]
wr_8d = (wr * (1.0 / s)[:, None]).view(-1, 8)
t0 = time.time()
# Process in chunks of 8-D vectors to keep GPU memory light
CHUNK_8D = 1 << 20 # 1M vectors per chunk (~8M floats)
ids_list, q_list = [], []
for i in range(0, wr_8d.shape[0], CHUNK_8D):
    i_end = min(i + CHUNK_8D, wr_8d.shape[0])
    ids_chunk, q_chunk = vq.encode(wr_8d[i:i_end], (16,), cbs)
    ids_list.append(ids_chunk[0])
    q_list.append(q_chunk)
q_all = torch.cat(q_list, dim=0).view_as(wr)
dt_quant = time.time() - t0
print(f"Quantizacao 2.0 bpw em {dt_quant:.2f}s! ({w.numel() / 1e6 / dt_quant:.1f} Mparams/s)")

# Compute reconstruction SNR and MSE
rec = q_all * s[:, None]
mse = ((wr - rec)**2).mean().item()
orig_var = wr.var().item()
snr = 10 * torch.log10(torch.tensor(orig_var / mse)).item()
print(f"MSE: {mse:.6f} | SNR: {snr:.2f} dB")
