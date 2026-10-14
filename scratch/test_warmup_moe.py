import sys, time
sys.path.insert(0, r"e:\compressAI")
import torch
from vq2.load_and_test_full_35b import PackedMoEExpertUniversal

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
s1 = torch.load("e:/compressAI/vq2_35b_out/quant_model-00001-of-00021.safetensors.pt", weights_only=False, map_location="cpu")["quant"]
cbs_data = torch.load("e:/compressAI/vq2_out/codebooks.pt", weights_only=False)
cb16 = cbs_data["16"].to(device=DEVICE, dtype=torch.bfloat16)

expert_mod = PackedMoEExpertUniversal(
    s1["model.language_model.layers.0.mlp.experts.gate_up_proj"],
    s1["model.language_model.layers.0.mlp.experts.down_proj"],
    cb16, device=DEVICE
)

h = torch.randn(1, 2048, device=DEVICE, dtype=torch.bfloat16)
topk_idx = torch.tensor([[5, 12, 47, 88, 120, 155, 201, 240]], device=DEVICE)
topk_w = torch.tensor([[0.25, 0.20, 0.15, 0.10, 0.10, 0.08, 0.07, 0.05]], device=DEVICE, dtype=torch.bfloat16)

# Warmup Triton JIT compilation!
for _ in range(10):
    _ = expert_mod(h, topk_idx, topk_w)
torch.cuda.synchronize()

N = 200
t0 = time.perf_counter()
for _ in range(N):
    _ = expert_mod(h, topk_idx, topk_w)
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / N * 1000
print(f"PackedMoEExpertUniversal apos warmup: {dt:.4f} ms ({dt*1000:.1f} us)!")
print(f"40 camadas MoE: {dt * 40:.2f} ms")
print(f"Velocidade MoE: {1000.0 / (dt * 40):.1f} tok/s")
