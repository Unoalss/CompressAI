import sys, time
sys.path.insert(0, r"e:\compressAI")
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeTextConfig
)
from transformers.cache_utils import DynamicCache
from accelerate import init_empty_weights

from vq2 import vq
from vq2.kernels import HAVE_TRITON, patch_transformers_kernels
from vq2.load_and_test_full_35b import (
    PackedMoEExpertUniversal,
    dequant_matrix,
    set_nested_param
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_35B = r"e:\compressAI\vq2_35b_out"
MODEL_DIR = r"e:\compressAI\modelo_35b"

print("1. Perfilando custos isolados:")

# Teste A: Custo do lm_head na CPU vs GPU
print("Avaliando lm_head [248320, 2048]...")
last_hidden_cpu = torch.randn(1, 1, 2048, dtype=torch.float32)
lm_head_cpu = torch.randn(248320, 2048, dtype=torch.float32)

t0 = time.perf_counter()
for _ in range(10):
    logits = torch.matmul(last_hidden_cpu, lm_head_cpu.T)
t_lm_head_cpu = (time.perf_counter() - t0) / 10 * 1000
print(f"  lm_head na CPU (FP32 matmul): {t_lm_head_cpu:.2f} ms")

# Teste B: lm_head em FP16 / BF16 ou na GPU com float8 / bf16
last_hidden_cuda = torch.randn(1, 1, 2048, device=DEVICE, dtype=torch.bfloat16)
lm_head_cuda = lm_head_cpu.to(device=DEVICE, dtype=torch.bfloat16)
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(10):
    logits_cuda = torch.matmul(last_hidden_cuda, lm_head_cuda.T)
torch.cuda.synchronize()
t_lm_head_cuda = (time.perf_counter() - t0) / 10 * 1000
print(f"  lm_head na GPU (BF16 matmul): {t_lm_head_cuda:.2f} ms (Economia: {t_lm_head_cpu - t_lm_head_cuda:.2f} ms)")

# Teste C: Custo de 1 camada MoE Universal (Triton)
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

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(100):
    _ = expert_mod(h, topk_idx, topk_w)
torch.cuda.synchronize()
t_moe = (time.perf_counter() - t0) / 100 * 1000
print(f"  MoE Triton (1 camada, 8 experts): {t_moe:.3f} ms")
print(f"  Total MoE nas 40 camadas: {t_moe * 40:.2f} ms")

print("\nConclusão inicial:")
print(f"Se MoE = {t_moe*40:.1f}ms e lm_head_cpu = {t_lm_head_cpu:.1f}ms, onde estão os outros milissegundos?")
