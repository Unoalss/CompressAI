import sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vq2.kernels import triton_conv1d_update
import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as qwen_mod

def test_equivalence():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo de teste: {device}")
    assert device == "cuda", "CUDA é necessário para testar kernel Triton"

    torch.manual_seed(42)
    batch_size = 1
    hidden_size = 8192
    state_len = 4
    dtype = torch.bfloat16

    # 1. Tensores de entrada
    weight = torch.randn(hidden_size, state_len, device=device, dtype=dtype)
    bias = None

    # Estado inicial aleatório
    cs_ref = torch.randn(batch_size, hidden_size, state_len, device=device, dtype=dtype)
    cs_tri = cs_ref.clone()

    pytorch_fn = qwen_mod.causal_conv1d_update

    print("\n--- Testando 5 passos sequenciais (t = 1..5) ---")
    for step in range(1, 6):
        hs = torch.randn(batch_size, hidden_size, 1, device=device, dtype=dtype)
        
        # Referência PyTorch
        out_ref = pytorch_fn(hs, cs_ref, weight, bias=bias, activation="silu")
        
        # Triton Kernel
        out_tri = triton_conv1d_update(hs, cs_tri, weight, b=bias)

        # Comparação da saída do token
        diff_out = torch.max(torch.abs(out_ref.float() - out_tri.float())).item()
        cos_out = F.cosine_similarity(out_ref.flatten().float(), out_tri.flatten().float(), dim=0).item()
        
        # Comparação do estado causal atualizado
        diff_cs = torch.max(torch.abs(cs_ref.float() - cs_tri.float())).item()
        cos_cs = F.cosine_similarity(cs_ref.flatten().float(), cs_tri.flatten().float(), dim=0).item()

        print(f"Passo {step}:")
        print(f"  Saída token:     Max Diff = {diff_out:.6f} | Cosseno = {cos_out:.8f}")
        print(f"  Estado causal:   Max Diff = {diff_cs:.6f} | Cosseno = {cos_cs:.8f}")

        # Tolerância numérica para bfloat16:
        # BF16 tem mantissa de 8 bits (eps ~ 0.0078), então diff < 0.05 e cosseno > 0.9999
        if diff_out > 0.05 or diff_cs > 0.05:
            print(f"  [ALERTA] Diferença acima do esperado no passo {step}!")
            print(f"  Amostra out_ref[:5]: {out_ref[0, :5, 0].tolist()}")
            print(f"  Amostra out_tri[:5]: {out_tri[0, :5, 0].tolist()}")
            print(f"  Amostra cs_ref[:2, :]: {cs_ref[0, 0, :].tolist()}")
            print(f"  Amostra cs_tri[:2, :]: {cs_tri[0, 0, :].tolist()}")
            return False

    print("\n[SUCESSO] Kernel Triton e PyTorch são equivalentes nos 5 passos!")
    return True

if __name__ == "__main__":
    test_equivalence()
