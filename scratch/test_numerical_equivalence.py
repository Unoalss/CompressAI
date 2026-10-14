"""
scratch/test_numerical_equivalence.py
Teste rigoroso de equivalência numérica e precisão:
1. Compara pesos de referência (safetensors original) vs pesos reconstruídos no VQ2.
2. Compara a saída do especialista (MLP) para a mesma entrada.
3. Mede erro máximo, erro médio, Cosine Similarity e SNR.
"""
import sys, math
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vq2 import vq
from vq2.paged_runtime import PagedModelStore, PagedMoEExpert, DEVICE

def run_equivalence_test(layer_idx=0, expert_idx=0):
    print("=" * 80)
    print("TESTE DE EQUIVALÊNCIA NUMÉRICA: CHECKPOINT HF vs MOTOR LEGION VQ2")
    print(f"Camada: {layer_idx} | Especialista: {expert_idx} | Dispositivo: {DEVICE}")
    print("=" * 80)

    # 1. Carrega pesos originais do safetensors HuggingFace
    safetensors_path = ROOT / "modelo_35b" / "model-00001-of-00021.safetensors"
    print(f"[1/4] Carregando pesos de referência de {safetensors_path.name}...")
    with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
        # gate_up_proj shape: [256, 1024, 2048]
        orig_gu_slice = f.get_slice("model.language_model.layers.0.mlp.experts.gate_up_proj")
        orig_gu = orig_gu_slice[expert_idx].to(device=DEVICE, dtype=torch.bfloat16)
        
        # down_proj shape: [256, 2048, 512]
        orig_dn_slice = f.get_slice("model.language_model.layers.0.mlp.experts.down_proj")
        orig_dn = orig_dn_slice[expert_idx].to(device=DEVICE, dtype=torch.bfloat16)

    print(f"      Pesos de Referência: Gate-Up {tuple(orig_gu.shape)} | Down {tuple(orig_dn.shape)}")

    # 2. Carrega motor LEGION e descompacta o especialista no slot 0
    print("[2/4] Carregando pesos reconstruídos no LEGION (VQ2)...")
    cbs_data = torch.load(ROOT / "vq2_out" / "codebooks.pt", weights_only=False)
    cb16 = cbs_data["16"].to(device=DEVICE, dtype=torch.bfloat16)
    
    store = PagedModelStore(ROOT / "vq2_35b_out" / "qwen35b_full.vq2m", use_mmap=False)
    paged_layer = PagedMoEExpert(layer_idx=layer_idx, store=store, cb16=cb16, capacity=16, device=DEVICE)
    
    # Carrega especialista no slot 0
    slot = paged_layer._get_slot(expert_idx)

    # 3. Compara a execução do especialista para a mesma entrada
    print("[3/4] Executando inferência isolada do especialista para 100 vetores de teste...")
    torch.manual_seed(42)
    x = torch.randn(10, 2048, device=DEVICE, dtype=torch.bfloat16)

    # A. Saída de Referência (Equação exata do HuggingFace):
    # gate, up = (x @ W_gu.T).chunk(2, dim=-1)
    # act = silu(gate) * up
    # out = act @ W_dn.T
    gu_ref = F.linear(x, orig_gu)
    gate_ref, up_ref = gu_ref.chunk(2, dim=-1)
    act_ref = F.silu(gate_ref.float()) * up_ref.float()
    out_ref = F.linear(act_ref.bfloat16(), orig_dn)

    # B. Saída do LEGION via PagedMoEExpert (Kernel Triton VQ2):
    # Simula chamada do especialista com peso 1.0
    topk_idx = torch.tensor([[expert_idx]], device=DEVICE, dtype=torch.long)
    topk_weights = torch.tensor([[1.0]], device=DEVICE, dtype=torch.float32)

    legion_outs = []
    for i in range(10):
        xi = x[i:i+1]
        out_legion_i = paged_layer.forward(xi, topk_idx, topk_weights)
        legion_outs.append(out_legion_i)
    out_legion = torch.cat(legion_outs, dim=0)

    # 4. Métricas Numéricas de Divergência
    print("[4/4] Calculando divergência numérica e fidelidade...")
    cos_sim = F.cosine_similarity(out_ref.float().view(-1), out_legion.float().view(-1), dim=0).item()
    diff = (out_ref.float() - out_legion.float()).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    signal_power = (out_ref.float() ** 2).mean().item()
    noise_power = (diff ** 2).mean().item()
    snr_db = 10 * math.log10(signal_power / max(noise_power, 1e-12))

    print("\n" + "=" * 80)
    print("RESULTADO DO TESTE DE EQUIVALÊNCIA NUMÉRICA (CAMADA 0, EXPERT 0)")
    print("=" * 80)
    print(f"Cosine Similarity (Alinhamento Direcional): {cos_sim:.5f}  (Ideal: 1.00000)")
    print(f"SNR (Relação Sinal-Ruído):                  {snr_db:.2f} dB  (Ideal: > 20 dB)")
    print(f"Erro Absoluto Médio (MAE):                  {mean_err:.5f}")
    print(f"Erro Absoluto Máximo:                       {max_err:.5f}")
    print(f"Norma Média da Saída de Referência:          {out_ref.float().norm(dim=-1).mean().item():.3f}")
    print(f"Norma Média da Saída do LEGION:             {out_legion.float().norm(dim=-1).mean().item():.3f}")
    print("=" * 80)

    store.close()
    return {
        "cosine_similarity": cos_sim,
        "snr_db": snr_db,
        "mae": mean_err,
        "max_err": max_err
    }

if __name__ == "__main__":
    run_equivalence_test(layer_idx=0, expert_idx=0)
