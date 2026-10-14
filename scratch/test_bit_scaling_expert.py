"""
scratch/test_bit_scaling_expert.py
Mede a sensibilidade e o ganho empírico de precisão ao quantizar o especialista (Camada 0, Expert 0)
em 2 bits (16,), 3 bits (16, 8) e 4 bits (16, 16) usando os codebooks existentes.
"""
import sys, math
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vq2 import vq
from vq2.common import DEV

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def test_bit_scaling(layer_idx=0, expert_idx=0, num_samples=1000):
    print("=" * 80)
    print(f"BENCHMARK EMPÍRICO DE ESCALONAMENTO DE PRECISÃO: 2 vs 3 vs 4 BITS")
    print(f"Camada: {layer_idx} | Especialista: {expert_idx} | Dispositivo: {DEVICE}")
    print("=" * 80)

    # 1. Carrega pesos originais HuggingFace (bfloat16)
    safetensors_path = ROOT / "modelo_35b" / "model-00001-of-00021.safetensors"
    with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
        orig_gu = f.get_slice("model.language_model.layers.0.mlp.experts.gate_up_proj")[expert_idx].to(DEVICE, torch.bfloat16)
        orig_dn = f.get_slice("model.language_model.layers.0.mlp.experts.down_proj")[expert_idx].to(DEVICE, torch.bfloat16)

    # 2. Carrega Codebooks
    cbs_data = torch.load(ROOT / "vq2_out" / "codebooks.pt", weights_only=False)
    cbs = vq.Codebooks({k: v.to(DEVICE, torch.bfloat16) for k, v in cbs_data.items()})

    rot_2048 = vq.get_rot("moe_expert_2048", 2048, DEVICE)
    rot_512 = vq.get_rot("moe_expert_512", 512, DEVICE)

    # Vetores de teste para o MLP
    torch.manual_seed(42)
    x = torch.randn(num_samples, 2048, device=DEVICE, dtype=torch.bfloat16)

    # Saída de Referência FP16 / BF16
    gu_ref = F.linear(x, orig_gu)
    gate_ref, up_ref = gu_ref.chunk(2, dim=-1)
    act_ref = F.silu(gate_ref.float()) * up_ref.float()
    out_ref = F.linear(act_ref.bfloat16(), orig_dn)
    norm_ref = out_ref.float().norm(dim=-1).mean().item()

    configs = [
        ("2 bits (16,)", (16,), 792576),
        ("3 bits (16, 8)", (16, 8), int(792576 * 1.5)),
        ("4 bits (16, 16)", (16, 16), int(792576 * 2.0)),
    ]

    results = []

    for name, cfg, bytes_est in configs:
        # Quantiza e reconstrói Gate-Up: [1024, 2048]
        # Rotaciona para base de incoerência:
        W_gu_rot = rot_2048.fwd(orig_gu)
        s_gu = W_gu_rot.float().norm(dim=1, keepdim=True) / math.sqrt(2048)
        u_gu = (W_gu_rot.float() / s_gu.clamp_min(1e-6)).reshape(-1, 8)
        idxs_gu, rec_gu = vq.encode(u_gu.to(DEVICE, torch.bfloat16), cfg, cbs)
        Q_gu = (rec_gu.view(1024, 2048) * s_gu.to(torch.bfloat16)).to(torch.bfloat16)
        W_rec_gu = rot_2048.inv(Q_gu)

        # Quantiza e reconstrói Down: [2048, 512]
        W_dn_rot = rot_512.fwd(orig_dn)
        s_dn = W_dn_rot.float().norm(dim=1, keepdim=True) / math.sqrt(512)
        u_dn = (W_dn_rot.float() / s_dn.clamp_min(1e-6)).reshape(-1, 8)
        idxs_dn, rec_dn = vq.encode(u_dn.to(DEVICE, torch.bfloat16), cfg, cbs)
        Q_dn = (rec_dn.view(2048, 512) * s_dn.to(torch.bfloat16)).to(torch.bfloat16)
        W_rec_dn = rot_512.inv(Q_dn)

        # Métricas dos Pesos
        cos_w_gu = F.cosine_similarity(orig_gu.float().view(-1), W_rec_gu.float().view(-1), dim=0).item()
        cos_w_dn = F.cosine_similarity(orig_dn.float().view(-1), W_rec_dn.float().view(-1), dim=0).item()
        snr_w_gu = 10 * math.log10((orig_gu.float()**2).mean().item() / max(((orig_gu.float() - W_rec_gu.float())**2).mean().item(), 1e-12))
        snr_w_dn = 10 * math.log10((orig_dn.float()**2).mean().item() / max(((orig_dn.float() - W_rec_dn.float())**2).mean().item(), 1e-12))

        # Execução PyTorch do MLP com os pesos quantizados
        y_gu = F.linear(x, W_rec_gu)
        g, u = y_gu.chunk(2, dim=-1)
        act = (F.silu(g.float()) * u.float()).bfloat16()
        out_vq = F.linear(act, W_rec_dn)

        # Métricas de Saída do Especialista
        cos_out = F.cosine_similarity(out_ref.float().view(-1), out_vq.float().view(-1), dim=0).item()
        diff_out = (out_ref.float() - out_vq.float()).abs()
        snr_out = 10 * math.log10((out_ref.float()**2).mean().item() / max((diff_out**2).mean().item(), 1e-12))
        norm_vq = out_vq.float().norm(dim=-1).mean().item()
        norm_ratio = norm_vq / norm_ref

        results.append({
            "name": name,
            "cfg": cfg,
            "bytes_kb": bytes_est / 1024,
            "cos_w_gu": cos_w_gu,
            "snr_w_gu": snr_w_gu,
            "cos_w_dn": cos_w_dn,
            "snr_w_dn": snr_w_dn,
            "cos_out": cos_out,
            "snr_out": snr_out,
            "norm_ratio": norm_ratio,
        })

    print(f"\n{'Configuração':<18} | {'Tamanho':<10} | {'W_gu Cos':<9} | {'W_gu SNR':<9} | {'Saída Cos':<10} | {'Saída SNR':<10} | {'Norm Ratio':<10}")
    print("-" * 88)
    for r in results:
        print(f"{r['name']:<18} | {r['bytes_kb']:>7.1f} KB | {r['cos_w_gu']:>9.4f} | {r['snr_w_gu']:>6.2f} dB | {r['cos_out']:>10.5f} | {r['snr_out']:>7.2f} dB | {r['norm_ratio']:>10.4f}")

    print("=" * 88)

if __name__ == "__main__":
    test_bit_scaling(layer_idx=0, expert_idx=0, num_samples=1000)
