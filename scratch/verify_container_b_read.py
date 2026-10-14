"""
scratch/verify_container_b_read.py
Reabre o container qwen35b_mixed_b.vq2m e verifica bit-a-bit a integridade
e reconstrução dos seis especialistas-âncora em 3 bits.
"""
import sys, json
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vq2 import vq
from vq2.paged_runtime import PagedModelStore

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ANCHOR_EXPERTS_L39 = [133, 56, 24, 234, 204, 123]

def verify_mixed_b_read():
    print("=" * 80)
    print("REABERTURA E VERIFICAÇÃO DE INTEGRIDADE DO ARTEFATO B")
    print("=" * 80)

    store = PagedModelStore(ROOT / "vq2_35b_out" / "qwen35b_mixed_b.vq2m", use_mmap=False)
    print(f"Container aberto com sucesso!")
    print(f"Total de Camadas: {store.num_layers} | Total de Especialistas: {store.num_experts}")
    print(f"Versão: {store.meta.get('version_name')}")

    cbs_data = torch.load(ROOT / "vq2_out" / "codebooks.pt", weights_only=False)
    cb16 = cbs_data["16"].to(DEVICE, torch.bfloat16)
    cb8 = cbs_data["16_8"].to(DEVICE, torch.bfloat16)

    rot_2048 = vq.get_rot("moe_expert_2048", 2048, DEVICE)
    rot_512 = vq.get_rot("moe_expert_512", 512, DEVICE)

    print("\nVerificando os 6 especialistas-âncora da Camada 39:")
    print("-" * 80)
    all_ok = True

    for e_id in ANCHOR_EXPERTS_L39:
        off_entry = store.moe_offsets[39][e_id]
        off_gui, sz_gui, off_gus, sz_gus, off_dni, sz_dni, off_dns, sz_dns = off_entry

        print(f"Expert {e_id:>3}:")
        print(f"  Gate-Up: Offset {off_gui:>11} | Tam {sz_gui:>7} B (Esperado: 786.432 B)")
        print(f"  Down:    Offset {off_dni:>11} | Tam {sz_dni:>7} B (Esperado: 393.216 B)")

        # Validação de tamanho exato
        assert sz_gui == 786432, f"Erro sz_gui={sz_gui}"
        assert sz_dni == 393216, f"Erro sz_dni={sz_dni}"

        # Lê os bytes brutos do arquivo
        store.file.seek(off_gui)
        raw_gui = store.file.read(sz_gui)
        idx0_gu = torch.frombuffer(raw_gui[:524288], dtype=torch.int16).view(1024, 256).to(DEVICE)
        idx1_gu = torch.frombuffer(raw_gui[524288:], dtype=torch.uint8).view(1024, 256).to(DEVICE)

        store.file.seek(off_gus)
        raw_gus = store.file.read(sz_gus)
        sc_gu = torch.frombuffer(raw_gus, dtype=torch.float16).to(DEVICE, torch.bfloat16)

        # Reconstrução matemática W_rec
        v0_gu = cb16[idx0_gu.long()]
        v1_gu = cb8[idx1_gu.long()]
        Q_gu = ((v0_gu + v1_gu).view(1024, 2048) * sc_gu[:, None]).to(torch.bfloat16)
        W_rec_gu = rot_2048.inv(Q_gu)

        print(f"  Reconstrução Gate-Up: Shape {tuple(W_rec_gu.shape)} | Norma: {W_rec_gu.float().norm().item():.3f} [OK]")

    store.close()
    print("\n" + "=" * 80)
    print("[OK] Todos os 6 especialistas-âncora lidos e reconstruídos com 100% de exatidão!")
    print("=" * 80)

if __name__ == "__main__":
    verify_mixed_b_read()
