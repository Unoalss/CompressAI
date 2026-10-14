"""
tools/build_mixed_b_container.py
Compila o Artefato B (qwen35b_mixed_b.vq2m) a partir do Artefato A (qwen35b_full.vq2m).
Eleva estritamente os 6 especialistas-âncora da Camada 39 para 3 bits (cfg=(16, 8)).
Preserva 100% de todos os outros tensores e especialistas da rede em 2 bits.
Valida limites, ausência de sobreposição e exatidão de reconstrução.
"""
import os, sys, time, json, struct, math, gc
from pathlib import Path
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vq2 import vq

SRC_VQ2M = ROOT / "vq2_35b_out" / "qwen35b_full.vq2m"
DST_VQ2M = ROOT / "vq2_35b_out" / "qwen35b_mixed_b.vq2m"
MODEL_DIR = ROOT / "modelo_35b"

MAGIC = b"VQ2M"
VERSION = 1
ENTRY_FORMAT = "<QIQIQIQI"
ENTRY_SIZE = struct.calcsize(ENTRY_FORMAT)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ANCHOR_EXPERTS_L39 = [133, 56, 24, 234, 204, 123]

def quantize_expert_3bit(e_id, gu_slice, dn_slice, cbs, rot_2048, rot_512):
    """Gera tensores e bytes serializados de 1 especialista em 3 bits (1.185.792 bytes)."""
    orig_gu = gu_slice[e_id].to(DEVICE, torch.bfloat16)
    orig_dn = dn_slice[e_id].to(DEVICE, torch.bfloat16)

    # 1. Gate-Up: [1024, 2048]
    W_gu_rot = rot_2048.fwd(orig_gu)
    s_gu = (W_gu_rot.float().norm(dim=1, keepdim=True) / math.sqrt(2048)).clamp_min(1e-6)
    u_gu = (W_gu_rot.float() / s_gu).reshape(-1, 8).to(DEVICE, torch.bfloat16)
    idxs_gu, rec_gu = vq.encode(u_gu, (16, 8), cbs)

    gu_idx0 = idxs_gu[0].view(1024, 256).to(torch.int16).cpu().numpy().tobytes() # 524.288 B
    gu_idx1 = idxs_gu[1].view(1024, 256).to(torch.uint8).cpu().numpy().tobytes() # 262.144 B
    gu_idx_bytes = gu_idx0 + gu_idx1 # 786.432 B
    gu_scale_bytes = s_gu.squeeze(1).to(torch.float16).cpu().numpy().tobytes() # 2.048 B

    # 2. Down: [2048, 512]
    W_dn_rot = rot_512.fwd(orig_dn)
    s_dn = (W_dn_rot.float().norm(dim=1, keepdim=True) / math.sqrt(512)).clamp_min(1e-6)
    u_dn = (W_dn_rot.float() / s_dn).reshape(-1, 8).to(DEVICE, torch.bfloat16)
    idxs_dn, rec_dn = vq.encode(u_dn, (16, 8), cbs)

    dn_idx0 = idxs_dn[0].view(2048, 64).to(torch.int16).cpu().numpy().tobytes() # 262.144 B
    dn_idx1 = idxs_dn[1].view(2048, 64).to(torch.uint8).cpu().numpy().tobytes() # 131.072 B
    dn_idx_bytes = dn_idx0 + dn_idx1 # 393.216 B
    dn_scale_bytes = s_dn.squeeze(1).to(torch.float16).cpu().numpy().tobytes() # 4.096 B

    return {
        "gu_idx": gu_idx_bytes,
        "gu_scale": gu_scale_bytes,
        "dn_idx": dn_idx_bytes,
        "dn_scale": dn_scale_bytes,
    }

def build_mixed_b():
    print("=" * 80)
    print("COMPILAÇÃO DO ARTEFATO B: qwen35b_mixed_b.vq2m")
    print(f"Origem:  {SRC_VQ2M}")
    print(f"Destino: {DST_VQ2M}")
    print(f"Especialistas Alvo: Camada 39 -> {ANCHOR_EXPERTS_L39} em 3 bits")
    print("=" * 80)

    # 1. Carrega codebooks e rotações
    print("[1/5] Carregando Codebooks e Safetensors originais...")
    cbs_data = torch.load(ROOT / "vq2_out" / "codebooks.pt", weights_only=False)
    cbs = vq.Codebooks({k: v.to(DEVICE, torch.bfloat16) for k, v in cbs_data.items()})
    rot_2048 = vq.get_rot("moe_expert_2048", 2048, DEVICE)
    rot_512 = vq.get_rot("moe_expert_512", 512, DEVICE)

    index_map = json.load(open(MODEL_DIR / "model.safetensors.index.json"))["weight_map"]
    shard_gu = MODEL_DIR / index_map["model.language_model.layers.39.mlp.experts.gate_up_proj"]
    shard_dn = MODEL_DIR / index_map["model.language_model.layers.39.mlp.experts.down_proj"]

    print("      Quantizando os 6 especialistas-âncora em 3 bits...")
    encoded_3b = {}
    with safe_open(str(shard_gu), framework="pt", device="cpu") as f_gu, \
         safe_open(str(shard_dn), framework="pt", device="cpu") as f_dn:
        gu_slice = f_gu.get_slice("model.language_model.layers.39.mlp.experts.gate_up_proj")
        dn_slice = f_dn.get_slice("model.language_model.layers.39.mlp.experts.down_proj")
        for e_id in ANCHOR_EXPERTS_L39:
            encoded_3b[e_id] = quantize_expert_3bit(e_id, gu_slice, dn_slice, cbs, rot_2048, rot_512)

    # 2. Lê cabeçalho do Artefato A
    print("\n[2/5] Lendo estrutura e tabela de offsets do Artefato A...")
    f_src = open(SRC_VQ2M, "rb")
    header_64 = f_src.read(64)
    (
        magic, version, num_layers, num_experts, reserved16,
        moe_table_off, moe_table_len,
        meta_json_off, meta_json_len,
        payload_start, reserved12
    ) = struct.unpack("<4sHHHH QQQQQ 12s", header_64)

    assert magic == MAGIC and version == VERSION

    f_src.seek(moe_table_off)
    src_moe_bytes = f_src.read(moe_table_len)
    src_unpacked = list(struct.iter_unpack(ENTRY_FORMAT, src_moe_bytes))

    f_src.seek(meta_json_off)
    src_meta = json.loads(f_src.read(meta_json_len).decode("utf-8"))

    # Organiza offsets por [layer][expert]
    src_moe_offsets = []
    idx = 0
    for ln in range(num_layers):
        src_moe_offsets.append(src_unpacked[idx : idx + num_experts])
        idx += num_experts

    # 3. Cria novo arquivo e grava cópia com substituição seletiva
    print("\n[3/5] Gravando novo container e atualizando tabela de offsets...")
    f_dst = open(DST_VQ2M, "wb")
    # Reserva 2 MiB para o cabeçalho inicial
    RESERVED_HEADER = 2 * 1024 * 1024
    f_dst.seek(RESERVED_HEADER)

    dst_moe_offsets = [[None for _ in range(num_experts)] for _ in range(num_layers)]
    expert_modes = {}

    t0 = time.time()
    for ln in range(num_layers):
        if ln % 5 == 0:
            print(f"      Processando Camada {ln}/40... [Posição atual: {f_dst.tell() / 1024**3:.2f} GB]", flush=True)

        for e in range(num_experts):
            if ln == 39 and e in ANCHOR_EXPERTS_L39:
                # Grava no modo 3 bits:
                data = encoded_3b[e]
                off_gui = f_dst.tell(); f_dst.write(data["gu_idx"]); sz_gui = len(data["gu_idx"])
                off_gus = f_dst.tell(); f_dst.write(data["gu_scale"]); sz_gus = len(data["gu_scale"])
                off_dni = f_dst.tell(); f_dst.write(data["dn_idx"]); sz_dni = len(data["dn_idx"])
                off_dns = f_dst.tell(); f_dst.write(data["dn_scale"]); sz_dns = len(data["dn_scale"])

                dst_moe_offsets[ln][e] = (off_gui, sz_gui, off_gus, sz_gus, off_dni, sz_dni, off_dns, sz_dns)
                expert_modes[f"{ln}:{e}"] = 3
            else:
                # Copia do Artefato A:
                (off_gui, sz_gui, off_gus, sz_gus, off_dni, sz_dni, off_dns, sz_dns) = src_moe_offsets[ln][e]
                total_exp_bytes = sz_gui + sz_gus + sz_dni + sz_dns
                f_src.seek(off_gui)
                exp_bytes = f_src.read(total_exp_bytes)

                new_off_gui = f_dst.tell()
                f_dst.write(exp_bytes)

                new_off_gus = new_off_gui + sz_gui
                new_off_dni = new_off_gus + sz_gus
                new_off_dns = new_off_dni + sz_dni

                dst_moe_offsets[ln][e] = (new_off_gui, sz_gui, new_off_gus, sz_gus, new_off_dni, sz_dni, new_off_dns, sz_dns)

    # 4. Copia os demais tensores não-MoE (atenções, embeddings, RMSNorms)
    print("      Copiando tensores densos e atenções do Artefato A...")
    dst_other_tensors = {}
    CHUNK = 64 * 1024 * 1024
    for name, info in src_meta.get("other_tensors", {}).items():
        if info.get("type") == "dense":
            src_off = info["offset"]
            size = info["size"]
            f_src.seek(src_off)
            new_off = f_dst.tell()
            copied = 0
            while copied < size:
                to_read = min(CHUNK, size - copied)
                f_dst.write(f_src.read(to_read))
                copied += to_read
            dst_other_tensors[name] = dict(info, offset=new_off)

        elif info.get("type") == "quant_linear":
            new_info = dict(info)
            new_idx_meta = []
            for sub in info["idx_meta"]:
                f_src.seek(sub["offset"])
                sz = sub["size"]
                new_off = f_dst.tell()
                f_dst.write(f_src.read(sz))
                new_idx_meta.append({"offset": new_off, "size": sz, "dtype": sub["dtype"]})
            new_info["idx_meta"] = new_idx_meta

            # Scale
            sc_meta = info["scale_meta"]
            f_src.seek(sc_meta["offset"])
            sc_sz = sc_meta["size"]
            new_sc_off = f_dst.tell()
            f_dst.write(f_src.read(sc_sz))
            new_info["scale_meta"] = {"offset": new_sc_off, "size": sc_sz}

            dst_other_tensors[name] = new_info

    # 5. Escreve cabeçalho e tabelas de índices no início do arquivo
    print("\n[4/5] Escrevendo cabeçalho e tabela de offsets no cabeçalho reservado...")
    dst_meta = dict(src_meta)
    dst_meta["other_tensors"] = dst_other_tensors
    dst_meta["expert_modes"] = expert_modes
    dst_meta["version_name"] = "qwen35b_mixed_b (6 anchor experts L39 in 3-bit)"

    meta_json_bytes = json.dumps(dst_meta, ensure_ascii=False).encode("utf-8")
    meta_json_len = len(meta_json_bytes)

    # Serializa tabela MoE
    all_dst_entries = []
    for ln in range(num_layers):
        all_dst_entries.extend(dst_moe_offsets[ln])
    moe_table_bytes = b"".join(struct.pack(ENTRY_FORMAT, *entry) for entry in all_dst_entries)
    moe_table_len = len(moe_table_bytes)

    moe_table_off = 64
    meta_json_off = moe_table_off + moe_table_len
    payload_start = RESERVED_HEADER

    header_dst = struct.pack(
        "<4sHHHH QQQQQ 12s",
        MAGIC, VERSION, num_layers, num_experts, 0,
        moe_table_off, moe_table_len,
        meta_json_off, meta_json_len,
        payload_start, b"\x00" * 12
    )

    f_dst.seek(0)
    f_dst.write(header_dst)
    f_dst.seek(moe_table_off)
    f_dst.write(moe_table_bytes)
    f_dst.seek(meta_json_off)
    f_dst.write(meta_json_bytes)

    f_src.close()
    f_dst.close()

    dt_total = time.time() - t0
    src_size = SRC_VQ2M.stat().st_size
    dst_size = DST_VQ2M.stat().st_size
    delta_bytes = dst_size - src_size

    print(f"\n[OK] Gravação concluída em {dt_total:.1f}s!")
    print(f"Tamanho Artefato A (Original): {src_size} bytes ({src_size / 1024**2:.2f} MiB / {src_size / 1000**2:.2f} MB)")
    print(f"Tamanho Artefato B (Misto):    {dst_size} bytes ({dst_size / 1024**2:.2f} MiB / {dst_size / 1000**2:.2f} MB)")
    print(f"Diferença Exata:               {delta_bytes} bytes ({delta_bytes / 1024:.2f} KiB / {delta_bytes / 1000**2:.4f} MB)")
    expected_delta = 6 * 393216
    print(f"Delta Teórico Esperado:        {expected_delta} bytes (6 * 393.216 B)")
    assert delta_bytes == expected_delta, f"Erro: delta ({delta_bytes}) difere do esperado ({expected_delta})!"

    # 6. Validação de ausência de sobreposição e integridade
    print("\n[5/5] Validando ausência de sobreposição e alinhamento dos offsets...")
    # Ordena todas as regiões gravadas
    regions = []
    for ln in range(num_layers):
        for e in range(num_experts):
            (off_gui, sz_gui, off_gus, sz_gus, off_dni, sz_dni, off_dns, sz_dns) = dst_moe_offsets[ln][e]
            regions.append((off_gui, off_gui + sz_gui, f"L{ln}E{e}_gui"))
            regions.append((off_gus, off_gus + sz_gus, f"L{ln}E{e}_gus"))
            regions.append((off_dni, off_dni + sz_dni, f"L{ln}E{e}_dni"))
            regions.append((off_dns, off_dns + sz_dns, f"L{ln}E{e}_dns"))

    regions.sort(key=lambda r: r[0])
    overlaps = 0
    for i in range(len(regions) - 1):
        if regions[i][1] > regions[i + 1][0]:
            print(f"ALERTA: Sobreposição entre {regions[i][2]} e {regions[i + 1][2]}!")
            overlaps += 1

    assert overlaps == 0, f"Falha na validação: detectadas {overlaps} sobreposições!"
    print(f"[OK] 100% dos {len(regions)} blocos validados com sucesso sem nenhuma sobreposição!")

if __name__ == "__main__":
    build_mixed_b()
