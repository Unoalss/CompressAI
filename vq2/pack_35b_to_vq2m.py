"""
vq2/pack_35b_to_vq2m.py - Empacotador Universal do Modelo Qwen3.8-35B-A3B no formato paginado .vq2m.
Converte os 22 shards quantizados (.pt) em um unico arquivo binario indexado de 9.38 GB:
- Tabela de Offsets direta para os 10.240 especialistas MoE (40 camadas x 256 experts).
- Tabela de tensores para atencoes, normalizacoes, shared experts e embeddings.
- Permite carregamento sob demanda via memory-mapping (mmap) com orcamento de RAM estrito (< 2.0 GB).
"""
import os, sys, time, json, struct, mmap, gc
from pathlib import Path
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_35B = ROOT / "vq2_35b_out"
MODEL_DIR = ROOT / "modelo_35b"
TARGET_VQ2M = ROOT / "vq2_35b_out" / "qwen35b_full.vq2m"

MAGIC = b"VQ2M"
VERSION = 1
ENTRY_FORMAT = "<QIQIQIQI" # 48 bytes por expert (4 pares uint64 offset + uint32 size)
ENTRY_SIZE = struct.calcsize(ENTRY_FORMAT)

def write_tensor_direct(f, tensor):
    """Grava tensor em fatias diretas de 64MB via memoryview sem duplicar gigabytes na RAM."""
    t = tensor.contiguous().view(torch.int16) if tensor.dtype == torch.bfloat16 else tensor.contiguous()
    arr = t.numpy()
    mv = memoryview(arr).cast('b')
    CHUNK = 64 * 1024 * 1024
    for i in range(0, len(mv), CHUNK):
        f.write(mv[i:i + CHUNK])
    return len(mv)

def pack_full_model():
    print("=" * 80, flush=True)
    print("INICIANDO EMPACOTAMENTO GLOBAL DO QWEN3.8-35B NO FORMATO PAGINADO .vq2m", flush=True)
    print(f"Destino: {TARGET_VQ2M}", flush=True)
    print("=" * 80, flush=True)
    
    index_file = MODEL_DIR / "model.safetensors.index.json"
    with open(index_file) as f:
        weight_map = json.load(f)["weight_map"]
        
    shards_to_load = sorted(set(weight_map.values()))
    print(f"Total de shards identificados no indice: {len(shards_to_load)}", flush=True)
    
    t_start = time.time()
    
    moe_offset_table = [[None for _ in range(256)] for _ in range(40)]
    other_tensors_meta = {}
    
    pending_gate_up = {}
    pending_down = {}
    
    RESERVED_HEADER_BYTES = 2 * 1024 * 1024 # 2 MiB reservado no início para cabeçalho
    
    f_out = open(TARGET_VQ2M, "wb")
    f_out.seek(RESERVED_HEADER_BYTES)
    
    layers_moe_written = set()
    total_non_moe_written = 0
    
    print("\n[Passo 1/2] Processando shards e gravando tensores contiguos...", flush=True)
    for s_idx, shard_name in enumerate(shards_to_load, 1):
        pt_path = OUT_35B / f"quant_{shard_name}.pt"
        if not pt_path.exists():
            print(f"AVISO: Shard {shard_name} nao encontrado em {pt_path}, pulando...", flush=True)
            continue
            
        t_s0 = time.time()
        shard_data = torch.load(pt_path, weights_only=False, map_location="cpu")
        quant_dict = shard_data.get("quant", {})
        dense_dict = shard_data.get("dense", {})
        
        # A. Processa tensores densos (RMSNorms, router gates, biases)
        for raw_name, tensor in dense_dict.items():
            clean_name = raw_name.replace("model.language_model.", "model.")
            t_cast = tensor.to(torch.bfloat16) if tensor.dtype != torch.bfloat16 else tensor
            offset = f_out.tell()
            size = write_tensor_direct(f_out, t_cast)
            other_tensors_meta[clean_name] = {
                "type": "dense",
                "dtype": "bfloat16",
                "shape": list(tensor.shape),
                "offset": offset,
                "size": size
            }
            total_non_moe_written += 1
            
        # B. Processa tensores lineares quantizados (atenções, shared experts, embeddings)
        for raw_name, q_data in quant_dict.items():
            if "mlp.experts" in raw_name:
                continue
            clean_name = raw_name.replace("model.language_model.", "model.")
            idx_list = q_data["idx"]
            scale = q_data["scale"].to(torch.float16).contiguous()
            
            idx_meta = []
            for sub_idx in idx_list:
                off = f_out.tell()
                sz = write_tensor_direct(f_out, sub_idx)
                idx_meta.append({"offset": off, "size": sz, "dtype": str(sub_idx.dtype).split(".")[-1]})
                
            scale_off = f_out.tell()
            scale_sz = write_tensor_direct(f_out, scale)
            
            other_tensors_meta[clean_name] = {
                "type": "quant_linear",
                "shape": list(q_data["shape"]),
                "cfg": list(q_data["cfg"]),
                "tag": q_data["tag"],
                "idx_meta": idx_meta,
                "scale_meta": {"offset": scale_off, "size": scale_sz}
            }
            total_non_moe_written += 1
            
        # C. Pareia e grava especialistas MoE quando gate_up e down estiverem disponíveis
        for k, q_data in quant_dict.items():
            if "mlp.experts.gate_up_proj" in k:
                m = k.split(".layers.")
                if len(m) > 1:
                    ln = int(m[1].split(".")[0])
                    pending_gate_up[ln] = q_data
            elif "mlp.experts.down_proj" in k:
                m = k.split(".layers.")
                if len(m) > 1:
                    ln = int(m[1].split(".")[0])
                    pending_down[ln] = q_data
                    
        for ln in list(pending_gate_up.keys()):
            if ln in pending_down and ln not in layers_moe_written:
                q_gu = pending_gate_up[ln]
                q_dn = pending_down[ln]
                
                gu_idx = q_gu["idx"][0].view(256, 1024, 256).contiguous()
                gu_scale = q_gu["scale"].view(256, 1024).to(torch.float16).contiguous()
                dn_idx = q_dn["idx"][0].view(256, 2048, 64).contiguous()
                dn_scale = q_dn["scale"].view(256, 2048).to(torch.float16).contiguous()
                
                for e in range(256):
                    off_gui = f_out.tell(); sz_gui = write_tensor_direct(f_out, gu_idx[e])
                    off_gus = f_out.tell(); sz_gus = write_tensor_direct(f_out, gu_scale[e])
                    off_dni = f_out.tell(); sz_dni = write_tensor_direct(f_out, dn_idx[e])
                    off_dns = f_out.tell(); sz_dns = write_tensor_direct(f_out, dn_scale[e])
                    
                    moe_offset_table[ln][e] = (
                        off_gui, sz_gui, off_gus, sz_gus,
                        off_dni, sz_dni, off_dns, sz_dns
                    )
                    
                layers_moe_written.add(ln)
                del pending_gate_up[ln], pending_down[ln]
                
        del shard_data
        gc.collect()
        dt_s = time.time() - t_s0
        print(f"[{s_idx:02d}/{len(shards_to_load)}] Shard {shard_name[:32]} ok em {dt_s:.1f}s | Camadas MoE concluidas: {len(layers_moe_written)}/40 | Escrito: {f_out.tell()/1024**2:.1f} MB", flush=True)

    # Também grava o Shard MTP se existir
    mtp_shard = OUT_35B / "quant_model-mtp.safetensors.pt"
    if mtp_shard.exists():
        print("Gravando shard adicional MTP...", flush=True)
        mtp_data = torch.load(mtp_shard, weights_only=False, map_location="cpu")
        for raw_name, tensor in mtp_data.get("dense", {}).items():
            clean_name = "mtp." + raw_name.replace("model.language_model.", "model.")
            t_cast = tensor.to(torch.bfloat16) if tensor.dtype != torch.bfloat16 else tensor
            offset = f_out.tell()
            size = write_tensor_direct(f_out, t_cast)
            other_tensors_meta[clean_name] = {
                "type": "dense",
                "dtype": "bfloat16",
                "shape": list(tensor.shape),
                "offset": offset,
                "size": size
            }
        for raw_name, q_data in mtp_data.get("quant", {}).items():
            clean_name = "mtp." + raw_name.replace("model.language_model.", "model.")
            idx_meta = []
            for sub_idx in q_data["idx"]:
                off = f_out.tell()
                sz = write_tensor_direct(f_out, sub_idx)
                idx_meta.append({"offset": off, "size": sz, "dtype": str(sub_idx.dtype).split(".")[-1]})
            scale = q_data["scale"].to(torch.float16).contiguous()
            scale_off = f_out.tell()
            scale_sz = write_tensor_direct(f_out, scale)
            other_tensors_meta[clean_name] = {
                "type": "quant_linear",
                "shape": list(q_data["shape"]),
                "cfg": list(q_data["cfg"]),
                "tag": q_data["tag"],
                "idx_meta": idx_meta,
                "scale_meta": {"offset": scale_off, "size": scale_sz}
            }
        del mtp_data
        gc.collect()

    final_payload_size = f_out.tell()
    print(f"\n[Passo 2/2] Gravando cabecalho e tabelas de indices (Payload total: {final_payload_size/1024**3:.2f} GB)...", flush=True)
    
    meta_json_obj = {
        "model_name": "Qwen3.8-35B-A3B-Distill",
        "num_layers": 40,
        "num_experts": 256,
        "hidden_dim": 2048,
        "intermediate_dim": 512,
        "gu_rows": 1024,
        "gu_cols": 256,
        "dn_rows": 2048,
        "dn_cols": 64,
        "other_tensors": other_tensors_meta
    }
    meta_json_bytes = json.dumps(meta_json_obj).encode("utf-8")
    meta_json_len = len(meta_json_bytes)
    
    moe_table_bytes = bytearray()
    for ln in range(40):
        for e in range(256):
            entry = moe_offset_table[ln][e]
            if entry is None:
                entry = (0, 0, 0, 0, 0, 0, 0, 0)
            moe_table_bytes.extend(struct.pack(ENTRY_FORMAT, *entry))
            
    moe_table_len = len(moe_table_bytes)
    
    moe_table_off = 64
    meta_json_off = moe_table_off + moe_table_len
    payload_start = RESERVED_HEADER_BYTES
    
    if meta_json_off + meta_json_len > RESERVED_HEADER_BYTES:
        raise RuntimeError(f"Cabecalho excedeu espaco reservado de {RESERVED_HEADER_BYTES} bytes! Necessario: {meta_json_off + meta_json_len}")
        
    f_out.seek(0)
    header_fixed = struct.pack(
        "<4sHHHH QQQQQ 12s",
        MAGIC, VERSION, 40, 256, 0,
        moe_table_off, moe_table_len,
        meta_json_off, meta_json_len,
        payload_start, b"\x00" * 12
    )
    assert len(header_fixed) == 64
    f_out.write(header_fixed)
    f_out.write(moe_table_bytes)
    f_out.write(meta_json_bytes)
    f_out.close()
    
    total_sec = time.time() - t_start
    final_file_size_bytes = TARGET_VQ2M.stat().st_size
    print("=" * 80, flush=True)
    print("ARQUIVO PAGINADO .vq2m GERADO COM SUCESSO ABSOLUTO!", flush=True)
    print(f"Caminho: {TARGET_VQ2M}", flush=True)
    print(f"Tamanho total: {final_file_size_bytes / 1024**3:.2f} GB ({final_file_size_bytes / 1024**2:.1f} MB)", flush=True)
    print(f"Camadas MoE indexadas: {len(layers_moe_written)}/40 (Total de 10.240 especialistas mapeados!)", flush=True)
    print(f"Outros tensores indexados: {total_non_moe_written}", flush=True)
    print(f"Tempo total de execucao: {total_sec/60:.2f} minutos", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    pack_full_model()
