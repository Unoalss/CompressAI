"""
vq2/quantize_35b.py - Pipeline ultra-otimizado de quantizacao vetorial RDO para Qwen3.8-35B-A3B (MoE 256 experts).
Processa shard por shard de safetensors com controle estrito de VRAM (< 2.0 GB pico),
garantindo execucao 100% acelerada na GPU sem paginacao WDDM.
Meta de tamanho: ~9.5 a 10.2 GB (85% menor que o original de 67 GB).
"""
import os, sys, time, json
from pathlib import Path
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vq2 import vq

MODEL_DIR = ROOT / "modelo_35b"
OUT_35B = ROOT / "vq2_35b_out"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CB_CACHE = {}

def init_cb_cache(cbs):
    global CB_CACHE
    CB_CACHE = {}
    for k, cb in cbs.cb.items():
        cb16 = cb.half()
        c2 = (cb.float() ** 2).sum(1).half()
        CB_CACHE[k] = (cb16.T.contiguous(), c2.contiguous(), cb.float())

@torch.no_grad()
def fast_quantize_matrix(W_2d, tag, cfg, dev=DEVICE, row_chunk=32768, v_chunk=4096):
    """
    Quantiza matriz [out, n] em blocos de vetores 8-D rotacionados com baixo consumo de VRAM.
    VRAM pico: < 1.8 GB mesmo para matrizes MoE de 536M parametros.
    """
    out, n = W_2d.shape
    rot = vq.get_rot(tag, n, dev)
    scales = torch.empty(out, dtype=torch.half)
    
    if cfg == (16,):
        final_idx = [torch.empty((out, n // 8), dtype=torch.int16)]
        cb_T, c2, _ = CB_CACHE["16"]
        for r in range(0, out, row_chunk):
            r_end = min(r + row_chunk, out)
            w_blk = W_2d[r:r_end].float().to(dev)
            wr = rot.fwd(w_blk)
            s = wr.pow(2).mean(1).sqrt().clamp_min(1e-6).half()
            scales[r:r_end] = s.cpu()
            wr.div_(s.float()[:, None])
            wr_8d = wr.view(-1, 8).half()
            
            nv = wr_8d.shape[0]
            idx_blk = torch.empty(nv, dtype=torch.int16)
            for vi in range(0, nv, v_chunk):
                ve = min(vi + v_chunk, nv)
                dist = torch.addmm(c2, wr_8d[vi:ve], cb_T, alpha=-2.0)
                idx_blk[vi:ve] = dist.argmin(1).to(torch.int16).cpu()
            final_idx[0][r:r_end] = idx_blk.view(r_end - r, -1)
            del w_blk, wr, wr_8d
    elif cfg == (16, 8):
        final_idx = [
            torch.empty((out, n // 8), dtype=torch.int16),
            torch.empty((out, n // 8), dtype=torch.uint8)
        ]
        cb0_T, c2_0, cb0_raw = CB_CACHE["16"]
        cb1_T, c2_1, _ = CB_CACHE["16_8"]
        for r in range(0, out, row_chunk):
            r_end = min(r + row_chunk, out)
            w_blk = W_2d[r:r_end].float().to(dev)
            wr = rot.fwd(w_blk)
            s = wr.pow(2).mean(1).sqrt().clamp_min(1e-6).half()
            scales[r:r_end] = s.cpu()
            wr.div_(s.float()[:, None])
            wr_8d = wr.view(-1, 8).half()
            
            nv = wr_8d.shape[0]
            idx0_blk = torch.empty(nv, dtype=torch.int16)
            idx1_blk = torch.empty(nv, dtype=torch.uint8)
            for vi in range(0, nv, v_chunk):
                ve = min(vi + v_chunk, nv)
                sub_v = wr_8d[vi:ve]
                dist0 = torch.addmm(c2_0, sub_v, cb0_T, alpha=-2.0)
                a0 = dist0.argmin(1)
                idx0_blk[vi:ve] = a0.to(torch.int16).cpu()
                
                # Residuo do estagio 1
                res = (sub_v - cb0_raw[a0].half()).half()
                dist1 = torch.addmm(c2_1, res, cb1_T, alpha=-2.0)
                a1 = dist1.argmin(1)
                idx1_blk[vi:ve] = a1.to(torch.uint8).cpu()
                
            final_idx[0][r:r_end] = idx0_blk.view(r_end - r, -1)
            final_idx[1][r:r_end] = idx1_blk.view(r_end - r, -1)
            del w_blk, wr, wr_8d
    else:
        raise ValueError(f"cfg nao suportado: {cfg}")
        
    return {
        "shape": (out, n),
        "cfg": list(cfg),
        "tag": tag,
        "idx": final_idx,
        "scale": scales
    }

def process_shard(shard_name, shard_idx, total_shards, out_dir):
    shard_path = MODEL_DIR / shard_name
    out_shard_path = out_dir / f"quant_{shard_name}.pt"
    if out_shard_path.exists():
        size_mb = out_shard_path.stat().st_size / 1024**2
        print(f"[{shard_idx}/{total_shards}] Shard {shard_name} ja processado ({size_mb:.1f} MB). Pulando...", flush=True)
        return
        
    print("\n" + "=" * 80, flush=True)
    print(f"[{shard_idx}/{total_shards}] PROCESSANDO SHARD: {shard_name}", flush=True)
    print("=" * 80, flush=True)
    
    t_shard_0 = time.time()
    quant_tensors = {}
    dense_tensors = {}
    
    f = safe_open(str(shard_path), framework="pt")
    keys = list(f.keys())
    
    for k_idx, k in enumerate(keys):
        t0 = time.time()
        tensor = f.get_tensor(k)
        shape = tensor.shape
        numel = tensor.numel()
        
        # Decide se e quantizado ou denso
        is_expert = ("experts.down_proj" in k or "experts.gate_up_proj" in k)
        is_embed = ("embed_tokens" in k or "lm_head" in k)
        is_linear_attn = ("linear_attn.in_proj" in k or "linear_attn.out_proj" in k)
        is_self_attn = ("self_attn.q_proj" in k or "self_attn.k_proj" in k or "self_attn.v_proj" in k or "self_attn.o_proj" in k)
        is_shared_exp = ("shared_expert." in k and ".weight" in k and tensor.ndim == 2)
        
        # 1D, normas, biases e roteadores ficam densos em BF16
        if tensor.ndim < 2 or "norm" in k or "gate.weight" in k or "conv1d" in k:
            dense_tensors[k] = tensor.to(torch.bfloat16)
            continue
            
        # Alocacao de bits
        if is_expert:
            cfg = (16,) # 2.0 bpw para os 256 experts esparsos
            tag = "moe_expert_2048" if shape[-1] == 2048 else "moe_expert_512"
            w_2d = tensor.view(-1, shape[-1])
        elif is_embed:
            cfg = (16,) # 2.0 bpw para embedding/lm_head
            tag = "embed"
            w_2d = tensor.view(-1, shape[-1])
        elif is_linear_attn or is_self_attn or is_shared_exp:
            cfg = (16, 8) # 3.0 bpw para atencoes centrais e expert compartilhado
            tag = "attn"
            w_2d = tensor.view(-1, shape[-1])
        else:
            dense_tensors[k] = tensor.to(torch.bfloat16)
            continue
            
        print(f"  [{k_idx+1}/{len(keys)}] Quantizando {k} | Shape {list(shape)} -> 2D {list(w_2d.shape)} | cfg={cfg}...", end=" ", flush=True)
        qdata = fast_quantize_matrix(w_2d, tag, cfg)
        quant_tensors[k] = qdata
        dt = time.time() - t0
        mb_orig = numel * 2 / 1024**2
        mb_quant = (sum(x.numel() * x.element_size() for x in qdata["idx"]) + qdata["scale"].numel() * 2) / 1024**2
        print(f"OK em {dt:.1f}s ({mb_orig:.1f} MB -> {mb_quant:.1f} MB, -{100*(1-mb_quant/mb_orig):.1f}%)", flush=True)
        
    # Salva o shard quantizado
    payload = {
        "shard_name": shard_name,
        "quant": quant_tensors,
        "dense": dense_tensors
    }
    torch.save(payload, out_shard_path)
    dt_shard = time.time() - t_shard_0
    shard_size_mb = out_shard_path.stat().st_size / 1024**2
    print(f"\nShard {shard_name} concluido em {dt_shard/60:.1f} min! Salvo em {out_shard_path.name} ({shard_size_mb:.1f} MB)", flush=True)

def run_35b_pipeline():
    OUT_35B.mkdir(parents=True, exist_ok=True)
    index_file = MODEL_DIR / "model.safetensors.index.json"
    with open(index_file) as f:
        weight_map = json.load(f)["weight_map"]
    shards = sorted(set(weight_map.values()))
    
    print("=" * 80, flush=True)
    print("PIPELINE DE QUANTIZACAO VQ2 - QWEN3.8-35B-A3B", flush=True)
    print(f"Total de Shards: {len(shards)} | Saida: {OUT_35B}", flush=True)
    print(f"Dispositivo GPU: {torch.cuda.get_device_name(0)} (VRAM Max 12 GB)", flush=True)
    print("=" * 80, flush=True)
    
    # Carrega dicionarios compartilhados e inicializa cache rapido
    cbs_data = torch.load(ROOT / "vq2_out" / "codebooks.pt", weights_only=False)
    cbs = vq.Codebooks({k: v.float().to(DEVICE) for k, v in cbs_data.items()})
    init_cb_cache(cbs)
    print("Codebooks carregados e cacheados em VRAM com sucesso!\n", flush=True)
    
    t_global_0 = time.time()
    for idx, shard in enumerate(shards, 1):
        process_shard(shard, idx, len(shards), OUT_35B)
        torch.cuda.empty_cache()
        
    dt_total = time.time() - t_global_0
    print("\n" + "=" * 80, flush=True)
    print(f"TODOS OS {len(shards)} SHARDS DO MODELO 35B FORAM QUANTIZADOS EM {dt_total/60:.1f} MINUTOS!", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    run_35b_pipeline()
