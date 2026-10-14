"""
vq2/paged_runtime.py - Motor de Inferência Paginado Memory-Mapped (.vq2m) com Cache LRU de Especialistas.
Permite executar o modelo Qwen3.8-35B-A3B (40 camadas, 256 experts) mantendo os 9.38 GB mapeados em disco
e apenas os especialistas ativos e camadas fixas residentes na RAM/VRAM (< 2.0 GB).
"""
import os, sys, time, json, struct, mmap, threading
from pathlib import Path
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vq2 import vq
from vq2.kernels import HAVE_TRITON, vq_moe_parallel_gate_up, vq_moe_parallel_down

def vq_moe_parallel_gate_up_cpu(xr, slots, pool_idx, cb16, pool_scale, y_out, OUT=1024, NG=256, rs=1.0):
    """Execução MoE Gate-Up de referência em CPU via PyTorch einsum/gather vetorizado."""
    K = slots.shape[0]
    xg = xr.view(NG, 8).to(dtype=cb16.dtype)
    for k in range(K):
        s = slots[k].item()
        idx = (pool_idx[s].to(torch.int32) & 0xFFFF).long()
        sc = pool_scale[s].float()
        rec = cb16[idx]  # [1024, 256, 8]
        acc = torch.einsum('org,rg->o', rec, xg)
        y_out[k] = (acc.float() * sc * rs).to(y_out.dtype)

def vq_moe_parallel_down_cpu(act_r, slots, pool_idx, cb16, pool_scale, y_out, OUT=2048, NG=64, rs=1.0):
    """Execução MoE Down de referência em CPU via PyTorch einsum/gather vetorizado."""
    K = slots.shape[0]
    for k in range(K):
        s = slots[k].item()
        idx = (pool_idx[s].to(torch.int32) & 0xFFFF).long()
        sc = pool_scale[s].float()
        act_g = act_r[k].view(NG, 8).to(dtype=cb16.dtype)
        rec = cb16[idx]  # [2048, 64, 8]
        acc = torch.einsum('org,rg->o', rec, act_g)
        y_out[k] = (acc.float() * sc * rs).to(y_out.dtype)

MAGIC = b"VQ2M"
VERSION = 1
ENTRY_FORMAT = "<QIQIQIQI"
ENTRY_SIZE = struct.calcsize(ENTRY_FORMAT)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class PagedModelStore:
    """
    Leitor do container .vq2m de 9.38 GB.
    Suporta dois modos:
      - use_mmap=False (Modo Offset / Orçamento Rígido de RAM): Leitura sob demanda por seek/readinto
        usando um buffer único reutilizável de 774 KB. Evita acumulação de páginas no Working Set da RAM.
      - use_mmap=True (Modo Memory-Mapped clássico do SO).
    """
    def __init__(self, file_path, use_mmap=False, device=None):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Arquivo .vq2m nao encontrado em {self.file_path}")
            
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_mmap = use_mmap
        self.file = open(self.file_path, "rb")
        self.lock = threading.Lock()
        
        if self.use_mmap:
            import psutil
            avail_ram_gb = psutil.virtual_memory().available / 1024**3
            if avail_ram_gb >= 11.0:
                print(f"      [RAM Turbo] Detectados {avail_ram_gb:.1f} GB de RAM livre: pré-carregando container de 9.4 GB na memória RAM do sistema...")
                self.file.seek(0)
                self.mm = self.file.read()
                print("      [RAM Turbo] 100% dos 10.240 especialistas residentes na RAM física (zero page faults e I/O instantâneo)!")
            else:
                self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
            header_fixed = self.mm[:64]
        else:
            self.mm = None
            if self.device != "cpu" and torch.cuda.is_available():
                self.expert_scratch = torch.empty(792576, dtype=torch.uint8, pin_memory=True)
                self.scratch_mv = memoryview(self.expert_scratch.numpy())
            else:
                self.expert_scratch = bytearray(792576)
                self.scratch_mv = memoryview(self.expert_scratch)
            self.file.seek(0)
            header_fixed = self.file.read(64)
            
        (
            magic, version, self.num_layers, self.num_experts, _,
            moe_table_off, moe_table_len,
            meta_json_off, meta_json_len,
            self.payload_start, _
        ) = struct.unpack("<4sHHHH QQQQQ 12s", header_fixed)
        
        if magic != MAGIC or version != VERSION:
            raise ValueError("Arquivo .vq2m invalido ou versao incompativel!")
            
        if self.use_mmap:
            all_moe_bytes = self.mm[moe_table_off : moe_table_off + moe_table_len]
            meta_json_bytes = self.mm[meta_json_off : meta_json_off + meta_json_len]
        else:
            self.file.seek(moe_table_off)
            all_moe_bytes = self.file.read(moe_table_len)
            self.file.seek(meta_json_off)
            meta_json_bytes = self.file.read(meta_json_len)
            
        unpacked_entries = list(struct.iter_unpack(ENTRY_FORMAT, all_moe_bytes))
        self.moe_offsets = []
        idx = 0
        for ln in range(self.num_layers):
            self.moe_offsets.append(unpacked_entries[idx : idx + self.num_experts])
            idx += self.num_experts
            
        del all_moe_bytes
        self.meta = json.loads(meta_json_bytes.decode("utf-8"))
        del meta_json_bytes
        self.other_tensors = self.meta.get("other_tensors", {})
        
        self.gu_rows = self.meta.get("gu_rows", 1024)
        self.gu_cols = self.meta.get("gu_cols", 256)
        self.dn_rows = self.meta.get("dn_rows", 2048)
        self.dn_cols = self.meta.get("dn_cols", 64)

    def load_expert_into_slot(self, layer_idx: int, expert_id: int, slot: int,
                              pool_gu_idx, pool_gu_scale, pool_dn_idx, pool_dn_scale, device=None):
        """Transfere os pesos de 1 especialista diretamente para os buffers de VRAM do slot indicado."""
        (
            off_gui, sz_gui, off_gus, sz_gus,
            off_dni, sz_dni, off_dns, sz_dns
        ) = self.moe_offsets[layer_idx][expert_id]
        
        if self.use_mmap:
            gui = torch.frombuffer(self.mm, dtype=torch.int16, count=sz_gui // 2, offset=off_gui).view(self.gu_rows, self.gu_cols)
            gus = torch.frombuffer(self.mm, dtype=torch.float16, count=sz_gus // 2, offset=off_gus)
            dni = torch.frombuffer(self.mm, dtype=torch.int16, count=sz_dni // 2, offset=off_dni).view(self.dn_rows, self.dn_cols)
            dns = torch.frombuffer(self.mm, dtype=torch.float16, count=sz_dns // 2, offset=off_dns)
            is_cuda = pool_gu_idx.is_cuda
            pool_gu_idx[slot].copy_(gui, non_blocking=is_cuda)
            pool_gu_scale[slot].copy_(gus.to(dtype=torch.bfloat16), non_blocking=is_cuda)
            pool_dn_idx[slot].copy_(dni, non_blocking=is_cuda)
            pool_dn_scale[slot].copy_(dns.to(dtype=torch.bfloat16), non_blocking=is_cuda)
        else:
            with self.lock:
                self.file.seek(off_gui)
                self.file.readinto(self.scratch_mv)
                
            if isinstance(self.expert_scratch, torch.Tensor):
                gui = self.expert_scratch[0 : sz_gui].view(dtype=torch.int16).view(self.gu_rows, self.gu_cols)
                gus = self.expert_scratch[sz_gui : sz_gui + sz_gus].view(dtype=torch.float16)
                dni = self.expert_scratch[sz_gui + sz_gus : sz_gui + sz_gus + sz_dni].view(dtype=torch.int16).view(self.dn_rows, self.dn_cols)
                dns = self.expert_scratch[sz_gui + sz_gus + sz_dni : sz_gui + sz_gus + sz_dni + sz_dns].view(dtype=torch.float16)
                
                is_cuda = pool_gu_idx.is_cuda
                pool_gu_idx[slot].copy_(gui, non_blocking=is_cuda)
                pool_gu_scale[slot].copy_(gus.to(dtype=torch.bfloat16), non_blocking=is_cuda)
                pool_dn_idx[slot].copy_(dni, non_blocking=is_cuda)
                pool_dn_scale[slot].copy_(dns.to(dtype=torch.bfloat16), non_blocking=is_cuda)
            else:
                gui_raw = self.scratch_mv[0 : sz_gui]
                gus_raw = self.scratch_mv[sz_gui : sz_gui + sz_gus]
                dni_raw = self.scratch_mv[sz_gui + sz_gus : sz_gui + sz_gus + sz_dni]
                dns_raw = self.scratch_mv[sz_gui + sz_gus + sz_dni : sz_gui + sz_gus + sz_dni + sz_dns]
                
                gui = torch.frombuffer(gui_raw, dtype=torch.int16, count=sz_gui // 2).view(self.gu_rows, self.gu_cols)
                gus = torch.frombuffer(gus_raw, dtype=torch.float16, count=sz_gus // 2)
                dni = torch.frombuffer(dni_raw, dtype=torch.int16, count=sz_dni // 2).view(self.dn_rows, self.dn_cols)
                dns = torch.frombuffer(dns_raw, dtype=torch.float16, count=sz_dns // 2)
                
                pool_gu_idx[slot].copy_(gui)
                pool_gu_scale[slot].copy_(gus.to(dtype=torch.bfloat16))
                pool_dn_idx[slot].copy_(dni)
                pool_dn_scale[slot].copy_(dns.to(dtype=torch.bfloat16))

    def get_expert_weights(self, layer_idx: int, expert_id: int, device="cpu"):
        """Retorna os 4 tensores de 1 especialista específico."""
        (
            off_gui, sz_gui, off_gus, sz_gus,
            off_dni, sz_dni, off_dns, sz_dns
        ) = self.moe_offsets[layer_idx][expert_id]
        
        if self.use_mmap:
            gui = torch.frombuffer(self.mm, dtype=torch.int16, count=sz_gui // 2, offset=off_gui).view(self.gu_rows, self.gu_cols)
            gus = torch.frombuffer(self.mm, dtype=torch.float16, count=sz_gus // 2, offset=off_gus)
            dni = torch.frombuffer(self.mm, dtype=torch.int16, count=sz_dni // 2, offset=off_dni).view(self.dn_rows, self.dn_cols)
            dns = torch.frombuffer(self.mm, dtype=torch.float16, count=sz_dns // 2, offset=off_dns)
            if device != "cpu":
                return gui.to(device), gus.to(device=device, dtype=torch.bfloat16), dni.to(device), dns.to(device=device, dtype=torch.bfloat16)
            return gui, gus, dni, dns
        else:
            with self.lock:
                self.file.seek(off_gui)
                self.file.readinto(self.scratch_mv)
                
            gui_raw = self.scratch_mv[0 : sz_gui]
            gus_raw = self.scratch_mv[sz_gui : sz_gui + sz_gus]
            dni_raw = self.scratch_mv[sz_gui + sz_gus : sz_gui + sz_gus + sz_dni]
            dns_raw = self.scratch_mv[sz_gui + sz_gus + sz_dni : sz_gui + sz_gus + sz_dni + sz_dns]
            
            gui = torch.frombuffer(gui_raw, dtype=torch.int16, count=sz_gui // 2).view(self.gu_rows, self.gu_cols)
            gus = torch.frombuffer(gus_raw, dtype=torch.float16, count=sz_gus // 2)
            dni = torch.frombuffer(dni_raw, dtype=torch.int16, count=sz_dni // 2).view(self.dn_rows, self.dn_cols)
            dns = torch.frombuffer(dns_raw, dtype=torch.float16, count=sz_dns // 2)
            
            if device != "cpu":
                return gui.to(device), gus.to(device=device, dtype=torch.bfloat16), dni.to(device), dns.to(device=device, dtype=torch.bfloat16)
            return gui.clone(), gus.clone(), dni.clone(), dns.clone()

    def get_dense_tensor(self, name: str, device="cpu"):
        """Retorna um tensor denso (RMSNorm, router, bias)."""
        info = self.other_tensors.get(name)
        if not info or info.get("type") != "dense":
            return None
        offset = info["offset"]
        size = info["size"]
        shape = info["shape"]
        if self.use_mmap:
            t = torch.frombuffer(self.mm, dtype=torch.bfloat16, count=size // 2, offset=offset).view(*shape)
            return t.to(device)
        else:
            with self.lock:
                self.file.seek(offset)
                raw = self.file.read(size)
            t = torch.frombuffer(raw, dtype=torch.bfloat16, count=size // 2).view(*shape).to(device)
            del raw
            return t

    def get_quant_linear(self, name: str, cbs_data=None, device="cpu"):
        """Retorna tensores quantizados lineares reconstruídos com limpeza imediata de rascunhos."""
        info = self.other_tensors.get(name)
        if not info or info.get("type") != "quant_linear":
            return None
        shape = info["shape"]
        cfg = info["cfg"]
        tag = info["tag"]
        
        idx_list = []
        for im in info["idx_meta"]:
            off = im["offset"]
            sz = im["size"]
            dt = im.get("dtype", "int16")
            torch_dt = torch.int16 if dt == "int16" else torch.uint8
            if self.use_mmap:
                t = torch.frombuffer(self.mm, dtype=torch_dt, count=sz // (2 if torch_dt == torch.int16 else 1), offset=off)
            else:
                with self.lock:
                    self.file.seek(off)
                    raw = self.file.read(sz)
                t = torch.frombuffer(raw, dtype=torch_dt, count=sz // (2 if torch_dt == torch.int16 else 1))
            idx_list.append(t.to(device))
            
        scale_off = info["scale_meta"]["offset"]
        scale_sz = info["scale_meta"]["size"]
        if self.use_mmap:
            scale = torch.frombuffer(self.mm, dtype=torch.float16, count=scale_sz // 2, offset=scale_off).to(device).float()
        else:
            with self.lock:
                self.file.seek(scale_off)
                s_raw = self.file.read(scale_sz)
            scale = torch.frombuffer(s_raw, dtype=torch.float16, count=scale_sz // 2).to(device).float()
            del s_raw
            
        if cbs_data is not None:
            rec = None
            for i in range(len(cfg)):
                cb_key = "_".join(str(b) for b in cfg[: i + 1])
                cb = cbs_data[cb_key].to(device, dtype=torch.bfloat16)
                part = cb[idx_list[i].long()]
                if rec is None:
                    rec = part
                else:
                    rec = rec + part
                del part
            Q = rec.reshape(shape[0], shape[1]) * scale[:, None].to(dtype=torch.bfloat16)
            del rec, idx_list, scale
            W = vq.get_rot(tag, shape[1], device).inv(Q)
            del Q
            return W.to(torch.bfloat16)
        return {"shape": shape, "cfg": cfg, "tag": tag, "idx": idx_list, "scale": scale}

    def close(self):
        if self.mm is not None:
            self.mm.close()
        self.file.close()


class LRUExpertCache:
    """
    Cache LRU para manter os especialistas mais frequentes na VRAM/RAM.
    Reduz a latência de I/O em >70% durante conversas contínuas.
    """
    def __init__(self, capacity=128, device=DEVICE):
        self.capacity = capacity
        self.device = device
        self.cache = OrderedDict() # (layer_idx, expert_id) -> (gui, gus, dni, dns)
        self.hits = 0
        self.misses = 0

    def get(self, store: PagedModelStore, layer_idx: int, expert_id: int):
        key = (layer_idx, expert_id)
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
            
        self.misses += 1
        gui, gus, dni, dns = store.get_expert_weights(layer_idx, expert_id, device=self.device)
        self.cache[key] = (gui, gus, dni, dns)
        
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
            
        return gui, gus, dni, dns


class PagedMoEExpert(nn.Module):
    """
    Submódulo MoE com pool estático de alta velocidade e ZERO-COPY.
    Mantém uma reserva estática de C especialistas na VRAM (ex: 96 experts por camada).
    Para os 8 especialistas ativos, indexa os slots diretamente no kernel Triton
    eliminando cópias de tensores e reduzindo a latência MoE para ~7 ms!
    """
    def __init__(self, layer_idx: int, store: PagedModelStore, cb16, capacity=96, device=DEVICE):
        super().__init__()
        self.layer_idx = layer_idx
        self.store = store
        self.cb16 = cb16
        self.capacity = capacity
        self.device = device
        
        self.hidden_dim = 2048
        self.intermediate_dim = 512
        self.rot_2048 = vq.get_rot("moe_expert_2048", self.hidden_dim, device)
        self.rot_512 = vq.get_rot("moe_expert_512", self.intermediate_dim, device)
        self.RS_2048 = 1.0
        self.RS_512 = 1.0
        
        # Pool estático em VRAM: C slots (ex: 96 experts = ~74 MB por camada)
        self.register_buffer("pool_gu_idx", torch.empty((capacity, 1024, 256), dtype=torch.int16, device=device), persistent=False)
        self.register_buffer("pool_gu_scale", torch.empty((capacity, 1024), dtype=torch.bfloat16, device=device), persistent=False)
        self.register_buffer("pool_dn_idx", torch.empty((capacity, 2048, 64), dtype=torch.int16, device=device), persistent=False)
        self.register_buffer("pool_dn_scale", torch.empty((capacity, 2048), dtype=torch.bfloat16, device=device), persistent=False)
        
        self.expert_to_slot = {}
        self.free_slots = list(range(capacity))
        self.access_order = OrderedDict()
        
        self.hits = 0
        self.misses = 0
        self.activated_experts_set = set()
        
        self.register_buffer("active_slots", torch.zeros(8, dtype=torch.int32, device=device), persistent=False)
        self.register_buffer("y_gu", torch.empty((8, 1024), device=device, dtype=torch.bfloat16), persistent=False)
        self.register_buffer("y_down", torch.empty((8, 2048), device=device, dtype=torch.bfloat16), persistent=False)
        self.register_buffer("act_r", torch.empty((8, 512), device=device, dtype=torch.bfloat16), persistent=False)

        # Suporte nativo a Precisão Mista Adaptativa (Especialistas 3 bits em VRAM)
        self.promoted_3b = {}
        expert_modes = self.store.meta.get("expert_modes", {})
        promoted_in_layer = [e for e in range(self.store.num_experts) if expert_modes.get(f"{self.layer_idx}:{e}") == 3]
        if promoted_in_layer:
            cbs_data = torch.load(ROOT / "vq2_out" / "codebooks.pt", weights_only=False)
            cb8 = cbs_data["16_8"].to(self.device, torch.bfloat16)
            for e_id in promoted_in_layer:
                (off_gui, sz_gui, off_gus, sz_gus, off_dni, sz_dni, off_dns, sz_dns) = self.store.moe_offsets[self.layer_idx][e_id]
                with self.store.lock:
                    self.store.file.seek(off_gui)
                    raw_gui = self.store.file.read(sz_gui)
                    self.store.file.seek(off_gus)
                    raw_gus = self.store.file.read(sz_gus)
                    self.store.file.seek(off_dni)
                    raw_dni = self.store.file.read(sz_dni)
                    self.store.file.seek(off_dns)
                    raw_dns = self.store.file.read(sz_dns)

                idx0_gu = torch.frombuffer(raw_gui[:524288], dtype=torch.int16).view(1024, 256).to(self.device)
                idx1_gu = torch.frombuffer(raw_gui[524288:], dtype=torch.uint8).view(1024, 256).to(self.device)
                sc_gu = torch.frombuffer(raw_gus, dtype=torch.float16).to(self.device, torch.bfloat16)
                idx0_gu_m = (idx0_gu.to(torch.int32) & 0xFFFF).long()
                W_gu = self.rot_2048.inv(((self.cb16[idx0_gu_m] + cb8[idx1_gu.long()]).view(1024, 2048) * sc_gu[:, None]).to(torch.bfloat16))

                idx0_dn = torch.frombuffer(raw_dni[:262144], dtype=torch.int16).view(2048, 64).to(self.device)
                idx1_dn = torch.frombuffer(raw_dni[262144:], dtype=torch.uint8).view(2048, 64).to(self.device)
                sc_dn = torch.frombuffer(raw_dns, dtype=torch.float16).to(self.device, torch.bfloat16)
                idx0_dn_m = (idx0_dn.to(torch.int32) & 0xFFFF).long()
                W_dn = self.rot_512.inv(((self.cb16[idx0_dn_m] + cb8[idx1_dn.long()]).view(2048, 512) * sc_dn[:, None]).to(torch.bfloat16))

                self.promoted_3b[e_id] = {"gu": W_gu, "dn": W_dn}
            del cbs_data, cb8

    def _run_gate_up(self, xr, slots_t, y_gu, OUT=1024, NG=256):
        if self.device == "cpu":
            vq_moe_parallel_gate_up_cpu(
                xr, slots_t, self.pool_gu_idx, self.cb16, self.pool_gu_scale, y_gu,
                OUT=OUT, NG=NG, rs=self.RS_2048
            )
        else:
            vq_moe_parallel_gate_up(
                xr, slots_t, self.pool_gu_idx, self.cb16, self.pool_gu_scale, y_gu,
                OUT=OUT, NG=NG, rs=self.RS_2048
            )

    def _run_down(self, act_r, slots_t, y_down, OUT=2048, NG=64):
        if self.device == "cpu":
            vq_moe_parallel_down_cpu(
                act_r, slots_t, self.pool_dn_idx, self.cb16, self.pool_dn_scale, y_down,
                OUT=OUT, NG=NG, rs=self.RS_512
            )
        else:
            vq_moe_parallel_down(
                act_r, slots_t, self.pool_dn_idx, self.cb16, self.pool_dn_scale, y_down,
                OUT=OUT, NG=NG, rs=self.RS_512
            )

    def _get_slot(self, e_id: int) -> int:
        self.activated_experts_set.add(e_id)
        if e_id in self.expert_to_slot:
            self.hits += 1
            self.access_order.move_to_end(e_id)
            return self.expert_to_slot[e_id]
            
        self.misses += 1
        if self.free_slots:
            slot = self.free_slots.pop()
        else:
            evict_id, _ = self.access_order.popitem(last=False)
            slot = self.expert_to_slot.pop(evict_id)
            
        self.store.load_expert_into_slot(
            self.layer_idx, e_id, slot,
            self.pool_gu_idx, self.pool_gu_scale, self.pool_dn_idx, self.pool_dn_scale,
            device=self.device
        )
        
        self.expert_to_slot[e_id] = slot
        self.access_order[e_id] = None
        return slot

    def forward(self, hidden_states, topk_idx, topk_weights):
        T = hidden_states.shape[0]
        K = topk_idx.shape[-1]
        if T == 1:
            topk_ids_list = topk_idx.view(-1).tolist()
            topk_w = topk_weights.view(-1).bfloat16()

            if self.promoted_3b and any(e in self.promoted_3b for e in topk_ids_list):
                promoted_out = torch.zeros(1, 2048, device=self.device, dtype=torch.bfloat16)
                regular_slots = []
                reg_mask = []
                xt = hidden_states.bfloat16()
                for k_idx, e_id in enumerate(topk_ids_list):
                    w_k = topk_w[k_idx]
                    if e_id in self.promoted_3b:
                        W_gu = self.promoted_3b[e_id]["gu"]
                        W_dn = self.promoted_3b[e_id]["dn"]
                        y_gu = F.linear(xt, W_gu)
                        g, u = y_gu[:, :512], y_gu[:, 512:]
                        act = (F.silu(g.float()) * u.float()).bfloat16()
                        promoted_out += F.linear(act, W_dn) * w_k
                        reg_mask.append(False)
                    else:
                        regular_slots.append(self._get_slot(e_id))
                        reg_mask.append(True)

                if regular_slots:
                    K_reg = len(regular_slots)
                    for i, s in enumerate(regular_slots):
                        self.active_slots[i] = s
                    slots_t = self.active_slots[:K_reg]
                    reg_w = topk_w[torch.tensor(reg_mask, device=self.device, dtype=torch.bool)]

                    xr = self.rot_2048.fwd(xt.view(-1))
                    y_gu = self.y_gu[:K_reg]
                    self._run_gate_up(
                        xr, slots_t, y_gu,
                        OUT=1024, NG=256
                    )
                    gate = y_gu[:, :512]
                    up = y_gu[:, 512:]
                    act = (F.silu(gate.float()) * up.float()).bfloat16()
                    act_r = self.rot_512.fwd(act)
                    y_down = self.y_down[:K_reg]
                    self._run_down(
                        act_r, slots_t, y_down,
                        OUT=2048, NG=64
                    )
                    reg_out = (y_down * reg_w[:, None]).sum(dim=0, keepdim=True)
                    return promoted_out + reg_out
                return promoted_out

            for i, e in enumerate(topk_ids_list):
                self.active_slots[i] = self._get_slot(e)
            slots_t = self.active_slots[:K]
            
            xr = self.rot_2048.fwd(hidden_states.view(-1).bfloat16())
            y_gu = self.y_gu[:K]
            self._run_gate_up(
                xr, slots_t, y_gu,
                OUT=1024, NG=256
            )
            gate = y_gu[:, :512]
            up = y_gu[:, 512:]
            act = (F.silu(gate.float()) * up.float()).bfloat16()
            act_r = self.rot_512.fwd(act)
            y_down = self.y_down[:K]
            self._run_down(
                act_r, slots_t, y_down,
                OUT=2048, NG=64
            )
            out = (y_down * topk_w[:, None]).sum(dim=0, keepdim=True)
            return out
        else:
            outs = []
            for t in range(T):
                topk_ids_list = topk_idx[t].view(-1).tolist()
                topk_w = topk_weights[t].view(-1).bfloat16()
                xt = hidden_states[t:t+1].bfloat16()

                if self.promoted_3b and any(e in self.promoted_3b for e in topk_ids_list):
                    promoted_out = torch.zeros(1, 2048, device=self.device, dtype=torch.bfloat16)
                    regular_slots = []
                    regular_weights = []
                    for k_idx, e_id in enumerate(topk_ids_list):
                        w_k = topk_w[k_idx]
                        if e_id in self.promoted_3b:
                            W_gu = self.promoted_3b[e_id]["gu"]
                            W_dn = self.promoted_3b[e_id]["dn"]
                            y_gu = F.linear(xt, W_gu)
                            g, u = y_gu[:, :512], y_gu[:, 512:]
                            act = (F.silu(g.float()) * u.float()).bfloat16()
                            promoted_out += F.linear(act, W_dn) * w_k
                        else:
                            regular_slots.append(self._get_slot(e_id))
                            regular_weights.append(w_k)

                    if regular_slots:
                        slots_t = torch.tensor(regular_slots, dtype=torch.int32, device=self.device)
                        reg_w = torch.stack(regular_weights)
                        K_reg = len(regular_slots)
                        xr = self.rot_2048.fwd(xt.view(-1))
                        y_gu = self.y_gu[:K_reg]
                        self._run_gate_up(xr, slots_t, y_gu, OUT=1024, NG=256)
                        gate = y_gu[:, :512]
                        up = y_gu[:, 512:]
                        act = (F.silu(gate.float()) * up.float()).bfloat16()
                        act_r = self.act_r[:K_reg]
                        act_r.copy_(self.rot_512.fwd(act))
                        y_down = self.y_down[:K_reg]
                        self._run_down(act_r, slots_t, y_down, OUT=2048, NG=64)
                        outs.append(promoted_out + (y_down * reg_w[:, None]).sum(dim=0, keepdim=True))
                    else:
                        outs.append(promoted_out)
                else:
                    slots = [self._get_slot(e) for e in topk_ids_list]
                    slots_t = torch.tensor(slots, dtype=torch.int32, device=self.device)
                    xr = self.rot_2048.fwd(xt.view(-1))
                    y_gu = self.y_gu[:K]
                    self._run_gate_up(xr, slots_t, y_gu, OUT=1024, NG=256)
                    gate = y_gu[:, :512]
                    up = y_gu[:, 512:]
                    act = (F.silu(gate.float()) * up.float()).bfloat16()
                    act_r = self.act_r[:K]
                    act_r.copy_(self.rot_512.fwd(act))
                    y_down = self.y_down[:K]
                    self._run_down(act_r, slots_t, y_down, OUT=2048, NG=64)
                    outs.append((y_down * topk_w[:, None]).sum(dim=0, keepdim=True))

            return torch.cat(outs, dim=0)

