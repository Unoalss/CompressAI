import sys
sys.path.insert(0, ".")
import time, torch
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
from vq2.paged_runtime import PagedModelStore, PagedMoEExpert
from vq2 import vq
from vq2.kernels import vq_moe_parallel_gate_up, vq_moe_parallel_down

device = "cuda"
store = PagedModelStore("E:/compressAI/vq2_35b_out/qwen35b_full.vq2m")
cbs_data = torch.load("E:/compressAI/vq2_out/codebooks.pt", weights_only=False)
cb16 = cbs_data["16"].to(device=device, dtype=torch.bfloat16)

class FastPagedMoEExpert(nn.Module):
    def __init__(self, layer_idx: int, store: PagedModelStore, cb16, capacity=96, device=device):
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
        
        self.active_slots = torch.zeros(8, dtype=torch.int32, device=device)
        self.register_buffer("y_gu", torch.empty((8, 1024), device=device, dtype=torch.bfloat16), persistent=False)
        self.register_buffer("y_down", torch.empty((8, 2048), device=device, dtype=torch.bfloat16), persistent=False)
        self.register_buffer("act_r", torch.empty((8, 512), device=device, dtype=torch.bfloat16), persistent=False)

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
            
        gui, gus, dni, dns = self.store.get_expert_weights(self.layer_idx, e_id, device=self.device)
        self.pool_gu_idx[slot].copy_(gui)
        self.pool_gu_scale[slot].copy_(gus)
        self.pool_dn_idx[slot].copy_(dni)
        self.pool_dn_scale[slot].copy_(dns)
        
        self.expert_to_slot[e_id] = slot
        self.access_order[e_id] = None
        return slot

    def forward(self, hidden_states, topk_idx, topk_weights):
        T = hidden_states.shape[0]
        if T == 1:
            topk_ids_list = topk_idx.view(-1).tolist()
            topk_w = topk_weights.view(-1).bfloat16()
            
            for i in range(8):
                self.active_slots[i] = self._get_slot(topk_ids_list[i])
            
            xr = self.rot_2048.fwd(hidden_states.view(-1).bfloat16())
            vq_moe_parallel_gate_up(
                xr, self.active_slots, self.pool_gu_idx, self.cb16, self.pool_gu_scale, self.y_gu,
                OUT=1024, NG=256, rs=self.RS_2048
            )
            gate = self.y_gu[:, :512]
            up = self.y_gu[:, 512:]
            act = (F.silu(gate.float()) * up.float()).bfloat16()
            self.act_r.copy_(self.rot_512.fwd(act))
            vq_moe_parallel_down(
                self.act_r, self.active_slots, self.pool_dn_idx, self.cb16, self.pool_dn_scale, self.y_down,
                OUT=2048, NG=64, rs=self.RS_512
            )
            out = (self.y_down * topk_w[:, None]).sum(dim=0, keepdim=True)
            return out

orig_layer = PagedMoEExpert(layer_idx=0, store=store, cb16=cb16, device=device)
fast_layer = FastPagedMoEExpert(layer_idx=0, store=store, cb16=cb16, capacity=96, device=device)

# Gera sequencia realista de 100 passos com distribuicao Zipfiana entre 50 experts frequentes
torch.manual_seed(42)
hidden = torch.randn(1, 2048, device=device, dtype=torch.bfloat16)
steps_ids = []
for _ in range(100):
    # Escolhe 8 experts de um conjunto frequente de 50 experts
    e_ids = torch.randperm(50)[:8].unsqueeze(0).to(device)
    w = torch.softmax(torch.randn(1, 8, device=device), dim=-1).to(torch.bfloat16)
    steps_ids.append((e_ids, w))

# Warmup dos caches
for e_ids, w in steps_ids[:20]:
    _ = orig_layer(hidden, e_ids, w)
    _ = fast_layer(hidden, e_ids, w)
torch.cuda.synchronize()

# Mede tempo em 80 passos realistas
torch.cuda.synchronize()
t0 = time.perf_counter()
for e_ids, w in steps_ids[20:]:
    _ = orig_layer(hidden, e_ids, w)
torch.cuda.synchronize()
dt_orig = (time.perf_counter() - t0) / 80 * 1000

torch.cuda.synchronize()
t0 = time.perf_counter()
for e_ids, w in steps_ids[20:]:
    _ = fast_layer(hidden, e_ids, w)
torch.cuda.synchronize()
dt_fast = (time.perf_counter() - t0) / 80 * 1000

print(f"Orig Layer (com permutacoes de slots): {dt_orig:.3f} ms")
print(f"Fast Layer (com Zero-Copy Pool):      {dt_fast:.3f} ms")
print(f"Aceleracao: {dt_orig / dt_fast:.2f}x!")
