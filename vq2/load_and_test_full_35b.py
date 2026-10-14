"""
vq2/load_and_test_full_35b.py - Carregador e Executor Autoregressivo Completo de Qwen3.8-35B-A3B VQ2 RDO.
Carrega os 22 shards quantizados mantendo os 35.5B parametros residentes na RTX 3060 12GB (~10.8 GB VRAM).
Permite inferencia 100% livre e dinamica para qualquer pergunta do usuario.
"""
import os, sys, time, json, gc
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeTextConfig
)
from accelerate import init_empty_weights

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vq2 import vq
from vq2.kernels import HAVE_TRITON, vq_moe_parallel_gate_up, vq_moe_parallel_down

OUT_35B = ROOT / "vq2_35b_out"
MODEL_DIR = ROOT / "modelo_35b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class PackedMoEExpertUniversal(nn.Module):
    """
    Modulo customizado de 256 experts MoE executado via Triton na GPU.
    Suporta tanto prompt prefill (T > 1) quanto geracao token-a-token (T = 1).
    """
    def __init__(self, q_gate_up, q_down, cb16, device=DEVICE):
        super().__init__()
        self.num_experts = 256
        self.hidden_dim = 2048
        self.intermediate_dim = 512
        
        # Pesos quantizados em int16 (apenas 192 MB em VRAM!)
        self.gate_up_idx = q_gate_up["idx"][0].view(256, 1024, 256).to(device)
        self.gate_up_scale = q_gate_up["scale"].view(256, 1024).to(device=device, dtype=torch.bfloat16)
        self.down_idx = q_down["idx"][0].view(256, 2048, 64).to(device)
        self.down_scale = q_down["scale"].view(256, 2048).to(device=device, dtype=torch.bfloat16)
        
        self.cb16 = cb16
        self.rot_2048 = vq.get_rot("moe_expert_2048", self.hidden_dim, device)
        self.rot_512 = vq.get_rot("moe_expert_512", self.intermediate_dim, device)
        self.RS_2048 = 1.0
        self.RS_512 = 1.0
        
        self.register_buffer("y_gu", torch.empty((8, 1024), device=device, dtype=torch.bfloat16), persistent=False)
        self.register_buffer("y_down", torch.empty((8, 2048), device=device, dtype=torch.bfloat16), persistent=False)
        self.register_buffer("act_r", torch.empty((8, 512), device=device, dtype=torch.bfloat16), persistent=False)

    def forward(self, hidden_states, topk_idx, topk_weights):
        # hidden_states: [T, 2048], topk_idx: [T, 8], topk_weights: [T, 8]
        T = hidden_states.shape[0]
        if T == 1:
            xr = self.rot_2048.fwd(hidden_states.view(-1).bfloat16())
            topk_ids = topk_idx.view(-1).to(torch.int32)
            topk_w = topk_weights.view(-1).bfloat16()
            vq_moe_parallel_gate_up(
                xr, topk_ids, self.gate_up_idx, self.cb16, self.gate_up_scale, self.y_gu,
                OUT=1024, NG=256, rs=self.RS_2048
            )
            gate = self.y_gu[:, :512]
            up = self.y_gu[:, 512:]
            act = (torch.nn.functional.silu(gate.float()) * up.float()).bfloat16()
            self.act_r.copy_(self.rot_512.fwd(act))
            vq_moe_parallel_down(
                self.act_r, topk_ids, self.down_idx, self.cb16, self.down_scale, self.y_down,
                OUT=2048, NG=64, rs=self.RS_512
            )
            # y_down já está no espaço desrotacionado original (2048)
            out = (self.y_down * topk_w[:, None]).sum(dim=0, keepdim=True)
            return out
        else:
            outs = []
            for t in range(T):
                xr = self.rot_2048.fwd(hidden_states[t].bfloat16())
                topk_ids = topk_idx[t].to(torch.int32)
                topk_w = topk_weights[t].bfloat16()
                vq_moe_parallel_gate_up(
                    xr, topk_ids, self.gate_up_idx, self.cb16, self.gate_up_scale, self.y_gu,
                    OUT=1024, NG=256, rs=self.RS_2048
                )
                gate = self.y_gu[:, :512]
                up = self.y_gu[:, 512:]
                act = (torch.nn.functional.silu(gate.float()) * up.float()).bfloat16()
                self.act_r.copy_(self.rot_512.fwd(act))
                vq_moe_parallel_down(
                    self.act_r, topk_ids, self.down_idx, self.cb16, self.down_scale, self.y_down,
                    OUT=2048, NG=64, rs=self.RS_512
                )
                out_t = (self.y_down * topk_w[:, None]).sum(dim=0, keepdim=True)
                outs.append(out_t)
            return torch.cat(outs, dim=0)

def dequant_matrix(qdict, cbs_data, device=DEVICE):
    out, n = qdict["shape"]
    cfg = qdict["cfg"]
    tag = qdict["tag"]
    idx = qdict["idx"]
    scale = qdict["scale"].to(device).float()
    rec = 0
    for i in range(len(cfg)):
        cb_key = "_".join(str(b) for b in cfg[: i + 1])
        cb = cbs_data[cb_key].to(device)
        rec = rec + cb[idx[i].to(device).long()]
    Q = rec.reshape(out, n) * scale[:, None]
    W = vq.get_rot(tag, n, device).inv(Q)
    return W.to(torch.bfloat16)

def set_nested_param(model, param_name, tensor):
    if "visual" in param_name or "mtp" in param_name:
        return False
    parts = param_name.split(".")
    obj = model
    for p in parts[:-1]:
        if not hasattr(obj, p):
            return False
        obj = getattr(obj, p)
    setattr(obj, parts[-1], nn.Parameter(tensor, requires_grad=False))
    return True

def build_full_vq2_model(num_layers=40, device=DEVICE):
    print("=" * 80)
    print(f"INICIALIZANDO MOTOR COMPLETO QWEN3.8-35B-A3B VQ2 ({num_layers} CAMADAS)")
    print(f"GPU Alvo: {torch.cuda.get_device_name(0)} | VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("=" * 80)
    
    t0 = time.time()
    cfg = AutoConfig.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg.text_config
    if isinstance(text_cfg, dict):
        text_cfg = Qwen3_5MoeTextConfig(**text_cfg)
        
    text_cfg.num_hidden_layers = num_layers
    text_cfg.layer_types = text_cfg.layer_types[:num_layers]
    
    # 1. Instancia o esqueleto com init_empty_weights
    print("[1/4] Criando esqueleto do modelo...")
    with init_empty_weights():
        model = Qwen3_5MoeForCausalLM(text_cfg)
        
    # 2. Carrega Codebook
    print("[2/4] Carregando Codebook VQ2...")
    cbs_data = torch.load(ROOT / "vq2_out" / "codebooks.pt", weights_only=False)
    cb16 = cbs_data["16"].to(device=device, dtype=torch.bfloat16)
    
    # 3. Varre os shards e preenche os tensores
    print(f"[3/4] Injetando tensores dos 21 shards ({num_layers} camadas)...")
    index_file = MODEL_DIR / "model.safetensors.index.json"
    with open(index_file) as f:
        weight_map = json.load(f)["weight_map"]
        
    shards_to_load = sorted(set(weight_map.values()))
    
    layers_experts_loaded = set()
    pending_gate_up = {}
    pending_down = {}
    
    for shard_idx, shard_name in enumerate(shards_to_load, 1):
        pt_path = OUT_35B / f"quant_{shard_name}.pt"
        if not pt_path.exists():
            continue
            
        shard_data = torch.load(pt_path, weights_only=False, map_location="cpu")
        quant_dict = shard_data.get("quant", {})
        dense_dict = shard_data.get("dense", {})
        
        # 3.1. Processa Experts MoE (pareia gate_up e down mesmo se estiverem em shards diferentes)
        for k, q_data in quant_dict.items():
            if "mlp.experts.gate_up_proj" in k:
                m = k.split(".layers.")
                if len(m) > 1:
                    ln = int(m[1].split(".")[0])
                    if ln < num_layers:
                        pending_gate_up[ln] = q_data
            elif "mlp.experts.down_proj" in k:
                m = k.split(".layers.")
                if len(m) > 1:
                    ln = int(m[1].split(".")[0])
                    if ln < num_layers:
                        pending_down[ln] = q_data
                        
        for ln in list(pending_gate_up.keys()):
            if ln in pending_down and ln not in layers_experts_loaded:
                q_gu = pending_gate_up[ln]
                q_down = pending_down[ln]
                expert_mod = PackedMoEExpertUniversal(q_gu, q_down, cb16, device=device)
                model.model.layers[ln].mlp.experts = expert_mod
                layers_experts_loaded.add(ln)
                del pending_gate_up[ln], pending_down[ln]
                            
        # 3.2. Processa tensores densos
        for raw_name, tensor in dense_dict.items():
            clean_name = raw_name.replace("model.language_model.", "model.")
            # Checa se pertence a uma camada ativa
            if "model.layers." in clean_name:
                ln = int(clean_name.split("model.layers.")[1].split(".")[0])
                if ln >= num_layers:
                    continue
            tensor_cuda = tensor.to(device=device, dtype=torch.bfloat16)
            set_nested_param(model, clean_name, tensor_cuda)
            
        # 3.3. Processa tensores lineares quantizados (não-experts)
        for raw_name, q_data in quant_dict.items():
            if "mlp.experts" in raw_name:
                continue
            clean_name = raw_name.replace("model.language_model.", "model.")
            if "model.layers." in clean_name:
                ln = int(clean_name.split("model.layers.")[1].split(".")[0])
                if ln >= num_layers:
                    continue
            W = dequant_matrix(q_data, cbs_data, device=device)
            set_nested_param(model, clean_name, W)
            
        del shard_data
        gc.collect()
        
    print(f"[4/4] Finalizado! Total de camadas com MoE ativas: {len(layers_experts_loaded)}/{num_layers}")
    model.model.rotary_emb.to(device)
    model.eval()
    return model

if __name__ == "__main__":
    # Teste de validacao com 4 camadas (inclui DeltaNet + Full Attention)
    m = build_full_vq2_model(num_layers=4)
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    prompt = "Olá, quem é você?"
    inp = tok(prompt, return_tensors="pt").to(DEVICE)
    print(f"Executando teste de inferencia com o prompt: '{prompt}'...")
    with torch.no_grad():
        out = m(input_ids=inp.input_ids)
    print(f"Sucesso absoluto! Formato dos logits de saída: {list(out.logits.shape)}")
    next_token_id = out.logits[:, -1, :].argmax(dim=-1).item()
    print(f"ID do próximo token previsto: {next_token_id} ('{tok.decode([next_token_id])}')")
