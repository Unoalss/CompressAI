import sys, time
from pathlib import Path
sys.path.insert(0, r"e:\compressAI")
import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeTextConfig
)
from accelerate import init_empty_weights
from vq2 import vq
from vq2.inferencia_35b import PackedMoEExpert

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

cfg = AutoConfig.from_pretrained("e:/compressAI/modelo_35b", trust_remote_code=True)
text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg.text_config
if isinstance(text_cfg, dict):
    text_cfg = Qwen3_5MoeTextConfig(**text_cfg)

print("1. Instanciando Camada 0...")
with init_empty_weights():
    layer0 = Qwen3_5MoeDecoderLayer(text_cfg, layer_idx=0)

print("2. Carregando Shard 1 e Codebook...")
s1 = torch.load("e:/compressAI/vq2_35b_out/quant_model-00001-of-00021.safetensors.pt", weights_only=False, map_location="cpu")
cbs_data = torch.load("e:/compressAI/vq2_out/codebooks.pt", weights_only=False)
cb16 = cbs_data["16"].to(device=DEVICE, dtype=torch.bfloat16)

# Carrega MoE experts com Triton
q_down = s1["quant"]["model.language_model.layers.0.mlp.experts.down_proj"]
q_gate_up = s1["quant"]["model.language_model.layers.0.mlp.experts.gate_up_proj"]
moe_experts = PackedMoEExpert(q_gate_up, q_down, cb16, device=DEVICE)
layer0.mlp.experts = moe_experts

def dequant_matrix(qdict):
    out, n = qdict["shape"]
    cfg = qdict["cfg"]
    tag = qdict["tag"]
    idx = qdict["idx"]
    scale = qdict["scale"].to(DEVICE).float()
    rec = 0
    for i in range(len(cfg)):
        cb_key = "_".join(str(b) for b in cfg[: i + 1])
        cb = cbs_data[cb_key].to(DEVICE)
        rec = rec + cb[idx[i].to(DEVICE).long()]
    Q = rec.reshape(out, n) * scale[:, None]
    W = vq.get_rot(tag, n, DEVICE).inv(Q)
    return W.to(torch.bfloat16)

# Carrega todos os outros parametros da camada 0
print("3. Preenchendo todos os pesos da Camada 0...")
state_dict = {}
for name, _ in layer0.named_parameters():
    if "experts" in name:
        continue
    full_name = f"model.language_model.layers.0.{name}"
    if full_name in s1["dense"]:
        t = s1["dense"][full_name].to(device=DEVICE, dtype=torch.bfloat16)
        # Ajusta nome para a camada
        state_dict[name] = t
    elif full_name in s1["quant"]:
        W = dequant_matrix(s1["quant"][full_name])
        state_dict[name] = W
    else:
        print(f"AVISO: {name} nao encontrado no Shard 1!")

# Atribui os pesos na camada
for n, p in state_dict.items():
    submod_names = n.split(".")
    obj = layer0
    for sm in submod_names[:-1]:
        obj = getattr(obj, sm)
    setattr(obj, submod_names[-1], nn.Parameter(p, requires_grad=False))

print("4. Executando Forward Pass na Camada 0 com tensor real na GPU...")
layer0.to(DEVICE)
x = torch.randn(1, 1, 2048, device=DEVICE, dtype=torch.bfloat16)

with torch.no_grad():
    out = layer0(x)

if isinstance(out, tuple):
    y = out[0]
else:
    y = out

print(f"Sucesso absoluto!")
print(f"Input shape:  {list(x.shape)}")
print(f"Output shape: {list(y.shape)}")
print(f"Output norm:  {y.norm().item():.4f}")
