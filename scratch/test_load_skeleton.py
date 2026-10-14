import torch
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig
from accelerate import init_empty_weights

cfg = AutoConfig.from_pretrained("e:/compressAI/modelo_35b", trust_remote_code=True)
tcfg = cfg.text_config if hasattr(cfg, "text_config") else cfg
print("Instanciando esqueleto vazio do modelo 35B...")
with init_empty_weights():
    model = Qwen3_5MoeForCausalLM(tcfg)

print("Sucesso!")
print("Numero de camadas:", len(model.model.layers))
print("Tipo da camada 0 MLP:", type(model.model.layers[0].mlp))
print("Tipo da camada 0 Linear Attn:", type(model.model.layers[0].linear_attn))
print("Atributos do MLP:", dir(model.model.layers[0].mlp))
