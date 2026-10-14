import os
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "E:/agente/modelo"
device = "cpu"
torch.set_num_threads(os.cpu_count() or 8)

print("Carregando modelo...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
model.eval()

calib_texts = [
    "Quantum mechanics describes nature at the smallest scales of energy levels of atoms.",
    "def binary_search(array, target): low = 0; high = len(array) - 1; while low <= high: mid = (low + high) // 2",
    "The solar system formed approximately 4.6 billion years ago from the gravitational collapse of a molecular cloud.",
    "The capital of France is Paris, located on the Seine River in northern France."
]

print("Registrando hooks de ativacao em todas as camadas...")
activation_stats = {}
hooks = []

def get_hook(name):
    def hook(module, input, output):
        x = input[0].detach()
        x_flat = x.reshape(-1, x.shape[-1]).float()
        rms = torch.sqrt(torch.mean(x_flat ** 2, dim=0))
        if name not in activation_stats:
            activation_stats[name] = []
        activation_stats[name].append(rms)
    return hook

for i, layer in enumerate(model.model.layers):
    # FFN
    hooks.append(layer.mlp.gate_proj.register_forward_hook(get_hook(f"L{i}.mlp.gate_proj")))
    hooks.append(layer.mlp.up_proj.register_forward_hook(get_hook(f"L{i}.mlp.up_proj")))
    hooks.append(layer.mlp.down_proj.register_forward_hook(get_hook(f"L{i}.mlp.down_proj")))
    # Attn ou DeltaNet
    if hasattr(layer, "self_attn"):
        hooks.append(layer.self_attn.q_proj.register_forward_hook(get_hook(f"L{i}.self_attn.q_proj")))
        hooks.append(layer.self_attn.k_proj.register_forward_hook(get_hook(f"L{i}.self_attn.k_proj")))
        hooks.append(layer.self_attn.v_proj.register_forward_hook(get_hook(f"L{i}.self_attn.v_proj")))
        hooks.append(layer.self_attn.o_proj.register_forward_hook(get_hook(f"L{i}.self_attn.o_proj")))
    elif hasattr(layer, "linear_attn"):
        hooks.append(layer.linear_attn.in_proj_qkv.register_forward_hook(get_hook(f"L{i}.linear_attn.in_proj_qkv")))
        hooks.append(layer.linear_attn.in_proj_z.register_forward_hook(get_hook(f"L{i}.linear_attn.in_proj_z")))
        hooks.append(layer.linear_attn.out_proj.register_forward_hook(get_hook(f"L{i}.linear_attn.out_proj")))

t0 = time.time()
with torch.no_grad():
    for text in calib_texts:
        inp = tokenizer(text, return_tensors="pt")
        model(**inp)

for h in hooks:
    h.remove()

t_pass = time.time() - t0
print(f"Coleta de ativacoes concluida em {t_pass:.2f}s para {len(activation_stats)} componentes!")

# Consolidar RMS medio por componente
final_rms = {}
summary = []
for name, rms_list in activation_stats.items():
    stacked = torch.stack(rms_list, dim=0) # [num_texts, dim]
    mean_rms = torch.mean(stacked, dim=0)
    final_rms[name] = mean_rms
    ratio = (mean_rms.max() / (mean_rms.median() + 1e-6)).item()
    summary.append((name, mean_rms.min().item(), mean_rms.median().item(), mean_rms.max().item(), ratio))

summary.sort(key=lambda x: x[4], reverse=True)
print(f"\nTop 15 maiores razoes de outliers de ativacao (max/median):")
print(f"{'Componente':<30} | {'Min RMS':<10} | {'Med RMS':<10} | {'Max RMS':<10} | {'Outlier Ratio'}")
print("-" * 75)
for item in summary[:15]:
    print(f"{item[0]:<30} | {item[1]:<10.4f} | {item[2]:<10.4f} | {item[3]:<10.4f} | {item[4]:.2f}x")

torch.save(final_rms, "e:/agente/activation_rms_profile.pt")
print("\nPerfil de ativacao salvo com sucesso em e:/agente/activation_rms_profile.pt!")
