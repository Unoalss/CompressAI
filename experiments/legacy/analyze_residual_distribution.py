import os
import sys
import time
import json
import torch

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from transformers import AutoModelForCausalLM, AutoTokenizer
from test_qwen_ffn_single_layer import quantize_matrix_h5a_batched

MODEL_DIR = "E:/agente/modelo"
device = "cpu"
torch.set_num_threads(os.cpu_count() or 8)

print("=" * 100)
print("ANÁLISE DE DISTRIBUIÇÃO DO ERRO RESIDUAL: FROBENIUS VS FUNCIONAL")
print("=" * 100)

act_profile_path = "e:/agente/activation_rms_profile.pt"
if not os.path.exists(act_profile_path):
    raise FileNotFoundError("Perfil de ativacao nao encontrado!")

act_profiles = torch.load(act_profile_path)
print(f"Perfis de ativacao carregados: {len(act_profiles)} componentes.")

print("Carregando modelo BF16...")
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
model.eval()

attn_layers = [3, 7, 11, 15, 19, 23]
deltanet_layers = [i for i in range(24) if i not in attn_layers]

def quantize_h5a_v2_baseline(W, act_rms=None, p_prot=0.0, seed=42):
    orig_dtype = W.dtype
    W_float = W.float()
    W_rec, _, _, _, _ = quantize_matrix_h5a_batched(W_float, num_ctrl=3, num_spec=3, p_crit=0.05, seed=seed)
    W_rec = W_rec.float()
    if p_prot > 0 and act_rms is not None and act_rms.numel() == W.shape[1]:
        E = W_float - W_rec
        col_weight_err = torch.sum(E ** 2, dim=0)
        col_func_err = col_weight_err * (act_rms.float() ** 2)
        k_crit = max(1, int(p_prot * W.shape[1]))
        top_cols = torch.topk(col_func_err, k=k_crit).indices
        W_rec[:, top_cols] = W_float[:, top_cols]
    return W_rec.to(orig_dtype)

# Mapear todas as 150 matrizes de calculo
matrix_records = []

print("Calculando erros residuais em todas as matrizes do modelo...")
t0 = time.time()

# 1. FFN (72 matrizes)
for i in range(24):
    layer = model.model.layers[i]
    targets = [
        ("gate_proj", layer.mlp.gate_proj, 0.0075, "Tier 2"),
        ("up_proj",   layer.mlp.up_proj,   0.0075, "Tier 2"),
        ("down_proj", layer.mlp.down_proj, 0.0200, "Tier 1")
    ]
    for w_name, mod, p_prot, tier in targets:
        name = f"L{i}.mlp.{w_name}"
        W_orig = mod.weight.data.clone().float()
        rms = act_profiles.get(name)
        W_q = quantize_h5a_v2_baseline(W_orig, act_rms=rms, p_prot=p_prot, seed=42+i).float()
        
        E = W_orig - W_q
        frob_err = torch.norm(E).item()
        rel_frob = frob_err / torch.norm(W_orig).item()
        
        if rms is not None and rms.numel() == W_orig.shape[1]:
            # ||E[:, j]||^2 * RMS(X_j)^2
            col_weight_err = torch.sum(E ** 2, dim=0)
            func_err = torch.sum(col_weight_err * (rms.float() ** 2)).item()
            # erro funcional relativo a W_orig * X
            col_orig_power = torch.sum(W_orig ** 2, dim=0)
            orig_func_power = torch.sum(col_orig_power * (rms.float() ** 2)).item()
            rel_func = (func_err / (orig_func_power + 1e-12)) ** 0.5
        else:
            func_err = 0.0
            rel_func = 0.0
            
        matrix_records.append({
            "name": name,
            "layer": i,
            "type": "FFN",
            "tier": tier,
            "shape": list(W_orig.shape),
            "frob_err": frob_err,
            "rel_frob": rel_frob,
            "func_err": func_err,
            "rel_func": rel_func
        })

# 2. Self-Attention (24 matrizes)
for i in attn_layers:
    layer = model.model.layers[i]
    targets = [
        ("q_proj", layer.self_attn.q_proj, 0.0, "Tier 3"),
        ("k_proj", layer.self_attn.k_proj, 0.0, "Tier 3"),
        ("v_proj", layer.self_attn.v_proj, 0.0075, "Tier 2"),
        ("o_proj", layer.self_attn.o_proj, 0.015 if i <= 11 else 0.0075, "Tier 1" if i <= 11 else "Tier 2")
    ]
    for w_name, mod, p_prot, tier in targets:
        name = f"L{i}.self_attn.{w_name}"
        W_orig = mod.weight.data.clone().float()
        rms = act_profiles.get(name)
        W_q = quantize_h5a_v2_baseline(W_orig, act_rms=rms, p_prot=p_prot, seed=100+i).float()
        E = W_orig - W_q
        frob_err = torch.norm(E).item()
        rel_frob = frob_err / torch.norm(W_orig).item()
        if rms is not None and rms.numel() == W_orig.shape[1]:
            col_weight_err = torch.sum(E ** 2, dim=0)
            func_err = torch.sum(col_weight_err * (rms.float() ** 2)).item()
            col_orig_power = torch.sum(W_orig ** 2, dim=0)
            orig_func_power = torch.sum(col_orig_power * (rms.float() ** 2)).item()
            rel_func = (func_err / (orig_func_power + 1e-12)) ** 0.5
        else:
            func_err = 0.0
            rel_func = 0.0
        matrix_records.append({
            "name": name,
            "layer": i,
            "type": "Self-Attn",
            "tier": tier,
            "shape": list(W_orig.shape),
            "frob_err": frob_err,
            "rel_frob": rel_frob,
            "func_err": func_err,
            "rel_func": rel_func
        })

# 3. DeltaNet (54 matrizes)
for i in deltanet_layers:
    layer = model.model.layers[i]
    targets = [
        ("in_proj_qkv", layer.linear_attn.in_proj_qkv, 0.0, "Tier 3"),
        ("in_proj_z",   layer.linear_attn.in_proj_z,   0.0075, "Tier 2"),
        ("out_proj",    layer.linear_attn.out_proj,    0.010, "Tier 2")
    ]
    for w_name, mod, p_prot, tier in targets:
        name = f"L{i}.linear_attn.{w_name}"
        W_orig = mod.weight.data.clone().float()
        rms = act_profiles.get(name)
        W_q = quantize_h5a_v2_baseline(W_orig, act_rms=rms, p_prot=p_prot, seed=200+i).float()
        E = W_orig - W_q
        frob_err = torch.norm(E).item()
        rel_frob = frob_err / torch.norm(W_orig).item()
        if rms is not None and rms.numel() == W_orig.shape[1]:
            col_weight_err = torch.sum(E ** 2, dim=0)
            func_err = torch.sum(col_weight_err * (rms.float() ** 2)).item()
            col_orig_power = torch.sum(W_orig ** 2, dim=0)
            orig_func_power = torch.sum(col_orig_power * (rms.float() ** 2)).item()
            rel_func = (func_err / (orig_func_power + 1e-12)) ** 0.5
        else:
            func_err = 0.0
            rel_func = 0.0
        matrix_records.append({
            "name": name,
            "layer": i,
            "type": "DeltaNet",
            "tier": tier,
            "shape": list(W_orig.shape),
            "frob_err": frob_err,
            "rel_frob": rel_frob,
            "func_err": func_err,
            "rel_func": rel_func
        })

dt = time.time() - t0
print(f"-> 150 matrizes analisadas em {dt:.1f}s!\n")

# Classificar por Erro Frobenius Absoluto ||E||_F
sorted_by_frob = sorted(matrix_records, key=lambda x: x["frob_err"], reverse=True)

# Classificar por Erro Funcional
sorted_by_func = sorted(matrix_records, key=lambda x: x["func_err"], reverse=True)

print("=" * 105)
print("TOP 15 MATRIZES COM MAIOR ERRO FROBENIUS: ||E||_F")
print("=" * 105)
print(f"{'Pos':<4} | {'Nome da Matriz':<28} | {'Tipo':<10} | {'Tier':<8} | {'Shape':<14} | {'||E||_F':<10} | {'Rel Frob':<10} | {'Func Err'}")
print("-" * 105)
for idx, m in enumerate(sorted_by_frob[:15], 1):
    shape_str = f"{m['shape'][0]}x{m['shape'][1]}"
    print(f"{idx:<4} | {m['name']:<28} | {m['type']:<10} | {m['tier']:<8} | {shape_str:<14} | {m['frob_err']:<10.4f} | {m['rel_frob']*100:<9.2f}% | {m['func_err']:.6f}")

print("\n" + "=" * 105)
print("TOP 15 MATRIZES COM MAIOR ERRO FUNCIONAL: ||E[:,j]||² × RMS(X_j)²")
print("=" * 105)
print(f"{'Pos':<4} | {'Nome da Matriz':<28} | {'Tipo':<10} | {'Tier':<8} | {'Shape':<14} | {'Func Err':<12} | {'Rel Func':<10} | {'||E||_F'}")
print("-" * 105)
for idx, m in enumerate(sorted_by_func[:15], 1):
    shape_str = f"{m['shape'][0]}x{m['shape'][1]}"
    print(f"{idx:<4} | {m['name']:<28} | {m['type']:<10} | {m['tier']:<8} | {shape_str:<14} | {m['func_err']:<12.6f} | {m['rel_func']*100:<9.2f}% | {m['frob_err']:.4f}")

# Calcular rank correlation ou overlap entre Top 20
top20_frob_names = set(m["name"] for m in sorted_by_frob[:20])
top20_func_names = set(m["name"] for m in sorted_by_func[:20])
overlap_top20 = len(top20_frob_names.intersection(top20_func_names))
print("\n" + "=" * 105)
print(f"Sobreposicao entre Top 20 Frobenius e Top 20 Funcional: {overlap_top20}/20 ({overlap_top20*5:.1f}%)")
print("=" * 105)

output_data = {
    "sorted_by_frob": sorted_by_frob,
    "sorted_by_func": sorted_by_func,
    "overlap_top20": overlap_top20
}

with open("e:/agente/residual_error_distribution.json", "w") as f:
    json.dump(output_data, f, indent=2)

print("Resultados salvos em e:/agente/residual_error_distribution.json!")
