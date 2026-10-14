import os
import sys
import time
import json
import gc
import psutil
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

print("=" * 115)
print("H5-A v2.1: INVESTIGAÇÃO E ABLATION NAS 8 MATRIZES DELTANET in_proj_qkv")
print("Objetivo: Testar proteção funcional (0% a 4%) e residual Rank 8 no maior concentrador de erro residual")
print("Baseline oficial congelado: H5-A v2 (847.8 MB, PPL 9.98)")
print("=" * 115)

# -----------------------------------------------------------------------------
# 1. CARREGAR PERFIL DE ATIVAÇÕES
# -----------------------------------------------------------------------------
act_profile_path = "e:/agente/activation_rms_profile.pt"
if not os.path.exists(act_profile_path):
    raise FileNotFoundError("Perfil de ativacao nao encontrado!")

act_profiles = torch.load(act_profile_path)
print(f"Perfis de ativacao carregados: {len(act_profiles)} componentes.")

# -----------------------------------------------------------------------------
# 2. FUNÇÃO BASELINE H5-A v2
# -----------------------------------------------------------------------------
def quantize_matrix_h5a_functional(W, act_rms=None, p_prot=0.0, seed=42):
    orig_dtype = W.dtype
    W_float = W.float()
    W_rec, _, _, _, _ = quantize_matrix_h5a_batched(
        W_float, num_ctrl=3, num_spec=3, p_crit=0.05, seed=seed
    )
    W_rec = W_rec.float()
    if p_prot > 0 and act_rms is not None and act_rms.numel() == W.shape[1]:
        E = W_float - W_rec
        col_weight_err = torch.sum(E ** 2, dim=0)
        col_func_err = col_weight_err * (act_rms.float() ** 2)
        k_crit = max(1, int(p_prot * W.shape[1]))
        top_cols = torch.topk(col_func_err, k=k_crit).indices
        W_rec[:, top_cols] = W_float[:, top_cols]
    return W_rec.to(orig_dtype)

# -----------------------------------------------------------------------------
# 3. PIPELINE DE AVALIAÇÃO
# -----------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

domains = {
    "Ciencia": "Quantum mechanics describes nature at the smallest scales of energy levels of atoms and subatomic particles.",
    "Codigo": "def binary_search(array, target): low = 0; high = len(array) - 1; while low <= high: mid = (low + high) // 2",
    "Historia": "The ancient library of Alexandria was once the largest repository of knowledge in the Hellenistic world.",
    "Enciclopedia": "The solar system formed approximately 4.6 billion years ago from the gravitational collapse of a molecular cloud."
}

prompts = [
    ("Capital", "The capital of France is"),
    ("Fibonacci", "def fibonacci(n):\n    \"\"\"Return nth Fibonacci number\"\"\"\n"),
    ("Newton", "Newton's second law of motion states that")
]

def eval_ppl(m):
    scores = {}
    with torch.no_grad():
        for d, txt in domains.items():
            inputs = tokenizer(txt, return_tensors="pt")
            loss = m(**inputs, labels=inputs.input_ids).loss
            scores[d] = torch.exp(loss).item()
    scores["Media"] = sum(scores.values()) / len(scores)
    return scores

def eval_gen(m, max_tokens=20):
    results = {}
    for name, p in prompts:
        inp = tokenizer(p, return_tensors="pt")
        t0 = time.time()
        with torch.no_grad():
            out = m.generate(**inp, max_new_tokens=max_tokens, do_sample=False)
        dt = time.time() - t0
        gen_tokens = out[0][inp.input_ids.shape[1]:]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip().replace("\n", " ")
        tok_s = len(gen_tokens) / dt if dt > 0 else 0
        results[name] = {"text": text, "tok_per_sec": tok_s}
    return results

# -----------------------------------------------------------------------------
# 4. CARREGAR MODELO BF16
# -----------------------------------------------------------------------------
print("\n[1/5] Carregando modelo base Qwen3.5-2B em BF16 (model.safetensors)...")
t0_load = time.time()
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
model.eval()
print(f"-> Carregado em {time.time()-t0_load:.2f}s!")

attn_layers = [3, 7, 11, 15, 19, 23]
deltanet_layers = [i for i in range(24) if i not in attn_layers]

# As 8 maiores matrizes DeltaNet em erro residual:
target_dnet_layers = [18, 17, 20, 21, 16, 22, 4, 12]
print(f"Camadas DeltaNet críticas alvo: {target_dnet_layers}")

# -----------------------------------------------------------------------------
# 5. QUANTIZAÇÃO BASELINE DO MODELO COMPLETO
#    E armazenamento de W_orig das matrizes-alvo para testes
# -----------------------------------------------------------------------------
print("\n[2/5] Quantizando modelo com baseline H5-A v2 e preparando matrizes alvo...")
t0_quant = time.time()

# Guardar dados de SVD e pesos base das 24 down_proj (para combinações com FFN Rank 8)
ffn_svd_data = {}

for i in range(24):
    l = model.model.layers[i]
    rms_gate = act_profiles.get(f"L{i}.mlp.gate_proj")
    rms_up   = act_profiles.get(f"L{i}.mlp.up_proj")
    rms_down = act_profiles.get(f"L{i}.mlp.down_proj")
    
    rg = quantize_matrix_h5a_functional(l.mlp.gate_proj.weight.data, act_rms=rms_gate, p_prot=0.0075, seed=42+i)
    ru = quantize_matrix_h5a_functional(l.mlp.up_proj.weight.data,   act_rms=rms_up,   p_prot=0.0075, seed=42+i)
    
    w_orig_down = l.mlp.down_proj.weight.data.clone().float()
    rd = quantize_matrix_h5a_functional(l.mlp.down_proj.weight.data, act_rms=rms_down, p_prot=0.020, seed=42+i)
    
    E_down = w_orig_down - rd.float()
    U_f, S_f, V_f = torch.svd_lowrank(E_down, q=8, niter=4)
    ffn_svd_data[i] = {
        "W_base": rd.clone(),
        "C_rank8": ((U_f * S_f) @ V_f.T).to(torch.bfloat16)
    }
    
    l.mlp.gate_proj.weight.data.copy_(rg)
    l.mlp.up_proj.weight.data.copy_(ru)
    l.mlp.down_proj.weight.data.copy_(rd)

for i in attn_layers:
    l = model.model.layers[i]
    rms_q = act_profiles.get(f"L{i}.self_attn.q_proj")
    rms_k = act_profiles.get(f"L{i}.self_attn.k_proj")
    rms_v = act_profiles.get(f"L{i}.self_attn.v_proj")
    rms_o = act_profiles.get(f"L{i}.self_attn.o_proj")
    rq = quantize_matrix_h5a_functional(l.self_attn.q_proj.weight.data, act_rms=rms_q, p_prot=0.0, seed=100+i)
    rk = quantize_matrix_h5a_functional(l.self_attn.k_proj.weight.data, act_rms=rms_k, p_prot=0.0, seed=100+i)
    rv = quantize_matrix_h5a_functional(l.self_attn.v_proj.weight.data, act_rms=rms_v, p_prot=0.0075, seed=100+i)
    p_o = 0.015 if i <= 11 else 0.0075
    ro = quantize_matrix_h5a_functional(l.self_attn.o_proj.weight.data, act_rms=rms_o, p_prot=p_o, seed=100+i)
    l.self_attn.q_proj.weight.data.copy_(rq)
    l.self_attn.k_proj.weight.data.copy_(rk)
    l.self_attn.v_proj.weight.data.copy_(rv)
    l.self_attn.o_proj.weight.data.copy_(ro)

# Guardar dados das 8 matrizes DeltaNet in_proj_qkv
dnet_targets_data = {}

for i in deltanet_layers:
    l = model.model.layers[i]
    rms_qkv = act_profiles.get(f"L{i}.linear_attn.in_proj_qkv")
    rms_z   = act_profiles.get(f"L{i}.linear_attn.in_proj_z")
    rms_out = act_profiles.get(f"L{i}.linear_attn.out_proj")
    
    w_orig_qkv = l.linear_attn.in_proj_qkv.weight.data.clone().float()
    rqkv = quantize_matrix_h5a_functional(l.linear_attn.in_proj_qkv.weight.data, act_rms=rms_qkv, p_prot=0.0, seed=200+i)
    rz   = quantize_matrix_h5a_functional(l.linear_attn.in_proj_z.weight.data,   act_rms=rms_z,   p_prot=0.0075, seed=200+i)
    rout = quantize_matrix_h5a_functional(l.linear_attn.out_proj.weight.data,    act_rms=rms_out, p_prot=0.010, seed=200+i)
    
    if i in target_dnet_layers:
        # Calcular SVD Rank 8 do erro de in_proj_qkv
        E_qkv = w_orig_qkv - rqkv.float()
        U_d, S_d, V_d = torch.svd_lowrank(E_qkv, q=8, niter=4)
        dnet_targets_data[i] = {
            "W_orig": w_orig_qkv,
            "W_base": rqkv.clone(),
            "rms": rms_qkv,
            "C_rank8": ((U_d * S_d) @ V_d.T).to(torch.bfloat16)
        }
        
    l.linear_attn.in_proj_qkv.weight.data.copy_(rqkv)
    l.linear_attn.in_proj_z.weight.data.copy_(rz)
    l.linear_attn.out_proj.weight.data.copy_(rout)

# Embeddings Híbridos
w_emb = model.model.embed_tokens.weight.data
total_tokens = w_emb.shape[0]
for start in range(32768, total_tokens, 50000):
    end = min(start + 50000, total_tokens)
    rec, _, _, _, _ = quantize_matrix_h5a_batched(w_emb[start:end], num_ctrl=3, num_spec=3, p_crit=0.05, seed=500+start)
    w_emb[start:end].copy_(rec)

print(f"-> Modelo quantizado com baseline H5-A v2 em {time.time()-t0_quant:.1f}s.")

# -----------------------------------------------------------------------------
# 6. TESTE A: SWEEP DE PROTEÇÃO FUNCIONAL NAS 8 in_proj_qkv
#    Níveis: 0%, 0.5%, 1.0%, 2.0%, 4.0%
# -----------------------------------------------------------------------------
base_model_mb = 847.8
total_model_params = 2274069824
prot_levels = [0.0, 0.005, 0.010, 0.020, 0.040]
sweep_func_results = []

print("\n" + "=" * 115)
print("TESTE A: SWEEP DE PROTEÇÃO FUNCIONAL NAS 8 in_proj_qkv (0% a 4%)")
print("=" * 115)

base_ppl_media = None

for p in prot_levels:
    t0_p = time.time()
    
    # Aplicar proteção funcional de p% nas 8 matrizes
    for layer_idx in target_dnet_layers:
        mod = model.model.layers[layer_idx].linear_attn.in_proj_qkv
        data = dnet_targets_data[layer_idx]
        if p == 0.0:
            mod.weight.data.copy_(data["W_base"])
        else:
            w_rec = quantize_matrix_h5a_functional(data["W_orig"], act_rms=data["rms"], p_prot=p, seed=200+layer_idx)
            mod.weight.data.copy_(w_rec)
            
    # Calcular custo em MB: 8 matrizes com p * 2048 canais em BF16 vs 3.05 BPW
    # canais protegidos: p * 2048 colunas * 6144 linhas * (16 - 3.05) bits
    extra_bytes = 8 * (p * 2048) * 6144 * ((16.0 - 3.05) / 8.0)
    extra_mb = extra_bytes / (1024 * 1024)
    final_mb = base_model_mb + extra_mb
    effective_bpw = (final_mb * 1024 * 1024 * 8) / total_model_params
    
    ppl = eval_ppl(model)
    ppl_media = ppl["Media"]
    
    if p == 0.0:
        base_ppl_media = ppl_media
        ganho = 0.0
        eficiencia = 0.0
    else:
        ganho = base_ppl_media - ppl_media
        eficiencia = ganho / extra_mb if extra_mb > 0 else 0.0
        
    dt_eval = time.time() - t0_p
    sweep_func_results.append({
        "prot_level": f"{p*100:.1f}%",
        "final_mb": round(final_mb, 2),
        "extra_mb": round(extra_mb, 2),
        "effective_bpw": round(effective_bpw, 3),
        "ppl_codigo": round(ppl["Codigo"], 2),
        "ppl_enciclopedia": round(ppl["Enciclopedia"], 2),
        "ppl_ciencia": round(ppl["Ciencia"], 2),
        "ppl_historia": round(ppl["Historia"], 2),
        "ppl_media": round(ppl_media, 2),
        "ganho_ppl": round(ganho, 3),
        "eficiencia": round(eficiencia, 4)
    })
    
    print(f"Proteção {p*100:4.1f}% | Tamanho: {final_mb:6.1f} MB (+{extra_mb:4.1f} MB) | PPL: {ppl_media:5.2f} (Ganho: {ganho:+5.2f}) | Eficiência: {eficiencia:.4f} PPL/MB")

# Restaurar baseline
for layer_idx in target_dnet_layers:
    model.model.layers[layer_idx].linear_attn.in_proj_qkv.weight.data.copy_(dnet_targets_data[layer_idx]["W_base"])

# -----------------------------------------------------------------------------
# 7. TESTE B: RESIDUAL LOW-RANK (RANK 8) NAS 8 in_proj_qkv
# -----------------------------------------------------------------------------
print("\n" + "=" * 115)
print("TESTE B: RESIDUAL LOW-RANK (RANK 8) NAS 8 in_proj_qkv")
print("=" * 115)

# Aplicar Rank 8 nas 8 in_proj_qkv
for layer_idx in target_dnet_layers:
    mod = model.model.layers[layer_idx].linear_attn.in_proj_qkv
    data = dnet_targets_data[layer_idx]
    w_with_r8 = data["W_base"] + data["C_rank8"]
    mod.weight.data.copy_(w_with_r8)

extra_params_dnet_r8 = 8 * 8 * (6144 + 2048) # 8 matrizes * rank 8 * (6144 + 2048)
extra_mb_dnet_r8 = (extra_params_dnet_r8 * 2) / (1024 * 1024)
final_mb_dnet_r8 = base_model_mb + extra_mb_dnet_r8

ppl_dnet_r8 = eval_ppl(model)
ganho_dnet_r8 = base_ppl_media - ppl_dnet_r8["Media"]
eficiencia_dnet_r8 = ganho_dnet_r8 / extra_mb_dnet_r8

print(f"DeltaNet Rank 8 | Tamanho: {final_mb_dnet_r8:6.1f} MB (+{extra_mb_dnet_r8:4.1f} MB) | PPL: {ppl_dnet_r8['Media']:5.2f} (Ganho: {ganho_dnet_r8:+5.2f}) | Eficiência: {eficiencia_dnet_r8:.4f} PPL/MB")

# -----------------------------------------------------------------------------
# 8. TESTE C: COMPARAÇÃO ESTRATÉGICA DE COMBINAÇÕES
# -----------------------------------------------------------------------------
print("\n" + "=" * 115)
print("TESTE C: COMPARAÇÃO DE COMBINAÇÕES MULTI-ESCALA")
print("=" * 115)

# Configurações para comparar:
# 1. v2 Baseline (Rank 0)
# 2. v2 + FFN Rank 8 (+3.0 MB)
# 3. v2 + DeltaNet Rank 8 (+1.0 MB)
# 4. v2 + DeltaNet Funcional 1% (+1.5 MB)
# 5. v2 + DeltaNet Funcional 2% (+3.0 MB)
# 6. v2 + DeltaNet Rank 8 + FFN Rank 8 (+4.0 MB)
# 7. v2 + DeltaNet Funcional 1% + FFN Rank 8 (+4.5 MB)

combinations_results = []

def reset_model():
    for layer_idx in target_dnet_layers:
        model.model.layers[layer_idx].linear_attn.in_proj_qkv.weight.data.copy_(dnet_targets_data[layer_idx]["W_base"])
    for i in range(24):
        model.model.layers[i].mlp.down_proj.weight.data.copy_(ffn_svd_data[i]["W_base"])

# 1. v2 Baseline
reset_model()
ppl_c1 = eval_ppl(model)
combinations_results.append({
    "config": "1. v2 Baseline",
    "mb": base_model_mb,
    "delta_mb": 0.0,
    "ppl_media": round(ppl_c1["Media"], 2),
    "ppl_codigo": round(ppl_c1["Codigo"], 2),
    "ppl_encic": round(ppl_c1["Enciclopedia"], 2),
    "ppl_ciencia": round(ppl_c1["Ciencia"], 2),
    "ppl_hist": round(ppl_c1["Historia"], 2),
    "ganho": 0.0
})

# 2. v2 + FFN Rank 8
reset_model()
for i in range(24):
    model.model.layers[i].mlp.down_proj.weight.data.copy_(ffn_svd_data[i]["W_base"] + ffn_svd_data[i]["C_rank8"])
ppl_c2 = eval_ppl(model)
combinations_results.append({
    "config": "2. v2 + FFN Rank 8",
    "mb": round(base_model_mb + 3.0, 1),
    "delta_mb": 3.0,
    "ppl_media": round(ppl_c2["Media"], 2),
    "ppl_codigo": round(ppl_c2["Codigo"], 2),
    "ppl_encic": round(ppl_c2["Enciclopedia"], 2),
    "ppl_ciencia": round(ppl_c2["Ciencia"], 2),
    "ppl_hist": round(ppl_c2["Historia"], 2),
    "ganho": round(base_ppl_media - ppl_c2["Media"], 2)
})

# 3. v2 + DeltaNet Rank 8
reset_model()
for layer_idx in target_dnet_layers:
    model.model.layers[layer_idx].linear_attn.in_proj_qkv.weight.data.copy_(dnet_targets_data[layer_idx]["W_base"] + dnet_targets_data[layer_idx]["C_rank8"])
ppl_c3 = eval_ppl(model)
combinations_results.append({
    "config": "3. v2 + DeltaNet Rank 8",
    "mb": round(base_model_mb + extra_mb_dnet_r8, 1),
    "delta_mb": round(extra_mb_dnet_r8, 1),
    "ppl_media": round(ppl_c3["Media"], 2),
    "ppl_codigo": round(ppl_c3["Codigo"], 2),
    "ppl_encic": round(ppl_c3["Enciclopedia"], 2),
    "ppl_ciencia": round(ppl_c3["Ciencia"], 2),
    "ppl_hist": round(ppl_c3["Historia"], 2),
    "ganho": round(base_ppl_media - ppl_c3["Media"], 2)
})

# 4. v2 + DeltaNet Funcional 1%
reset_model()
for layer_idx in target_dnet_layers:
    data = dnet_targets_data[layer_idx]
    w_rec = quantize_matrix_h5a_functional(data["W_orig"], act_rms=data["rms"], p_prot=0.010, seed=200+layer_idx)
    model.model.layers[layer_idx].linear_attn.in_proj_qkv.weight.data.copy_(w_rec)
ppl_c4 = eval_ppl(model)
combinations_results.append({
    "config": "4. v2 + DeltaNet Func 1%",
    "mb": round(base_model_mb + 1.5, 1),
    "delta_mb": 1.5,
    "ppl_media": round(ppl_c4["Media"], 2),
    "ppl_codigo": round(ppl_c4["Codigo"], 2),
    "ppl_encic": round(ppl_c4["Enciclopedia"], 2),
    "ppl_ciencia": round(ppl_c4["Ciencia"], 2),
    "ppl_hist": round(ppl_c4["Historia"], 2),
    "ganho": round(base_ppl_media - ppl_c4["Media"], 2)
})

# 5. v2 + DeltaNet Funcional 2%
reset_model()
for layer_idx in target_dnet_layers:
    data = dnet_targets_data[layer_idx]
    w_rec = quantize_matrix_h5a_functional(data["W_orig"], act_rms=data["rms"], p_prot=0.020, seed=200+layer_idx)
    model.model.layers[layer_idx].linear_attn.in_proj_qkv.weight.data.copy_(w_rec)
ppl_c5 = eval_ppl(model)
combinations_results.append({
    "config": "5. v2 + DeltaNet Func 2%",
    "mb": round(base_model_mb + 3.0, 1),
    "delta_mb": 3.0,
    "ppl_media": round(ppl_c5["Media"], 2),
    "ppl_codigo": round(ppl_c5["Codigo"], 2),
    "ppl_encic": round(ppl_c5["Enciclopedia"], 2),
    "ppl_ciencia": round(ppl_c5["Ciencia"], 2),
    "ppl_hist": round(ppl_c5["Historia"], 2),
    "ganho": round(base_ppl_media - ppl_c5["Media"], 2)
})

# 6. v2 + DeltaNet Rank 8 + FFN Rank 8
reset_model()
for layer_idx in target_dnet_layers:
    model.model.layers[layer_idx].linear_attn.in_proj_qkv.weight.data.copy_(dnet_targets_data[layer_idx]["W_base"] + dnet_targets_data[layer_idx]["C_rank8"])
for i in range(24):
    model.model.layers[i].mlp.down_proj.weight.data.copy_(ffn_svd_data[i]["W_base"] + ffn_svd_data[i]["C_rank8"])
ppl_c6 = eval_ppl(model)
combinations_results.append({
    "config": "6. v2 + DeltaNet R8 + FFN R8",
    "mb": round(base_model_mb + 4.0, 1),
    "delta_mb": 4.0,
    "ppl_media": round(ppl_c6["Media"], 2),
    "ppl_codigo": round(ppl_c6["Codigo"], 2),
    "ppl_encic": round(ppl_c6["Enciclopedia"], 2),
    "ppl_ciencia": round(ppl_c6["Ciencia"], 2),
    "ppl_hist": round(ppl_c6["Historia"], 2),
    "ganho": round(base_ppl_media - ppl_c6["Media"], 2)
})

# 7. v2 + DeltaNet Func 1% + FFN Rank 8
reset_model()
for layer_idx in target_dnet_layers:
    data = dnet_targets_data[layer_idx]
    w_rec = quantize_matrix_h5a_functional(data["W_orig"], act_rms=data["rms"], p_prot=0.010, seed=200+layer_idx)
    model.model.layers[layer_idx].linear_attn.in_proj_qkv.weight.data.copy_(w_rec)
for i in range(24):
    model.model.layers[i].mlp.down_proj.weight.data.copy_(ffn_svd_data[i]["W_base"] + ffn_svd_data[i]["C_rank8"])
ppl_c7 = eval_ppl(model)
combinations_results.append({
    "config": "7. v2 + DeltaNet F1% + FFN R8",
    "mb": round(base_model_mb + 4.5, 1),
    "delta_mb": 4.5,
    "ppl_media": round(ppl_c7["Media"], 2),
    "ppl_codigo": round(ppl_c7["Codigo"], 2),
    "ppl_encic": round(ppl_c7["Enciclopedia"], 2),
    "ppl_ciencia": round(ppl_c7["Ciencia"], 2),
    "ppl_hist": round(ppl_c7["Historia"], 2),
    "ganho": round(base_ppl_media - ppl_c7["Media"], 2)
})

# Teste de Geração na Melhor Configuração
best_config = min(combinations_results, key=lambda x: x["ppl_media"])
print(f"\nMelhor configuracao encontrada: {best_config['config']} com PPL {best_config['ppl_media']}")
gen_results = eval_gen(model)

print("\n" + "=" * 115)
print("TABELA COMPARATIVA CONSOLIDADA (COMBINAÇÕES MULTI-ESCALA)")
print("=" * 115)
print(f"{'Configuração':<32} | {'Tamanho':<10} | {'Delta MB':<9} | {'Código':<7} | {'Encic':<7} | {'Ciência':<8} | {'Hist':<7} | {'PPL Média':<10} | {'Ganho'}")
print("-" * 115)
for c in combinations_results:
    print(f"{c['config']:<32} | {c['mb']:6.1f} MB | +{c['delta_mb']:4.1f} MB  | {c['ppl_codigo']:<7.2f} | {c['ppl_encic']:<7.2f} | {c['ppl_ciencia']:<8.2f} | {c['ppl_hist']:<7.2f} | {c['ppl_media']:<10.2f} | {c['ganho']:<+5.2f}")

print("\n--- GERAÇÃO COMPARATIVA (MELHOR CONFIG) ---")
for k in ["Capital", "Fibonacci", "Newton"]:
    print(f"  [{k}]: {gen_results[k]['text']} ({gen_results[k]['tok_per_sec']:.2f} tok/s)")

output_all = {
    "sweep_func_levels": sweep_func_results,
    "dnet_rank8": {
        "final_mb": round(final_mb_dnet_r8, 2),
        "extra_mb": round(extra_mb_dnet_r8, 2),
        "ppl": ppl_dnet_r8,
        "ganho": round(ganho_dnet_r8, 3)
    },
    "combinations": combinations_results,
    "best_config": best_config,
    "generation": gen_results
}

with open("e:/agente/deltanet_functional_residual_results.json", "w") as f:
    json.dump(output_all, f, indent=2)

print("\nResultados salvos com sucesso em e:/agente/deltanet_functional_residual_results.json!")
