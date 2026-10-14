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

print("=" * 110)
print("SWEEP CONTROLADO DO RESIDUAL LOW-RANK (SVD): RANKS 0 -> 8 -> 16 -> 32 -> 64")
print("Objetivo: Descobrir o ponto de maior eficiência do residual (Ganho PPL / Custo MB)")
print("Baseline congelado: H5-A v2 Funcional Adaptativo (847.8 MB, PPL 9.98)")
print("=" * 110)

# -----------------------------------------------------------------------------
# 1. CARREGAR PERFIL DE ATIVAÇÕES
# -----------------------------------------------------------------------------
act_profile_path = "e:/agente/activation_rms_profile.pt"
if not os.path.exists(act_profile_path):
    raise FileNotFoundError("Perfil de ativacao nao encontrado!")

act_profiles = torch.load(act_profile_path)
print(f"Perfis de ativacao carregados: {len(act_profiles)} componentes.")

# -----------------------------------------------------------------------------
# 2. FUNÇÃO BASELINE H5-A v2 CONGELADA
# -----------------------------------------------------------------------------
def quantize_matrix_h5a_functional(W, act_rms=None, p_prot=0.02, seed=42):
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
# 3. PIPELINE DE AVALIAÇÃO CONGELADO
# -----------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

domains = {
    "Ciencia": "Quantum mechanics describes nature at the smallest scales of energy levels of atoms and subatomic particles.",
    "Codigo": "def binary_search(array, target): low = 0; high = len(array) - 1; while low <= high: mid = (low + high) // 2",
    "Historia": "The ancient library of Alexandria was once the largest repository of knowledge in the Hellenistic world.",
    "Enciclopedia": "The solar system formed approximately 4.6 billion years ago from the gravitational collapse of a molecular cloud."
}

def eval_ppl(m):
    scores = {}
    with torch.no_grad():
        for d, txt in domains.items():
            inputs = tokenizer(txt, return_tensors="pt")
            loss = m(**inputs, labels=inputs.input_ids).loss
            scores[d] = torch.exp(loss).item()
    scores["Media"] = sum(scores.values()) / len(scores)
    return scores

# -----------------------------------------------------------------------------
# 4. CARREGAR MODELO BF16
# -----------------------------------------------------------------------------
print("\n[1/4] Carregando modelo base Qwen3.5-2B em BF16 (model.safetensors)...")
t0_load = time.time()
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
model.eval()
print(f"-> Carregado em {time.time()-t0_load:.2f}s!")

attn_layers = [3, 7, 11, 15, 19, 23]
deltanet_layers = [i for i in range(24) if i not in attn_layers]

# -----------------------------------------------------------------------------
# 5. QUANTIZAÇÃO DO MODELO BASELINE H5-A v2
#    Armazenamos W_orig e W_h5a nas matrizes críticas (24 down_proj) para SVD
# -----------------------------------------------------------------------------
print("\n[2/4] Quantizando modelo completo com Baseline H5-A v2...")
t0_quant = time.time()

# Dicionário para guardar decomposição SVD do erro nas matrizes críticas Tier 1 (24 down_proj)
critical_svd_data = {}

# FFN (72 matrizes)
for i in range(24):
    l = model.model.layers[i]
    rms_gate = act_profiles.get(f"L{i}.mlp.gate_proj")
    rms_up   = act_profiles.get(f"L{i}.mlp.up_proj")
    rms_down = act_profiles.get(f"L{i}.mlp.down_proj")
    
    rg = quantize_matrix_h5a_functional(l.mlp.gate_proj.weight.data, act_rms=rms_gate, p_prot=0.0075, seed=42+i)
    ru = quantize_matrix_h5a_functional(l.mlp.up_proj.weight.data,   act_rms=rms_up,   p_prot=0.0075, seed=42+i)
    
    # Matriz crítica Tier 1: down_proj
    w_orig_down = l.mlp.down_proj.weight.data.clone().float()
    rd = quantize_matrix_h5a_functional(l.mlp.down_proj.weight.data, act_rms=rms_down, p_prot=0.020, seed=42+i)
    
    # Calcular e guardar o erro residual E = W_orig - W_h5a
    E_down = (w_orig_down - rd.float())
    # SVD low-rank até Rank 64: U [2048, 64], S [64], Vh [64, 6144]
    U, S, V = torch.svd_lowrank(E_down, q=64, niter=4)
    critical_svd_data[f"L{i}.mlp.down_proj"] = {
        "U": U, # [2048, 64]
        "S": S, # [64]
        "V": V, # [6144, 64]
        "W_base": rd.clone()
    }
    
    l.mlp.gate_proj.weight.data.copy_(rg)
    l.mlp.up_proj.weight.data.copy_(ru)
    l.mlp.down_proj.weight.data.copy_(rd)

# Self-Attention (24 matrizes)
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

# DeltaNet (54 matrizes)
for i in deltanet_layers:
    l = model.model.layers[i]
    rms_qkv = act_profiles.get(f"L{i}.linear_attn.in_proj_qkv")
    rms_z   = act_profiles.get(f"L{i}.linear_attn.in_proj_z")
    rms_out = act_profiles.get(f"L{i}.linear_attn.out_proj")
    
    rqkv = quantize_matrix_h5a_functional(l.linear_attn.in_proj_qkv.weight.data, act_rms=rms_qkv, p_prot=0.0, seed=200+i)
    rz   = quantize_matrix_h5a_functional(l.linear_attn.in_proj_z.weight.data,   act_rms=rms_z,   p_prot=0.0075, seed=200+i)
    rout = quantize_matrix_h5a_functional(l.linear_attn.out_proj.weight.data,    act_rms=rms_out, p_prot=0.010, seed=200+i)
    
    l.linear_attn.in_proj_qkv.weight.data.copy_(rqkv)
    l.linear_attn.in_proj_z.weight.data.copy_(rz)
    l.linear_attn.out_proj.weight.data.copy_(rout)

# Embeddings Híbridos (Top 32k BF16 + Resto 3.05)
w_emb = model.model.embed_tokens.weight.data
total_tokens = w_emb.shape[0]
for start in range(32768, total_tokens, 50000):
    end = min(start + 50000, total_tokens)
    rec, _, _, _, _ = quantize_matrix_h5a_batched(w_emb[start:end], num_ctrl=3, num_spec=3, p_crit=0.05, seed=500+start)
    w_emb[start:end].copy_(rec)

print(f"-> Modelo quantizado com H5-A v2 e SVD preparado em {time.time()-t0_quant:.1f}s.")

# -----------------------------------------------------------------------------
# 6. SWEEP CONTROLADO: RANKS 0 -> 8 -> 16 -> 32 -> 64
# -----------------------------------------------------------------------------
base_model_mb = 847.8
total_model_params = 2274069824 # 2.27B params
ranks_to_test = [0, 8, 16, 32, 64]
sweep_results = []

print("\n" + "=" * 115)
print("INICIANDO SWEEP DE RANKS RESIDUAIS")
print("=" * 115)

base_ppl_media = None

for r in ranks_to_test:
    t0_rank = time.time()
    
    # 1. Aplicar correção de Rank r nas 24 matrizes críticas
    for i in range(24):
        layer = model.model.layers[i]
        key = f"L{i}.mlp.down_proj"
        data = critical_svd_data[key]
        if r == 0:
            layer.mlp.down_proj.weight.data.copy_(data["W_base"])
        else:
            # C_r = U[:, :r] * S[:r] @ V[:, :r].T
            U_r = data["U"][:, :r] # [2048, r]
            S_r = data["S"][:r]    # [r]
            V_r = data["V"][:, :r] # [6144, r]
            C_r = (U_r * S_r) @ V_r.T # [2048, 6144]
            W_final = data["W_base"].float() + C_r
            layer.mlp.down_proj.weight.data.copy_(W_final.to(torch.bfloat16))
            
    # 2. Calcular parâmetros extras e tamanho em MB
    # 24 matrizes de tamanho 2048x6144: cada rank r adiciona r * (2048 + 6144) parâmetros em BF16
    extra_params = 24 * r * (2048 + 6144)
    extra_mb = (extra_params * 2) / (1024 * 1024) # BF16 = 2 bytes
    final_mb = base_model_mb + extra_mb
    
    # BPW efetivo global: (bytes_total * 8) / params_total
    total_bytes = final_mb * (1024 * 1024)
    effective_bpw = (total_bytes * 8) / total_model_params
    
    # 3. Avaliar PPL nos 4 domínios
    ppl = eval_ppl(model)
    ppl_media = ppl["Media"]
    
    if r == 0:
        base_ppl_media = ppl_media
        ganho_ppl = 0.0
        eficiencia = 0.0
    else:
        ganho_ppl = base_ppl_media - ppl_media # positivo = melhoria
        eficiencia = ganho_ppl / extra_mb if extra_mb > 0 else 0.0
        
    dt_eval = time.time() - t0_rank
    
    sweep_results.append({
        "rank": r,
        "final_mb": round(final_mb, 2),
        "extra_mb": round(extra_mb, 2),
        "effective_bpw": round(effective_bpw, 3),
        "ppl_codigo": round(ppl["Codigo"], 2),
        "ppl_enciclopedia": round(ppl["Enciclopedia"], 2),
        "ppl_ciencia": round(ppl["Ciencia"], 2),
        "ppl_historia": round(ppl["Historia"], 2),
        "ppl_media": round(ppl_media, 2),
        "ganho_ppl": round(ganho_ppl, 3),
        "eficiencia_ppl_mb": round(eficiencia, 4),
        "eval_time_s": round(dt_eval, 1)
    })
    
    print(f"Rank {r:<2} | Tamanho: {final_mb:6.1f} MB (+{extra_mb:4.1f} MB) | BPW: {effective_bpw:.3f} | PPL: {ppl_media:5.2f} (Ganho: {ganho_ppl:+5.2f}) | Eficiência: {eficiencia:.4f} PPL/MB")

# -----------------------------------------------------------------------------
# 7. TABELA CONSOLIDADA DOS RESULTADOS
# -----------------------------------------------------------------------------
print("\n" + "=" * 115)
print("TABELA CONSOLIDADA DO SWEEP LOW-RANK")
print("=" * 115)
header = f"{'Rank':<6} | {'Tamanho':<10} | {'Custo MB':<10} | {'BPW':<6} | {'Código':<8} | {'Encicl':<8} | {'Ciência':<8} | {'História':<8} | {'Média':<8} | {'Ganho PPL':<10} | {'PPL / MB'}"
print(header)
print("-" * 115)
for res in sweep_results:
    row = (f"Rank {res['rank']:<2} | {res['final_mb']:6.1f} MB | +{res['extra_mb']:4.1f} MB  | {res['effective_bpw']:.3f} | "
           f"{res['ppl_codigo']:<8.2f} | {res['ppl_enciclopedia']:<8.2f} | {res['ppl_ciencia']:<8.2f} | {res['ppl_historia']:<8.2f} | "
           f"{res['ppl_media']:<8.2f} | {res['ganho_ppl']:<+10.2f} | {res['eficiencia_ppl_mb']:<8.4f}")
    print(row)

# Salvar resultados em JSON
with open("e:/agente/lowrank_residual_sweep_results.json", "w") as f:
    json.dump(sweep_results, f, indent=2)

print("\nResultados completos salvos em e:/agente/lowrank_residual_sweep_results.json!")
