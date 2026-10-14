"""
run_qwen_h5a_v2_2_deltanet_rank16.py
H5-A v2.2 Diagnostic: DeltaNet in_proj_qkv Rank 8 -> Rank 16 nas 8 matrizes críticas.
Mantém todo o restante idêntico ao H5-A v2.1 Rebalanced:
- 24 FFN down_proj com Rank 8
- Vocabulário híbrido com 31.744 tokens em BF16 puro
- 8 DeltaNet in_proj_qkv (L18, L17, L20, L21, L16, L22, L4, L12) com Rank 16 (SVD q=16)
Mede PPL em todos os 4 domínios e avalia geração qualitativa.
"""
import os
import sys
import time
import math
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

print("=" * 115)
print("H5-A v2.2 DIAGNÓSTICO: DELTANET in_proj_qkv RANK 16 NAS 8 MATRIZES CRÍTICAS")
print("Controles congelados:")
print("  - H5-A v2 Baseline:         847.8 MB | PPL 9.98")
print("  - H5-A v2.1 Rebalanced:     848.4 MB | PPL 9.69 (DeltaNet Rank 8 + FFN Rank 8)")
print("Hipótese a testar: DeltaNet Rank 16 traz ganho relevante ou satura?")
print("=" * 115)

act_profile_path = "e:/agente/activation_rms_profile.pt"
act_profiles = torch.load(act_profile_path)

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

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    low_cpu_mem_usage=True
)

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
        toks = out.shape[1] - inp.input_ids.shape[1]
        results[name] = {
            "text": tokenizer.decode(out[0], skip_special_tokens=True).strip(),
            "tok_per_sec": toks / dt if dt > 0 else 0
        }
    return results

print("\nQuantizando modelo com H5-A v2.2 (DeltaNet Rank 16)...")
t0_quant = time.time()

# 1. FFN: 24 camadas com Rank 8 na down_proj (idêntico a v2.1)
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
    corr_down = ((U_f * S_f) @ V_f.T).to(torch.bfloat16)
    
    l.mlp.gate_proj.weight.data.copy_(rg)
    l.mlp.up_proj.weight.data.copy_(ru)
    l.mlp.down_proj.weight.data.copy_(rd + corr_down)

# 2. Atenção Convencional (idêntico a v2.1)
attn_layers = [3, 7, 11, 15, 19, 23]
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

# 3. DeltaNet com RANK 16 nas 8 mais críticas: L18, L17, L20, L21, L16, L22, L4, L12
target_dnet_layers = [18, 17, 20, 21, 16, 22, 4, 12]
deltanet_layers = [i for i in range(24) if i not in attn_layers]
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
        # AQUI: SVD Rank 16 (q=16)
        E_qkv = w_orig_qkv - rqkv.float()
        U_d, S_d, V_d = torch.svd_lowrank(E_qkv, q=16, niter=4)
        corr_qkv = ((U_d * S_d) @ V_d.T).to(torch.bfloat16)
        l.linear_attn.in_proj_qkv.weight.data.copy_(rqkv + corr_qkv)
    else:
        l.linear_attn.in_proj_qkv.weight.data.copy_(rqkv)
        
    l.linear_attn.in_proj_z.weight.data.copy_(rz)
    l.linear_attn.out_proj.weight.data.copy_(rout)

# 4. Vocabulário Híbrido: corte em 31.744 tokens BF16 puro (idêntico a v2.1)
vocab_cutoff = 31744
w_emb = model.model.embed_tokens.weight.data
total_tokens = w_emb.shape[0]
for start in range(vocab_cutoff, total_tokens, 50000):
    end = min(start + 50000, total_tokens)
    rec, _, _, _, _ = quantize_matrix_h5a_batched(w_emb[start:end], num_ctrl=3, num_spec=3, p_crit=0.05, seed=500+start)
    w_emb[start:end].copy_(rec)

print(f"Modelo quantizado em {time.time()-t0_quant:.1f}s.")

# Cálculo do tamanho exato em MB:
# v2.1 Rebalanced: 848.36 MB
# + Rank 16 vs Rank 8 na DeltaNet: 8 matrizes * (16 - 8) * (6144 + 2048) * 2 bytes = 1,048,576 bytes = 1.00 MB
# Total = 848.36 + 1.00 = 849.36 MB ≈ 849.4 MB
base_v2_1_mb = 848.36
extra_mb_r16 = (8 * (16 - 8) * (6144 + 2048) * 2) / (1024 * 1024)
final_mb = base_v2_1_mb + extra_mb_r16
print(f"Tamanho total final: {final_mb:.2f} MB (Delta vs v2.1: +{extra_mb_r16:.2f} MB)")

ppl = eval_ppl(model)
ppl_media = ppl['Media']
ganho_vs_v2_1 = 9.69 - ppl_media
eficiencia_r16 = ganho_vs_v2_1 / extra_mb_r16 if extra_mb_r16 > 0 else 0.0

print("\n" + "=" * 80)
print(f"PPL Médio H5-A v2.2 (DeltaNet R16): {ppl_media:.2f} (vs 9.69 em v2.1)")
print(f"  Ganho vs v2.1: {ganho_vs_v2_1:+.2f} PPL | Eficiência: {eficiencia_r16:.4f} PPL/MB")
print(f"  Código:       {ppl['Codigo']:.2f}")
print(f"  Enciclopédia: {ppl['Enciclopedia']:.2f}")
print(f"  Ciência:      {ppl['Ciencia']:.2f}")
print(f"  História:     {ppl['Historia']:.2f}")
print("=" * 80)

gen = eval_gen(model)
print("\n--- GERAÇÃO QUALITATIVA ---")
for k in ["Capital", "Fibonacci", "Newton"]:
    print(f"  [{k}]: {gen[k]['text']}")

output = {
    "model": "H5-A v2.2 (DeltaNet Rank 16)",
    "final_mb": round(final_mb, 2),
    "delta_vs_v2_1_mb": round(extra_mb_r16, 2),
    "ppl": ppl,
    "ganho_vs_v2_1": round(ganho_vs_v2_1, 3),
    "eficiencia_vs_v2_1": round(eficiencia_r16, 4),
    "generation": gen
}

with open("e:/agente/h5a_v2_2_deltanet_rank16_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("\nResultados salvos com sucesso em e:/agente/h5a_v2_2_deltanet_rank16_results.json!")
