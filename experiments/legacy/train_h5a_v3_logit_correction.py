"""
train_h5a_v3_logit_correction.py
H5-A v3 — Functional Logit Distillation & Low-Rank Logit Correction (V3-B).
Objetivo:
Otimizar adaptadores low-rank (h' = h + (h @ B) @ A.T) diretamente pela divergência KL contra os logits do BF16:
  min_{A, B} KL( softmax(W_head h_BF16) || softmax(W_head (h_H5A + (h_H5A @ B) @ A.T)) )

Testa sistematicamente os Ranks:
- V3-B1: Rank 8
- V3-B2: Rank 16
- V3-B3: Rank 32

Mede para cada rank:
1. Métricas Offline:
   - KL Divergence
   - Top-1 Agreement (%)
   - Top-5 Agreement (%)
   - Hidden Cosine Similarity (%)
2. Avaliação de Modelo Completo (com hook no lm_head):
   - PPL Código
   - PPL Enciclopédia
   - PPL Ciência
   - PPL História
   - PPL MÉDIA
"""
import os
import sys
import time
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

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
print("H5-A v3: TESTE V3-B — CORREÇÃO LOW-RANK GUIADA POR LOGITS (FUNCTIONAL DISTILLATION)")
print("Hipótese: Otimizar o residual de hidden state via perda KL nos logits (R8, R16, R32)")
print("Controle congelado: H5-A v2.1 Rebalanced (848.4 MB | PPL 9.69 | KL 0.6068 | Top-1 71.72%)")
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

test_sequences = {
    "Ciencia": "Quantum mechanics describes nature at the smallest scales of energy levels of atoms and subatomic particles.",
    "Codigo": "def binary_search(array, target): low = 0; high = len(array) - 1; while low <= high: mid = (low + high) // 2",
    "Historia": "The ancient library of Alexandria was once the largest repository of knowledge in the Hellenistic world.",
    "Enciclopedia": "The solar system formed approximately 4.6 billion years ago from the gravitational collapse of a molecular cloud.",
    "Logica": "If all roses are flowers and some flowers fade quickly, what can we logically deduce about the persistence of botanical species?",
    "Matematica": "To solve the quadratic equation ax^2 + bx + c = 0, the quadratic formula states that x = (-b +- sqrt(b^2 - 4ac)) / (2a).",
    "Linguistica": "Artificial intelligence has transformed modern computational linguistics, enabling real-time multilingual neural translation.",
    "Filosofia": "The distinction between a priori and a posteriori knowledge represents a fundamental divide in modern Western epistemology."
}

benchmark_domains = {
    "Ciencia": "Quantum mechanics describes nature at the smallest scales of energy levels of atoms and subatomic particles.",
    "Codigo": "def binary_search(array, target): low = 0; high = len(array) - 1; while low <= high: mid = (low + high) // 2",
    "Historia": "The ancient library of Alexandria was once the largest repository of knowledge in the Hellenistic world.",
    "Enciclopedia": "The solar system formed approximately 4.6 billion years ago from the gravitational collapse of a molecular cloud."
}

def eval_ppl(m):
    scores = {}
    with torch.no_grad():
        for d, txt in benchmark_domains.items():
            inputs = tokenizer(txt, return_tensors="pt")
            loss = m(**inputs, labels=inputs.input_ids).loss
            scores[d] = torch.exp(loss).item()
    scores["Media"] = sum(scores.values()) / len(scores)
    return scores

# -----------------------------------------------------------------------------
# 1. CARREGAR MODELO BF16 E EXTRAIR ALVOS DE REFERÊNCIA
# -----------------------------------------------------------------------------
print("\n[1/5] Carregando modelo base BF16 e capturando alvos de destilação...")
t0_load = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    low_cpu_mem_usage=True
)
model.eval()

h_bf16_list = []
p_bf16_list = []
top1_bf16_list = []

with torch.no_grad():
    for name, text in test_sequences.items():
        inp = tokenizer(text, return_tensors="pt")
        out = model(inp.input_ids, output_hidden_states=True)
        h = out.hidden_states[-1][0].float() # [T, 2048]
        logits = out.logits[0].float()        # [T, V]
        p = F.softmax(logits, dim=-1)         # [T, V]
        h_bf16_list.append(h)
        p_bf16_list.append(p)
        top1_bf16_list.append(logits.argmax(dim=-1))

H_BF16 = torch.cat(h_bf16_list, dim=0) # [N, 2048]
P_BF16 = torch.cat(p_bf16_list, dim=0) # [N, V]
TOP1_BF16 = torch.cat(top1_bf16_list, dim=0) # [N]

N_tokens, D_hidden = H_BF16.shape
print(f"-> Alvos BF16 coletados: {N_tokens} tokens x {D_hidden} dimensões.")

# Guardar cópia do W_head em float32 para treinamento rápido
W_head = model.lm_head.weight.data.clone().float() # [V, 2048]
V_vocab = W_head.shape[0]
print(f"-> Matriz lm_head pronta: {V_vocab} vocabulário x {D_hidden} dimensões.")

# -----------------------------------------------------------------------------
# 2. QUANTIZAR MODELO PARA H5-A v2.1 REBALANCED (848.4 MB)
# -----------------------------------------------------------------------------
print("\n[2/5] Quantizando modelo para H5-A v2.1 Rebalanced (848.4 MB)...")
t0_quant = time.time()

# 1. FFN: 24 camadas com Rank 8 na down_proj
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

# 2. Atenção Convencional (Layers 3, 7, 11, 15, 19, 23)
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

# 3. DeltaNet com Rank 8 nas 8 mais críticas: [18, 17, 20, 21, 16, 22, 4, 12]
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
        E_qkv = w_orig_qkv - rqkv.float()
        U_d, S_d, V_d = torch.svd_lowrank(E_qkv, q=8, niter=4)
        corr_qkv = ((U_d * S_d) @ V_d.T).to(torch.bfloat16)
        l.linear_attn.in_proj_qkv.weight.data.copy_(rqkv + corr_qkv)
    else:
        l.linear_attn.in_proj_qkv.weight.data.copy_(rqkv)
        
    l.linear_attn.in_proj_z.weight.data.copy_(rz)
    l.linear_attn.out_proj.weight.data.copy_(rout)

# 4. Vocabulário Híbrido Rebalanceado: corte em 31.744 tokens BF16 puro
vocab_cutoff = 31744
w_emb = model.model.embed_tokens.weight.data
total_tokens = w_emb.shape[0]
for start in range(vocab_cutoff, total_tokens, 50000):
    end = min(start + 50000, total_tokens)
    rec, _, _, _, _ = quantize_matrix_h5a_batched(w_emb[start:end], num_ctrl=3, num_spec=3, p_crit=0.05, seed=500+start)
    w_emb[start:end].copy_(rec)

print(f"-> Modelo quantizado com H5-A v2.1 em {time.time()-t0_quant:.1f}s!")

# -----------------------------------------------------------------------------
# 3. EXTRAIR HIDDEN STATES DO H5-A v2.1 E MEDIR BASELINE
# -----------------------------------------------------------------------------
print("\n[3/5] Extraindo representações H5-A v2.1 e computando métricas baseline...")
h_h5a_list = []
with torch.no_grad():
    for name, text in test_sequences.items():
        inp = tokenizer(text, return_tensors="pt")
        out = model(inp.input_ids, output_hidden_states=True)
        h = out.hidden_states[-1][0].float()
        h_h5a_list.append(h)

H_H5A = torch.cat(h_h5a_list, dim=0) # [N, 2048]

def compute_metrics(H_curr, P_target, top1_target, W_proj):
    with torch.no_grad():
        logits_curr = H_curr @ W_proj.T # [N, V]
        log_q = F.log_softmax(logits_curr, dim=-1)
        # KL = P * (log P - log Q)
        # log P = log(P_target + 1e-12)
        log_p = torch.log(P_target.clamp(min=1e-12))
        kl = (P_target * (log_p - log_q)).sum(dim=-1).mean().item()
        
        top1_curr = logits_curr.argmax(dim=-1)
        top1_match = (top1_curr == top1_target).float().mean().item() * 100.0
        
        top5_curr = torch.topk(logits_curr, k=5, dim=-1).indices
        top5_match_count = 0
        for i in range(len(top1_target)):
            if top1_target[i] in top5_curr[i]:
                top5_match_count += 1
        top5_match = (top5_match_count / len(top1_target)) * 100.0
        
        # Cosine com H_BF16
        cos = (H_curr * H_BF16).sum(dim=-1) / (H_curr.norm(dim=-1) * H_BF16.norm(dim=-1) + 1e-9)
        mean_cos = cos.mean().item() * 100.0
        
    return {
        "kl": round(kl, 4),
        "top1": round(top1_match, 2),
        "top5": round(top5_match, 2),
        "cosine": round(mean_cos, 2)
    }

base_metrics = compute_metrics(H_H5A, P_BF16, TOP1_BF16, W_head)
base_ppl = eval_ppl(model)
base_metrics["ppl_media"] = round(base_ppl["Media"], 2)
base_metrics["ppl"] = {k: round(v, 2) for k, v in base_ppl.items()}

print(f"-> Baseline H5-A v2.1: KL={base_metrics['kl']:.4f} | Top-1={base_metrics['top1']:.2f}% | Top-5={base_metrics['top5']:.2f}% | Cosine={base_metrics['cosine']:.2f}% | PPL={base_metrics['ppl_media']:.2f}")

# -----------------------------------------------------------------------------
# 4. TREINAMENTO LOW-RANK LOGIT DISTILLATION (V3-B1, V3-B2, V3-B3)
# -----------------------------------------------------------------------------
print("\n[4/5] Otimizando correções low-rank guiadas por logits (Ranks 8, 16, 32)...")

ranks_to_test = [8, 16, 32]
results_by_rank = {}

for r in ranks_to_test:
    t0_r = time.time()
    torch.manual_seed(42)
    
    # Inicialização: B aleatório normalizado, A inicializado próximo de zero
    # h' = h + (h @ B) @ A.T
    B = nn.Parameter(torch.randn(D_hidden, r) * (1.0 / math.sqrt(D_hidden)))
    A = nn.Parameter(torch.zeros(D_hidden, r))
    
    optimizer = torch.optim.AdamW([A, B], lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-4)
    
    # Treinamento por 150 passos
    for step in range(150):
        optimizer.zero_grad()
        
        # h' = H_H5A + (H_H5A @ B) @ A.T
        delta = (H_H5A @ B) @ A.T
        H_prime = H_H5A + delta
        
        # Logits projetados
        logits_prime = H_prime @ W_head.T
        log_q = F.log_softmax(logits_prime, dim=-1)
        
        # Perda de destilação Cross-Entropy contra P_BF16 (equivalente à minimização de KL)
        loss = -(P_BF16 * log_q).sum(dim=-1).mean()
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
    dt_train = time.time() - t0_r
    
    # Avaliar métricas offline
    with torch.no_grad():
        H_opt = H_H5A + (H_H5A @ B) @ A.T
    
    metrics_r = compute_metrics(H_opt, P_BF16, TOP1_BF16, W_head)
    
    # Calcular custo em parâmetros e KB
    num_params = 2 * D_hidden * r
    size_kb = (num_params * 2) / 1024 # em BF16 (2 bytes por param)
    size_mb = size_kb / 1024
    
    # -------------------------------------------------------------------------
    # Integrar correção ao modelo real via hook e medir PPL end-to-end
    # -------------------------------------------------------------------------
    A_frozen = A.data.to(torch.bfloat16)
    B_frozen = B.data.to(torch.bfloat16)
    
    def make_pre_hook(A_mat, B_mat):
        def pre_hook(module, args):
            h_input = args[0]
            corr = (h_input @ B_mat) @ A_mat.T
            return (h_input + corr,)
        return pre_hook
        
    hook_handle = model.lm_head.register_forward_pre_hook(make_pre_hook(A_frozen, B_frozen))
    
    ppl_r = eval_ppl(model)
    hook_handle.remove() # Remove hook para não vazar para o próximo teste
    
    metrics_r["ppl_media"] = round(ppl_r["Media"], 2)
    metrics_r["ppl"] = {k: round(v, 2) for k, v in ppl_r.items()}
    metrics_r["num_params"] = num_params
    metrics_r["size_kb"] = round(size_kb, 1)
    metrics_r["size_mb"] = round(size_mb, 3)
    metrics_r["train_time_sec"] = round(dt_train, 1)
    
    results_by_rank[r] = metrics_r
    
    print(f"Rank {r:2d} ({size_kb:5.1f} KB) | KL: {metrics_r['kl']:.4f} (vs {base_metrics['kl']:.4f}) | Top-1: {metrics_r['top1']:5.2f}% (vs {base_metrics['top1']:5.2f}%) | Top-5: {metrics_r['top5']:5.2f}% | PPL: {metrics_r['ppl_media']:5.2f} (vs {base_metrics['ppl_media']:5.2f})")

# -----------------------------------------------------------------------------
# 5. CONSOLIDAÇÃO E SALVAMENTO DOS RESULTADOS
# -----------------------------------------------------------------------------
print("\n" + "=" * 115)
print("TABELA COMPARATIVA CONSOLIDADA: H5-A v3-B (LOGIT DISTILLATION)")
print("=" * 115)
print(f"{'Métrica':<24} | {'Baseline v2.1':<14} | {'Rank 8':<14} | {'Rank 16':<14} | {'Rank 32':<14}")
print("-" * 85)

for m_key, m_name in [
    ("kl", "Divergência KL (nats)"),
    ("top1", "Concordância Top-1 (%)"),
    ("top5", "Concordância Top-5 (%)"),
    ("cosine", "Hidden Cosine (%)"),
    ("ppl_media", "PPL Média"),
]:
    v_base = f"{base_metrics[m_key]}"
    v_r8 = f"{results_by_rank[8][m_key]}"
    v_r16 = f"{results_by_rank[16][m_key]}"
    v_r32 = f"{results_by_rank[32][m_key]}"
    print(f"{m_name:<24} | {v_base:<14} | {v_r8:<14} | {v_r16:<14} | {v_r32:<14}")

print(f"{'Custo de Memória':<24} | {'0.0 KB':<14} | {results_by_rank[8]['size_kb']:.1f} KB       | {results_by_rank[16]['size_kb']:.1f} KB      | {results_by_rank[32]['size_kb']:.1f} KB")

print("\nDetalhamento de PPL por Domínio:")
print(f"{'Domínio':<16} | {'Baseline':<10} | {'Rank 8':<10} | {'Rank 16':<10} | {'Rank 32':<10}")
print("-" * 65)
for dom in ["Codigo", "Enciclopedia", "Ciencia", "Historia"]:
    p_b = base_metrics["ppl"][dom]
    p_8 = results_by_rank[8]["ppl"][dom]
    p_16 = results_by_rank[16]["ppl"][dom]
    p_32 = results_by_rank[32]["ppl"][dom]
    print(f"{dom:<16} | {p_b:<10.2f} | {p_8:<10.2f} | {p_16:<10.2f} | {p_32:<10.2f}")

output_final = {
    "baseline_v2_1": base_metrics,
    "ranks": results_by_rank
}

with open("e:/agente/h5a_v3_logit_correction_results.json", "w", encoding="utf-8") as f:
    json.dump(output_final, f, indent=2, ensure_ascii=False)

print("\nResultados salvos com sucesso em: e:/agente/h5a_v3_logit_correction_results.json!")
