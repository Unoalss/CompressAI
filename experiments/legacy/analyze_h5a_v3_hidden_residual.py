"""
analyze_h5a_v3_hidden_residual.py
H5-A v3 — Functional Logit Distillation & Hidden Residual Diagnostic.
Fase 1: Diagnóstico Comparativo BF16 vs H5-A v2.1 Rebalanced.
Métricas:
- Hidden State: Cosine Similarity, MSE, Relative Frobenius Error, Layer-by-layer Divergence
- SVD do Residual Delta_h: Espectro de autovalores, energia acumulada por Rank (1, 2, 4, 8, 16, 32, 64, 128)
- Logits & Decisões: Divergência KL (P || Q), Top-1 Agreement, Top-5 Agreement, Correlação Delta_h vs KL
Nenhuma modificação permanente no modelo. Nenhum treino. Apenas diagnóstico empírico.
"""
import os
import sys
import time
import math
import json
import torch
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
print("H5-A v3: DIAGNÓSTICO DO RESIDUAL FUNCIONAL E ESPETRO DO HIDDEN STATE (BF16 vs v2.1)")
print("Objetivo: Investigar se Delta_h = h_BF16 - h_H5A possui estrutura de baixa dimensão (SVD)")
print("e medir o impacto na distribuição de probabilidade (KL) e decisões do modelo.")
print("=" * 115)

# Carregar perfil de ativações para reconstruir exatamente o H5-A v2.1 Rebalanced
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
print("\n[1/4] Carregando modelo base Qwen3.5-2B em BF16 puro...")
t0_load = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    low_cpu_mem_usage=True
)
model.eval()
print(f"-> Modelo BF16 carregado em {time.time()-t0_load:.2f}s!")

# Sequências de teste ricas cobrindo os 4 domínios canônicos + domínios analíticos adicionais
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

# -----------------------------------------------------------------------------
# FASE 1: EXECUÇÃO DO BASELINE BF16 (EXTRAÇÃO DE HIDDEN STATES E LOGITS)
# -----------------------------------------------------------------------------
print("\n[2/4] Executando inferência BF16 e capturando representações de referência...")
bf16_data = {}
total_tokens_count = 0

with torch.no_grad():
    for name, text in test_sequences.items():
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs.input_ids
        num_toks = input_ids.shape[1]
        total_tokens_count += num_toks
        
        # Forward pass com output_hidden_states=True
        out = model(input_ids, output_hidden_states=True)
        logits_bf16 = out.logits[0].float() # [T, V]
        
        # Hidden states em todas as camadas
        # out.hidden_states é uma tupla de 25 tensores: [0] = embeddings, [1..24] = camadas 1 a 24
        all_h_bf16 = [h[0].float() for h in out.hidden_states]
        final_h_bf16 = all_h_bf16[-1] # [T, 2048]
        
        # Calcular log-probabilities para cálculo estável de KL
        log_p_bf16 = F.log_softmax(logits_bf16, dim=-1)
        top1_bf16 = logits_bf16.argmax(dim=-1)
        
        bf16_data[name] = {
            "input_ids": input_ids,
            "num_tokens": num_toks,
            "final_h": final_h_bf16,
            "all_h": all_h_bf16,
            "log_p": log_p_bf16,
            "top1": top1_bf16,
            "loss": out.loss.item() if out.loss is not None else None
        }

print(f"-> BF16 concluído! Total de {total_tokens_count} tokens avaliados em {len(test_sequences)} domínios.")

# -----------------------------------------------------------------------------
# FASE 2: QUANTIZAÇÃO IN-PLACE PARA H5-A v2.1 REBALANCED (848.4 MB)
# -----------------------------------------------------------------------------
print("\n[3/4] Quantizando modelo para H5-A v2.1 Rebalanced (848.4 MB)...")
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
# FASE 3: EXECUÇÃO DO H5-A v2.1 E COMPARAÇÃO RESIDUAL
# -----------------------------------------------------------------------------
print("\n[4/4] Executando H5-A v2.1 e extraindo métricas funcionais comparativas...")

delta_h_list = []
token_cosines = []
token_mses = []
token_rel_errs = []
kl_divs = []
top1_matches = 0
top5_matches = 0
per_domain_stats = {}

# Camadas para acompanhar progressão de divergência: [0, 4, 8, 12, 16, 20, 24]
check_layers = [0, 4, 8, 12, 16, 20, 24]
layer_cosines = {l: [] for l in check_layers}
layer_rel_errs = {l: [] for l in check_layers}

with torch.no_grad():
    for name, text in test_sequences.items():
        data_bf16 = bf16_data[name]
        input_ids = data_bf16["input_ids"]
        T = data_bf16["num_tokens"]
        
        out_h5a = model(input_ids, output_hidden_states=True)
        logits_h5a = out_h5a.logits[0].float() # [T, V]
        log_q_h5a = F.log_softmax(logits_h5a, dim=-1)
        
        all_h_h5a = [h[0].float() for h in out_h5a.hidden_states]
        final_h_h5a = all_h_h5a[-1] # [T, 2048]
        final_h_bf16 = data_bf16["final_h"] # [T, 2048]
        
        # Delta_h para este domínio
        delta_h = final_h_bf16 - final_h_h5a # [T, 2048]
        delta_h_list.append(delta_h)
        
        # Cosine e Erros por token
        dot = (final_h_bf16 * final_h_h5a).sum(dim=-1)
        norm_bf16 = final_h_bf16.norm(dim=-1)
        norm_h5a = final_h_h5a.norm(dim=-1)
        cos = dot / (norm_bf16 * norm_h5a + 1e-9)
        token_cosines.extend(cos.tolist())
        
        diff_sq = (delta_h ** 2).mean(dim=-1)
        token_mses.extend(diff_sq.tolist())
        
        rel_err = delta_h.norm(dim=-1) / (norm_bf16 + 1e-9)
        token_rel_errs.extend(rel_err.tolist())
        
        # KL Divergence: P * (log P - log Q)
        log_p = data_bf16["log_p"] # [T, V]
        p = torch.exp(log_p)
        kl = (p * (log_p - log_q_h5a)).sum(dim=-1) # [T]
        kl_divs.extend(kl.tolist())
        
        # Top-1 e Top-5 Agreement
        top1_h5a = logits_h5a.argmax(dim=-1)
        top5_h5a = torch.topk(logits_h5a, k=5, dim=-1).indices
        
        top1_match = (data_bf16["top1"] == top1_h5a)
        top1_matches += top1_match.sum().item()
        
        for t in range(T):
            if data_bf16["top1"][t] in top5_h5a[t]:
                top5_matches += 1
                
        # Divergência camada por camada
        for l in check_layers:
            h_b = data_bf16["all_h"][l]
            h_h = all_h_h5a[l]
            c_l = (h_b * h_h).sum(dim=-1) / (h_b.norm(dim=-1) * h_h.norm(dim=-1) + 1e-9)
            r_l = (h_b - h_h).norm(dim=-1) / (h_b.norm(dim=-1) + 1e-9)
            layer_cosines[l].extend(c_l.tolist())
            layer_rel_errs[l].extend(r_l.tolist())
            
        per_domain_stats[name] = {
            "tokens": T,
            "mean_cosine": round(cos.mean().item(), 4),
            "min_cosine": round(cos.min().item(), 4),
            "mean_rel_err": round(rel_err.mean().item(), 4),
            "mean_kl": round(kl.mean().item(), 4),
            "top1_agreement": round((top1_match.float().mean().item()) * 100, 2)
        }

# -----------------------------------------------------------------------------
# FASE 4: DECOMPOSIÇÃO SVD DO DELTA_H E ANÁLISE DE ENERGIA ESPECTRAL
# -----------------------------------------------------------------------------
print("\n" + "=" * 115)
print("DECOMPOSIÇÃO SVD DO RESIDUAL DO HIDDEN STATE (Delta_h = h_BF16 - h_H5A)")
print("=" * 115)

Delta_H_all = torch.cat(delta_h_list, dim=0) # [N_total, 2048]
N_tokens, D_hidden = Delta_H_all.shape
print(f"Matriz de residual Delta_H construída: {N_tokens} tokens x {D_hidden} dimensões.")

# SVD completa do residual
U, S, Vh = torch.linalg.svd(Delta_H_all, full_matrices=False)
eigenvalues = S ** 2
total_energy = eigenvalues.sum().item()

target_ranks = [1, 2, 4, 8, 16, 32, 64, 128, 256]
energy_by_rank = {}

print(f"{'Rank':<8} | {'Singular Value (S_k)':<22} | {'Energia Acumulada (%)':<24} | {'Energia Marginal (%)'}")
print("-" * 75)

prev_cum = 0.0
for k in target_ranks:
    if k <= len(S):
        cum_energy = (eigenvalues[:k].sum().item() / total_energy) * 100.0
        marginal = cum_energy - prev_cum
        prev_cum = cum_energy
        s_val = S[k-1].item()
        energy_by_rank[k] = {
            "singular_val": round(s_val, 4),
            "cum_energy_pct": round(cum_energy, 2),
            "marginal_pct": round(marginal, 2)
        }
        print(f"Rank {k:<3} | S_{k:<2} = {s_val:<16.4f} | {cum_energy:6.2f}%                  | +{marginal:5.2f}%")

# Métricas globais consolidadas
mean_cosine = sum(token_cosines) / len(token_cosines)
min_cosine = min(token_cosines)
mean_mse = sum(token_mses) / len(token_mses)
mean_rel_err = sum(token_rel_errs) / len(token_rel_errs)
mean_kl = sum(kl_divs) / len(kl_divs)
top1_acc = (top1_matches / total_tokens_count) * 100.0
top5_acc = (top5_matches / total_tokens_count) * 100.0

# Correlação entre ||Delta_h|| e KL Divergence
t_norms = torch.tensor(token_rel_errs)
t_kls = torch.tensor(kl_divs)
mean_norm = t_norms.mean()
mean_k = t_kls.mean()
corr_num = ((t_norms - mean_norm) * (t_kls - mean_k)).sum()
corr_den = torch.sqrt(((t_norms - mean_norm) ** 2).sum() * ((t_kls - mean_k) ** 2).sum())
corr_norm_kl = (corr_num / (corr_den + 1e-9)).item()

print("\n" + "=" * 115)
print("SÍNTESE DO DIAGNÓSTICO FUNCIONAL: BF16 vs H5-A v2.1 REBALANCED")
print("=" * 115)
print(f"1. FIDELIDADE DE REPRESENTAÇÃO (Hidden States - Final Layer):")
print(f"   - Similaridade de Cosseno Média: {mean_cosine * 100:.2f}% (Mínimo: {min_cosine * 100:.2f}%)")
print(f"   - Erro Médio Quadrático (MSE):   {mean_mse:.6f}")
print(f"   - Erro Relativo de Frobenius:    {mean_rel_err * 100:.2f}%")
print(f"\n2. FIDELIDADE DE DECISÃO E DISTRIBUIÇÃO (Logits):")
print(f"   - Divergência KL Média (P || Q): {mean_kl:.4f} nats/token")
print(f"   - Concordância Top-1 (Argmax):   {top1_acc:.2f}%")
print(f"   - Concordância Top-5:            {top5_acc:.2f}%")
print(f"   - Correlação ||Delta_h|| x KL:   {corr_norm_kl:.4f}")

print("\n3. DIVERGÊNCIA AO LONGO DA PROFUNDIDADE DA REDE:")
print(f"{'Camada':<10} | {'Cosine Médio':<16} | {'Erro Relativo Médio'}")
print("-" * 50)
layer_summary = {}
for l in check_layers:
    l_name = "Embedding" if l == 0 else f"Camada {l}"
    cos_l = sum(layer_cosines[l]) / len(layer_cosines[l])
    err_l = sum(layer_rel_errs[l]) / len(layer_rel_errs[l])
    layer_summary[l_name] = {"cosine": round(cos_l, 4), "rel_err": round(err_l, 4)}
    print(f"{l_name:<10} | {cos_l * 100:6.2f}%          | {err_l * 100:6.2f}%")

print("\n4. ANÁLISE POR DOMÍNIO:")
for dom, st in per_domain_stats.items():
    print(f"   - {dom:<12}: Cosine {st['mean_cosine']*100:5.2f}% | KL {st['mean_kl']:6.4f} | Top-1 {st['top1_agreement']:5.2f}%")

# Salvar relatório consolidado
diagnostic_results = {
    "model_comparison": "Qwen3.5-2B BF16 vs H5-A v2.1 Rebalanced (848.4 MB)",
    "total_tokens_evaluated": total_tokens_count,
    "hidden_representation": {
        "mean_cosine": round(mean_cosine, 5),
        "min_cosine": round(min_cosine, 5),
        "mean_mse": round(mean_mse, 6),
        "mean_rel_err": round(mean_rel_err, 5)
    },
    "logit_distribution": {
        "mean_kl_divergence": round(mean_kl, 5),
        "top1_agreement_pct": round(top1_acc, 2),
        "top5_agreement_pct": round(top5_acc, 2),
        "correlation_delta_h_vs_kl": round(corr_norm_kl, 4)
    },
    "svd_spectrum_energy": energy_by_rank,
    "top10_singular_values": [round(s.item(), 4) for s in S[:10]],
    "layer_by_layer_divergence": layer_summary,
    "per_domain": per_domain_stats
}

with open("e:/agente/h5a_v3_hidden_residual_analysis.json", "w", encoding="utf-8") as f:
    json.dump(diagnostic_results, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 115)
print("Relatório diagnóstico salvo em: e:/agente/h5a_v3_hidden_residual_analysis.json")
print("=" * 115)
