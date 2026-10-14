"""
test_h5a_v3_c_generalization.py
H5-A v3-C: Generalização e Validação em Holdout da Logit Distillation.
Objetivo:
Avaliar se a correção low-rank (h' = h + (h @ B) @ A.T) treinada exclusivamente em um conjunto
de calibração (Código, Ciência, Computação, Enciclopédia) generaliza sem perda para um conjunto
Holdout NUNCA VISTO (História Antiga, Iluminismo, Álgebra Abstrata, Topologia, Filosofia, Lógica).

Ranks avaliados:
- Rank 8 (64 KB)
- Rank 16 (128 KB)
- Rank 32 (256 KB)

Métricas medidas em TREINO e HOLDOUT:
- Divergência KL (P_BF16 || Q_H5A)
- Concordância Top-1 (Argmax) %
- Concordância Top-5 %
- Hidden Cosine Similarity %
- PPL End-to-End no Benchmark Completo
- Gap de Generalização (Delta KL, Retenção de Top-1)
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
print("H5-A v3-C: EXPERIMENTO DE GENERALIZAÇÃO E VALIDAÇÃO EM HOLDOUT (LOGIT DISTILLATION)")
print("Hipótese a testar: A correção low-rank aprende uma distorção estrutural sistemática da quantização,")
print("generalizando com alta fidelidade para domínios e sequências nunca vistos durante o treinamento.")
print("Controle congelado: H5-A v2.1 Rebalanced (848.4 MB | PPL 9.69)")
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

# -----------------------------------------------------------------------------
# CONJUNTOS DE DADOS: TREINO (IN-DOMAIN) vs HOLDOUT (OUT-OF-DOMAIN & UNSEEN)
# -----------------------------------------------------------------------------
train_corpus = {
    "code_algorithms": "def quicksort(arr): if len(arr) <= 1: return arr; pivot = arr[len(arr) // 2]; left = [x for x in arr if x < pivot]; middle = [x for x in arr if x == pivot]; right = [x for x in arr if x > pivot]; return quicksort(left) + middle + quicksort(right)\nclass LRUCache: def __init__(self, capacity: int): self.capacity = capacity; self.cache = {}; self.order = []\n    def get(self, key: int) -> int: if key not in self.cache: return -1; self.order.remove(key); self.order.append(key); return self.cache[key]",
    "code_data_structures": "class TrieNode: def __init__(self): self.children = {}; self.is_end_of_word = False\nclass Trie: def __init__(self): self.root = TrieNode()\n    def insert(self, word: str) -> None: current = self.root; for char in word: if char not in current.children: current.children[char] = TrieNode(); current = current.children[char]; current.is_end_of_word = True\n    def search(self, word: str) -> bool: current = self.root; for char in word: if char not in current.children: return False; current = current.children[char]; return current.is_end_of_word",
    "science_quantum_mechanics": "In quantum mechanics, the wave function describes the quantum state of an isolated system of particles. The Schrodinger equation governs how this wave function evolves deterministically over time according to unitary operators. When an observable operator is measured, the system undergoes state reduction according to the Born rule, where the probability density is given by the squared magnitude of the complex amplitude. Entanglement between composite systems creates non-local correlations that violate Bell inequalities, demonstrating that no local hidden variable theory can reproduce all predictions of quantum physics.",
    "science_thermodynamics": "The second law of thermodynamics asserts that the total entropy of an isolated physical system can never decrease over time; it remains constant in reversible processes and increases in irreversible spontaneous processes. Statistical mechanics interprets entropy as the logarithm of the number of microstates corresponding to a given macrostate, scaled by Boltzmann constant S = k_B * ln(Omega). Thermodynamic equilibrium maximizes thermodynamic entropy, which corresponds to the state of maximum statistical probability and thermal homogenization across all available degrees of freedom.",
    "encyclopedia_cosmology": "The standard Lambda-CDM cosmological model describes an accelerating universe dominated by dark energy and cold dark matter, with baryonic matter constituting less than five percent of total energy density. The cosmic microwave background radiation provides a snapshot of the universe approximately 380,000 years after the Big Bang, during the recombination epoch when neutral hydrogen atoms formed and photons decoupled from plasma. Gravitational instability in primordial density perturbations caused matter to collapse hierarchically into filaments, galaxy clusters, and stellar systems.",
    "computer_architecture": "Modern superscalar microprocessors achieve instruction-level parallelism through dynamic branch prediction, speculative execution, and out-of-order execution pipelines. Register renaming maps architectural registers to physical register files to eliminate write-after-read and write-after-write hazards. Multi-level hierarchical memory caches, governed by cache coherence protocols like MESI, bridge the memory latency gap between high-speed arithmetic logic units and slower main dynamic random-access memory."
}

holdout_corpus = {
    "history_ancient_world": "The Peloponnesian War reshaped the ancient Greek geopolitical landscape, ending the golden age of Athenian democracy and hegemony. Following the Peace of Nicias, the disastrous Sicilian Expedition decimated Athenian naval superiority and resources. Sparta, bolstered by financial subsidies from the Persian Achaemenid Empire, ultimately defeated Athens at the Battle of Aegospotami, establishing a short-lived Spartan hegemony that was later shattered by Thebes under Epaminondas at the Battle of Leuctra.",
    "history_enlightenment": "The European Enlightenment was characterized by a profound reorientation in politics, science, and philosophical inquiry. Thinkers such as Voltaire, Montesquieu, and Jean-Jacques Rousseau challenged traditional ecclesiastical authority, promoting rational empiricism, individual liberties, and the separation of powers. The publication of the Encyclopedie by Denis Diderot and Jean le Rond d'Alembert aimed to democratize secular knowledge, laying intellectual groundwork for both the American and French Revolutions.",
    "math_abstract_algebra": "A Galois field is a finite algebraic field with q = p^n elements, where p is a prime characteristic and n is a positive integer. In modern commutative algebra, any prime ideal in a commutative Noetherian ring satisfies Krull principal ideal theorem, which bounds the height of the ideal by the number of minimal generators. Homological algebra formalizes these geometric intuitions through derived functors, Ext and Tor modules, and spectral sequences that compute cohomology groups of algebraic varieties.",
    "math_topology": "A topological space is compact if every open cover admits a finite subcover, a property that generalizes the Heine-Borel theorem for closed and bounded subsets of Euclidean space. The fundamental group of a pointed topological space classifies equivalence classes of continuous loops up to homotopy. Under the Seifert-van Kampen theorem, the fundamental group of a union of two path-connected open spaces can be expressed as the amalgamated free product of their individual fundamental groups over their intersection.",
    "philosophy_epistemology": "Immanuel Kant Critique of Pure Reason attempted a Copernican revolution in philosophy by reconciling rationalism with British empiricism. Kant distinguished between analytic judgments, where the predicate is contained in the concept of the subject, and synthetic judgments, which expand our conceptual understanding. He famously argued for the existence of synthetic a priori judgments, asserting that human perception is mediated through pure forms of sensible intuition—namely space and time—and twelve categories of the understanding.",
    "logic_formal_semantics": "Kurt Godel first incompleteness theorem proved that any consistent formal axiomatic system capable of expressing elementary Peano arithmetic contains true statements that cannot be proven within the system itself. The second incompleteness theorem established that such a formal system cannot demonstrate its own internal consistency unless it is inconsistent. Tarski undefinability theorem similarly established that arithmetical truth cannot be defined in the language of arithmetic itself, exposing intrinsic limitations in formal proof theory."
}

canonical_benchmark = {
    "Ciencia": "Quantum mechanics describes nature at the smallest scales of energy levels of atoms and subatomic particles.",
    "Codigo": "def binary_search(array, target): low = 0; high = len(array) - 1; while low <= high: mid = (low + high) // 2",
    "Historia": "The ancient library of Alexandria was once the largest repository of knowledge in the Hellenistic world.",
    "Enciclopedia": "The solar system formed approximately 4.6 billion years ago from the gravitational collapse of a molecular cloud."
}

def eval_ppl(m):
    scores = {}
    with torch.no_grad():
        for d, txt in canonical_benchmark.items():
            inputs = tokenizer(txt, return_tensors="pt")
            loss = m(**inputs, labels=inputs.input_ids).loss
            scores[d] = torch.exp(loss).item()
    scores["Media"] = sum(scores.values()) / len(scores)
    return scores

# -----------------------------------------------------------------------------
# 1. CARREGAR MODELO BF16 E EXTRAIR ALVOS PARA TREINO E HOLDOUT
# -----------------------------------------------------------------------------
print("\n[1/5] Carregando modelo base BF16 e extraindo representações de referência...")
t0_load = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    low_cpu_mem_usage=True
)
model.eval()

def collect_data(corpus_dict):
    h_list, p_list, top1_list = [], [], []
    with torch.no_grad():
        for name, text in corpus_dict.items():
            inp = tokenizer(text, return_tensors="pt")
            out = model(inp.input_ids, output_hidden_states=True)
            h = out.hidden_states[-1][0].float()
            logits = out.logits[0].float()
            p = F.softmax(logits, dim=-1)
            h_list.append(h)
            p_list.append(p)
            top1_list.append(logits.argmax(dim=-1))
    return torch.cat(h_list, dim=0), torch.cat(p_list, dim=0), torch.cat(top1_list, dim=0)

H_BF16_train, P_BF16_train, TOP1_BF16_train = collect_data(train_corpus)
H_BF16_hold,  P_BF16_hold,  TOP1_BF16_hold  = collect_data(holdout_corpus)

N_train = H_BF16_train.shape[0]
N_hold  = H_BF16_hold.shape[0]
D_hidden = H_BF16_train.shape[1]

print(f"-> Corpus de Treino:  {N_train} tokens em 6 sub-domínios (Código, Ciência, Computação, Enciclopédia)")
print(f"-> Corpus de Holdout: {N_hold} tokens em 6 sub-domínios (História, Álgebra, Topologia, Filosofia, Lógica)")

W_head = model.lm_head.weight.data.clone().float()

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
# 3. EXTRAIR HIDDEN STATES DO H5-A v2.1 PARA TREINO E HOLDOUT
# -----------------------------------------------------------------------------
print("\n[3/5] Extraindo representações H5-A v2.1 e computando métricas baseline...")

def collect_h5a_h(corpus_dict):
    h_list = []
    with torch.no_grad():
        for name, text in corpus_dict.items():
            inp = tokenizer(text, return_tensors="pt")
            out = model(inp.input_ids, output_hidden_states=True)
            h = out.hidden_states[-1][0].float()
            h_list.append(h)
    return torch.cat(h_list, dim=0)

H_H5A_train = collect_h5a_h(train_corpus)
H_H5A_hold  = collect_h5a_h(holdout_corpus)

def compute_metrics_batched(H_curr, H_target_ref, P_target, top1_target, W_proj, batch_size=256):
    total_tokens = H_curr.shape[0]
    total_kl = 0.0
    total_top1 = 0
    total_top5 = 0
    total_cos = 0.0
    
    with torch.no_grad():
        for i in range(0, total_tokens, batch_size):
            end_idx = min(i + batch_size, total_tokens)
            h_b = H_curr[i:end_idx]
            p_b = P_target[i:end_idx]
            top1_b = top1_target[i:end_idx]
            ref_b = H_target_ref[i:end_idx]
            B_curr = h_b.shape[0]
            
            logits_b = h_b @ W_proj.T # [B_curr, V]
            log_q = F.log_softmax(logits_b, dim=-1)
            log_p = torch.log(p_b.clamp(min=1e-12))
            kl_b = (p_b * (log_p - log_q)).sum(dim=-1).sum().item()
            total_kl += kl_b
            
            top1_pred = logits_b.argmax(dim=-1)
            total_top1 += (top1_pred == top1_b).sum().item()
            
            top5_pred = torch.topk(logits_b, k=5, dim=-1).indices
            for t_idx in range(B_curr):
                if top1_b[t_idx] in top5_pred[t_idx]:
                    total_top5 += 1
                    
            cos_b = (h_b * ref_b).sum(dim=-1) / (h_b.norm(dim=-1) * ref_b.norm(dim=-1) + 1e-9)
            total_cos += cos_b.sum().item()
            
    return {
        "kl": round(total_kl / total_tokens, 4),
        "top1": round((total_top1 / total_tokens) * 100.0, 2),
        "top5": round((total_top5 / total_tokens) * 100.0, 2),
        "cosine": round((total_cos / total_tokens) * 100.0, 2)
    }

base_train_metrics = compute_metrics_batched(H_H5A_train, H_BF16_train, P_BF16_train, TOP1_BF16_train, W_head)
base_hold_metrics  = compute_metrics_batched(H_H5A_hold,  H_BF16_hold,  P_BF16_hold,  TOP1_BF16_hold,  W_head)
base_ppl = eval_ppl(model)

print(f"-> Baseline TREINO:  KL={base_train_metrics['kl']:.4f} | Top-1={base_train_metrics['top1']:.2f}% | Top-5={base_train_metrics['top5']:.2f}% | Cos={base_train_metrics['cosine']:.2f}%")
print(f"-> Baseline HOLDOUT: KL={base_hold_metrics['kl']:.4f} | Top-1={base_hold_metrics['top1']:.2f}% | Top-5={base_hold_metrics['top5']:.2f}% | Cos={base_hold_metrics['cosine']:.2f}%")
print(f"-> Baseline PPL Geral: {base_ppl['Media']:.2f}")

# -----------------------------------------------------------------------------
# 4. TREINAMENTO EXCLUSIVO EM TREINO E AVALIAÇÃO EM HOLDOUT (R8, R16, R32)
# -----------------------------------------------------------------------------
print("\n[4/5] Otimizando correções low-rank EXCLUSIVAMENTE em Treino e testando em Holdout...")

ranks_to_test = [8, 16, 32]
v3_c_results = {}

for r in ranks_to_test:
    t0_r = time.time()
    torch.manual_seed(42)
    
    B_param = nn.Parameter(torch.randn(D_hidden, r) * (1.0 / math.sqrt(D_hidden)))
    A_param = nn.Parameter(torch.zeros(D_hidden, r))
    
    optimizer = torch.optim.AdamW([A_param, B_param], lr=2.5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120, eta_min=1e-4)
    
    # Treinamento em mini-batches de 256 tokens para manter consumo de RAM baixo e alta generalização
    batch_size_train = 256
    num_train_batches = math.ceil(N_train / batch_size_train)
    
    # 120 passos de otimização
    step = 0
    while step < 120:
        permutation = torch.randperm(N_train)
        for b_idx in range(num_train_batches):
            if step >= 120:
                break
            indices = permutation[b_idx * batch_size_train : min((b_idx + 1) * batch_size_train, N_train)]
            h_b = H_H5A_train[indices]
            p_b = P_BF16_train[indices]
            
            optimizer.zero_grad()
            h_corr = h_b + (h_b @ B_param) @ A_param.T
            logits_corr = h_corr @ W_head.T
            log_q = F.log_softmax(logits_corr, dim=-1)
            
            loss = -(p_b * log_q).sum(dim=-1).mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            
    dt_train = time.time() - t0_r
    
    # -------------------------------------------------------------------------
    # Avaliação em TREINO
    # -------------------------------------------------------------------------
    with torch.no_grad():
        H_opt_train = H_H5A_train + (H_H5A_train @ B_param) @ A_param.T
        metrics_train = compute_metrics_batched(H_opt_train, H_BF16_train, P_BF16_train, TOP1_BF16_train, W_head)
        
    # -------------------------------------------------------------------------
    # Avaliação em HOLDOUT (FROZEN A e B - NUNCA VISTO)
    # -------------------------------------------------------------------------
    with torch.no_grad():
        H_opt_hold = H_H5A_hold + (H_H5A_hold @ B_param) @ A_param.T
        metrics_hold = compute_metrics_batched(H_opt_hold, H_BF16_hold, P_BF16_hold, TOP1_BF16_hold, W_head)
        
    # Gap de generalização
    delta_kl = metrics_hold["kl"] - metrics_train["kl"]
    top1_retention = (metrics_hold["top1"] / metrics_train["top1"]) * 100.0
    
    # -------------------------------------------------------------------------
    # PPL no Benchmark Geral via Forward Pre-Hook
    # -------------------------------------------------------------------------
    A_frozen = A_param.data.to(torch.bfloat16)
    B_frozen = B_param.data.to(torch.bfloat16)
    
    def make_pre_hook(A_mat, B_mat):
        def pre_hook(module, args):
            h_input = args[0]
            corr = (h_input @ B_mat) @ A_mat.T
            return (h_input + corr,)
        return pre_hook
        
    hook_handle = model.lm_head.register_forward_pre_hook(make_pre_hook(A_frozen, B_frozen))
    ppl_r = eval_ppl(model)
    hook_handle.remove()
    
    num_params = 2 * D_hidden * r
    size_kb = (num_params * 2) / 1024
    
    v3_c_results[r] = {
        "rank": r,
        "size_kb": size_kb,
        "size_mb": round(size_kb / 1024, 3),
        "train_time_sec": round(dt_train, 1),
        "train_metrics": metrics_train,
        "holdout_metrics": metrics_hold,
        "generalization_gap": {
            "delta_kl": round(delta_kl, 4),
            "top1_retention_pct": round(top1_retention, 2)
        },
        "ppl_media": round(ppl_r["Media"], 2),
        "ppl": {k: round(v, 2) for k, v in ppl_r.items()}
    }
    
    print(f"\n--- RESULTADOS RANK {r:2d} ({size_kb:.1f} KB) ---")
    print(f"  TREINO:  KL: {metrics_train['kl']:.4f} (Base: {base_train_metrics['kl']:.4f}) | Top-1: {metrics_train['top1']:.2f}% | Top-5: {metrics_train['top5']:.2f}% | Cos: {metrics_train['cosine']:.2f}%")
    print(f"  HOLDOUT: KL: {metrics_hold['kl']:.4f} (Base: {base_hold_metrics['kl']:.4f}) | Top-1: {metrics_hold['top1']:.2f}% | Top-5: {metrics_hold['top5']:.2f}% | Cos: {metrics_hold['cosine']:.2f}%")
    print(f"  GAP GEN: Delta KL = {delta_kl:+.4f} | Retenção Top-1 = {top1_retention:.2f}% | PPL Benchmark = {ppl_r['Media']:.2f}")

# -----------------------------------------------------------------------------
# 5. CONSOLIDAÇÃO E SALVAMENTO
# -----------------------------------------------------------------------------
print("\n" + "=" * 115)
print("TABELA COMPARATIVA CONSOLIDADA: H5-A v3-C GENERALIZAÇÃO (TREINO vs HOLDOUT)")
print("=" * 115)
print(f"{'Métrica':<24} | {'Baseline':<12} | {'Rank 8 (64 KB)':<18} | {'Rank 16 (128 KB)':<18} | {'Rank 32 (256 KB)':<18}")
print("-" * 100)

print(f"{'KL Treino (nats)':<24} | {base_train_metrics['kl']:<12.4f} | {v3_c_results[8]['train_metrics']['kl']:<18.4f} | {v3_c_results[16]['train_metrics']['kl']:<18.4f} | {v3_c_results[32]['train_metrics']['kl']:<18.4f}")
print(f"{'KL Holdout (nats)':<24} | {base_hold_metrics['kl']:<12.4f} | {v3_c_results[8]['holdout_metrics']['kl']:<18.4f} | {v3_c_results[16]['holdout_metrics']['kl']:<18.4f} | {v3_c_results[32]['holdout_metrics']['kl']:<18.4f}")
print(f"{'Top-1 Treino (%)':<24} | {base_train_metrics['top1']:<12.2f} | {v3_c_results[8]['train_metrics']['top1']:<18.2f} | {v3_c_results[16]['train_metrics']['top1']:<18.2f} | {v3_c_results[32]['train_metrics']['top1']:<18.2f}")
print(f"{'Top-1 Holdout (%)':<24} | {base_hold_metrics['top1']:<12.2f} | {v3_c_results[8]['holdout_metrics']['top1']:<18.2f} | {v3_c_results[16]['holdout_metrics']['top1']:<18.2f} | {v3_c_results[32]['holdout_metrics']['top1']:<18.2f}")
print(f"{'Top-5 Holdout (%)':<24} | {base_hold_metrics['top5']:<12.2f} | {v3_c_results[8]['holdout_metrics']['top5']:<18.2f} | {v3_c_results[16]['holdout_metrics']['top5']:<18.2f} | {v3_c_results[32]['holdout_metrics']['top5']:<18.2f}")
print(f"{'Hidden Cosine Holdout (%)':<24} | {base_hold_metrics['cosine']:<12.2f} | {v3_c_results[8]['holdout_metrics']['cosine']:<18.2f} | {v3_c_results[16]['holdout_metrics']['cosine']:<18.2f} | {v3_c_results[32]['holdout_metrics']['cosine']:<18.2f}")
print(f"{'PPL Benchmark Geral':<24} | {base_ppl['Media']:<12.2f} | {v3_c_results[8]['ppl_media']:<18.2f} | {v3_c_results[16]['ppl_media']:<18.2f} | {v3_c_results[32]['ppl_media']:<18.2f}")

print("\nDetalhamento de PPL no Benchmark Geral:")
print(f"{'Domínio':<16} | {'Baseline':<10} | {'Rank 8':<10} | {'Rank 16':<10} | {'Rank 32':<10}")
print("-" * 65)
for dom in ["Codigo", "Enciclopedia", "Ciencia", "Historia"]:
    p_b = base_ppl[dom]
    p_8 = v3_c_results[8]["ppl"][dom]
    p_16 = v3_c_results[16]["ppl"][dom]
    p_32 = v3_c_results[32]["ppl"][dom]
    print(f"{dom:<16} | {p_b:<10.2f} | {p_8:<10.2f} | {p_16:<10.2f} | {p_32:<10.2f}")

output_full = {
    "baseline": {
        "train": base_train_metrics,
        "holdout": base_hold_metrics,
        "ppl": {k: round(v, 2) for k, v in base_ppl.items()}
    },
    "ranks": v3_c_results
}

with open("e:/agente/h5a_v3_c_generalization_results.json", "w", encoding="utf-8") as f:
    json.dump(output_full, f, indent=2, ensure_ascii=False)

print("\nResultados salvos com sucesso em: e:/agente/h5a_v3_c_generalization_results.json!")
