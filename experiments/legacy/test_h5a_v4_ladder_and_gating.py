"""
test_h5a_v4_ladder_and_gating.py
H5-A v4: Teste da Arquitetura Ladder (Micro-Adaptadores Inter-Camadas) e Adaptive Gating.

Hipóteses testadas:
1. Ladder Duplo (Camada 12 + Camada 24, Rank 8 em cada):
   Interromper o acúmulo de erro na metade do modelo (Camada 12) permite que as camadas 13-24
   operem em um espaço de representação muito mais próximo do BF16 original, superando a correção
   isolada no final (V3-B Rank 16) com EXATAMENTE o mesmo orçamento de memória (128 KB).
2. Adaptive Gating (w_g):
   Modulação não-linear (1 + tanh(h @ w_g)) impede a sobre-correção e melhora a generalização
   para o conjunto Holdout (domínios nunca vistos).

Configurações avaliadas:
1. Baseline: H5-A v2.1 Rebalanced (848.40 MB | sem adaptadores)
2. Controle V3-B: Rank 16 apenas na Camada 24 (128 KB | 848.525 MB)
3. Controle V3-Gated: Rank 16 na Camada 24 com Adaptive Gating (132 KB)
4. H5-A v4-Ladder: Rank 8 na Camada 12 + Rank 8 na Camada 24 (128 KB | 848.525 MB)
5. H5-A v4-Ladder-Gated: Rank 8-Gated na Camada 12 + Rank 8-Gated na Camada 24 (136 KB)
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

MODEL_DIR = "E:/agente/compress/modelo"
device = "cpu"
torch.set_num_threads(os.cpu_count() or 8)

print("=" * 115, flush=True)
print("H5-A v4: EXPERIMENTO OUSADO — LADDER DE MICRO-ADAPTADORES E ADAPTIVE GATING", flush=True)
print("Objetivo: Testar se recalibrar a Camada 12 (R8) + Camada 24 (R8) destrói o drift cumulativo", flush=True)
print("e se o Adaptive Gating elimina o gap de generalização no Holdout mantendo o modelo < 850 MB.", flush=True)
print("=" * 115, flush=True)

act_profile_path = "e:/agente/compress/activation_rms_profile.pt"
if not os.path.exists(act_profile_path):
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

# Corpora de Treino e Holdout rigorosos (mesmos de v3-C)
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
# 1. CARREGAR MODELO BF16 E EXTRAIR REPRESENTAÇÕES DE REFERÊNCIA
# -----------------------------------------------------------------------------
targets_cache_path = "e:/agente/compress/h5a_v4_targets_cache.pt"

print("\n[1/6] Carregando modelo base BF16 e capturando alvos nas Camadas 12 e 24...", flush=True)
t0_load = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    low_cpu_mem_usage=True
)
model.eval()

W_head = model.lm_head.weight.data.clone().float()

if os.path.exists(targets_cache_path):
    print("-> Alvos BF16 encontrados no cache! Carregando instantaneamente...", flush=True)
    targets_data = torch.load(targets_cache_path)
    H12_bf16_train = targets_data["H12_bf16_train"]
    H24_bf16_train = targets_data["H24_bf16_train"]
    P_bf16_train   = targets_data["P_bf16_train"]
    Top1_bf16_train = targets_data["Top1_bf16_train"]
    H12_bf16_hold  = targets_data["H12_bf16_hold"]
    H24_bf16_hold  = targets_data["H24_bf16_hold"]
    P_bf16_hold    = targets_data["P_bf16_hold"]
    Top1_bf16_hold = targets_data["Top1_bf16_hold"]
else:
    def collect_layer_targets(corpus_dict):
        h12_list = []
        h24_list = []
        p_list   = []
        top1_list = []
        with torch.no_grad():
            for name, text in corpus_dict.items():
                inp = tokenizer(text, return_tensors="pt")
                out = model(inp.input_ids, output_hidden_states=True)
                # hidden_states[12] é a entrada da camada 12 (saída da camada 11)
                # hidden_states[-1] é a saída da camada 24 (entrada do lm_head)
                h12 = out.hidden_states[12][0].float()
                h24 = out.hidden_states[-1][0].float()
                logits = out.logits[0].float()
                p = F.softmax(logits, dim=-1)
                
                h12_list.append(h12)
                h24_list.append(h24)
                p_list.append(p)
                top1_list.append(logits.argmax(dim=-1))
                
        H12 = torch.cat(h12_list, dim=0)
        H24 = torch.cat(h24_list, dim=0)
        P   = torch.cat(p_list, dim=0)
        T1  = torch.cat(top1_list, dim=0)
        return H12, H24, P, T1

    H12_bf16_train, H24_bf16_train, P_bf16_train, Top1_bf16_train = collect_layer_targets(train_corpus)
    H12_bf16_hold,  H24_bf16_hold,  P_bf16_hold,  Top1_bf16_hold  = collect_layer_targets(holdout_corpus)

    torch.save({
        "H12_bf16_train": H12_bf16_train,
        "H24_bf16_train": H24_bf16_train,
        "P_bf16_train": P_bf16_train,
        "Top1_bf16_train": Top1_bf16_train,
        "H12_bf16_hold": H12_bf16_hold,
        "H24_bf16_hold": H24_bf16_hold,
        "P_bf16_hold": P_bf16_hold,
        "Top1_bf16_hold": Top1_bf16_hold,
    }, targets_cache_path)
    print("-> Alvos BF16 cacheados em disco com sucesso!", flush=True)

N_train, D_hidden = H12_bf16_train.shape
N_hold, _         = H12_bf16_hold.shape
print(f"-> Alvos prontos: Treino ({N_train} tokens) | Holdout ({N_hold} tokens) | Dimensão={D_hidden}", flush=True)

# -----------------------------------------------------------------------------
# 2. QUANTIZAR MODELO PARA H5-A v2.1 REBALANCED (848.4 MB)
# -----------------------------------------------------------------------------
quant_weights_cache = "e:/agente/compress/h5a_v2_1_model_weights.pt"

print("\n[2/6] Quantizando modelo para H5-A v2.1 Rebalanced (848.4 MB)...", flush=True)
t0_quant = time.time()

if os.path.exists(quant_weights_cache):
    print("-> Pesos quantizados H5-A v2.1 encontrados no cache! Carregando...", flush=True)
    model.load_state_dict(torch.load(quant_weights_cache))
    print(f"-> Modelo quantizado carregado em {time.time()-t0_quant:.1f}s!", flush=True)
else:
    # FFN
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

    # Attention
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

    # DeltaNet
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

    # Embedding
    vocab_cutoff = 31744
    w_emb = model.model.embed_tokens.weight.data
    total_tokens = w_emb.shape[0]
    for start in range(vocab_cutoff, total_tokens, 50000):
        end = min(start + 50000, total_tokens)
        rec, _, _, _, _ = quantize_matrix_h5a_batched(w_emb[start:end], num_ctrl=3, num_spec=3, p_crit=0.05, seed=500+start)
        w_emb[start:end].copy_(rec)

    torch.save(model.state_dict(), quant_weights_cache)
    print(f"-> Modelo quantizado e salvo no cache em {time.time()-t0_quant:.1f}s!", flush=True)

# -----------------------------------------------------------------------------
# 3. EXTRAÇÃO DAS REPRESENTAÇÕES H5-A SEM ADAPTADORES (BASELINE)
# -----------------------------------------------------------------------------
print("\n[3/6] Extraindo representações H5-A descalibradas e medindo Baseline...", flush=True)

h5a_reps_cache = "e:/agente/compress/h5a_v4_baseline_reps_cache.pt"

if os.path.exists(h5a_reps_cache):
    print("-> Representações H5-A encontradas no cache! Carregando...", flush=True)
    c_reps = torch.load(h5a_reps_cache)
    H12_h5a_train = c_reps["H12_h5a_train"]
    H24_h5a_train = c_reps["H24_h5a_train"]
    H12_h5a_hold  = c_reps["H12_h5a_hold"]
    H24_h5a_hold  = c_reps["H24_h5a_hold"]
else:
    def collect_h5a_representations(corpus_dict):
        h12_list, h24_list = [], []
        with torch.no_grad():
            for name, text in corpus_dict.items():
                inp = tokenizer(text, return_tensors="pt")
                out = model(inp.input_ids, output_hidden_states=True)
                h12_list.append(out.hidden_states[12][0].float())
                h24_list.append(out.hidden_states[-1][0].float())
        return torch.cat(h12_list, dim=0), torch.cat(h24_list, dim=0)

    H12_h5a_train, H24_h5a_train = collect_h5a_representations(train_corpus)
    H12_h5a_hold,  H24_h5a_hold  = collect_h5a_representations(holdout_corpus)

    torch.save({
        "H12_h5a_train": H12_h5a_train,
        "H24_h5a_train": H24_h5a_train,
        "H12_h5a_hold": H12_h5a_hold,
        "H24_h5a_hold": H24_h5a_hold,
    }, h5a_reps_cache)
    print("-> Representações H5-A salvas no cache!", flush=True)

# Diagnóstico de Drift na Camada 12
cos_l12_train = ((H12_h5a_train * H12_bf16_train).sum(dim=-1) / (H12_h5a_train.norm(dim=-1) * H12_bf16_train.norm(dim=-1) + 1e-9)).mean().item() * 100.0
rel_err_l12_train = ((H12_h5a_train - H12_bf16_train).norm(dim=-1) / (H12_bf16_train.norm(dim=-1) + 1e-9)).mean().item() * 100.0
print(f"-> Drift Camada 12 Inicial (Treino): Cosine={cos_l12_train:.2f}% | Erro Relativo={rel_err_l12_train:.2f}%", flush=True)

def eval_metrics(H_curr, P_target, top1_target, H_target_ref):
    with torch.no_grad():
        logits_curr = H_curr @ W_head.T
        log_q = F.log_softmax(logits_curr, dim=-1)
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
        
        cos = (H_curr * H_target_ref).sum(dim=-1) / (H_curr.norm(dim=-1) * H_target_ref.norm(dim=-1) + 1e-9)
        mean_cos = cos.mean().item() * 100.0
        
    return {
        "kl": round(kl, 4),
        "top1": round(top1_match, 2),
        "top5": round(top5_match, 2),
        "cosine": round(mean_cos, 2)
    }

base_train_m = eval_metrics(H24_h5a_train, P_bf16_train, Top1_bf16_train, H24_bf16_train)
base_hold_m  = eval_metrics(H24_h5a_hold,  P_bf16_hold,  Top1_bf16_hold,  H24_bf16_hold)
base_ppl     = eval_ppl(model)

print(f"-> Baseline H5-A v2.1: Treino KL={base_train_m['kl']:.4f} (Top1={base_train_m['top1']}%) | Holdout KL={base_hold_m['kl']:.4f} (Top1={base_hold_m['top1']}%) | PPL={base_ppl['Media']:.2f}", flush=True)

# -----------------------------------------------------------------------------
# 4. IMPLEMENTAÇÃO DOS MÓDULOS DE ADAPTAÇÃO: LOW-RANK E ADAPTIVE GATING
# -----------------------------------------------------------------------------
class LowRankCorrection(nn.Module):
    def __init__(self, d_model=2048, rank=8, use_gating=False):
        super().__init__()
        self.rank = rank
        self.use_gating = use_gating
        self.B = nn.Parameter(torch.randn(d_model, rank) * (1.0 / math.sqrt(d_model)))
        self.A = nn.Parameter(torch.zeros(d_model, rank))
        if use_gating:
            self.w_gate = nn.Parameter(torch.zeros(d_model))
        else:
            self.register_parameter('w_gate', None)
            
    def forward(self, h):
        orig_dtype = h.dtype
        h_f = h.float()
        delta = (h_f @ self.B.float()) @ self.A.float().T
        if self.use_gating:
            # Gating dinâmico suave: quando w_gate=0, gate=1.0 exato
            gate = 1.0 + torch.tanh(h_f @ self.w_gate.float()).unsqueeze(-1)
            delta = delta * gate
        return (h_f + delta).to(orig_dtype)
        
    def frobenius_norm_sq(self):
        AtA = self.A.T @ self.A
        BtB = self.B.T @ self.B
        return torch.sum(AtA * BtB)

# Funções auxiliares para criar hooks sem vazamento de dtype ou argumentos
def make_l12_hook(module_corr):
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            h_corr = module_corr(output[0])
            return (h_corr,) + output[1:]
        else:
            return module_corr(output)
    return hook_fn

def make_l24_hook(module_corr):
    def hook_fn(module, args):
        h_corr = module_corr(args[0])
        return (h_corr,) + tuple(args[1:])
    return hook_fn

# -----------------------------------------------------------------------------
# 5. TREINAMENTO E AVALIAÇÃO DAS CONFIGURAÇÕES
# -----------------------------------------------------------------------------
print("\n[4/6] Otimizando as configurações de Adaptação...", flush=True)

experiments_results = {}

# CONFIG 1: Controle V3-B (Rank 16 na Camada 24, linear)
print("\n--- Treinando Config 1: Controle V3-B (L24 Rank 16 - 128 KB) ---", flush=True)
torch.manual_seed(42)
mod_v3b = LowRankCorrection(d_model=D_hidden, rank=16, use_gating=False)
opt = torch.optim.AdamW(mod_v3b.parameters(), lr=3e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=150, eta_min=1e-4)

for step in range(150):
    opt.zero_grad()
    h_corr = mod_v3b(H24_h5a_train)
    logits = h_corr @ W_head.T
    log_q = F.log_softmax(logits, dim=-1)
    loss = -(P_bf16_train * log_q).sum(dim=-1).mean()
    loss.backward()
    opt.step()
    sched.step()

with torch.no_grad():
    h_train_v3b = mod_v3b(H24_h5a_train)
    h_hold_v3b  = mod_v3b(H24_h5a_hold)

h_handle = model.lm_head.register_forward_pre_hook(make_l24_hook(mod_v3b))
ppl_v3b = eval_ppl(model)
h_handle.remove()

m_train_v3b = eval_metrics(h_train_v3b, P_bf16_train, Top1_bf16_train, H24_bf16_train)
m_hold_v3b  = eval_metrics(h_hold_v3b,  P_bf16_hold,  Top1_bf16_hold,  H24_bf16_hold)
experiments_results["Control_V3B_R16"] = {
    "name": "Controle V3-B (L24 Rank 16)",
    "size_kb": 128.0,
    "train": m_train_v3b,
    "holdout": m_hold_v3b,
    "gap_kl": round(m_hold_v3b["kl"] - m_train_v3b["kl"], 4),
    "top1_retention": round((m_hold_v3b["top1"] / m_train_v3b["top1"]) * 100.0, 2),
    "ppl": {k: round(v, 2) for k, v in ppl_v3b.items()}
}
print(f"V3-B R16: Treino KL={m_train_v3b['kl']:.4f} | Holdout KL={m_hold_v3b['kl']:.4f} | Gap={experiments_results['Control_V3B_R16']['gap_kl']:.4f} | PPL={ppl_v3b['Media']:.2f}", flush=True)

# CONFIG 2: Controle V3-Gated (Rank 16 na Camada 24 com Adaptive Gating - 132 KB)
print("\n--- Treinando Config 2: Controle V3-Gated (L24 Rank 16 + Gate - 132 KB) ---", flush=True)
torch.manual_seed(42)
mod_v3g = LowRankCorrection(d_model=D_hidden, rank=16, use_gating=True)
opt = torch.optim.AdamW(mod_v3g.parameters(), lr=3e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=150, eta_min=1e-4)

for step in range(150):
    opt.zero_grad()
    h_corr = mod_v3g(H24_h5a_train)
    logits = h_corr @ W_head.T
    log_q = F.log_softmax(logits, dim=-1)
    loss = -(P_bf16_train * log_q).sum(dim=-1).mean()
    loss.backward()
    opt.step()
    sched.step()

with torch.no_grad():
    h_train_v3g = mod_v3g(H24_h5a_train)
    h_hold_v3g  = mod_v3g(H24_h5a_hold)

h_handle = model.lm_head.register_forward_pre_hook(make_l24_hook(mod_v3g))
ppl_v3g = eval_ppl(model)
h_handle.remove()

m_train_v3g = eval_metrics(h_train_v3g, P_bf16_train, Top1_bf16_train, H24_bf16_train)
m_hold_v3g  = eval_metrics(h_hold_v3g,  P_bf16_hold,  Top1_bf16_hold,  H24_bf16_hold)
experiments_results["Control_V3Gated_R16"] = {
    "name": "Controle V3-Gated (L24 Rank 16 + Gate)",
    "size_kb": 132.0,
    "train": m_train_v3g,
    "holdout": m_hold_v3g,
    "gap_kl": round(m_hold_v3g["kl"] - m_train_v3g["kl"], 4),
    "top1_retention": round((m_hold_v3g["top1"] / m_train_v3g["top1"]) * 100.0, 2),
    "ppl": {k: round(v, 2) for k, v in ppl_v3g.items()}
}
print(f"V3-Gated R16: Treino KL={m_train_v3g['kl']:.4f} | Holdout KL={m_hold_v3g['kl']:.4f} | Gap={experiments_results['Control_V3Gated_R16']['gap_kl']:.4f} | PPL={ppl_v3g['Media']:.2f}", flush=True)

# CONFIG 3: H5-A v4-Ladder (L12 Rank 8 + L24 Rank 8 - EXATOS 128 KB)
print("\n--- Treinando Config 3: H5-A v4-Ladder (L12 Rank 8 + L24 Rank 8 - 128 KB) ---", flush=True)
torch.manual_seed(42)
mod_l12 = LowRankCorrection(d_model=D_hidden, rank=8, use_gating=False)
opt12 = torch.optim.AdamW(mod_l12.parameters(), lr=5e-3, weight_decay=1e-4)

# Etapa 1 do Ladder: Otimizar adaptador da Camada 12 para restaurar a representação intermediária do BF16
for step in range(150):
    opt12.zero_grad()
    h12_corr = mod_l12(H12_h5a_train)
    mse = F.mse_loss(h12_corr, H12_bf16_train)
    cos = 1.0 - ((h12_corr * H12_bf16_train).sum(dim=-1) / (h12_corr.norm(dim=-1) * H12_bf16_train.norm(dim=-1) + 1e-9)).mean()
    loss12 = mse + 0.5 * cos
    loss12.backward()
    opt12.step()

with torch.no_grad():
    h12_fixed_train = mod_l12(H12_h5a_train)
    cos_fixed = ((h12_fixed_train * H12_bf16_train).sum(dim=-1) / (h12_fixed_train.norm(dim=-1) * H12_bf16_train.norm(dim=-1) + 1e-9)).mean().item() * 100.0
    rel_err_fixed = ((h12_fixed_train - H12_bf16_train).norm(dim=-1) / (H12_bf16_train.norm(dim=-1) + 1e-9)).mean().item() * 100.0
    print(f"-> Camada 12 APÓS Adaptação: Cosine={cos_fixed:.2f}% (era {cos_l12_train:.2f}%) | Erro Relativo={rel_err_fixed:.2f}% (era {rel_err_l12_train:.2f}%)", flush=True)

# Etapa 2 do Ladder: Coletar H24 com o hook ativo na Camada 11 (alimentando a Camada 12)
h12_handle = model.model.layers[11].register_forward_hook(make_l12_hook(mod_l12))

h24_ladder_train_list, h24_ladder_hold_list = [], []
with torch.no_grad():
    for name, text in train_corpus.items():
        inp = tokenizer(text, return_tensors="pt")
        out = model(inp.input_ids, output_hidden_states=True)
        h24_ladder_train_list.append(out.hidden_states[-1][0].float())
    for name, text in holdout_corpus.items():
        inp = tokenizer(text, return_tensors="pt")
        out = model(inp.input_ids, output_hidden_states=True)
        h24_ladder_hold_list.append(out.hidden_states[-1][0].float())

H24_ladder_train = torch.cat(h24_ladder_train_list, dim=0)
H24_ladder_hold  = torch.cat(h24_ladder_hold_list, dim=0)

# Etapa 3 do Ladder: Otimizar adaptador da Camada 24 (Rank 8) com a representação intermediária sanada
mod_l24 = LowRankCorrection(d_model=D_hidden, rank=8, use_gating=False)
opt24 = torch.optim.AdamW(mod_l24.parameters(), lr=3e-3, weight_decay=1e-4)
sched24 = torch.optim.lr_scheduler.CosineAnnealingLR(opt24, T_max=150, eta_min=1e-4)

for step in range(150):
    opt24.zero_grad()
    h_corr = mod_l24(H24_ladder_train)
    logits = h_corr @ W_head.T
    log_q = F.log_softmax(logits, dim=-1)
    loss = -(P_bf16_train * log_q).sum(dim=-1).mean()
    loss.backward()
    opt24.step()
    sched24.step()

with torch.no_grad():
    h_train_ladder = mod_l24(H24_ladder_train)
    h_hold_ladder  = mod_l24(H24_ladder_hold)

h24_handle = model.lm_head.register_forward_pre_hook(make_l24_hook(mod_l24))
ppl_ladder = eval_ppl(model)
h12_handle.remove()
h24_handle.remove()

m_train_ladder = eval_metrics(h_train_ladder, P_bf16_train, Top1_bf16_train, H24_bf16_train)
m_hold_ladder  = eval_metrics(h_hold_ladder,  P_bf16_hold,  Top1_bf16_hold,  H24_bf16_hold)
experiments_results["H5A_v4_Ladder"] = {
    "name": "H5-A v4-Ladder (L12 R8 + L24 R8)",
    "size_kb": 128.0,
    "train": m_train_ladder,
    "holdout": m_hold_ladder,
    "gap_kl": round(m_hold_ladder["kl"] - m_train_ladder["kl"], 4),
    "top1_retention": round((m_hold_ladder["top1"] / m_train_ladder["top1"]) * 100.0, 2),
    "ppl": {k: round(v, 2) for k, v in ppl_ladder.items()}
}
print(f"v4-Ladder: Treino KL={m_train_ladder['kl']:.4f} | Holdout KL={m_hold_ladder['kl']:.4f} | Gap={experiments_results['H5A_v4_Ladder']['gap_kl']:.4f} | PPL={ppl_ladder['Media']:.2f}", flush=True)

# CONFIG 4: H5-A v4-Ladder-Gated (L12 R8-Gated + L24 R8-Gated - 136 KB)
print("\n--- Treinando Config 4: H5-A v4-Ladder-Gated (L12 R8-G + L24 R8-G - 136 KB) ---", flush=True)
torch.manual_seed(42)
mod_l12_g = LowRankCorrection(d_model=D_hidden, rank=8, use_gating=True)
opt12_g = torch.optim.AdamW(mod_l12_g.parameters(), lr=5e-3, weight_decay=1e-4)

for step in range(150):
    opt12_g.zero_grad()
    h12_corr = mod_l12_g(H12_h5a_train)
    mse = F.mse_loss(h12_corr, H12_bf16_train)
    cos = 1.0 - ((h12_corr * H12_bf16_train).sum(dim=-1) / (h12_corr.norm(dim=-1) * H12_bf16_train.norm(dim=-1) + 1e-9)).mean()
    loss12 = mse + 0.5 * cos
    loss12.backward()
    opt12_g.step()

h12_g_handle = model.model.layers[11].register_forward_hook(make_l12_hook(mod_l12_g))

h24_lg_train_list, h24_lg_hold_list = [], []
with torch.no_grad():
    for name, text in train_corpus.items():
        inp = tokenizer(text, return_tensors="pt")
        out = model(inp.input_ids, output_hidden_states=True)
        h24_lg_train_list.append(out.hidden_states[-1][0].float())
    for name, text in holdout_corpus.items():
        inp = tokenizer(text, return_tensors="pt")
        out = model(inp.input_ids, output_hidden_states=True)
        h24_lg_hold_list.append(out.hidden_states[-1][0].float())

H24_lg_train = torch.cat(h24_lg_train_list, dim=0)
H24_lg_hold  = torch.cat(h24_lg_hold_list, dim=0)

mod_l24_g = LowRankCorrection(d_model=D_hidden, rank=8, use_gating=True)
opt24_g = torch.optim.AdamW(mod_l24_g.parameters(), lr=3e-3, weight_decay=1e-4)
sched24_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt24_g, T_max=150, eta_min=1e-4)

for step in range(150):
    opt24_g.zero_grad()
    h_corr = mod_l24_g(H24_lg_train)
    logits = h_corr @ W_head.T
    log_q = F.log_softmax(logits, dim=-1)
    loss = -(P_bf16_train * log_q).sum(dim=-1).mean()
    loss.backward()
    opt24_g.step()
    sched24_g.step()

with torch.no_grad():
    h_train_lg = mod_l24_g(H24_lg_train)
    h_hold_lg  = mod_l24_g(H24_lg_hold)

h24_g_handle = model.lm_head.register_forward_pre_hook(make_l24_hook(mod_l24_g))
ppl_ladder_gated = eval_ppl(model)
h12_g_handle.remove()
h24_g_handle.remove()

m_train_lg = eval_metrics(h_train_lg, P_bf16_train, Top1_bf16_train, H24_bf16_train)
m_hold_lg  = eval_metrics(h_hold_lg,  P_bf16_hold,  Top1_bf16_hold,  H24_bf16_hold)
experiments_results["H5A_v4_Ladder_Gated"] = {
    "name": "H5-A v4-Ladder-Gated (L12 R8-G + L24 R8-G)",
    "size_kb": 136.0,
    "train": m_train_lg,
    "holdout": m_hold_lg,
    "gap_kl": round(m_hold_lg["kl"] - m_train_lg["kl"], 4),
    "top1_retention": round((m_hold_lg["top1"] / m_train_lg["top1"]) * 100.0, 2),
    "ppl": {k: round(v, 2) for k, v in ppl_ladder_gated.items()}
}
print(f"v4-Ladder-Gated: Treino KL={m_train_lg['kl']:.4f} | Holdout KL={m_hold_lg['kl']:.4f} | Gap={experiments_results['H5A_v4_Ladder_Gated']['gap_kl']:.4f} | PPL={ppl_ladder_gated['Media']:.2f}", flush=True)

# -----------------------------------------------------------------------------
# 6. RELATÓRIO CONSOLIDADO E PERSISTÊNCIA DOS RESULTADOS
# -----------------------------------------------------------------------------
print("\n" + "=" * 120, flush=True)
print(f"{'Configuração':<34} | {'Size':<8} | {'Treino KL':<10} | {'Hold KL':<9} | {'Gap KL':<8} | {'Ret. Top1':<10} | {'PPL Média':<10}", flush=True)
print("-" * 120, flush=True)
print(f"{'Baseline H5-A v2.1 Rebalanced':<34} | {'0.0 KB':<8} | {base_train_m['kl']:<10.4f} | {base_hold_m['kl']:<9.4f} | {base_hold_m['kl']-base_train_m['kl']:<+8.4f} | {(base_hold_m['top1']/base_train_m['top1'])*100.0:<9.2f}% | {base_ppl['Media']:<10.2f}", flush=True)
for key, res in experiments_results.items():
    print(f"{res['name']:<34} | {res['size_kb']:<5.1f} KB | {res['train']['kl']:<10.4f} | {res['holdout']['kl']:<9.4f} | {res['gap_kl']:<+8.4f} | {res['top1_retention']:<9.2f}% | {res['ppl']['Media']:<10.2f}", flush=True)

print("\nDetalhamento da PPL Canônica por Domínio:", flush=True)
print(f"{'Domínio':<16} | {'Baseline':<10} | {'V3-B R16':<10} | {'V3-Gated':<10} | {'v4-Ladder':<10} | {'v4-Ladder-G':<10}", flush=True)
print("-" * 75, flush=True)
for dom in ["Codigo", "Enciclopedia", "Ciencia", "Historia"]:
    p_b   = base_ppl[dom]
    p_v3b = experiments_results["Control_V3B_R16"]["ppl"][dom]
    p_v3g = experiments_results["Control_V3Gated_R16"]["ppl"][dom]
    p_lad = experiments_results["H5A_v4_Ladder"]["ppl"][dom]
    p_lg  = experiments_results["H5A_v4_Ladder_Gated"]["ppl"][dom]
    print(f"{dom:<16} | {p_b:<10.2f} | {p_v3b:<10.2f} | {p_v3g:<10.2f} | {p_lad:<10.2f} | {p_lg:<10.2f}", flush=True)

output_final = {
    "baseline": {
        "train": base_train_m,
        "holdout": base_hold_m,
        "ppl": {k: round(v, 2) for k, v in base_ppl.items()}
    },
    "experiments": experiments_results
}

save_path = "e:/agente/compress/h5a_v4_ladder_and_gating_results.json"
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(output_final, f, indent=2, ensure_ascii=False)

print(f"\nResultados salvos com sucesso em: {save_path}", flush=True)
print("=" * 120, flush=True)
