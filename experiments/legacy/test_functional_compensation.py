import os
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from test_qwen_ffn_single_layer import quantize_matrix_h5a_batched

MODEL_DIR = "E:/agente/modelo"
device = "cpu"
torch.set_num_threads(os.cpu_count() or 8)

print("=" * 90)
print("TESTE DE COMPENSAÇÃO FUNCIONAL ORIENTADA À ATIVAÇÃO NO QWEN3.5-2B")
print("=" * 90)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
calib_texts = [
    "Quantum mechanics describes nature at the smallest scales of energy levels of atoms.",
    "def binary_search(array, target): low = 0; high = len(array) - 1; while low <= high: mid = (low + high) // 2",
    "The solar system formed approximately 4.6 billion years ago from the gravitational collapse of a molecular cloud."
]

print("Carregando modelo BF16...")
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
model.eval()

# Coletar ativacoes de entrada no Layer 0 FFN (down_proj)
# down_proj recebe a ativacao da camada intermediaria (hidden_dim = 6144)
captured_acts = []
def hook_fn(module, input, output):
    captured_acts.append(input[0].detach().float())

handle = model.model.layers[0].mlp.down_proj.register_forward_hook(hook_fn)

with torch.no_grad():
    for text in calib_texts:
        inp = tokenizer(text, return_tensors="pt")
        model(**inp)

handle.remove()

# Concatenar ativacoes de todos os tokens
X = torch.cat([act.reshape(-1, act.shape[-1]) for act in captured_acts], dim=0) # [T, 6144]
print(f"Tokens de calibracao coletados: {X.shape[0]}, dimensao: {X.shape[1]}")

# Analise das ativacoes
act_rms = torch.sqrt(torch.mean(X ** 2, dim=0)) # [6144]
print(f"RMS das Ativacoes: min={act_rms.min():.4f}, median={act_rms.median():.4f}, max={act_rms.max():.4f}, ratio={act_rms.max()/act_rms.median():.2f}x")

# Pegar o peso real do down_proj
W_orig = model.model.layers[0].mlp.down_proj.weight.data.clone().float() # [2048, 6144]
Y_orig = X @ W_orig.T # [T, 2048]
norm_Y = torch.norm(Y_orig)

# 1. H5-A Base (3.05 BPW)
print("\n1. Quantizando W com H5-A Base (3.05 BPW)...")
W_h5a, _, _, _, _ = quantize_matrix_h5a_batched(W_orig, num_ctrl=3, num_spec=3, p_crit=0.05, seed=42)
W_h5a = W_h5a.float()
Y_h5a = X @ W_h5a.T

err_weight_base = torch.norm(W_orig - W_h5a) / torch.norm(W_orig)
err_func_base = torch.norm(Y_orig - Y_h5a) / norm_Y
cos_weight_base = torch.nn.functional.cosine_similarity(W_orig.flatten(), W_h5a.flatten(), dim=0)
print(f"Base H5-A: Erro Peso={err_weight_base*100:.2f}%, Erro Saida Funcional={err_func_base*100:.2f}%, Cosine={cos_weight_base*100:.2f}%")

# 2. Selecao de 2% Canais por Erro de Peso vs Erro de Saida Funcional
num_cols = W_orig.shape[1]
k_crit = int(0.02 * num_cols) # 122 colunas

# Metodo A: Selecao por energia do erro do peso
E = W_orig - W_h5a
col_weight_err = torch.sum(E ** 2, dim=0)
topk_weight = torch.topk(col_weight_err, k=k_crit).indices

W_prot_weight = W_h5a.clone()
W_prot_weight[:, topk_weight] = W_orig[:, topk_weight]
Y_prot_weight = X @ W_prot_weight.T
err_func_weight_prot = torch.norm(Y_orig - Y_prot_weight) / norm_Y
print(f"Protecao 2% por Erro de Peso: Erro Saida Funcional={err_func_weight_prot*100:.2f}% (Reducao: {(err_func_base - err_func_weight_prot)*100:.2f}pp)")

# Metodo B: Selecao por energia do erro funcional de saida: Error_j = act_rms_j^2 * col_weight_err_j
col_func_err = col_weight_err * (act_rms ** 2)
topk_func = torch.topk(col_func_err, k=k_crit).indices

overlap = len(set(topk_weight.tolist()).intersection(set(topk_func.tolist())))
print(f"Sobreposicao entre canais de peso vs canais funcionais: {overlap}/{k_crit} ({overlap/k_crit*100:.1f}%)")

W_prot_func = W_h5a.clone()
W_prot_func[:, topk_func] = W_orig[:, topk_func]
Y_prot_func = X @ W_prot_func.T
err_func_func_prot = torch.norm(Y_orig - Y_prot_func) / norm_Y
print(f"Protecao 2% por Erro Funcional: Erro Saida Funcional={err_func_func_prot*100:.2f}% (Reducao: {(err_func_base - err_func_func_prot)*100:.2f}pp)")

# 3. Correcao Compacta Low-Rank Funcional (C = U V^T)
# Queremos minimizar ||X (E - C)^T||_F
# Seja X = U_x S_x V_x^T ou simplesmente ponderar E pelo RMS da ativacao: E_scaled = E * act_rms
# SVD de E_scaled:
print("\n3. Testando Correcao Funcional Low-Rank (sem restaurar canais BF16 inteiros)...")
for rank in [2, 4, 8, 16]:
    # Ponderar erro pelas ativacoes: cada coluna multiplicada por act_rms
    E_scaled = E * act_rms.unsqueeze(0) # [2048, 6144]
    U, S_vals, Vh = torch.linalg.svd(E_scaled, full_matrices=False)
    
    # Reconstruir correcao low-rank no espaco ponderado
    C_scaled = (U[:, :rank] * S_vals[:rank]) @ Vh[:rank, :]
    # Desescalar
    C = C_scaled / (act_rms.unsqueeze(0) + 1e-8)
    
    W_corrected = W_h5a + C
    Y_corrected = X @ W_corrected.T
    err_func_lr = torch.norm(Y_orig - Y_corrected) / norm_Y
    param_count = rank * (W_orig.shape[0] + W_orig.shape[1])
    bytes_bf16 = param_count * 2
    print(f"Rank-{rank:<2}: Params={param_count:<6} ({bytes_bf16/1024:.1f} KB, {param_count/W_orig.numel()*100:.3f}% da matriz) -> Erro Saida Funcional={err_func_lr*100:.2f}%")

# 4. Avaliar Perplexidade no Modelo Real com L0 down_proj modificado
eval_text = "The solar system formed approximately 4.6 billion years ago from the gravitational collapse of a molecular cloud. Quantum mechanics describes nature at the smallest scales."
inp_eval = tokenizer(eval_text, return_tensors="pt")

with torch.no_grad():
    base_loss = model(**inp_eval, labels=inp_eval.input_ids).loss.item()
    base_ppl = torch.exp(torch.tensor(base_loss)).item()

print(f"\nModelo Base BF16 PPL: {base_ppl:.3f}")

# Testar Base H5-A
model.model.layers[0].mlp.down_proj.weight.data.copy_(W_h5a.to(torch.bfloat16))
with torch.no_grad():
    ppl_h5a = torch.exp(model(**inp_eval, labels=inp_eval.input_ids).loss).item()
print(f"Com L0 down_proj H5-A Base: PPL={ppl_h5a:.3f} (Delta={ppl_h5a-base_ppl:+.3f})")

# Testar Protecao Peso
model.model.layers[0].mlp.down_proj.weight.data.copy_(W_prot_weight.to(torch.bfloat16))
with torch.no_grad():
    ppl_prot_w = torch.exp(model(**inp_eval, labels=inp_eval.input_ids).loss).item()
print(f"Com L0 down_proj Prot 2% Peso: PPL={ppl_prot_w:.3f} (Delta={ppl_prot_w-base_ppl:+.3f})")

# Testar Protecao Funcional
model.model.layers[0].mlp.down_proj.weight.data.copy_(W_prot_func.to(torch.bfloat16))
with torch.no_grad():
    ppl_prot_f = torch.exp(model(**inp_eval, labels=inp_eval.input_ids).loss).item()
print(f"Com L0 down_proj Prot 2% Funcional: PPL={ppl_prot_f:.3f} (Delta={ppl_prot_f-base_ppl:+.3f})")

# Testar Correcao Low-Rank Rank-8
model.model.layers[0].mlp.down_proj.weight.data.copy_(W_corrected.to(torch.bfloat16))
with torch.no_grad():
    ppl_lr = torch.exp(model(**inp_eval, labels=inp_eval.input_ids).loss).item()
print(f"Com L0 down_proj Rank-8 Funcional: PPL={ppl_lr:.3f} (Delta={ppl_lr-base_ppl:+.3f})")
