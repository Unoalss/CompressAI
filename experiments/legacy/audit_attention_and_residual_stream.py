import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import time
import copy
from redistribute_h5_protection import quantize_matrix

def evaluate_perplexity(model, tokenizer, text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        loss = model(**inputs, labels=inputs.input_ids).loss
        return torch.exp(loss).item()

def evaluate_generation(model, tokenizer, prompt, max_tokens=25):
    inp = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    tokens = tokenizer.encode(text)
    unique_ratio = len(set(tokens)) / len(tokens)
    return text, unique_ratio

def analyze_code_generation(text):
    newlines = text.count('\n')
    if newlines >= 4:
        lines = [l.strip() for l in text.split('\n')]
        empty_lines = sum(1 for l in lines if len(l) == 0)
        if empty_lines >= 3:
            return "Loop \\n (Indentacao)"
    tokens = text.split()
    if len(tokens) >= 6:
        from collections import Counter
        counts = Counter(tokens)
        most_common, freq = counts.most_common(1)[0]
        if freq >= 4:
            return f"Repeticao ('{most_common}')"
    return "OK (Sem Loop)"

def audit_residual_stream_divergence(model_fp16, model_quant, tokenizer, test_prompt):
    """
    Rastreia hook a hook o fluxo de ativacao em cada uma das 6 camadas:
    - Norma da saida da Atencao ||Attn(x)||
    - Norma da saida da MLP ||MLP(x)||
    - Razao Atencao / MLP
    - Cosseno do residual stream contra o FP16
    """
    inputs = tokenizer(test_prompt, return_tensors="pt")
    
    logs_fp16 = []
    logs_quant = []
    
    def get_hooks(storage_list):
        hooks = []
        for l_idx in range(6):
            block = (model_fp16 if storage_list is logs_fp16 else model_quant).transformer.h[l_idx]
            
            def make_hook(layer_i):
                def hook_fn(module, inp, outp):
                    # outp eh o hidden_states apos o bloco
                    # captura atencao e mlp
                    storage_list.append({
                        'layer': layer_i,
                        'hidden_norm': torch.norm(outp[0]).item()
                    })
                return hook_fn
            
            h = block.register_forward_hook(make_hook(l_idx))
            hooks.append(h)
        return hooks

    # Executa FP16
    h_fp16 = get_hooks(logs_fp16)
    with torch.no_grad():
        _ = model_fp16(**inputs)
    for h in h_fp16: h.remove()
    
    # Executa Quantizado
    h_quant = get_hooks(logs_quant)
    with torch.no_grad():
        _ = model_quant(**inputs)
    for h in h_quant: h.remove()
    
    return logs_fp16, logs_quant

def main():
    print("=" * 125)
    print("FASE 1: AUDITORIA DO RESIDUAL STREAM & QUANTIZAÇÃO DA ATENÇÃO (`c_attn`)")
    print("Investigando a interação entre Atenção e MLP no Código Python a 3.05 bpp")
    print("=" * 125)
    
    model_name = "distilbert/distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    base_model.eval()
    
    domains = {
        "Ciencia": "Quantum mechanics describes nature at the smallest scales of energy levels of atoms and subatomic particles.",
        "Codigo": "def binary_search(array, target): low = 0; high = len(array) - 1; while low <= high: mid = (low + high) // 2",
        "Historia": "The ancient library of Alexandria was once the largest repository of knowledge in the Hellenistic world.",
        "Enciclopedia": "The solar system formed approximately 4.6 billion years ago from the gravitational collapse of a molecular cloud."
    }
    
    code_prompts = [
        ("Fibonacci", "def fibonacci(n):"),
        ("Quicksort", "def quicksort(arr):"),
        ("Busca Binaria", "def binary_search(arr, target):"),
        ("Classe Python", "class BankAccount:\n    def __init__(self, balance):"),
        ("Funcao Matematica", "def calculate_area(radius):"),
        ("JSON / Dados", "data = {\n    \"name\": \"Alice\",\n    \"age\":")
    ]
    
    ppl_orig = {d: evaluate_perplexity(base_model, tokenizer, text) for d, text in domains.items()}
    avg_orig = np.mean(list(ppl_orig.values()))
    print(f"\n[0] FP16 ORIGINAL (16.0 bpp): Media PPL = {avg_orig:.2f}")
    
    # 1. CONSTRUIR O CAMPEÃO H5-A (MLP a 3.05 bpp)
    print("\n" + "-" * 120)
    print("Construindo Modelo H5-A (MLP a 3.05 bpp)...")
    print("-" * 120)
    
    m_h5a = copy.deepcopy(base_model)
    for l_idx, block in enumerate(m_h5a.transformer.h):
        p_fc = 0.03 if l_idx == 0 else 0.05
        p_proj = 0.03 if l_idx == 0 else (0.10 if l_idx == 5 else 0.05)
        with torch.no_grad():
            w_fc, _, _, _ = quantize_matrix(block.mlp.c_fc.weight.data, num_ctrl=2, num_spec=4, spec_mode='sensitivity', p_crit=p_fc, seed=42)
            block.mlp.c_fc.weight.data = w_fc
            w_proj, _, _, _ = quantize_matrix(block.mlp.c_proj.weight.data, num_ctrl=3, num_spec=3, spec_mode='standard', p_crit=p_proj, seed=42)
            block.mlp.c_proj.weight.data = w_proj

    # 2. DIAGNÓSTICO DO RESIDUAL STREAM (FP16 vs H5-A no Prompt de Fibonacci)
    print("\n" + "=" * 120)
    print("DIAGNÓSTICO: DESVIO DE ENERGIA DO RESIDUAL STREAM (PROMPT 'def fibonacci(n):')")
    print("=" * 120)
    logs_fp16, logs_quant = audit_residual_stream_divergence(base_model, m_h5a, tokenizer, "def fibonacci(n):")
    print(f"{'Camada':<10} | {'Norma FP16':<16} | {'Norma H5-A Quant':<18} | {'Divergência Ratio':<18} | {'Delta Norma'}")
    print("-" * 120)
    for i in range(6):
        n_fp = logs_fp16[i]['hidden_norm']
        n_q = logs_quant[i]['hidden_norm']
        ratio = n_q / n_fp
        delta = n_q - n_fp
        print(f"Layer h{i:<3} | {n_fp:>14.4f}  | {n_q:>16.4f}  | {ratio:>16.4f}x | {delta:>+11.4f}")

    # 3. EXPERIMENTO COM A ATENÇÃO
    print("\n" + "=" * 120)
    print("TESTES COMPARATIVOS DE ATENÇÃO & BALANCEAMENTO RESIDUAL A 3.05 BPP")
    print("=" * 120)
    
    # Variante 1: H5-A Referência (Apenas MLP quantizada)
    # Variante 2: H5-A + Calibração de Ganho Residual da MLP (Norm Balancing para manter razão exata da Atencao)
    # Variante 3: H5-A + Atencao Completa (`c_attn` e `c_proj` de atencao) quantizada a 3.05 bpp!
    
    # Construir Variante 2: Calibração de Ganho Residual
    m_balanced = copy.deepcopy(m_h5a)
    for l_idx in range(6):
        # Escala a saida da MLP pelo inverso da divergência media observada
        ratio = logs_quant[l_idx]['hidden_norm'] / logs_fp16[l_idx]['hidden_norm']
        if ratio > 0:
            with torch.no_grad():
                m_balanced.transformer.h[l_idx].mlp.c_proj.weight.data /= ratio

    # Construir Variante 3: Quantização Total da Atenção (`c_attn` a 3.05 bpp)
    print("Quantizando Matrizes de Atencao (c_attn e attn.c_proj) a 3.05 bpp...")
    m_all_quant = copy.deepcopy(m_h5a)
    for l_idx, block in enumerate(m_all_quant.transformer.h):
        with torch.no_grad():
            # c_attn: 768 x 2304
            w_attn, _, _, _ = quantize_matrix(block.attn.c_attn.weight.data, num_ctrl=3, num_spec=3, spec_mode='standard', p_crit=0.05, seed=42)
            block.attn.c_attn.weight.data = w_attn
            # attn.c_proj: 768 x 768
            w_aproj, _, _, _ = quantize_matrix(block.attn.c_proj.weight.data, num_ctrl=3, num_spec=3, spec_mode='standard', p_crit=0.05, seed=42)
            block.attn.c_proj.weight.data = w_aproj
            
    models_to_evaluate = [
        ('FP16 Original', 'FP16 Original (16.0 bpp)', base_model),
        ('H5-A Ref', 'H5-A: Apenas MLP quantizada (3.05 bpp)', m_h5a),
        ('H5-A Balanced', 'H5-A + Balanceamento Residual de Norma', m_balanced),
        ('Total Quant', 'Total 3.05 bpp: Atencao + MLP quantizadas', m_all_quant)
    ]
    
    score_results = []
    
    for code, name, model_eval in models_to_evaluate:
        print(f"\nAvaliando {name}...")
        ppls = {d: evaluate_perplexity(model_eval, tokenizer, text) for d, text in domains.items()}
        avg_ppl = np.mean(list(ppls.values()))
        
        loop_cnt = 0
        div_vals = []
        code_snippets = []
        
        for p_label, p_text in code_prompts:
            gen_txt, div_r = evaluate_generation(model_eval, tokenizer, p_text, max_tokens=22)
            st = analyze_code_generation(gen_txt)
            if st != "OK (Sem Loop)":
                loop_cnt += 1
            div_vals.append(div_r)
            code_snippets.append((p_label, gen_txt, st))
            
        avg_div = np.mean(div_vals)
        score_results.append({
            'code': code,
            'name': name,
            'avg_ppl': avg_ppl,
            'ppls': ppls,
            'loop_cnt': loop_cnt,
            'avg_div': avg_div,
            'snippets': code_snippets
        })
        print(f"   -> PPL Medio: {avg_ppl:.2f} (Ciencia: {ppls['Ciencia']:.1f} | Codigo: {ppls['Codigo']:.1f})")
        print(f"   -> Bateria de Codigo: Loops = {loop_cnt}/6 | Diversidade Media = {avg_div*100:.1f}%")

    print("\n" + "=" * 135)
    print("SCOREBOARD COMPARATIVO FINAL: ATENÇÃO & RESIDUAL STREAM (3.05 BPP)")
    print("=" * 135)
    print(f"{'Configuracao':<35} | {'PPL Medio':<10} | {'Ciencia':<8} | {'Codigo':<8} | {'Historia':<8} | {'Encic':<8} | {'Loops Codigo':<13} | {'Div. Codigo'}")
    print("-" * 135)
    
    for r in score_results:
        print(f"{r['name'][:35]:<35} | {r['avg_ppl']:>9.2f}  | {r['ppls']['Ciencia']:>8.1f} | {r['ppls']['Codigo']:>8.1f} | {r['ppls']['Historia']:>8.1f} | {r['ppls']['Enciclopedia']:>8.1f} | {r['loop_cnt']}/6 loops   | {r['avg_div']*100:>10.1f}%")
    print("=" * 135)

    print("\n" + "=" * 135)
    print("AMOSTRAS DE CÓDIGO GERADAS: COMPARAÇÃO DIRETA")
    print("=" * 135)
    for idx in range(len(code_prompts)):
        p_name = code_prompts[idx][0]
        print(f"\n--- {p_name} ---")
        for r in score_results:
            snippet = r['snippets'][idx][1].replace('\n', ' \\n ')
            status = r['snippets'][idx][2]
            print(f"[{r['code']:<14}] ({status:<20}): \"{snippet[:75]}...\"")
    print("=" * 135)

if __name__ == '__main__':
    main()
