import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import time
import copy
from refine_model_c import quantize_layer_c_refined
from test_2plus4_variations import quantize_layer_2plus4

def evaluate_perplexity(model, tokenizer, text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        loss = model(**inputs, labels=inputs.input_ids).loss
        return torch.exp(loss).item()

def evaluate_generation(model, tokenizer, prompt):
    inp = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=22, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    tokens = tokenizer.encode(text)
    unique_ratio = len(set(tokens)) / len(tokens)
    return text, unique_ratio

def main():
    print("=" * 115)
    print("ALOCAÇÃO HIERÁRQUICA MISTA: 3+3 (C.3) x 2+4-D ENTRE CAMADAS E PROJEÇÕES")
    print("Objetivo: Unir a vitória em Ciência do 2+4-D (128.3) com a estabilidade do C.3 (82.5)")
    print("Orçamento rigorosamente fixo: 3.05 bpp | DistilGPT-2 | 12 Camadas MLP")
    print("=" * 115)
    
    model_name = "distilbert/distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    base_model.eval()
    
    domains = {
        "Ciência": "Quantum mechanics describes nature at the smallest scales of energy levels of atoms and subatomic particles.",
        "Código": "def binary_search(array, target): low = 0; high = len(array) - 1; while low <= high: mid = (low + high) // 2",
        "História": "The ancient library of Alexandria was once the largest repository of knowledge in the Hellenistic world.",
        "Enciclopédia": "The solar system formed approximately 4.6 billion years ago from the gravitational collapse of a molecular cloud."
    }
    
    # 0. FP16 Original
    ppl_orig = {d: evaluate_perplexity(base_model, tokenizer, text) for d, text in domains.items()}
    avg_orig = np.mean(list(ppl_orig.values()))
    prompt = "The future of artificial intelligence is"
    gen_orig, div_orig = evaluate_generation(base_model, tokenizer, prompt)
    
    print(f"\n[0] FP16 ORIGINAL (16.0 bpp): Média PPL = {avg_orig:.2f}")
    print(f"    Geração: \"{gen_orig}\" (Diversidade: {div_orig*100:.1f}%)")
    
    configs = [
        ('C3_Puro', 'Referência C.3 Puro: 3+3 em todas as 12 camadas',
         lambda l_idx, p_name: '3+3'),
        ('2+4D_Puro', 'Referência 2+4-D Puro: 2+4-D em todas as 12 camadas',
         lambda l_idx, p_name: '2+4-D'),
        ('M1_Profundidade', 'M.1 Profundidade: 3+3 nas camadas h[0..3] e 2+4-D no topo h[4..5]',
         lambda l_idx, p_name: '2+4-D' if l_idx >= 4 else '3+3'),
        ('M2_Funcional', 'M.2 Funcional: 2+4-D na Expansão (c_fc) e 3+3 na Projeção (c_proj)',
         lambda l_idx, p_name: '2+4-D' if p_name == 'c_fc' else '3+3'),
        ('M3_Invertida', 'M.3 Invertida: 2+4-D na Projeção (c_proj) e 3+3 na Expansão (c_fc)',
         lambda l_idx, p_name: '3+3' if p_name == 'c_fc' else '2+4-D'),
        ('M4_Piramide', 'M.4 Pirâmide: 3+3 nas pontas (h[0], h[5]) e 2+4-D no miolo (h[1..4])',
         lambda l_idx, p_name: '3+3' if l_idx in [0, 5] else '2+4-D')
    ]
    
    results = {}
    
    for code, name, alloc_fn in configs:
        print(f"\n" + "-" * 115)
        print(f"Quantizando modelo completo: {name}...")
        print("-" * 115)
        
        m_test = copy.deepcopy(base_model)
        t0 = time.time()
        
        for l_idx, block in enumerate(m_test.transformer.h):
            with torch.no_grad():
                # c_fc
                mode_fc = alloc_fn(l_idx, 'c_fc')
                if mode_fc == '3+3':
                    w_fc, _, _ = quantize_layer_c_refined(block.mlp.c_fc.weight.data, variant='C3_error_subfamilies', p_crit=0.05, seed=42)
                else:
                    w_fc, _, _ = quantize_layer_2plus4(block.mlp.c_fc.weight.data, mode='2+4-D', p_crit=0.05, seed=42)
                block.mlp.c_fc.weight.data = w_fc
                
                # c_proj
                mode_proj = alloc_fn(l_idx, 'c_proj')
                if mode_proj == '3+3':
                    w_proj, _, _ = quantize_layer_c_refined(block.mlp.c_proj.weight.data, variant='C3_error_subfamilies', p_crit=0.05, seed=42)
                else:
                    w_proj, _, _ = quantize_layer_2plus4(block.mlp.c_proj.weight.data, mode='2+4-D', p_crit=0.05, seed=42)
                block.mlp.c_proj.weight.data = w_proj
                
        elapsed = time.time() - t0
        
        ppls = {d: evaluate_perplexity(m_test, tokenizer, text) for d, text in domains.items()}
        avg_ppl = np.mean(list(ppls.values()))
        gen_text, div_ratio = evaluate_generation(m_test, tokenizer, prompt)
        
        results[code] = {
            'name': name,
            'avg_ppl': avg_ppl,
            'ppls': ppls,
            'gen_text': gen_text,
            'div_ratio': div_ratio,
            'time': elapsed
        }
        
        print(f"   -> Média PPL: {avg_ppl:.2f} (Ciência: {ppls['Ciência']:.1f} | Código: {ppls['Código']:.1f} | Hist: {ppls['História']:.1f} | Encic: {ppls['Enciclopédia']:.1f})")
        print(f"   -> Diversidade Léxica: {div_ratio*100:.1f}%")
        print(f"   -> Geração: \"{gen_text}\"")

    print("\n" + "=" * 135)
    print("SCOREBOARD COMPARATIVO FINAL: ALOCAÇÕES MISTAS (3+3 vs 2+4-D) A 3.05 BPP")
    print("=" * 135)
    print(f"{'Configuração':<45} | {'PPL Médio':<10} | {'Ciência':<8} | {'Código':<8} | {'História':<8} | {'Encic':<8} | {'Diversidade':<12} | {'Status Loop'}")
    print("-" * 135)
    print(f"{'FP16 Original (16.0 bpp)':<45} | {avg_orig:>9.2f}  | {ppl_orig['Ciência']:>8.1f} | {ppl_orig['Código']:>8.1f} | {ppl_orig['História']:>8.1f} | {ppl_orig['Enciclopédia']:>8.1f} | {div_orig*100:>10.1f}% | {'Perfeito (Ref)'}")
    print("-" * 135)
    
    for code, name, _ in configs:
        r = results[code]
        loop_status = "LOOP DETECTADO" if r['div_ratio'] < 0.45 else "100% Sem Loop"
        print(f"{r['name'][:45]:<45} | {r['avg_ppl']:>9.2f}  | {r['ppls']['Ciência']:>8.1f} | {r['ppls']['Código']:>8.1f} | {r['ppls']['História']:>8.1f} | {r['ppls']['Enciclopédia']:>8.1f} | {r['div_ratio']*100:>10.1f}% | {loop_status}")
    print("=" * 135)

if __name__ == '__main__':
    main()
