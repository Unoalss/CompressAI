import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import copy

def quantize_tensor_rvq(weight_tensor, block_size=8, bits_per_code=5, stages=8, n_iter=6):
    """
    Aplica RVQ no tensor de pesos e retorna o tensor reconstruído.
    """
    orig_shape = weight_tensor.shape
    weights_flat = weight_tensor.float().reshape(-1, block_size)
    N = weights_flat.shape[0]
    K = 2 ** bits_per_code # 32 para [5]
    
    current_residual = weights_flat.clone()
    reconstructed = torch.zeros_like(weights_flat)
    
    for _ in range(stages):
        # K-Means rápido
        perm = torch.randperm(N)[:K]
        centroids = current_residual[perm].clone()
        for _ in range(n_iter):
            dists = torch.cdist(current_residual, centroids)
            labels = torch.argmin(dists, dim=1)
            for k in range(K):
                mask = (labels == k)
                if mask.sum() > 0:
                    centroids[k] = current_residual[mask].mean(dim=0)
        
        dists = torch.cdist(current_residual, centroids)
        labels = torch.argmin(dists, dim=1)
        reconstructed += centroids[labels]
        current_residual = weights_flat - reconstructed
        
    return reconstructed.view(orig_shape)

def evaluate_generation(model, tokenizer, prompt, max_new_tokens=25):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=False, # Ganância determinística para comparação exata
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def compute_perplexity(model, tokenizer, text):
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        return torch.exp(loss).item()

def run_experiment():
    print("=" * 75)
    print("EXPERIMENTO EM LLM REAL: DISTILGPT-2 (82M Parâmetros)")
    print("Avaliando Perplexidade e Coerência de Texto Real com Sistema [5]")
    print("=" * 75)
    
    print("\n[1] Carregando modelo base...")
    model_name = "distilbert/distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_orig = AutoModelForCausalLM.from_pretrained(model_name)
    model_orig.eval()
    
    test_text = (
        "Artificial intelligence is transforming science, industry, and daily life. "
        "Researchers are developing models that can reason, write code, and assist humans."
    )
    test_prompt = "The future of artificial intelligence is"
    
    # 1. Original
    ppl_orig = compute_perplexity(model_orig, tokenizer, test_text)
    gen_orig = evaluate_generation(model_orig, tokenizer, test_prompt)
    
    print(f"\n-> MODELO ORIGINAL (FP32/FP16):")
    print(f"   - Perplexidade no texto teste: {ppl_orig:.2f}")
    print(f"   - Geração: \"{gen_orig}\"")
    
    # Testar estágios do sistema [5]: [5]x2, [5]x4, [5]x8 (5 bytes por bloco)
    stages_to_test = [2, 4, 8]
    results = []
    
    for stages in stages_to_test:
        print(f"\n" + "-" * 75)
        bits_pw = (stages * 5) / 8.0
        bytes_block = (stages * 5) / 8.0
        print(f"-> QUANTIZANDO COM [5]x{stages} ({stages*5} bits por bloco de 8 = {bits_pw:.2f} bits/peso)...")
        
        # Copiar modelo
        model_q = copy.deepcopy(model_orig)
        start_t = time.time()
        
        # Quantizar camadas lineares dos blocos Transformer (MLP: c_fc e c_proj)
        total_layers_quantized = 0
        for block in model_q.transformer.h:
            # c_fc.weight em Conv1D do GPT-2 tem formato (in_features, out_features)
            with torch.no_grad():
                block.mlp.c_fc.weight.data = quantize_tensor_rvq(
                    block.mlp.c_fc.weight.data, block_size=8, bits_per_code=5, stages=stages
                )
                block.mlp.c_proj.weight.data = quantize_tensor_rvq(
                    block.mlp.c_proj.weight.data, block_size=8, bits_per_code=5, stages=stages
                )
            total_layers_quantized += 2
            
        elapsed = time.time() - start_t
        print(f"   [Quantizadas {total_layers_quantized} camadas MLP em {elapsed:.1f}s]")
        
        # Avaliar PPL e Geração
        ppl_q = compute_perplexity(model_q, tokenizer, test_text)
        gen_q = evaluate_generation(model_q, tokenizer, test_prompt)
        
        print(f"   - Perplexidade: {ppl_q:.2f} (Original era {ppl_orig:.2f})")
        print(f"   - Geração: \"{gen_q}\"")
        
        results.append({
            'stages': stages,
            'name': f"[5]x{stages}",
            'bits_pw': bits_pw,
            'ppl': ppl_q,
            'gen': gen_q
        })
        
    print("\n" + "=" * 75)
    print("QUADRO RESUMO: DEGRADAÇÃO REAL NA LLM")
    print("=" * 75)
    print(f"{'Configuração':<12} | {'Bits/Peso':<10} | {'Perplexidade':<14} | {'Estado da Resposta'}")
    print("-" * 75)
    print(f"{'Original':<12} | {'16.00 b':<10} | {ppl_orig:<14.2f} | Coerente (Referência)")
    for r in results:
        status = "Preservada" if abs(r['ppl'] - ppl_orig) < 5.0 else ("Degradada" if r['ppl'] < 100 else "Colapso")
        print(f"{r['name']:<12} | {r['bits_pw']:>6.2f} b   | {r['ppl']:<14.2f} | {status}")
    print("=" * 75)

if __name__ == '__main__':
    run_experiment()
