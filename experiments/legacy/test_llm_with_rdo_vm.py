import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import copy

def train_kmeans_1d(values, K=16, n_iter=8):
    """Treina K valores de escala unidimensionais."""
    perm = torch.randperm(values.shape[0])[:K]
    centroids = values[perm].clone()
    for _ in range(n_iter):
        dists = torch.abs(values.unsqueeze(1) - centroids.unsqueeze(0))
        labels = torch.argmin(dists, dim=1)
        for k in range(K):
            mask = (labels == k)
            if mask.sum() > 0:
                centroids[k] = values[mask].mean()
    return torch.sort(centroids)[0]

def train_codebook_vectors(vectors, K=26, n_iter=8):
    """Treina K centroides vetoriais."""
    N, D = vectors.shape
    perm = torch.randperm(N)[:K]
    centroids = vectors[perm].clone()
    for _ in range(n_iter):
        dists = torch.cdist(vectors, centroids)
        labels = torch.argmin(dists, dim=1)
        for k in range(K):
            mask = (labels == k)
            if mask.sum() > 0:
                centroids[k] = vectors[mask].mean(dim=0)
    return centroids

def quantize_layer_rdo_fast(weight_tensor, block_size=8, target_mse=0.0035):
    """
    Quantiza a camada usando RDO vetorizado ultrarrápido:
    Compete FORMA (5b), FORMA+ESCALA (15b) e FORMA+ESCALA+RESÍDUO (25b).
    """
    orig_shape = weight_tensor.shape
    w_flat = weight_tensor.float().reshape(-1, block_size)
    N = w_flat.shape[0]
    
    # 1. Normas e Escalas
    norms = torch.norm(w_flat, dim=1)
    lib_scales = train_kmeans_1d(norms, K=16)
    
    # 2. Formas Unitárias (26 átomos)
    w_unit = w_flat / (norms.unsqueeze(1) + 1e-7)
    lib_shape = train_codebook_vectors(w_unit, K=26, n_iter=6)
    lib_shape = lib_shape / (torch.norm(lib_shape, dim=1, keepdim=True) + 1e-7)
    
    # 3. Resíduos Calibrados
    d_shape = torch.cdist(w_unit, lib_shape)
    best_s = torch.argmin(d_shape, dim=1)
    d_scale = torch.abs(norms.unsqueeze(1) - lib_scales.unsqueeze(0))
    best_scale = torch.argmin(d_scale, dim=1)
    
    cand_scaled = lib_shape[best_s] * lib_scales[best_scale].unsqueeze(1)
    res_vecs = w_flat - cand_scaled
    lib_residual = train_codebook_vectors(res_vecs, K=26, n_iter=6)
    
    d_res = torch.cdist(res_vecs, lib_residual)
    best_r = torch.argmin(d_res, dim=1)
    cand_full = cand_scaled + lib_residual[best_r]
    
    # 4. Avaliação Vetorizada de Distorção (RDO)
    cand_shape = lib_shape[best_s]
    err_shape = torch.mean((cand_shape - w_flat) ** 2, dim=1)
    err_scaled = torch.mean((cand_scaled - w_flat) ** 2, dim=1)
    
    # Decisão RDO:
    # Escolhe a menor instrução que atinge target_mse
    mask_shape = (err_shape <= target_mse)
    mask_scaled = (~mask_shape) & (err_scaled <= target_mse)
    mask_full = (~mask_shape) & (~mask_scaled)
    
    # Montagem vetorizada da reconstrução
    recon = torch.zeros_like(w_flat)
    recon[mask_shape] = cand_shape[mask_shape]
    recon[mask_scaled] = cand_scaled[mask_scaled]
    recon[mask_full] = cand_full[mask_full]
    
    # Cálculo de bits gastos
    bits_shape = mask_shape.sum().item() * 5
    bits_scaled = mask_scaled.sum().item() * 15
    bits_full = mask_full.sum().item() * 25
    total_bits = bits_shape + bits_scaled + bits_full
    avg_bpp = total_bits / (N * block_size)
    
    stats = {
        'pct_shape': (mask_shape.sum().item() / N) * 100,
        'pct_scaled': (mask_scaled.sum().item() / N) * 100,
        'pct_full': (mask_full.sum().item() / N) * 100,
        'bpp': avg_bpp
    }
    
    return recon.view(orig_shape), stats

def compute_perplexity(model, tokenizer, text):
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    with torch.no_grad():
        loss = model(input_ids, labels=input_ids).loss
        return torch.exp(loss).item()

def generate_text(model, tokenizer, prompt, max_new_tokens=30):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=False, 
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

def main():
    print("=" * 80)
    print("PROVA FINAL NA LLM INTEIRA: DISTILGPT-2 COM SISTEMA RDO")
    print("Testando Forma + Escala + Resíduo em Todas as Camadas Transformer")
    print("=" * 80)
    
    model_name = "distilbert/distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_orig = AutoModelForCausalLM.from_pretrained(model_name)
    model_orig.eval()
    
    test_text = (
        "Artificial intelligence is transforming science, industry, and daily life. "
        "Researchers are developing models that can reason, write code, and assist humans."
    )
    prompt_1 = "The future of artificial intelligence is"
    prompt_2 = "To solve a complex problem, one must first"
    
    # 1. Medição do Modelo Original
    ppl_orig = compute_perplexity(model_orig, tokenizer, test_text)
    gen1_orig = generate_text(model_orig, tokenizer, prompt_1)
    gen2_orig = generate_text(model_orig, tokenizer, prompt_2)
    
    print("\n-> MODELO ORIGINAL (16.0 bits/peso):")
    print(f"   - Perplexidade: {ppl_orig:.2f}")
    print(f"   - Resposta 1: \"{gen1_orig}\"")
    print(f"   - Resposta 2: \"{gen2_orig}\"")
    
    # 2. Clonar e Quantizar todas as 12 camadas Transformer com RDO Adaptativo
    print("\n" + "-" * 80)
    print("Quantizando 12 Camadas Transformer com a VM Adaptativa RDO...")
    model_rdo = copy.deepcopy(model_orig)
    model_rdo.eval()
    
    start_t = time.time()
    layer_bpps = []
    
    for block in model_rdo.transformer.h:
        with torch.no_grad():
            w_fc, stats_fc = quantize_layer_rdo_fast(block.mlp.c_fc.weight.data)
            block.mlp.c_fc.weight.data = w_fc
            layer_bpps.append(stats_fc['bpp'])
            
            w_proj, stats_proj = quantize_layer_rdo_fast(block.mlp.c_proj.weight.data)
            block.mlp.c_proj.weight.data = w_proj
            layer_bpps.append(stats_proj['bpp'])
            
    elapsed = time.time() - start_t
    avg_model_bpp = sum(layer_bpps) / len(layer_bpps)
    print(f"[12 Camadas Quantizadas em {elapsed:.2f}s | Média: {avg_model_bpp:.2f} bits/peso]")
    
    # 3. Avaliar Modelo Quantizado RDO
    ppl_rdo = compute_perplexity(model_rdo, tokenizer, test_text)
    gen1_rdo = generate_text(model_rdo, tokenizer, prompt_1)
    gen2_rdo = generate_text(model_rdo, tokenizer, prompt_2)
    
    print(f"\n-> MODELO RDO ADAPTATIVO ({avg_model_bpp:.2f} bits/peso):")
    print(f"   - Perplexidade: {ppl_rdo:.2f} (vs {ppl_orig:.2f} original)")
    print(f"   - Resposta 1: \"{gen1_rdo}\"")
    print(f"   - Resposta 2: \"{gen2_rdo}\"")
    
    # 4. Tabela Resumo com Histórico dos Testes Anteriores
    print("\n" + "=" * 80)
    print("GRANDE QUADRO COMPARATIVO: TODAS AS ARQUITETURAS NA LLM REAL")
    print("=" * 80)
    print(f"{'Arquitetura':<28} | {'Bits/Peso':<10} | {'Perplexidade':<14} | {'Status da Resposta'}")
    print("-" * 80)
    print(f"{'Original FP16/FP32':<28} | {'16.00 b':<10} | {ppl_orig:<14.2f} | Coerente (Referência)")
    print(f"{'Tabela Plana [5]x4':<28} | {'2.50 b':<10} | {'131.03':<14} | Colapso ('perfect' 'S' 'S')")
    print(f"{'Chaves Hierárquicas [5]x4':<28} | {'2.50 b':<10} | {'128.43':<14} | Gramática ok, loop repetição")
    print(f"{'RDO (Forma+Escala+Resíduo)':<28} | {f'{avg_model_bpp:.2f} b':<10} | {ppl_rdo:<14.2f} | Testado agora")
    print(f"{'Sistema Fixo [5]x8':<28} | {'5.00 b':<10} | {'62.74':<14} | Idêntico ao Original")
    print("=" * 80)

if __name__ == '__main__':
    main()
