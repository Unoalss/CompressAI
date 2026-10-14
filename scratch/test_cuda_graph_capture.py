import sys, os, time
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legion.engine import LegionEngine
from transformers.cache_utils import DynamicCache

def test_capture():
    print("=" * 70)
    print("DIAGNÓSTICO EXPERIMENTAL DE CAPTURA DE CUDA GRAPHS NO LEGION")
    print("=" * 70)

    engine = LegionEngine()
    device = "cuda"

    prompt = "A inteligência artificial avançada permite"
    prompt_ids = engine.tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()

    with torch.no_grad():
        out = engine.model(input_ids=prompt_ids, use_cache=True)
        past_kv = out.past_key_values
        next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()

    curr = torch.tensor([[next_tok]], device=device)
    
    # Executa 3 tokens normais para estabilizar caches
    for _ in range(3):
        with torch.no_grad():
            out = engine.model(input_ids=curr, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()
            curr = torch.tensor([[next_tok]], device=device)
    torch.cuda.synchronize()

    # TESTE 1: Captura de uma camada DeltaNet isolada (Atenção Linear)
    print("\n--- TESTE 1: Captura de DeltaNet Linear Attention (Camada 0) ---")
    layer_0 = engine.model.model.layers[0] # DeltaNet
    static_hidden = torch.randn(1, 1, 2048, dtype=torch.bfloat16, device=device)
    
    # Warmup
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            with torch.no_grad():
                _ = layer_0.linear_attn(static_hidden, cache_params=past_kv)
    torch.cuda.current_stream().wait_stream(s)

    # Capture
    g_deltanet = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g_deltanet, stream=s):
            out_delta = layer_0.linear_attn(static_hidden, cache_params=past_kv)
        print("  [SUCESSO] DeltaNet capturada em CUDA Graph perfeitamente!")
        print(f"  Número de nós no grafo DeltaNet: {g_deltanet.pool()}")
        # Replay test
        g_deltanet.replay()
        torch.cuda.synchronize()
        print("  [SUCESSO] Replay de DeltaNet executado sem erros!")
    except Exception as e:
        print(f"  [FALHA] DeltaNet capture error: {e}")

    # TESTE 2: Captura de uma camada Full Attention isolada (Camada 3)
    print("\n--- TESTE 2: Captura de Full Attention (Camada 3) ---")
    layer_fa = engine.model.model.layers[3] # Full Attention
    pos_ids_3d = torch.zeros(3, 1, 1, dtype=torch.long, device=device) + 10
    pos_emb = engine.model.model.rotary_emb(static_hidden, position_ids=pos_ids_3d)
    
    s2 = torch.cuda.Stream()
    s2.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s2):
        for _ in range(3):
            with torch.no_grad():
                # Eager FA sem DynamicCache update para testar se os kernels de atenção capturam
                _ = layer_fa.self_attn(static_hidden, position_embeddings=pos_emb, past_key_values=None)
    torch.cuda.current_stream().wait_stream(s2)

    g_fa = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g_fa, stream=s2):
            out_fa, _ = layer_fa.self_attn(static_hidden, position_embeddings=pos_emb, past_key_values=None)
        print("  [SUCESSO] Full Attention capturada em CUDA Graph perfeitamente!")
        g_fa.replay()
        torch.cuda.synchronize()
        print("  [SUCESSO] Replay de Full Attention executado sem erros!")
    except Exception as e:
        print(f"  [FALHA] Full Attention capture error: {e}")

    # TESTE 3: Avaliação de PagedMoEExpert durante captura
    print("\n--- TESTE 3: Captura de PagedMoEExpert (Camada 0) ---")
    paged_expert = layer_0.mlp.experts
    dummy_topk_idx = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], device=device)
    dummy_topk_w = torch.ones(1, 8, device=device) / 8.0

    s3 = torch.cuda.Stream()
    s3.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s3):
        for _ in range(3):
            with torch.no_grad():
                _ = paged_expert(static_hidden, dummy_topk_idx, dummy_topk_w)
    torch.cuda.current_stream().wait_stream(s3)

    g_moe = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g_moe, stream=s3):
            out_moe = paged_expert(static_hidden, dummy_topk_idx, dummy_topk_w)
        print("  [SUCESSO] PagedMoEExpert capturado em CUDA Graph!")
    except Exception as e:
        print(f"  [ESPERADO/FALHA] PagedMoEExpert capture error: {e}")

    print("\n" + "=" * 70)
    print("DIAGNÓSTICO CONCLUÍDO")
    print("=" * 70)

if __name__ == "__main__":
    test_capture()
