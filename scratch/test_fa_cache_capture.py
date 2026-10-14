import sys, os, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legion.engine import LegionEngine
from transformers.cache_utils import DynamicCache

def test_fa_cache_capture():
    print("=" * 70)
    print("TESTANDO CAPTURA DE FULL ATTENTION COM KV CACHE")
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
    for _ in range(3):
        with torch.no_grad():
            out = engine.model(input_ids=curr, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()
            curr = torch.tensor([[next_tok]], device=device)
    torch.cuda.synchronize()

    # Camada 3 é Full Attention
    layer_3 = engine.model.model.layers[3]
    static_hidden = torch.randn(1, 1, 2048, dtype=torch.bfloat16, device=device)
    pos_ids_3d = torch.zeros(3, 1, 1, dtype=torch.long, device=device) + 15
    pos_emb = engine.model.model.rotary_emb(static_hidden, position_ids=pos_ids_3d)

    # Teste com past_key_values (DynamicCache)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            with torch.no_grad():
                _ = layer_3.self_attn(static_hidden, position_embeddings=pos_emb, past_key_values=past_kv)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g, stream=s):
            out_fa, _ = layer_3.self_attn(static_hidden, position_embeddings=pos_emb, past_key_values=past_kv)
        print("  [SUCESSO] Full Attention COM DynamicCache capturada em CUDA Graph!")
        g.replay()
        torch.cuda.synchronize()
        print("  [SUCESSO] Replay de Full Attention executado sem erros!")
    except Exception as e:
        print(f"  [FALHA] Full Attention capture error: {e}")

if __name__ == "__main__":
    test_fa_cache_capture()
