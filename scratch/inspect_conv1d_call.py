import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legion.engine import LegionEngine

def test():
    print("Inicializando LegionEngine...")
    engine = LegionEngine()
    tokenizer = engine.tokenizer

    prompt = "Teste de verificação causal."
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")

    intercepted = []

    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as mod
    orig_fn = mod.causal_conv1d_update

    def spy_fn(hidden_states, conv_state, weight, bias=None, activation=None):
        if len(intercepted) < 3:
            intercepted.append({
                "hs_shape": list(hidden_states.shape),
                "hs_stride": list(hidden_states.stride()),
                "hs_dtype": str(hidden_states.dtype),
                "cs_shape": list(conv_state.shape),
                "cs_stride": list(conv_state.stride()),
                "cs_dtype": str(conv_state.dtype),
                "w_shape": list(weight.shape),
                "w_stride": list(weight.stride()),
                "w_dtype": str(weight.dtype),
                "bias": bias is not None,
                "activation": activation,
                "is_cuda": hidden_states.is_cuda
            })
        return orig_fn(hidden_states, conv_state, weight, bias, activation)

    mod.causal_conv1d_update = spy_fn

    print("Executando prefill...")
    with torch.no_grad():
        out = engine.model(input_ids=inputs, use_cache=True)
        past_kv = out.past_key_values
        next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()

    print(f"Prefill concluído. Interceptações no prefill: {len(intercepted)}")

    print("Executando 1 token de decodificação...")
    curr = torch.tensor([[next_tok]], device="cuda")
    with torch.no_grad():
        out = engine.model(input_ids=curr, past_key_values=past_kv, use_cache=True)

    print(f"Decodificação concluída! Interceptações no decode: {len(intercepted)}")
    for i, item in enumerate(intercepted):
        print(f"Chamada {i+1}: {item}")

if __name__ == "__main__":
    test()
