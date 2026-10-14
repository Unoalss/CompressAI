import sys, os, time
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legion.engine import LegionEngine

def test_deltanet_advancement():
    print("=" * 80)
    print("TESTE DE AVANÇO DO ESTADO RECORRENTE DA DELTANET EM CUDA GRAPHS")
    print("=" * 80)

    engine = LegionEngine()
    device = "cuda"

    prompt = "A inteligência artificial avançada permite"
    prompt_ids = engine.tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()

    with torch.no_grad():
        out = engine.model(input_ids=prompt_ids, use_cache=True)
        past_kv = out.past_key_values

    layer_0 = engine.model.model.layers[0] # DeltaNet
    static_x = torch.randn(1, 1, 2048, dtype=torch.bfloat16, device=device)

    # Verifica os estados atuais
    cs = past_kv.layers[0].conv_states[0]
    rs = past_kv.layers[0].recurrent_states[0]
    print(f"Formato do Conv State:      {cs.shape} (Endereço: {cs.data_ptr()})")
    print(f"Formato do Recurrent State: {rs.shape} (Endereço: {rs.data_ptr()})")

    # Executa 1 passo eager
    with torch.no_grad():
        out1 = layer_0.linear_attn(static_x, cache_params=past_kv)
    rs_after1 = past_kv.layers[0].recurrent_states[0]
    print(f"Após passo 1 Eager: Endereço do Recurrent State: {rs_after1.data_ptr()}")
    print(f"O endereço mudou? {rs.data_ptr() != rs_after1.data_ptr()} (PyTorch alocou novo tensor!)")

if __name__ == "__main__":
    test_deltanet_advancement()
