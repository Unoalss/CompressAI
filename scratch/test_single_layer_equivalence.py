import sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legion.engine import LegionEngine
from vq2.kernels import triton_conv1d_update
import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as mod

def test_layer_equivalence():
    print("=" * 80)
    print("FASE 2(A-B): TESTE DE EQUIVALÊNCIA NA CAMADA 0 DO MODELO REAL")
    print("=" * 80)

    engine = LegionEngine()
    tokenizer = engine.tokenizer

    prompt = "A inteligência artificial avançada demonstra que"
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")

    # 1. Executa prefill para popular o cache de estados convolucionais
    print("\n1. Executando prefill...")
    with torch.no_grad():
        out = engine.model(input_ids=inputs, use_cache=True)
        past_kv = out.past_key_values
        next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()

    curr = torch.tensor([[next_tok]], device="cuda")

    # 2. Vamos testar a Camada 0 (que é linear_attention) em 10 tokens sucessivos de decodificação!
    print("\n2. Testando Camada 0 em 10 passos sucessivos de decodificação...")
    
    orig_conv_fn = mod.causal_conv1d_update

    metrics_history = []

    # Interceptador que executa AMBOS na Camada 0 e compara
    def verifying_conv_fn(hidden_states, conv_state, weight, bias=None, activation=None):
        # Clonamos os estados para executar a referência PyTorch sem alterar o estado de teste
        cs_py = conv_state.clone()
        cs_tri = conv_state.clone()

        # Executa PyTorch
        out_py = orig_conv_fn(hidden_states, cs_py, weight, bias, activation)

        # Executa Triton
        out_tri = triton_conv1d_update(hidden_states, cs_tri, weight, bias)

        # Atualiza o estado oficial com o do Triton (para testar se os próximos passos divergem)
        conv_state.copy_(cs_tri)

        # Medições
        diff_out = torch.max(torch.abs(out_py.float() - out_tri.float())).item()
        cos_out = F.cosine_similarity(out_py.flatten().float(), out_tri.flatten().float(), dim=0).item()
        diff_cs = torch.max(torch.abs(cs_py.float() - cs_tri.float())).item()
        cos_cs = F.cosine_similarity(cs_py.flatten().float(), cs_tri.flatten().float(), dim=0).item()

        metrics_history.append({
            "diff_out": diff_out,
            "cos_out": cos_out,
            "diff_cs": diff_cs,
            "cos_cs": cos_cs
        })

        return out_tri

    # Aplica na Camada 0 (e nas demais se chamado)
    mod.causal_conv1d_update = verifying_conv_fn

    for step in range(1, 11):
        with torch.no_grad():
            out = engine.model(input_ids=curr, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()
            curr = torch.tensor([[next_tok]], device="cuda")

    print(f"\nTotal de convoluções causal_conv1d_update interceptadas: {len(metrics_history)}")
    
    # Amostra das métricas da Camada 0 nos passos
    print("\nMétricas dos passos na Camada 0 (passos 1 a 10):")
    # Como há 30 camadas lineares por token, o índice da Camada 0 a cada token é step * 30
    for s in range(10):
        m = metrics_history[s * 30]
        print(f"  Token {s+1} (Camada 0): MaxDiff Out = {m['diff_out']:.6f} (Cosseno = {m['cos_out']:.8f}) | MaxDiff CS = {m['diff_cs']:.6f}")

    # Médias gerais em todas as 300 chamadas (10 tokens x 30 camadas)
    avg_diff_out = sum(m["diff_out"] for m in metrics_history) / len(metrics_history)
    avg_cos_out = sum(m["cos_out"] for m in metrics_history) / len(metrics_history)
    max_diff_cs = max(m["diff_cs"] for m in metrics_history)

    print("\n" + "=" * 80)
    print("RESUMO DA VALIDAÇÃO EM 300 CHAMADAS (30 CAMADAS x 10 TOKENS):")
    print(f"  Diferença Média Máxima de Saída: {avg_diff_out:.6f}")
    print(f"  Similaridade Cosseno Média:       {avg_cos_out:.8f}")
    print(f"  Diferença Máxima no Estado Causal: {max_diff_cs:.6f}")
    print("=" * 80)

    assert avg_cos_out > 0.9999, "Similaridade cosseno insuficiente!"
    assert max_diff_cs < 1e-4, "Divergência detectada no estado causal!"
    print("\n[SUCESSO] Equivalência causal e numérica comprovada com 100% de estabilidade!")

if __name__ == "__main__":
    test_layer_equivalence()
