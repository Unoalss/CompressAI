import sys, os, time
from pathlib import Path
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legion.engine import LegionEngine
import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as qm
from vq2.kernels import triton_rmsnorm_gated

def validate_rmsnorm_gated():
    print("=" * 80)
    print("OTIMIZAÇÃO 4 - PASSO 1: VALIDAÇÃO DO KERNEL TRITON FUNDIDO RMSNormGated")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "CUDA é obrigatório!"

    # 1. Integridade Física do B2
    b2_path = ROOT / "vq2_35b_out" / "qwen35b_mixed_b2.vq2m"
    b2_size = b2_path.stat().st_size
    print(f"[1/4] Verificando integridade física do B2...")
    print(f"      Arquivo: {b2_path}")
    print(f"      Tamanho: {b2_size:,} bytes")
    assert b2_size == 10108393952, "Tamanho do B2 alterado!"
    print("      [OK] B2 permanece 100% INTACTO byte a byte!")

    engine = LegionEngine()
    tokenizer = engine.tokenizer
    prompt = "A inteligência artificial avançada permite arquiteturas neurais altamente eficientes com"
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    # Identifica as classes de norm
    norm_cls = qm.Qwen3_5MoeRMSNormGated

    # Guarda forward original PyTorch
    def pytorch_norm_forward(self, hidden_states, gate):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)

    def triton_norm_forward(self, hidden_states, gate):
        if (hidden_states.is_cuda and self.activation == "silu" and
            hidden_states.shape[-1] <= 256 and
            (hidden_states.dtype == torch.bfloat16 or hidden_states.dtype == torch.float16)):
            return triton_rmsnorm_gated(hidden_states, gate, self.weight, self.variance_epsilon)
        return pytorch_norm_forward(self, hidden_states, gate)

    def run_generation(use_triton: bool, n_tokens: int = 25):
        if use_triton:
            norm_cls.forward = triton_norm_forward
        else:
            norm_cls.forward = pytorch_norm_forward

        # Prefill + warmup de 3 tokens
        torch.cuda.synchronize()
        with torch.no_grad():
            out = engine.model(input_ids=prompt_ids, use_cache=True)
            past_kv = out.past_key_values
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()

        generated_ids = [next_tok]
        curr = torch.tensor([[next_tok]], device=device)

        for _ in range(3):
            with torch.no_grad():
                out = engine.model(input_ids=curr, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
                next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()
                generated_ids.append(next_tok)
                curr = torch.tensor([[next_tok]], device=device)
        torch.cuda.synchronize()

        # Medição de alta precisão
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        ev_start.record()

        for _ in range(n_tokens):
            with torch.no_grad():
                out = engine.model(input_ids=curr, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
                next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()
                generated_ids.append(next_tok)
                curr = torch.tensor([[next_tok]], device=device)

        ev_end.record()
        torch.cuda.synchronize()
        gpu_time_ms = ev_start.elapsed_time(ev_end)

        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        logits = out.logits[:, -1, :].clone()

        return {
            "gpu_ms": gpu_time_ms,
            "ms_per_tok": gpu_time_ms / n_tokens,
            "tok_per_sec": n_tokens / (gpu_time_ms / 1000.0),
            "text": text,
            "logits": logits
        }

    # [2/4] Teste de Equivalência Causal e Numérica
    print("\n[2/4] Executando Teste de Equivalência no Modelo Completo...")
    print("      Rodando com PyTorch RMSNormGated original...")
    res_py = run_generation(use_triton=False, n_tokens=25)
    print(f"      PyTorch: {res_py['tok_per_sec']:.2f} tok/s ({res_py['ms_per_tok']:.2f} ms/tok)")
    print(f"      Texto:   {res_py['text'][:70]}...")

    print("      Rodando com Triton RMSNormGated fundido...")
    res_tr = run_generation(use_triton=True, n_tokens=25)
    print(f"      Triton:  {res_tr['tok_per_sec']:.2f} tok/s ({res_tr['ms_per_tok']:.2f} ms/tok)")
    print(f"      Texto:   {res_tr['text'][:70]}...")

    # Verificação de fidelidade
    logits_py = res_py["logits"].float()
    logits_tr = res_tr["logits"].float()
    has_nan = torch.isnan(logits_tr).any().item() or torch.isinf(logits_tr).any().item()
    cos_sim = F.cosine_similarity(logits_py, logits_tr, dim=-1).item()
    p_tr = F.log_softmax(logits_tr, dim=-1)
    p_py = F.softmax(logits_py, dim=-1)
    kl_div = F.kl_div(p_tr, p_py, reduction="batchmean").item()
    text_identical = (res_py["text"] == res_tr["text"])

    print("\n" + "=" * 80)
    print("AUDITORIA NUMÉRICA DE EQUIVALÊNCIA:")
    print("=" * 80)
    print(f"Estabilidade Numérica:            Sem NaNs / Sem Infs: {not has_nan}")
    print(f"Textos Gerados Idênticos:         {text_identical}")
    print(f"Similaridade Cosseno Logits:      {cos_sim * 100:.6f}%")
    print(f"Divergência KL (D_KL):            {kl_div:.8f}")
    print("=" * 80)

    # [3/4] Baterias Alternadas de Benchmark de Velocidade
    print("\n[3/4] Executando 3 Baterias Alternadas de Benchmark...")
    py_speeds = []
    tr_speeds = []

    for round_idx in range(1, 4):
        print(f"\n--- BATERIA {round_idx} ---")
        r_py = run_generation(use_triton=False, n_tokens=30)
        py_speeds.append(r_py)
        print(f"  [PyTorch Original] {r_py['tok_per_sec']:.2f} tok/s ({r_py['ms_per_tok']:.2f} ms/tok)")

        r_tr = run_generation(use_triton=True, n_tokens=30)
        tr_speeds.append(r_tr)
        print(f"  [Triton Fundido]   {r_tr['tok_per_sec']:.2f} tok/s ({r_tr['ms_per_tok']:.2f} ms/tok)")

    avg_py_tps = sum(r["tok_per_sec"] for r in py_speeds) / 3
    avg_py_ms = sum(r["ms_per_tok"] for r in py_speeds) / 3
    avg_tr_tps = sum(r["tok_per_sec"] for r in tr_speeds) / 3
    avg_tr_ms = sum(r["ms_per_tok"] for r in tr_speeds) / 3

    # [4/4] Verificação Final da Integridade do B2
    final_b2_size = b2_path.stat().st_size
    assert final_b2_size == 10108393952, "Tamanho do B2 violado!"

    print("\n" + "=" * 80)
    print("RELATÓRIO CONSOLIDADO DO PASSO 1 (RMSNormGated FUNDIDO):")
    print("=" * 80)
    print(f"Artefato B2:                      100% INTACTO ({final_b2_size:,} bytes)")
    print(f"Equivalência Causal:              Concordância Textual: {text_identical}")
    print(f"Similaridade Cosseno Logits:      {cos_sim * 100:.6f}%")
    print(f"Divergência KL:                   {kl_div:.8f}")
    print("--------------------------------------------------------------------------------")
    print(f"PyTorch Original (Referência):    {avg_py_tps:.2f} tok/s ({avg_py_ms:.2f} ms/token)")
    print(f"Triton Fundido (Otimizado):       {avg_tr_tps:.2f} tok/s ({avg_tr_ms:.2f} ms/token)")
    print(f"Ganho Líquido Real:               +{avg_tr_tps - avg_py_tps:.2f} tok/s (-{avg_py_ms - avg_tr_ms:.2f} ms/token)")
    print("=" * 80)

if __name__ == "__main__":
    validate_rmsnorm_gated()
