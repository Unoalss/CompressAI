import sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legion.engine import LegionEngine
from vq2 import vq

def test_hadamard_equivalence():
    print("=" * 80)
    print("OTIMIZAÇÃO 2 - FASE C: VALIDAÇÃO DE EQUIVALÊNCIA NO MODELO REAL")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda"

    # 1. Carrega instâncias de rotação e pré-computa as matrizes fundidas M
    rot_2048 = vq.get_rot("moe_expert_2048", 2048, device)
    rot_512 = vq.get_rot("moe_expert_512", 512, device)

    # Pré-computa matrizes de base canônica
    I_2048 = torch.eye(2048, device=device, dtype=torch.bfloat16)
    M_2048 = rot_2048.fwd(I_2048).contiguous()

    I_512 = torch.eye(512, device=device, dtype=torch.bfloat16)
    M_512 = rot_512.fwd(I_512).contiguous()

    print("Matrizes fundidas pré-computadas:")
    print(f"  M_2048: {M_2048.shape} ({M_2048.element_size() * M_2048.nelement() / (1024*1024):.2f} MB)")
    print(f"  M_512:  {M_512.shape} ({M_512.element_size() * M_512.nelement() / (1024*1024):.2f} MB)")

    # 2. Inicializa o motor neural
    engine = LegionEngine()
    tokenizer = engine.tokenizer

    test_prompt = "A inteligência artificial avançada demonstra que o raciocínio estruturado"
    prompt_ids = tokenizer(test_prompt, return_tensors="pt")["input_ids"].to(device)

    # Função para executar N tokens com um método de rotação
    def run_generation(mode_label, use_fused, n_tokens=25):
        orig_fwd_2048 = rot_2048.fwd
        orig_fwd_512 = rot_512.fwd

        if use_fused:
            def fused_fwd_2048(x):
                if x.dim() == 1:
                    return torch.matmul(x, M_2048)
                return torch.matmul(x, M_2048)

            def fused_fwd_512(act):
                return torch.matmul(act, M_512)

            rot_2048.fwd = fused_fwd_2048
            rot_512.fwd = fused_fwd_512

        torch.cuda.synchronize()
        with torch.no_grad():
            out = engine.model(input_ids=prompt_ids, use_cache=True)
            past_kv = out.past_key_values
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()

        generated_ids = [next_tok]
        curr = torch.tensor([[next_tok]], device=device)

        # Warmup de 3 tokens
        for _ in range(3):
            with torch.no_grad():
                out = engine.model(input_ids=curr, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
                next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()
                generated_ids.append(next_tok)
                curr = torch.tensor([[next_tok]], device=device)
        torch.cuda.synchronize()

        # Medição precisa
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

        # Restaura
        rot_2048.fwd = orig_fwd_2048
        rot_512.fwd = orig_fwd_512

        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        last_logits = out.logits[:, -1, :].clone()

        return {
            "mode": mode_label,
            "gpu_ms": gpu_time_ms,
            "ms_per_tok": gpu_time_ms / n_tokens,
            "tok_per_sec": n_tokens / (gpu_time_ms / 1000.0),
            "text": text,
            "logits": last_logits
        }

    print("\nExecutando Geração com Rotação Original (4 passos / 320 kernels)...")
    res_orig = run_generation("Original (4 passos)", use_fused=False, n_tokens=25)
    print(f"  Velocidade: {res_orig['tok_per_sec']:.2f} tok/s ({res_orig['ms_per_tok']:.2f} ms/tok)")
    print(f"  Texto: {res_orig['text'][:70]}...")

    print("\nExecutando Geração com Rotação Fundida Pré-Computada (1 passo / 80 kernels)...")
    res_fused = run_generation("Fundida (1 passo)", use_fused=True, n_tokens=25)
    print(f"  Velocidade: {res_fused['tok_per_sec']:.2f} tok/s ({res_fused['ms_per_tok']:.2f} ms/tok)")
    print(f"  Texto: {res_fused['text'][:70]}...")

    # Comparações de fidelidade
    logits_orig = res_orig["logits"].float()
    logits_fused = res_fused["logits"].float()

    cos_sim = F.cosine_similarity(logits_orig, logits_fused, dim=-1).item()
    p_fused = F.log_softmax(logits_fused, dim=-1)
    p_orig = F.softmax(logits_orig, dim=-1)
    kl_div = F.kl_div(p_fused, p_orig, reduction="batchmean").item()

    has_nan = torch.isnan(logits_fused).any().item() or torch.isinf(logits_fused).any().item()

    print("\n" + "=" * 80)
    print("RELATÓRIO DE EQUIVALÊNCIA DA OTIMIZAÇÃO 2:")
    print("=" * 80)
    print(f"Estabilidade Numérica:            Sem NaNs / Sem Infs: {not has_nan}")
    print(f"Concordância dos Logits:")
    print(f"  - Similaridade Cosseno:         {cos_sim * 100:.4f}%")
    print(f"  - Divergência KL:               {kl_div:.6f}")
    print(f"Comparativo de Performance:")
    print(f"  - Rotação Original:             {res_orig['tok_per_sec']:.2f} tok/s ({res_orig['ms_per_tok']:.2f} ms/token)")
    print(f"  - Rotação Fundida:              {res_fused['tok_per_sec']:.2f} tok/s ({res_fused['ms_per_tok']:.2f} ms/token)")
    print(f"  - Ganho Líquido:                +{res_fused['tok_per_sec'] - res_orig['tok_per_sec']:.2f} tok/s (-{res_orig['ms_per_tok'] - res_fused['ms_per_tok']:.2f} ms/token)")
    print(f"  - Textos Gerados Idênticos:     {res_orig['text'] == res_fused['text']}")
    print("=" * 80)

if __name__ == "__main__":
    test_hadamard_equivalence()
