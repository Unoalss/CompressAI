import sys, time
from pathlib import Path
from collections import OrderedDict
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legion.engine import LegionEngine
from vq2.kernels import vq_moe_parallel_gate_up, vq_moe_parallel_down

def test_optimized_paged_expert():
    print("=" * 80)
    print("OTIMIZAÇÃO 3: TESTE DE OTIMIZAÇÕES NO RUNTIME DO PagedMoEExpert")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda"

    engine = LegionEngine()
    tokenizer = engine.tokenizer

    prompt = "A inteligência artificial avançada permite arquiteturas neurais altamente eficientes com"
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    # 1. Armazena o forward original para comparação A/B
    orig_forward = engine.layers[0].__class__.forward
    orig_get_slot = engine.layers[0].__class__._get_slot

    # 2. Vamos definir o forward otimizado
    def optimized_forward(self, hidden_states, topk_idx, topk_weights):
        T = hidden_states.shape[0]
        K = topk_idx.shape[-1]
        if T == 1:
            topk_ids_list = topk_idx.view(-1).tolist()
            topk_w = topk_weights.view(-1).bfloat16()

            # Caso haja especialistas 3b promovidos
            if self.promoted_3b and any(e in self.promoted_3b for e in topk_ids_list):
                promoted_out = torch.zeros(1, 2048, device=self.device, dtype=torch.bfloat16)
                regular_slots = []
                reg_mask = []
                xt = hidden_states.bfloat16()
                for k_idx, e_id in enumerate(topk_ids_list):
                    w_k = topk_w[k_idx]
                    if e_id in self.promoted_3b:
                        W_gu = self.promoted_3b[e_id]["gu"]
                        W_dn = self.promoted_3b[e_id]["dn"]
                        y_gu = F.linear(xt, W_gu)
                        g, u = y_gu[:, :512], y_gu[:, 512:]
                        act = (F.silu(g.float()) * u.float()).bfloat16()
                        promoted_out += F.linear(act, W_dn) * w_k
                        reg_mask.append(False)
                    else:
                        regular_slots.append(self._get_slot(e_id))
                        reg_mask.append(True)

                if regular_slots:
                    K_reg = len(regular_slots)
                    # Otimização 1: Reutiliza self.active_slots pré-alocado
                    for i, s in enumerate(regular_slots):
                        self.active_slots[i] = s
                    slots_t = self.active_slots[:K_reg]

                    # Otimização 2: Slice direto do tensor de pesos (sem torch.stack)
                    reg_w = topk_w[torch.tensor(reg_mask, device=self.device, dtype=torch.bool)]

                    xr = self.rot_2048.fwd(xt.view(-1))
                    y_gu = self.y_gu[:K_reg]
                    vq_moe_parallel_gate_up(
                        xr, slots_t, self.pool_gu_idx, self.cb16, self.pool_gu_scale, y_gu,
                        OUT=1024, NG=256, rs=self.RS_2048
                    )
                    gate = y_gu[:, :512]
                    up = y_gu[:, 512:]
                    act = (F.silu(gate.float()) * up.float()).bfloat16()
                    
                    # Otimização 3: Sem cópia redundante act_r.copy_
                    act_r = self.rot_512.fwd(act)
                    y_down = self.y_down[:K_reg]
                    vq_moe_parallel_down(
                        act_r, slots_t, self.pool_dn_idx, self.cb16, self.pool_dn_scale, y_down,
                        OUT=2048, NG=64, rs=self.RS_512
                    )
                    reg_out = (y_down * reg_w[:, None]).sum(dim=0, keepdim=True)
                    return promoted_out + reg_out
                return promoted_out

            # Caso 100% especialistas comuns (a maioria esmagadora das camadas)
            # Otimização 1: Batch resolution e preenchimento in-place do active_slots
            for i, e_id in enumerate(topk_ids_list):
                self.active_slots[i] = self._get_slot(e_id)
            slots_t = self.active_slots[:K]

            xr = self.rot_2048.fwd(hidden_states.view(-1).bfloat16())
            y_gu = self.y_gu[:K]
            vq_moe_parallel_gate_up(
                xr, slots_t, self.pool_gu_idx, self.cb16, self.pool_gu_scale, y_gu,
                OUT=1024, NG=256, rs=self.RS_2048
            )
            gate = y_gu[:, :512]
            up = y_gu[:, 512:]
            act = (F.silu(gate.float()) * up.float()).bfloat16()
            
            # Otimização 3: Sem cópia redundante act_r.copy_
            act_r = self.rot_512.fwd(act)
            y_down = self.y_down[:K]
            vq_moe_parallel_down(
                act_r, slots_t, self.pool_dn_idx, self.cb16, self.pool_dn_scale, y_down,
                OUT=2048, NG=64, rs=self.RS_512
            )
            out = (y_down * topk_w[:, None]).sum(dim=0, keepdim=True)
            return out
        else:
            return orig_forward(self, hidden_states, topk_idx, topk_weights)

    def run_generation(use_opt, n_tokens=25):
        if use_opt:
            engine.layers[0].__class__.forward = optimized_forward
        else:
            engine.layers[0].__class__.forward = orig_forward

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
        return gpu_time_ms / n_tokens, n_tokens / (gpu_time_ms / 1000.0), text, out.logits[:, -1, :]

    print("\n[1/2] Testando Runtime MoE Original...")
    ms_orig, tps_orig, text_orig, logits_orig = run_generation(use_opt=False, n_tokens=25)
    print(f"      Original: {ms_orig:.2f} ms/tok ({tps_orig:.2f} tok/s)")
    print(f"      Texto:    {text_orig[:60]}...")

    print("\n[2/2] Testando Runtime MoE Otimizado...")
    ms_opt, tps_opt, text_opt, logits_opt = run_generation(use_opt=True, n_tokens=25)
    print(f"      Otimizado:{ms_opt:.2f} ms/tok ({tps_opt:.2f} tok/s)")
    print(f"      Texto:    {text_opt[:60]}...")

    # Verificações de fidelidade
    cos_sim = F.cosine_similarity(logits_orig.float(), logits_opt.float(), dim=-1).item()
    p_opt = F.log_softmax(logits_opt.float(), dim=-1)
    p_orig = F.softmax(logits_orig.float(), dim=-1)
    kl_div = F.kl_div(p_opt, p_orig, reduction="batchmean").item()

    print("\n" + "=" * 80)
    print("RELATÓRIO DE EQUIVALÊNCIA E PERFORMANCE:")
    print("=" * 80)
    print(f"Similaridade Cosseno Logits:   {cos_sim * 100:.4f}%")
    print(f"Divergência KL:                {kl_div:.6f}")
    print(f"Textos Idênticos:              {text_orig == text_opt}")
    print(f"Original:                      {ms_orig:.2f} ms/tok ({tps_orig:.2f} tok/s)")
    print(f"Otimizado:                     {ms_opt:.2f} ms/tok ({tps_opt:.2f} tok/s)")
    print(f"Ganho Líquido:                 +{tps_opt - tps_orig:.2f} tok/s (-{ms_orig - ms_opt:.2f} ms/tok)")
    print("=" * 80)

if __name__ == "__main__":
    test_optimized_paged_expert()
