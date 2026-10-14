"""
scratch/audit_a_b1_b2_logits.py
Auditoria Causal e Incremental de Logits:
Compara 4 configurações estritas nos mesmos prompts não vistos:
  1. Ref: HF Original BF16 nos 16 especialistas promovidos
  2. Config A: Baseline 100% 2 bits (qwen35b_full.vq2m)
  3. Config B1: Passagem 1 (6 âncoras da L39 em 3 bits)
  4. Config B2: Passagem 2 (6 âncoras L39 + 10 novos candidatos de L31-L38 em 3 bits)

Métricas medidas:
  - Divergência KL em relação ao BF16 de referência (KL_A, KL_B1, KL_B2)
  - Redução incremental de KL: (A -> B1) e (B1 -> B2)
  - Similaridade cosseno dos logits (Cos_A, Cos_B1, Cos_B2)
  - Concordância Top-1 com o token previsto pelo BF16
  - Teste de geração greedy (50 tokens) sem penalidade de repetição para avaliar estabilidade textual.
"""
import os, sys, math, json, time
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F
from safetensors import safe_open

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vq2 import vq
from vq2.kernels import vq_moe_parallel_gate_up, vq_moe_parallel_down
from legion.engine import LegionEngine

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_EXPERTS = {
    39: [133, 56, 24, 234, 204, 123], # B1 âncoras
    38: [209, 67],                    # B2 candidatos
    37: [186, 19],
    36: [101],
    34: [203],
    33: [127],
    32: [165],
    31: [35, 162],
}

# Prompts totalmente inéditos (fora dos domínios da calibração)
UNSEEN_PROMPTS = [
    {
        "domain": "Termodinâmica Química",
        "prompt": "Na termodinâmica química, a energia livre de Gibbs é uma função de estado que determina a espontaneidade de uma reação a pressão e"
    },
    {
        "domain": "Lógica Matemática",
        "prompt": "Em lógica proposicional, o princípio do terceiro excluído estabelece que para qualquer proposição p, a disjunção de p e sua negação é"
    },
    {
        "domain": "Sistemas Distribuídos",
        "prompt": "Em arquiteturas de microsserviços distribuídos, o teorema CAP afirma que é impossível fornecer simultaneamente consistência, disponibilidade e"
    },
    {
        "domain": "Astronomia / Espaço",
        "prompt": "A missão Apollo 11 da NASA alcançou a superfície lunar em julho de 1969, quando o módulo Eagle pousou no"
    },
    {
        "domain": "Neurociência Cognitiva",
        "prompt": "Na neurociência cognitiva, as sinapses químicas utilizam mensageiros moleculares chamados neurotransmissores que atravessam a fenda sináptica para se ligarem aos"
    },
    {
        "domain": "Macroeconomia",
        "prompt": "No modelo macroeconômico IS-LM, o equilíbrio geral é alcançado na interseção entre o mercado de bens e o mercado de"
    }
]

def load_promoted_original_weights():
    """Carrega os tensores originais BF16 para os 16 especialistas das 8 camadas."""
    index_path = ROOT / "modelo_35b" / "model.safetensors.index.json"
    weight_map = json.load(open(index_path))["weight_map"]
    
    orig_weights = defaultdict(dict)
    for ln, exp_list in TARGET_EXPERTS.items():
        k_gu = f"model.language_model.layers.{ln}.mlp.experts.gate_up_proj"
        k_dn = f"model.language_model.layers.{ln}.mlp.experts.down_proj"
        shard_gu = ROOT / "modelo_35b" / weight_map[k_gu]
        shard_dn = ROOT / "modelo_35b" / weight_map[k_dn]

        with safe_open(str(shard_gu), framework="pt", device="cpu") as f_gu, \
             safe_open(str(shard_dn), framework="pt", device="cpu") as f_dn:
            gu_slice = f_gu.get_slice(k_gu)
            dn_slice = f_dn.get_slice(k_dn)
            for e_id in exp_list:
                orig_weights[ln][e_id] = {
                    "gu": gu_slice[e_id].to(DEVICE, torch.bfloat16),
                    "dn": dn_slice[e_id].to(DEVICE, torch.bfloat16)
                }
    return orig_weights

def build_3bit_weights(orig_weights):
    """Reconstrói os pesos em 3 bits (cfg=(16, 8)) para os especialistas promovidos."""
    cbs_data = torch.load(ROOT / "vq2_out" / "codebooks.pt", weights_only=False)
    cbs = vq.Codebooks({k: v.to(DEVICE, torch.bfloat16) for k, v in cbs_data.items()})
    rot_2048 = vq.get_rot("moe_expert_2048", 2048, DEVICE)
    rot_512 = vq.get_rot("moe_expert_512", 512, DEVICE)

    weights_3b = defaultdict(dict)
    for ln, exps in orig_weights.items():
        for e_id, w in exps.items():
            orig_gu = w["gu"]
            orig_dn = w["dn"]

            # Gate-Up 3 bits
            W_gu_rot = rot_2048.fwd(orig_gu)
            s_gu = W_gu_rot.float().norm(dim=1, keepdim=True) / math.sqrt(2048)
            u_gu = (W_gu_rot.float() / s_gu.clamp_min(1e-6)).reshape(-1, 8).to(DEVICE, torch.bfloat16)
            _, rec_gu = vq.encode(u_gu, (16, 8), cbs)
            Q_gu = (rec_gu.view(1024, 2048) * s_gu.to(torch.bfloat16)).to(torch.bfloat16)
            W_rec_gu = rot_2048.inv(Q_gu)

            # Down 3 bits
            W_dn_rot = rot_512.fwd(orig_dn)
            s_dn = W_dn_rot.float().norm(dim=1, keepdim=True) / math.sqrt(512)
            u_dn = (W_dn_rot.float() / s_dn.clamp_min(1e-6)).reshape(-1, 8).to(DEVICE, torch.bfloat16)
            _, rec_dn = vq.encode(u_dn, (16, 8), cbs)
            Q_dn = (rec_dn.view(2048, 512) * s_dn.to(torch.bfloat16)).to(torch.bfloat16)
            W_rec_dn = rot_512.inv(Q_dn)

            weights_3b[ln][e_id] = {"gu": W_rec_gu, "dn": W_rec_dn}
    return weights_3b

def create_layer_forward_hook(layer_module, active_custom_weights):
    """Cria um forward hook para substituir a inferência de especialistas específicos."""
    orig_fwd = layer_module.forward

    def custom_forward(hidden_states, topk_idx, topk_weights):
        T = hidden_states.shape[0]
        K = topk_idx.shape[-1]
        outs = []
        for t in range(T):
            ids_list = topk_idx[t].view(-1).tolist()
            w_t = topk_weights[t].view(-1).bfloat16()
            xt = hidden_states[t:t+1].bfloat16()

            token_out = torch.zeros(1, 2048, device=DEVICE, dtype=torch.bfloat16)
            for k_idx, e_id in enumerate(ids_list):
                if e_id in active_custom_weights:
                    W_gu = active_custom_weights[e_id]["gu"]
                    W_dn = active_custom_weights[e_id]["dn"]
                    y_gu = F.linear(xt, W_gu)
                    g, u = y_gu.chunk(2, dim=-1)
                    act = (F.silu(g.float()) * u.float()).bfloat16()
                    y_k = F.linear(act, W_dn)
                    token_out += y_k * w_t[k_idx]
                else:
                    # Executa via slot original 2 bits
                    slot = layer_module._get_slot(e_id)
                    slots_t = torch.tensor([slot], dtype=torch.int32, device=DEVICE)
                    xr = layer_module.rot_2048.fwd(xt.view(-1))
                    y_gu = torch.empty((1, 1024), device=DEVICE, dtype=torch.bfloat16)
                    vq_moe_parallel_gate_up(xr, slots_t, layer_module.pool_gu_idx, layer_module.cb16, layer_module.pool_gu_scale, y_gu, 1024, 256, layer_module.RS_2048)
                    g, u = y_gu[:, :512], y_gu[:, 512:]
                    act = (F.silu(g.float()) * u.float()).bfloat16()
                    act_r = layer_module.rot_512.fwd(act)
                    y_dn = torch.empty((1, 2048), device=DEVICE, dtype=torch.bfloat16)
                    vq_moe_parallel_down(act_r, slots_t, layer_module.pool_dn_idx, layer_module.cb16, layer_module.pool_dn_scale, y_dn, 2048, 64, layer_module.RS_512)
                    token_out += y_dn * w_t[k_idx]
            outs.append(token_out)
        return torch.cat(outs, dim=0)

    return orig_fwd, custom_forward

def run_causal_audit():
    print("=" * 80)
    print("AUDITORIA CAUSAL E INCREMENTAL DE LOGITS: CONFIGURAÇÕES A vs B1 vs B2")
    print("=" * 80)

    # 1. Carrega pesos originais BF16
    print("[1/4] Carregando pesos originais HF BF16 dos 16 especialistas promovidos...")
    orig_weights = load_promoted_original_weights()
    for ln in sorted(orig_weights.keys()):
        print(f"      Camada {ln}: especialistas {list(orig_weights[ln].keys())}")

    # 2. Sintetiza pesos reconstruídos em 3 bits
    print("\n[2/4] Sintetizando pesos 3 bits (cfg=(16, 8)) via RVQ...")
    weights_3b = build_3bit_weights(orig_weights)

    # 3. Inicializa Engine do LEGION (baseada no container A original)
    print("\n[3/4] Inicializando Motor Neural LEGION (Baseline A)...")
    engine = LegionEngine()
    tokenizer = engine.tokenizer

    # 4. Configuração das 4 variantes experimentais:
    #   - A: nenhuma substituição (100% 2 bits)
    #   - B1: substituição em 3 bits apenas para Camada 39 (6 âncoras)
    #   - B2: substituição em 3 bits para todas as 8 camadas (16 especialistas)
    #   - Ref: substituição com pesos originais BF16 para todas as 8 camadas
    b1_weights_3b = {39: weights_3b[39]}
    b2_weights_3b = weights_3b
    ref_weights = orig_weights

    print("\n[4/4] Executando auditoria causal nos 6 prompts inéditos...")
    print("=" * 80)

    audit_records = []

    for idx, item in enumerate(UNSEEN_PROMPTS, 1):
        domain = item["domain"]
        prompt = item["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        # Função auxiliar para forward com mapa de substituição
        def forward_with_map(custom_map):
            orig_forwards = {}
            for ln, exp_dict in custom_map.items():
                l_mod = engine.layers[ln]
                orig_f, custom_f = create_layer_forward_hook(l_mod, exp_dict)
                orig_forwards[ln] = orig_f
                l_mod.forward = custom_f

            with torch.no_grad():
                out = engine.model(**inputs)
                logits = out.logits[0, -1, :].float()

            for ln, orig_f in orig_forwards.items():
                engine.layers[ln].forward = orig_f
            return logits

        # A. Execução A (100% 2 bits)
        with torch.no_grad():
            out_A = engine.model(**inputs)
            logits_A = out_A.logits[0, -1, :].float()

        # B1. Execução B1 (3 bits nos 6 da L39)
        logits_B1 = forward_with_map(b1_weights_3b)

        # B2. Execução B2 (3 bits nos 16 promovidos)
        logits_B2 = forward_with_map(b2_weights_3b)

        # Ref. Execução Referência BF16 (16 especialistas em BF16 exato)
        logits_Ref = forward_with_map(ref_weights)

        # Cálculo de probabilidades softmax
        probs_Ref = F.softmax(logits_Ref, dim=-1)
        probs_A = F.softmax(logits_A, dim=-1)
        probs_B1 = F.softmax(logits_B1, dim=-1)
        probs_B2 = F.softmax(logits_B2, dim=-1)

        # Divergências KL em relação à Ref
        kl_A = F.kl_div(probs_A.log(), probs_Ref, reduction='sum').item()
        kl_B1 = F.kl_div(probs_B1.log(), probs_Ref, reduction='sum').item()
        kl_B2 = F.kl_div(probs_B2.log(), probs_Ref, reduction='sum').item()

        # Similaridades Cosseno
        cos_A = F.cosine_similarity(logits_Ref, logits_A, dim=0).item()
        cos_B1 = F.cosine_similarity(logits_Ref, logits_B1, dim=0).item()
        cos_B2 = F.cosine_similarity(logits_Ref, logits_B2, dim=0).item()

        # Top-1 Tokens
        top1_Ref = torch.argmax(logits_Ref).item()
        top1_A = torch.argmax(logits_A).item()
        top1_B1 = torch.argmax(logits_B1).item()
        top1_B2 = torch.argmax(logits_B2).item()

        tok_Ref = tokenizer.decode([top1_Ref]).strip()
        tok_A = tokenizer.decode([top1_A]).strip()
        tok_B1 = tokenizer.decode([top1_B1]).strip()
        tok_B2 = tokenizer.decode([top1_B2]).strip()

        print(f"\n[Prompt {idx} - Domínio: {domain}]")
        print(f"  Prompt: \"{prompt[:70]}...\"")
        print(f"  Token Previsto Ref (BF16): '{tok_Ref}' (id: {top1_Ref})")
        print(f"  Config A  (2 bits):   '{tok_A}' | Match: {top1_A == top1_Ref} | Cos: {cos_A:.5f} | KL: {kl_A:.5f}")
        print(f"  Config B1 (L39 3b):   '{tok_B1}' | Match: {top1_B1 == top1_Ref} | Cos: {cos_B1:.5f} | KL: {kl_B1:.5f} (delta-KL vs A: {(kl_B1-kl_A):.5f})")
        print(f"  Config B2 (16 exp 3b): '{tok_B2}' | Match: {top1_B2 == top1_Ref} | Cos: {cos_B2:.5f} | KL: {kl_B2:.5f} (delta-KL vs B1: {(kl_B2-kl_B1):.5f})")

        audit_records.append({
            "id": idx,
            "domain": domain,
            "prompt": prompt,
            "kl_A": kl_A, "kl_B1": kl_B1, "kl_B2": kl_B2,
            "cos_A": cos_A, "cos_B1": cos_B1, "cos_B2": cos_B2,
            "match_A": top1_A == top1_Ref,
            "match_B1": top1_B1 == top1_Ref,
            "match_B2": top1_B2 == top1_Ref,
            "tok_Ref": tok_Ref, "tok_A": tok_A, "tok_B1": tok_B1, "tok_B2": tok_B2
        })

    # Consolidação Geral
    print("\n" + "=" * 80)
    print("CONSOLIDAÇÃO ESTATÍSTICA DA AUDITORIA CAUSAL (A vs B1 vs B2):")
    print("=" * 80)

    mean_kl_A = sum(r["kl_A"] for r in audit_records) / len(audit_records)
    mean_kl_B1 = sum(r["kl_B1"] for r in audit_records) / len(audit_records)
    mean_kl_B2 = sum(r["kl_B2"] for r in audit_records) / len(audit_records)

    mean_cos_A = sum(r["cos_A"] for r in audit_records) / len(audit_records)
    mean_cos_B1 = sum(r["cos_B1"] for r in audit_records) / len(audit_records)
    mean_cos_B2 = sum(r["cos_B2"] for r in audit_records) / len(audit_records)

    top1_A = sum(r["match_A"] for r in audit_records) / len(audit_records)
    top1_B1 = sum(r["match_B1"] for r in audit_records) / len(audit_records)
    top1_B2 = sum(r["match_B2"] for r in audit_records) / len(audit_records)

    red_B1_A = (1 - mean_kl_B1 / mean_kl_A) * 100
    red_B2_B1 = (1 - mean_kl_B2 / mean_kl_B1) * 100
    red_B2_A = (1 - mean_kl_B2 / mean_kl_A) * 100

    print(f"1. Divergência KL Média (vs HF BF16):")
    print(f"   - Baseline A  (100% 2 bits):         {mean_kl_A:.6f}")
    print(f"   - Artefato B1 (6 âncoras L39):       {mean_kl_B1:.6f}  (Redução vs A: -{red_B1_A:.1f}%)")
    print(f"   - Artefato B2 (16 especialistas):    {mean_kl_B2:.6f}  (Redução vs B1: -{red_B2_B1:.1f}% | Total vs A: -{red_B2_A:.1f}%)")
    print()
    print(f"2. Similaridade Cosseno dos Logits:")
    print(f"   - Baseline A:  {mean_cos_A:.6f}")
    print(f"   - Artefato B1: {mean_cos_B1:.6f}")
    print(f"   - Artefato B2: {mean_cos_B2:.6f}")
    print()
    print(f"3. Concordância Top-1 Token:")
    print(f"   - Baseline A:  {top1_A*100:.1f}%")
    print(f"   - Artefato B1: {top1_B1*100:.1f}%")
    print(f"   - Artefato B2: {top1_B2*100:.1f}%")
    print("=" * 80)

    # 5. Teste de Geração Greedy Pura (50 tokens, sem penalidade de repetição)
    print("\n[5/5] Teste de Geração Greedy Pura (Estabilidade e Ausência de Repetição)...")
    test_p = "Em arquiteturas de microsserviços distribuídos, o teorema CAP afirma que"
    
    # Gera com B2
    orig_forwards = {}
    for ln, exp_dict in b2_weights_3b.items():
        l_mod = engine.layers[ln]
        orig_f, custom_f = create_layer_forward_hook(l_mod, exp_dict)
        orig_forwards[ln] = orig_f
        l_mod.forward = custom_f

    tokens = tokenizer(test_p, return_tensors="pt")["input_ids"].to(DEVICE)
    all_tokens = tokens[0].tolist()
    past_kv = None
    gen_text_tokens = []
    
    for step in range(50):
        inp = tokens if past_kv is None else tokens[:, -1:]
        with torch.no_grad():
            out = engine.model(input_ids=inp, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()
        all_tokens.append(next_tok)
        gen_text_tokens.append(next_tok)
        tokens = torch.tensor([[next_tok]], device=DEVICE)
        if next_tok in engine.eos_ids:
            break

    # Restaura forwards
    for ln, orig_f in orig_forwards.items():
        engine.layers[ln].forward = orig_f

    generated_output = tokenizer.decode(gen_text_tokens, skip_special_tokens=True)
    print(f"Prompt de teste: \"{test_p}\"")
    print(f"Geração Greedy B2 (50 tokens):\n{generated_output}")
    print("=" * 80)

    # Salva relatório JSON completo
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": {
            "mean_kl_A": mean_kl_A,
            "mean_kl_B1": mean_kl_B1,
            "mean_kl_B2": mean_kl_B2,
            "reduction_B1_vs_A_percent": red_B1_A,
            "reduction_B2_vs_B1_percent": red_B2_B1,
            "reduction_B2_vs_A_percent": red_B2_A,
            "mean_cos_A": mean_cos_A,
            "mean_cos_B1": mean_cos_B1,
            "mean_cos_B2": mean_cos_B2,
            "top1_acc_A": top1_A,
            "top1_acc_B1": top1_B1,
            "top1_acc_B2": top1_B2,
        },
        "records": audit_records,
        "greedy_sample": generated_output
    }
    with open(ROOT / "scratch" / "causal_audit_a_b1_b2_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Resultados da Auditoria Causal salvos em: {ROOT / 'scratch' / 'causal_audit_a_b1_b2_results.json'}")

if __name__ == "__main__":
    run_causal_audit()
