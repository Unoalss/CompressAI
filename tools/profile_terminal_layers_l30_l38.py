"""
tools/profile_terminal_layers_l30_l38.py
Passagem 2 — Amostragem e Perfilamento das Camadas Terminais L30 a L38.
Objetivo: Classificar candidatos a 3 bits por ganho marginal normalizado por byte (S_e),
separando explicitamente distorção relativa de escala de ativação e frequência de roteamento.
"""
import os, sys, time, json, math
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vq2 import vq
from legion.engine import LegionEngine

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DELTA_BYTES = 393216 # 384 KiB por especialista

TARGET_LAYERS = list(range(30, 39)) # [30, 31, 32, 33, 34, 35, 36, 37, 38]

DIVERSE_CALIB_PROMPTS = [
    "A teoria da relatividade geral de Albert Einstein descreve a gravidade como a curvatura do espaco-tempo causada pela distribuicao de massa e energia.",
    "Implement a high-performance LRU cache in Python with O(1) get and put operations using a doubly linked list and a hash map.",
    "Se um modelo MoE possui 40 camadas, 256 especialistas por camada e ativa 8 especialistas por token, qual e o numero total de despachos?",
    "Explique o impacto do Iluminismo no desenvolvimento das democracias constitucionais modernas e no pensamento cientifico experimental.",
    "SELECT u.id, u.name, COUNT(o.id) FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id HAVING COUNT(o.id) > 5;",
    "Compare o imperativo categorico de Immanuel Kant com a teoria etica utilitarista de John Stuart Mill."
]

def collect_l30_l38_activations(engine):
    cache_path = ROOT / "scratch" / "calibration_l30_l38_data.pt"
    if cache_path.exists():
        print(f"\n[1/4] Carregando ativacoes L30-L38 do cache ({cache_path.name})...")
        return torch.load(cache_path, weights_only=False)

    print("\n[1/4] Coletando ativacoes reais de entrada para as Camadas L30 a L38...")
    activations = {l: [] for l in TARGET_LAYERS}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, args, output):
            x = args[0].detach()
            activations[layer_idx].append(x.view(-1, x.shape[-1]))
        return hook_fn

    for l in TARGET_LAYERS:
        h = engine.model.model.layers[l].mlp.experts.register_forward_hook(make_hook(l))
        hooks.append(h)

    with torch.no_grad():
        for text in DIVERSE_CALIB_PROMPTS:
            inputs = engine.tokenizer(text, return_tensors="pt").to(DEVICE)
            engine.model(**inputs)

    for h in hooks:
        h.remove()

    layer_data = {}
    for l in TARGET_LAYERS:
        all_x = torch.cat(activations[l], dim=0) # [Tokens, 2048]
        gate_w = engine.model.model.layers[l].mlp.gate.weight
        logits = F.linear(all_x.bfloat16(), gate_w.bfloat16())
        probs = F.softmax(logits.float(), dim=-1)
        topk_w, topk_ids = torch.topk(probs, k=8, dim=-1)
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)

        layer_data[l] = {
            "x": all_x.cpu(),
            "probs": probs.cpu(),
            "topk_ids": topk_ids.cpu(),
            "topk_weights": topk_w.cpu(),
            "total_tokens": all_x.shape[0]
        }
        print(f"      Camada {l}: {all_x.shape[0]} vetores coletados.")

    cache_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(layer_data, cache_path)
    return layer_data

def evaluate_expert_relative_distortion(l_idx, e_id, orig_gu, orig_dn, cbs, rot_2048, rot_512, x_all, pi_e):
    # 1. Referência Original
    gu_ref = F.linear(x_all, orig_gu)
    gate_ref, up_ref = gu_ref.chunk(2, dim=-1)
    act_ref = F.silu(gate_ref.float()) * up_ref.float()
    y_orig = F.linear(act_ref.bfloat16(), orig_dn)
    
    # Energia do sinal: sum(||y_orig||^2)
    signal_energy = (y_orig.float() ** 2).sum(dim=-1).mean().item()
    norm_orig_mean = y_orig.float().norm(dim=-1).mean().item()

    # Rotação de pesos
    W_gu_rot = rot_2048.fwd(orig_gu)
    s_gu = (W_gu_rot.float().norm(dim=1, keepdim=True) / math.sqrt(2048)).clamp_min(1e-6)
    u_gu = (W_gu_rot.float() / s_gu).reshape(-1, 8).to(DEVICE, torch.bfloat16)

    W_dn_rot = rot_512.fwd(orig_dn)
    s_dn = (W_dn_rot.float().norm(dim=1, keepdim=True) / math.sqrt(512)).clamp_min(1e-6)
    u_dn = (W_dn_rot.float() / s_dn).reshape(-1, 8).to(DEVICE, torch.bfloat16)

    # 2. Reconstrução 2 bits (16,)
    _, rec_gu_2b = vq.encode(u_gu, (16,), cbs)
    Q_gu_2b = (rec_gu_2b.view(1024, 2048) * s_gu.to(torch.bfloat16)).to(torch.bfloat16)
    W_rec_gu_2b = rot_2048.inv(Q_gu_2b)

    _, rec_dn_2b = vq.encode(u_dn, (16,), cbs)
    Q_dn_2b = (rec_dn_2b.view(2048, 512) * s_dn.to(torch.bfloat16)).to(torch.bfloat16)
    W_rec_dn_2b = rot_512.inv(Q_dn_2b)

    y_gu_2b = F.linear(x_all, W_rec_gu_2b)
    g2, u2 = y_gu_2b.chunk(2, dim=-1)
    act2 = (F.silu(g2.float()) * u2.float()).bfloat16()
    y_2b = F.linear(act2, W_rec_dn_2b)

    diff_sq_2b = ((y_orig.float() - y_2b.float()) ** 2).sum(dim=-1)
    d_raw_2b = diff_sq_2b.mean().item()
    d_rel_2b = d_raw_2b / max(signal_energy, 1e-6)
    cos_2b = F.cosine_similarity(y_orig.float(), y_2b.float(), dim=-1).mean().item()

    # 3. Reconstrução 3 bits (16, 8)
    _, rec_gu_3b = vq.encode(u_gu, (16, 8), cbs)
    Q_gu_3b = (rec_gu_3b.view(1024, 2048) * s_gu.to(torch.bfloat16)).to(torch.bfloat16)
    W_rec_gu_3b = rot_2048.inv(Q_gu_3b)

    _, rec_dn_3b = vq.encode(u_dn, (16, 8), cbs)
    Q_dn_3b = (rec_dn_3b.view(2048, 512) * s_dn.to(torch.bfloat16)).to(torch.bfloat16)
    W_rec_dn_3b = rot_512.inv(Q_dn_3b)

    y_gu_3b = F.linear(x_all, W_rec_gu_3b)
    g3, u3 = y_gu_3b.chunk(2, dim=-1)
    act3 = (F.silu(g3.float()) * u3.float()).bfloat16()
    y_3b = F.linear(act3, W_rec_dn_3b)

    diff_sq_3b = ((y_orig.float() - y_3b.float()) ** 2).sum(dim=-1)
    d_raw_3b = diff_sq_3b.mean().item()
    d_rel_3b = d_raw_3b / max(signal_energy, 1e-6)
    cos_3b = F.cosine_similarity(y_orig.float(), y_3b.float(), dim=-1).mean().item()

    # Ganho Marginal Normalizado por Byte: S_e = (D_rel_2b - D_rel_3b) / DELTA_BYTES
    delta_d_rel = d_rel_2b - d_rel_3b
    s_e_rel = delta_d_rel / DELTA_BYTES # Ganho relativo por byte

    # Ganho Ponderado pela probabilidade de roteamento
    mean_pi = pi_e.mean().item()
    s_e_weighted = s_e_rel * mean_pi

    return {
        "d_rel_2b": d_rel_2b,
        "d_rel_3b": d_rel_3b,
        "delta_d_rel": delta_d_rel,
        "cos_2b": cos_2b,
        "cos_3b": cos_3b,
        "signal_energy": signal_energy,
        "s_e_rel": s_e_rel,
        "s_e_weighted": s_e_weighted,
        "mean_pi": mean_pi,
    }

def run_l30_l38_profiling():
    print("=" * 80)
    print("PASSAGEM 2: PERFILAMENTO DE SENSIBILIDADE RELATIVA DAS CAMADAS L30 A L38")
    print(f"Camadas: {TARGET_LAYERS} (9 camadas) | Amostragem: 16 especialistas/camada")
    print("=" * 80)

    # 1. Coleta ou Carrega Ativações
    cache_path = ROOT / "scratch" / "calibration_l30_l38_data.pt"
    if cache_path.exists():
        print(f"\n[1/4] Carregando ativacoes L30-L38 do cache ({cache_path.name})...")
        layer_data = torch.load(cache_path, weights_only=False)
    else:
        print("\n[Inicializando Motor LEGION para Coleta de Ativacoes...]")
        engine = LegionEngine()
        layer_data = collect_l30_l38_activations(engine)
        del engine
        torch.cuda.empty_cache()

    # 2. Carrega Codebooks
    print("\n[2/4] Carregando Codebooks VQ2 e Rotações...")
    cbs_data = torch.load(ROOT / "vq2_out" / "codebooks.pt", weights_only=False)
    cbs = vq.Codebooks({k: v.to(DEVICE, torch.bfloat16) for k, v in cbs_data.items()})
    rot_2048 = vq.get_rot("moe_expert_2048", 2048, DEVICE)
    rot_512 = vq.get_rot("moe_expert_512", 512, DEVICE)

    index_map = json.load(open(ROOT / "modelo_35b" / "model.safetensors.index.json"))["weight_map"]

    # 3. Avaliação de 16 Especialistas por Camada
    print("\n[3/4] Avaliando Distorção Relativa e Ganho Marginal Normalizado por Byte...")
    all_evaluated_candidates = []
    layer_profiles = {}

    for l_idx in TARGET_LAYERS:
        ld = layer_data[l_idx]
        x_all = ld["x"].to(DEVICE, torch.bfloat16)
        probs = ld["probs"] # [N, 256]
        topk_ids = ld["topk_ids"] # [N, 8]
        total_tokens = ld["total_tokens"]

        # Estatísticas de roteamento dos 256 especialistas
        exp_stats = []
        for e_id in range(256):
            mask = (topk_ids == e_id).any(dim=-1)
            sel_count = mask.sum().item()
            freq = sel_count / total_tokens
            pi_e = probs[:, e_id]
            mean_pi = pi_e.mean().item()
            exp_stats.append((e_id, freq, sel_count, mean_pi, pi_e))

        exp_stats.sort(key=lambda item: item[1], reverse=True)
        # Amostra: 8 mais frequentes, 4 intermediários (ranks 40..60), 4 de cauda (ranks 120+)
        sample_top = exp_stats[:8]
        sample_mid = exp_stats[40:44] if len(exp_stats) > 44 else exp_stats[8:12]
        sample_tail = [item for item in exp_stats[80:] if item[2] > 0][:4]
        if len(sample_tail) < 4:
            sample_tail = exp_stats[12:16]

        eval_sample = sample_top + sample_mid + sample_tail

        # Abre os shards safetensors
        k_gu = f"model.language_model.layers.{l_idx}.mlp.experts.gate_up_proj"
        k_dn = f"model.language_model.layers.{l_idx}.mlp.experts.down_proj"
        shard_gu = ROOT / "modelo_35b" / index_map[k_gu]
        shard_dn = ROOT / "modelo_35b" / index_map[k_dn]

        layer_results = []
        print(f"\n--- Camada {l_idx} ---")
        with safe_open(str(shard_gu), framework="pt", device="cpu") as f_gu, \
             safe_open(str(shard_dn), framework="pt", device="cpu") as f_dn:
            gu_slice = f_gu.get_slice(k_gu)
            dn_slice = f_dn.get_slice(k_dn)

            for item in eval_sample:
                e_id, freq, sel_count, mean_pi, pi_e = item
                orig_gu = gu_slice[e_id].to(DEVICE, torch.bfloat16)
                orig_dn = dn_slice[e_id].to(DEVICE, torch.bfloat16)

                metrics = evaluate_expert_relative_distortion(
                    l_idx, e_id, orig_gu, orig_dn, cbs, rot_2048, rot_512, x_all, pi_e.to(DEVICE)
                )

                cand = {
                    "layer": l_idx,
                    "expert_id": e_id,
                    "routing_freq": freq,
                    "mean_pi": mean_pi,
                    "d_rel_2b": metrics["d_rel_2b"],
                    "d_rel_3b": metrics["d_rel_3b"],
                    "delta_d_rel": metrics["delta_d_rel"],
                    "cos_2b": metrics["cos_2b"],
                    "cos_3b": metrics["cos_3b"],
                    "s_e_rel": metrics["s_e_rel"],
                    "s_e_weighted": metrics["s_e_weighted"]
                }
                layer_results.append(cand)
                all_evaluated_candidates.append(cand)

                print(f"  Exp {e_id:>3} | Freq: {freq*100:>5.1f}% | D_rel: 2b={metrics['d_rel_2b']:.4f} -> 3b={metrics['d_rel_3b']:.4f} (Cos: {metrics['cos_2b']:.4f} -> {metrics['cos_3b']:.4f}) | S_rel: {metrics['s_e_rel']*1e6:>6.3f} /MB | S_w: {metrics['s_e_weighted']*1e6:>6.3f}")

        layer_profiles[l_idx] = layer_results

    # 4. Compilação do Ranking Global
    print("\n" + "=" * 80)
    print("[4/4] RANKING GLOBAL DE CANDIDATOS (TOP 25) POR GANHO MARGINAL PONDERADO (S_w):")
    print("=" * 80)
    all_evaluated_candidates.sort(key=lambda x: x["s_e_weighted"], reverse=True)

    print(f"{'Rank':<5} | {'Layer':<6} | {'Exp':<5} | {'Freq (%)':<9} | {'Prob (pi)':<10} | {'D_rel (2b->3b)':<17} | {'S_rel (x1e6)':<13} | {'S_weighted (x1e6)':<16}")
    print("-" * 92)
    for r_idx, c in enumerate(all_evaluated_candidates[:25], 1):
        print(f"{r_idx:<5} | L{c['layer']:<5} | {c['expert_id']:>4} | {c['routing_freq']*100:>7.1f}% | {c['mean_pi']:>10.4f} | {c['d_rel_2b']:.4f} -> {c['d_rel_3b']:.4f} | {c['s_e_rel']*1e6:>12.3f} | {c['s_e_weighted']*1e6:>15.4f}")

    print("=" * 92)

    # Salva JSON estruturado
    out_json = ROOT / "scratch" / "l30_l38_sensitivity_profile.json"
    with open(out_json, "w", encoding="utf-8") as fp:
        json.dump({
            "ranking": all_evaluated_candidates,
            "by_layer": layer_profiles
        }, fp, indent=2)

    print(f"\n[OK] Perfilamento de L30-L38 concluído! Resultados salvos em: {out_json}")

if __name__ == "__main__":
    run_l30_l38_profiling()
