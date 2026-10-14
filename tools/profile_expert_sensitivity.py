"""
tools/profile_expert_sensitivity.py
LEGION Adaptive Precision — Profiler de Sensibilidade por Especialista (Etapa A).

Metodologia em conformidade com as 3 correções de auditoria:
1. Medição pareada estrita contra pesos originais HF (safetensors BF16) em 2, 3 e 4 bits.
2. Relatório duplo: erro bruto (Média, P95, Máx) e erro ponderado por roteamento π(e|x),
   com frequência de ativação para não ocultar especialistas raros.
3. Ganho marginal por byte adicional (G_2->3 e G_3->4) baseado nos bytes reais serializados
   (Δ = 393.216 bytes por passo de 1-bit).
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

INDEX_PATH = ROOT / "modelo_35b" / "model.safetensors.index.json"
INDEX_MAP = json.load(open(INDEX_PATH))["weight_map"] if INDEX_PATH.exists() else {}

CALIBRATION_PROMPTS = [
    "A teoria da relatividade geral de Einstein descreve a gravidade não como uma força mística, mas como a curvatura do espaço-tempo provocada pela distribuição de massa e energia.",
    "Implement a high-performance LRU cache in Python with O(1) get and put operations using a doubly linked list and a hash map.",
    "Se um modelo MoE possui 40 camadas, 256 especialistas por camada e ativa 8 especialistas por token, qual é o número total de despachos por token e como funciona o carregamento paginado?",
    "Explique o impacto do Iluminismo no desenvolvimento das democracias constitucionais modernas, no método científico experimental e na separação de poderes.",
    "Você é o LEGION, um motor de inteligência artificial de alta performance focado em computação vetorial e quantização extrema de 2 bits."
]

SERIALIZED_BYTES = {
    2: 792576,    # cfg=(16,) -> 2.0 bpw
    3: 1185792,   # cfg=(16, 8) -> 3.0 bpw
    4: 1579008,   # cfg=(16, 16) -> 4.0 bpw
}
DELTA_BYTES = 393216 # 384 KiB por passo

def collect_layer_activations(engine, target_layers):
    """Executa os prompts de calibração no modelo e extrai as ativações reais de entrada do MoE."""
    cache_path = ROOT / "scratch" / "calibration_layer_data.pt"
    if cache_path.exists():
        print(f"\n[1/3] Carregando ativações de calibração do cache ({cache_path.name})...")
        return torch.load(cache_path, weights_only=False)

    print("\n[1/3] Coletando ativações reais de calibração via Forward Hooks...")
    activations = {l: [] for l in target_layers}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, args, output):
            x = args[0].detach()
            activations[layer_idx].append(x.view(-1, x.shape[-1]))
        return hook_fn

    for l in target_layers:
        h = engine.model.model.layers[l].mlp.experts.register_forward_hook(make_hook(l))
        hooks.append(h)

    # Executa prefill dos textos de calibração
    with torch.no_grad():
        for i, text in enumerate(CALIBRATION_PROMPTS):
            inputs = engine.tokenizer(text, return_tensors="pt").to(DEVICE)
            engine.model(**inputs)

    for h in hooks:
        h.remove()

    # Concatena ativações por camada
    layer_data = {}
    for l in target_layers:
        all_x = torch.cat(activations[l], dim=0) # [Total_Tokens, 2048]
        gate_w = engine.model.model.layers[l].mlp.gate.weight # [256, 2048]
        logits = F.linear(all_x.bfloat16(), gate_w.bfloat16())
        probs = F.softmax(logits.float(), dim=-1)
        topk_weights, topk_ids = torch.topk(probs, k=8, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        layer_data[l] = {
            "x": all_x.cpu(),
            "probs": probs.cpu(),
            "topk_ids": topk_ids.cpu(),
            "topk_weights": topk_weights.cpu(),
            "total_tokens": all_x.shape[0]
        }
        print(f"      Camada {l}: {all_x.shape[0]} vetores de ativação coletados.")

    cache_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(layer_data, cache_path)
    return layer_data

def evaluate_expert_precisions(l_idx, e_idx, orig_gu, orig_dn, cbs, rot_2048, rot_512, x_all, routing_w_all):
    """Avalia o especialista sob 2, 3 e 4 bits usando ativações reais e pesos originais."""
    t0 = time.time()
    
    # 1. Saída de Referência Original (BF16)
    gu_ref = F.linear(x_all, orig_gu)
    gate_ref, up_ref = gu_ref.chunk(2, dim=-1)
    act_ref = F.silu(gate_ref.float()) * up_ref.float()
    y_orig = F.linear(act_ref.bfloat16(), orig_dn)
    norm_orig = y_orig.float().norm(dim=-1)
    mean_norm_orig = norm_orig.mean().item()

    # 2. Rotação dos pesos para base de Hadamard
    W_gu_rot = rot_2048.fwd(orig_gu)
    s_gu = W_gu_rot.float().norm(dim=1, keepdim=True) / math.sqrt(2048)
    u_gu = (W_gu_rot.float() / s_gu.clamp_min(1e-6)).reshape(-1, 8).to(DEVICE, torch.bfloat16)

    W_dn_rot = rot_512.fwd(orig_dn)
    s_dn = W_dn_rot.float().norm(dim=1, keepdim=True) / math.sqrt(512)
    u_dn = (W_dn_rot.float() / s_dn.clamp_min(1e-6)).reshape(-1, 8).to(DEVICE, torch.bfloat16)

    configs = [(2, (16,)), (3, (16, 8)), (4, (16, 16))]
    metrics = {}

    for b, cfg in configs:
        # Reconstrução Gate-Up
        _, rec_gu = vq.encode(u_gu, cfg, cbs)
        Q_gu = (rec_gu.view(1024, 2048) * s_gu.to(torch.bfloat16)).to(torch.bfloat16)
        W_rec_gu = rot_2048.inv(Q_gu)

        # Reconstrução Down
        _, rec_dn = vq.encode(u_dn, cfg, cbs)
        Q_dn = (rec_dn.view(2048, 512) * s_dn.to(torch.bfloat16)).to(torch.bfloat16)
        W_rec_dn = rot_512.inv(Q_dn)

        # Execução com pesos reconstruídos
        y_gu_rec = F.linear(x_all, W_rec_gu)
        g_rec, u_rec = y_gu_rec.chunk(2, dim=-1)
        act_rec = (F.silu(g_rec.float()) * u_rec.float()).bfloat16()
        y_b = F.linear(act_rec, W_rec_dn)

        # Erro pontual por amostra: ||y_orig - y_b||^2
        diff_sq = ((y_orig.float() - y_b.float()) ** 2).sum(dim=-1) # [N]
        e_mean = diff_sq.mean().item()
        e_p95 = torch.quantile(diff_sq, 0.95).item()
        e_max = diff_sq.max().item()

        # Erro ponderado por roteamento: sum(diff_sq * pi(e|x)) / N
        e_weighted = (diff_sq * routing_w_all).mean().item()

        # Alinhamento e Norma
        cos_sim = F.cosine_similarity(y_orig.float(), y_b.float(), dim=-1).mean().item()
        norm_b = y_b.float().norm(dim=-1).mean().item()
        norm_ratio = norm_b / max(mean_norm_orig, 1e-6)

        metrics[b] = {
            "error_mean": e_mean,
            "error_p95": e_p95,
            "error_max": e_max,
            "error_weighted": e_weighted,
            "cosine_similarity": cos_sim,
            "norm_ratio": norm_ratio,
        }

    # Ganhos Marginais por Byte:
    g_2to3_weighted = (metrics[2]["error_weighted"] - metrics[3]["error_weighted"]) / DELTA_BYTES
    g_3to4_weighted = (metrics[3]["error_weighted"] - metrics[4]["error_weighted"]) / DELTA_BYTES

    g_2to3_raw = (metrics[2]["error_mean"] - metrics[3]["error_mean"]) / DELTA_BYTES
    g_3to4_raw = (metrics[3]["error_mean"] - metrics[4]["error_mean"]) / DELTA_BYTES

    t_eval = time.time() - t0

    return {
        "metrics": metrics,
        "g_2to3_weighted": g_2to3_weighted,
        "g_3to4_weighted": g_3to4_weighted,
        "g_2to3_raw": g_2to3_raw,
        "g_3to4_raw": g_3to4_raw,
        "eval_time_sec": t_eval
    }

def run_sensitivity_profiling(target_layers=[0, 10, 20, 39], max_experts_per_layer=16):
    print("=" * 80)
    print("LEGION ADAPTIVE PRECISION — PROFILER DE SENSIBILIDADE POR ESPECIALISTA")
    print(f"Camadas Alvo: {target_layers} | Amostras por Camada: até {max_experts_per_layer} especialistas")
    print("=" * 80)

    # 1. Carrega ou Coleta ativações reais de calibração
    cache_path = ROOT / "scratch" / "calibration_layer_data.pt"
    if cache_path.exists():
        print(f"\n[1/3] Carregando ativações de calibração do cache ({cache_path.name})...")
        layer_data = torch.load(cache_path, weights_only=False)
    else:
        print("\n[Inicializando Motor Neural para Calibração...]")
        engine = LegionEngine()
        layer_data = collect_layer_activations(engine, target_layers)
        del engine
        torch.cuda.empty_cache()

    # 2. Carrega Codebooks
    print("\n[2/3] Carregando Codebooks VQ2 e Rotações Ortogonais...")
    cbs_data = torch.load(ROOT / "vq2_out" / "codebooks.pt", weights_only=False)
    cbs = vq.Codebooks({k: v.to(DEVICE, torch.bfloat16) for k, v in cbs_data.items()})
    rot_2048 = vq.get_rot("moe_expert_2048", 2048, DEVICE)
    rot_512 = vq.get_rot("moe_expert_512", 512, DEVICE)

    # 3. Avalia especialistas nas camadas alvo
    print("\n[3/3] Avaliando sensibilidade pareada contra pesos originais HF...")
    profile_results = {}

    for l_idx in target_layers:
        print(f"\n--- Camada {l_idx} ---")
        
        ld = layer_data[l_idx]
        x_all = ld["x"].to(DEVICE, torch.bfloat16)
        probs = ld["probs"].to(DEVICE)
        topk_ids = ld["topk_ids"].to(DEVICE)
        total_tokens = ld["total_tokens"]

        # Calcula estatísticas de roteamento para cada especialista
        expert_routing_stats = []
        for e_id in range(256):
            # Quantas vezes e_id aparece em topk_ids?
            mask_sel = (topk_ids == e_id).any(dim=-1)
            sel_count = mask_sel.sum().item()
            freq = sel_count / total_tokens
            # Vetor de peso atribuído a e_id em cada token:
            pi_e = probs[:, e_id] # [N]
            mean_pi = pi_e.mean().item()
            max_pi = pi_e.max().item()
            expert_routing_stats.append((e_id, freq, sel_count, mean_pi, max_pi, pi_e))

        # Seleciona uma mistura representativa: especialistas mais frequentes + especialistas raros
        expert_routing_stats.sort(key=lambda x: x[1], reverse=True)
        # Pega os 12 mais ativos e 4 raros com menor frequência para cobrir os dois perfis
        top_active = expert_routing_stats[:12]
        rare_active = [item for item in expert_routing_stats[12:] if item[2] > 0][:4]
        if len(rare_active) < 4:
            rare_active = expert_routing_stats[12:16]
        
        eval_list = (top_active + rare_active)[:max_experts_per_layer]

        layer_results = []
        k_gu = f"model.language_model.layers.{l_idx}.mlp.experts.gate_up_proj"
        k_dn = f"model.language_model.layers.{l_idx}.mlp.experts.down_proj"
        shard_gu = ROOT / "modelo_35b" / INDEX_MAP[k_gu]
        shard_dn = ROOT / "modelo_35b" / INDEX_MAP[k_dn]

        with safe_open(str(shard_gu), framework="pt", device="cpu") as f_gu, \
             safe_open(str(shard_dn), framework="pt", device="cpu") as f_dn:
            gu_slice = f_gu.get_slice(k_gu)
            dn_slice = f_dn.get_slice(k_dn)

            for e_info in eval_list:
                e_id, freq, sel_count, mean_pi, max_pi, pi_e = e_info
                
                # Carrega pesos originais do especialista
                orig_gu = gu_slice[e_id].to(DEVICE, torch.bfloat16)
                orig_dn = dn_slice[e_id].to(DEVICE, torch.bfloat16)

                res = evaluate_expert_precisions(
                    l_idx, e_id, orig_gu, orig_dn, cbs, rot_2048, rot_512, x_all, pi_e.to(DEVICE)
                )

                res["layer_idx"] = l_idx
                res["expert_idx"] = e_id
                res["routing_freq"] = freq
                res["selection_count"] = sel_count
                res["mean_routing_prob"] = mean_pi
                res["max_routing_prob"] = max_pi

                layer_results.append(res)
                
                # Exibe linha de progresso
                m = res["metrics"]
                print(f"  Exp {e_id:>3} | Freq: {freq*100:>5.1f}% | Cos: 2b={m[2]['cosine_similarity']:.4f} -> 3b={m[3]['cosine_similarity']:.4f} -> 4b={m[4]['cosine_similarity']:.4f} | G_2->3: {res['g_2to3_weighted']*1e6:>7.3f} µE/KB")

        profile_results[l_idx] = layer_results

    # Salva arquivo JSON estruturado
    out_file = ROOT / "scratch" / "expert_sensitivity_profile.json"
    out_file.parent.mkdir(exist_ok=True, parents=True)
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(profile_results, fp, indent=2)

    print(f"\n[OK] Profiler concluído com sucesso! Resultados salvos em: {out_file}")
    return profile_results

if __name__ == "__main__":
    run_sensitivity_profiling(target_layers=[0, 10, 20, 39], max_experts_per_layer=16)
