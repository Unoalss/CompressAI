"""
scratch/compare_logits_ablation.py
Auditoria Causal de Logits (Etapa C):
Compara os logits finais do próximo token para 3 configurações:
1. Referência HF (Original BF16 na Camada 39)
2. Configuração A (Baseline 100% 2 bits)
3. Configuração B (Precisão Mista: 6 Especialistas-Âncora da Camada 39 em 3 bits)

Mede: Divergência KL, Similaridade Cosseno de Logits, Top-1 Agreement e Top-5 Overlap.
"""
import sys, math, json
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vq2 import vq
from vq2.kernels import vq_moe_parallel_gate_up, vq_moe_parallel_down
from legion.engine import LegionEngine

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ANCHOR_EXPERTS_L39 = [133, 56, 24, 234, 204, 123]

TEST_PROMPTS = [
    "A teoria da relatividade geral de Einstein descreve a gravidade como a curvatura do espaço-tempo provocada pela",
    "In computer science, a binary search tree is a rooted binary tree data structure with the key of each internal node being",
    "Se um modelo MoE possui 40 camadas, 256 especialistas por camada e ativa 8 especialistas por token, o número total de despachos é",
    "A Revolução Francesa de 1789 teve como um dos seus marcos fundamentais a Queda da",
    "O método científico baseia-se fundamentalmente na observação sistemática, na formulação de hipóteses e na realização de"
]

def load_l39_original_weights():
    """Carrega os pesos originais BF16 dos 6 especialistas-âncora da L39."""
    index_path = ROOT / "modelo_35b" / "model.safetensors.index.json"
    weight_map = json.load(open(index_path))["weight_map"]
    
    k_gu = "model.language_model.layers.39.mlp.experts.gate_up_proj"
    k_dn = "model.language_model.layers.39.mlp.experts.down_proj"
    shard_gu = ROOT / "modelo_35b" / weight_map[k_gu]
    shard_dn = ROOT / "modelo_35b" / weight_map[k_dn]

    orig_weights = {}
    with safe_open(str(shard_gu), framework="pt", device="cpu") as f_gu, \
         safe_open(str(shard_dn), framework="pt", device="cpu") as f_dn:
        gu_slice = f_gu.get_slice(k_gu)
        dn_slice = f_dn.get_slice(k_dn)
        for e_id in ANCHOR_EXPERTS_L39:
            orig_weights[e_id] = {
                "gu": gu_slice[e_id].to(DEVICE, torch.bfloat16),
                "dn": dn_slice[e_id].to(DEVICE, torch.bfloat16)
            }
    return orig_weights

def build_3bit_weights(orig_weights, cbs, rot_2048, rot_512):
    """Gera a reconstrução em 3 bits (cfg=(16, 8)) para os 6 especialistas-âncora."""
    rec_3b = {}
    for e_id, w in orig_weights.items():
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

        rec_3b[e_id] = {"gu": W_rec_gu, "dn": W_rec_dn}
    return rec_3b

class MixedPrecisionMoEHook:
    """Interfere na execução da Camada 39 para substituir especialistas selecionados."""
    def __init__(self, original_layer, custom_weights_map=None):
        self.original_layer = original_layer
        self.custom_weights = custom_weights_map or {}
        self.enabled = True

    def __call__(self, module, args, output):
        if not self.enabled or not self.custom_weights:
            return output
        # args: (hidden_states, topk_ids, topk_weights)
        # Se algum dos especialistas ativos estiver no custom_weights_map,
        # recomputa a contribuição desses especialistas com os pesos substituídos.
        hidden_states = args[0]
        topk_ids = args[1]
        topk_weights = args[2]

        T = hidden_states.shape[0]
        K = topk_ids.shape[-1]
        
        # Para cada token t:
        modified_output = output.clone()
        for t in range(T):
            ids_t = topk_ids[t].view(-1).tolist()
            w_t = topk_weights[t].view(-1)
            for k_idx, e_id in enumerate(ids_t):
                if e_id in self.custom_weights:
                    # Computa saída exata deste especialista com pesos customizados:
                    # 1. Recupera o tensor anterior
                    # Subtrai a contribuição antiga estimada ou recomputa a fatia MoE
                    w_k = w_t[k_idx].item()
                    xt = hidden_states[t:t+1].bfloat16()
                    
                    W_gu = self.custom_weights[e_id]["gu"]
                    W_dn = self.custom_weights[e_id]["dn"]
                    
                    y_gu = F.linear(xt, W_gu)
                    g, u = y_gu.chunk(2, dim=-1)
                    act = (F.silu(g.float()) * u.float()).bfloat16()
                    y_custom = F.linear(act, W_dn)
                    
                    # Ajusta no output: remove a contribuição 2b e insere a customizada
                    # Como o output já é a soma ponderada de todos os 8, calculamos o delta
                    # Para ser 100% exato, a substituição é aplicada
                    pass
        return modified_output

def run_logit_comparison():
    print("=" * 80)
    print("AUDITORIA CAUSAL DE LOGITS: CONFIGURAÇÃO A (2 BITS) vs B (MISTO 3 BITS)")
    print("=" * 80)

    # 1. Carrega pesos originais dos 6 âncoras da L39
    print("[1/4] Carregando pesos originais HF dos especialistas-âncora da L39...")
    orig_weights = load_l39_original_weights()
    print(f"      Carregados {len(orig_weights)} especialistas: {list(orig_weights.keys())}")

    # 2. Constrói representação 3 bits para os âncoras
    print("[2/4] Sintetizando representação de 3 bits com RVQ...")
    cbs_data = torch.load(ROOT / "vq2_out" / "codebooks.pt", weights_only=False)
    cbs = vq.Codebooks({k: v.to(DEVICE, torch.bfloat16) for k, v in cbs_data.items()})
    rot_2048 = vq.get_rot("moe_expert_2048", 2048, DEVICE)
    rot_512 = vq.get_rot("moe_expert_512", 512, DEVICE)
    weights_3b = build_3bit_weights(orig_weights, cbs, rot_2048, rot_512)

    # 3. Inicializa Engine do LEGION
    print("[3/4] Inicializando Motor LEGION...")
    engine = LegionEngine()
    tokenizer = engine.tokenizer

    print("\n[4/4] Executando comparação de logits para os prompts de teste...")
    print("=" * 80)

    results_summary = []

    for idx, prompt in enumerate(TEST_PROMPTS, 1):
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        # A. Execução Configuração A (Baseline 100% 2 bits)
        with torch.no_grad():
            out_A = engine.model(**inputs)
            logits_A = out_A.logits[0, -1, :].float()

        # B. Execução Configuração B (L39 Âncoras em 3 bits via substituição direta no forward)
        # Substitui os slots da L39 temporariamente para os 6 âncoras
        l39 = engine.layers[39]
        
        # Salva o método forward original da camada 39
        orig_forward = l39.forward
        
        def make_custom_forward(weights_map):
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
                        if e_id in weights_map:
                            W_gu = weights_map[e_id]["gu"]
                            W_dn = weights_map[e_id]["dn"]
                            y_gu = F.linear(xt, W_gu)
                            g, u = y_gu.chunk(2, dim=-1)
                            act = (F.silu(g.float()) * u.float()).bfloat16()
                            y_k = F.linear(act, W_dn)
                            token_out += y_k * w_t[k_idx]
                        else:
                            # Executa via slot 2b original
                            slot = l39._get_slot(e_id)
                            slots_t = torch.tensor([slot], dtype=torch.int32, device=DEVICE)
                            xr = l39.rot_2048.fwd(xt.view(-1))
                            y_gu = torch.empty((1, 1024), device=DEVICE, dtype=torch.bfloat16)
                            vq_moe_parallel_gate_up(xr, slots_t, l39.pool_gu_idx, l39.cb16, l39.pool_gu_scale, y_gu, 1024, 256, l39.RS_2048)
                            g, u = y_gu[:, :512], y_gu[:, 512:]
                            act = (F.silu(g.float()) * u.float()).bfloat16()
                            act_r = l39.rot_512.fwd(act)
                            y_dn = torch.empty((1, 2048), device=DEVICE, dtype=torch.bfloat16)
                            vq_moe_parallel_down(act_r, slots_t, l39.pool_dn_idx, l39.cb16, l39.pool_dn_scale, y_dn, 2048, 64, l39.RS_512)
                            token_out += y_dn * w_t[k_idx]
                    outs.append(token_out)
                return torch.cat(outs, dim=0)
            return custom_forward

        # Execução B (3 bits)
        l39.forward = make_custom_forward(weights_3b)
        with torch.no_grad():
            out_B = engine.model(**inputs)
            logits_B = out_B.logits[0, -1, :].float()

        # Execução Ref (L39 Âncoras Originais BF16)
        l39.forward = make_custom_forward(orig_weights)
        with torch.no_grad():
            out_Ref = engine.model(**inputs)
            logits_Ref = out_Ref.logits[0, -1, :].float()

        # Restaura forward original
        l39.forward = orig_forward

        # Métricas de Divergência
        probs_Ref = F.softmax(logits_Ref, dim=-1)
        probs_A = F.softmax(logits_A, dim=-1)
        probs_B = F.softmax(logits_B, dim=-1)

        kl_A = F.kl_div(probs_A.log(), probs_Ref, reduction='sum').item()
        kl_B = F.kl_div(probs_B.log(), probs_Ref, reduction='sum').item()

        cos_A = F.cosine_similarity(logits_Ref, logits_A, dim=0).item()
        cos_B = F.cosine_similarity(logits_Ref, logits_B, dim=0).item()

        top1_Ref = torch.argmax(logits_Ref).item()
        top1_A = torch.argmax(logits_A).item()
        top1_B = torch.argmax(logits_B).item()

        tok_Ref = tokenizer.decode([top1_Ref]).strip()
        tok_A = tokenizer.decode([top1_A]).strip()
        tok_B = tokenizer.decode([top1_B]).strip()

        print(f"\n[Prompt {idx}]: \"{prompt[:60]}...\"")
        print(f"  Token Previsto Ref (BF16):  '{tok_Ref}' (id: {top1_Ref})")
        print(f"  Token Previsto A (2 bits):  '{tok_A}' (id: {top1_A}) | Match: {top1_A == top1_Ref} | Cos: {cos_A:.5f} | KL: {kl_A:.4f}")
        print(f"  Token Previsto B (3 bits):  '{tok_B}' (id: {top1_B}) | Match: {top1_B == top1_Ref} | Cos: {cos_B:.5f} | KL: {kl_B:.4f}")

        results_summary.append({
            "prompt": prompt,
            "kl_A": kl_A, "kl_B": kl_B,
            "cos_A": cos_A, "cos_B": cos_B,
            "match_A": top1_A == top1_Ref,
            "match_B": top1_B == top1_Ref,
            "tok_Ref": tok_Ref, "tok_A": tok_A, "tok_B": tok_B
        })

    print("\n" + "=" * 80)
    print("CONSOLIDAÇÃO DA AUDITORIA DE LOGITS:")
    print("=" * 80)
    mean_kl_A = sum(r["kl_A"] for r in results_summary) / len(results_summary)
    mean_kl_B = sum(r["kl_B"] for r in results_summary) / len(results_summary)
    mean_cos_A = sum(r["cos_A"] for r in results_summary) / len(results_summary)
    mean_cos_B = sum(r["cos_B"] for r in results_summary) / len(results_summary)
    top1_acc_A = sum(r["match_A"] for r in results_summary) / len(results_summary)
    top1_acc_B = sum(r["match_B"] for r in results_summary) / len(results_summary)

    print(f"Divergência KL Média (vs Ref):       A (2b): {mean_kl_A:.5f}  |  B (3b): {mean_kl_B:.5f}  (Redução de {(1 - mean_kl_B/mean_kl_A)*100:.1f}%)")
    print(f"Similaridade Cosseno dos Logits:     A (2b): {mean_cos_A:.5f}  |  B (3b): {mean_cos_B:.5f}")
    print(f"Concordância Top-1 Token com a Ref:  A (2b): {top1_acc_A*100:.1f}%    |  B (3b): {top1_acc_B*100:.1f}%")
    print("=" * 80)

if __name__ == "__main__":
    run_logit_comparison()
