"""
scratch/verify_anchor_experts.py
Verifica a hipótese dos 'Especialistas-Âncora' da Camada 39 em 10 prompts totalmente distintos.
"""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legion.engine import LegionEngine

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DIVERSE_PROMPTS = [
    # 1. Poesia e Literatura
    "Escreva um soneto barroco sobre a brevidade da vida e a passagem do tempo.",
    # 2. Programação e SQL
    "SELECT u.id, u.name, COUNT(o.id) FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id HAVING COUNT(o.id) > 5;",
    # 3. Matemática e Álgebra Linear
    "Seja A uma matriz simétrica real de dimensão n x n. Prove que todos os autovalores de A são números reais.",
    # 4. Medicina e Biologia
    "Explique o mecanismo fisiopatológico da cetoacidose diabética e a relação com a deficiência de insulina e corpos cetônicos.",
    # 5. Culinária e Receita
    "Receita tradicional de pão de queijo mineiro com polvilho azedo, queijo da canastra e leite.",
    # 6. Comandos de Sistema / Linux
    "find /var/log -type f -name '*.log' -mtime +30 -exec gzip {} \\; && df -h",
    # 7. Filosofia Moral
    "Compare o imperativo categórico de Immanuel Kant com a teoria ética utilitarista de John Stuart Mill.",
    # 8. Diálogo Curto
    "Olá! Como você está se sentindo hoje? Pode me ajudar a planejar minha rotina de estudos?",
    # 9. Física Quântica
    "Explique o paradoxo EPR e as desigualdades de Bell no contexto do entrelaçamento quântico.",
    # 10. Economia e Finanças
    "Qual é o impacto de uma elevação da taxa básica de juros (Selic) sobre a inflação e o crédito no Brasil?"
]

def verify_anchors():
    print("=" * 80)
    print("VERIFICAÇÃO DE ROBUSTEZ: ESPECIALISTAS-ÂNCORA DA CAMADA 39")
    print(f"Testando 10 domínios completamente heterogêneos...")
    print("=" * 80)

    engine = LegionEngine()
    
    # Captura ativações de entrada da Camada 39
    x_l39 = []
    def hook_fn(module, args, output):
        x = args[0].detach()
        x_l39.append(x.view(-1, x.shape[-1]))

    h = engine.model.model.layers[39].mlp.experts.register_forward_hook(hook_fn)

    with torch.no_grad():
        for i, text in enumerate(DIVERSE_PROMPTS):
            inp = engine.tokenizer(text, return_tensors="pt").to(DEVICE)
            engine.model(**inp)

    h.remove()

    all_x = torch.cat(x_l39, dim=0) # [Total_Tokens, 2048]
    total_tokens = all_x.shape[0]

    # Distribuição do Roteador L39
    gate_w = engine.model.model.layers[39].mlp.gate.weight
    logits = F.linear(all_x.bfloat16(), gate_w.bfloat16())
    probs = F.softmax(logits.float(), dim=-1)
    topk_w, topk_ids = torch.topk(probs, k=8, dim=-1)

    print(f"\nTotal de Tokens Analisados: {total_tokens} em 10 domínios distintos.")
    print("-" * 80)
    print(f"{'Especialista L39':<18} | {'Freq. Ativação':<16} | {'Contagem':<10} | {'Prob. Média':<12}")
    print("-" * 80)

    target_experts = [133, 24, 56, 234, 204, 123, 144, 255]
    for e_id in target_experts:
        mask = (topk_ids == e_id).any(dim=-1)
        count = mask.sum().item()
        freq = count / total_tokens
        mean_p = probs[:, e_id].mean().item()
        print(f"Expert {e_id:>3}          | {freq * 100:>13.1f}% | {count:>8} | {mean_p:>10.4f}")

    print("=" * 80)

if __name__ == "__main__":
    verify_anchors()
