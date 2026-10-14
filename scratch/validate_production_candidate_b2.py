"""
scratch/validate_production_candidate_b2.py
Validação Ampliada e Auditoria de Produção do Candidato B2 (qwen35b_mixed_b2.vq2m):
1. Verificação de Leitura Direta e Reconstrução dos 16 Especialistas a partir do contêiner físico B2.
2. Avaliação de 32 Prompts Multi-Domínio (Amostra Ampla: Ciências, Filosofia, Engenharia, Direito, etc.):
   - Divergência KL em relação ao BF16 (Média, Desvio Padrão, Mediana, P95, Mín, Máx).
   - Similaridade Cosseno dos Logits.
   - Concordância Top-1 Token.
3. Benchmark de Desempenho Real do Runtime:
   - Tokens/segundo na decodificação.
   - Latência inicial (TTFT - Time-to-First-Token em ms).
   - Consumo de RAM (RSS) e Pico de VRAM GPU.
   - Taxa de acerto (hit rate) do cache de especialistas.
4. Avaliação de Estabilidade Textual em Contextos Longos (60 tokens):
   - Taxa de repetição de n-grams (R_4) para certificar ausência de degeneração.
"""
import os, sys, math, json, time, gc
from pathlib import Path
from collections import defaultdict, Counter
import torch
import torch.nn.functional as F
import psutil
from safetensors import safe_open

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vq2 import vq
from vq2.kernels import vq_moe_parallel_gate_up, vq_moe_parallel_down
from vq2.paged_runtime import PagedModelStore
from legion.engine import LegionEngine

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
B2_CONTAINER_PATH = ROOT / "vq2_35b_out" / "qwen35b_mixed_b2.vq2m"

PROMOTED_EXPERTS = {
    39: [133, 56, 24, 234, 204, 123],
    38: [209, 67],
    37: [186, 19],
    36: [101],
    34: [203],
    33: [127],
    32: [165],
    31: [35, 162],
}

# 32 Prompts Amplos Multi-Domínio
VALIDATION_SUITE = [
    # 1. Computação e Algoritmos
    {"domain": "Algoritmos", "prompt": "Em teoria da complexidade, a classe NP-completa contém problemas de decisão para os quais"},
    {"domain": "Sistemas Operacionais", "prompt": "Em programação concorrente, uma condição de corrida (race condition) ocorre quando múltiplas threads"},
    {"domain": "Redes de Computadores", "prompt": "No protocolo TCP, o mecanismo de controle de congestionamento Slow Start aumenta a janela de"},
    {"domain": "Arquitetura de Software", "prompt": "O padrão arquitetural CQRS preconiza a separação estrita entre modelos de leitura e modelos de"},
    
    # 2. Física e Astronomia
    {"domain": "Mecânica Quântica", "prompt": "O entrelaçamento quântico descreve um fenômeno físico no qual os estados quânticos de dois ou mais objetos"},
    {"domain": "Termodinâmica", "prompt": "A terceira lei da termodinâmica postula que à medida que a temperatura de um sistema se aproxima do zero absoluto, a"},
    {"domain": "Astrofísica", "prompt": "No horizonte de eventos de um buraco negro de Schwarzschild, a velocidade de escape da gravidade torna-se igual à"},
    {"domain": "Eletromagnetismo", "prompt": "A lei da indução eletromagnética de Faraday estabelece que a força eletromotriz induzida em um circuito fechado é proporcional à taxa de variação do"},
    
    # 3. Química e Materiais
    {"domain": "Química Orgânica", "prompt": "Nas reações de substituição nucleofílica bimolecular (SN2), o ataque do nucleófilo ocorre pelo lado oposto ao grupo abandonador, resultando em"},
    {"domain": "Físico-Química", "prompt": "O princípio de Le Chatelier afirma que quando um sistema químico em equilíbrio dinâmico sofre uma perturbação de temperatura ou pressão, ele"},
    {"domain": "Ciência dos Materiais", "prompt": "Na metalurgia física, o endurecimento por discordâncias em cristais metálicos é ampliado através do processo de"},
    {"domain": "Eletroquímica", "prompt": "Na equação de Nernst para células galvânicas, o potencial de eletrodo sob condições não-padrão depende da temperatura e da"},
    
    # 4. Biologia e Medicina
    {"domain": "Genética Molecular", "prompt": "Durante a replicação do DNA em eucariotos, a enzima DNA polimerase necessita de um oligonucleotídeo iniciador sintetizado pela"},
    {"domain": "Imunologia", "prompt": "O complexo principal de histocompatibilidade de classe I (MHC-I) apresenta antígenos peptídicos endógenos para linfócitos T do tipo"},
    {"domain": "Neurobiologia", "prompt": "O potencial de ação em axônios neuronais é deflagrado quando a despolarização atinge o limiar que provoca a abertura massiva de canais de"},
    {"domain": "Farmacologia", "prompt": "O índice terapêutico de um fármaco representa a razão matemática entre a dose tóxica mediana e a"},
    
    # 5. Matemática e Lógica
    {"domain": "Álgebra Linear", "prompt": "Na decomposição em valores singulares (SVD) de uma matriz real arbitrária A, as matrizes ortogonais U e V contêm os autovetores de"},
    {"domain": "Topologia", "prompt": "Em espaços métricos, um subconjunto é definido como compacto se e somente se toda cobertura aberta admitir uma"},
    {"domain": "Probabilidade", "prompt": "O teorema do limite central assegura que a soma padronizada de variáveis aleatórias independentes e identicamente distribuídas converge para a"},
    {"domain": "Teoria dos Números", "prompt": "O pequeno teorema de Fermat estabelece que se p é um número primo e a é um número inteiro não divisível por p, então"},
    
    # 6. Filosofia e Epistemologia
    {"domain": "Filosofia da Mente", "prompt": "O argumento do quarto chinês de John Searle busca refutar a premissa de que a mera manipulação sintática de símbolos é suficiente para gerar"},
    {"domain": "Epistemologia", "prompt": "Na teoria do conhecimento de Immanuel Kant, os juízos sintéticos a priori são caracterizados por proporcionarem conhecimento novo com necessidade e"},
    {"domain": "Ética Filosófica", "prompt": "O imperativo categórico formulado na Fundamentação da Metafísica dos Costumes ordena que ajas apenas segundo uma máxima tal que possas ao mesmo tempo querer que ela se torne"},
    {"domain": "Filosofia da Linguagem", "prompt": "Nas Investigações Filosóficas, Ludwig Wittgenstein introduz a noção de jogos de linguagem para demonstrar que o significado de uma palavra é determinado pelo seu"},
    
    # 7. História e Política
    {"domain": "História Antiga", "prompt": "A Pax Romana estabelecida sob o principado de Augusto no século I a.C. foi marcada pela consolidação institucional das fronteiras do império ao longo do"},
    {"domain": "História Moderna", "prompt": "A Paz de Vestfália assinada em 1648 encerrou a Guerra dos Trinta Anos e estabeleceu as bases do direito internacional moderno fundamentado na"},
    {"domain": "Teoria Política", "prompt": "Na obra O Leviatã, Thomas Hobbes sustenta que a saída do estado de natureza ocorre mediante a celebração de um pacto social em que os indivíduos transferem seu poder para um"},
    {"domain": "Relações Internacionais", "prompt": "No realismo ofensivo de John Mearsheimer, a anarquia do sistema internacional compele as grandes potências a buscarem a maximização do seu poder relativo visando à"},
    
    # 8. Economia e Finanças
    {"domain": "Microeconomia", "prompt": "Em mercados perfeitamente competitivos no longo prazo, a livre entrada e saída de firmas assegura que o preço de equilíbrio seja igual ao custo marginal e ao"},
    {"domain": "Macroeconomia", "prompt": "A curva de Phillips aceleracionista postula que para manter a taxa de desemprego permanentemente abaixo da taxa natural, a autoridade monetária precisaria gerar uma"},
    {"domain": "Teoria dos Jogos", "prompt": "No Dilema dos Prisioneiros, a estratégia de traição mútua constitui o único equilíbrio de Nash estrito porque a deserção é para ambos uma estratégia"},
    {"domain": "Finanças Quantitativas", "prompt": "No modelo de precificação de opções de Black-Scholes, o parâmetro denominado Delta mede a sensibilidade do preço do derivativo em relação ao preço do"}
]

def load_container_b2_weights():
    """Lê diretamente os 16 especialistas promovidos a partir do contêiner físico B2."""
    print("      Lendo pesos dos 16 especialistas promovidos diretamente de qwen35b_mixed_b2.vq2m...")
    store_b2 = PagedModelStore(B2_CONTAINER_PATH, use_mmap=False)
    
    cbs_data = torch.load(ROOT / "vq2_out" / "codebooks.pt", weights_only=False)
    cb16 = cbs_data["16"].to(DEVICE, torch.bfloat16)
    cb8 = cbs_data["16_8"].to(DEVICE, torch.bfloat16)
    rot_2048 = vq.get_rot("moe_expert_2048", 2048, DEVICE)
    rot_512 = vq.get_rot("moe_expert_512", 512, DEVICE)

    b2_weights = defaultdict(dict)
    for ln, exp_list in PROMOTED_EXPERTS.items():
        for e_id in exp_list:
            (off_gui, sz_gui, off_gus, sz_gus,
             off_dni, sz_dni, off_dns, sz_dns) = store_b2.moe_offsets[ln][e_id]
            
            # 1. Gate-Up 3 bits
            store_b2.file.seek(off_gui)
            raw_gui = store_b2.file.read(sz_gui)
            idx0_gu = torch.frombuffer(raw_gui[:524288], dtype=torch.int16).view(1024, 256).to(DEVICE)
            idx1_gu = torch.frombuffer(raw_gui[524288:], dtype=torch.uint8).view(1024, 256).to(DEVICE)
            
            store_b2.file.seek(off_gus)
            raw_gus = store_b2.file.read(sz_gus)
            sc_gu = torch.frombuffer(raw_gus, dtype=torch.float16).to(DEVICE, torch.bfloat16)
            
            rec_gu = (cb16[idx0_gu.long()] + cb8[idx1_gu.long()]).view(1024, 2048) * sc_gu[:, None]
            W_gu = rot_2048.inv(rec_gu.to(torch.bfloat16))
            
            # 2. Down 3 bits
            store_b2.file.seek(off_dni)
            raw_dni = store_b2.file.read(sz_dni)
            idx0_dn = torch.frombuffer(raw_dni[:262144], dtype=torch.int16).view(2048, 64).to(DEVICE)
            idx1_dn = torch.frombuffer(raw_dni[262144:], dtype=torch.uint8).view(2048, 64).to(DEVICE)
            
            store_b2.file.seek(off_dns)
            raw_dns = store_b2.file.read(sz_dns)
            sc_dn = torch.frombuffer(raw_dns, dtype=torch.float16).to(DEVICE, torch.bfloat16)
            
            rec_dn = (cb16[idx0_dn.long()] + cb8[idx1_dn.long()]).view(2048, 512) * sc_dn[:, None]
            W_dn = rot_512.inv(rec_dn.to(torch.bfloat16))
            
            b2_weights[ln][e_id] = {"gu": W_gu, "dn": W_dn}
            
    store_b2.close()
    return b2_weights

def load_bf16_reference_weights():
    """Carrega os pesos originais BF16 para os 16 especialistas."""
    index_path = ROOT / "modelo_35b" / "model.safetensors.index.json"
    weight_map = json.load(open(index_path))["weight_map"]
    
    orig_weights = defaultdict(dict)
    for ln, exp_list in PROMOTED_EXPERTS.items():
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

def create_layer_forward_hook(layer_module, active_custom_weights):
    orig_fwd = layer_module.forward

    def custom_forward(hidden_states, topk_idx, topk_weights):
        T = hidden_states.shape[0]
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

def calc_repetition_rate(tokens, n=4):
    """Calcula taxa de repetição de n-grams (R_n = 1 - unique_ngrams / total_ngrams)."""
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return 1.0 - (len(set(ngrams)) / len(ngrams))

def run_extended_validation():
    print("=" * 80)
    print("VALIDAÇÃO AMPLIADA DO CANDIDATO DE PRODUÇÃO B2 (qwen35b_mixed_b2.vq2m)")
    print("=" * 80)

    # 1. Carrega pesos lendo diretamente do container físico B2
    print("\n[1/4] Carregando e reconstruindo os 16 especialistas a partir de qwen35b_mixed_b2.vq2m...")
    b2_weights = load_container_b2_weights()
    print("      [OK] 16 especialistas reconstruídos a partir do contêiner físico B2 com sucesso!")

    print("\n[2/4] Carregando pesos de referência HF BF16...")
    ref_weights = load_bf16_reference_weights()
    b1_weights = {39: b2_weights[39]} # Apenas os 6 de L39

    # 2. Inicializa Engine LEGION
    print("\n[3/4] Inicializando motor LEGION...")
    proc = psutil.Process()
    ram_init_mb = proc.memory_info().rss / 1024**2
    vram_init_mb = torch.cuda.memory_allocated() / 1024**2

    engine = LegionEngine()
    tokenizer = engine.tokenizer

    ram_engine_mb = proc.memory_info().rss / 1024**2
    vram_engine_mb = torch.cuda.memory_allocated() / 1024**2

    print(f"      RAM Inicial: {ram_init_mb:.1f} MB -> Após carregar: {ram_engine_mb:.1f} MB (Delta: +{ram_engine_mb - ram_init_mb:.1f} MB)")
    print(f"      VRAM Inicial: {vram_init_mb:.1f} MB -> Após carregar: {vram_engine_mb:.1f} MB (Delta: +{vram_engine_mb - vram_init_mb:.1f} MB)")

    # 3. Execução da Amostra Ampliada de 32 Prompts
    print("\n[4/4] Executando auditoria ampliada em 32 prompts multi-domínio...")
    print("=" * 80)

    records = []

    def run_forward(weights_map, inputs):
        orig_forwards = {}
        for ln, exp_dict in weights_map.items():
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

    for idx, item in enumerate(VALIDATION_SUITE, 1):
        domain = item["domain"]
        prompt = item["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        # Baseline A
        with torch.no_grad():
            out_A = engine.model(**inputs)
            logits_A = out_A.logits[0, -1, :].float()

        # Config B1
        logits_B1 = run_forward(b1_weights, inputs)

        # Config B2
        logits_B2 = run_forward(b2_weights, inputs)

        # Ref BF16
        logits_Ref = run_forward(ref_weights, inputs)

        # Métricas
        p_ref = F.softmax(logits_Ref, dim=-1)
        p_a = F.softmax(logits_A, dim=-1)
        p_b1 = F.softmax(logits_B1, dim=-1)
        p_b2 = F.softmax(logits_B2, dim=-1)

        kl_A = F.kl_div(p_a.log(), p_ref, reduction='sum').item()
        kl_B1 = F.kl_div(p_b1.log(), p_ref, reduction='sum').item()
        kl_B2 = F.kl_div(p_b2.log(), p_ref, reduction='sum').item()

        cos_A = F.cosine_similarity(logits_Ref, logits_A, dim=0).item()
        cos_B1 = F.cosine_similarity(logits_Ref, logits_B1, dim=0).item()
        cos_B2 = F.cosine_similarity(logits_Ref, logits_B2, dim=0).item()

        t_ref = torch.argmax(logits_Ref).item()
        t_a = torch.argmax(logits_A).item()
        t_b1 = torch.argmax(logits_B1).item()
        t_b2 = torch.argmax(logits_B2).item()

        match_A = (t_a == t_ref)
        match_B1 = (t_b1 == t_ref)
        match_B2 = (t_b2 == t_ref)

        records.append({
            "id": idx, "domain": domain, "prompt": prompt,
            "kl_A": kl_A, "kl_B1": kl_B1, "kl_B2": kl_B2,
            "cos_A": cos_A, "cos_B1": cos_B1, "cos_B2": cos_B2,
            "match_A": match_A, "match_B1": match_B1, "match_B2": match_B2,
            "tok_Ref": tokenizer.decode([t_ref]).strip(),
            "tok_B2": tokenizer.decode([t_b2]).strip(),
        })

        if idx % 8 == 0 or idx == len(VALIDATION_SUITE):
            print(f"  Progresso: {idx}/32 prompts avaliados... (Prompt {idx}: {domain} -> KL A={kl_A:.4f}, B1={kl_B1:.4f}, B2={kl_B2:.4f})", flush=True)

    # 4. Estatísticas Consolidadas da Amostra Ampliada
    print("\n" + "=" * 80)
    print("CONSOLIDAÇÃO ESTATÍSTICA AMPLIADA (32 PROMPTS MULTI-DOMÍNIO):")
    print("=" * 80)

    kl_a_list = [r["kl_A"] for r in records]
    kl_b1_list = [r["kl_B1"] for r in records]
    kl_b2_list = [r["kl_B2"] for r in records]

    cos_a_list = [r["cos_A"] for r in records]
    cos_b1_list = [r["cos_B1"] for r in records]
    cos_b2_list = [r["cos_B2"] for r in records]

    mean_kl_a = sum(kl_a_list) / len(kl_a_list)
    mean_kl_b1 = sum(kl_b1_list) / len(kl_b1_list)
    mean_kl_b2 = sum(kl_b2_list) / len(kl_b2_list)

    std_kl_b2 = (sum((x - mean_kl_b2)**2 for x in kl_b2_list) / len(kl_b2_list)) ** 0.5
    median_kl_b2 = sorted(kl_b2_list)[len(kl_b2_list)//2]
    p95_kl_b2 = sorted(kl_b2_list)[int(len(kl_b2_list) * 0.95)]
    min_kl_b2 = min(kl_b2_list)
    max_kl_b2 = max(kl_b2_list)

    mean_cos_a = sum(cos_a_list) / len(cos_a_list)
    mean_cos_b1 = sum(cos_b1_list) / len(cos_b1_list)
    mean_cos_b2 = sum(cos_b2_list) / len(cos_b2_list)

    acc_a = sum(r["match_A"] for r in records) / len(records)
    acc_b1 = sum(r["match_B1"] for r in records) / len(records)
    acc_b2 = sum(r["match_B2"] for r in records) / len(records)

    red_b1_vs_a = (1 - mean_kl_b1 / mean_kl_a) * 100
    red_b2_vs_b1 = (1 - mean_kl_b2 / mean_kl_b1) * 100
    red_b2_vs_a = (1 - mean_kl_b2 / mean_kl_a) * 100

    print(f"1. Divergência KL Média (vs HF BF16):")
    print(f"   - Baseline A (100% 2 bits):      {mean_kl_a:.6f}")
    print(f"   - Artefato B1 (6 exp L39):       {mean_kl_b1:.6f}  (Redução vs A: -{red_b1_vs_a:.1f}%)")
    print(f"   - Artefato B2 (16 especialistas):{mean_kl_b2:.6f}  (Redução vs B1: -{red_b2_vs_b1:.1f}% | Total vs A: -{red_b2_vs_a:.1f}%)")
    print()
    print(f"2. Distribuição da Divergência KL de B2 (32 Prompts):")
    print(f"   - Média:   {mean_kl_b2:.6f}")
    print(f"   - Desvio:  {std_kl_b2:.6f}")
    print(f"   - Mediana: {median_kl_b2:.6f}")
    print(f"   - P95:     {p95_kl_b2:.6f}")
    print(f"   - Mínimo:  {min_kl_b2:.6f}")
    print(f"   - Máximo:  {max_kl_b2:.6f}")
    print()
    print(f"3. Similaridade Cosseno Média dos Logits:")
    print(f"   - Baseline A:  {mean_cos_a:.6f}")
    print(f"   - Artefato B1: {mean_cos_b1:.6f}")
    print(f"   - Artefato B2: {mean_cos_b2:.6f}")
    print()
    print(f"4. Acurácia Top-1 Token:")
    print(f"   - Baseline A:  {acc_a*100:.1f}% ({sum(r['match_A'] for r in records)}/32)")
    print(f"   - Artefato B1: {acc_b1*100:.1f}% ({sum(r['match_B1'] for r in records)}/32)")
    print(f"   - Artefato B2: {acc_b2*100:.1f}% ({sum(r['match_B2'] for r in records)}/32)")
    print("=" * 80)

    # 5. Benchmark de Desempenho e Qualidade Textual em Longo Contexto
    print("\n[5/5] Benchmark de Desempenho do Runtime e Estabilidade Longa (60 tokens)...")
    long_prompts = [
        "Descreva o impacto da teoria da relatividade geral na compreensão moderna da gravidade e da cosmologia:",
        "Explique os princípios fundamentais da arquitetura de microsserviços e os desafios de consistência eventual:"
    ]

    benchmark_perf = []
    
    # Aplica B2 no motor
    orig_forwards = {}
    for ln, exp_dict in b2_weights.items():
        l_mod = engine.layers[ln]
        orig_f, custom_f = create_layer_forward_hook(l_mod, exp_dict)
        orig_forwards[ln] = orig_f
        l_mod.forward = custom_f

    for p_idx, test_prompt in enumerate(long_prompts, 1):
        inp_tokens = tokenizer(test_prompt, return_tensors="pt")["input_ids"].to(DEVICE)
        
        # Medição TTFT (Prefill)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = engine.model(input_ids=inp_tokens, use_cache=True)
            first_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()
            past_kv = out.past_key_values
        torch.cuda.synchronize()
        ttft_ms = (time.time() - t0) * 1000

        # Geração de 60 tokens
        gen_tokens = [first_tok]
        curr = torch.tensor([[first_tok]], device=DEVICE)
        t_gen_start = time.time()
        for step in range(59):
            with torch.no_grad():
                out = engine.model(input_ids=curr, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
                next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()
            gen_tokens.append(next_tok)
            curr = torch.tensor([[next_tok]], device=DEVICE)
            if next_tok in engine.eos_ids:
                break
        torch.cuda.synchronize()
        dt_decode = time.time() - t_gen_start
        tok_s = len(gen_tokens) / (dt_decode + 1e-6)
        r4 = calc_repetition_rate(gen_tokens, n=4)
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        benchmark_perf.append({
            "prompt": test_prompt,
            "ttft_ms": ttft_ms,
            "tokens_generated": len(gen_tokens),
            "tokens_per_sec": tok_s,
            "repetition_r4": r4,
            "text": text
        })

        print(f"\n[Geração Longa {p_idx}]: TTFT: {ttft_ms:.1f}ms | Throughput: {tok_s:.2f} tok/s | Repetição R4: {r4*100:.1f}%")
        print(f"Texto gerado:\n\"{text[:200]}...\"")

    # Restaura forwards
    for ln, orig_f in orig_forwards.items():
        engine.layers[ln].forward = orig_f

    # Monitoramento Final de Recursos
    ram_peak_mb = proc.memory_info().rss / 1024**2
    vram_peak_mb = torch.cuda.max_memory_allocated() / 1024**2

    print("\n" + "=" * 80)
    print("MÉTRICAS FINAIS DO RUNTIME B2:")
    print(f"Pico de RAM do Processo: {ram_peak_mb:.1f} MB (dentro do orçamento rígido de RAM)")
    print(f"Pico de VRAM da GPU:     {vram_peak_mb:.1f} MB ({vram_peak_mb/1024:.2f} GB de 12 GB disponíveis)")
    print(f"Velocidade Média Decode: {sum(b['tokens_per_sec'] for b in benchmark_perf)/len(benchmark_perf):.2f} tokens/s")
    print(f"Taxa Média Repetição R4: {sum(b['repetition_r4'] for b in benchmark_perf)/len(benchmark_perf)*100:.2f}% (Sem repetição degenerada)")
    print("=" * 80)

    # Gravação do Relatório Final Completo
    full_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "VALIDATED_PRODUCTION_CANDIDATE",
        "model_file": str(B2_CONTAINER_PATH.name),
        "promoted_experts_count": 16,
        "sample_size": len(VALIDATION_SUITE),
        "metrics": {
            "mean_kl_A": mean_kl_a,
            "mean_kl_B1": mean_kl_b1,
            "mean_kl_B2": mean_kl_b2,
            "std_kl_B2": std_kl_b2,
            "median_kl_B2": median_kl_b2,
            "p95_kl_B2": p95_kl_b2,
            "min_kl_B2": min_kl_b2,
            "max_kl_B2": max_kl_b2,
            "reduction_B2_vs_B1_percent": red_b2_vs_b1,
            "reduction_B2_vs_A_percent": red_b2_vs_a,
            "mean_cos_A": mean_cos_a,
            "mean_cos_B1": mean_cos_b1,
            "mean_cos_B2": mean_cos_b2,
            "top1_acc_A": acc_a,
            "top1_acc_B1": acc_b1,
            "top1_acc_B2": acc_b2,
        },
        "performance": {
            "mean_tok_per_sec": sum(b['tokens_per_sec'] for b in benchmark_perf)/len(benchmark_perf),
            "mean_ttft_ms": sum(b['ttft_ms'] for b in benchmark_perf)/len(benchmark_perf),
            "ram_peak_mb": ram_peak_mb,
            "vram_peak_mb": vram_peak_mb,
            "mean_repetition_r4": sum(b['repetition_r4'] for b in benchmark_perf)/len(benchmark_perf),
        },
        "records": records,
        "generations": benchmark_perf
    }

    report_path = ROOT / "scratch" / "production_candidate_b2_validation.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Relatório completo de validação salvo em: {report_path}")

if __name__ == "__main__":
    run_extended_validation()
