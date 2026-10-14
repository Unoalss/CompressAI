import sys, time
sys.path.insert(0, r"e:\compressAI")
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeTextConfig
)
from transformers.cache_utils import DynamicCache
from accelerate import init_empty_weights

from vq2 import vq
from vq2.kernels import patch_transformers_kernels
from vq2.load_and_test_full_35b import (
    PackedMoEExpertUniversal,
    dequant_matrix,
    set_nested_param
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = r"e:\compressAI\modelo_35b"

patch_transformers_kernels()

print("1. Carregando modelo e tokenizer...")
tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
EOS_IDS = {tok.eos_token_id, 248044, 248045}

cfg = AutoConfig.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg.text_config
with init_empty_weights():
    model = Qwen3_5MoeForCausalLM(text_cfg)

cbs_data = torch.load("e:/compressAI/vq2_out/codebooks.pt", weights_only=False)
cb16 = cbs_data["16"].to(device=DEVICE, dtype=torch.bfloat16)

import json
with open("e:/compressAI/modelo_35b/model.safetensors.index.json") as f:
    shards_to_load = sorted(set(json.load(f)["weight_map"].values()))

layers_experts_loaded = set()
pending_gate_up = {}
pending_down = {}

for shard_name in shards_to_load:
    pt_path = f"e:/compressAI/vq2_35b_out/quant_{shard_name}.pt"
    try:
        s = torch.load(pt_path, weights_only=False, map_location="cpu")
    except Exception:
        continue
    quant_dict = s.get("quant", {})
    dense_dict = s.get("dense", {})

    for k, q_data in quant_dict.items():
        if "mlp.experts.gate_up_proj" in k:
            m = k.split(".layers.")
            if len(m) > 1: pending_gate_up[int(m[1].split(".")[0])] = q_data
        elif "mlp.experts.down_proj" in k:
            m = k.split(".layers.")
            if len(m) > 1: pending_down[int(m[1].split(".")[0])] = q_data

    for ln in list(pending_gate_up.keys()):
        if ln in pending_down and ln not in layers_experts_loaded:
            expert_mod = PackedMoEExpertUniversal(pending_gate_up[ln], pending_down[ln], cb16, device=DEVICE)
            model.model.layers[ln].mlp.experts = expert_mod
            layers_experts_loaded.add(ln)
            del pending_gate_up[ln], pending_down[ln]

    for raw_name, tensor in dense_dict.items():
        clean_name = raw_name.replace("model.language_model.", "model.")
        set_nested_param(model, clean_name, tensor.to(device=DEVICE, dtype=torch.bfloat16))

    for raw_name, q_data in quant_dict.items():
        if "mlp.experts" in raw_name: continue
        clean_name = raw_name.replace("model.language_model.", "model.")
        dev = "cpu" if "embed_tokens" in clean_name else DEVICE
        W = dequant_matrix(q_data, cbs_data, device=dev)
        set_nested_param(model, clean_name, W)

model.model.rotary_emb.to(DEVICE)
model.eval()
lm_head = model.lm_head.weight

def find_candidate_tokens(token_list, ngram_size=3, max_candidates=5):
    """Busca subsequência n-gram na história para especulação."""
    if len(token_list) < ngram_size:
        return []
    suffix = token_list[-ngram_size:]
    
    # Procura a última ocorrência antes do sufixo atual
    for i in range(len(token_list) - ngram_size - 1, -1, -1):
        if token_list[i:i + ngram_size] == suffix:
            start_cand = i + ngram_size
            candidates = token_list[start_cand:start_cand + max_candidates]
            if candidates:
                return candidates
    return []

def speculative_generate(prompt: str, max_new_tokens: int = 60, K: int = 5):
    messages = [
        {"role": "system", "content": "Você é um assistente de IA prestativo e responde em português."},
        {"role": "user", "content": prompt}
    ]
    formatted = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    input_ids = tok.encode(formatted)
    
    cache = DynamicCache(config=text_cfg)
    
    print(f"\n--- PROMPT: {prompt} ---")
    print("Gerando: ", end="", flush=True)
    
    # 1. Prefill
    t0 = time.perf_counter()
    prompt_tensor = torch.tensor([input_ids], device="cpu")
    prompt_emb = model.model.embed_tokens(prompt_tensor).to(device=DEVICE, dtype=torch.bfloat16)
    with torch.no_grad():
        out = model.model(inputs_embeds=prompt_emb, past_key_values=cache, use_cache=True)
        logits = F.linear(out.last_hidden_state[:, -1:, :], lm_head).squeeze(0).squeeze(0)
    
    first_tok = logits.argmax().item()
    all_tokens = list(input_ids) + [first_tok]
    generated_tokens = [first_tok]
    print(tok.decode([first_tok]), end="", flush=True)
    
    t_decode_start = time.perf_counter()
    num_steps = 0
    accepted_speculative = 0
    
    while len(generated_tokens) < max_new_tokens:
        last_tok = all_tokens[-1]
        if last_tok in EOS_IDS:
            break
            
        num_steps += 1
        # Busca candidatos via lookahead de n-grams
        candidates = find_candidate_tokens(all_tokens, ngram_size=2, max_candidates=K)
        if not candidates:
            candidates = find_candidate_tokens(all_tokens, ngram_size=1, max_candidates=K)
            
        if not candidates:
            # Passo padrão de 1 token
            cur_tensor = torch.tensor([[last_tok]], device="cpu")
            cur_emb = model.model.embed_tokens(cur_tensor).to(device=DEVICE, dtype=torch.bfloat16)
            with torch.no_grad():
                out = model.model(inputs_embeds=cur_emb, past_key_values=cache, use_cache=True)
                logits = F.linear(out.last_hidden_state[:, -1:, :], lm_head).squeeze(0).squeeze(0)
            next_t = logits.argmax().item()
            all_tokens.append(next_t)
            generated_tokens.append(next_t)
            print(tok.decode([next_t]), end="", flush=True)
            if next_t in EOS_IDS:
                break
            continue
            
        # Verificação paralela de K candidatos em 1 forward pass!
        cand_tensor = torch.tensor([[last_tok] + candidates], device="cpu")
        cand_emb = model.model.embed_tokens(cand_tensor).to(device=DEVICE, dtype=torch.bfloat16)
        
        # Salva ponto de restauração do cache caso candidatos falhem
        # DynamicCache permite truncar ou avaliar
        with torch.no_grad():
            out = model.model(inputs_embeds=cand_emb, past_key_values=cache, use_cache=True)
            step_logits = F.linear(out.last_hidden_state, lm_head)[0] # (K+1, vocab)
            
        # Verifica quais candidatos foram aceitos
        # step_logits[0] prevê o primeiro candidato
        accepted = []
        for idx, cand in enumerate(candidates):
            pred = step_logits[idx].argmax().item()
            if pred == cand:
                accepted.append(cand)
            else:
                # Primeiro mismatch: o modelo emitiu 'pred'
                accepted.append(pred)
                break
        else:
            # Todos os K candidatos foram aceitos! Emite também o bônus predito pelo último candidato
            bonus = step_logits[len(candidates)].argmax().item()
            accepted.append(bonus)
            
        # Imprime tokens aceitos
        for t in accepted:
            all_tokens.append(t)
            generated_tokens.append(t)
            print(tok.decode([t]), end="", flush=True)
            if t in EOS_IDS:
                break
        accepted_speculative += len(accepted)
        
    dt = time.perf_counter() - t_decode_start
    total_tokens = len(generated_tokens)
    speed = total_tokens / max(dt, 1e-4)
    print(f"\n\n[Relatório de Desempenho]:")
    print(f"Tokens Gerados: {total_tokens}")
    print(f"Tempo de Decodificação: {dt:.2f}s")
    print(f"Passos de Execução: {num_steps}")
    print(f"Tokens por Passo: {total_tokens / max(num_steps, 1):.2f}")
    print(f"VELOCIDADE: {speed:.1f} TOKENS/SEGUNDO!")

speculative_generate("Qual é a capital do Brasil e qual é a capital da França?")
