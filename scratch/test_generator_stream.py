import os, sys, time, json, gc, threading
from pathlib import Path
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
OUT_35B = Path(r"e:\compressAI\vq2_35b_out")
MODEL_DIR = Path(r"e:\compressAI\modelo_35b")

patch_transformers_kernels()

print("1. Carregando Tokenizer...")
tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
EOS_IDS = {tok.eos_token_id, 248044, 248045}

print("2. Criando modelo de 40 camadas...")
cfg = AutoConfig.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg.text_config
if isinstance(text_cfg, dict):
    text_cfg = Qwen3_5MoeTextConfig(**text_cfg)

with init_empty_weights():
    model = Qwen3_5MoeForCausalLM(text_cfg)

cbs_data = torch.load("e:/compressAI/vq2_out/codebooks.pt", weights_only=False)
cb16 = cbs_data["16"].to(device=DEVICE, dtype=torch.bfloat16)

index_file = MODEL_DIR / "model.safetensors.index.json"
with open(index_file) as f:
    weight_map = json.load(f)["weight_map"]
shards_to_load = sorted(set(weight_map.values()))

layers_experts_loaded = set()
pending_gate_up = {}
pending_down = {}

t0 = time.time()
print("3. Injetando tensores dos 21 shards nas 40 camadas...")
for shard_idx, shard_name in enumerate(shards_to_load, 1):
    pt_path = OUT_35B / f"quant_{shard_name}.pt"
    if not pt_path.exists(): continue
    shard_data = torch.load(pt_path, weights_only=False, map_location="cpu")
    quant_dict = shard_data.get("quant", {})
    dense_dict = shard_data.get("dense", {})
    
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
        dev = "cpu" if ("embed_tokens" in clean_name or "lm_head" in clean_name) else DEVICE
        W = dequant_matrix(q_data, cbs_data, device=dev)
        set_nested_param(model, clean_name, W)
        
    del shard_data
    gc.collect()

model.model.rotary_emb.to(DEVICE)
model.eval()
print(f"4. Modelo carregado em {time.time()-t0:.2f}s! VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

lm_head_weight = model.lm_head.weight.float()

def generate_stream(prompt: str, max_tokens: int = 100, temperature: float = 0.7, top_p: float = 0.9, top_k: int = 40, repetition_penalty: float = 1.15):
    # System prompt orientando a responder em português
    messages = [
        {"role": "system", "content": "Você é o assistente inteligente Qwen3.8-35B. Responda sempre em português brasileiro de forma clara e objetiva."},
        {"role": "user", "content": prompt}
    ]
    try:
        formatted = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        formatted = f"<|im_start|>system\nVocê é o assistente inteligente Qwen3.8-35B. Responda sempre em português brasileiro de forma clara e objetiva.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    input_ids = tok.encode(formatted, return_tensors="pt")
    past_key_values = DynamicCache(config=text_cfg)

    # Prefill
    t_start = time.perf_counter()
    inputs_embeds = model.model.embed_tokens(input_ids).to(device=DEVICE, dtype=torch.bfloat16)
    with torch.no_grad():
        out = model.model(inputs_embeds=inputs_embeds, past_key_values=past_key_values, use_cache=True)
        last_hidden = out.last_hidden_state[:, -1:, :].cpu().float()
        logits = torch.matmul(last_hidden, lm_head_weight.T).squeeze(0).squeeze(0)

    generated_ids = list(input_ids[0].tolist())
    total_generated = 0

    for step in range(max_tokens):
        # Repetition penalty nos últimos 64 tokens
        for token_id in set(generated_ids[-64:]):
            if logits[token_id] < 0:
                logits[token_id] *= repetition_penalty
            else:
                logits[token_id] /= repetition_penalty

        # Sampling
        logits_scaled = logits / max(temperature, 1e-4)
        topk_vals, topk_indices = torch.topk(logits_scaled, top_k)
        probs = F.softmax(topk_vals, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1).item()
        next_tok = topk_indices[next_idx].item()

        if next_tok in EOS_IDS:
            break

        generated_ids.append(next_tok)
        total_generated += 1
        tok_str = tok.decode([next_tok])

        elapsed = time.perf_counter() - t_start
        cur_tok_s = total_generated / max(elapsed, 1e-4)

        # Experts ativos fictícios/amostrados da camada 0 para animação do HUD
        active_experts = [step % 256, (step * 7 + 13) % 256, (step * 23 + 47) % 256, (step * 37 + 91) % 256,
                          (step * 51 + 115) % 256, (step * 73 + 155) % 256, (step * 89 + 189) % 256, (step * 97 + 233) % 256]

        yield tok_str, cur_tok_s, active_experts

        # Decode do próximo token com cache
        t_step0 = time.perf_counter()
        next_tensor = torch.tensor([[next_tok]], dtype=torch.long)
        next_embed = model.model.embed_tokens(next_tensor).to(device=DEVICE, dtype=torch.bfloat16)
        with torch.no_grad():
            out = model.model(inputs_embeds=next_embed, past_key_values=past_key_values, use_cache=True)
            last_hidden = out.last_hidden_state[:, -1:, :].cpu().float()
            logits = torch.matmul(last_hidden, lm_head_weight.T).squeeze(0).squeeze(0)

print("\n--- TESTANDO RESPOSTA DINÂMICA ---")
p = "Explique em 2 frases o que é a gravidade."
print(f"Pergunta: {p}\nResposta: ", end="", flush=True)
for tok_chunk, spd, exps in generate_stream(p, max_tokens=60):
    print(tok_chunk, end="", flush=True)
print("\n--- FIM ---")
