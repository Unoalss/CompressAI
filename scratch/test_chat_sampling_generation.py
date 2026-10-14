import os, sys, time, json, gc
from pathlib import Path
sys.path.insert(0, r"e:\compressAI")
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeTextConfig
)
from accelerate import init_empty_weights

from vq2.load_and_test_full_35b import (
    PackedMoEExpertUniversal,
    dequant_matrix,
    set_nested_param
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_35B = Path(r"e:\compressAI\vq2_35b_out")
MODEL_DIR = Path(r"e:\compressAI\modelo_35b")

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

# Teste com ChatML Template
user_prompt = "Olá, quem é você?"
messages = [
    {"role": "system", "content": "Você é o assistente inteligente Qwen3.8-35B."},
    {"role": "user", "content": user_prompt}
]
try:
    formatted_prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
except Exception:
    formatted_prompt = f"<|im_start|>system\nVocê é o assistente inteligente Qwen3.8-35B.<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

print(f"\nPrompt formatado ChatML:\n{formatted_prompt}")

input_ids = tok.encode(formatted_prompt, return_tensors="pt")
generated_ids = list(input_ids[0].tolist())
lm_head_weight = model.lm_head.weight.float()

max_new_tokens = 30
temperature = 0.7
top_p = 0.9
top_k = 40
repetition_penalty = 1.2

print("\n--- INÍCIO DA GERAÇÃO REAL COM AMOSTRAGEM INTELIGENTE ---")
response_tokens = []

for step in range(max_new_tokens):
    curr_input_ids = torch.tensor([generated_ids], dtype=torch.long)
    inputs_embeds = model.model.embed_tokens(curr_input_ids).to(device=DEVICE, dtype=torch.bfloat16)
    
    with torch.no_grad():
        out = model.model(inputs_embeds=inputs_embeds)
        last_hidden = out.last_hidden_state[:, -1:, :].cpu().float()
        logits = torch.matmul(last_hidden, lm_head_weight.T).squeeze(0).squeeze(0)
        
    # Repetition Penalty
    for token_id in set(generated_ids):
        if logits[token_id] < 0:
            logits[token_id] *= repetition_penalty
        else:
            logits[token_id] /= repetition_penalty
            
    # Temperature & Top-K / Top-P
    logits = logits / temperature
    topk_vals, topk_indices = torch.topk(logits, top_k)
    probs = F.softmax(topk_vals, dim=-1)
    
    # Amostra próximo token
    next_idx = torch.multinomial(probs, num_samples=1).item()
    next_tok = topk_indices[next_idx].item()
    
    if next_tok in EOS_IDS:
        print(f"\n[Fim de geração pelo token EOS {next_tok}]")
        break
        
    generated_ids.append(next_tok)
    response_tokens.append(next_tok)
    tok_str = tok.decode([next_tok])
    print(tok_str, end="", flush=True)

print("\n--- FIM DA GERAÇÃO ---")
final_text = tok.decode(response_tokens)
print("Texto puro da resposta:", repr(final_text))
