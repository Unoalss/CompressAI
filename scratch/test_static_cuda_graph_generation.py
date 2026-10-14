import sys, time
sys.path.insert(0, r"e:\compressAI")
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoConfig, StaticCache
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeTextConfig
)
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

# Teste com StaticCache para geração
prompt = "Olá! Quem foi Galileu Galilei?"
messages = [
    {"role": "system", "content": "Você é um assistente prestativo. Responda em português."},
    {"role": "user", "content": prompt}
]
formatted = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
input_ids = tok.encode(formatted, return_tensors="pt")
seq_len = input_ids.shape[1]

max_cache_len = seq_len + 64
print(f"Instanciando StaticCache (len={max_cache_len})...")
try:
    cache = StaticCache(config=text_cfg, max_batch_size=1, max_cache_len=max_cache_len, device=DEVICE, dtype=torch.bfloat16)
    
    # 1. Prefill
    prompt_emb = model.model.embed_tokens(input_ids).to(device=DEVICE, dtype=torch.bfloat16)
    with torch.no_grad():
        out = model.model(inputs_embeds=prompt_emb, past_key_values=cache, use_cache=True)
        logits = F.linear(out.last_hidden_state[:, -1:, :], model.lm_head.weight).squeeze(0).squeeze(0).float()
        
    next_tok = logits.argmax().item()
    print("Primeiro token:", tok.decode([next_tok]))
    
    # Decodificação de 15 tokens com StaticCache
    generated = [next_tok]
    t0 = time.perf_counter()
    for step in range(15):
        next_t = torch.tensor([[generated[-1]]], device="cpu")
        next_emb = model.model.embed_tokens(next_t).to(device=DEVICE, dtype=torch.bfloat16)
        with torch.no_grad():
            out = model.model(inputs_embeds=next_emb, past_key_values=cache, use_cache=True)
            logits = F.linear(out.last_hidden_state[:, -1:, :], model.lm_head.weight).squeeze(0).squeeze(0).float()
        next_tok = logits.argmax().item()
        generated.append(next_tok)
        print(tok.decode([next_tok]), end="", flush=True)
        if next_tok in EOS_IDS: break
    dt = time.perf_counter() - t0
    print(f"\nVelocidade StaticCache: {len(generated)} tokens em {dt:.3f}s -> {len(generated)/dt:.2f} tok/s")
except Exception as e:
    import traceback
    print("Erro com StaticCache:", e)
    traceback.print_exc()
