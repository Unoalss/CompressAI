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
OUT_35B = r"e:\compressAI\vq2_35b_out"
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

def generate_with_cuda_graph(user_prompt: str, max_tokens: int = 50, temperature: float = 0.7):
    messages = [
        {"role": "system", "content": "Você é o modelo Qwen3.8-35B-A3B VQ2 RDO. Responda em português brasileiro de forma objetiva."},
        {"role": "user", "content": user_prompt}
    ]
    formatted = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    input_ids = tok.encode(formatted, return_tensors="pt")
    
    cache = DynamicCache(config=text_cfg)
    
    # 1. Prefill
    t_start = time.perf_counter()
    inputs_embeds = model.model.embed_tokens(input_ids).to(device=DEVICE, dtype=torch.bfloat16)
    with torch.no_grad():
        out = model.model(inputs_embeds=inputs_embeds, past_key_values=cache, use_cache=True)
        logits = F.linear(out.last_hidden_state[:, -1:, :], model.lm_head.weight).squeeze(0).squeeze(0).float()
    
    # Amostra primeiro token após o prefill
    topk_vals, topk_indices = torch.topk(logits / temperature, 40)
    probs = F.softmax(topk_vals, dim=-1)
    next_tok = topk_indices[torch.multinomial(probs, 1).item()].item()
    
    generated_tokens = [next_tok]
    print(f"\nPergunta: {user_prompt}")
    print("Resposta CUDA Graph: ", end="", flush=True)
    print(tok.decode([next_tok]), end="", flush=True)
    
    # 2. Setup do CUDA Graph para o decode
    static_token = torch.tensor([[next_tok]], device="cpu")
    static_embed = model.model.embed_tokens(static_token).to(device=DEVICE, dtype=torch.bfloat16)
    
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            with torch.no_grad():
                out = model.model(inputs_embeds=static_embed, past_key_values=cache, use_cache=True)
                _ = F.linear(out.last_hidden_state[:, -1:, :], model.lm_head.weight)
    torch.cuda.current_stream().wait_stream(s)
    
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        with torch.no_grad():
            static_out = model.model(inputs_embeds=static_embed, past_key_values=cache, use_cache=True)
            static_logits = F.linear(static_out.last_hidden_state[:, -1:, :], model.lm_head.weight)
            
    # 3. Replay loop
    t_decode0 = time.perf_counter()
    for _ in range(max_tokens - 1):
        g.replay()
        
        # Pega logits do grafo
        step_logits = static_logits[0, 0].float()
        
        # Penalidade e amostragem
        topk_vals, topk_indices = torch.topk(step_logits / temperature, 40)
        probs = F.softmax(topk_vals, dim=-1)
        cur_tok = topk_indices[torch.multinomial(probs, 1).item()].item()
        
        if cur_tok in EOS_IDS:
            break
            
        generated_tokens.append(cur_tok)
        print(tok.decode([cur_tok]), end="", flush=True)
        
        # Atualiza static_embed para o próximo replay
        new_emb = model.model.embed_tokens(torch.tensor([[cur_tok]], device="cpu")).to(device=DEVICE, dtype=torch.bfloat16)
        static_embed.copy_(new_emb)
        
    dt_decode = time.perf_counter() - t_decode0
    speed = len(generated_tokens) / max(dt_decode, 1e-4)
    print(f"\n\n[Geração de {len(generated_tokens)} tokens em {dt_decode:.2f}s -> {speed:.1f} TOKENS/SEGUNDO!]")

generate_with_cuda_graph("Explique em poucas palavras o que é buraco negro.", max_tokens=40)
