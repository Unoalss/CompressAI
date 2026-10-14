import sys, os
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legion.engine import LegionEngine
import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as qm
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from scratch.test_fused_qwen35_rope import triton_fused_rmsnorm_rope_qwen

engine = LegionEngine()
full_layers = [l.self_attn for l in engine.model.model.layers if hasattr(l, "self_attn")]

def make_fa_forward(use_fused: bool):
    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        seq_len = hidden_states.shape[1]

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)
        cos, sin = position_embeddings

        if use_fused and seq_len == 1 and hidden_states.is_cuda:
            # Fused RMSNormZeroCentered + RoPE Triton
            query_states = triton_fused_rmsnorm_rope_qwen(
                query_states.view(hidden_shape), self.q_norm.weight, cos, sin, eps=self.q_norm.eps
            )
            key_raw = self.k_proj(hidden_states).view(hidden_shape)
            key_states = triton_fused_rmsnorm_rope_qwen(
                key_raw, self.k_norm.weight, cos, sin, eps=self.k_norm.eps
            )
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        else:
            # Original PyTorch
            query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            query_states, key_states = qm.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, qm.eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0, scaling=self.scaling, **kwargs
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
    return forward

prompt = "A inteligência artificial avançada permite arquiteturas neurais altamente eficientes com"
prompt_ids = engine.tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()

# Sequência fixa de teste (20 tokens) gerada pelo PyTorch
full_layers[0].__class__.forward = make_fa_forward(use_fused=False)
with torch.no_grad():
    out = engine.model(input_ids=prompt_ids, use_cache=True)
    past_kv = out.past_key_values
    next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()

test_tokens = [next_tok]
curr = torch.tensor([[next_tok]], device="cuda")
for _ in range(15):
    with torch.no_grad():
        out = engine.model(input_ids=curr, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()
        test_tokens.append(next_tok)
        curr = torch.tensor([[next_tok]], device="cuda")

print(f"Sequência de Referência Fixa ({len(test_tokens)} tokens):")
print("Texto:", engine.tokenizer.decode(test_tokens))

# Agora executamos Teacher-Forcing em ambos os modos na mesma sequência fixa:
def run_teacher_forcing(use_fused: bool):
    full_layers[0].__class__.forward = make_fa_forward(use_fused)
    torch.cuda.synchronize()
    
    with torch.no_grad():
        out = engine.model(input_ids=prompt_ids, use_cache=True)
        past_kv = out.past_key_values
        
    step_logits = []
    for tok in test_tokens:
        curr = torch.tensor([[tok]], device="cuda")
        with torch.no_grad():
            out = engine.model(input_ids=curr, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            step_logits.append(out.logits[:, -1, :].clone())
    return step_logits

logits_pt = run_teacher_forcing(use_fused=False)
logits_tri = run_teacher_forcing(use_fused=True)

print("\n--- COMPARAÇÃO COM PREFIXO IDÊNTICO (TEACHER FORCING) ---")
cosines = []
kls = []
top1_matches = 0

for i in range(len(test_tokens)):
    l_pt = logits_pt[i].float()
    l_tri = logits_tri[i].float()
    cos = F.cosine_similarity(l_pt, l_tri, dim=-1).item()
    p_fus = F.log_softmax(l_tri, dim=-1)
    p_pt = F.softmax(l_pt, dim=-1)
    kl = F.kl_div(p_fus, p_pt, reduction="batchmean").item()
    
    top1_pt = torch.argmax(l_pt, dim=-1).item()
    top1_tri = torch.argmax(l_tri, dim=-1).item()
    match = (top1_pt == top1_tri)
    if match:
        top1_matches += 1
    cosines.append(cos)
    kls.append(kl)
    print(f"Token {i:2d} | Top-1 Match: {str(match):5s} | Cosine: {cos*100:.5f}% | KL: {kl:.6f} | PT Top: {top1_pt} vs TRI Top: {top1_tri}")

print("\n" + "=" * 80)
print(f"Média Cosseno Logits: {sum(cosines)/len(cosines)*100:.6f}%")
print(f"Média Divergência KL: {sum(kls)/len(kls):.8f}")
print(f"Top-1 Concordância:  {top1_matches}/{len(test_tokens)} ({top1_matches/len(test_tokens)*100:.1f}%)")
print("=" * 80)
