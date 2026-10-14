import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as qm
print("Has causal_conv1d_update in qwen3_5_moe:", hasattr(qm, "causal_conv1d_update"))
print("Has causal_conv1d_fn in qwen3_5_moe:", hasattr(qm, "causal_conv1d_fn"))
