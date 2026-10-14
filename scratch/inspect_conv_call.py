import inspect
import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as qm
print("qm.causal_conv1d_update:", qm.causal_conv1d_update)
for name, cls in inspect.getmembers(qm, inspect.isclass):
    if "LinearAttention" in name or "Delta" in name:
        print("Found class:", name)
        src = inspect.getsource(cls.forward)
        for line in src.split("\n"):
            if "conv" in line:
                print("  ", line)
