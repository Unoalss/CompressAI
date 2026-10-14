import inspect
import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as qm
source = inspect.getsource(qm)
for i, line in enumerate(source.split("\n")[:60]):
    if "conv1d" in line.lower():
        print(f"{i}: {line}")
