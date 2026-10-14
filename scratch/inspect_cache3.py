import inspect
import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as qm
source = inspect.getsource(qm.Qwen3_5MoeModel.forward)
for i, line in enumerate(source.split("\n")[:80]):
    print(f"{i}: {line}")
