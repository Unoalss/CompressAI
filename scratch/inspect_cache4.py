import inspect
import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as qm
source = inspect.getsource(qm.Qwen3_5MoeTextModel.forward)
for i, line in enumerate(source.split("\n")[:70]):
    print(f"{i}: {line}")
