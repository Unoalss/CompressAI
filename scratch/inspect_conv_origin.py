import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as qm
fn = qm.causal_conv1d_update
print("fn module:", fn.__module__)
print("fn name:", fn.__name__)
import inspect
print("fn file:", inspect.getsourcefile(fn))
