import sys, time, psutil
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def probe():
    p = psutil.Process()
    print(f"[0] RSS Inicial: {p.memory_info().rss / 1024**2:.1f} MB")
    
    t0 = time.perf_counter()
    from transformers import AutoTokenizer, AutoConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig
    from accelerate import init_empty_weights
    from vq2.paged_runtime import PagedModelStore, PagedMoEExpert
    from vq2.kernels import patch_transformers_kernels, optimize_deltanet_layers
    patch_transformers_kernels()
    print(f"[1] Imports e patches: {(time.perf_counter()-t0)*1000:.1f} ms | RSS: {p.memory_info().rss / 1024**2:.1f} MB")
    
    t0 = time.perf_counter()
    MODEL_DIR = ROOT / "modelo_35b"
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    cbs_data = torch.load(ROOT / "vq2_out" / "codebooks.pt", weights_only=False, map_location="cpu")
    cb16 = cbs_data["16"].to(dtype=torch.bfloat16)
    print(f"[2] Tokenizer e Codebook: {(time.perf_counter()-t0)*1000:.1f} ms | RSS: {p.memory_info().rss / 1024**2:.1f} MB")

    t0 = time.perf_counter()
    model_path = ROOT / "vq2_35b_out" / "qwen35b_mixed_b2.vq2m"
    store = PagedModelStore(model_path, use_mmap=False, device="cpu")
    print(f"[3] Container Store: {(time.perf_counter()-t0)*1000:.1f} ms | RSS: {p.memory_info().rss / 1024**2:.1f} MB")

    t0 = time.perf_counter()
    cfg = AutoConfig.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg.text_config
    if isinstance(text_cfg, dict):
        text_cfg = Qwen3_5MoeTextConfig(**text_cfg)
    text_cfg.num_experts_per_tok = 8

    with init_empty_weights():
        model = Qwen3_5MoeForCausalLM(text_cfg)
    print(f"[4] Esqueleto vazio: {(time.perf_counter()-t0)*1000:.1f} ms | RSS: {p.memory_info().rss / 1024**2:.1f} MB")

    t0 = time.perf_counter()
    layers = []
    for l_idx in range(40):
        layer = PagedMoEExpert(layer_idx=l_idx, store=store, cb16=cb16, capacity=16, device="cpu")
        model.model.layers[l_idx].mlp.experts = layer
        if hasattr(model.model.layers[l_idx].mlp, "gate") and hasattr(model.model.layers[l_idx].mlp.gate, "top_k"):
            model.model.layers[l_idx].mlp.gate.top_k = 8
        layers.append(layer)
    print(f"[5] 40 camadas MoE instanciadas: {(time.perf_counter()-t0)*1000:.1f} ms | RSS: {p.memory_info().rss / 1024**2:.1f} MB")

    t0 = time.perf_counter()
    num_other = len(store.other_tensors)
    print(f"      Injetando {num_other} outros tensores (densos e quant_linear)...")
    count = 0
    t_start_inject = time.perf_counter()
    for name, info in store.other_tensors.items():
        if info.get("type") == "dense":
            t = store.get_dense_tensor(name, device="cpu")
            # set param
            parts = name.split(".")
            obj = model
            ok = True
            for p_part in parts[:-1]:
                if not hasattr(obj, p_part):
                    ok = False; break
                obj = getattr(obj, p_part)
            if ok:
                setattr(obj, parts[-1], torch.nn.Parameter(t, requires_grad=False))
        elif info.get("type") == "quant_linear":
            W = store.get_quant_linear(name, cbs_data=cbs_data, device="cpu")
            if W is not None:
                parts = name.split(".")
                obj = model
                ok = True
                for p_part in parts[:-1]:
                    if not hasattr(obj, p_part):
                        ok = False; break
                    obj = getattr(obj, p_part)
                if ok:
                    setattr(obj, parts[-1], torch.nn.Parameter(W, requires_grad=False))
                del W
        count += 1
        if count % 100 == 0 or count == num_other:
            print(f"        Progresso tensores: {count}/{num_other} ({(time.perf_counter()-t_start_inject):.1f}s) | RSS: {p.memory_info().rss / 1024**2:.1f} MB")
    
    print(f"[6] Tensores densos/quant_linear injetados: {(time.perf_counter()-t0):.2f} s | RSS: {p.memory_info().rss / 1024**2:.1f} MB")

if __name__ == "__main__":
    probe()
