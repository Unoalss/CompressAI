import sys, time, psutil
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vq2.paged_runtime import PagedModelStore, PagedMoEExpert

def test():
    p = psutil.Process()
    print(f"RSS inicial: {p.memory_info().rss / 1024**2:.2f} MB")
    
    b2_path = ROOT / "vq2_35b_out" / "qwen35b_mixed_b2.vq2m"
    store = PagedModelStore(b2_path, use_mmap=False, device="cpu")
    cbs_data = torch.load(ROOT / "vq2_out" / "codebooks.pt", weights_only=False, map_location="cpu")
    cb16 = cbs_data["16"].bfloat16()
    
    print("Instanciando camada 0 em CPU...")
    t0 = time.perf_counter()
    layer = PagedMoEExpert(layer_idx=0, store=store, cb16=cb16, capacity=16, device="cpu")
    print(f"Camada 0 instanciada em {(time.perf_counter()-t0)*1000:.2f} ms")
    print(f"RSS apos instanciacao: {p.memory_info().rss / 1024**2:.2f} MB")
    
    # Test forward
    hidden_states = torch.randn(1, 2048, dtype=torch.bfloat16)
    topk_idx = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.int32)
    topk_weights = torch.ones(1, 8, dtype=torch.float32) / 8.0
    
    print("Executando forward de decode (T=1, 8 experts) em CPU...")
    t0 = time.perf_counter()
    out = layer(hidden_states, topk_idx, topk_weights)
    dt = (time.perf_counter() - t0) * 1000
    print(f"Forward concluiu com sucesso em {dt:.2f} ms!")
    print(f"Saida: shape={out.shape}, dtype={out.dtype}, any_nan={torch.isnan(out).any().item()}")
    print(f"RSS apos forward: {p.memory_info().rss / 1024**2:.2f} MB")

if __name__ == "__main__":
    test()
