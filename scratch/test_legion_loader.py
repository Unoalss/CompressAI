"""
scratch/test_legion_loader.py - Teste de montagem do motor LEGION diretamente do Pendrive D:\.
"""
import sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vq2.paged_runtime import PagedModelStore

PENDRIVE_MODEL = Path("D:/qwen35b_full.vq2m")
LOCAL_MODEL = Path("E:/compressAI/vq2_35b_out/qwen35b_full.vq2m")
MODEL_PATH = PENDRIVE_MODEL if PENDRIVE_MODEL.exists() else LOCAL_MODEL

print(f"Testando carregamento a partir de: {MODEL_PATH}")
t0 = time.perf_counter()
store = PagedModelStore(MODEL_PATH)
dt = (time.perf_counter() - t0) * 1000
print(f"Store montado com sucesso em {dt:.2f} ms!")
print(f"Camadas MoE: {store.num_layers} | Experts: {store.num_experts}")
print(f"Total de outros tensores: {len(store.other_tensors)}")

# Testa leitura de 1 tensor denso
t_dense = store.get_dense_tensor("model.layers.0.input_layernorm.weight", device="cuda" if torch.cuda.is_available() else "cpu")
print(f"RMSNorm da Camada 0: shape={t_dense.shape}, dtype={t_dense.dtype}")

# Carrega codebooks
cbs_data = torch.load(ROOT / "vq2_out" / "codebooks.pt", weights_only=False)

# Testa desquantização de uma atenção direta do pendrive
t_q = store.get_quant_linear("model.layers.0.self_attn.q_proj.weight", cbs_data=cbs_data, device="cuda" if torch.cuda.is_available() else "cpu")
print(f"q_proj da Camada 0 desquantizado do pendrive: shape={t_q.shape}, dtype={t_q.dtype}")

store.close()
print("Teste do loader LEGION concluído com 100% de sucesso!")
