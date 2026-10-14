import os
import time
import torch
from safetensors import safe_open

MODEL_DIR = "E:/agente/modelo"
SAFETENSORS_PATH = os.path.join(MODEL_DIR, "model.safetensors-00001-of-00001.safetensors")

print("=" * 90)
print("FASE 3: ABLAÇÃO ESTRUTURAL 2+4 vs 3+3 NAS MATRIZES FFN DO QWEN3.5-2B")
print("=" * 90)

# Importar o quantizador testado
from test_qwen_ffn_single_layer import quantize_matrix_h5a_batched

# Carregar camadas representativas: 0 (inicial), 6, 12 (intermediária), 18, 23 (final)
layers_to_test = [0, 6, 12, 18, 23]
results = {}

with safe_open(SAFETENSORS_PATH, framework="pt", device="cpu") as f:
    for lyr in layers_to_test:
        w_gate = f.get_tensor(f"model.language_model.layers.{lyr}.mlp.gate_proj.weight")
        w_up = f.get_tensor(f"model.language_model.layers.{lyr}.mlp.up_proj.weight")
        w_down = f.get_tensor(f"model.language_model.layers.{lyr}.mlp.down_proj.weight")
        
        print(f"\n--- AVALIANDO CAMADA {lyr} ---")
        
        # 1. Teste no gate_proj (Expansão): 2+4-D vs 3+3
        _, _, mse_g_24, cos_g_24, _ = quantize_matrix_h5a_batched(w_gate, num_ctrl=2, num_spec=4, spec_mode='sensitivity', p_crit=0.05, seed=42+lyr)
        _, _, mse_g_33, cos_g_33, _ = quantize_matrix_h5a_batched(w_gate, num_ctrl=3, num_spec=3, spec_mode='sensitivity', p_crit=0.05, seed=42+lyr)
        
        # 2. Teste no up_proj (Expansão): 2+4-D vs 3+3
        _, _, mse_u_24, cos_u_24, _ = quantize_matrix_h5a_batched(w_up, num_ctrl=2, num_spec=4, spec_mode='sensitivity', p_crit=0.05, seed=42+lyr)
        _, _, mse_u_33, cos_u_33, _ = quantize_matrix_h5a_batched(w_up, num_ctrl=3, num_spec=3, spec_mode='sensitivity', p_crit=0.05, seed=42+lyr)
        
        # 3. Teste no down_proj (Redução): 3+3 vs 2+4-D
        _, _, mse_d_33, cos_d_33, _ = quantize_matrix_h5a_batched(w_down, num_ctrl=3, num_spec=3, spec_mode='sensitivity', p_crit=0.05, seed=42+lyr)
        _, _, mse_d_24, cos_d_24, _ = quantize_matrix_h5a_batched(w_down, num_ctrl=2, num_spec=4, spec_mode='sensitivity', p_crit=0.05, seed=42+lyr)
        
        # 4. Teste de ativação SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
        # Gerar uma entrada sintética x com a norma típica do hidden_size 2048
        torch.manual_seed(42)
        x = torch.randn(32, 2048, dtype=torch.bfloat16) * 0.05
        
        with torch.no_grad():
            orig_act = torch.nn.functional.silu(torch.matmul(x, w_gate.t())) * torch.matmul(x, w_up.t())
            orig_out = torch.matmul(orig_act, w_down.t())
            
            # Recriar matrizes H5-A Asimétrica (2+4 / 2+4 / 3+3)
            rec_g_asim, _, _, _, _ = quantize_matrix_h5a_batched(w_gate, num_ctrl=2, num_spec=4, spec_mode='sensitivity', p_crit=0.05, seed=42+lyr)
            rec_u_asim, _, _, _, _ = quantize_matrix_h5a_batched(w_up, num_ctrl=2, num_spec=4, spec_mode='sensitivity', p_crit=0.05, seed=42+lyr)
            rec_d_asim, _, _, _, _ = quantize_matrix_h5a_batched(w_down, num_ctrl=3, num_spec=3, spec_mode='sensitivity', p_crit=0.05, seed=42+lyr)
            
            act_asim = torch.nn.functional.silu(torch.matmul(x, rec_g_asim.t())) * torch.matmul(x, rec_u_asim.t())
            out_asim = torch.matmul(act_asim, rec_d_asim.t())
            
            # Recriar matrizes Simétricas 3+3 (3+3 / 3+3 / 3+3)
            rec_g_33, _, _, _, _ = quantize_matrix_h5a_batched(w_gate, num_ctrl=3, num_spec=3, spec_mode='sensitivity', p_crit=0.05, seed=42+lyr)
            rec_u_33, _, _, _, _ = quantize_matrix_h5a_batched(w_up, num_ctrl=3, num_spec=3, spec_mode='sensitivity', p_crit=0.05, seed=42+lyr)
            rec_d_33, _, _, _, _ = quantize_matrix_h5a_batched(w_down, num_ctrl=3, num_spec=3, spec_mode='sensitivity', p_crit=0.05, seed=42+lyr)
            
            act_33 = torch.nn.functional.silu(torch.matmul(x, rec_g_33.t())) * torch.matmul(x, rec_u_33.t())
            out_33 = torch.matmul(act_33, rec_d_33.t())
            
            cos_act_asim = torch.nn.functional.cosine_similarity(orig_act.view(1, -1), act_asim.view(1, -1)).item()
            cos_out_asim = torch.nn.functional.cosine_similarity(orig_out.view(1, -1), out_asim.view(1, -1)).item()
            
            cos_act_33 = torch.nn.functional.cosine_similarity(orig_act.view(1, -1), act_33.view(1, -1)).item()
            cos_out_33 = torch.nn.functional.cosine_similarity(orig_out.view(1, -1), out_33.view(1, -1)).item()
            
        print(f"gate_proj:  2+4-D Cos={cos_g_24:.6f} (MSE={mse_g_24:.8f}) | 3+3 Cos={cos_g_33:.6f} (MSE={mse_g_33:.8f}) -> Vencedor: {'2+4-D' if cos_g_24 >= cos_g_33 else '3+3'}")
        print(f"up_proj:    2+4-D Cos={cos_u_24:.6f} (MSE={mse_u_24:.8f}) | 3+3 Cos={cos_u_33:.6f} (MSE={mse_u_33:.8f}) -> Vencedor: {'2+4-D' if cos_u_24 >= cos_u_33 else '3+3'}")
        print(f"down_proj:  3+3   Cos={cos_d_33:.6f} (MSE={mse_d_33:.8f}) | 2+4-D Cos={cos_d_24:.6f} (MSE={mse_d_24:.8f}) -> Vencedor: {'3+3' if cos_d_33 >= cos_d_24 else '2+4-D'}")
        print(f"SwiGLU Out: Asimétrico (2+4/3+3) Cos={cos_out_asim:.6f} | Simétrico (3+3/3+3) Cos={cos_out_33:.6f} -> Ganho Asimétrico: {(cos_out_asim - cos_out_33)*1e4:+.2f} bps")
        
        results[lyr] = {
            "cos_g_24": cos_g_24, "cos_g_33": cos_g_33,
            "cos_u_24": cos_u_24, "cos_u_33": cos_u_33,
            "cos_d_33": cos_d_33, "cos_d_24": cos_d_24,
            "cos_out_asim": cos_out_asim, "cos_out_33": cos_out_33
        }

print("\n" + "=" * 90)
print("SÍNTESE DA ABLAÇÃO FASE 3:")
print("=" * 90)
mean_gain = sum((r["cos_out_asim"] - r["cos_out_33"]) * 1e4 for r in results.values()) / len(results)
print(f"Ganho Médio da Alocação Asimétrica na saída do Bloco SwiGLU: {mean_gain:+.2f} bps de similaridade cosseno!")
