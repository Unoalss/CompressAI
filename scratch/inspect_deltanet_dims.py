import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legion.engine import LegionEngine

def inspect_deltanet_dims():
    engine = LegionEngine()
    model = engine.model.model
    
    linear_layers = [l for l in model.layers if hasattr(l, "linear_attn")]
    full_layers = [l for l in model.layers if hasattr(l, "self_attn")]
    
    print("=" * 80)
    print("ARQUITETURA DE ATENÇÃO DO QWEN 3.5 MoE:")
    print("=" * 80)
    print(f"Total de camadas: {len(model.layers)}")
    print(f"Camadas Linear Attention (DeltaNet): {len(linear_layers)} (Ex: Camada 0)")
    print(f"Camadas Full Attention (Self-Attn):  {len(full_layers)}")
    
    dn0 = linear_layers[0].linear_attn
    print("\nDimensões da DeltaNet (linear_attn):")
    print(f"  hidden_size:          {dn0.hidden_size}")
    print(f"  num_v_heads:          {dn0.num_v_heads}")
    print(f"  num_k_heads:          {dn0.num_k_heads}")
    print(f"  head_k_dim:           {dn0.head_k_dim}")
    print(f"  head_v_dim:           {dn0.head_v_dim}")
    print(f"  key_dim:              {dn0.key_dim}")
    print(f"  value_dim:            {dn0.value_dim}")
    print(f"  conv_dim:             {dn0.conv_dim}")
    print(f"  conv_kernel_size:     {dn0.conv_kernel_size}")
    
    print("\nPesos e Projeções da DeltaNet (linear_attn):")
    print(f"  in_proj_qkv: {dn0.in_proj_qkv.weight.shape} ({dn0.in_proj_qkv.weight.element_size() * dn0.in_proj_qkv.weight.nelement() / (1024*1024):.2f} MB)")
    print(f"  in_proj_z:   {dn0.in_proj_z.weight.shape}   ({dn0.in_proj_z.weight.element_size() * dn0.in_proj_z.weight.nelement() / (1024*1024):.2f} MB)")
    print(f"  in_proj_b:   {dn0.in_proj_b.weight.shape}   ({dn0.in_proj_b.weight.element_size() * dn0.in_proj_b.weight.nelement() / (1024*1024):.2f} MB)")
    print(f"  in_proj_a:   {dn0.in_proj_a.weight.shape}   ({dn0.in_proj_a.weight.element_size() * dn0.in_proj_a.weight.nelement() / (1024*1024):.2f} MB)")
    print(f"  out_proj:    {dn0.out_proj.weight.shape}    ({dn0.out_proj.weight.element_size() * dn0.out_proj.weight.nelement() / (1024*1024):.2f} MB)")
    print(f"  conv1d:      {dn0.conv1d.weight.shape}")
    print(f"  norm:        Qwen3_5MoeRMSNormGated(dim={dn0.norm.weight.shape})")

    if full_layers:
        fa0 = full_layers[0].self_attn
        print("\nDimensões da Full Attention (self_attn):")
        print(f"  hidden_size:   {fa0.hidden_size}")
        print(f"  num_heads:     {fa0.num_heads}")
        print(f"  num_kv_heads:  {fa0.num_key_value_heads}")
        print(f"  head_dim:      {fa0.head_dim}")
        print(f"  q_proj:        {fa0.q_proj.weight.shape}")
        print(f"  k_proj:        {fa0.k_proj.weight.shape}")
        print(f"  v_proj:        {fa0.v_proj.weight.shape}")
        print(f"  o_proj:        {fa0.o_proj.weight.shape}")
    print("=" * 80)

if __name__ == "__main__":
    inspect_deltanet_dims()
