import sys, os, time
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scratch.test_cuda_graph_generator import LegionEngine, CUDAGraphDecoder

def find_nan():
    engine = LegionEngine()
    prompt = "A inteligência artificial avançada permite"
    prompt_ids = engine.tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
    with torch.no_grad():
        out = engine.model(input_ids=prompt_ids, use_cache=True)
        past_kv = out.past_key_values
        next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()

    decoder = CUDAGraphDecoder(engine)
    decoder.capture(past_kv, initial_pos=prompt_ids.shape[1])

    curr_in = torch.tensor([[next_tok]], device="cuda")
    h = decoder.model.model.embed_tokens(curr_in)
    print("Embed has NaN:", torch.isnan(h).any().item())

    for l_idx in range(decoder.num_layers):
        buf = decoder.buffers[l_idx]
        buf["x_in"].copy_(h)
        decoder.graphs_a[l_idx].replay()

        exp = decoder.model.model.layers[l_idx].mlp.experts
        topk_ids = buf["ri"].view(-1).tolist()
        for i, e in enumerate(topk_ids):
            decoder.pinned_slots[i] = exp._get_slot(e)
        buf["slots"].copy_(decoder.pinned_slots, non_blocking=True)
        buf["topk_weights"].copy_(buf["rw"].view(-1), non_blocking=True)

        decoder.graphs_b[l_idx].replay()
        h = buf["x_out"]

        if torch.isnan(h).any().item():
            print(f"Layer {l_idx} PRODUZIU NaN!")
            print(f"  res has NaN:       {torch.isnan(buf['res']).any().item()}")
            print(f"  post_norm has NaN: {torch.isnan(buf['post_norm']).any().item()}")
            print(f"  sh_final has NaN:  {torch.isnan(buf['sh_final']).any().item()}")
            print(f"  rw has NaN:        {torch.isnan(buf['rw']).any().item()}")
            print(f"  ri values:         {buf['ri'].tolist()}")
            break
        else:
            print(f"Layer {l_idx:02d} ({decoder.model.model.layers[l_idx].block_type[:4]}): OK (min={h.min().item():.3f}, max={h.max().item():.3f})")

if __name__ == "__main__":
    find_nan()
