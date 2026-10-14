"""
run_refs.py - referencias e ablacao por componente (passo 0 do plano).

  q4km_full  : GGUF Q4_K_M inteiro (dequantizado) - o alvo a bater
  q4km_ffn   : somente as 72 matrizes FFN vindas do GGUF (backbone BF16 no resto)
  abl_*_rtn4 : RTN 4-bit g128 aplicado em UM componente por vez (FFN / atencao+DeltaNet / embeddings)
               -> mostra de onde vem a perda de PPL
"""
import sys, time, torch
import vq_lab as L
import run_vq_lab as R

log = R.log


def body_linears(model, skip_mlp=True):
    out = {}
    for i, layer in enumerate(model.model.layers):
        for n, m in layer.named_modules():
            if isinstance(m, torch.nn.Linear):
                if skip_mlp and n.startswith("mlp"):
                    continue
                out[f"l{i}.{n}"] = m
    return out


def rtn_inplace(mods):
    saved = {k: m.weight.data.clone() for k, m in mods.items()}
    for k, m in mods.items():
        What, _ = L.rtn_scalar(m.weight.data.float(), 4, 128)
        m.weight.data.copy_(What)
    return saved


def restore(mods, saved):
    for k, m in mods.items():
        m.weight.data.copy_(saved[k])


def main(which):
    tok = L.load_tokens()
    done = R.load_results()
    if "q4km_full" in which and "q4km_full" not in done:
        g = L.load_model(gguf=True)
        r = L.eval_ppl(g, tok)
        log("q4km_full", r)
        R.save_result("q4km_full", r)
        del g
    if "q4km_ffn" in which and "q4km_ffn" not in done:
        m = L.load_model()
        g = L.load_model(gguf=True)
        mm, gm = L.ffn_modules(m), L.ffn_modules(g)
        for k in mm:
            mm[k].weight.data.copy_(gm[k].weight.data)
        del g
        r = L.eval_ppl(m, tok)
        r["bpw"] = 4.84  # media real do GGUF nas FFN: gate/up Q4_K 4.5; down 50% Q4_K + 50% Q6_K
        log("q4km_ffn", r)
        R.save_result("q4km_ffn", r)
        del m
    if any(w.startswith("abl_") for w in which):
        m = L.load_model()
        groups = {
            "abl_ffn_rtn4": L.ffn_modules(m),
            "abl_attn_rtn4": body_linears(m),
        }
        for name, mods in groups.items():
            if name in which and name not in done:
                saved = rtn_inplace(mods)
                r = L.eval_ppl(m, tok)
                log(name, r)
                R.save_result(name, r)
                restore(mods, saved)
        if "abl_emb_rtn4" in which and "abl_emb_rtn4" not in done:
            emb = m.model.embed_tokens
            sv = emb.weight.data.clone()
            W, _ = L.rtn_scalar(sv.float(), 4, 128)
            emb.weight.data.copy_(W)
            r = L.eval_ppl(m, tok)
            log("abl_emb_rtn4", r)
            R.save_result("abl_emb_rtn4", r)
            emb.weight.data.copy_(sv)


if __name__ == "__main__":
    main(sys.argv[1:])
