"""
tools/upload_to_hf.py - Script para upload dos modelos VQ2/VQ2M para o Hugging Face Hub.
"""
import os, sys, time, argparse
from pathlib import Path
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
REPO_ID = "ArthurLss/CompressAI"

def upload_models(token: str):
    api = HfApi(token=token)
    print("=" * 80)
    print(f"INICIANDO UPLOAD PARA O HUGGING FACE: https://huggingface.co/{REPO_ID}")
    print("=" * 80)
    
    files_to_upload = [
        ("VQ2_MARCO_FINAL/qwen3.5_2b_vq2_560mb.vq2r", "qwen3.5_2b_vq2_560mb.vq2r", "Modelo 2B RDO 560MB"),
        ("vq2_35b_out/qwen35b_mixed_b2.vq2m", "qwen35b_mixed_b2.vq2m", "Modelo MoE 35B Artefato B2 (10 GB)"),
    ]
    
    for local_rel, repo_path, label in files_to_upload:
        local_path = ROOT / local_rel
        if not local_path.exists():
            print(f"[PULANDO] Arquivo nao encontrado: {local_path}")
            continue
            
        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"\n--> Uploading {label} ({size_mb:.2f} MB)...")
        t0 = time.time()
        try:
            commit_info = api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=repo_path,
                repo_id=REPO_ID,
                repo_type="model"
            )
            dt = time.time() - t0
            print(f"[SUCESSO] {label} enviado em {dt:.1f}s ({size_mb / max(dt, 0.001):.2f} MB/s)!")
            print(f"URL: {commit_info}")
        except Exception as e:
            print(f"[ERRO no upload de {label}]: {e}")
            raise e

    print("\n" + "=" * 80)
    print("TODOS OS ARQUIVOS FORAM ENVIADOS COM SUCESSO AO HUGGING FACE!")
    print(f"Acesse: https://huggingface.co/{REPO_ID}")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN", ""))
    args = parser.parse_args()
    if not args.token:
        print("Erro: Forneca o token via --token ou variavel HF_TOKEN")
        sys.exit(1)
    upload_models(token=args.token)
