"""
tools/download_35b.py - Download resiliente e com progresso do modelo Qwen3.8-35B-A3B-Distill.
Salva diretamente na pasta modelo_35b sem symlinks.
"""
import os, sys, time, argparse
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGET_DIR = ROOT / "modelo_35b"
DEFAULT_REPO_ID = "empero-ai/Qwen3.8-35B-A3B-Distill"

def download_model(repo_id: str = DEFAULT_REPO_ID, target_dir: Path = DEFAULT_TARGET_DIR):
    print("=" * 80)
    print(f"INICIANDO DOWNLOAD DO MODELO: {repo_id}")
    print(f"Destino no disco: {target_dir}")
    print("=" * 80)
    
    target_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    
    try:
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=4
        )
        dt = time.time() - t0
        print("\n" + "=" * 80)
        print(f"DOWNLOAD CONCLUIDO COM SUCESSO EM {dt/60:.1f} MINUTOS!")
        print(f"Arquivos salvos em: {path}")
        print("=" * 80)
    except Exception as e:
        print(f"\n[ERRO NO DOWNLOAD]: {e}")
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Qwen 35B MoE checkpoint")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face repo ID")
    parser.add_argument("--target-dir", default=str(DEFAULT_TARGET_DIR), help="Destination folder")
    args = parser.parse_args()
    download_model(repo_id=args.repo_id, target_dir=Path(args.target_dir))
