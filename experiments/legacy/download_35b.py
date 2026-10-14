"""
download_35b.py - Download resiliente e com progresso do modelo Qwen3.8-35B-A3B-Distill.
Salva diretamente em e:\\compressAI\\modelo_35b sem symlinks.
"""
import os, sys, time
from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ID = "empero-ai/Qwen3.8-35B-A3B-Distill"
TARGET_DIR = Path(r"e:\compressAI\modelo_35b")

def download_model():
    print("=" * 80)
    print(f"INICIANDO DOWNLOAD DO MODELO: {REPO_ID}")
    print(f"Destino no disco E: {TARGET_DIR}")
    print("=" * 80)
    
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    
    try:
        path = snapshot_download(
            repo_id=REPO_ID,
            local_dir=str(TARGET_DIR),
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
    download_model()
