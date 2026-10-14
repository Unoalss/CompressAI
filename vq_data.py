"""
vq_data.py - Wrapper de compatibilidade. Redireciona para tools.prepare_data.
"""
from tools.prepare_data import main, prepare_dataset, fetch_wiki, code_corpus, windows

if __name__ == "__main__":
    prepare_dataset()
