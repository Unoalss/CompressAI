import sys
sys.path.insert(0, r"e:\compressAI")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("e:/compressAI/modelo_35b", trust_remote_code=True)
print("Chat template:\n", tok.chat_template)
