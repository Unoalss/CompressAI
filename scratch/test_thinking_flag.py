import sys
sys.path.insert(0, r"e:\compressAI")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("e:/compressAI/modelo_35b", trust_remote_code=True)
msgs = [{"role": "user", "content": "Olá"}]
t1 = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
print("t1 (no thinking):\n", repr(t1))
t2 = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
print("t2 (with thinking):\n", repr(t2))
