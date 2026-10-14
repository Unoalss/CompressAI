"""
scratch/test_chat_stream.py
Testa a API de streaming do LEGION Chatbot e exibe a resposta em tempo real.
"""
import sys, json, urllib.request

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

url = "http://127.0.0.1:7777/api/chat"
payload = {
    "message": "Olá LEGION! Apresente-se e explique como funciona sua legião de especialistas em precisão mista.",
    "max_tokens": 150,
    "temperature": 0.6
}

req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

print("-" * 60)
print(f"Enviando mensagem para: {url}...")
print(f"Pergunta: \"{payload['message']}\"")
print("-" * 60)
print("LEGION: ", end="", flush=True)

with urllib.request.urlopen(req) as resp:
    for line in resp:
        line_str = line.decode("utf-8").strip()
        if line_str.startswith("data: ") and "[DONE]" not in line_str:
            data = json.loads(line_str[6:])
            tok = data.get("token", "")
            print(tok, end="", flush=True)

print("\n" + "-" * 60)
print("[OK] Teste de chat concluído com sucesso!")
