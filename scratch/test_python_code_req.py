import urllib.request, json, time

req_data = json.dumps({
    "prompt": "Escreva uma função em Python para calcular o fatorial de um número.",
    "max_tokens": 60,
    "temperature": 0.7
}).encode("utf-8")

req = urllib.request.Request(
    "http://127.0.0.1:8080/api/generate",
    data=req_data,
    headers={"Content-Type": "application/json"}
)

print("Enviando requisição de código Python...")
with urllib.request.urlopen(req) as resp:
    for line in resp:
        line_str = line.decode("utf-8")
        if line_str.startswith("data: "):
            payload_str = line_str[6:].strip()
            if payload_str:
                data = json.loads(payload_str)
                if data.get("type") == "token":
                    print(data["token"], end="", flush=True)
                elif data.get("type") == "done":
                    print(f"\n\n[Concluído: {data['total_tokens']} tokens]")
                    break
