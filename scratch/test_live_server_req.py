import urllib.request, json, time

req_data = json.dumps({
    "prompt": "Explique brevemente o que é a Teoria da Relatividade de Einstein.",
    "max_tokens": 50,
    "temperature": 0.7
}).encode("utf-8")

req = urllib.request.Request(
    "http://127.0.0.1:8080/api/generate",
    data=req_data,
    headers={"Content-Type": "application/json"}
)

print("Enviando requisição SSE para o servidor web...")
t0 = time.time()
with urllib.request.urlopen(req) as resp:
    buffer = ""
    for line in resp:
        line_str = line.decode("utf-8")
        if line_str.startswith("data: "):
            payload_str = line_str[6:].strip()
            if payload_str:
                data = json.loads(payload_str)
                if data.get("type") == "token":
                    print(data["token"], end="", flush=True)
                elif data.get("type") == "done":
                    print(f"\n\n[Concluído: {data['total_tokens']} tokens em {data['elapsed_s']}s -> {data['final_tok_s']} tok/s]")
                    break

print(f"\nTempo total da requisição: {time.time()-t0:.2f}s")
