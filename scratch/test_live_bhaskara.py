import urllib.request, json, time, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

req_data = json.dumps({
    "prompt": "Explique de forma resumida a fórmula de Bhaskara e para que ela serve.",
    "max_tokens": 60,
    "temperature": 0.7
}).encode("utf-8")

req = urllib.request.Request(
    "http://127.0.0.1:8080/api/generate",
    data=req_data,
    headers={"Content-Type": "application/json"}
)

print("Enviando pergunta sobre Bhaskara...")
t0 = time.time()
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
                    print(f"\n\n[Concluído: {data['total_tokens']} tokens em {data['elapsed_s']}s -> {data['final_tok_s']} tok/s]")
                    break

print(f"\nTempo total: {time.time()-t0:.2f}s")
