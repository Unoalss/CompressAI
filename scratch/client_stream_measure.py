import urllib.request
import json
import time

url = "http://127.0.0.1:7777/api/chat"
payload = {
    "message": "Explique o que e a teoria da relatividade em poucas palavras.",
    "max_tokens": 30,
    "temperature": 0.0
}

req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

print("Enviando requisicao...")
t_req_start = time.perf_counter()
t_prev = t_req_start
tok_count = 0

with urllib.request.urlopen(req) as resp:
    t_first = None
    for line in resp:
        line = line.decode("utf-8").strip()
        if not line or not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str == "[DONE]":
            break
        data = json.loads(data_str)
        t_now = time.perf_counter()
        if t_first is None:
            t_first = t_now
            prefill_time = (t_first - t_req_start) * 1000
            print(f"Prefill: {prefill_time:.1f} ms")
        dt_ms = (t_now - t_prev) * 1000
        t_prev = t_now
        tok_count += 1
        stats = data.get("stats", {})
        print(f"Tok {tok_count:2d}: {data.get('token')!r:15s} | dt: {dt_ms:6.1f} ms | server step_ms: {stats.get('step_ms')} ms | tok/s: {stats.get('tok_s')} | Hit: {stats.get('hit_rate_pct')}%")

t_end = time.perf_counter()
print(f"\nTotal: {tok_count} tokens em {t_end - t_first:.2f}s (Decode Speed: {tok_count / (t_end - t_first):.2f} tok/s)")
