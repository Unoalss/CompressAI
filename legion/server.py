"""
legion/server.py - Servidor HTTP Offline do Chatbot LEGION.
Inicia o serviço na porta 7777 e conecta o navegador a interface com telemetria em tempo real.
"""
import os, sys, time, json, mimetypes, threading, uuid
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import urllib.parse
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legion.engine import LegionEngine

PORT = 7777
STATIC_DIR = Path(__file__).resolve().parent / "static"

print("=" * 80)
print(f"INICIANDO SERVIDOR WEB LEGION (PORTA {PORT})")
print("=" * 80)

model_arg = sys.argv[1] if len(sys.argv) > 1 else None
engine = LegionEngine(model_path=model_arg, cache_capacity=160)

class LegionHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Desativa log flood para manter o terminal limpo
        pass

    def send_json(self, status_code, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self.serve_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            return
        elif path == "/style.css":
            self.serve_file(STATIC_DIR / "style.css", "text/css")
            return
        elif path == "/app.js":
            self.serve_file(STATIC_DIR / "app.js", "application/javascript")
            return
        elif path == "/api/status":
            self.send_json(200, engine.get_stats())
            return
        else:
            local_file = STATIC_DIR / path.lstrip("/")
            if local_file.exists() and local_file.is_file():
                mime, _ = mimetypes.guess_type(str(local_file))
                self.serve_file(local_file, mime or "application/octet-stream")
                return
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_len = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_len) if content_len > 0 else b"{}"
        try:
            payload = json.loads(body_bytes.decode("utf-8"))
        except Exception:
            payload = {}

        if path == "/api/chat":
            self.handle_streaming_chat(payload)
        elif path == "/v1/chat/completions":
            self.handle_openai_chat(payload)
        else:
            self.send_response(404)
            self.end_headers()

    def serve_file(self, filepath, content_type):
        if not filepath.exists():
            self.send_response(404)
            self.end_headers()
            return
        with open(filepath, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def handle_streaming_chat(self, payload):
        prompt = payload.get("message", "") or payload.get("prompt", "")
        max_tokens = int(payload.get("max_tokens", 300))
        temperature = float(payload.get("temperature", 0.0))
        repetition_penalty = float(payload.get("repetition_penalty", 1.0))
        system_prompt = payload.get("system_prompt", "Você é o LEGION, uma inteligência artificial que comanda uma legião de 10.240 especialistas MoE rodando diretamente a partir de um pendrive USB. Responda em português brasileiro com precisão, inteligência e clareza.")

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        for chunk in engine.generate_stream(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty
        ):
            line = f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")
            try:
                self.wfile.write(line)
                self.wfile.flush()
            except (ConnectionResetError, BrokenPipeError):
                break

        try:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except Exception:
            pass
        self.close_connection = True

    def handle_openai_chat(self, payload):
        messages = payload.get("messages", [])
        prompt = messages[-1].get("content", "") if messages else ""
        system = next((m.get("content") for m in messages if m.get("role") == "system"), None)

        full_text = []
        for chunk in engine.generate_stream(prompt=prompt, system_prompt=system or "Responda em português.", max_tokens=payload.get("max_tokens", 256)):
            full_text.append(chunk["token"])

        res = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "legion-qwen35b-moe",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "".join(full_text)},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": len(full_text), "total_tokens": len(prompt.split()) + len(full_text)}
        }
        self.send_json(200, res)

def run():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), LegionHandler)
    print(f"\n========================================================")
    print(f"  LEGION QUANTUM CHATBOT ONLINE")
    print(f"  URL Local: http://127.0.0.1:{PORT}")
    print(f"  Pressione Ctrl+C para encerrar")
    print(f"========================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nEncerrando servidor...")
        server.server_close()

if __name__ == "__main__":
    run()
