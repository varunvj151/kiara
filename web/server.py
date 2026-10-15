"""
web/server.py — High-Performance Streaming Backend Server for Kiara Web App
=============================================================================
Zero third-party web dependencies (uses Python's standard ThreadingHTTPServer).
- Serves static assets (HTML, CSS, JS).
- Streaming chat generation endpoint (chunked streaming / SSE).
- Model status and checkpoint switcher endpoint.
- Thermal-safe: caps CPU threads to 4.
"""

from __future__ import annotations
import argparse
import json
import mimetypes
import os
import sys
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch

# Protect laptop thermals: cap CPU threads
torch.set_num_threads(4)

from kiara.model import KiaraConfig, KiaraForCausalLM
from kiara.tokenizer import KiaraTokenizer
from kiara.trainer import load_checkpoint

STATIC_DIR = Path(__file__).resolve().parent

# Global Model Container
CURRENT_MODEL: KiaraForCausalLM | None = None
CURRENT_TOKENIZER: KiaraTokenizer | None = None
CURRENT_CONFIG: KiaraConfig | None = None
CURRENT_CKPT_PATH: Path | None = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


mimetypes.init()
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("text/html", ".html")
mimetypes.add_type("application/json", ".json")

def find_available_checkpoints() -> list[dict[str, str]]:
    """Scan local checkpoints directory and Google Drive (if present)."""
    ckpts = []
    ckpt_root = REPO_ROOT / "checkpoints"

    search_dirs = [
        ckpt_root,
        Path("/content/drive/MyDrive/kiara_checkpoints"),
    ]

    seen = set()
    for d in search_dirs:
        if d.exists():
            for pt in d.rglob("*.pt"):
                # Only include checkpoint_final or steps to avoid clutter
                rel_name = f"{pt.parent.name}/{pt.name}"
                if str(pt) not in seen:
                    seen.add(str(pt))
                    ckpts.append({
                        "name": rel_name,
                        "path": str(pt),
                        "size_mb": round(pt.stat().st_size / (1024 * 1024), 1),
                    })
    # Sort with 'final' first, then newer stages
    ckpts.sort(key=lambda x: (0 if "final" in x["name"] else 1, x["name"]), reverse=False)
    return ckpts


def load_kiara_model(ckpt_path: Path):
    global CURRENT_MODEL, CURRENT_TOKENIZER, CURRENT_CONFIG, CURRENT_CKPT_PATH
    print(f"\n[Kiara Server] Loading checkpoint: {ckpt_path} on {DEVICE}...")

    CURRENT_TOKENIZER = KiaraTokenizer.load()

    # Read checkpoint metadata to detect architecture
    raw = torch.load(ckpt_path, map_location="cpu")
    if "config" in raw and isinstance(raw["config"], dict):
        config = KiaraConfig(**raw["config"])
    elif (ckpt_path.parent / "config.json").exists():
        config = KiaraConfig.from_pretrained(ckpt_path.parent)
    elif "26d1" in str(ckpt_path).lower():
        config = KiaraConfig(
            vocab_size=32768, hidden_size=768, intermediate_size=2048,
            num_hidden_layers=12, num_attention_heads=12, num_key_value_heads=3,
            max_position_embeddings=512, tie_word_embeddings=True
        )
    else:
        config = KiaraConfig(
            vocab_size=32768, hidden_size=512, intermediate_size=1536,
            num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=2,
            max_position_embeddings=512, tie_word_embeddings=True
        )

    CURRENT_CONFIG = config
    model = KiaraForCausalLM(config)
    model.load_state_dict(raw["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    CURRENT_MODEL = model
    CURRENT_CKPT_PATH = ckpt_path
    params = sum(p.numel() for p in model.parameters())
    print(f"[Kiara Server] Model online! Parameters: {params:,} | Device: {DEVICE}")


def init_default_model():
    # Pick best available checkpoint
    ckpts = find_available_checkpoints()
    preferred = [
        "kiara_instruct/checkpoint_final.pt",
        "stage_26D2_256/checkpoint_final.pt",
        "stage_26D1/checkpoint_final.pt",
        "stage_26C/checkpoint_final.pt",
    ]

    selected_path = None
    for pref in preferred:
        for c in ckpts:
            if pref in c["path"].replace("\\", "/"):
                selected_path = Path(c["path"])
                break
        if selected_path:
            break

    if not selected_path and ckpts:
        selected_path = Path(ckpts[0]["path"])

    if selected_path:
        load_kiara_model(selected_path)
    else:
        print("[Kiara Server] Warning: No .pt checkpoint found. Initializing random config for demo.")
        global CURRENT_MODEL, CURRENT_TOKENIZER, CURRENT_CONFIG
        CURRENT_TOKENIZER = KiaraTokenizer.load()
        CURRENT_CONFIG = KiaraConfig(vocab_size=32768, hidden_size=256, intermediate_size=512, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
        CURRENT_MODEL = KiaraForCausalLM(CURRENT_CONFIG).to(DEVICE).eval()


class KiaraHTTPHandler(SimpleHTTPRequestHandler):
    """Custom HTTP Request Handler for Kiara Web Application."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)

        # Status endpoint
        if parsed.path == "/api/status":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            params = sum(p.numel() for p in CURRENT_MODEL.parameters()) if CURRENT_MODEL else 0
            status_data = {
                "status": "online",
                "model_name": "Kiara",
                "checkpoint": f"{CURRENT_CKPT_PATH.parent.name}/{CURRENT_CKPT_PATH.name}" if CURRENT_CKPT_PATH else "demo",
                "checkpoint_path": str(CURRENT_CKPT_PATH) if CURRENT_CKPT_PATH else "",
                "parameters": params,
                "device": str(DEVICE),
                "threads": torch.get_num_threads(),
                "available_checkpoints": find_available_checkpoints(),
            }
            self.wfile.write(json.dumps(status_data).encode("utf-8"))
            return

        # Serve static frontend files
        if parsed.path == "/" or parsed.path == "":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)

        # Switch checkpoint endpoint
        if parsed.path == "/api/switch_checkpoint":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body)
            ckpt_path = Path(data.get("path", ""))

            if ckpt_path.exists() and ckpt_path.is_file():
                load_kiara_model(ckpt_path)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "success": True,
                    "checkpoint": f"{ckpt_path.parent.name}/{ckpt_path.name}",
                    "checkpoint_path": str(ckpt_path)
                }).encode("utf-8"))
            else:
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"File not found: {ckpt_path}"}).encode("utf-8"))
            return

        # Streaming Chat Generation endpoint
        if parsed.path == "/api/chat":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(body)
            except Exception:
                data = {}

            prompt = data.get("prompt", "").strip()
            max_new_tokens = int(data.get("max_tokens", 80))
            temperature = float(data.get("temperature", 0.7))
            top_k = int(data.get("top_k", 40))
            instruct = bool(data.get("instruct", True))

            if not prompt:
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.end_headers()
                return

            # Format prompt
            if instruct:
                formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
            else:
                formatted_prompt = prompt

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            # Autoregressive Streaming Generation
            input_ids = CURRENT_TOKENIZER.encode(formatted_prompt, bos=True)
            generated = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
            past_key_values = None
            eos_id = CURRENT_TOKENIZER.eos_token_id

            t0 = time.time()
            tokens_generated = 0

            with torch.no_grad():
                for _ in range(max_new_tokens):
                    if past_key_values is not None:
                        cur_input = generated[:, -1:]
                    else:
                        cur_input = generated

                    outputs = CURRENT_MODEL.forward(
                        input_ids=cur_input,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    past_key_values = outputs.past_key_values
                    next_token_logits = outputs.logits[:, -1, :]

                    if temperature > 0:
                        next_token_logits = next_token_logits / temperature
                        if top_k > 0:
                            v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                            next_token_logits[next_token_logits < v[:, [-1]]] = float("-inf")
                        probs = torch.softmax(next_token_logits, dim=-1)
                        next_tok = torch.multinomial(probs, num_samples=1)
                    else:
                        next_tok = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                    tok_id = next_tok.item()
                    generated = torch.cat((generated, next_tok), dim=-1)
                    tokens_generated += 1

                    # Decode single token
                    token_str = CURRENT_TOKENIZER.decode([tok_id], skip_special_tokens=False)

                    # Stop conditions
                    if tok_id == eos_id or "[EOS]" in token_str or "<|im_end|>" in token_str:
                        break
                    if "###" in token_str:
                        break

                    # Send SSE chunk
                    chunk_payload = json.dumps({"token": token_str, "done": False})
                    try:
                        self.wfile.write(f"data: {chunk_payload}\n\n".encode("utf-8"))
                        self.wfile.flush()
                    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                        print("[Kiara Server] Client disconnected early.")
                        return

            elapsed = max(0.001, time.time() - t0)
            tok_per_sec = round(tokens_generated / elapsed, 1)

            # Send done event with metrics
            done_payload = json.dumps({
                "token": "",
                "done": True,
                "tokens_generated": tokens_generated,
                "elapsed_sec": round(elapsed, 2),
                "tok_per_sec": tok_per_sec,
            })
            try:
                self.wfile.write(f"data: {done_payload}\n\n".encode("utf-8"))
                self.wfile.flush()
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                pass
            self.close_connection = True
            return

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def run_server(port: int = 8000):
    init_default_model()
    server_address = ("127.0.0.1", port)
    httpd = ThreadingHTTPServer(server_address, KiaraHTTPHandler)
    print("\n" + "=" * 70)
    print(f"  ✨ KIARA CHAT WEB APP IS RUNNING!")
    print(f"  👉 Open in your browser: http://127.0.0.1:{port}")
    print("=" * 70 + "\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Kiara Server.")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000, help="Port to run server on")
    args = parser.parse_args()
    run_server(port=args.port)
