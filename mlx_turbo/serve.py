"""
mlx-turbo serve: OpenAI-compatible API server with TurboQuant KV cache.

Usage:
    python -m mlx_turbo.serve --model mlx-community/Qwen3.5-9B-OptiQ-4bit --bits 3
    python -m mlx_turbo.serve --model mlx-community/Qwen3.5-9B-OptiQ-4bit --bits 3 --port 8080
"""

import argparse
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache, KVCache

from mlx_turbo.patch import patch, _detect_head_dim
from mlx_turbo.kv_cache import TurboQuantKVCache

MODEL = None
TOKENIZER = None
ARGS = None


def make_cache():
    """Create a fresh TurboQuant-patched cache."""
    default = make_prompt_cache(MODEL)
    hd = _detect_head_dim(MODEL)
    if hd & (hd - 1) != 0:
        hd = 1 << (hd - 1).bit_length()
    return [
        TurboQuantKVCache(bits=ARGS.bits, head_dim=hd) if isinstance(c, KVCache) else c
        for c in default
    ]


def generate(messages, max_tokens=256, temperature=0.0, stream=False):
    """Run generation with TurboQuant cache, yield token strings."""
    prompt_tokens = TOKENIZER.apply_chat_template(messages, add_generation_prompt=True)
    prompt = mx.array(prompt_tokens)
    cache = make_cache()

    sampler = None
    if temperature > 0:
        from mlx_lm.generate import make_sampler

        sampler = make_sampler(temperature)

    for token, _ in generate_step(
        prompt,
        MODEL,
        max_tokens=max_tokens,
        prompt_cache=cache,
        sampler=sampler,
    ):
        tok_id = token if isinstance(token, int) else token.item()
        text = TOKENIZER.decode([tok_id])
        yield tok_id, text


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        pass  # Suppress default logging

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models":
            self._send_json(
                {
                    "object": "list",
                    "data": [{"id": ARGS.model, "object": "model"}],
                }
            )
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._send_json({"error": "not found"}, 404)
            return

        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", 256)
        temperature = body.get("temperature", 0.0)
        stream = body.get("stream", False)
        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            for _, text in generate(messages, max_tokens, temperature):
                chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": text},
                            "finish_reason": None,
                        }
                    ],
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()

            # Final chunk
            done = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            self.wfile.write(f"data: {json.dumps(done)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        else:
            tokens = []
            full_text = ""
            t0 = time.time()
            for tok_id, text in generate(messages, max_tokens, temperature):
                tokens.append(tok_id)
                full_text += text
            elapsed = time.time() - t0

            self._send_json(
                {
                    "id": req_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": ARGS.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": full_text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": len(tokens),
                        "total_tokens": len(tokens),
                    },
                }
            )
            print(
                f"  [{len(tokens)} tok, {elapsed:.1f}s, {len(tokens) / elapsed:.0f} tok/s]"
            )


def main():
    global MODEL, TOKENIZER, ARGS

    parser = argparse.ArgumentParser(description="mlx-turbo: OpenAI-compatible server")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--bits", type=int, default=3, choices=[2, 3, 4])
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    ARGS = parser.parse_args()

    print(f"[mlx-turbo] Loading {ARGS.model}...")
    MODEL, TOKENIZER = load(ARGS.model)
    hd = _detect_head_dim(MODEL)
    n_layers = len(MODEL.layers)
    n_kv = sum(1 for c in make_prompt_cache(MODEL) if isinstance(c, KVCache))
    print(
        f"[mlx-turbo] {ARGS.bits}-bit TurboQuant | head_dim={hd} | {n_kv}/{n_layers} KV layers"
    )
    print(f"[mlx-turbo] Serving on http://{ARGS.host}:{ARGS.port}")
    print()

    server = HTTPServer((ARGS.host, ARGS.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[mlx-turbo] Shutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
