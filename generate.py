"""
TurboQuant generation: run LLMs with compressed KV cache on Apple Silicon.

Patches mlx-lm's generation pipeline to use TurboQuant KV cache compression,
enabling longer context windows within the same memory budget.

Usage:
    # GPT-OSS 20B (recommended for 24GB Mac)
    python generate.py --model NexaAI/gpt-oss-20b-MLX-4bit \
                       --prompt "Explain quantum computing simply." \
                       --bits 3 --max-tokens 200

    # Qwen3.5 27B
    python generate.py --model mlx-community/Qwen3.5-27B-4bit \
                       --prompt "Hello" --bits 3

    # Baseline comparison (no TurboQuant)
    python generate.py --model NexaAI/gpt-oss-20b-MLX-4bit \
                       --prompt "Hello" --baseline

    # Memory projection only (no model download)
    python generate.py --model NexaAI/gpt-oss-20b-MLX-4bit --project-only
"""

import argparse
import sys
import time
import functools


def _cfg_get(obj, key, default=None):
    """Get a config value from an object or dict."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def detect_model_config(model):
    """Extract architecture parameters from loaded model."""
    config = model.args if hasattr(model, "args") else model.config

    hidden_size = _cfg_get(config, "hidden_size", 4096)
    num_heads = _cfg_get(config, "num_attention_heads", 32)
    num_kv_heads = _cfg_get(config, "num_key_value_heads", num_heads)
    head_dim = _cfg_get(config, "head_dim", hidden_size // num_heads)
    n_layers = _cfg_get(config, "num_hidden_layers", 32)
    model_type = _cfg_get(config, "model_type", "unknown")

    # Some models nest config (e.g., Qwen3.5 with text_config dict)
    tc = _cfg_get(config, "text_config", None)
    if tc is not None:
        hidden_size = _cfg_get(tc, "hidden_size", hidden_size)
        num_heads = _cfg_get(tc, "num_attention_heads", num_heads)
        num_kv_heads = _cfg_get(tc, "num_key_value_heads", num_kv_heads)
        head_dim = _cfg_get(tc, "head_dim", hidden_size // num_heads)
        n_layers = _cfg_get(tc, "num_hidden_layers", n_layers)
        model_type = _cfg_get(tc, "model_type", model_type)

    return {
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "n_layers": n_layers,
        "model_type": model_type,
    }


def make_turboquant_cache(model, bits=3):
    """
    Create a cache list for a model, replacing KVCache entries with TurboQuant.

    For hybrid models (e.g., Qwen3.5 with linear + full attention), only the
    full attention layers use KVCache. Linear attention layers use ArraysCache
    which we leave untouched.
    """
    from mlx_turbo.kv_cache import TurboQuantKVCache
    from mlx_lm.models.cache import make_prompt_cache, KVCache

    cfg = detect_model_config(model)
    head_dim = cfg["head_dim"]

    # TurboQuant requires power-of-2 head_dim
    if head_dim & (head_dim - 1) != 0:
        padded = 1 << (head_dim - 1).bit_length()
        print(f"  WARNING: head_dim={head_dim} not power of 2, padding to {padded}")
        head_dim = padded

    # Get the model's default cache layout (respects hybrid architectures)
    default_cache = make_prompt_cache(model)

    # Replace only KVCache entries with TurboQuantKVCache
    caches = []
    tq_count = 0
    for i, c in enumerate(default_cache):
        if isinstance(c, KVCache):
            caches.append(TurboQuantKVCache(bits=bits, head_dim=head_dim))
            tq_count += 1
        else:
            caches.append(c)

    print(
        f"  TurboQuant: {bits}-bit, head_dim={head_dim}, "
        f"{tq_count}/{len(caches)} layers patched (Rust backend)"
    )
    return caches


def project_memory(model_name, bits=3):
    """Print memory projections without loading the model."""
    from mlx_turbo._core import TurboEngine

    # Known model configs
    configs = {
        "gpt-oss-20b": {
            "n_layers": 24,
            "num_kv_heads": 8,
            "head_dim": 64,
            "model_bytes_q4": 11e9,
            "name": "GPT-OSS 20B (Q4)",
        },
        "qwen3.5-27b": {
            "n_layers": 64,
            "num_kv_heads": 4,
            "head_dim": 256,
            "model_bytes_q4": 15e9,
            "name": "Qwen3.5 27B (Q4)",
            # Note: only 16/64 layers have full attention KV cache
            "kv_layers": 16,
        },
    }

    # Try to match
    key = None
    model_lower = model_name.lower()
    for k in configs:
        if k.replace("-", "").replace(".", "") in model_lower.replace("-", "").replace(
            ".", ""
        ):
            key = k
            break

    if key is None:
        print(f"Unknown model '{model_name}'. Known configs: {list(configs.keys())}")
        return

    c = configs[key]
    n_kv_layers = c.get("kv_layers", c["n_layers"])
    hd = c["head_dim"]
    nkv = c["num_kv_heads"]

    # Ensure head_dim is power of 2 for engine
    engine_dim = hd if (hd & (hd - 1)) == 0 else (1 << (hd - 1).bit_length())
    engine = TurboEngine(bits, engine_dim)

    print(f"\n{'=' * 65}")
    print(f"  Memory Projection: {c['name']}")
    print(f"{'=' * 65}")
    print(f"  KV layers: {n_kv_layers}, KV heads: {nkv}, head_dim: {hd}")
    print(f"  Model weights: ~{c['model_bytes_q4'] / 1e9:.0f} GB")
    print(
        f"  TurboQuant: {bits}-bit, {engine.bits_per_value():.2f} bits/val, "
        f"{engine.compression_ratio():.1f}x compression"
    )
    print(
        f"  Available for KV (24GB Mac): ~{(24e9 - c['model_bytes_q4'] - 3e9) / 1e9:.0f} GB"
    )
    print()

    avail = 24e9 - c["model_bytes_q4"] - 3e9  # Leave 3GB for OS/overhead

    print(
        f"  {'Context':>8}  {'FP16 KV':>10}  {'TQ' + str(bits) + ' KV':>10}  "
        f"{'Savings':>8}  {'Fits 24GB?':>10}"
    )
    print(
        f"  {'-------':>8}  {'-------':>10}  {'-------':>10}  "
        f"{'-------':>8}  {'----------':>10}"
    )

    for ctx in [4096, 8192, 16384, 32768, 65536, 131072]:
        # FP16: 2 bytes per value, K+V
        fp16 = n_kv_layers * 2 * nkv * ctx * hd * 2
        # TurboQuant
        bpvec = engine.bytes_per_vector()
        tq = n_kv_layers * 2 * nkv * ctx * bpvec
        fits = "YES" if tq < avail else "no"
        print(
            f"  {ctx:>8}  {fp16 / 1e6:>8.0f} MB  {tq / 1e6:>8.0f} MB  "
            f"{fp16 / max(tq, 1):>7.1f}x  {fits:>10}"
        )

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate text with TurboQuant KV cache compression"
    )
    parser.add_argument(
        "--model", type=str, default="mlx-community/Qwen3.5-9B-OptiQ-4bit"
    )
    parser.add_argument(
        "--prompt", type=str, default="Explain quantum computing in simple terms."
    )
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--bits", type=int, default=3, choices=[2, 3, 4])
    parser.add_argument(
        "--baseline", action="store_true", help="Run without TurboQuant"
    )
    parser.add_argument(
        "--project-only", action="store_true", help="Memory projections only"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.project_only:
        project_memory(args.model, bits=args.bits)
        return

    try:
        from mlx_lm import load
        from mlx_lm.generate import generate_step
        import mlx.core as mx
    except ImportError:
        print("ERROR: mlx-lm not installed. Run: pip install mlx-lm")
        sys.exit(1)

    print(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)

    cfg = detect_model_config(model)
    print(
        f"  Architecture: {cfg['model_type']}, {cfg['n_layers']} layers, "
        f"head_dim={cfg['head_dim']}, kv_heads={cfg['num_kv_heads']}"
    )

    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": args.prompt}]
        prompt_tokens = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
    else:
        prompt_tokens = tokenizer.encode(args.prompt)

    prompt = mx.array(prompt_tokens)
    print(f"  Prompt: {len(prompt_tokens)} tokens")

    if args.baseline:
        print(f"\n--- BASELINE (standard KV cache) ---\n")
        t0 = time.time()
        tokens = []
        for token, _ in generate_step(prompt, model, max_tokens=args.max_tokens):
            tokens.append(token if isinstance(token, int) else token.item())
            if len(tokens) == 1:
                ttft = time.time() - t0
        elapsed = time.time() - t0
        text = tokenizer.decode(tokens)
        print(text)
        print(
            f"\n  TTFT: {ttft:.2f}s | {len(tokens)} tok | {elapsed:.1f}s | "
            f"{len(tokens) / elapsed:.1f} tok/s"
        )
    else:
        print(f"\n--- TurboQuant {args.bits}-bit (Rust backend) ---\n")
        cache = make_turboquant_cache(model, bits=args.bits)

        t0 = time.time()
        tokens = []
        for token, _ in generate_step(
            prompt, model, max_tokens=args.max_tokens, prompt_cache=cache
        ):
            tokens.append(token if isinstance(token, int) else token.item())
            if len(tokens) == 1:
                ttft = time.time() - t0
            if args.verbose and len(tokens) % 20 == 0:
                from mlx_turbo.kv_cache import TurboQuantKVCache

                tq_bytes = sum(
                    c.nbytes for c in cache if isinstance(c, TurboQuantKVCache)
                )
                print(f"  [{len(tokens)} tok] TQ KV: {tq_bytes / 1e6:.1f} MB")

        elapsed = time.time() - t0
        text = tokenizer.decode(tokens)
        print(text)

        from mlx_turbo.kv_cache import TurboQuantKVCache

        tq_bytes = sum(c.nbytes for c in cache if isinstance(c, TurboQuantKVCache))
        tq_layers = sum(1 for c in cache if isinstance(c, TurboQuantKVCache))
        max_off = max(
            (c.offset for c in cache if isinstance(c, TurboQuantKVCache)), default=0
        )
        fp16_equiv = max_off * cfg["head_dim"] * 2 * 2 * tq_layers

        print(
            f"\n  TTFT: {ttft:.2f}s | {len(tokens)} tok | {elapsed:.1f}s | "
            f"{len(tokens) / elapsed:.1f} tok/s"
        )
        print(
            f"  KV: {tq_bytes / 1e6:.1f} MB compressed (vs {fp16_equiv / 1e6:.1f} MB FP16, "
            f"{fp16_equiv / max(tq_bytes, 1):.1f}x)"
        )


if __name__ == "__main__":
    main()
