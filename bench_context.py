"""
Long-context benchmark: TurboQuant vs baseline at 4K, 8K, 16K, 32K, 64K.

Strategy: fill context with a long document containing a hidden fact (needle),
then ask the model to retrieve it. Measures:
  - Prefill time
  - KV cache memory
  - Generation speed
  - Whether the model finds the needle (quality check)
"""

import time
import argparse
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache, KVCache
from mlx_turbo.kv_cache import TurboQuantKVCache


FILLER_PARAGRAPH = (
    "The history of artificial intelligence research spans decades of innovation "
    "and setbacks. Early pioneers like Alan Turing and John McCarthy laid the "
    "groundwork for what would become one of the most transformative fields in "
    "computer science. The development of neural networks, expert systems, and "
    "machine learning algorithms has led to remarkable advances in natural language "
    "processing, computer vision, and robotics. Modern large language models "
    "represent the culmination of years of research into transformer architectures, "
    "attention mechanisms, and scaling laws. These models have demonstrated "
    "surprising emergent capabilities as they grow in size and training data. "
)

NEEDLE = "The secret password for Project Aurora is 'crystalline-nebula-7742'."

QUESTION = "What is the secret password for Project Aurora?"


def build_long_prompt(tokenizer, target_tokens, needle_position=0.5):
    """Build a prompt of approximately target_tokens length with a hidden needle."""
    # Tokenize filler to estimate tokens per paragraph
    filler_tokens = len(tokenizer.encode(FILLER_PARAGRAPH))
    needle_tokens = len(tokenizer.encode(NEEDLE))

    # Calculate how many filler paragraphs we need
    overhead = 100  # for system prompt, question, etc.
    available = target_tokens - overhead - needle_tokens
    n_paragraphs = max(1, available // filler_tokens)

    # Insert needle at the specified position
    needle_idx = int(n_paragraphs * needle_position)

    paragraphs = []
    for i in range(n_paragraphs):
        if i == needle_idx:
            paragraphs.append(NEEDLE)
        paragraphs.append(FILLER_PARAGRAPH)

    document = "\n\n".join(paragraphs)

    messages = [
        {
            "role": "user",
            "content": (
                f"Read the following document carefully, then answer the question.\n\n"
                f"--- DOCUMENT START ---\n{document}\n--- DOCUMENT END ---\n\n"
                f"Question: {QUESTION}\n"
                f"Answer concisely with just the password."
            ),
        }
    ]

    prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    return prompt_tokens


def make_tq_cache(model, bits):
    """Create hybrid cache with TurboQuant on KVCache layers."""
    default = make_prompt_cache(model)
    cache = []
    cfg = model.args if hasattr(model, "args") else model.config
    # Get head_dim - handle nested config
    if hasattr(cfg, "text_config") and isinstance(cfg.text_config, dict):
        head_dim = cfg.text_config.get("head_dim", 128)
    elif hasattr(cfg, "head_dim"):
        head_dim = cfg.head_dim
    else:
        head_dim = 128

    for c in default:
        if isinstance(c, KVCache):
            cache.append(TurboQuantKVCache(bits=bits, head_dim=head_dim))
        else:
            cache.append(c)
    return cache


def run_benchmark(model, tokenizer, prompt_tokens, cache, label, max_gen=60):
    """Run generation and collect metrics."""
    prompt = mx.array(prompt_tokens)

    t0 = time.time()
    tokens = []
    for token, _ in generate_step(
        prompt, model, max_tokens=max_gen, prompt_cache=cache
    ):
        tok = token if isinstance(token, int) else token.item()
        tokens.append(tok)
        if len(tokens) == 1:
            ttft = time.time() - t0

    total = time.time() - t0
    text = tokenizer.decode(tokens)
    gen_speed = len(tokens) / (total - ttft) if total > ttft else 0

    # Memory from TQ layers
    tq_bytes = sum(c.nbytes for c in cache if isinstance(c, TurboQuantKVCache))
    kv_bytes = 0
    for c in cache:
        if isinstance(c, KVCache) and c.keys is not None:
            kv_bytes += c.keys.nbytes + c.values.nbytes
        elif isinstance(c, TurboQuantKVCache):
            kv_bytes += c.nbytes

    # Check if needle was found
    found = "crystalline-nebula-7742" in text.lower() or "7742" in text

    return {
        "label": label,
        "ttft": ttft,
        "total": total,
        "gen_speed": gen_speed,
        "n_tokens": len(tokens),
        "kv_bytes": kv_bytes,
        "tq_bytes": tq_bytes,
        "found_needle": found,
        "text": text.strip()[:200],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen3.5-9B-OptiQ-4bit")
    parser.add_argument(
        "--contexts", nargs="+", type=int, default=[4096, 8192, 16384, 32768]
    )
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--max-gen", type=int, default=60)
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    model, tokenizer = load(args.model)
    print("Loaded.\n")

    results = []

    for ctx in args.contexts:
        print(f"{'=' * 70}")
        print(f"  CONTEXT LENGTH: {ctx:,} tokens")
        print(f"{'=' * 70}")

        prompt_tokens = build_long_prompt(tokenizer, ctx)
        actual_len = len(prompt_tokens)
        print(f"  Actual prompt length: {actual_len:,} tokens")

        # --- Baseline ---
        if not args.skip_baseline:
            print(f"\n  [Baseline FP16]")
            cache_bl = make_prompt_cache(model)
            try:
                r = run_benchmark(
                    model, tokenizer, prompt_tokens, cache_bl, "baseline", args.max_gen
                )
                results.append({"ctx": ctx, **r})
                print(
                    f"    TTFT: {r['ttft']:.1f}s | Gen: {r['gen_speed']:.1f} tok/s | "
                    f"KV: {r['kv_bytes'] / 1e6:.1f} MB | Needle: {'YES' if r['found_needle'] else 'no'}"
                )
                print(f"    Output: {r['text'][:120]}")
            except Exception as e:
                print(f"    FAILED: {e}")
                results.append({"ctx": ctx, "label": "baseline", "error": str(e)})

            # Clear memory
            del cache_bl
            mx.clear_cache()

        # --- TurboQuant ---
        print(f"\n  [TurboQuant {args.bits}-bit]")
        cache_tq = make_tq_cache(model, args.bits)
        try:
            r = run_benchmark(
                model,
                tokenizer,
                prompt_tokens,
                cache_tq,
                f"tq{args.bits}",
                args.max_gen,
            )
            results.append({"ctx": ctx, **r})
            print(
                f"    TTFT: {r['ttft']:.1f}s | Gen: {r['gen_speed']:.1f} tok/s | "
                f"KV: {r['kv_bytes'] / 1e6:.1f} MB | Needle: {'YES' if r['found_needle'] else 'no'}"
            )
            print(f"    Output: {r['text'][:120]}")
        except Exception as e:
            print(f"    FAILED: {e}")
            results.append({"ctx": ctx, "label": f"tq{args.bits}", "error": str(e)})

        del cache_tq
        mx.clear_cache()

        print()

    # --- Summary Table ---
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(
        f"  {'Context':>8}  {'Mode':>12}  {'TTFT':>7}  {'tok/s':>7}  {'KV MB':>7}  {'Needle':>7}"
    )
    print(
        f"  {'-------':>8}  {'----':>12}  {'----':>7}  {'-----':>7}  {'-----':>7}  {'------':>7}"
    )
    for r in results:
        if "error" in r:
            print(f"  {r['ctx']:>8}  {r['label']:>12}  {'FAILED':>7}")
        else:
            print(
                f"  {r['ctx']:>8}  {r['label']:>12}  {r['ttft']:>6.1f}s  "
                f"{r['gen_speed']:>6.1f}  {r['kv_bytes'] / 1e6:>6.1f}  "
                f"{'YES' if r['found_needle'] else 'no':>6}"
            )


if __name__ == "__main__":
    main()
