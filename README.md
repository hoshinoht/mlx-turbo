# mlx-turbo

TurboQuant KV cache compression for MLX on Apple Silicon. Rust core, Python integration.

Compresses KV cache to 3 bits per value (~4.7x) with near-zero accuracy loss, letting you run longer contexts in the same memory.

Based on Google Research's [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026).

## Install

```bash
# Build from source (requires Rust toolchain + maturin)
pip install maturin
git clone https://github.com/hoshinoht/mlx-turbo.git
cd mlx-turbo
maturin build --release
pip install target/wheels/*.whl
```

## Quick start

```python
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache, KVCache
from mlx_turbo.kv_cache import TurboQuantKVCache

model, tokenizer = load("mlx-community/Qwen3.5-9B-OptiQ-4bit")

# Patch KV cache layers with 3-bit TurboQuant compression
cache = [
    TurboQuantKVCache(bits=3, head_dim=256) if isinstance(c, KVCache) else c
    for c in make_prompt_cache(model)
]

# Generate as usual — cache compression is transparent
prompt = mx.array(tokenizer.encode("Hello"))
for token, _ in generate_step(prompt, model, max_tokens=200, prompt_cache=cache):
    print(tokenizer.decode([token]), end="", flush=True)
```

Or use the included `generate.py`:

```bash
python generate.py --model mlx-community/Qwen3.5-9B-OptiQ-4bit --prompt "Hello" --bits 3
python generate.py --model mlx-community/Qwen3.5-9B-OptiQ-4bit --prompt "Hello" --baseline
```

## How it works

1. **Random rotation** (Fast Walsh-Hadamard Transform) makes vector coordinates approximately Gaussian
2. **Lloyd-Max quantization** maps each coordinate to optimal centroids for that distribution
3. **Bit packing** stores 3-bit indices into u32 words

The algorithm is **model-agnostic** — works with any mlx-lm model that uses `KVCache`. Hybrid models (Qwen3.5 with linear attention layers) are handled automatically.

## Benchmarks

**Qwen3.5-9B-OptiQ-4bit on 24GB Mac (M4 Pro)**

| Context | Baseline tok/s | TurboQuant tok/s | Baseline KV | TurboQuant KV | Compression |
|---------|---------------|-----------------|-------------|---------------|-------------|
| 4K | 42.6 | 36.6 | 131 MB | 28 MB | **4.7x** |
| 8K | — | 35.1 | 268 MB | 56 MB | **4.7x** |
| 16K | — | 32.3 | 524 MB | 112 MB | **4.7x** |
| 32K | — | 30.0 | 1,074 MB | 226 MB | **4.7x** |

MSE distortion within paper bounds at all bit widths:

| Bits | MSE (measured) | MSE (paper bound) |
|------|---------------|-------------------|
| 2 | 0.115 | 0.170 |
| 3 | 0.034 | 0.043 |
| 4 | 0.009 | 0.011 |

## Architecture

```
src/           Rust core (PyO3 + numpy)
  lib.rs         PyO3 bindings
  fwht.rs        Fast Walsh-Hadamard Transform
  codebook.rs    Lloyd-Max solver
  quantizer.rs   Quantize + pack/unpack
  engine.rs      Compress/decompress API

mlx_turbo/     Python integration
  __init__.py    Exports from Rust _core
  kv_cache.py    mlx-lm KVCache drop-in replacement
```

## License

MIT
