# mlx-turbo Evaluation Results

**Date:** 2026-03-29  
**Hardware:** Apple M4 Pro, 24GB unified memory  
**TurboQuant:** 3-bit MSE-only, Rust backend  

## Models

| | Mistral Nemo 12B | Qwen3.5 9B |
|---|---|---|
| Repo | `mlx-community/Mistral-Nemo-Instruct-2407-4bit` | `mlx-community/Qwen3.5-9B-OptiQ-4bit` |
| Architecture | Standard transformer | Hybrid (linear + full attention) |
| Total layers | 40 | 32 |
| KVCache layers | 40 (100%) | 8 (25%) |
| head_dim | 128 | 256 |
| KV heads | 8 | 4 |

## Memory Compression

Constant ~4.7x regardless of model, context length, or prompt content.

| Context | Mistral BL | Mistral TQ | Ratio | Qwen BL | Qwen TQ | Ratio |
|---------|-----------|-----------|-------|---------|---------|-------|
| 500 | 84 MB | 18 MB | 4.6x | 17 MB | 4 MB | 4.7x |
| 2,000 | 329 MB | 72 MB | 4.6x | 66 MB | 14 MB | 4.7x |
| 2,700 | 445 MB | 97 MB | 4.6x | 89 MB | 19 MB | 4.7x |
| 4,096 | — | — | — | 131 MB | 28 MB | 4.7x |
| 8,192 | — | — | — | 268 MB | 56 MB | 4.7x |
| 16,384 | — | — | — | 524 MB | 112 MB | 4.7x |
| 32,768 | — | — | — | 1,074 MB | 226 MB | 4.7x |

## Decode Speed

| | Mistral Nemo 12B | Qwen3.5 9B |
|---|---|---|
| Baseline | 34.4 tok/s | 43.6 tok/s |
| TurboQuant | 21.9 tok/s | 36.9 tok/s |
| Overhead | -36% | -15% |

Overhead is proportional to KVCache layer count (40 vs 8) due to MLX↔numpy↔Rust marshalling per layer.

### Qwen3.5 9B Long Context Decode

| Context | tok/s |
|---------|-------|
| 4K | 36.6 |
| 8K | 35.1 |
| 16K | 32.3 |
| 32K | 30.0 |

## Quality (200 tok generation)

### Mistral Nemo 12B

| Task | BL tok/s | TQ tok/s | Token Match |
|------|----------|----------|-------------|
| Factual | 33.8 | 21.5 | 13% |
| Math | 34.5 | 22.0 | 4% |
| Code | 34.6 | 21.8 | 4% |
| Reasoning | 34.6 | 22.1 | 4% |
| Creative | 34.5 | 22.0 | 14% |
| **Average** | **34.4** | **21.9** | **8%** |

### Qwen3.5 9B

| Task | BL tok/s | TQ tok/s | Token Match |
|------|----------|----------|-------------|
| Factual | 43.8 | 37.0 | 53% |
| Math | 43.9 | 37.0 | 22% |
| Code | 43.6 | 36.9 | 23% |
| Reasoning | 43.2 | 36.9 | 10% |
| Creative | 43.4 | 36.6 | 22% |
| **Average** | **43.6** | **36.9** | **26%** |

Low token match ≠ wrong answers. Both models produce correct, coherent outputs. Divergence is from lossy compression perturbing attention scores, causing different (but valid) sampling paths.

## MSE Distortion (unit vectors)

| Bits | d=64 | d=128 | d=256 | Paper Bound |
|------|------|-------|-------|-------------|
| 2 | 0.114 | 0.115 | 0.117 | 0.170 |
| 3 | 0.033 | 0.034 | 0.034 | 0.043 |
| 4 | 0.009 | 0.009 | 0.010 | 0.011 |

## Rust Core Performance

| Vectors | Compress | Decompress | Throughput |
|---------|----------|------------|------------|
| 1K | 2.7ms | 0.7ms | 301K vec/s |
| 10K | 25.5ms | 7.6ms | 303K vec/s |
| 100K | 247ms | 78ms | 308K vec/s |
