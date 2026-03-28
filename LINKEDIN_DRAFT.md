# LinkedIn Post Draft

---

Google dropped the TurboQuant paper 3 days ago. I spent the weekend implementing it.

TurboQuant is a KV cache compression algorithm from Google Research (ICLR 2026). The idea: rotate vectors with a Walsh-Hadamard transform, then quantize each coordinate independently using optimal Lloyd-Max centroids. No training, no calibration data. Just math.

The KV cache is the hidden memory killer in LLM inference. At 32K context, a 9B model's KV cache eats over 1GB. On a 24GB Mac, that's the difference between "it works" and "it crashes."

I wanted to see if I could get it running locally on Apple Silicon.

**What I built:**

mlx-turbo — a Rust-core TurboQuant implementation for Apple's MLX framework. The hot path (FWHT, quantization, bit packing) is in Rust via PyO3. The KV cache patcher drops into any mlx-lm model with one line:

```python
import mlx_turbo
model, tokenizer = mlx_turbo.load("Qwen3.5-9B", bits=3)
```

It also ships with an OpenAI-compatible server:
```bash
python -m mlx_turbo.serve --model Qwen3.5-9B --bits 3
```

**What the numbers say:**

Tested on Qwen3.5-9B and Mistral Nemo 12B:
- 4.7x KV cache compression (1,074 MB → 226 MB at 32K context)
- MSE within paper bounds at all bit widths (68-82% of theoretical maximum)
- 30 tok/s at 32K context on a Mac
- Extends usable context from ~47K to ~225K in the same memory

**What I learned:**

The compression works exactly as the paper claims. The output quality is semantically identical to baseline — the model gives correct answers to math, code, reasoning, and creative prompts.

The bottleneck isn't the algorithm. It's the data marshalling. Every token crosses the MLX → numpy → Rust → numpy → MLX boundary per layer. On Mistral (40 KV layers), that's 36% overhead. On Qwen3.5 (8 KV layers), it's 15%. The Rust core itself processes 300K vectors/sec — the bridge is what costs.

Hybrid architectures (linear attention + full attention) benefit most. Qwen3.5 only compresses 8 of 32 layers, leaving 75% of the network untouched. This is probably the future — models where only a fraction of layers need full KV cache.

**What's next:**

Metal kernels via MLX's `mx.fast.metal_kernel()` would eliminate the marshalling entirely. That's the path to <5% overhead for all models.

Repo: https://github.com/hoshinoht/mlx-turbo

---

**Suggested images (attach 2-3):**
1. `charts/01_kv_memory.png` — The money shot: 1,074 MB → 226 MB
2. `charts/06_memory_projection.png` — Context limit extension from 47K to 225K
3. `charts/05_overhead_analysis.png` — Honest overhead analysis

**Suggested hashtags:**
#LLM #AppleSilicon #Rust #MachineLearning #MLX #OpenSource
