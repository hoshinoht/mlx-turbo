"""
TurboQuant KV cache for mlx-lm — backed by Rust core.

The compress/decompress hot path runs in native Rust (mlx_turbo._core).
This wrapper handles:
  - MLX array <-> numpy conversion at the boundary
  - Pre-allocated decompressed buffer management (mlx-lm KVCache pattern)
  - mlx-lm cache interface (update_and_fetch, state, make_mask, etc.)
"""

from typing import Optional

import numpy as np
import mlx.core as mx

from mlx_turbo._core import TurboEngine


class TurboQuantKVCache:
    """
    Drop-in KVCache replacement using Rust-backed TurboQuant compression.

    Compatible with mlx-lm's cache interface:
      update_and_fetch, state, offset, size, empty, is_trimmable, trim, make_mask
    """

    step: int = 256

    def __init__(
        self,
        bits: int = 3,
        head_dim: int = 128,
        step: int = 256,
        seed: int = 42,
        val_seed: int = 73,
    ):
        # DO NOT name this self.bits — mlx-lm routes to quantized SDPA if it exists
        self._tq_bits = bits
        self.head_dim = head_dim
        self.step = step

        # Rust engines for keys and values
        self._key_engine = TurboEngine(bits, head_dim, seed)
        self._val_engine = TurboEngine(min(bits, 4), head_dim, val_seed)

        # State
        self.offset: int = 0
        self._input_dtype = mx.float16

        # Pre-allocated decompressed buffers (mlx-lm KVCache pattern)
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None

        # Compressed storage for memory accounting
        self._compressed_bytes: int = 0

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Compress new K/V via Rust, write decompressed into pre-allocated buffers."""
        B, n_kv, num_new, hd = keys.shape
        prev = self.offset
        self._input_dtype = keys.dtype

        # --- Compress + decompress via Rust (per KV-head, flattened) ---
        new_keys = self._rust_roundtrip(self._key_engine, keys, B, n_kv, num_new, hd)
        new_values = self._rust_roundtrip(
            self._val_engine, values, B, n_kv, num_new, hd
        )

        # --- Write into pre-allocated MLX buffers ---
        if self.keys is None or (prev + num_new) > self.keys.shape[2]:
            n_steps = (self.step + num_new - 1) // self.step
            new_cap = n_steps * self.step
            new_k = mx.zeros((B, n_kv, new_cap, hd), keys.dtype)
            new_v = mx.zeros((B, n_kv, new_cap, hd), values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.keys[..., prev : prev + num_new, :] = new_keys
        self.values[..., prev : prev + num_new, :] = new_values

        # Track compressed size
        kbpv = self._key_engine.bytes_per_vector()
        vbpv = self._val_engine.bytes_per_vector()
        self._compressed_bytes += num_new * n_kv * (kbpv + vbpv)

        self.offset += num_new
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    def _rust_roundtrip(
        self, engine: TurboEngine, data: mx.array, B: int, n_kv: int, seq: int, hd: int
    ) -> mx.array:
        """
        MLX array -> numpy -> Rust compress+decompress -> numpy -> MLX array.

        Processes each (batch, head) slice as a flat f32 buffer of (seq * hd,).
        """
        # MLX -> numpy (this triggers mx.eval if lazy)
        data_np = np.array(data.astype(mx.float32))  # (B, n_kv, seq, hd)
        out_np = np.empty_like(data_np)

        for b in range(B):
            for h in range(n_kv):
                flat = np.ascontiguousarray(data_np[b, h].reshape(-1))  # (seq * hd,)
                packed, norms = engine.compress(flat, seq)
                recon = engine.decompress(packed, norms, seq)
                out_np[b, h] = recon.reshape(seq, hd)

        return mx.array(out_np).astype(self._input_dtype)

    # ---- mlx-lm interface ----

    @property
    def state(self):
        if self.keys is not None:
            if self.offset == self.keys.shape[2]:
                return self.keys, self.values
            return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
        return mx.zeros((0,)), mx.zeros((0,))

    @state.setter
    def state(self, v):
        keys, values = v
        if keys.size == 0:
            return
        self.offset = 0
        self.keys = None
        self.values = None
        self._compressed_bytes = 0
        self.update_and_fetch(keys, values)

    def size(self) -> int:
        return self.offset

    def empty(self) -> bool:
        return self.keys is None

    @property
    def nbytes(self) -> int:
        return self._compressed_bytes

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int):
        self.offset = max(0, self.offset - n)

    def make_mask(self, *args, **kwargs):
        from mlx_lm.models.cache import create_attention_mask

        if args:
            N = args[0]
            rest = args[1:]
            return create_attention_mask(N, self.offset, *rest, **kwargs)
        return create_attention_mask(offset=self.offset, **kwargs)

    def memory_report(self) -> dict:
        fp16_bytes = self.offset * self.head_dim * 2 * 2
        compressed = self.nbytes
        return {
            "offset": self.offset,
            "fp16_bytes": fp16_bytes,
            "compressed_bytes": compressed,
            "compression_ratio": fp16_bytes / max(compressed, 1),
            "bits_per_value": 8 * compressed / max(self.offset * self.head_dim * 2, 1),
        }
