"""
TurboQuant KV cache for mlx-lm.

Uses Metal GPU kernels for compress/decompress — no CPU bridge crossing.
Data stays as mx.array the entire time.
Rust backend used only for one-time codebook precomputation at init.
"""

from typing import Optional

import numpy as np
import mlx.core as mx

from mlx_turbo._core import TurboEngine, build_codebook, generate_sign_flips
from mlx_turbo.metal_ops import metal_compress, metal_decompress


class TurboQuantKVCache:
    """
    Drop-in KVCache replacement with Metal-accelerated TurboQuant compression.
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
        self._tq_bits = bits
        self.head_dim = head_dim
        self.step = step

        # Precompute codebooks via Rust (once), store as persistent MLX arrays
        k_c, k_b = build_codebook(bits, head_dim)
        v_c, v_b = build_codebook(min(bits, 4), head_dim)
        k_signs = np.array(generate_sign_flips(head_dim, seed))
        v_signs = np.array(generate_sign_flips(head_dim, val_seed))

        self._k_centroids = mx.array(k_c)
        self._k_boundaries = mx.array(k_b)
        self._k_signs = mx.array(k_signs)
        self._v_centroids = mx.array(v_c)
        self._v_boundaries = mx.array(v_b)
        self._v_signs = mx.array(v_signs)
        self._v_bits = min(bits, 4)

        # For compressed byte tracking
        k_vpw = 32 // bits
        v_vpw = 32 // self._v_bits
        self._k_bpv = ((head_dim + k_vpw - 1) // k_vpw) * 4 + 4
        self._v_bpv = ((head_dim + v_vpw - 1) // v_vpw) * 4 + 4

        # State
        self.offset: int = 0
        self._input_dtype = mx.float16

        # Pre-allocated decompressed buffers
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None

        self._compressed_bytes: int = 0

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Compress/decompress via Metal kernels. No CPU bridge."""
        B, n_kv, num_new, hd = keys.shape
        prev = self.offset
        self._input_dtype = keys.dtype

        # Flatten to (n_vecs * hd,) — stays as mx.array on GPU
        n_vecs = B * n_kv * num_new
        k_flat = keys.astype(mx.float32).reshape(-1)
        v_flat = values.astype(mx.float32).reshape(-1)

        # Metal compress + decompress (all GPU, no bridge)
        k_packed, k_norms = metal_compress(
            k_flat,
            self._k_signs,
            self._k_boundaries,
            n_vecs,
            hd,
            self._tq_bits,
        )
        new_keys = (
            metal_decompress(
                k_packed,
                k_norms,
                self._k_centroids,
                self._k_signs,
                n_vecs,
                hd,
                self._tq_bits,
            )
            .reshape(B, n_kv, num_new, hd)
            .astype(self._input_dtype)
        )

        v_packed, v_norms = metal_compress(
            v_flat,
            self._v_signs,
            self._v_boundaries,
            n_vecs,
            hd,
            self._v_bits,
        )
        new_values = (
            metal_decompress(
                v_packed,
                v_norms,
                self._v_centroids,
                self._v_signs,
                n_vecs,
                hd,
                self._v_bits,
            )
            .reshape(B, n_kv, num_new, hd)
            .astype(self._input_dtype)
        )

        # Write into pre-allocated MLX buffers
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

        self._compressed_bytes += num_new * n_kv * (self._k_bpv + self._v_bpv)
        self.offset += num_new
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

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
            return create_attention_mask(args[0], self.offset, *args[1:], **kwargs)
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
