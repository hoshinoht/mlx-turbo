"""
TurboQuant Metal kernels — loaded from .metal files.

Kernels live in mlx_turbo/kernels/*.metal with proper syntax and
are loaded once at import time.
"""

from pathlib import Path
import mlx.core as mx

_KERNEL_DIR = Path(__file__).parent / "kernels"


def _load(name: str) -> str:
    return (_KERNEL_DIR / f"{name}.metal").read_text()


# ─── Kernel objects (created once, reused across calls) ───────────────────────

_decompress_kernel = mx.fast.metal_kernel(
    name="turboquant_decompress",
    input_names=["packed", "norms", "centroids", "signs", "meta"],
    output_names=["out"],
    source=_load("decompress"),
)

_compress_kernel = mx.fast.metal_kernel(
    name="turboquant_compress",
    input_names=["data", "signs", "boundaries", "params"],
    output_names=["packed", "norms"],
    source=_load("compress"),
    atomic_outputs=True,
)


# ─── Python wrappers ─────────────────────────────────────────────────────────


def metal_decompress(
    packed: mx.array,
    norms: mx.array,
    centroids: mx.array,
    signs: mx.array,
    n_vectors: int,
    dim: int,
    bits: int,
) -> mx.array:
    """Decompress on GPU. Returns flat (n_vectors * dim,) float32."""
    vpw = 32 // bits
    n_words = (dim + vpw - 1) // vpw
    meta = mx.array([bits, vpw, len(centroids)], dtype=mx.uint32)

    outputs = _decompress_kernel(
        inputs=[packed, norms, centroids, signs, meta],
        grid=(n_vectors * dim, 1, 1),
        threadgroup=(dim, 1, 1),
        output_shapes=[(n_vectors * dim,)],
        output_dtypes=[mx.float32],
    )
    return outputs[0]


def metal_compress(
    data: mx.array,
    signs: mx.array,
    boundaries: mx.array,
    n_vectors: int,
    dim: int,
    bits: int,
) -> tuple[mx.array, mx.array]:
    """Compress on GPU. Returns (packed uint32, norms float32)."""
    vpw = 32 // bits
    n_words = (dim + vpw - 1) // vpw
    params = mx.array([len(boundaries), bits, vpw], dtype=mx.uint32)

    outputs = _compress_kernel(
        inputs=[data, signs, boundaries, params],
        grid=(n_vectors * dim, 1, 1),
        threadgroup=(dim, 1, 1),
        output_shapes=[(n_vectors * n_words,), (n_vectors,)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return outputs[0], outputs[1]
