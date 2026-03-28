"""
Model patching: swap KVCache layers with TurboQuant-compressed caches.

Usage:
    import mlx_turbo
    model, tokenizer = mlx_turbo.load("mlx-community/Qwen3.5-9B-OptiQ-4bit", bits=3)

    # Or patch an already-loaded model:
    from mlx_lm import load
    model, tokenizer = load("mlx-community/Qwen3.5-9B-OptiQ-4bit")
    mlx_turbo.patch(model, bits=3)
"""

from typing import Optional

import mlx.nn as nn


def _detect_head_dim(model: nn.Module) -> int:
    """Extract head_dim from model config, handling nested text_config dicts."""
    config = model.args if hasattr(model, "args") else getattr(model, "config", None)
    if config is None:
        return 128

    def _get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    tc = _get(config, "text_config", None)
    source = tc if tc is not None else config

    hidden = _get(source, "hidden_size", 4096)
    n_heads = _get(source, "num_attention_heads", 32)
    head_dim = _get(source, "head_dim", hidden // n_heads)
    return head_dim


def patch(model: nn.Module, bits: int = 3, seed: int = 42) -> nn.Module:
    """
    Monkey-patch a model so make_prompt_cache returns TurboQuant caches.

    This patches the model's `make_cache` method. When mlx-lm's
    `make_prompt_cache(model)` is called, it checks for `model.make_cache()`
    first -- so our patched version is used automatically.

    Args:
        model: An mlx-lm model (already loaded).
        bits: Quantization bits (2, 3, or 4).
        seed: Random seed for rotation.

    Returns:
        The same model, patched in-place.
    """
    from mlx_lm.models.cache import KVCache

    head_dim = _detect_head_dim(model)
    # Ensure power of 2
    if head_dim & (head_dim - 1) != 0:
        head_dim = 1 << (head_dim - 1).bit_length()

    # Save original make_cache if it exists
    original_make_cache = getattr(model, "make_cache", None)

    def _turbo_make_cache():
        from mlx_turbo.kv_cache import TurboQuantKVCache

        if original_make_cache is not None:
            # Hybrid model (e.g., Qwen3.5) -- get default cache layout,
            # replace only KVCache entries
            default = original_make_cache()
        else:
            # Standard model -- create KVCache for each layer
            n_layers = len(model.layers)
            default = [KVCache() for _ in range(n_layers)]

        caches = []
        tq_count = 0
        for c in default:
            if isinstance(c, KVCache):
                caches.append(
                    TurboQuantKVCache(bits=bits, head_dim=head_dim, seed=seed)
                )
                tq_count += 1
            else:
                caches.append(c)

        return caches

    model.make_cache = _turbo_make_cache
    model._turbo_bits = bits
    model._turbo_head_dim = head_dim
    return model


def load(
    model_path: str,
    bits: int = 3,
    seed: int = 42,
    tokenizer_config: Optional[dict] = None,
    adapter_path: Optional[str] = None,
):
    """
    Load a model with TurboQuant KV cache compression pre-applied.

    Drop-in replacement for mlx_lm.load().

    Args:
        model_path: HuggingFace model name or local path.
        bits: TurboQuant bits (2, 3, or 4).
        seed: Random seed for rotation.
        tokenizer_config: Passed to mlx_lm.load().
        adapter_path: Passed to mlx_lm.load().

    Returns:
        (model, tokenizer) with TurboQuant patched in.
    """
    from mlx_lm import load as mlx_load

    kwargs = {}
    if tokenizer_config is not None:
        kwargs["tokenizer_config"] = tokenizer_config
    if adapter_path is not None:
        kwargs["adapter_path"] = adapter_path

    model, tokenizer = mlx_load(model_path, **kwargs)
    patch(model, bits=bits, seed=seed)

    head_dim = _detect_head_dim(model)
    n_layers = len(model.layers)
    print(f"[mlx-turbo] Patched: {bits}-bit, head_dim={head_dim}, {n_layers} layers")

    return model, tokenizer
