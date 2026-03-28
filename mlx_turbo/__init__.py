"""
mlx-turbo: TurboQuant KV cache compression for MLX on Apple Silicon.

Core algorithm (FWHT, Lloyd-Max, quantize, pack/unpack) runs in Rust.
This Python layer provides MLX integration (KV cache patching, array conversion).

Papers:
  - TurboQuant: https://arxiv.org/abs/2504.19874
  - PolarQuant: https://arxiv.org/abs/2502.02617
  - QJL: https://arxiv.org/abs/2406.03482
"""

from mlx_turbo._core import TurboEngine, build_codebook, generate_sign_flips
from mlx_turbo.patch import patch, load

__version__ = "0.1.0"
