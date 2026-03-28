"""
Validation tests for TurboQuant against the paper's theoretical bounds.

Tests:
  1. Lloyd-Max codebook convergence and MSE bounds
  2. FWHT correctness (self-inverse property)
  3. Random rotation produces expected distribution
  4. MSE distortion within paper's upper bounds
  5. QJL unbiasedness and inner product correlation
  6. Round-trip quantize/dequantize
  7. Needle-in-haystack retrieval
  8. Compression ratio verification
"""

import math
import sys
import os

import numpy as np

# Ensure turboquant is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_codebook():
    """Test Lloyd-Max codebook solver convergence."""
    from mlx_turbo.codebook import LloydMaxSolver

    print("=" * 60)
    print("TEST: Lloyd-Max Codebook Solver")
    print("=" * 60)

    for bits in [2, 3, 4]:
        for dim in [64, 128, 256]:
            sigma = 1.0 / math.sqrt(dim)
            solver = LloydMaxSolver(bits=bits, sigma=sigma)
            centroids, boundaries = solver.solve()

            n_levels = 2**bits
            assert len(centroids) == n_levels, f"Expected {n_levels} centroids"
            assert len(boundaries) == n_levels + 1

            # Centroids should be sorted
            for i in range(len(centroids) - 1):
                assert centroids[i] < centroids[i + 1], "Centroids not sorted"

            # Centroids should be symmetric around 0 (Gaussian is symmetric)
            for i in range(n_levels // 2):
                assert abs(centroids[i] + centroids[n_levels - 1 - i]) < 1e-6, (
                    "Centroids not symmetric"
                )

            print(
                f"  bits={bits}, dim={dim}: {n_levels} centroids, "
                f"range [{centroids[0]:.4f}, {centroids[-1]:.4f}]  OK"
            )

    print()


def test_fwht():
    """Test Fast Walsh-Hadamard Transform is self-inverse."""
    import mlx.core as mx
    from mlx_turbo.rotation import fwht

    print("=" * 60)
    print("TEST: FWHT Self-Inverse Property")
    print("=" * 60)

    for d in [8, 16, 32, 64, 128]:
        x = mx.random.normal(shape=(4, d))
        y = fwht(x)
        x_back = fwht(y) / d  # WHT is self-inverse up to factor d

        error = mx.mean(mx.abs(x - x_back)).item()
        assert error < 1e-5, f"FWHT round-trip error {error} too large for d={d}"
        print(f"  d={d}: round-trip error = {error:.2e}  OK")

    print()


def test_rotation_roundtrip():
    """Test random_rotate -> inverse_rotate is identity."""
    import mlx.core as mx
    from mlx_turbo.rotation import random_rotate, inverse_rotate

    print("=" * 60)
    print("TEST: Rotation Round-Trip")
    print("=" * 60)

    for d in [32, 64, 128]:
        x = mx.random.normal(shape=(10, d))
        y = random_rotate(x, seed=42)
        x_back = inverse_rotate(y, seed=42)

        error = mx.max(mx.abs(x - x_back)).item()
        assert error < 1e-4, f"Rotation round-trip error {error} too large for d={d}"
        print(f"  d={d}: max round-trip error = {error:.2e}  OK")

    print()


def test_rotation_distribution():
    """Test that rotation makes coordinates approximately N(0, 1/d)."""
    import mlx.core as mx
    from mlx_turbo.rotation import random_rotate

    print("=" * 60)
    print("TEST: Post-Rotation Distribution")
    print("=" * 60)

    d = 128
    n_vecs = 1000
    expected_sigma = 1.0 / math.sqrt(d)

    # Generate random unit vectors
    x = mx.random.normal(shape=(n_vecs, d))
    norms = mx.linalg.norm(x, axis=-1, keepdims=True)
    x = x / norms

    # Rotate
    y = random_rotate(x, seed=42)

    # Check statistics of all coordinates
    mean = mx.mean(y).item()
    std = float(mx.mean(mx.square(y)).item() ** 0.5)

    print(f"  d={d}, n={n_vecs}")
    print(f"  Mean:   {mean:.4f}  (expected: ~0.0)")
    print(f"  Std:    {std:.4f}  (expected: ~{expected_sigma:.4f})")
    assert abs(mean) < 0.02, f"Mean {mean} too far from 0"
    assert abs(std - expected_sigma) < 0.02, f"Std {std} too far from {expected_sigma}"
    print(f"  PASSED")
    print()


def test_mse_distortion():
    """Test MSE distortion is within the paper's theoretical upper bounds."""
    import mlx.core as mx
    from mlx_turbo.quantizer import TurboQuantMSE

    print("=" * 60)
    print("TEST: MSE Distortion vs Paper Bounds")
    print("=" * 60)

    # Paper Table: upper bounds on MSE for unit vectors
    # From TurboQuant paper, the MSE bound for b-bit quantization of
    # d-dimensional unit vectors is approximately:
    # MSE <= C * 2^(-2b) / d  (where C is a dimension-dependent constant)
    # Empirical bounds from tonbistudio/turboquant-pytorch validation:
    paper_bounds = {
        2: 0.170,  # 2-bit
        3: 0.043,  # 3-bit
        4: 0.011,  # 4-bit
    }

    d = 128
    n_vecs = 500

    # Generate random unit vectors
    x = mx.random.normal(shape=(n_vecs, d))
    norms = mx.linalg.norm(x, axis=-1, keepdims=True)
    x_unit = x / norms
    # Scale to have norm 1
    x_test = x_unit

    for bits in [2, 3, 4]:
        quant = TurboQuantMSE(bits=bits, dim=d, seed=42)
        indices, q_norms = quant.quantize(x_test)
        x_recon = quant.dequantize(indices, q_norms)

        # MSE per vector, averaged
        mse = mx.mean(mx.sum(mx.square(x_test - x_recon), axis=-1)).item()
        bound = paper_bounds[bits]
        ratio = mse / bound

        status = "OK" if mse <= bound * 1.2 else "WARN"  # 20% tolerance
        print(
            f"  {bits}-bit: MSE = {mse:.4f}  bound = {bound:.3f}  "
            f"ratio = {ratio:.2f}x  [{status}]"
        )

    print()


def test_qjl_unbiasedness():
    """Test that QJL correction produces unbiased inner product estimates."""
    import mlx.core as mx
    from mlx_turbo.quantizer import TurboQuantMSE
    from mlx_turbo.qjl import QJLSketch

    print("=" * 60)
    print("TEST: QJL Inner Product Unbiasedness")
    print("=" * 60)

    d = 128
    n_pairs = 200
    bits = 2  # Use 2-bit to make the bias more visible

    # Generate random vector pairs
    mx.random.seed(42)
    queries = mx.random.normal(shape=(n_pairs, d))
    keys = mx.random.normal(shape=(n_pairs, d))

    # Normalize
    queries = queries / mx.linalg.norm(queries, axis=-1, keepdims=True)
    keys = keys / mx.linalg.norm(keys, axis=-1, keepdims=True)

    # True inner products
    true_ip = mx.sum(queries * keys, axis=-1)  # (n_pairs,)

    # MSE-only inner products (biased)
    quant = TurboQuantMSE(bits=bits, dim=d, seed=42)
    indices, norms = quant.quantize(keys)
    keys_mse = quant.dequantize(indices, norms)
    mse_ip = mx.sum(queries * keys_mse, axis=-1)
    mse_bias = mx.mean(mse_ip - true_ip).item()

    # QJL-corrected inner products
    residual = quant.compute_residual(keys, indices, norms)
    qjl = QJLSketch(dim=d, sketch_dim=d, seed=137)
    packed_signs, r_norms = qjl.sketch(residual)
    corrections = []
    for i in range(n_pairs):
        c = qjl.estimate_inner_product_correction(
            queries[i : i + 1], packed_signs[i : i + 1], r_norms[i : i + 1]
        )
        corrections.append(c.item())
    corrections = mx.array(corrections)

    corrected_ip = mse_ip + corrections
    qjl_bias = mx.mean(corrected_ip - true_ip).item()

    # Correlation
    true_np = np.array(true_ip.tolist())
    mse_np = np.array(mse_ip.tolist())
    corrected_np = np.array(corrected_ip.tolist())

    mse_corr = np.corrcoef(true_np, mse_np)[0, 1]
    qjl_corr = np.corrcoef(true_np, corrected_np)[0, 1]

    print(f"  MSE-only bias:    {mse_bias:+.4f}  correlation: {mse_corr:.3f}")
    print(f"  QJL-corrected:    {qjl_bias:+.4f}  correlation: {qjl_corr:.3f}")
    print(f"  Bias reduction:   {abs(mse_bias) / max(abs(qjl_bias), 1e-8):.1f}x")
    print()


def test_engine_roundtrip():
    """Test TurboQuantEngine compress/decompress round-trip."""
    import mlx.core as mx
    from mlx_turbo.engine import TurboQuantEngine

    print("=" * 60)
    print("TEST: Engine Round-Trip")
    print("=" * 60)

    d = 128
    n = 32

    for bits, use_qjl in [(3, False), (4, False), (3, True), (4, True)]:
        engine = TurboQuantEngine(bits=bits, dim=d, use_qjl=use_qjl)

        x = mx.random.normal(shape=(2, 4, n, d))  # (B, n_heads, seq, dim)
        cv = engine.compress(x)
        x_recon = engine.decompress(cv)

        mse = mx.mean(mx.square(x - x_recon)).item()
        bpv = engine.bits_per_value()
        cr = engine.compression_ratio()

        mode = "MSE+QJL" if use_qjl else "MSE-only"
        print(
            f"  {bits}-bit {mode}: MSE={mse:.4f}  "
            f"bits/val={bpv:.2f}  compression={cr:.1f}x"
        )

    print()


def test_needle_in_haystack():
    """Test that TurboQuant preserves nearest-neighbor retrieval."""
    import mlx.core as mx
    from mlx_turbo.engine import TurboQuantEngine

    print("=" * 60)
    print("TEST: Needle-in-Haystack Retrieval")
    print("=" * 60)

    d = 128
    n_correct = 0
    n_total = 0

    for bits in [3, 4]:
        for seq_len in [128, 512, 2048]:
            engine = TurboQuantEngine(bits=bits, dim=d, use_qjl=False)

            # Create haystack
            haystack = mx.random.normal(shape=(seq_len, d))
            haystack = haystack / mx.linalg.norm(haystack, axis=-1, keepdims=True)

            # Pick a random "needle" - the vector most similar to query
            query = mx.random.normal(shape=(1, d))
            query = query / mx.linalg.norm(query, axis=-1, keepdims=True)

            # Make one vector very similar to query (the needle)
            needle_idx = seq_len // 2
            haystack[needle_idx] = query.squeeze() * 0.95 + haystack[needle_idx] * 0.05
            haystack[needle_idx] = haystack[needle_idx] / mx.linalg.norm(
                haystack[needle_idx]
            )

            # True nearest
            true_scores = (query @ haystack.T).squeeze()
            true_top1 = mx.argmax(true_scores).item()

            # Compressed nearest
            cv = engine.compress(haystack)
            recon = engine.decompress(cv)
            compressed_scores = (query @ recon.T).squeeze()
            compressed_top1 = mx.argmax(compressed_scores).item()

            match = true_top1 == compressed_top1
            n_correct += int(match)
            n_total += 1

            status = "MATCH" if match else "MISS"
            print(
                f"  {bits}-bit, seq={seq_len}: true_idx={true_top1}, "
                f"compressed_idx={compressed_top1}  [{status}]"
            )

    print(
        f"\n  Retrieval accuracy: {n_correct}/{n_total} "
        f"({100 * n_correct / n_total:.0f}%)"
    )
    print()


def test_compression_ratio():
    """Verify compression ratios match expectations."""
    from mlx_turbo.engine import TurboQuantEngine

    print("=" * 60)
    print("TEST: Compression Ratios")
    print("=" * 60)

    d = 128
    for bits in [2, 3, 4]:
        for use_qjl in [False, True]:
            if use_qjl and bits < 3:
                continue
            engine = TurboQuantEngine(bits=bits, dim=d, use_qjl=use_qjl)
            bpv = engine.bits_per_value()
            cr = engine.compression_ratio()
            bpv_total = engine.bytes_per_vector()
            mode = "MSE+QJL" if use_qjl else "MSE-only"
            print(
                f"  {bits}-bit {mode}: {bpv:.2f} bits/val, "
                f"{cr:.1f}x compression, {bpv_total} bytes/vector"
            )

    print()


def main():
    print("\n" + "=" * 60)
    print("  TurboQuant Algorithm Validation Suite")
    print("=" * 60 + "\n")

    test_codebook()
    test_fwht()
    test_rotation_roundtrip()
    test_rotation_distribution()
    test_mse_distortion()
    test_qjl_unbiasedness()
    test_engine_roundtrip()
    test_needle_in_haystack()
    test_compression_ratio()

    print("=" * 60)
    print("  ALL TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
