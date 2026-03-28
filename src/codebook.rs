//! Lloyd-Max optimal scalar quantizer for the post-rotation Gaussian distribution.
//!
//! After rotation, each coordinate ≈ N(0, σ²) where σ = 1/√d.
//! Lloyd-Max finds centroids that minimise E[(X − Q(X))²].
//!
//! Codebooks are small (≤16 centroids) and computed once at init,
//! so this doesn't need to be blazing fast — correctness matters more.

use std::f64::consts::PI;

/// Gaussian PDF: (1 / (σ√(2π))) * exp(-x²/(2σ²))
fn gauss_pdf(x: f64, sigma: f64) -> f64 {
    let s2 = sigma * sigma;
    (1.0 / (sigma * (2.0 * PI).sqrt())) * (-x * x / (2.0 * s2)).exp()
}

/// Gaussian CDF via the error function approximation.
fn gauss_cdf(x: f64, sigma: f64) -> f64 {
    0.5 * (1.0 + erf(x / (sigma * std::f64::consts::SQRT_2)))
}

/// Error function approximation (Abramowitz & Stegun 7.1.26, max error 1.5e-7).
fn erf(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    sign * (1.0 - poly * (-x * x).exp())
}

/// Numerically integrate f(x) over [a, b] using Simpson's rule.
fn integrate<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> f64 {
    let n = if n % 2 == 0 { n } else { n + 1 };
    let h = (b - a) / n as f64;
    let mut sum = f(a) + f(b);
    for i in 1..n {
        let x = a + i as f64 * h;
        sum += if i % 2 == 0 { 2.0 } else { 4.0 } * f(x);
    }
    sum * h / 3.0
}

/// Solve Lloyd-Max for N(0, σ²) with 2^bits levels.
/// Returns (centroids, boundaries) where boundaries includes ±∞ sentinels.
pub fn solve_lloyd_max(bits: u32, sigma: f64, max_iter: usize, tol: f64) -> (Vec<f64>, Vec<f64>) {
    let n = 1u32 << bits;
    let nl = n as usize;

    // Initialise centroids uniformly in [-3σ, 3σ]
    let mut c: Vec<f64> = (0..nl)
        .map(|i| -3.0 * sigma + 6.0 * sigma * (i as f64) / (nl as f64 - 1.0))
        .collect();

    let mut bounds = vec![0.0f64; nl + 1];

    for _ in 0..max_iter {
        // Boundaries = midpoints
        bounds[0] = -8.0 * sigma; // proxy for -∞
        bounds[nl] = 8.0 * sigma; // proxy for +∞
        for i in 0..nl - 1 {
            bounds[i + 1] = (c[i] + c[i + 1]) / 2.0;
        }

        // New centroids = E[X | bounds[i] < X < bounds[i+1]]
        let mut new_c = vec![0.0f64; nl];
        let mut max_shift: f64 = 0.0;
        for i in 0..nl {
            let lo = bounds[i];
            let hi = bounds[i + 1];
            let num = integrate(|x| x * gauss_pdf(x, sigma), lo, hi, 200);
            let den = gauss_cdf(hi, sigma) - gauss_cdf(lo, sigma);
            new_c[i] = if den.abs() > 1e-15 { num / den } else { c[i] };
            max_shift = max_shift.max((new_c[i] - c[i]).abs());
        }
        c = new_c;
        if max_shift < tol {
            break;
        }
    }

    // Final boundaries with real ±∞
    bounds[0] = f64::NEG_INFINITY;
    bounds[nl] = f64::INFINITY;
    for i in 0..nl - 1 {
        bounds[i + 1] = (c[i] + c[i + 1]) / 2.0;
    }

    (c, bounds)
}

/// Convenience: solve and return (centroids_f32, interior_boundaries_f32).
/// Interior boundaries exclude ±∞, shape (2^bits - 1,).
pub fn build_codebook(bits: u32, dim: usize) -> (Vec<f32>, Vec<f32>) {
    let sigma = 1.0 / (dim as f64).sqrt();
    let (centroids, bounds) = solve_lloyd_max(bits, sigma, 200, 1e-12);
    let c32: Vec<f32> = centroids.iter().map(|&x| x as f32).collect();
    // Interior boundaries: skip first (-∞) and last (+∞)
    let b32: Vec<f32> = bounds[1..bounds.len() - 1]
        .iter()
        .map(|&x| x as f32)
        .collect();
    (c32, b32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codebook_3bit_128d() {
        let (c, b) = build_codebook(3, 128);
        assert_eq!(c.len(), 8);
        assert_eq!(b.len(), 7);
        // Centroids should be sorted
        for i in 0..c.len() - 1 {
            assert!(c[i] < c[i + 1], "centroids not sorted at {i}");
        }
        // Should be symmetric
        for i in 0..4 {
            assert!(
                (c[i] + c[7 - i]).abs() < 1e-5,
                "not symmetric at {i}: {} vs {}",
                c[i],
                c[7 - i]
            );
        }
    }

    #[test]
    fn codebook_dimensions() {
        for bits in [2, 3, 4] {
            for dim in [64, 128, 256] {
                let (c, b) = build_codebook(bits, dim);
                assert_eq!(c.len(), 1 << bits);
                assert_eq!(b.len(), (1 << bits) - 1);
            }
        }
    }
}
