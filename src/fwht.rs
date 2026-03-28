//! Fast Walsh-Hadamard Transform + randomized sign flips.
//!
//! After rotation by D @ H / sqrt(d), coordinates of a unit vector follow
//! N(0, 1/d), enabling independent Lloyd-Max scalar quantization.

use std::f64;

/// Generate deterministic sign flips via golden-ratio hashing.
/// Returns Vec of +1.0 / -1.0, length `d`.
pub fn generate_sign_flips(d: usize, seed: u64) -> Vec<f32> {
    let phi: f64 = (1.0 + 5.0_f64.sqrt()) / 2.0;
    (0..d)
        .map(|i| {
            let h = ((i as f64 + seed as f64) * phi * 1e6).floor() as i64 % 2;
            if h == 0 {
                1.0f32
            } else {
                -1.0f32
            }
        })
        .collect()
}

/// In-place FWHT on a single vector. `d` must be power of 2.
#[inline]
pub fn fwht_inplace(x: &mut [f32]) {
    let d = x.len();
    debug_assert!(d > 0 && (d & (d - 1)) == 0);
    let mut h = 1usize;
    while h < d {
        for i in (0..d).step_by(2 * h) {
            for j in i..i + h {
                let a = x[j];
                let b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h <<= 1;
    }
}

/// Batch FWHT over `n` vectors of dimension `d`, stored row-major in `data`.
pub fn fwht_batch(data: &mut [f32], n: usize, d: usize) {
    debug_assert_eq!(data.len(), n * d);
    for i in 0..n {
        fwht_inplace(&mut data[i * d..(i + 1) * d]);
    }
}

/// Forward rotation: y = D @ H @ x / sqrt(d). Operates in-place.
pub fn rotate_batch(data: &mut [f32], n: usize, d: usize, signs: &[f32]) {
    fwht_batch(data, n, d);
    let s = 1.0 / (d as f32).sqrt();
    for i in 0..n {
        let off = i * d;
        for j in 0..d {
            data[off + j] *= s * signs[j];
        }
    }
}

/// Inverse rotation: x = H @ (D @ y) / sqrt(d). Operates in-place.
pub fn inv_rotate_batch(data: &mut [f32], n: usize, d: usize, signs: &[f32]) {
    for i in 0..n {
        let off = i * d;
        for j in 0..d {
            data[off + j] *= signs[j];
        }
    }
    fwht_batch(data, n, d);
    let s = 1.0 / (d as f32).sqrt();
    for i in 0..n {
        let off = i * d;
        for j in 0..d {
            data[off + j] *= s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_fwht() {
        let d = 128;
        let orig: Vec<f32> = (0..d).map(|i| (i as f32) * 0.01).collect();
        let mut buf = orig.clone();
        fwht_inplace(&mut buf);
        fwht_inplace(&mut buf);
        // H @ H = d * I
        for i in 0..d {
            assert!((buf[i] - orig[i] * d as f32).abs() < 1e-3);
        }
    }

    #[test]
    fn roundtrip_rotation() {
        let d = 256;
        let signs = generate_sign_flips(d, 42);
        let orig: Vec<f32> = (0..d).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut buf = orig.clone();
        rotate_batch(&mut buf, 1, d, &signs);
        inv_rotate_batch(&mut buf, 1, d, &signs);
        for i in 0..d {
            assert!(
                (buf[i] - orig[i]).abs() < 1e-4,
                "idx {i}: {} vs {}",
                buf[i],
                orig[i]
            );
        }
    }
}
