//! Stage 1: MSE-optimal scalar quantization + bit packing.
//!
//! Per vector: extract norm → normalise → rotate → quantize → pack.
//! This is the hot path. Every token passes through here.

use crate::fwht;

/// Quantize a flat f32 buffer of `n` vectors of dimension `d`.
///
/// Returns `(packed_u32, norms_f32)`.
///   - `packed`: length `n * n_words` where `n_words = ceil(d / vals_per_word)`.
///   - `norms`: length `n`.
pub fn quantize_batch(
    data: &[f32],
    n: usize,
    d: usize,
    bits: u32,
    boundaries: &[f32],
    signs: &[f32],
) -> (Vec<u32>, Vec<f32>) {
    let vpw = (32 / bits) as usize; // values per u32 word
    let n_words = (d + vpw - 1) / vpw;

    let mut packed = Vec::with_capacity(n * n_words);
    let mut norms = Vec::with_capacity(n);
    let mut scratch = vec![0.0f32; d];

    for v in 0..n {
        let off = v * d;
        let vec = &data[off..off + d];

        // --- norm + normalise ---
        let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        let inv = if norm > 1e-8 { 1.0 / norm } else { 0.0 };
        norms.push(norm);

        for j in 0..d {
            scratch[j] = vec[j] * inv;
        }

        // --- rotate in-place ---
        fwht::fwht_inplace(&mut scratch);
        let s = 1.0 / (d as f32).sqrt();
        for j in 0..d {
            scratch[j] *= s * signs[j];
        }

        // --- quantize + pack ---
        // Pack `vpw` values per u32 word (vpw = 32 / bits, truncated).
        // For 3-bit: 10 values per word. For 2-bit: 16. For 4-bit: 8.
        let mut wc = 0usize;
        let mut j = 0usize;

        while j < d {
            let mut word: u32 = 0;
            for slot in 0..vpw {
                if j >= d {
                    break;
                }
                let val = scratch[j];
                let mut bucket = 0u32;
                for &b in boundaries.iter() {
                    if val > b {
                        bucket += 1;
                    } else {
                        break;
                    }
                }
                word |= bucket << (slot as u32 * bits);
                j += 1;
            }
            packed.push(word);
            wc += 1;
        }
        while wc < n_words {
            packed.push(0);
            wc += 1;
        }
    }

    (packed, norms)
}

/// Dequantize packed u32 indices back to f32 vectors.
///
/// Returns flat f32 buffer of length `n * d`.
pub fn dequantize_batch(
    packed: &[u32],
    norms: &[f32],
    n: usize,
    d: usize,
    bits: u32,
    centroids: &[f32],
    signs: &[f32],
) -> Vec<f32> {
    let vpw = (32 / bits) as usize;
    let n_words = (d + vpw - 1) / vpw;
    let mask = (1u32 << bits) - 1;
    let max_idx = centroids.len() - 1;

    let mut out = vec![0.0f32; n * d];

    for v in 0..n {
        let po = v * n_words;
        let oo = v * d;

        // --- unpack + centroid lookup ---
        let mut coord = 0;
        for w in 0..n_words {
            let word = packed[po + w];
            let mut sh = 0u32;
            for _ in 0..vpw {
                if coord >= d {
                    break;
                }
                let idx = ((word >> sh) & mask) as usize;
                out[oo + coord] = centroids[idx.min(max_idx)];
                coord += 1;
                sh += bits;
            }
        }

        // --- inverse rotation ---
        let sl = &mut out[oo..oo + d];
        for j in 0..d {
            sl[j] *= signs[j];
        }
        fwht::fwht_inplace(sl);
        let s = 1.0 / (d as f32).sqrt();
        let norm = norms[v];
        for j in 0..d {
            sl[j] *= s * norm;
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebook::build_codebook;
    use crate::fwht::generate_sign_flips;

    #[test]
    fn roundtrip_quantize() {
        let d = 128;
        let bits = 3u32;
        let signs = generate_sign_flips(d, 42);
        let (centroids, boundaries) = build_codebook(bits, d);

        // Unit-ish vector
        let mut data: Vec<f32> = (0..d).map(|i| (i as f32 * 0.37).sin()).collect();
        let norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in data.iter_mut() {
            *x /= norm;
        }
        // Now it's unit length, scale back up
        for x in data.iter_mut() {
            *x *= norm;
        }

        let (packed, norms) = quantize_batch(&data, 1, d, bits, &boundaries, &signs);
        let recon = dequantize_batch(&packed, &norms, 1, d, bits, &centroids, &signs);

        // MSE should be within paper bounds (~0.043 for 3-bit, d=128)
        let mse: f32 = data
            .iter()
            .zip(recon.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / d as f32;
        assert!(
            mse < 0.1,
            "MSE {mse} too high for 3-bit d=128 (bound ~0.043)"
        );
    }
}
