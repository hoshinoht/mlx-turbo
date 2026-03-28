//! TurboQuant engine: wraps codebook + quantizer into a stateful compressor.
//!
//! Holds precomputed codebooks and sign flips, exposes compress/decompress.

use crate::codebook;
use crate::fwht;
use crate::quantizer;

/// Pre-initialised TurboQuant engine for a fixed (bits, dim) configuration.
pub struct Engine {
    pub bits: u32,
    pub dim: usize,
    pub centroids: Vec<f32>,
    pub boundaries: Vec<f32>,
    pub signs: Vec<f32>,
}

/// Compressed representation of a batch of vectors.
pub struct Compressed {
    pub packed: Vec<u32>, // bit-packed indices, length n * n_words
    pub norms: Vec<f32>,  // vector norms, length n
    pub n: usize,
    pub d: usize,
}

impl Engine {
    /// Create a new engine for the given bits and dimension.
    /// Precomputes Lloyd-Max codebook and sign flips.
    pub fn new(bits: u32, dim: usize, seed: u64) -> Self {
        assert!(matches!(bits, 2 | 3 | 4), "bits must be 2, 3, or 4");
        assert!(dim > 0 && (dim & (dim - 1)) == 0, "dim must be power of 2");

        let (centroids, boundaries) = codebook::build_codebook(bits, dim);
        let signs = fwht::generate_sign_flips(dim, seed);

        Engine {
            bits,
            dim,
            centroids,
            boundaries,
            signs,
        }
    }

    /// Compress a flat f32 buffer of `n` vectors of dimension `dim`.
    pub fn compress(&self, data: &[f32], n: usize) -> Compressed {
        assert_eq!(data.len(), n * self.dim);
        let (packed, norms) =
            quantizer::quantize_batch(data, n, self.dim, self.bits, &self.boundaries, &self.signs);
        Compressed {
            packed,
            norms,
            n,
            d: self.dim,
        }
    }

    /// Decompress back to flat f32 buffer of length `n * dim`.
    pub fn decompress(&self, comp: &Compressed) -> Vec<f32> {
        quantizer::dequantize_batch(
            &comp.packed,
            &comp.norms,
            comp.n,
            comp.d,
            self.bits,
            &self.centroids,
            &self.signs,
        )
    }

    /// Number of u32 words per vector in packed representation.
    pub fn words_per_vector(&self) -> usize {
        let vpw = (32 / self.bits) as usize;
        (self.dim + vpw - 1) / vpw
    }

    /// Bytes per compressed vector (packed indices + f32 norm).
    pub fn bytes_per_vector(&self) -> usize {
        self.words_per_vector() * 4 + 4 // u32 words + f32 norm
    }

    /// Effective bits per value including norm overhead.
    pub fn bits_per_value(&self) -> f32 {
        (self.bytes_per_vector() * 8) as f32 / self.dim as f32
    }

    /// Compression ratio vs float16 (2 bytes per value).
    pub fn compression_ratio(&self) -> f32 {
        (self.dim as f32 * 2.0) / self.bytes_per_vector() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_compress_decompress() {
        let e = Engine::new(3, 128, 42);
        let data: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();

        let comp = e.compress(&data, 1);
        let recon = e.decompress(&comp);

        assert_eq!(recon.len(), 128);

        let mse: f32 = data
            .iter()
            .zip(recon.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / 128.0;

        assert!(mse < 0.1, "MSE {mse} too high");
    }

    #[test]
    fn engine_batch() {
        let e = Engine::new(3, 256, 42);
        let n = 64;
        let data: Vec<f32> = (0..n * 256).map(|i| (i as f32 * 0.01).sin()).collect();

        let comp = e.compress(&data, n);
        assert_eq!(comp.norms.len(), n);
        assert_eq!(comp.packed.len(), n * e.words_per_vector());

        let recon = e.decompress(&comp);
        assert_eq!(recon.len(), n * 256);
    }

    #[test]
    fn compression_stats() {
        for (bits, dim) in [(2, 128), (3, 128), (4, 128), (3, 256)] {
            let e = Engine::new(bits, dim, 42);
            let bpv = e.bits_per_value();
            let cr = e.compression_ratio();
            println!(
                "bits={bits} dim={dim}: {bpv:.2} bits/val, {cr:.1}x compression, {} bytes/vec",
                e.bytes_per_vector()
            );
            assert!(cr > 1.0, "compression ratio should be > 1");
        }
    }
}
