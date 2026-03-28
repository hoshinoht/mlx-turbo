//! mlx-turbo-core: Rust backend for TurboQuant KV cache compression.
//!
//! Python calls into this crate via PyO3 for the hot path:
//!   compress(data, n, d) -> (packed, norms)
//!   decompress(packed, norms, n, d) -> data
//!
//! Data crosses the boundary as numpy arrays (zero-copy via PyO3-numpy).
//! The Python side converts between mlx.core.array and numpy as needed.

pub mod codebook;
pub mod engine;
pub mod fwht;
pub mod quantizer;

use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyList;

use std::sync::Mutex;

// ---------------------------------------------------------------------------
// Stateful engine wrapper for Python
// ---------------------------------------------------------------------------

/// Python-visible TurboQuant engine. Holds precomputed codebook + signs.
#[pyclass]
struct TurboEngine {
    inner: Mutex<engine::Engine>,
}

#[pymethods]
impl TurboEngine {
    /// Create a new engine for the given (bits, dim).
    #[new]
    #[pyo3(signature = (bits, dim, seed=42))]
    fn new(bits: u32, dim: usize, seed: u64) -> Self {
        TurboEngine {
            inner: Mutex::new(engine::Engine::new(bits, dim, seed)),
        }
    }

    /// Compress a flat f32 numpy array of shape (n * d,).
    /// Returns (packed_u32, norms_f32) as numpy arrays.
    fn compress<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray1<f32>,
        n: usize,
    ) -> (Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<f32>>) {
        let eng = self.inner.lock().unwrap();
        let slice = data.as_slice().expect("contiguous f32");
        let comp = eng.compress(slice, n);
        (
            Array1::from(comp.packed).into_pyarray(py),
            Array1::from(comp.norms).into_pyarray(py),
        )
    }

    /// Decompress packed data back to flat f32 array of shape (n * d,).
    fn decompress<'py>(
        &self,
        py: Python<'py>,
        packed: PyReadonlyArray1<u32>,
        norms: PyReadonlyArray1<f32>,
        n: usize,
    ) -> Bound<'py, PyArray1<f32>> {
        let eng = self.inner.lock().unwrap();
        let comp = engine::Compressed {
            packed: packed.as_slice().expect("contiguous u32").to_vec(),
            norms: norms.as_slice().expect("contiguous f32").to_vec(),
            n,
            d: eng.dim,
        };
        let out = eng.decompress(&comp);
        Array1::from(out).into_pyarray(py)
    }

    /// Bytes per compressed vector.
    fn bytes_per_vector(&self) -> usize {
        self.inner.lock().unwrap().bytes_per_vector()
    }

    /// Effective bits per value.
    fn bits_per_value(&self) -> f32 {
        self.inner.lock().unwrap().bits_per_value()
    }

    /// Compression ratio vs FP16.
    fn compression_ratio(&self) -> f32 {
        self.inner.lock().unwrap().compression_ratio()
    }

    /// Get centroids as numpy array.
    fn centroids<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let eng = self.inner.lock().unwrap();
        Array1::from(eng.centroids.clone()).into_pyarray(py)
    }

    /// Get interior boundaries as numpy array.
    fn boundaries<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let eng = self.inner.lock().unwrap();
        Array1::from(eng.boundaries.clone()).into_pyarray(py)
    }

    /// Get sign flips as numpy array.
    fn sign_flips<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let eng = self.inner.lock().unwrap();
        Array1::from(eng.signs.clone()).into_pyarray(py)
    }
}

// ---------------------------------------------------------------------------
// Standalone functions for fine-grained control
// ---------------------------------------------------------------------------

/// Build Lloyd-Max codebook for given (bits, dim).
/// Returns (centroids, interior_boundaries) as numpy arrays.
#[pyfunction]
fn build_codebook<'py>(
    py: Python<'py>,
    bits: u32,
    dim: usize,
) -> (Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>) {
    let (c, b) = codebook::build_codebook(bits, dim);
    (
        Array1::from(c).into_pyarray(py),
        Array1::from(b).into_pyarray(py),
    )
}

/// Generate sign flips as numpy array.
#[pyfunction]
#[pyo3(signature = (d, seed=42))]
fn generate_sign_flips<'py>(py: Python<'py>, d: usize, seed: u64) -> Bound<'py, PyArray1<f32>> {
    Array1::from(fwht::generate_sign_flips(d, seed)).into_pyarray(py)
}

/// In-place FWHT on a flat f32 array.
#[pyfunction]
fn fwht_batch<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<f32>,
    n: usize,
    d: usize,
) -> Bound<'py, PyArray1<f32>> {
    let mut buf = data.as_slice().expect("contiguous f32").to_vec();
    fwht::fwht_batch(&mut buf, n, d);
    Array1::from(buf).into_pyarray(py)
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

/// The native Rust module exposed as `mlx_turbo._core`.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TurboEngine>()?;
    m.add_function(wrap_pyfunction!(build_codebook, m)?)?;
    m.add_function(wrap_pyfunction!(generate_sign_flips, m)?)?;
    m.add_function(wrap_pyfunction!(fwht_batch, m)?)?;
    Ok(())
}
