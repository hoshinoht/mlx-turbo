// TurboQuant decompress: (packed uint32, norms) -> float vectors
//
// Fused pipeline per vector:
//   1. Unpack bit index from uint32
//   2. Centroid lookup + sign flip
//   3. Inverse FWHT butterfly (shared memory)
//   4. Scale by 1/sqrt(D) * norm
//
// Grid:  (n_vectors * dim, 1, 1)
// Group: (dim, 1, 1)
//
// Inputs:
//   packed    — (n_vectors * n_words,) uint32
//   norms     — (n_vectors,) float32
//   centroids — (n_levels,) float32
//   signs     — (dim,) float32
//   meta      — [bits, vals_per_word, n_centroids] uint32
//
// Outputs:
//   out — (n_vectors * dim,) float32

    uint gid = threadgroup_position_in_grid.x;   // vector index
    uint tid = thread_position_in_threadgroup.x;  // coordinate index
    uint D   = threads_per_threadgroup.x;

    // --- 1. Unpack ---
    uint bits        = meta[0];
    uint vpw         = meta[1];
    uint n_centroids = meta[2];
    uint n_words     = (D + vpw - 1) / vpw;
    uint word_idx    = tid / vpw;
    uint slot        = tid % vpw;
    uint mask        = (1u << bits) - 1u;

    uint pack_base = gid * n_words;
    uint idx = (packed[pack_base + word_idx] >> (slot * bits)) & mask;
    idx = metal::min(idx, n_centroids - 1u);

    // --- 2. Centroid lookup + sign flip into shared ---
    threadgroup float s[256];
    s[tid] = centroids[idx] * signs[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- 3. Inverse FWHT butterfly ---
    for (uint h = 1; h < D; h <<= 1) {
        uint block_pos = tid % (2 * h);
        uint lo = (block_pos < h) ? tid     : tid - h;
        uint hi = (block_pos < h) ? tid + h : tid;

        float a = s[lo];
        float b = s[hi];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        s[tid] = (block_pos < h) ? (a + b) : (a - b);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // --- 4. Scale ---
    float inv_sqrt_d = 1.0f / metal::sqrt((float)D);
    out[gid * D + tid] = s[tid] * inv_sqrt_d * norms[gid];
