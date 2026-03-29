// TurboQuant compress: float vectors -> (packed uint32, norms)
//
// Fused pipeline per vector:
//   1. Parallel norm reduction
//   2. Normalize
//   3. FWHT butterfly (shared memory)
//   4. Scale by 1/sqrt(D) + sign flip
//   5. Boundary quantize
//   6. Atomic bit-pack into uint32 words
//
// Grid:  (n_vectors * dim, 1, 1)
// Group: (dim, 1, 1)
//
// Inputs:
//   data       — (n_vectors * dim,) float32
//   signs      — (dim,) float32
//   boundaries — (n_levels - 1,) float32
//   params     — [n_boundaries, bits, vals_per_word] uint32
//
// Outputs:
//   packed — (n_vectors * n_words,) uint32  (atomic)
//   norms  — (n_vectors,) float32

    uint gid = threadgroup_position_in_grid.x;   // vector index
    uint tid = thread_position_in_threadgroup.x;  // coordinate index
    uint D   = threads_per_threadgroup.x;
    uint base = gid * D;

    float val = data[base + tid];

    // --- 1. Parallel norm via reduction ---
    threadgroup float s[256];
    threadgroup float r[256];

    r[tid] = val * val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = D / 2; step > 0; step >>= 1) {
        if (tid < step) r[tid] += r[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float norm = metal::sqrt(r[0]);
    float inv_norm = (norm > 1e-8f) ? (1.0f / norm) : 0.0f;
    if (tid == 0) atomic_store_explicit(&norms[gid], norm, memory_order_relaxed);

    // --- 2. Normalize into shared ---
    s[tid] = val * inv_norm;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- 3. FWHT butterfly ---
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

    // --- 4. Scale + sign flip ---
    float inv_sqrt_d = 1.0f / metal::sqrt((float)D);
    float rotated = s[tid] * inv_sqrt_d * signs[tid];

    // --- 5. Quantize ---
    uint n_bounds = params[0];
    uint bucket = 0;
    for (uint i = 0; i < n_bounds; i++) {
        if (rotated > boundaries[i]) bucket++;
        else break;
    }

    // --- 6. Atomic pack ---
    uint bits = params[1];
    uint vpw  = params[2];
    uint word_idx  = tid / vpw;
    uint slot      = tid % vpw;
    uint pack_base = gid * ((D + vpw - 1) / vpw);

    atomic_fetch_or_explicit(&packed[pack_base + word_idx], bucket << (slot * bits), memory_order_relaxed);
