// Fast Walsh-Hadamard Transform — shared memory butterfly.
//
// Grid:  (n_vectors * dim, 1, 1)  — one thread per coordinate
// Group: (dim, 1, 1)              — one threadgroup per vector
//
// Matches Rust fwht_inplace: sequential butterfly at stride h,
// pairs (j, j+h) within blocks of 2*h.

    uint gid = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint D   = threads_per_threadgroup.x;

    threadgroup float s[256];
    s[tid] = inp[gid * D + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

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

    out[gid * D + tid] = s[tid];
