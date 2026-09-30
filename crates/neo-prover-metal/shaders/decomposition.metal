// Pi_DEC writes fourteen child masks per parent scan. The first group also checks
// that every centered coefficient fits in the fixed base-2 child count. For
// base two the balanced digits of a value are the bits of its magnitude with
// its sign. Child masks are planar: the positive columns, then the negative
// columns, of each child.
constant ushort DEC_SPLIT_CHILDREN_PER_THREAD = 14;

kernel void dec_split_base2_masks(
    device const ulong *parent [[buffer(0)]],
    device const ulong *shape [[buffer(1)]],
    device ulong *masks [[buffer(2)]],
    device atomic_uint *child_nonzero [[buffer(3)]],
    device atomic_uint *status [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    ulong child_count = shape[1];
    ulong cols = shape[3];
    ulong first_child = (index / cols) * DEC_SPLIT_CHILDREN_PER_THREAD;
    ulong column = index % cols;

    ulong positive[DEC_SPLIT_CHILDREN_PER_THREAD];
    ulong negative[DEC_SPLIT_CHILDREN_PER_THREAD];
    for (ushort local = 0; local < DEC_SPLIT_CHILDREN_PER_THREAD; ++local) {
        positive[local] = 0;
        negative[local] = 0;
    }
    for (ulong coefficient = 0; coefficient < RING_DEGREE; ++coefficient) {
        ulong word = gl_from_word(parent[coefficient * cols + column]);
        bool is_negative = word > (GOLDILOCKS_MODULUS - 1) / 2;
        ulong magnitude = is_negative ? GOLDILOCKS_MODULUS - word : word;
        if (first_child == 0 && (magnitude >> child_count) != 0) {
            atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
        }
        for (ushort local = 0; local < DEC_SPLIT_CHILDREN_PER_THREAD; ++local) {
            ulong child = first_child + local;
            if (child < child_count && ((magnitude >> child) & 1ul) != 0) {
                if (is_negative) {
                    negative[local] |= 1ul << coefficient;
                } else {
                    positive[local] |= 1ul << coefficient;
                }
            }
        }
    }

    // Lanes in the first lane's child group share one atomic per child. A
    // SIMD group that crosses into the next group records those lanes alone.
    bool first_group = index / cols == simd_broadcast_first(index) / cols;
    for (ushort local = 0; local < DEC_SPLIT_CHILDREN_PER_THREAD; ++local) {
        ulong child = first_child + local;
        uint nonzero = (positive[local] | negative[local]) != 0 ? 1u : 0u;
        if (child < child_count) {
            masks[2 * child * cols + column] = positive[local];
            masks[(2 * child + 1) * cols + column] = negative[local];
        }
        uint shared = simd_or(first_group ? nonzero : 0u);
        if (child < child_count && (simd_is_first() ? shared : (first_group ? 0u : nonzero)) != 0) {
            atomic_fetch_or_explicit(&child_nonzero[child], 1u, memory_order_relaxed);
        }
    }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void dec_sparse_ring_partials(
    device const ulong *forms [[buffer(0)]],
    device const ulong *masks [[buffer(1)]],
    device const ulong *shape [[buffer(2)]],
    device ulong *partials [[buffer(3)]],
    device const uint *active_children [[buffer(4)]],
    device const uint *active_blocks [[buffer(5)]],
    device const uint *active_chunk_bases [[buffer(6)]],
    device const uint *active_chunk_matrices [[buffer(7)]],
    device const uint *matrix_active_offsets [[buffer(8)]],
    device const uint *child_nonzero [[buffer(9)]],
    uint index [[thread_position_in_grid]]) {
    ulong active_count = shape[1];
    ulong form_rows = shape[2];
    ulong blocks = shape[3];
    ulong chunk_count = shape[4];
    ulong magnitudes = shape[6];
    ulong coefficient = index % RING_PRODUCT_COEFFICIENTS;
    ulong rest = index / RING_PRODUCT_COEFFICIENTS;
    ulong component = rest % 2;
    rest /= 2;
    ulong chunk = rest % chunk_count;
    ulong active_child = rest / chunk_count;
    if (active_child >= active_count) {
        return;
    }
    ulong child = active_children[active_child];
    if (shape[5] != 0 && child_nonzero[child] == 0) {
        partials[index] = 0;
        return;
    }
    ulong matrix = active_chunk_matrices[chunk];
    if (2 * matrix + component >= form_rows) {
        return;
    }
    ulong start = active_chunk_bases[chunk];
    ulong end = min(start + DEC_CHUNK_COLUMNS, (ulong)matrix_active_offsets[matrix + 1]);
    ulong term_start = coefficient >= RING_DEGREE ? coefficient - (RING_DEGREE - 1) : 0;
    ulong term_end = coefficient < RING_DEGREE ? coefficient : RING_DEGREE - 1;
    ulong valid = (~0ul << term_start) & ((1ul << (term_end + 1)) - 1);
    ulong positive_lo = 0;
    ulong positive_hi = 0;
    ulong negative_lo = 0;
    ulong negative_hi = 0;
    for (ulong active = start; active < end; ++active) {
        ulong block = (ulong)active_blocks[active] % blocks;
        ulong mask_base = 2 * magnitudes * (child * blocks + block);
        for (ulong magnitude = 1; magnitude <= magnitudes; ++magnitude) {
            ulong positive = masks[mask_base + 2 * (magnitude - 1)] & valid;
            while (positive != 0) {
                uint term = (uint)ctz(positive);
                positive &= positive - 1;
                ulong value = forms[((active - shape[7]) * 2 + component) * RING_DEGREE + coefficient - term];
                if (magnitude != 1) {
                    value = gl_mul(value, magnitude);
                }
                ulong next = positive_lo + value;
                positive_hi += next < positive_lo;
                positive_lo = next;
            }
            ulong negative = masks[mask_base + 2 * (magnitude - 1) + 1] & valid;
            while (negative != 0) {
                uint term = (uint)ctz(negative);
                negative &= negative - 1;
                ulong value = forms[((active - shape[7]) * 2 + component) * RING_DEGREE + coefficient - term];
                if (magnitude != 1) {
                    value = gl_mul(value, magnitude);
                }
                ulong next = negative_lo + value;
                negative_hi += next < negative_lo;
                negative_lo = next;
            }
        }
    }
    partials[index] = gl_sub(gl_reduce_sum(positive_lo, positive_hi), gl_reduce_sum(negative_lo, negative_hi));
}

kernel void dec_sparse_ring_sum_chunks(
    device const ulong *partials [[buffer(0)]],
    device const ulong *shape [[buffer(1)]],
    device ulong *sums [[buffer(2)]],
    device const uint *matrix_chunk_offsets [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    ulong form_rows = shape[2];
    ulong chunk_count = shape[4];
    ulong group = index / RING_PRODUCT_COEFFICIENTS;
    ulong coefficient = index % RING_PRODUCT_COEFFICIENTS;
    ulong child = group / form_rows;
    ulong form_row = group % form_rows;
    ulong matrix = form_row / 2;
    ulong component = form_row % 2;
    ulong value = 0;
    ulong start = matrix_chunk_offsets[matrix];
    ulong end = matrix_chunk_offsets[matrix + 1];
    for (ulong chunk = start; chunk < end; ++chunk) {
        ulong partial = ((child * chunk_count + chunk) * 2 + component) * RING_PRODUCT_COEFFICIENTS + coefficient;
        value = gl_add(value, partials[partial]);
    }
    sums[index] = value;
}

kernel void dec_ring_reduce_phi81(
    device const ulong *sums [[buffer(0)]],
    device const ulong *shape [[buffer(1)]],
    device ulong *output [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    ulong groups = shape[1] * shape[2];
    ulong group = index / RING_DEGREE;
    ulong coefficient = index % RING_DEGREE;
    if (group >= groups) {
        return;
    }
    ulong base = group * RING_PRODUCT_COEFFICIENTS;
    ulong value = gl_from_word(sums[base + coefficient]);
    if (coefficient <= 26) {
        value = gl_sub(value, gl_from_word(sums[base + coefficient + 54]));
        if (coefficient <= 25) {
            value = gl_add(value, gl_from_word(sums[base + coefficient + 81]));
        }
    } else {
        value = gl_sub(value, gl_from_word(sums[base + coefficient + 27]));
    }
    output[index] = value;
}

