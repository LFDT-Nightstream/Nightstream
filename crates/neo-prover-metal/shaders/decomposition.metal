// Child masks are planar: the positive columns, then the negative columns, of each child.
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
        ulong word = fq_from_word(parent[coefficient * cols + column]);
        bool is_negative = word > (FQ_MODULUS - 1) / 2;
        ulong magnitude = is_negative ? FQ_MODULUS - word : word;
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

// A product lane of one chunk adds at most one term for each column and
// lane, in one exact sum.
static_assert(DEC_CHUNK_COLUMNS * RING_DEGREE <= PRODUCT_SUM_TERMS, "a dense product lane is one exact sum");
static_assert(DEC_CHUNK_COLUMNS * RING_DEGREE <= (1ul << 32), "a signed-unit product lane is one exact sum");

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
    ulong valid = (~0ul << term_start) & (~0ul >> (63 - term_end));
    HalfSums positive_sum = HalfSums{0, 0};
    HalfSums negative_sum = HalfSums{0, 0};
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
                    value = fq_mul(value, magnitude);
                }
                half_sums_add(positive_sum, value);
            }
            ulong negative = masks[mask_base + 2 * (magnitude - 1) + 1] & valid;
            while (negative != 0) {
                uint term = (uint)ctz(negative);
                negative &= negative - 1;
                ulong value = forms[((active - shape[7]) * 2 + component) * RING_DEGREE + coefficient - term];
                if (magnitude != 1) {
                    value = fq_mul(value, magnitude);
                }
                half_sums_add(negative_sum, value);
            }
        }
    }
    partials[index] = fq_sub(half_sums_reduce(positive_sum), half_sums_reduce(negative_sum));
}

// `dec_sparse_ring_partials` over running sources without norm terms, whose
// field values are stored source by source, block by block and lane by lane.
// Source `s` writes the partials of opening `shape[8] + s`. A thread adds
// one coefficient of the product of one source, chunk and component, with
// one reduction.
kernel void dec_dense_ring_partials(
    device const ulong *forms [[buffer(0)]],
    device const ulong *values [[buffer(1)]],
    device const ulong *shape [[buffer(2)]],
    device ulong *partials [[buffer(3)]],
    device const uint *sources [[buffer(4)]],
    device const uint *active_blocks [[buffer(5)]],
    device const uint *active_chunk_bases [[buffer(6)]],
    device const uint *active_chunk_matrices [[buffer(7)]],
    device const uint *matrix_active_offsets [[buffer(8)]],
    uint index [[thread_position_in_grid]]) {
    ulong blocks = shape[3];
    ulong chunk_count = shape[4];
    ulong coefficient = index % RING_PRODUCT_COEFFICIENTS;
    ulong group = index / RING_PRODUCT_COEFFICIENTS;
    ulong component = group % 2;
    ulong chunk = (group / 2) % chunk_count;
    ulong dense = (group / 2) / chunk_count;
    ulong matrix = active_chunk_matrices[chunk];
    if (2 * matrix + component >= shape[2]) {
        return;
    }
    ulong source = sources[dense];
    ulong start = active_chunk_bases[chunk];
    ulong end = min(start + DEC_CHUNK_COLUMNS, (ulong)matrix_active_offsets[matrix + 1]);
    ulong term_start = coefficient >= RING_DEGREE ? coefficient - (RING_DEGREE - 1) : 0;
    ulong term_end = coefficient < RING_DEGREE ? coefficient : RING_DEGREE - 1;
    ProductSum sum = ProductSum{0, 0, 0, 0};
    for (ulong active = start; active < end; ++active) {
        ulong block = (ulong)active_blocks[active] % blocks;
        device const ulong *value = values + (source * blocks + block) * RING_DEGREE;
        device const ulong *form = forms + ((active - shape[7]) * 2 + component) * RING_DEGREE + coefficient;
        for (ulong term = term_start; term <= term_end; ++term) {
            product_sum_add(sum, *(form - term), value[term]);
        }
    }
    ulong partial = ((shape[8] + dense) * chunk_count + chunk) * 2 + component;
    partials[partial * RING_PRODUCT_COEFFICIENTS + coefficient] = product_sum_reduce(sum);
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
        value = fq_add(value, partials[partial]);
    }
    sums[index] = value;
}

kernel void dec_ring_reduce(
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
    ulong value = fq_from_word(sums[base + coefficient]);
    // X^RING_DEGREE = -1.
    if (coefficient + RING_DEGREE < RING_PRODUCT_COEFFICIENTS) {
        value = fq_sub(value, fq_from_word(sums[base + coefficient + RING_DEGREE]));
    }
    output[index] = value;
}
