// Three Nebula lane commitments over one resident signed-mask batch.

kernel void ajtai_lane_ring_partials(
    device const ulong *ops_matrix [[buffer(0)]],
    device const ulong *mem_matrix [[buffer(1)]],
    device const ulong *masks [[buffer(2)]],
    device const ulong *shape [[buffer(3)]],
    device ulong *partials [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    ulong full_cols = shape[0];
    ulong witness_count = shape[1];
    ulong rows = shape[2];
    ulong ops_cols = shape[3];
    ulong mem_cols = shape[4];
    ulong ops_chunks = shape[5];
    ulong mem_chunks = shape[6];
    ulong ops_groups = witness_count * rows * ops_chunks;
    ulong mem_groups = witness_count * rows * mem_chunks;
    ulong coefficient = (ulong)index % RING_PRODUCT_COEFFICIENTS;
    ulong packed_group = (ulong)index / RING_PRODUCT_COEFFICIENTS;

    ulong lane;
    ulong local_group;
    if (packed_group < ops_groups) {
        lane = 0;
        local_group = packed_group;
    } else if (packed_group < ops_groups + mem_groups) {
        lane = 1;
        local_group = packed_group - ops_groups;
    } else {
        lane = 2;
        local_group = packed_group - ops_groups - mem_groups;
        if (local_group >= mem_groups) {
            return;
        }
    }

    ulong lane_cols = lane == 0 ? ops_cols : mem_cols;
    ulong chunks = lane == 0 ? ops_chunks : mem_chunks;
    ulong source_offset = shape[7 + lane];
    ulong chunk = local_group % chunks;
    ulong group = local_group / chunks;
    ulong witness = group / rows;
    ulong form_row = group % rows;
    ulong column_start = chunk * DEC_CHUNK_COLUMNS;
    ulong column_end = min(column_start + DEC_CHUNK_COLUMNS, lane_cols);
    ulong term_start = coefficient >= RING_DEGREE ? coefficient - (RING_DEGREE - 1) : 0;
    ulong term_end = coefficient < RING_DEGREE ? coefficient : RING_DEGREE - 1;
    ulong valid = (~0ul << term_start) & ((1ul << (term_end + 1)) - 1);
    ulong positive_lo = 0;
    ulong positive_hi = 0;
    ulong negative_lo = 0;
    ulong negative_hi = 0;
    for (ulong column = column_start; column < column_end; ++column) {
        ulong mask_base = 2 * (witness * full_cols + source_offset + column);
        ulong positive = masks[mask_base] & valid;
        while (positive != 0) {
            uint term = (uint)ctz(positive);
            positive &= positive - 1;
            ulong form_index = (form_row * lane_cols + column) * RING_DEGREE + coefficient - term;
            ulong value = lane == 0 ? ops_matrix[form_index] : mem_matrix[form_index];
            ulong next = positive_lo + value;
            positive_hi += next < positive_lo;
            positive_lo = next;
        }
        ulong negative = masks[mask_base + 1] & valid;
        while (negative != 0) {
            uint term = (uint)ctz(negative);
            negative &= negative - 1;
            ulong form_index = (form_row * lane_cols + column) * RING_DEGREE + coefficient - term;
            ulong value = lane == 0 ? ops_matrix[form_index] : mem_matrix[form_index];
            ulong next = negative_lo + value;
            negative_hi += next < negative_lo;
            negative_lo = next;
        }
    }
    partials[index] = gl_sub(gl_reduce_sum(positive_lo, positive_hi), gl_reduce_sum(negative_lo, negative_hi));
}

kernel void ajtai_lane_ring_sum_chunks(
    device const ulong *partials [[buffer(0)]],
    device const ulong *shape [[buffer(1)]],
    device ulong *sums [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    ulong witness_count = shape[1];
    ulong rows = shape[2];
    ulong ops_chunks = shape[5];
    ulong mem_chunks = shape[6];
    ulong groups_per_lane = witness_count * rows;
    ulong group = (ulong)index / RING_PRODUCT_COEFFICIENTS;
    ulong coefficient = (ulong)index % RING_PRODUCT_COEFFICIENTS;
    ulong lane = group / groups_per_lane;
    if (lane >= 3) {
        return;
    }
    ulong local_group = group % groups_per_lane;
    ulong chunks = lane == 0 ? ops_chunks : mem_chunks;
    ulong partial_base = lane == 0
        ? 0
        : (lane == 1
            ? groups_per_lane * ops_chunks
            : groups_per_lane * (ops_chunks + mem_chunks));
    ulong value = 0;
    for (ulong chunk = 0; chunk < chunks; ++chunk) {
        ulong partial_group = partial_base + local_group * chunks + chunk;
        value = gl_add(value, partials[partial_group * RING_PRODUCT_COEFFICIENTS + coefficient]);
    }
    sums[index] = value;
}

kernel void ajtai_lane_ring_reduce_phi81(
    device const ulong *sums [[buffer(0)]],
    device const ulong *shape [[buffer(1)]],
    device ulong *output [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    ulong groups = 3 * shape[1] * shape[2];
    ulong group = (ulong)index / RING_DEGREE;
    ulong coefficient = (ulong)index % RING_DEGREE;
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
