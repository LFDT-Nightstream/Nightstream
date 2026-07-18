// NC table folding and sumcheck kernels. Common field arithmetic remains in goldilocks.metal.
// Compact rows store a cyclic window; dense rows always store all 54 ring lanes.

constant ushort NC_SIMD_WIDTH = 32;
constant ushort NC_DENSE_PAIRS_PER_GROUP = 8;
constant ushort NC_REDUCTION_THREADS = 256;
constant ushort NC_REDUCTION_SIMD_GROUPS = 8;

inline ulong nc_simd_shuffle_xor_word(ulong value, ushort mask) {
    uint lo = simd_shuffle_xor((uint)value, mask);
    uint hi = simd_shuffle_xor((uint)(value >> 32), mask);
    return ((ulong)hi << 32) | lo;
}

inline Kx nc_simd_shuffle_xor(Kx value, ushort mask) {
    return Kx{
        nc_simd_shuffle_xor_word(value.c0, mask),
        nc_simd_shuffle_xor_word(value.c1, mask)};
}

inline Kx nc_simd_reduce(Kx value) {
    for (ushort mask = 1; mask < NC_SIMD_WIDTH; mask <<= 1) {
        value = kx_add(value, nc_simd_shuffle_xor(value, mask));
    }
    return value;
}

kernel void nc_reduce_partials(
    device const ulong *partials [[buffer(0)]],
    device const ulong *shape [[buffer(1)]],
    device ulong *output [[buffer(2)]],
    uint lane [[thread_index_in_threadgroup]],
    uint coefficient [[threadgroup_position_in_grid]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]]) {
    ulong rows = shape[0];
    ulong coefficient_count = shape[1];
    if ((ulong)coefficient >= coefficient_count) {
        return;
    }
    Kx total = Kx{0, 0};
    for (ulong row = lane; row < rows; row += NC_REDUCTION_THREADS) {
        total = kx_add(total, load_k(partials, row * coefficient_count + coefficient));
    }
    total = nc_simd_reduce(total);
    threadgroup Kx shared[NC_REDUCTION_SIMD_GROUPS];
    if (simd_lane == 0) {
        shared[simd_group] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        total = shared[0];
        for (ushort group = 1; group < NC_REDUCTION_SIMD_GROUPS; ++group) {
            total = kx_add(total, shared[group]);
        }
        output[2 * coefficient] = total.c0;
        output[2 * coefficient + 1] = total.c1;
    }
}

inline Kx nc_signed_mask_digit(
    ulong positive,
    ulong negative,
    ulong bit) {
    if ((positive & bit) != 0) {
        return Kx{1, 0};
    }
    if ((negative & bit) != 0) {
        return Kx{GOLDILOCKS_MODULUS - 1, 0};
    }
    return Kx{0, 0};
}

// Small mask inputs materialize their first folded width directly.
kernel void nc_fold_signed_masks(
    device const ulong *masks [[buffer(0)]],
    device const ulong *challenge_words [[buffer(1)]],
    device const ulong *shape [[buffer(2)]],
    device ulong *output [[buffer(3)]],
    device const uint *active_witnesses [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    ulong rows = shape[0];
    ulong witness_count = shape[1];
    ulong blocks = shape[2];
    ulong active_rows = shape[3];
    ulong half_rows = rows / 2;
    ulong live_output_rows = (active_rows + 1) / 2;
    ulong active_witness = index / live_output_rows;
    ulong out_row = index % live_output_rows;
    if (active_witness >= witness_count) {
        return;
    }
    ulong witness = (ulong)active_witnesses[active_witness];
    ulong lo_row = 2 * out_row;
    Kx lo = Kx{0, 0};
    Kx hi = Kx{0, 0};
    if (lo_row < active_rows) {
        // RING_DEGREE and lo_row are even, so the paired hi row is in this block too.
        ulong block = lo_row / RING_DEGREE;
        ulong mask_base = 2 * (witness * blocks + block);
        ulong positive = masks[mask_base];
        ulong negative = masks[mask_base + 1];
        lo = nc_signed_mask_digit(
            positive,
            negative,
            1ul << (lo_row % RING_DEGREE));
        if (lo_row + 1 < active_rows) {
            hi = nc_signed_mask_digit(
                positive,
                negative,
                1ul << ((lo_row + 1) % RING_DEGREE));
        }
    }
    Kx challenge = Kx{
        gl_from_word(challenge_words[0]),
        gl_from_word(challenge_words[1])};
    Kx zero = Kx{0, 0};
    Kx one_minus_challenge = kx_sub(Kx{1, 0}, challenge);
    Kx folded_lo = lo.c0 == 1
        ? one_minus_challenge
        : (lo.c0 == 0 ? zero : kx_sub(zero, one_minus_challenge));
    Kx folded_hi = hi.c0 == 1
        ? challenge
        : (hi.c0 == 0 ? zero : kx_sub(zero, challenge));
    ulong output_base = active_witness * rows + 2 * out_row;
    output[2 * output_base] = folded_lo.c0;
    output[2 * output_base + 1] = folded_lo.c1;
    output[2 * (output_base + 1)] = folded_hi.c0;
    output[2 * (output_base + 1) + 1] = folded_hi.c1;
}

// Fold only live rows while preserving padded per-witness input and output strides.
// Compact width doubles until overlap, then every later row stays dense.
kernel void nc_fold_compact(
    device const ulong *input [[buffer(0)]],
    device const ulong *challenge_words [[buffer(1)]],
    device const ulong *shape [[buffer(2)]],
    device ulong *output [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    ulong witness_count = shape[0];
    ulong rows = shape[1];
    ulong width = shape[2];
    bool dense = shape[3] != 0;
    ulong live_rows = shape[4];
    bool output_dense = dense || 2 * width > RING_DEGREE;
    ulong half_rows = (rows + 1) / 2;
    ulong live_output_rows = (live_rows + 1) / 2;
    ulong input_per_witness = dense ? rows * RING_DEGREE : rows * width;
    ulong output_width = output_dense ? RING_DEGREE : 2 * width;
    ulong output_per_witness = half_rows * output_width;
    ulong live_output_per_witness = live_output_rows * output_width;
    ulong witness = index / live_output_per_witness;
    ulong within = index % live_output_per_witness;
    ulong out_row = within / output_width;
    ulong slot = within % output_width;
    if (witness >= witness_count) {
        return;
    }
    ulong input_base = witness * input_per_witness;
    ulong lo_row = 2 * out_row;
    ulong hi_row = lo_row + 1;
    Kx challenge = Kx{challenge_words[0], challenge_words[1]};
    Kx lo = Kx{0, 0};
    Kx hi = Kx{0, 0};
    if (!output_dense) {
        if (slot < width) {
            lo = load_k(input, input_base + lo_row * width + slot);
        } else if (hi_row < live_rows) {
            hi = load_k(input, input_base + hi_row * width + slot - width);
        }
    } else if (dense) {
        lo = load_k(input, input_base + lo_row * RING_DEGREE + slot);
        if (hi_row < live_rows) {
            hi = load_k(input, input_base + hi_row * RING_DEGREE + slot);
        }
    } else {
        ulong start_lo = (lo_row * width) % RING_DEGREE;
        ulong lo_slot = (slot + RING_DEGREE - start_lo) % RING_DEGREE;
        if (lo_slot < width) {
            lo = load_k(input, input_base + lo_row * width + lo_slot);
        }
        if (hi_row < live_rows) {
            ulong start_hi = (hi_row * width) % RING_DEGREE;
            ulong hi_slot = (slot + RING_DEGREE - start_hi) % RING_DEGREE;
            if (hi_slot < width) {
                hi = load_k(input, input_base + hi_row * width + hi_slot);
            }
        }
    }
    Kx folded = kx_add(lo, kx_mul(challenge, kx_sub(hi, lo)));
    ulong output_index = witness * output_per_witness + out_row * output_width + slot;
    output[2 * output_index] = folded.c0;
    output[2 * output_index + 1] = folded.c1;
}

inline void nc_accumulate_digit_constraint(
    thread Kx *inner,
    Kx weight,
    Kx a,
    Kx b) {
    Kx one = Kx{1, 0};
    Kx weighted_a = kx_mul(weight, a);
    Kx weighted_b = kx_mul(weight, b);
    Kx a2 = kx_mul(a, a);
    Kx b2 = kx_mul(b, b);
    Kx three_a2 = kx_add(a2, kx_add(a2, a2));
    Kx three_b2 = kx_add(b2, kx_add(b2, b2));
    inner[0] = kx_add(inner[0], kx_mul(weighted_a, kx_sub(a2, one)));
    inner[1] = kx_add(inner[1], kx_mul(weighted_b, kx_sub(three_a2, one)));
    inner[2] = kx_add(inner[2], kx_mul(weighted_a, three_b2));
    inner[3] = kx_add(inner[3], kx_mul(weighted_b, b2));
}

inline void nc_accumulate_low_window_constraint(
    thread Kx *inner,
    Kx weight,
    Kx a) {
    Kx weighted_a = kx_mul(weight, a);
    Kx weighted_a3 = kx_mul(weighted_a, kx_mul(a, a));
    Kx three_weighted_a3 = kx_add(weighted_a3, kx_add(weighted_a3, weighted_a3));
    inner[0] = kx_add(inner[0], kx_sub(weighted_a3, weighted_a));
    inner[1] = kx_add(inner[1], kx_sub(weighted_a, three_weighted_a3));
    inner[2] = kx_add(inner[2], three_weighted_a3);
    inner[3] = kx_sub(inner[3], weighted_a3);
}

inline void nc_accumulate_high_window_constraint(
    thread Kx *inner,
    Kx weight,
    Kx b) {
    Kx weighted_b = kx_mul(weight, b);
    Kx weighted_b3 = kx_mul(weighted_b, kx_mul(b, b));
    inner[1] = kx_sub(inner[1], weighted_b);
    inner[3] = kx_add(inner[3], weighted_b3);
}

inline void nc_accumulate_signed_low_constraint(
    thread Kx *inner,
    Kx signed_weight) {
    Kx twice = kx_add(signed_weight, signed_weight);
    Kx three = kx_add(twice, signed_weight);
    inner[1] = kx_sub(inner[1], twice);
    inner[2] = kx_add(inner[2], three);
    inner[3] = kx_sub(inner[3], signed_weight);
}

inline void nc_accumulate_signed_high_constraint(
    thread Kx *inner,
    Kx signed_weight) {
    inner[1] = kx_sub(inner[1], signed_weight);
    inner[3] = kx_add(inner[3], signed_weight);
}

// Multiply a cubic by the equality line with two degree-one Karatsuba blocks.
inline void nc_multiply_cubic_by_eq(
    thread Kx *product,
    Kx eq_zero,
    Kx eq_one,
    thread Kx *cubic) {
    Kx eq_slope = kx_sub(eq_one, eq_zero);

    Kx lo_zero = kx_mul(eq_zero, cubic[0]);
    Kx lo_two = kx_mul(eq_slope, cubic[1]);
    Kx lo_one = kx_sub(
        kx_sub(kx_mul(eq_one, kx_add(cubic[0], cubic[1])), lo_zero),
        lo_two);

    Kx hi_zero = kx_mul(eq_zero, cubic[2]);
    Kx hi_two = kx_mul(eq_slope, cubic[3]);
    Kx hi_one = kx_sub(
        kx_sub(kx_mul(eq_one, kx_add(cubic[2], cubic[3])), hi_zero),
        hi_two);

    product[0] = lo_zero;
    product[1] = lo_one;
    product[2] = kx_add(lo_two, hi_zero);
    product[3] = hi_one;
    product[4] = hi_two;
}

inline Kx nc_mask_basis_digit(
    device const ulong *masks,
    ulong witness,
    ulong blocks,
    ulong active_rows,
    ulong row,
    Kx basis) {
    if (row >= active_rows) {
        return Kx{0, 0};
    }
    ulong block = row / RING_DEGREE;
    ulong bit = 1ul << (row % RING_DEGREE);
    ulong mask_base = 2 * (witness * blocks + block);
    if ((masks[mask_base] & bit) != 0) {
        return basis;
    }
    if ((masks[mask_base + 1] & bit) != 0) {
        return kx_sub(Kx{0, 0}, basis);
    }
    return Kx{0, 0};
}

inline Kx nc_mask_basis_digit_at(
    device const ulong *masks,
    ulong witness,
    ulong blocks,
    ulong block,
    ulong bit,
    Kx basis) {
    ulong mask_base = 2 * (witness * blocks + block);
    if ((masks[mask_base] & bit) != 0) {
        return basis;
    }
    if ((masks[mask_base + 1] & bit) != 0) {
        return kx_sub(Kx{0, 0}, basis);
    }
    return Kx{0, 0};
}

// Large mask inputs fold one shared basis instead of every witness row.
kernel void nc_expand_mask_basis(
    device const ulong *basis [[buffer(0)]],
    device const ulong *challenge_words [[buffer(1)]],
    device const ulong *shape [[buffer(2)]],
    device ulong *output [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    ulong width = shape[2];
    if ((ulong)index >= 2 * width) {
        return;
    }
    Kx challenge = Kx{challenge_words[0], challenge_words[1]};
    Kx scale = index < width ? kx_sub(Kx{1, 0}, challenge) : challenge;
    Kx value = kx_mul(load_k(basis, index % width), scale);
    output[2 * index] = value.c0;
    output[2 * index + 1] = value.c1;
}

// Materialize only live rows; unwritten padded suffix rows are never read again.
kernel void nc_materialize_mask_dense(
    device const ulong *masks [[buffer(0)]],
    device const ulong *basis [[buffer(1)]],
    device const ulong *mask_shape [[buffer(2)]],
    device const ulong *fold_shape [[buffer(3)]],
    device ulong *output [[buffer(4)]],
    device const uint *active_witnesses [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    ulong witness_count = fold_shape[0];
    ulong output_rows = fold_shape[1] / 2;
    ulong live_output_rows = (fold_shape[4] + 1) / 2;
    ulong values_per_witness = output_rows * RING_DEGREE;
    ulong live_values_per_witness = live_output_rows * RING_DEGREE;
    ulong active_witness = index / live_values_per_witness;
    ulong within = index % live_values_per_witness;
    if (active_witness >= witness_count) {
        return;
    }
    ulong output_row = within / RING_DEGREE;
    ulong ring_lane = within % RING_DEGREE;
    ulong source_width = 2 * fold_shape[2];
    ulong source_base = output_row * source_width;
    ulong source_start = source_base % RING_DEGREE;
    ulong slot = (ring_lane + RING_DEGREE - source_start) % RING_DEGREE;
    ulong witness = (ulong)active_witnesses[active_witness];
    ulong blocks = mask_shape[2];
    ulong active_rows = mask_shape[3];
    Kx value = nc_mask_basis_digit(
        masks,
        witness,
        blocks,
        active_rows,
        source_base + slot,
        load_k(basis, slot));
    for (ulong extra = slot + RING_DEGREE; extra < source_width; extra += RING_DEGREE) {
        value = kx_add(
            value,
            nc_mask_basis_digit(
                masks,
                witness,
                blocks,
                active_rows,
                source_base + extra,
                load_k(basis, extra)));
    }
    ulong output_index = active_witness * values_per_witness + output_row * RING_DEGREE + ring_lane;
    output[2 * output_index] = value.c0;
    output[2 * output_index + 1] = value.c1;
}

// Mask-native rounds traverse the compact active-witness index, not the full batch.
kernel void nc_round_mask_partials(
    device const ulong *eq_table [[buffer(0)]],
    device const ulong *masks [[buffer(1)]],
    device const ulong *shape [[buffer(2)]],
    device const ulong *mask_shape [[buffer(3)]],
    device const ulong *weights [[buffer(4)]],
    device const ulong *basis [[buffer(5)]],
    device ulong *partials [[buffer(6)]],
    device const uint *active_witnesses [[buffer(7)]],
    uint pair [[thread_position_in_grid]],
    uint lane_index [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]]) {
    ulong witness_count = shape[1];
    ulong width = shape[2];
    ulong live_rows = shape[5];
    ulong blocks = mask_shape[2];
    ulong active_rows = mask_shape[3];
    threadgroup Kx shared[2 * 5];
    Kx local[5] = {Kx{0, 0}, Kx{0, 0}, Kx{0, 0}, Kx{0, 0}, Kx{0, 0}};
    if (pair < (live_rows + 1) / 2) {
        ulong index = 2 * pair;
        Kx e0 = load_k(eq_table, index);
        Kx e_at_one = load_k(eq_table, index + 1);
        Kx inner[4] = {Kx{0, 0}, Kx{0, 0}, Kx{0, 0}, Kx{0, 0}};
        ulong lo_base = index * width;
        ulong hi_base = (index + 1) * width;
        ulong lo_start = lo_base % RING_DEGREE;
        ulong hi_start = hi_base % RING_DEGREE;
        if (width == 1) {
            if (index < active_rows) {
                ulong block = index / RING_DEGREE;
                ulong lo_bit = 1ul << (index % RING_DEGREE);
                ulong hi_bit = 1ul << ((index + 1) % RING_DEGREE);
                bool hi_active = index + 1 < active_rows;
                for (ulong active_witness = 0; active_witness < witness_count; ++active_witness) {
                    ulong witness = (ulong)active_witnesses[active_witness];
                    ulong mask_base = 2 * (witness * blocks + block);
                    ulong positive = masks[mask_base];
                    ulong negative = masks[mask_base + 1];
                    if (((positive | negative) & lo_bit) != 0) {
                        Kx weight = load_k(weights, active_witness * RING_DEGREE + index % RING_DEGREE);
                        Kx signed_weight = (positive & lo_bit) != 0 ? weight : kx_sub(Kx{0, 0}, weight);
                        nc_accumulate_signed_low_constraint(inner, signed_weight);
                    }
                    if (hi_active && ((positive | negative) & hi_bit) != 0) {
                        Kx weight = load_k(weights, active_witness * RING_DEGREE + (index + 1) % RING_DEGREE);
                        Kx signed_weight = (positive & hi_bit) != 0 ? weight : kx_sub(Kx{0, 0}, weight);
                        nc_accumulate_signed_high_constraint(inner, signed_weight);
                    }
                }
            }
        } else {
            for (ulong active_witness = 0; active_witness < witness_count; ++active_witness) {
                ulong witness = (ulong)active_witnesses[active_witness];
                if (2 * width <= RING_DEGREE) {
                for (ulong slot = 0; slot < width; ++slot) {
                    Kx a = nc_mask_basis_digit(
                        masks,
                        witness,
                        blocks,
                        active_rows,
                        lo_base + slot,
                        load_k(basis, slot));
                    if (a.c0 != 0 || a.c1 != 0) {
                        Kx weight = load_k(
                            weights,
                            active_witness * RING_DEGREE + (lo_start + slot) % RING_DEGREE);
                        nc_accumulate_low_window_constraint(inner, weight, a);
                    }
                }
                for (ulong slot = 0; slot < width; ++slot) {
                    Kx hi = nc_mask_basis_digit(
                        masks,
                        witness,
                        blocks,
                        active_rows,
                        hi_base + slot,
                        load_k(basis, slot));
                    if (hi.c0 != 0 || hi.c1 != 0) {
                        Kx weight = load_k(
                            weights,
                            active_witness * RING_DEGREE + (hi_start + slot) % RING_DEGREE);
                        nc_accumulate_high_window_constraint(inner, weight, hi);
                    }
                }
                } else {
                    ulong lo_block = lo_base / RING_DEGREE;
                    ulong hi_offset = lo_start + width;
                    ulong hi_block = lo_block;
                    if (hi_offset >= RING_DEGREE) {
                        hi_offset -= RING_DEGREE;
                        hi_block += 1;
                    }
                    if (hi_offset >= RING_DEGREE) {
                        hi_offset -= RING_DEGREE;
                        hi_block += 1;
                    }
                    for (ulong ring_lane = 0; ring_lane < RING_DEGREE; ++ring_lane) {
                    ulong lo_slot = ring_lane >= lo_start
                        ? ring_lane - lo_start
                        : ring_lane + RING_DEGREE - lo_start;
                    ulong hi_slot = ring_lane >= hi_start
                        ? ring_lane - hi_start
                        : ring_lane + RING_DEGREE - hi_start;
                    bool has_lo = lo_slot < width;
                    bool has_hi = hi_slot < width;
                    if (!has_lo && !has_hi) {
                        continue;
                    }
                    Kx a = Kx{0, 0};
                    if (has_lo && lo_base + lo_slot < active_rows) {
                        ulong offset = lo_start + lo_slot;
                        bool carry = offset >= RING_DEGREE;
                        ulong bit_index = carry ? offset - RING_DEGREE : offset;
                        a = nc_mask_basis_digit_at(
                            masks,
                            witness,
                            blocks,
                            lo_block + (carry ? 1ul : 0ul),
                            1ul << bit_index,
                            load_k(basis, lo_slot));
                        ulong second = lo_slot + RING_DEGREE;
                        if (second < width && lo_base + second < active_rows) {
                            a = kx_add(
                                a,
                                nc_mask_basis_digit_at(
                                    masks,
                                    witness,
                                    blocks,
                                    lo_block + (carry ? 2ul : 1ul),
                                    1ul << bit_index,
                                    load_k(basis, second)));
                        }
                    }
                    Kx hi = Kx{0, 0};
                    if (has_hi && hi_base + hi_slot < active_rows) {
                        ulong offset = hi_start + hi_slot;
                        bool carry = offset >= RING_DEGREE;
                        ulong bit_index = carry ? offset - RING_DEGREE : offset;
                        hi = nc_mask_basis_digit_at(
                            masks,
                            witness,
                            blocks,
                            hi_block + (carry ? 1ul : 0ul),
                            1ul << bit_index,
                            load_k(basis, hi_slot));
                        ulong second = hi_slot + RING_DEGREE;
                        if (second < width && hi_base + second < active_rows) {
                            hi = kx_add(
                                hi,
                                nc_mask_basis_digit_at(
                                    masks,
                                    witness,
                                    blocks,
                                    hi_block + (carry ? 2ul : 1ul),
                                    1ul << bit_index,
                                    load_k(basis, second)));
                        }
                    }
                    Kx weight = load_k(weights, active_witness * RING_DEGREE + ring_lane);
                    if (has_lo && has_hi) {
                        nc_accumulate_digit_constraint(inner, weight, a, kx_sub(hi, a));
                    } else if (has_lo) {
                        nc_accumulate_low_window_constraint(inner, weight, a);
                    } else {
                        nc_accumulate_high_window_constraint(inner, weight, hi);
                    }
                    }
                }
            }
        }
        nc_multiply_cubic_by_eq(local, e0, e_at_one, inner);
    }
    for (uint coefficient = 0; coefficient < 5; ++coefficient) {
        local[coefficient] = nc_simd_reduce(local[coefficient]);
        if (simd_lane == 0) {
            shared[simd_group * 5 + coefficient] = local[coefficient];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane_index == 0) {
        for (uint coefficient = 0; coefficient < 5; ++coefficient) {
            Kx value = kx_add(shared[coefficient], shared[5 + coefficient]);
            ulong output_index = group * 5 + coefficient;
            partials[2 * output_index] = value.c0;
            partials[2 * output_index + 1] = value.c1;
        }
    }
}

// Dense rows map one SIMD group to each row pair; compact rows map one group
// across many pairs and reconstruct only their cyclic windows.
kernel void nc_round_partials(
    device const ulong *eq_table [[buffer(0)]],
    device const ulong *shape [[buffer(1)]],
    device const ulong *digit_values [[buffer(2)]],
    device const ulong *weights [[buffer(3)]],
    device ulong *partials [[buffer(4)]],
    uint pair [[thread_position_in_grid]],
    uint lane_index [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]]) {
    ulong witness_count = shape[1];
    ulong width = shape[2];
    bool dense = shape[3] != 0;
    ulong values_per_witness = shape[4];
    ulong live_table_len = shape[5];
    threadgroup Kx shared[8 * 5];
    Kx local[5] = {Kx{0, 0}, Kx{0, 0}, Kx{0, 0}, Kx{0, 0}, Kx{0, 0}};
    if (dense) {
        ulong dense_pair = (ulong)group * NC_DENSE_PAIRS_PER_GROUP + simd_group;
        Kx inner[4] = {Kx{0, 0}, Kx{0, 0}, Kx{0, 0}, Kx{0, 0}};
        if (dense_pair < (live_table_len + 1) / 2) {
            ulong index = 2 * dense_pair;
            bool hi_active = index + 1 < live_table_len;
            ulong terms = witness_count * RING_DEGREE;
            ulong ring_lane = simd_lane;
            ulong witness_base = 0;
            for (ulong term = simd_lane; term < terms; term += NC_SIMD_WIDTH) {
                Kx weight = load_k(weights, term);
                Kx a = load_k(digit_values, witness_base + index * RING_DEGREE + ring_lane);
                Kx hi = Kx{0, 0};
                if (hi_active) {
                    hi = load_k(digit_values, witness_base + (index + 1) * RING_DEGREE + ring_lane);
                }
                nc_accumulate_digit_constraint(inner, weight, a, kx_sub(hi, a));
                ring_lane += NC_SIMD_WIDTH;
                if (ring_lane >= RING_DEGREE) {
                    ring_lane -= RING_DEGREE;
                    witness_base += values_per_witness;
                }
            }
        }
        for (uint coefficient = 0; coefficient < 4; ++coefficient) {
            inner[coefficient] = nc_simd_reduce(inner[coefficient]);
        }
        if (simd_lane == 0) {
            if (dense_pair < (live_table_len + 1) / 2) {
                ulong index = 2 * dense_pair;
                Kx e0 = load_k(eq_table, index);
                Kx e_at_one = load_k(eq_table, index + 1);
                nc_multiply_cubic_by_eq(local, e0, e_at_one, inner);
            }
            for (uint coefficient = 0; coefficient < 5; ++coefficient) {
                shared[simd_group * 5 + coefficient] = local[coefficient];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane_index == 0) {
            for (uint coefficient = 0; coefficient < 5; ++coefficient) {
                Kx value = shared[coefficient];
                for (uint pair_index = 1; pair_index < NC_DENSE_PAIRS_PER_GROUP; ++pair_index) {
                    value = kx_add(value, shared[pair_index * 5 + coefficient]);
                }
                ulong output_index = group * 5 + coefficient;
                partials[2 * output_index] = value.c0;
                partials[2 * output_index + 1] = value.c1;
            }
        }
        return;
    }
    if (pair < (live_table_len + 1) / 2) {
        ulong index = 2 * pair;
        bool hi_active = index + 1 < live_table_len;
        Kx e0 = load_k(eq_table, index);
        Kx e_at_one = load_k(eq_table, index + 1);
        Kx inner[4] = {Kx{0, 0}, Kx{0, 0}, Kx{0, 0}, Kx{0, 0}};
        ulong start_lo = (index * width) % RING_DEGREE;
        ulong start_hi = ((index + 1) * width) % RING_DEGREE;
        for (ulong witness = 0; witness < witness_count; ++witness) {
            ulong witness_base = witness * values_per_witness;
            if (2 * width <= RING_DEGREE) {
                for (ulong slot = 0; slot < width; ++slot) {
                    ulong ring_lane = (start_lo + slot) % RING_DEGREE;
                    Kx weight = load_k(weights, witness * RING_DEGREE + ring_lane);
                    Kx a = load_k(digit_values, witness_base + index * width + slot);
                    nc_accumulate_low_window_constraint(inner, weight, a);
                }
                if (hi_active) {
                    for (ulong slot = 0; slot < width; ++slot) {
                        ulong ring_lane = (start_hi + slot) % RING_DEGREE;
                        Kx weight = load_k(weights, witness * RING_DEGREE + ring_lane);
                        Kx hi = load_k(digit_values, witness_base + (index + 1) * width + slot);
                        nc_accumulate_high_window_constraint(inner, weight, hi);
                    }
                }
            } else {
                for (ulong ring_lane = 0; ring_lane < RING_DEGREE; ++ring_lane) {
                    Kx weight = load_k(weights, witness * RING_DEGREE + ring_lane);
                    ulong slot_lo = (ring_lane + RING_DEGREE - start_lo) % RING_DEGREE;
                    ulong slot_hi = (ring_lane + RING_DEGREE - start_hi) % RING_DEGREE;
                    bool has_lo = slot_lo < width;
                    bool has_hi = hi_active && slot_hi < width;
                    if (has_lo) {
                        Kx a = load_k(digit_values, witness_base + index * width + slot_lo);
                        if (has_hi) {
                            Kx hi = load_k(digit_values, witness_base + (index + 1) * width + slot_hi);
                            nc_accumulate_digit_constraint(inner, weight, a, kx_sub(hi, a));
                        } else {
                            nc_accumulate_low_window_constraint(inner, weight, a);
                        }
                    } else if (has_hi) {
                        Kx hi = load_k(digit_values, witness_base + (index + 1) * width + slot_hi);
                        nc_accumulate_high_window_constraint(inner, weight, hi);
                    }
                }
            }
        }
        nc_multiply_cubic_by_eq(local, e0, e_at_one, inner);
    }
    for (uint coefficient = 0; coefficient < 5; ++coefficient) {
        local[coefficient] = nc_simd_reduce(local[coefficient]);
        if (simd_lane == 0) {
            shared[simd_group * 5 + coefficient] = local[coefficient];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane_index == 0) {
        for (uint coefficient = 0; coefficient < 5; ++coefficient) {
            Kx value = kx_add(shared[coefficient], shared[5 + coefficient]);
            ulong output_index = group * 5 + coefficient;
            partials[2 * output_index] = value.c0;
            partials[2 * output_index + 1] = value.c1;
        }
    }
}
