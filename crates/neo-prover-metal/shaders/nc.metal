// NC table folding and sumcheck kernels. Common field arithmetic remains in goldilocks.metal.

constant ushort NC_SIMD_WIDTH = 32;
constant ushort NC_DENSE_PAIRS_PER_GROUP = 2;
constant ushort NC_MASK_DENSE_CROSSOVER = 64;

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
    ulong active_witness = index / half_rows;
    ulong out_row = index % half_rows;
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
    bool output_dense = dense || 2 * width > RING_DEGREE;
    ulong half_rows = (rows + 1) / 2;
    ulong input_per_witness = dense ? rows * RING_DEGREE : rows * width;
    ulong output_width = output_dense ? RING_DEGREE : 2 * width;
    ulong output_per_witness = half_rows * output_width;
    ulong witness = index / output_per_witness;
    ulong within = index % output_per_witness;
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
        } else if (hi_row < rows) {
            hi = load_k(input, input_base + hi_row * width + slot - width);
        }
    } else if (dense) {
        lo = load_k(input, input_base + lo_row * RING_DEGREE + slot);
        if (hi_row < rows) {
            hi = load_k(input, input_base + hi_row * RING_DEGREE + slot);
        }
    } else {
        ulong start_lo = (lo_row * width) % RING_DEGREE;
        ulong lo_slot = (slot + RING_DEGREE - start_lo) % RING_DEGREE;
        if (lo_slot < width) {
            lo = load_k(input, input_base + lo_row * width + lo_slot);
        }
        if (hi_row < rows) {
            ulong start_hi = (hi_row * width) % RING_DEGREE;
            ulong hi_slot = (slot + RING_DEGREE - start_hi) % RING_DEGREE;
            if (hi_slot < width) {
                hi = load_k(input, input_base + hi_row * width + hi_slot);
            }
        }
    }
    Kx folded = kx_add(lo, kx_mul(challenge, kx_sub(hi, lo)));
    output[2 * index] = folded.c0;
    output[2 * index + 1] = folded.c1;
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
    ulong values_per_witness = output_rows * RING_DEGREE;
    ulong active_witness = index / values_per_witness;
    ulong within = index % values_per_witness;
    if (active_witness >= witness_count) {
        return;
    }
    ulong output_row = within / RING_DEGREE;
    ulong ring_lane = within % RING_DEGREE;
    ulong source_base = output_row * NC_MASK_DENSE_CROSSOVER;
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
    if (slot + RING_DEGREE < NC_MASK_DENSE_CROSSOVER) {
        ulong second = slot + RING_DEGREE;
        value = kx_add(
            value,
            nc_mask_basis_digit(
                masks,
                witness,
                blocks,
                active_rows,
                source_base + second,
                load_k(basis, second)));
    }
    output[2 * index] = value.c0;
    output[2 * index + 1] = value.c1;
}

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
    ulong rows = shape[0];
    ulong witness_count = shape[1];
    ulong width = shape[2];
    ulong blocks = mask_shape[2];
    ulong active_rows = mask_shape[3];
    threadgroup Kx shared[2 * 5];
    Kx local[5] = {Kx{0, 0}, Kx{0, 0}, Kx{0, 0}, Kx{0, 0}, Kx{0, 0}};
    if (pair < rows / 2) {
        ulong index = 2 * pair;
        Kx e0 = load_k(eq_table, index);
        Kx e1 = kx_sub(load_k(eq_table, index + 1), e0);
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
                    for (ulong ring_lane = 0; ring_lane < RING_DEGREE; ++ring_lane) {
                    ulong lo_slot = (ring_lane + RING_DEGREE - lo_start) % RING_DEGREE;
                    ulong hi_slot = (ring_lane + RING_DEGREE - hi_start) % RING_DEGREE;
                    bool has_lo = lo_slot < width;
                    bool has_hi = hi_slot < width;
                    if (!has_lo && !has_hi) {
                        continue;
                    }
                    Kx a = has_lo
                        ? nc_mask_basis_digit(
                            masks,
                            witness,
                            blocks,
                            active_rows,
                            lo_base + lo_slot,
                            load_k(basis, lo_slot))
                        : Kx{0, 0};
                    Kx hi = has_hi
                        ? nc_mask_basis_digit(
                            masks,
                            witness,
                            blocks,
                            active_rows,
                            hi_base + hi_slot,
                            load_k(basis, hi_slot))
                        : Kx{0, 0};
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
        local[0] = kx_mul(e0, inner[0]);
        local[1] = kx_add(kx_mul(e0, inner[1]), kx_mul(e1, inner[0]));
        local[2] = kx_add(kx_mul(e0, inner[2]), kx_mul(e1, inner[1]));
        local[3] = kx_add(kx_mul(e0, inner[3]), kx_mul(e1, inner[2]));
        local[4] = kx_mul(e1, inner[3]);
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
    ulong table_len = shape[0];
    ulong witness_count = shape[1];
    ulong width = shape[2];
    bool dense = shape[3] != 0;
    ulong values_per_witness = shape[4];
    threadgroup Kx shared[2 * 5];
    Kx local[5] = {Kx{0, 0}, Kx{0, 0}, Kx{0, 0}, Kx{0, 0}, Kx{0, 0}};
    if (dense) {
        ulong dense_pair = (ulong)group * NC_DENSE_PAIRS_PER_GROUP + simd_group;
        if (dense_pair < table_len / 2) {
            ulong index = 2 * dense_pair;
            Kx inner[4] = {Kx{0, 0}, Kx{0, 0}, Kx{0, 0}, Kx{0, 0}};
            ulong terms = witness_count * RING_DEGREE;
            for (ulong term = simd_lane; term < terms; term += NC_SIMD_WIDTH) {
                ulong witness = term / RING_DEGREE;
                ulong ring_lane = term - witness * RING_DEGREE;
                ulong witness_base = witness * values_per_witness;
                Kx weight = load_k(weights, witness * RING_DEGREE + ring_lane);
                Kx a = load_k(digit_values, witness_base + index * RING_DEGREE + ring_lane);
                Kx hi = load_k(digit_values, witness_base + (index + 1) * RING_DEGREE + ring_lane);
                nc_accumulate_digit_constraint(inner, weight, a, kx_sub(hi, a));
            }
            Kx e0 = load_k(eq_table, index);
            Kx e1 = kx_sub(load_k(eq_table, index + 1), e0);
            local[0] = kx_mul(e0, inner[0]);
            local[1] = kx_add(kx_mul(e0, inner[1]), kx_mul(e1, inner[0]));
            local[2] = kx_add(kx_mul(e0, inner[2]), kx_mul(e1, inner[1]));
            local[3] = kx_add(kx_mul(e0, inner[3]), kx_mul(e1, inner[2]));
            local[4] = kx_mul(e1, inner[3]);
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
        return;
    }
    if (pair < table_len / 2) {
        ulong index = 2 * pair;
        Kx e0 = load_k(eq_table, index);
        Kx e1 = kx_sub(load_k(eq_table, index + 1), e0);
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
                for (ulong slot = 0; slot < width; ++slot) {
                    ulong ring_lane = (start_hi + slot) % RING_DEGREE;
                    Kx weight = load_k(weights, witness * RING_DEGREE + ring_lane);
                    Kx hi = load_k(digit_values, witness_base + (index + 1) * width + slot);
                    nc_accumulate_high_window_constraint(inner, weight, hi);
                }
            } else {
                for (ulong ring_lane = 0; ring_lane < RING_DEGREE; ++ring_lane) {
                    Kx weight = load_k(weights, witness * RING_DEGREE + ring_lane);
                    ulong slot_lo = (ring_lane + RING_DEGREE - start_lo) % RING_DEGREE;
                    ulong slot_hi = (ring_lane + RING_DEGREE - start_hi) % RING_DEGREE;
                    if (slot_lo < width) {
                        Kx a = load_k(digit_values, witness_base + index * width + slot_lo);
                        if (slot_hi < width) {
                            Kx hi = load_k(digit_values, witness_base + (index + 1) * width + slot_hi);
                            nc_accumulate_digit_constraint(inner, weight, a, kx_sub(hi, a));
                        } else {
                            nc_accumulate_low_window_constraint(inner, weight, a);
                        }
                    } else if (slot_hi < width) {
                        Kx hi = load_k(digit_values, witness_base + (index + 1) * width + slot_hi);
                        nc_accumulate_high_window_constraint(inner, weight, hi);
                    }
                }
            }
        }
        local[0] = kx_mul(e0, inner[0]);
        local[1] = kx_add(kx_mul(e0, inner[1]), kx_mul(e1, inner[0]));
        local[2] = kx_add(kx_mul(e0, inner[2]), kx_mul(e1, inner[1]));
        local[3] = kx_add(kx_mul(e0, inner[3]), kx_mul(e1, inner[2]));
        local[4] = kx_mul(e1, inner[3]);
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
