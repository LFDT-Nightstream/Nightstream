// One-joint padded-row PiCCS evaluator.
// The host owns Fiat-Shamir and checks every returned round message.

constant ulong JOINT_INVERSE_TWO = 0x7fffffff80000001ul;

inline ulong joint_mask_value(
    device const ulong *masks,
    ulong blocks,
    ulong witness,
    ulong column,
    ulong magnitudes) {
    ulong block = column / RING_DEGREE;
    ulong lane = column % RING_DEGREE;
    ulong bit = 1ul << lane;
    ulong base = 2 * magnitudes * (witness * blocks + block);
    for (ulong magnitude = 1; magnitude <= magnitudes; ++magnitude) {
        if ((masks[base + 2 * (magnitude - 1)] & bit) != 0) {
            return magnitude;
        }
        if ((masks[base + 2 * (magnitude - 1) + 1] & bit) != 0) {
            return gl_sub(0, magnitude);
        }
    }
    return 0;
}

kernel void joint_expand_mask_assignments_f(
    device const ulong *masks [[buffer(0)]],
    device const ulong *shape [[buffer(1)]],
    device ulong *output [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    ulong blocks = shape[0];
    ulong source_count = shape[1];
    ulong width = shape[2];
    if ((ulong)index >= source_count * width) {
        return;
    }
    ulong source = (ulong)index / width;
    ulong column = (ulong)index % width;
    output[index] = joint_mask_value(masks, blocks, source, column, shape[3]);
}

kernel void joint_build_application_tables(
    device const uchar *row_offsets [[buffer(0)]],
    device const uint *row_blocks [[buffer(1)]],
    device const uint *dense_offsets [[buffer(2)]],
    device const uchar *dense_locals [[buffer(3)]],
    device const ulong *dense_coefficients [[buffer(4)]],
    device const uchar *geometric_row_offsets [[buffer(5)]],
    device const ulong *geometric_runs [[buffer(6)]],
    device const ulong *assignments [[buffer(7)]],
    device const ulong *shape [[buffer(8)]],
    device ulong *output [[buffer(9)]],
    device const uint2 *dense_row_blocks [[buffer(10)]],
    uint row [[thread_position_in_grid]]) {
    ulong rows = shape[0];
    ulong blocks = shape[1];
    ulong n_eff = shape[2];
    ulong table_len = shape[3];
    ulong witness = shape[4];
    ulong output_table = shape[5];
    ulong offset_width = shape[6];
    bool identity = shape[7] != 0;
    ulong assignment_width = shape[8];
    ulong geometric_offset_width = shape[9];
    if ((ulong)row >= table_len) {
        return;
    }
    ulong value = 0;
    if (row < n_eff) {
        if (identity) {
            value = assignments[witness * assignment_width + row];
        } else {
            ulong start = compact_row_offset(row_offsets, row, offset_width);
            ulong end = compact_row_offset(row_offsets, row + 1, offset_width);
            for (ulong entry = start; entry < end; ++entry) {
                uint reference = row_blocks[entry];
                if ((reference & COMPACT_DENSE_BLOCK_TAG) == 0) {
                    ulong block = (ulong)(reference & COMPACT_SINGLE_BLOCK_MASK);
                    ulong local = (ulong)((reference >> COMPACT_SINGLE_LOCAL_SHIFT) & COMPACT_SINGLE_LOCAL_MASK);
                    ulong column = block * RING_DEGREE + local;
                    ulong input = assignments[witness * assignment_width + column];
                    value = (reference & COMPACT_NEGATIVE_BLOCK_TAG) == 0
                        ? gl_add(value, input)
                        : gl_sub(value, input);
                } else {
                    uint2 block = dense_row_blocks[reference & COMPACT_DENSE_INDEX_MASK];
                    uint dense = block.y;
                    for (uint coefficient = dense_offsets[dense]; coefficient < dense_offsets[dense + 1]; ++coefficient) {
                        ulong column = (ulong)block.x * RING_DEGREE + (ulong)dense_locals[coefficient];
                        ulong input = assignments[witness * assignment_width + column];
                        if (input != 0) {
                            value = gl_add(
                                value,
                                gl_mul(gl_from_word(dense_coefficients[coefficient]), input));
                        }
                    }
                }
            }
            if (geometric_offset_width != 0) {
                ulong geometric_start = compact_row_offset(geometric_row_offsets, row, geometric_offset_width);
                ulong geometric_end = compact_row_offset(geometric_row_offsets, row + 1, geometric_offset_width);
                for (ulong run = geometric_start; run < geometric_end; ++run) {
                    ulong packed = geometric_runs[3 * run];
                    ulong column = packed & 0xfffffffful;
                    ulong run_end = column + (packed >> 32);
                    ulong coefficient = gl_from_word(geometric_runs[3 * run + 1]);
                    ulong ratio = gl_from_word(geometric_runs[3 * run + 2]);
                    for (; column < run_end; ++column) {
                        ulong input = assignments[witness * assignment_width + column];
                        if (input != 0 && coefficient != 0) {
                            value = gl_add(value, gl_mul(coefficient, input));
                        }
                        coefficient = gl_mul(coefficient, ratio);
                    }
                }
            }
        }
    }
    output[output_table * table_len + row] = value;
}

// Canonical selective F-prime lowers each seeded Phi81 map as the R1CS row
// A(z) * 1 = C(z). For an honest satisfying witness, the expensive seeded
// A-table entry is therefore exactly the already-built C-table entry. Rows
// from inactive selective arms are gated by a zero selector, so the same
// substitution is harmless there.
kernel void joint_copy_seeded_satisfied_rows(
    device const ulong *group_headers [[buffer(0)]],
    device const ulong *shape [[buffer(1)]],
    device ulong *tables [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    ulong group = (ulong)index / RING_DEGREE;
    ulong coordinate = (ulong)index % RING_DEGREE;
    device const ulong *header = group_headers + 4 * group;
    ulong table_len = shape[0];
    ulong row = header[1] + coordinate;
    if (row >= table_len) {
        return;
    }
    ulong table_base = shape[1];
    ulong target = table_base + header[0];
    ulong source = table_base + shape[2];
    tables[target * table_len + row] = tables[source * table_len + row];
}

kernel void joint_zero_words(
    device ulong *words [[buffer(0)]],
    uint index [[thread_position_in_grid]]) {
    words[index] = 0;
}

kernel void joint_fold_base_tables(
    device const ulong *tables [[buffer(0)]],
    device const ulong *challenge_words [[buffer(1)]],
    device const ulong *shape [[buffer(2)]],
    device ulong *output [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    ulong table_len = shape[0];
    ulong table_count = shape[1];
    ulong folded_len = (table_len + 1) / 2;
    if ((ulong)index >= table_count * folded_len) {
        return;
    }
    ulong table = index / folded_len;
    ulong pair = index % folded_len;
    ulong input = table * table_len + 2 * pair;
    Kx left = Kx{gl_from_word(tables[input]), 0};
    Kx right = 2 * pair + 1 < table_len
        ? Kx{gl_from_word(tables[input + 1]), 0}
        : Kx{0, 0};
    Kx challenge = Kx{gl_from_word(challenge_words[0]), gl_from_word(challenge_words[1])};
    Kx folded = kx_add(left, kx_mul(challenge, kx_sub(right, left)));
    output[2 * index] = folded.c0;
    output[2 * index + 1] = folded.c1;
}

kernel void joint_fold_k_tables(
    device const ulong *tables [[buffer(0)]],
    device const ulong *challenge_words [[buffer(1)]],
    device const ulong *shape [[buffer(2)]],
    device ulong *output [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    ulong table_len = shape[0];
    ulong table_count = shape[1];
    ulong folded_len = (table_len + 1) / 2;
    if ((ulong)index >= table_count * folded_len) {
        return;
    }
    ulong table = (ulong)index / folded_len;
    ulong pair = (ulong)index % folded_len;
    ulong input = table * table_len + 2 * pair;
    Kx left = load_k(tables, input);
    Kx right = 2 * pair + 1 < table_len ? load_k(tables, input + 1) : Kx{0, 0};
    Kx challenge = Kx{gl_from_word(challenge_words[0]), gl_from_word(challenge_words[1])};
    Kx folded = kx_add(left, kx_mul(challenge, kx_sub(right, left)));
    output[2 * index] = folded.c0;
    output[2 * index + 1] = folded.c1;
}

kernel void joint_fold_mask_assignments(
    device const ulong *masks [[buffer(0)]],
    device const ulong *challenge_words [[buffer(1)]],
    device const ulong *shape [[buffer(2)]],
    device ulong *output [[buffer(3)]],
    device const ulong *assignment_sources [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    ulong table_len = shape[0];
    ulong source_count = shape[1];
    ulong blocks = shape[2];
    ulong assignment_width = shape[3];
    ulong magnitudes = shape[4];
    ulong folded_len = (table_len + 1) / 2;
    if ((ulong)index >= source_count * folded_len) {
        return;
    }
    ulong source = index / folded_len;
    ulong mask_source = assignment_sources[source];
    ulong pair = index % folded_len;
    ulong low_index = 2 * pair;
    ulong high_index = low_index + 1;
    Kx left = Kx{
        low_index < assignment_width ? joint_mask_value(masks, blocks, mask_source, low_index, magnitudes) : 0,
        0};
    Kx right = Kx{
        high_index < assignment_width ? joint_mask_value(masks, blocks, mask_source, high_index, magnitudes) : 0,
        0};
    Kx challenge = Kx{gl_from_word(challenge_words[0]), gl_from_word(challenge_words[1])};
    Kx folded = kx_add(left, kx_mul(challenge, kx_sub(right, left)));
    output[2 * index] = folded.c0;
    output[2 * index + 1] = folded.c1;
}

inline Kx joint_load_table(
    device const ulong *tables,
    ulong table_len,
    ulong table,
    ulong index,
    bool base_field) {
    if (index >= table_len) {
        return Kx{0, 0};
    }
    ulong position = table * table_len + index;
    return base_field ? Kx{gl_from_word(tables[position]), 0} : load_k(tables, position);
}

inline Kx joint_assignment_value(
    device const ulong *assignments,
    device const ulong *assignment_sources,
    ulong table_len,
    ulong source,
    ulong index,
    bool base_round,
    ulong blocks,
    ulong magnitudes,
    ulong assignment_width,
    ulong assignment_len) {
    if (base_round) {
        ulong mask_source = assignment_sources[source];
        return Kx{
            index < assignment_width ? joint_mask_value(assignments, blocks, mask_source, index, magnitudes) : 0,
            0};
    }
    return index < assignment_len ? load_k(assignments, source * assignment_len + index) : Kx{0, 0};
}

inline ulong joint_half_f(ulong value) {
    return gl_mul(value, JOINT_INVERSE_TWO);
}

inline Kx joint_mul_f(Kx value, ulong scalar) {
    return Kx{gl_mul(value.c0, scalar), gl_mul(value.c1, scalar)};
}

inline ulong joint_signed_root_f(int root) {
    return root < 0 ? gl_sub(0, (ulong)(-root)) : (ulong)root;
}

inline Kx joint_signed_root_k(int root) {
    return Kx{joint_signed_root_f(root), 0};
}

inline ulong joint_range_product_f(ulong value, uint base) {
    ulong product = 1;
    int bound = (int)base - 1;
    for (int root = -bound; root <= bound; ++root) {
        product = gl_mul(product, gl_sub(value, joint_signed_root_f(root)));
    }
    return product;
}

inline Kx joint_range_product_k(Kx value, uint base) {
    Kx product = Kx{1, 0};
    int bound = (int)base - 1;
    for (int root = -bound; root <= bound; ++root) {
        product = kx_mul(product, kx_sub(value, joint_signed_root_k(root)));
    }
    return product;
}

inline Kx joint_half_k(Kx value) {
    return joint_mul_f(value, JOINT_INVERSE_TWO);
}

inline ulong joint_fixed_borrow_step_f(uint bound, ulong digit, ulong borrow) {
    ulong negative = joint_half_f(gl_mul(digit, gl_sub(digit, 1)));
    ulong positive = gl_add(digit, negative);
    ulong zero = gl_sub(gl_sub(1, digit), gl_add(negative, negative));
    if (bound == 0) {
        return gl_sub(1, gl_mul(negative, gl_sub(1, borrow)));
    }
    if (bound == 1) {
        return gl_add(positive, gl_mul(zero, borrow));
    }
    return gl_mul(positive, borrow);
}

inline Kx joint_fixed_borrow_step_k(uint bound, Kx digit, Kx borrow) {
    Kx one = Kx{1, 0};
    Kx negative = joint_half_k(kx_mul(digit, kx_sub(digit, one)));
    Kx positive = kx_add(digit, negative);
    Kx zero = kx_sub(kx_sub(one, digit), kx_add(negative, negative));
    if (bound == 0) {
        return kx_sub(one, kx_mul(negative, kx_sub(one, borrow)));
    }
    if (bound == 1) {
        return kx_add(positive, kx_mul(zero, borrow));
    }
    return kx_mul(positive, borrow);
}

inline ulong joint_selective_polynomial_f(thread const ulong *x) {
    ulong bit_square = gl_mul(x[0], x[0]);
    ulong sbox = gl_sbox(x[5]);
    ulong centered_square = gl_mul(x[6], x[6]);
    ulong general = gl_sub(bit_square, x[0]);
    general = gl_add(general, gl_mul(x[2], x[3]));
    general = gl_sub(general, x[4]);
    general = gl_add(general, sbox);
    general = gl_add(general, gl_mul(centered_square, x[6]));
    general = gl_sub(general, x[6]);
    ulong result = gl_mul(x[1], general);

    ulong centered_delta = gl_sub(centered_square, 1);
    ulong correction = gl_mul(
        centered_delta,
        gl_sub(gl_mul(centered_square, centered_delta), x[6]));
    ulong a_square = gl_mul(x[2], x[2]);
    ulong a_delta = gl_sub(a_square, 1);
    correction = gl_sub(
        correction,
        gl_mul(7, gl_mul(a_square, gl_mul(a_delta, a_delta))));
    result = gl_add(result, gl_mul(gl_mul(x[1], x[7]), correction));

    ulong evaluation = gl_sub(0, x[4]);
    evaluation = gl_add(evaluation, gl_mul(x[0], x[2]));
    evaluation = gl_add(evaluation, gl_mul(x[3], x[5]));
    evaluation = gl_add(evaluation, gl_mul(x[6], x[8]));
    evaluation = gl_add(evaluation, gl_mul(x[9], x[10]));
    evaluation = gl_add(evaluation, gl_mul(x[11], x[12]));
    result = gl_add(result, gl_mul(x[7], evaluation));

    for (uint bound = 0; bound < 5; ++bound) {
        ulong first = joint_fixed_borrow_step_f(bound % 3, x[6], x[0]);
        ulong second = joint_fixed_borrow_step_f(bound / 3, x[2], first);
        ulong relation = gl_sub(x[4], second);
        result = gl_add(result, gl_mul(gl_mul(x[1], x[8 + bound]), relation));
    }
    return result;
}

inline Kx joint_selective_polynomial_k(thread const Kx *x) {
    Kx bit_square = kx_mul(x[0], x[0]);
    Kx sbox_square = kx_mul(x[5], x[5]);
    Kx sbox_fourth = kx_mul(sbox_square, sbox_square);
    Kx sbox = kx_mul(kx_mul(sbox_fourth, sbox_square), x[5]);
    Kx centered_square = kx_mul(x[6], x[6]);
    Kx general = kx_sub(bit_square, x[0]);
    general = kx_add(general, kx_mul(x[2], x[3]));
    general = kx_sub(general, x[4]);
    general = kx_add(general, sbox);
    general = kx_add(general, kx_mul(centered_square, x[6]));
    general = kx_sub(general, x[6]);
    Kx result = kx_mul(x[1], general);

    Kx one = Kx{1, 0};
    Kx centered_delta = kx_sub(centered_square, one);
    Kx correction = kx_mul(
        centered_delta,
        kx_sub(kx_mul(centered_square, centered_delta), x[6]));
    Kx a_square = kx_mul(x[2], x[2]);
    Kx a_delta = kx_sub(a_square, one);
    correction = kx_sub(
        correction,
        joint_mul_f(kx_mul(a_square, kx_mul(a_delta, a_delta)), 7));
    result = kx_add(result, kx_mul(kx_mul(x[1], x[7]), correction));

    Kx evaluation = kx_sub(Kx{0, 0}, x[4]);
    evaluation = kx_add(evaluation, kx_mul(x[0], x[2]));
    evaluation = kx_add(evaluation, kx_mul(x[3], x[5]));
    evaluation = kx_add(evaluation, kx_mul(x[6], x[8]));
    evaluation = kx_add(evaluation, kx_mul(x[9], x[10]));
    evaluation = kx_add(evaluation, kx_mul(x[11], x[12]));
    result = kx_add(result, kx_mul(x[7], evaluation));

    for (uint bound = 0; bound < 5; ++bound) {
        Kx first = joint_fixed_borrow_step_k(bound % 3, x[6], x[0]);
        Kx second = joint_fixed_borrow_step_k(bound / 3, x[2], first);
        Kx relation = kx_sub(x[4], second);
        result = kx_add(result, kx_mul(kx_mul(x[1], x[8 + bound]), relation));
    }
    return result;
}

inline Kx joint_equality_suffix(
    device const ulong *tables,
    ulong chunks_per_round,
    ulong round,
    ulong pair) {
    Kx value = Kx{1, 0};
    ulong base = round * chunks_per_round * 256;
    for (ulong chunk = 0; chunk < chunks_per_round; ++chunk) {
        ulong index = (pair >> (8 * chunk)) & 255;
        value = kx_mul(value, load_k(tables, base + chunk * 256 + index));
    }
    return value;
}

// Exact, factored evaluator for the selective low-norm F-prime polynomial.
// It returns the canonical values at 0..degree instead of first expanding
// the 66-term polynomial into monomial coefficients.
kernel void joint_selective_round_partials(
    device const ulong *application_tables [[buffer(0)]],
    device const ulong *assignments_or_masks [[buffer(1)]],
    device const ulong *common_tables [[buffer(2)]],
    device const ulong *shape [[buffer(3)]],
    device const ulong *weights [[buffer(4)]],
    device ulong *partials [[buffer(7)]],
    device const ulong *equality_chunks [[buffer(8)]],
    device const ulong *prior_equality_chunks [[buffer(9)]],
    device const ulong *assignment_sources [[buffer(10)]],
    uint pair [[thread_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    ulong table_len = shape[0];
    ulong fresh_count = shape[1];
    ulong matrix_count = shape[2];
    ulong assignment_count = shape[3];
    uint coefficient_count = (uint)shape[4];
    bool base_round = shape[6] != 0;
    ulong blocks = shape[7];
    ulong assignment_width = shape[8];
    ulong active_len = shape[9];
    ulong application_len = shape[10];
    ulong assignment_len = shape[11];
    bool has_prior = shape[12] != 0;
    ulong common_len = shape[13];
    ulong chunks_per_round = shape[14];
    ulong round = shape[15];
    Kx alpha_low_factor = Kx{gl_from_word(shape[16]), gl_from_word(shape[17])};
    Kx alpha_slope_factor = Kx{gl_from_word(shape[18]), gl_from_word(shape[19])};
    Kx prior_low_factor = Kx{gl_from_word(shape[20]), gl_from_word(shape[21])};
    Kx prior_slope_factor = Kx{gl_from_word(shape[22]), gl_from_word(shape[23])};
    uint range_base = (uint)shape[24];
    threadgroup Kx shared[SUMCHECK_REDUCTION_THREADS * SUMCHECK_MAX_COEFFS];
    Kx values[SUMCHECK_MAX_COEFFS];
    for (uint point = 0; point < SUMCHECK_MAX_COEFFS; ++point) {
        values[point] = Kx{0, 0};
    }

    ulong pairs = (active_len + 1) / 2;
    if ((ulong)pair < pairs) {
        ulong low_index = 2 * (ulong)pair;
        ulong high_index = low_index + 1;
        for (ulong source = 0; source < fresh_count; ++source) {
            Kx source_weight = load_k(weights, source);
            if (base_round) {
                ulong x[13];
                ulong slope[13];
                for (ulong matrix = 0; matrix < matrix_count; ++matrix) {
                    ulong table = source * matrix_count + matrix;
                    ulong offset = table * application_len;
                    x[matrix] = low_index < application_len
                        ? gl_from_word(application_tables[offset + low_index])
                        : 0;
                    ulong high = high_index < application_len
                        ? gl_from_word(application_tables[offset + high_index])
                        : 0;
                    slope[matrix] = gl_sub(high, x[matrix]);
                }
                for (uint point = 0; point < coefficient_count; ++point) {
                    values[point] = kx_add(
                        values[point],
                        joint_mul_f(source_weight, joint_selective_polynomial_f(x)));
                    for (ulong matrix = 0; matrix < matrix_count; ++matrix) {
                        x[matrix] = gl_add(x[matrix], slope[matrix]);
                    }
                }
            } else {
                Kx x[13];
                Kx slope[13];
                for (ulong matrix = 0; matrix < matrix_count; ++matrix) {
                    ulong table = source * matrix_count + matrix;
                    x[matrix] = low_index < application_len
                        ? load_k(application_tables, table * application_len + low_index)
                        : Kx{0, 0};
                    Kx high = high_index < application_len
                        ? load_k(application_tables, table * application_len + high_index)
                        : Kx{0, 0};
                    slope[matrix] = kx_sub(high, x[matrix]);
                }
                for (uint point = 0; point < coefficient_count; ++point) {
                    values[point] = kx_add(
                        values[point],
                        kx_mul(source_weight, joint_selective_polynomial_k(x)));
                    for (ulong matrix = 0; matrix < matrix_count; ++matrix) {
                        x[matrix] = kx_add(x[matrix], slope[matrix]);
                    }
                }
            }
        }

        for (ulong source = 0; source < assignment_count; ++source) {
            Kx source_weight = load_k(weights, fresh_count + source);
            if (base_round) {
                ulong mask_source = assignment_sources[source];
                ulong value = low_index < assignment_len
                    ? joint_mask_value(assignments_or_masks, blocks, mask_source, low_index, range_base - 1)
                    : 0;
                ulong high = high_index < assignment_len
                    ? joint_mask_value(assignments_or_masks, blocks, mask_source, high_index, range_base - 1)
                    : 0;
                ulong slope = gl_sub(high, value);
                for (uint point = 0; point < coefficient_count; ++point) {
                    ulong norm = joint_range_product_f(value, range_base);
                    values[point] = kx_add(values[point], joint_mul_f(source_weight, norm));
                    value = gl_add(value, slope);
                }
            } else {
                Kx value = low_index < assignment_len
                    ? load_k(assignments_or_masks, source * assignment_len + low_index)
                    : Kx{0, 0};
                Kx high = high_index < assignment_len
                    ? load_k(assignments_or_masks, source * assignment_len + high_index)
                    : Kx{0, 0};
                Kx slope = kx_sub(high, value);
                for (uint point = 0; point < coefficient_count; ++point) {
                    Kx norm = joint_range_product_k(value, range_base);
                    values[point] = kx_add(values[point], kx_mul(source_weight, norm));
                    value = kx_add(value, slope);
                }
            }
        }

        Kx equality_suffix = joint_equality_suffix(equality_chunks, chunks_per_round, round, pair);
        Kx equality = kx_mul(alpha_low_factor, equality_suffix);
        Kx equality_slope = kx_mul(alpha_slope_factor, equality_suffix);
        Kx prior_suffix = has_prior
            ? joint_equality_suffix(prior_equality_chunks, chunks_per_round, round, pair)
            : Kx{0, 0};
        Kx prior = kx_mul(prior_low_factor, prior_suffix);
        Kx prior_slope = kx_mul(prior_slope_factor, prior_suffix);
        Kx carried = has_prior && low_index < common_len
            ? load_k(common_tables, low_index)
            : Kx{0, 0};
        Kx carried_high = has_prior && high_index < common_len
            ? load_k(common_tables, high_index)
            : Kx{0, 0};
        Kx carried_slope = kx_sub(carried_high, carried);
        for (uint point = 0; point < coefficient_count; ++point) {
            values[point] = kx_add(kx_mul(equality, values[point]), kx_mul(prior, carried));
            equality = kx_add(equality, equality_slope);
            prior = kx_add(prior, prior_slope);
            carried = kx_add(carried, carried_slope);
        }
    }

    for (uint point = 0; point < coefficient_count; ++point) {
        shared[lane * SUMCHECK_MAX_COEFFS + point] = values[point];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = SUMCHECK_REDUCTION_THREADS / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            for (uint point = 0; point < coefficient_count; ++point) {
                uint destination = lane * SUMCHECK_MAX_COEFFS + point;
                uint source = (lane + stride) * SUMCHECK_MAX_COEFFS + point;
                shared[destination] = kx_add(shared[destination], shared[source]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        for (uint point = 0; point < coefficient_count; ++point) {
            ulong output = group * coefficient_count + point;
            partials[2 * output] = shared[point].c0;
            partials[2 * output + 1] = shared[point].c1;
        }
    }
}

inline void joint_poly_mul_affine(
    thread Kx *polynomial,
    Kx constant_term,
    Kx slope,
    thread uint &degree) {
    Kx previous = Kx{0, 0};
    for (uint coefficient = 0; coefficient <= degree + 1; ++coefficient) {
        Kx old = polynomial[coefficient];
        polynomial[coefficient] = kx_add(kx_mul(constant_term, old), kx_mul(slope, previous));
        previous = old;
    }
    degree += 1;
}

kernel void joint_round_partials(
    device const ulong *application_tables [[buffer(0)]],
    device const ulong *assignments_or_masks [[buffer(1)]],
    device const ulong *common_tables [[buffer(2)]],
    device const ulong *shape [[buffer(3)]],
    device const ulong *weights [[buffer(4)]],
    device const ulong *term_headers [[buffer(5)]],
    device const ulong *term_variables [[buffer(6)]],
    device ulong *partials [[buffer(7)]],
    device const ulong *equality_chunks [[buffer(8)]],
    device const ulong *prior_equality_chunks [[buffer(9)]],
    device const ulong *assignment_sources [[buffer(10)]],
    uint pair [[thread_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    ulong table_len = shape[0];
    ulong fresh_count = shape[1];
    ulong matrix_count = shape[2];
    ulong assignment_count = shape[3];
    uint coefficient_count = (uint)shape[4];
    ulong term_count = shape[5];
    bool base_round = shape[6] != 0;
    ulong blocks = shape[7];
    ulong assignment_width = shape[8];
    ulong active_len = shape[9];
    ulong application_len = shape[10];
    ulong assignment_len = shape[11];
    bool has_prior = shape[12] != 0;
    ulong common_len = shape[13];
    ulong chunks_per_round = shape[14];
    ulong round = shape[15];
    Kx alpha_low_factor = Kx{gl_from_word(shape[16]), gl_from_word(shape[17])};
    Kx alpha_slope_factor = Kx{gl_from_word(shape[18]), gl_from_word(shape[19])};
    Kx prior_low_factor = Kx{gl_from_word(shape[20]), gl_from_word(shape[21])};
    Kx prior_slope_factor = Kx{gl_from_word(shape[22]), gl_from_word(shape[23])};
    uint range_base = (uint)shape[24];
    threadgroup Kx shared[SUMCHECK_REDUCTION_THREADS * SUMCHECK_MAX_COEFFS];
    Kx local[SUMCHECK_MAX_COEFFS];
    for (uint coefficient = 0; coefficient < SUMCHECK_MAX_COEFFS; ++coefficient) {
        local[coefficient] = Kx{0, 0};
    }

    ulong pairs = (active_len + 1) / 2;
    if ((ulong)pair < pairs) {
        ulong low_index = 2 * (ulong)pair;
        ulong high_index = low_index + 1;
        Kx inner[SUMCHECK_MAX_COEFFS];
        for (uint coefficient = 0; coefficient < SUMCHECK_MAX_COEFFS; ++coefficient) {
            inner[coefficient] = Kx{0, 0};
        }

        for (ulong source = 0; source < fresh_count; ++source) {
            Kx source_weight = load_k(weights, source);
            for (ulong term = 0; term < term_count; ++term) {
                ulong header = 3 * term;
                Kx polynomial[SUMCHECK_MAX_COEFFS];
                for (uint coefficient = 0; coefficient < SUMCHECK_MAX_COEFFS; ++coefficient) {
                    polynomial[coefficient] = Kx{0, 0};
                }
                polynomial[0] = kx_mul(Kx{gl_from_word(term_headers[header]), 0}, source_weight);
                ulong variable_start = term_headers[header + 1];
                ulong variable_count = term_headers[header + 2];
                uint degree = 0;
                for (ulong variable = 0; variable < variable_count; ++variable) {
                    ulong variable_header = 2 * (variable_start + variable);
                    ulong matrix = term_variables[variable_header];
                    uint exponent = (uint)term_variables[variable_header + 1];
                    ulong table = source * matrix_count + matrix;
                    Kx low = joint_load_table(
                        application_tables, application_len, table, low_index, base_round);
                    Kx high = joint_load_table(
                        application_tables, application_len, table, high_index, base_round);
                    Kx slope = kx_sub(high, low);
                    for (uint power = 0; power < exponent; ++power) {
                        joint_poly_mul_affine(polynomial, low, slope, degree);
                    }
                }
                for (uint coefficient = 0; coefficient <= degree; ++coefficient) {
                    inner[coefficient] = kx_add(inner[coefficient], polynomial[coefficient]);
                }
            }
        }

        for (ulong source = 0; source < assignment_count; ++source) {
            Kx low = joint_assignment_value(
                assignments_or_masks,
                assignment_sources,
                table_len,
                source,
                low_index,
                base_round,
                blocks,
                range_base - 1,
                assignment_width,
                assignment_len);
            Kx high = joint_assignment_value(
                assignments_or_masks,
                assignment_sources,
                table_len,
                source,
                high_index,
                base_round,
                blocks,
                range_base - 1,
                assignment_width,
                assignment_len);
            Kx slope = kx_sub(high, low);
            Kx polynomial[SUMCHECK_MAX_COEFFS];
            for (uint coefficient = 0; coefficient < SUMCHECK_MAX_COEFFS; ++coefficient) {
                polynomial[coefficient] = Kx{0, 0};
            }
            polynomial[0] = load_k(weights, fresh_count + source);
            uint degree = 0;
            int bound = (int)range_base - 1;
            for (int root = -bound; root <= bound; ++root) {
                joint_poly_mul_affine(polynomial, kx_sub(low, joint_signed_root_k(root)), slope, degree);
            }
            for (uint coefficient = 0; coefficient <= degree; ++coefficient) {
                inner[coefficient] = kx_add(inner[coefficient], polynomial[coefficient]);
            }
        }

        Kx equality_suffix = joint_equality_suffix(equality_chunks, chunks_per_round, round, pair);
        Kx eq_low = kx_mul(alpha_low_factor, equality_suffix);
        Kx eq_slope = kx_mul(alpha_slope_factor, equality_suffix);
        local[0] = kx_mul(eq_low, inner[0]);
        for (uint coefficient = 1; coefficient < coefficient_count; ++coefficient) {
            local[coefficient] = kx_add(
                kx_mul(eq_low, inner[coefficient]),
                kx_mul(eq_slope, inner[coefficient - 1]));
        }

        if (has_prior) {
            Kx prior_suffix = joint_equality_suffix(prior_equality_chunks, chunks_per_round, round, pair);
            Kx prior_low = kx_mul(prior_low_factor, prior_suffix);
            Kx prior_slope = kx_mul(prior_slope_factor, prior_suffix);
            Kx carried_low = low_index < common_len ? load_k(common_tables, low_index) : Kx{0, 0};
            Kx carried_high = high_index < common_len ? load_k(common_tables, high_index) : Kx{0, 0};
            Kx carried_slope = kx_sub(carried_high, carried_low);
            local[0] = kx_add(local[0], kx_mul(prior_low, carried_low));
            local[1] = kx_add(
                local[1],
                kx_add(kx_mul(prior_low, carried_slope), kx_mul(prior_slope, carried_low)));
            local[2] = kx_add(local[2], kx_mul(prior_slope, carried_slope));
        }
    }

    for (uint coefficient = 0; coefficient < coefficient_count; ++coefficient) {
        shared[lane * SUMCHECK_MAX_COEFFS + coefficient] = local[coefficient];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = SUMCHECK_REDUCTION_THREADS / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            for (uint coefficient = 0; coefficient < coefficient_count; ++coefficient) {
                uint destination = lane * SUMCHECK_MAX_COEFFS + coefficient;
                uint source = (lane + stride) * SUMCHECK_MAX_COEFFS + coefficient;
                shared[destination] = kx_add(shared[destination], shared[source]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        for (uint coefficient = 0; coefficient < coefficient_count; ++coefficient) {
            ulong output = group * coefficient_count + coefficient;
            partials[2 * output] = shared[coefficient].c0;
            partials[2 * output + 1] = shared[coefficient].c1;
        }
    }
}

kernel void joint_add_identity_carried(
    device const ulong *qk [[buffer(0)]],
    device const ulong *coefficient_words [[buffer(1)]],
    device const ulong *shape [[buffer(2)]],
    device ulong *output [[buffer(3)]],
    uint row [[thread_position_in_grid]]) {
    ulong assignment_width = shape[0];
    ulong n_pad = shape[1];
    if ((ulong)row >= n_pad || (ulong)row >= assignment_width) {
        return;
    }
    Kx coefficient = load_k(coefficient_words, 0);
    Kx value = kx_add(load_k(output, row), kx_mul(coefficient, load_k(qk, row)));
    output[2 * row] = value.c0;
    output[2 * row + 1] = value.c1;
}

constant ulong JOINT_SEEDED_OUTPUT_HEADER_WORDS = 9;
constant ulong JOINT_SEEDED_WORK_HEADER_WORDS = 3;

inline ulong joint_seeded_column(
    device const ulong *header,
    device const uint *word_starts,
    ulong message_row,
    ulong message_col) {
    ulong message_cols = header[2];
    ulong word_width = header[3];
    ulong word_count = header[4];
    ulong bit_index = message_row * message_cols + message_col;
    if (bit_index >= word_count * word_width) {
        return ~0ul;
    }
    ulong word = bit_index / word_width;
    return (ulong)word_starts[header[5] + word] + bit_index % word_width;
}

kernel void joint_seeded_base_partials(
    device const ulong *output_headers [[buffer(0)]],
    device const ulong *work_headers [[buffer(1)]],
    device const uint *word_starts [[buffer(2)]],
    device const ulong *rotations [[buffer(3)]],
    device const ulong *masks [[buffer(4)]],
    device const ulong *shape [[buffer(5)]],
    device ulong *partials [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    ulong work = (ulong)index / RING_PRODUCT_COEFFICIENTS;
    ulong coefficient = (ulong)index % RING_PRODUCT_COEFFICIENTS;
    device const ulong *work_header = work_headers + work * JOINT_SEEDED_WORK_HEADER_WORDS;
    ulong output_index = work_header[0];
    device const ulong *header = output_headers + output_index * JOINT_SEEDED_OUTPUT_HEADER_WORDS;
    if (header[1] >= shape[2]) {
        partials[index] = 0;
        return;
    }
    ulong term_start = coefficient >= RING_DEGREE ? coefficient - (RING_DEGREE - 1) : 0;
    ulong term_end = min(coefficient, RING_DEGREE - 1);
    ulong value = 0;
    for (ulong message_col = work_header[1]; message_col < work_header[2]; ++message_col) {
        for (ulong message_row = term_start; message_row <= term_end; ++message_row) {
            ulong column = joint_seeded_column(header, word_starts, message_row, message_col);
            if (column == ~0ul || column >= shape[0] * RING_DEGREE) {
                continue;
            }
            ulong input = joint_mask_value(masks, shape[0], shape[1], column, shape[5]);
            if (input != 0) {
                ulong rotation = gl_from_word(
                    rotations[header[6] + message_col * RING_DEGREE + coefficient - message_row]);
                value = gl_add(value, gl_mul(input, rotation));
            }
        }
    }
    partials[index] = value;
}

inline ulong joint_seeded_base_polynomial_sum(
    device const ulong *header,
    device const ulong *partials,
    ulong coefficient) {
    ulong value = 0;
    ulong work_end = header[7] + header[8];
    for (ulong work = header[7]; work < work_end; ++work) {
        value = gl_add(value, gl_from_word(partials[work * RING_PRODUCT_COEFFICIENTS + coefficient]));
    }
    return value;
}

inline ulong joint_seeded_base_phi81(
    device const ulong *header,
    device const ulong *partials,
    ulong coefficient) {
    ulong value = joint_seeded_base_polynomial_sum(header, partials, coefficient);
    if (coefficient <= 26) {
        value = gl_sub(value, joint_seeded_base_polynomial_sum(header, partials, coefficient + 54));
        if (coefficient <= 25) {
            value = gl_add(value, joint_seeded_base_polynomial_sum(header, partials, coefficient + 81));
        }
    } else {
        value = gl_sub(value, joint_seeded_base_polynomial_sum(header, partials, coefficient + 27));
    }
    return value;
}

kernel void joint_seeded_base_reduce(
    device const ulong *output_headers [[buffer(0)]],
    device const ulong *group_headers [[buffer(1)]],
    device const uint *group_outputs [[buffer(2)]],
    device const ulong *shape [[buffer(3)]],
    device const ulong *partials [[buffer(4)]],
    device ulong *tables [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    ulong group = (ulong)index / RING_DEGREE;
    ulong coordinate = (ulong)index % RING_DEGREE;
    device const ulong *group_header = group_headers + 4 * group;
    ulong row = group_header[1] + coordinate;
    if (row >= shape[2]) {
        return;
    }
    ulong value = 0;
    ulong output_end = group_header[2] + group_header[3];
    for (ulong position = group_header[2]; position < output_end; ++position) {
        ulong output_index = group_outputs[position];
        device const ulong *header = output_headers + output_index * JOINT_SEEDED_OUTPUT_HEADER_WORDS;
        value = gl_add(value, joint_seeded_base_phi81(header, partials, coordinate));
    }
    ulong table = shape[4] + group_header[0];
    ulong destination = table * shape[3] + row;
    tables[destination] = gl_add(gl_from_word(tables[destination]), value);
}

kernel void joint_seeded_k_partials(
    device const ulong *output_headers [[buffer(0)]],
    device const ulong *work_headers [[buffer(1)]],
    device const uint *word_starts [[buffer(2)]],
    device const ulong *rotations [[buffer(3)]],
    device const ulong *qk [[buffer(4)]],
    device const ulong *mat_coeffs [[buffer(5)]],
    device const ulong *shape [[buffer(6)]],
    device ulong *partials [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
    ulong work = (ulong)index / RING_PRODUCT_COEFFICIENTS;
    ulong coefficient = (ulong)index % RING_PRODUCT_COEFFICIENTS;
    device const ulong *work_header = work_headers + work * JOINT_SEEDED_WORK_HEADER_WORDS;
    ulong output_index = work_header[0];
    device const ulong *header = output_headers + output_index * JOINT_SEEDED_OUTPUT_HEADER_WORDS;
    if (header[1] >= shape[4]) {
        partials[2 * index] = 0;
        partials[2 * index + 1] = 0;
        return;
    }
    ulong term_start = coefficient >= RING_DEGREE ? coefficient - (RING_DEGREE - 1) : 0;
    ulong term_end = min(coefficient, RING_DEGREE - 1);
    Kx value = Kx{0, 0};
    for (ulong message_col = work_header[1]; message_col < work_header[2]; ++message_col) {
        for (ulong message_row = term_start; message_row <= term_end; ++message_row) {
            ulong column = joint_seeded_column(header, word_starts, message_row, message_col);
            if (column == ~0ul || column >= shape[1] * RING_DEGREE) {
                continue;
            }
            ulong rotation = gl_from_word(
                rotations[header[6] + message_col * RING_DEGREE + coefficient - message_row]);
            Kx input = load_k(qk, column);
            value = kx_add(value, Kx{gl_mul(input.c0, rotation), gl_mul(input.c1, rotation)});
        }
    }
    value = kx_mul(load_k(mat_coeffs, header[0]), value);
    partials[2 * index] = value.c0;
    partials[2 * index + 1] = value.c1;
}

inline Kx joint_seeded_k_polynomial_sum(
    device const ulong *header,
    device const ulong *partials,
    ulong coefficient) {
    Kx value = Kx{0, 0};
    ulong work_end = header[7] + header[8];
    for (ulong work = header[7]; work < work_end; ++work) {
        value = kx_add(value, load_k(partials, work * RING_PRODUCT_COEFFICIENTS + coefficient));
    }
    return value;
}

inline Kx joint_seeded_k_phi81(
    device const ulong *header,
    device const ulong *partials,
    ulong coefficient) {
    Kx value = joint_seeded_k_polynomial_sum(header, partials, coefficient);
    if (coefficient <= 26) {
        value = kx_sub(value, joint_seeded_k_polynomial_sum(header, partials, coefficient + 54));
        if (coefficient <= 25) {
            value = kx_add(value, joint_seeded_k_polynomial_sum(header, partials, coefficient + 81));
        }
    } else {
        value = kx_sub(value, joint_seeded_k_polynomial_sum(header, partials, coefficient + 27));
    }
    return value;
}

kernel void joint_seeded_k_reduce(
    device const ulong *output_headers [[buffer(0)]],
    device const ulong *group_headers [[buffer(1)]],
    device const uint *group_outputs [[buffer(2)]],
    device const ulong *shape [[buffer(3)]],
    device const ulong *partials [[buffer(4)]],
    device ulong *table [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    ulong group = (ulong)index / RING_DEGREE;
    ulong coordinate = (ulong)index % RING_DEGREE;
    device const ulong *group_header = group_headers + 3 * group;
    ulong row = group_header[0] + coordinate;
    if (row >= shape[4]) {
        return;
    }
    Kx value = Kx{0, 0};
    ulong output_end = group_header[1] + group_header[2];
    for (ulong position = group_header[1]; position < output_end; ++position) {
        ulong output_index = group_outputs[position];
        device const ulong *header = output_headers + output_index * JOINT_SEEDED_OUTPUT_HEADER_WORDS;
        value = kx_add(value, joint_seeded_k_phi81(header, partials, coordinate));
    }
    Kx previous = load_k(table, row);
    value = kx_add(previous, value);
    table[2 * row] = value.c0;
    table[2 * row + 1] = value.c1;
}
