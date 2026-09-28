// One-joint padded-row PiCCS evaluator.
// The host owns Fiat-Shamir and checks every returned round message.

inline ulong joint_mask_digit(
    device const ulong *masks,
    ulong blocks,
    ulong witness,
    ulong block,
    ulong bit,
    ulong magnitudes) {
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

inline ulong joint_mask_value(
    device const ulong *masks,
    ulong blocks,
    ulong witness,
    ulong column,
    ulong magnitudes) {
    return joint_mask_digit(masks, blocks, witness,
                            column / RING_DEGREE, 1ul << (column % RING_DEGREE), magnitudes);
}

#include "assignments.metal"
#include "carried.metal"

kernel void joint_build_application_tables(
    device const uchar *row_offsets [[buffer(0)]],
    device const uint *row_blocks [[buffer(1)]],
    device const uint *dense_offsets [[buffer(2)]],
    device const uchar *dense_locals [[buffer(3)]],
    device const ulong *dense_coefficients [[buffer(4)]],
    device const uchar *geometric_row_offsets [[buffer(5)]],
    device const ulong *geometric_runs [[buffer(6)]],
    device const ulong *masks [[buffer(7)]],
    device const ulong *shape [[buffer(8)]],
    device ulong *output [[buffer(9)]],
    device const uint2 *dense_row_blocks [[buffer(10)]],
    uint local_row [[thread_position_in_grid]]) {
    ulong rows = shape[0];
    ulong blocks = shape[1];
    ulong n_eff = shape[2];
    ulong table_len = shape[3];
    ulong witness = shape[4];
    ulong output_table = shape[5];
    ulong offset_width = shape[6];
    bool identity = shape[7] != 0;
    ulong magnitudes = shape[8];
    ulong geometric_offset_width = shape[9];
    ulong row = shape[10] + (ulong)local_row;
    if ((ulong)local_row >= rows) {
        return;
    }
    ulong value = 0;
    if (row < n_eff) {
        if (identity) {
            value = joint_mask_value(masks, blocks, witness, row, magnitudes);
        } else {
            ulong start = compact_row_offset(row_offsets, local_row, offset_width);
            ulong end = compact_row_offset(row_offsets, (ulong)local_row + 1, offset_width);
            for (ulong entry = start; entry < end; ++entry) {
                uint reference = row_blocks[entry];
                if ((reference & COMPACT_DENSE_BLOCK_TAG) == 0) {
                    ulong block = (ulong)(reference & COMPACT_SINGLE_BLOCK_MASK);
                    ulong local = (ulong)((reference >> COMPACT_SINGLE_LOCAL_SHIFT) & COMPACT_SINGLE_LOCAL_MASK);
                    ulong input = joint_mask_digit(masks, blocks, witness, block, 1ul << local, magnitudes);
                    value = (reference & COMPACT_NEGATIVE_BLOCK_TAG) == 0
                        ? gl_add(value, input)
                        : gl_sub(value, input);
                } else {
                    uint2 block = dense_row_blocks[reference & COMPACT_DENSE_INDEX_MASK];
                    uint dense = block.y;
                    for (uint coefficient = dense_offsets[dense]; coefficient < dense_offsets[dense + 1]; ++coefficient) {
                        ulong bit = 1ul << dense_locals[coefficient];
                        ulong input = joint_mask_digit(masks, blocks, witness, block.x, bit, magnitudes);
                        if (input != 0) {
                            value = gl_add(
                                value,
                                gl_mul(gl_from_word(dense_coefficients[coefficient]), input));
                        }
                    }
                }
            }
            if (geometric_offset_width != 0) {
                ulong geometric_start = compact_row_offset(geometric_row_offsets, local_row, geometric_offset_width);
                ulong geometric_end = compact_row_offset(geometric_row_offsets, (ulong)local_row + 1, geometric_offset_width);
                for (ulong run = geometric_start; run < geometric_end; ++run) {
                    ulong packed = geometric_runs[3 * run];
                    ulong column = packed & 0xfffffffful;
                    ulong run_end = column + (packed >> 32);
                    ulong coefficient = gl_from_word(geometric_runs[3 * run + 1]);
                    ulong ratio = gl_from_word(geometric_runs[3 * run + 2]);
                    ulong block = column / RING_DEGREE;
                    ulong bit = 1ul << (column % RING_DEGREE);
                    for (; column < run_end; ++column) {
                        ulong input = joint_mask_digit(masks, blocks, witness, block, bit, magnitudes);
                        if (input != 0 && coefficient != 0) {
                            value = gl_add(value, gl_mul(coefficient, input));
                        }
                        coefficient = gl_mul(coefficient, ratio);
                        bit <<= 1;
                        if (bit == (1ul << RING_DEGREE)) {
                            bit = 1;
                            ++block;
                        }
                    }
                }
            }
        }
    }
    output[output_table * table_len + shape[11] + local_row] = value;
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

kernel void joint_accumulate_application(
    device const ulong *input [[buffer(0)]],
    device const ulong *shape [[buffer(1)]],
    device ulong *output [[buffer(2)]],
    uint table [[thread_position_in_grid]]) {
    if ((ulong)table >= shape[0]) {
        return;
    }
    ulong destination = (ulong)table * shape[1] + shape[2];
    Kx weight = Kx{gl_from_word(shape[3]), gl_from_word(shape[4])};
    Kx value = kx_add(load_k(output, destination), kx_mul(load_k(input, table), weight));
    output[2 * destination] = value.c0;
    output[2 * destination + 1] = value.c1;
}

inline Kx joint_load_table(
    device const ulong *tables,
    ulong table_len,
    ulong table,
    ulong index,
    bool base_field,
    ulong row_offset,
    ulong local_len) {
    if (index >= table_len || index < row_offset || index - row_offset >= local_len) {
        return Kx{0, 0};
    }
    ulong position = table * local_len + index - row_offset;
    return base_field ? Kx{gl_from_word(tables[position]), 0} : load_k(tables, position);
}

inline ulong joint_signed_root_f(int root) {
    return root < 0 ? gl_sub(0, (ulong)(-root)) : (ulong)root;
}

inline Kx joint_signed_root_k(int root) {
    return Kx{joint_signed_root_f(root), 0};
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
    device const ulong *assignment_values [[buffer(11)]],
    uint local_pair [[thread_position_in_grid]],
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
    bool zero_application_padding = shape[25] != 0;
    ulong pair = shape[27] + (ulong)local_pair;
    ulong pair_count = shape[28];
    ulong application_row_offset = shape[29];
    ulong application_local_len = shape[30];
    threadgroup Kx shared[SUMCHECK_REDUCTION_THREADS * SUMCHECK_MAX_COEFFS];
    Kx local[SUMCHECK_MAX_COEFFS];
    for (uint coefficient = 0; coefficient < SUMCHECK_MAX_COEFFS; ++coefficient) {
        local[coefficient] = Kx{0, 0};
    }

    ulong pairs = (active_len + 1) / 2;
    if ((ulong)local_pair < pair_count && pair < pairs) {
        ulong low_index = 2 * pair;
        ulong high_index = low_index + 1;
        Kx inner[SUMCHECK_MAX_COEFFS];
        for (uint coefficient = 0; coefficient < SUMCHECK_MAX_COEFFS; ++coefficient) {
            inner[coefficient] = Kx{0, 0};
        }

        // Beyond the application rows, every matrix coordinate is zero.
        // Skip the polynomial only when the host established f(0) == 0.
        for (ulong source = 0;
             source < fresh_count && (low_index < application_len || !zero_application_padding);
             ++source) {
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
                        application_tables, application_len, table, low_index, base_round,
                        application_row_offset, application_local_len);
                    Kx high = joint_load_table(
                        application_tables, application_len, table, high_index, base_round,
                        application_row_offset, application_local_len);
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
                assignment_values,
                source,
                low_index,
                shape[26],
                blocks,
                range_base - 1,
                assignment_len);
            Kx high = joint_assignment_value(
                assignments_or_masks,
                assignment_sources,
                assignment_values,
                source,
                high_index,
                shape[26],
                blocks,
                range_base - 1,
                assignment_len);
            Kx slope = kx_sub(high, low);
            // Base-table values are signed digits, hence roots of the range
            // polynomial. Equal endpoints give the zero polynomial. A pair
            // of zero extension values stays zero in every later round too.
            if ((base_round && low.c0 == high.c0 && low.c1 == high.c1)
                || ((low.c0 | low.c1 | high.c0 | high.c1) == 0)) {
                continue;
            }
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
