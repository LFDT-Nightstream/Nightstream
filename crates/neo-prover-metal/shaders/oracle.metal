// FE oracle and row-sumcheck kernels, included after common field helpers.
// Extension tables use interleaved [c0, c1]; base tables use one word per row.

kernel void copy_base_to_k(
    device const ulong *input [[buffer(0)]],
    device ulong *output [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    output[2 * index] = gl_from_word(input[index]);
    output[2 * index + 1] = 0;
}

kernel void fe_add_sparse_base_rows(
    device const ulong *indices [[buffer(0)]],
    device const ulong *values [[buffer(1)]],
    device ulong *tables [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    ulong destination = indices[index];
    tables[destination] = gl_add(gl_from_word(tables[destination]), gl_from_word(values[index]));
}

// Witness kind: 0 dense base plane, 1 local signed masks, 2 resident mask batch.
inline ulong fe_load_real_witness(
    device const ulong *witness,
    ulong blocks,
    ulong kind,
    ulong witness_index,
    ulong column) {
    if (kind == 0) {
        return gl_from_word(witness[column]);
    }
    ulong block = column / RING_DEGREE;
    ulong local = column % RING_DEGREE;
    ulong bit = 1ul << local;
    if (kind == 2) {
        ulong base = 2 * (witness_index * blocks + block);
        if ((witness[base] & bit) != 0) {
            return 1;
        }
        if ((witness[base + 1] & bit) != 0) {
            return GOLDILOCKS_MODULUS - 1;
        }
        return 0;
    }
    if ((witness[block] & bit) != 0) {
        return 1;
    }
    if ((witness[blocks + block] & bit) != 0) {
        return GOLDILOCKS_MODULUS - 1;
    }
    return 0;
}

kernel void fe_build_mcs_row_tables(
    device const uint *matrix_row_offsets [[buffer(0)]],
    device const ulong *matrix_entry_bases [[buffer(1)]],
    device const uint *matrix_identity [[buffer(2)]],
    device const uint *entry_columns [[buffer(3)]],
    device const ulong *entry_coefficients [[buffer(4)]],
    device const uint *selected_matrices [[buffer(5)]],
    device const ulong *witness [[buffer(6)]],
    device const ulong *shape [[buffer(7)]],
    device ulong *output [[buffer(8)]],
    uint index [[thread_position_in_grid]]) {
    ulong rows = shape[1];
    ulong blocks = shape[2];
    ulong n_eff = shape[3];
    ulong n_pad = shape[4];
    ulong table_count = shape[5];
    ulong witness_kind = shape[6];
    ulong witness_index = shape[7];
    ulong live_len = shape[8];
    ulong elements = table_count * live_len;
    if ((ulong)index >= elements) {
        return;
    }
    ulong table = index / live_len;
    ulong row = index % live_len;
    ulong value = 0;
    if (row < n_eff) {
        ulong matrix = selected_matrices[table];
        if (matrix_identity[matrix] != 0) {
            value = fe_load_real_witness(witness, blocks, witness_kind, witness_index, row);
        } else {
            ulong offset = matrix * (rows + 1) + row;
            ulong entry_base = matrix_entry_bases[matrix];
            ulong start = entry_base + matrix_row_offsets[offset];
            ulong end = entry_base + matrix_row_offsets[offset + 1];
            for (ulong entry = start; entry < end; ++entry) {
                ulong input = fe_load_real_witness(
                    witness,
                    blocks,
                    witness_kind,
                    witness_index,
                    entry_columns[entry]);
                if (input != 0) {
                    value = gl_add(value, gl_mul(gl_from_word(entry_coefficients[entry]), input));
                }
            }
        }
    }
    output[table * n_pad + row] = value;
    if (row + 1 == live_len && live_len < n_pad) {
        output[table * n_pad + live_len] = 0;
    }
}

// Seeded work is split into independent chunks, then reduced per logical output.
constant ulong FE_SEEDED_OUTPUT_HEADER_WORDS = 9;
constant ulong FE_SEEDED_WORK_HEADER_WORDS = 3;

inline ulong fe_seeded_column(
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

kernel void fe_seeded_k_partials(
    device const ulong *output_headers [[buffer(0)]],
    device const ulong *work_headers [[buffer(1)]],
    device const uint *word_starts [[buffer(2)]],
    device const ulong *rotations [[buffer(3)]],
    device const ulong *qk [[buffer(4)]],
    device const ulong *matrix_coefficients [[buffer(5)]],
    device const ulong *shape [[buffer(6)]],
    device ulong *partials [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
    ulong work = (ulong)index / RING_PRODUCT_COEFFICIENTS;
    ulong coefficient = (ulong)index % RING_PRODUCT_COEFFICIENTS;
    device const ulong *work_header = work_headers + work * FE_SEEDED_WORK_HEADER_WORDS;
    ulong output_index = work_header[0];
    device const ulong *header = output_headers + output_index * FE_SEEDED_OUTPUT_HEADER_WORDS;
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
            ulong column = fe_seeded_column(header, word_starts, message_row, message_col);
            if (column == ~0ul || column >= shape[1] * RING_DEGREE) {
                continue;
            }
            ulong rotation = gl_from_word(
                rotations[header[6] + message_col * RING_DEGREE + coefficient - message_row]);
            Kx input = load_k(qk, column);
            value = kx_add(value, Kx{gl_mul(input.c0, rotation), gl_mul(input.c1, rotation)});
        }
    }
    value = kx_mul(load_k(matrix_coefficients, header[0]), value);
    partials[2 * index] = value.c0;
    partials[2 * index + 1] = value.c1;
}

inline Kx fe_seeded_k_polynomial_sum(
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

inline Kx fe_seeded_k_phi81(
    device const ulong *header,
    device const ulong *partials,
    ulong coefficient) {
    Kx value = fe_seeded_k_polynomial_sum(header, partials, coefficient);
    if (coefficient <= 26) {
        value = kx_sub(value, fe_seeded_k_polynomial_sum(header, partials, coefficient + 54));
        if (coefficient <= 25) {
            value = kx_add(value, fe_seeded_k_polynomial_sum(header, partials, coefficient + 81));
        }
    } else {
        value = kx_sub(value, fe_seeded_k_polynomial_sum(header, partials, coefficient + 27));
    }
    return value;
}

kernel void fe_seeded_k_reduce(
    device const ulong *output_headers [[buffer(0)]],
    device const ulong *group_headers [[buffer(1)]],
    device const uint *group_outputs [[buffer(2)]],
    device const ulong *shape [[buffer(3)]],
    device const ulong *partials [[buffer(4)]],
    device ulong *output [[buffer(5)]],
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
        device const ulong *header = output_headers + output_index * FE_SEEDED_OUTPUT_HEADER_WORDS;
        value = kx_add(value, fe_seeded_k_phi81(header, partials, coordinate));
    }
    Kx previous = load_k(output, row);
    value = kx_add(previous, value);
    output[2 * row] = value.c0;
    output[2 * row + 1] = value.c1;
}

// Streaming rounds read each independently owned MCS buffer in place.
// Row-invariant channel weights are applied only after threadgroup reduction.
// Cropped producers keep row `live_len` as the zero high-half sentinel.
inline Kx fe_stream_load_mcs(
    device const ulong *tables,
    ulong table,
    ulong row,
    ulong table_len,
    bool base_mode) {
    ulong index = table * table_len + row;
    return base_mode ? Kx{gl_from_word(tables[index]), 0} : load_k(tables, index);
}

inline void fe_poly_mul_affine_compact(
    thread Kx *polynomial,
    Kx a,
    Kx b,
    uint current_degree) {
    polynomial[current_degree + 1] = kx_mul(polynomial[current_degree], b);
    for (uint degree = current_degree; degree > 0; --degree) {
        polynomial[degree] = kx_add(
            kx_mul(polynomial[degree], a),
            kx_mul(polynomial[degree - 1], b));
    }
    polynomial[0] = kx_mul(polynomial[0], a);
}

inline void fe_stream_accumulate_terms(
    thread Kx *sum,
    thread Kx *term_poly,
    device const ulong *mcs_tables,
    device const ulong *term_headers,
    device const ulong *term_variables,
    ulong first_term,
    ulong term_count,
    ulong index,
    ulong table_len,
    ulong table_count,
    bool base_mode,
    uint row_degree) {
    for (ulong term = first_term; term < first_term + term_count; ++term) {
        ulong header = 4 * term;
        Kx scale = Kx{term_headers[header], term_headers[header + 1]};
        ulong variable_start = term_headers[header + 2];
        ulong variable_count = term_headers[header + 3];
        if (variable_count == 0) {
            sum[0] = kx_add(sum[0], scale);
            continue;
        }

        ulong first_header = 2 * variable_start;
        ulong first_position = term_variables[first_header];
        uint first_exponent = (uint)term_variables[first_header + 1];
        if (variable_count == 1 && first_position < table_count) {
            Kx first_a = fe_stream_load_mcs(mcs_tables, first_position, index, table_len, base_mode);
            Kx first_b = kx_sub(
                fe_stream_load_mcs(mcs_tables, first_position, index + 1, table_len, base_mode),
                first_a);
            if (first_exponent == 1) {
                sum[0] = kx_add(sum[0], kx_mul(scale, first_a));
                sum[1] = kx_add(sum[1], kx_mul(scale, first_b));
                continue;
            }
            if (first_exponent == 2) {
                sum[0] = kx_add(sum[0], kx_mul(scale, kx_mul(first_a, first_a)));
                sum[1] = kx_add(
                    sum[1],
                    kx_mul(scale, kx_add(kx_mul(first_a, first_b), kx_mul(first_a, first_b))));
                sum[2] = kx_add(sum[2], kx_mul(scale, kx_mul(first_b, first_b)));
                continue;
            }
        }

        bool compact_product = variable_count >= 2 && variable_count <= 4;
        for (ulong variable = 0; compact_product && variable < variable_count; ++variable) {
            ulong variable_header = 2 * (variable_start + variable);
            compact_product = term_variables[variable_header] < table_count
                && term_variables[variable_header + 1] == 1;
        }
        if (compact_product) {
            term_poly[0] = scale;
            for (uint variable = 0; variable < (uint)variable_count; ++variable) {
                ulong variable_position = term_variables[2 * (variable_start + variable)];
                Kx a = fe_stream_load_mcs(mcs_tables, variable_position, index, table_len, base_mode);
                Kx b = kx_sub(
                    fe_stream_load_mcs(mcs_tables, variable_position, index + 1, table_len, base_mode),
                    a);
                fe_poly_mul_affine_compact(term_poly, a, b, variable);
            }
            uint limit = min((uint)variable_count, row_degree);
            for (uint degree = 0; degree <= limit; ++degree) {
                sum[degree] = kx_add(sum[degree], term_poly[degree]);
            }
            continue;
        }

        for (uint degree = 0; degree < SUMCHECK_MAX_COEFFS; ++degree) {
            term_poly[degree] = Kx{0, 0};
        }
        term_poly[0] = scale;
        uint current_degree = 0;
        for (ulong variable = 0; variable < variable_count; ++variable) {
            ulong variable_header = 2 * (variable_start + variable);
            ulong variable_position = term_variables[variable_header];
            uint exponent = (uint)term_variables[variable_header + 1];
            if (variable_position >= table_count) {
                continue;
            }
            Kx a = fe_stream_load_mcs(mcs_tables, variable_position, index, table_len, base_mode);
            Kx b = kx_sub(
                fe_stream_load_mcs(mcs_tables, variable_position, index + 1, table_len, base_mode),
                a);
            for (uint power = 0; power < exponent && current_degree < row_degree; ++power) {
                fe_poly_mul_affine_compact(term_poly, a, b, current_degree);
                current_degree += 1;
            }
        }
        uint limit = min(current_degree, row_degree);
        for (uint degree = 0; degree <= limit; ++degree) {
            sum[degree] = kx_add(sum[degree], term_poly[degree]);
        }
    }
}

kernel void fe_stream_mcs_round_partials(
    device const ulong *mcs_tables [[buffer(0)]],
    device const ulong *special_tables [[buffer(1)]],
    device const ulong *shape [[buffer(2)]],
    device const ulong *gamma_words [[buffer(3)]],
    device const ulong *term_headers [[buffer(4)]],
    device const ulong *term_variables [[buffer(5)]],
    device ulong *partials [[buffer(6)]],
    uint pair [[thread_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    ulong table_len = shape[0];
    ulong active_len = shape[1];
    uint coefficient_count = (uint)shape[2];
    uint row_degree = (uint)shape[3];
    uint active_coefficients = min(row_degree + 1, SUMCHECK_MAX_COEFFS);
    ulong table_count = shape[4];
    bool base_mode = shape[5] != 0;
    ulong term_count = shape[6];
    threadgroup Kx shared[SUMCHECK_REDUCTION_THREADS * SUMCHECK_MAX_COEFFS];
    Kx local[SUMCHECK_MAX_COEFFS];
    for (uint degree = 0; degree < SUMCHECK_MAX_COEFFS; ++degree) {
        local[degree] = Kx{0, 0};
    }
    ulong pairs = (active_len + 1) / 2;
    if (pair < pairs) {
        ulong index = 2 * pair;
        Kx eq0 = load_k(special_tables, index);
        Kx eq1 = kx_sub(load_k(special_tables, index + 1), eq0);
        Kx inner[SUMCHECK_MAX_COEFFS];
        Kx term_poly[SUMCHECK_MAX_COEFFS];
        for (uint degree = 0; degree < SUMCHECK_MAX_COEFFS; ++degree) {
            inner[degree] = Kx{0, 0};
        }
        fe_stream_accumulate_terms(
            inner,
            term_poly,
            mcs_tables,
            term_headers,
            term_variables,
            0,
            term_count,
            index,
            table_len,
            table_count,
            base_mode,
            row_degree);
        local[0] = kx_mul(eq0, inner[0]);
        for (uint coefficient = 1; coefficient < active_coefficients; ++coefficient) {
            local[coefficient] = kx_add(
                kx_mul(eq0, inner[coefficient]),
                kx_mul(eq1, inner[coefficient - 1]));
        }
    }
    for (uint coefficient = 0; coefficient < active_coefficients; ++coefficient) {
        shared[lane * SUMCHECK_MAX_COEFFS + coefficient] = local[coefficient];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = SUMCHECK_REDUCTION_THREADS / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            for (uint coefficient = 0; coefficient < active_coefficients; ++coefficient) {
                uint dst = lane * SUMCHECK_MAX_COEFFS + coefficient;
                uint src = (lane + stride) * SUMCHECK_MAX_COEFFS + coefficient;
                shared[dst] = kx_add(shared[dst], shared[src]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        Kx gamma = Kx{gamma_words[0], gamma_words[1]};
        for (uint coefficient = 0; coefficient < coefficient_count; ++coefficient) {
            Kx value = coefficient < active_coefficients ? kx_mul(shared[coefficient], gamma) : Kx{0, 0};
            ulong output = group * coefficient_count + coefficient;
            partials[2 * output] = value.c0;
            partials[2 * output + 1] = value.c1;
        }
    }
}

kernel void fe_stream_mcs_factored_round_partials(
    device const ulong *mcs_tables [[buffer(0)]],
    device const ulong *special_tables [[buffer(1)]],
    device const ulong *shape [[buffer(2)]],
    device const ulong *gamma_words [[buffer(3)]],
    device const ulong *group_headers [[buffer(4)]],
    device const ulong *term_headers [[buffer(5)]],
    device const ulong *term_variables [[buffer(6)]],
    device ulong *partials [[buffer(7)]],
    uint pair [[thread_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    ulong table_len = shape[0];
    ulong active_len = shape[1];
    uint coefficient_count = (uint)shape[2];
    uint row_degree = (uint)shape[3];
    uint active_coefficients = min(row_degree + 1, SUMCHECK_MAX_COEFFS);
    ulong table_count = shape[4];
    bool base_mode = shape[5] != 0;
    ulong factor_group_count = shape[6];
    threadgroup Kx shared[SUMCHECK_REDUCTION_THREADS * SUMCHECK_MAX_COEFFS];
    Kx local[SUMCHECK_MAX_COEFFS];
    for (uint degree = 0; degree < SUMCHECK_MAX_COEFFS; ++degree) {
        local[degree] = Kx{0, 0};
    }
    ulong pairs = (active_len + 1) / 2;
    if (pair < pairs) {
        ulong index = 2 * pair;
        Kx eq0 = load_k(special_tables, index);
        Kx eq1 = kx_sub(load_k(special_tables, index + 1), eq0);
        Kx factor[SUMCHECK_MAX_COEFFS];
        Kx term_poly[SUMCHECK_MAX_COEFFS];
        for (ulong factor_group = 0; factor_group < factor_group_count; ++factor_group) {
            ulong header = 3 * factor_group;
            ulong selector = group_headers[header];
            // Keep the selector endpoints out of the long term-program live range.
            // Active groups reload them below; zero groups skip all factor work.
            {
                Kx selector_low = fe_stream_load_mcs(mcs_tables, selector, index, table_len, base_mode);
                Kx selector_high = fe_stream_load_mcs(mcs_tables, selector, index + 1, table_len, base_mode);
                if (selector_low.c0 == 0 && selector_low.c1 == 0
                    && selector_high.c0 == 0 && selector_high.c1 == 0) {
                    continue;
                }
            }
            for (uint degree = 0; degree < SUMCHECK_MAX_COEFFS; ++degree) {
                factor[degree] = Kx{0, 0};
            }
            fe_stream_accumulate_terms(
                factor,
                term_poly,
                mcs_tables,
                term_headers,
                term_variables,
                group_headers[header + 1],
                group_headers[header + 2],
                index,
                table_len,
                table_count,
                base_mode,
                row_degree);
            Kx selector0 = fe_stream_load_mcs(mcs_tables, selector, index, table_len, base_mode);
            Kx selector1 = kx_sub(
                fe_stream_load_mcs(mcs_tables, selector, index + 1, table_len, base_mode),
                selector0);
            local[0] = kx_add(local[0], kx_mul(selector0, factor[0]));
            for (uint coefficient = 1; coefficient < active_coefficients; ++coefficient) {
                local[coefficient] = kx_add(
                    local[coefficient],
                    kx_add(
                        kx_mul(selector0, factor[coefficient]),
                        kx_mul(selector1, factor[coefficient - 1])));
            }
        }
        for (uint coefficient = active_coefficients - 1; coefficient > 0; --coefficient) {
            local[coefficient] = kx_add(
                kx_mul(eq0, local[coefficient]),
                kx_mul(eq1, local[coefficient - 1]));
        }
        local[0] = kx_mul(eq0, local[0]);
    }
    for (uint coefficient = 0; coefficient < active_coefficients; ++coefficient) {
        shared[lane * SUMCHECK_MAX_COEFFS + coefficient] = local[coefficient];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = SUMCHECK_REDUCTION_THREADS / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            for (uint coefficient = 0; coefficient < active_coefficients; ++coefficient) {
                uint dst = lane * SUMCHECK_MAX_COEFFS + coefficient;
                uint src = (lane + stride) * SUMCHECK_MAX_COEFFS + coefficient;
                shared[dst] = kx_add(shared[dst], shared[src]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        Kx gamma = Kx{gamma_words[0], gamma_words[1]};
        for (uint coefficient = 0; coefficient < coefficient_count; ++coefficient) {
            Kx value = coefficient < active_coefficients ? kx_mul(shared[coefficient], gamma) : Kx{0, 0};
            ulong output = group * coefficient_count + coefficient;
            partials[2 * output] = value.c0;
            partials[2 * output + 1] = value.c1;
        }
    }
}

kernel void fe_stream_eval_round_partials(
    device const ulong *special_tables [[buffer(0)]],
    device const ulong *shape [[buffer(1)]],
    device ulong *partials [[buffer(2)]],
    uint pair [[thread_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    ulong table_len = shape[0];
    ulong active_len = shape[1];
    uint coefficient_count = (uint)shape[2];
    uint active_coefficients = min(coefficient_count, 3u);
    ulong inputs_slot = shape[3];
    ulong eval_slot = shape[4];
    threadgroup Kx shared[SUMCHECK_REDUCTION_THREADS * SUMCHECK_MAX_COEFFS];
    Kx local[SUMCHECK_MAX_COEFFS];
    for (uint degree = 0; degree < SUMCHECK_MAX_COEFFS; ++degree) {
        local[degree] = Kx{0, 0};
    }
    ulong pairs = (active_len + 1) / 2;
    if (pair < pairs) {
        ulong index = 2 * pair;
        Kx r0 = load_k(special_tables, inputs_slot * table_len + index);
        Kx r1 = kx_sub(load_k(special_tables, inputs_slot * table_len + index + 1), r0);
        Kx v0 = load_k(special_tables, eval_slot * table_len + index);
        Kx v1 = kx_sub(load_k(special_tables, eval_slot * table_len + index + 1), v0);
        local[0] = kx_mul(r0, v0);
        local[1] = kx_add(kx_mul(r0, v1), kx_mul(r1, v0));
        local[2] = kx_mul(r1, v1);
    }
    for (uint coefficient = 0; coefficient < active_coefficients; ++coefficient) {
        shared[lane * SUMCHECK_MAX_COEFFS + coefficient] = local[coefficient];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = SUMCHECK_REDUCTION_THREADS / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            for (uint coefficient = 0; coefficient < active_coefficients; ++coefficient) {
                uint dst = lane * SUMCHECK_MAX_COEFFS + coefficient;
                uint src = (lane + stride) * SUMCHECK_MAX_COEFFS + coefficient;
                shared[dst] = kx_add(shared[dst], shared[src]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        Kx gamma_to_k = Kx{shape[5], shape[6]};
        for (uint coefficient = 0; coefficient < coefficient_count; ++coefficient) {
            Kx value = coefficient < active_coefficients ? kx_mul(gamma_to_k, shared[coefficient]) : Kx{0, 0};
            ulong output = group * coefficient_count + coefficient;
            partials[2 * output] = value.c0;
            partials[2 * output + 1] = value.c1;
        }
    }
}

kernel void fe_stream_constant_round_partials(
    device const ulong *special_tables [[buffer(0)]],
    device const ulong *shape [[buffer(1)]],
    device const ulong *constant_words [[buffer(2)]],
    device ulong *partials [[buffer(3)]],
    uint pair [[thread_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    ulong table_len = shape[0];
    ulong active_len = shape[1];
    uint coefficient_count = (uint)shape[2];
    uint active_coefficients = min(coefficient_count, 2u);
    threadgroup Kx shared[SUMCHECK_REDUCTION_THREADS * SUMCHECK_MAX_COEFFS];
    Kx local[SUMCHECK_MAX_COEFFS];
    for (uint degree = 0; degree < SUMCHECK_MAX_COEFFS; ++degree) {
        local[degree] = Kx{0, 0};
    }
    ulong pairs = (active_len + 1) / 2;
    if (pair < pairs) {
        ulong index = 2 * pair;
        Kx eq0 = load_k(special_tables, index);
        Kx eq1 = kx_sub(load_k(special_tables, index + 1), eq0);
        local[0] = eq0;
        local[1] = eq1;
    }
    for (uint coefficient = 0; coefficient < active_coefficients; ++coefficient) {
        shared[lane * SUMCHECK_MAX_COEFFS + coefficient] = local[coefficient];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = SUMCHECK_REDUCTION_THREADS / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            for (uint coefficient = 0; coefficient < active_coefficients; ++coefficient) {
                uint dst = lane * SUMCHECK_MAX_COEFFS + coefficient;
                uint src = (lane + stride) * SUMCHECK_MAX_COEFFS + coefficient;
                shared[dst] = kx_add(shared[dst], shared[src]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        Kx constant_value = Kx{constant_words[0], constant_words[1]};
        for (uint coefficient = 0; coefficient < coefficient_count; ++coefficient) {
            Kx value = coefficient < active_coefficients ? kx_mul(shared[coefficient], constant_value) : Kx{0, 0};
            ulong output = group * coefficient_count + coefficient;
            partials[2 * output] = value.c0;
            partials[2 * output + 1] = value.c1;
        }
    }
}

// Round zero converts base tables to K while folding, avoiding a full copy.
kernel void fe_fold_base_tables_in_place(
    device ulong *tables [[buffer(0)]],
    device const ulong *challenge_words [[buffer(1)]],
    device const ulong *shape [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    ulong table_count = shape[0];
    ulong table_len = shape[1];
    ulong live_len = shape[2];
    ulong next_len = table_len / 2;
    ulong next_live_len = (live_len + 1) / 2;
    ulong elements = table_count * next_live_len;
    if ((ulong)index >= elements) {
        return;
    }
    ulong table = index / next_live_len;
    ulong pair = index % next_live_len;
    ulong input = table * table_len + 2 * pair;
    ulong low = gl_from_word(tables[input]);
    ulong high = gl_from_word(tables[input + 1]);
    Kx challenge = Kx{gl_from_word(challenge_words[0]), gl_from_word(challenge_words[1])};
    Kx folded = kx_add(Kx{low, 0}, kx_mul(challenge, Kx{gl_sub(high, low), 0}));
    ulong output = 2 * (table * next_len + pair);
    tables[output] = folded.c0;
    tables[output + 1] = folded.c1;
    if (pair + 1 == next_live_len && next_live_len < next_len) {
        ulong sentinel = 2 * (table * next_len + next_live_len);
        tables[sentinel] = 0;
        tables[sentinel + 1] = 0;
    }
}

kernel void fe_fold_k_tables_live(
    device const ulong *tables [[buffer(0)]],
    device const ulong *challenge_words [[buffer(1)]],
    device const ulong *shape [[buffer(2)]],
    device ulong *output [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    ulong table_len = shape[1];
    ulong live_len = shape[2];
    ulong next_len = table_len / 2;
    ulong next_live_len = (live_len + 1) / 2;
    ulong table = (ulong)index / next_live_len;
    ulong pair = (ulong)index % next_live_len;
    ulong input = table * table_len + 2 * pair;
    Kx low = load_k(tables, input);
    Kx high = load_k(tables, input + 1);
    Kx challenge = Kx{gl_from_word(challenge_words[0]), gl_from_word(challenge_words[1])};
    Kx folded = kx_add(low, kx_mul(challenge, kx_sub(high, low)));
    ulong destination = table * next_len + pair;
    output[2 * destination] = folded.c0;
    output[2 * destination + 1] = folded.c1;
    if (pair + 1 == next_live_len && next_live_len < next_len) {
        ulong sentinel = table * next_len + next_live_len;
        output[2 * sentinel] = 0;
        output[2 * sentinel + 1] = 0;
    }
}

kernel void fe_copy_k_tables_live(
    device const ulong *tables [[buffer(0)]],
    device const ulong *shape [[buffer(1)]],
    device ulong *output [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    ulong next_len = shape[1] / 2;
    ulong next_live_len = (shape[2] + 1) / 2;
    ulong table = (ulong)index / next_live_len;
    ulong row = (ulong)index % next_live_len;
    ulong source = table * next_len + row;
    output[2 * source] = tables[2 * source];
    output[2 * source + 1] = tables[2 * source + 1];
    if (row + 1 == next_live_len && next_live_len < next_len) {
        ulong sentinel = table * next_len + next_live_len;
        output[2 * sentinel] = 0;
        output[2 * sentinel + 1] = 0;
    }
}
