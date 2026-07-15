// Pi_CCS/Pi_DEC compact ring-form contraction. Included after Goldilocks helpers.

constant ulong DEC_PARALLEL_FORM_LIST_THRESHOLD = 128;
constant uint DEC_FORM_REDUCTION_THREADS = 256;

inline ulong dec_compact_original_form(
    device const uint *active_local_offsets,
    device const ulong *active_entry_bases,
    device const uint *entry_rows,
    device const ulong *entry_coefficients,
    device const ulong *chi,
    ulong active,
    ulong local,
    ulong component,
    ulong n_eff,
    ulong chi_len) {
    ulong offset = active * (RING_DEGREE + 1) + local;
    ulong entry_base = active_entry_bases[active];
    ulong start = entry_base + (ulong)active_local_offsets[offset];
    ulong end = entry_base + (ulong)active_local_offsets[offset + 1];
    ulong value = 0;
    for (ulong entry = start; entry < end; ++entry) {
        ulong row = (ulong)entry_rows[entry];
        if (row < n_eff && row < chi_len) {
            value = gl_add(
                value,
                gl_mul(
                    gl_from_word(chi[2 * row + component]),
                    gl_from_word(entry_coefficients[entry])));
        }
    }
    return value;
}

// Serial contraction is cheapest for the overwhelmingly common short lists.
// Long lists are deliberately left for dec_build_parallel_original_forms.
kernel void dec_build_ring_forms(
    device const uint *active_local_offsets [[buffer(0)]],
    device const ulong *active_entry_bases [[buffer(1)]],
    device const uint *matrix_identity [[buffer(2)]],
    device const uint *entry_rows [[buffer(3)]],
    device const ulong *entry_coefficients [[buffer(4)]],
    device const ulong *chi [[buffer(5)]],
    device const ulong *shape [[buffer(6)]],
    device ulong *forms [[buffer(7)]],
    device const uint *active_blocks [[buffer(8)]],
    uint index [[thread_position_in_grid]]) {
    ulong blocks = shape[1];
    ulong n_eff = shape[2];
    ulong chi_len = shape[3];
    ulong local = index % RING_DEGREE;
    ulong rest = index / RING_DEGREE;
    ulong component = rest % 2;
    ulong active = rest / 2;
    ulong encoded = active_blocks[active];
    ulong matrix = encoded / blocks;
    ulong block = encoded % blocks;
    ulong output = (active * 2 + component) * RING_DEGREE + local;
    if (matrix_identity[matrix] != 0) {
        ulong row = block * RING_DEGREE + local;
        forms[output] = row < n_eff && row < chi_len
            ? gl_from_word(chi[2 * row + component])
            : 0;
        return;
    }

    ulong offset = active * (RING_DEGREE + 1) + local;
    ulong entries = (ulong)active_local_offsets[offset + 1]
        - (ulong)active_local_offsets[offset];
    if (entries >= DEC_PARALLEL_FORM_LIST_THRESHOLD) {
        return;
    }
    forms[output] = dec_compact_original_form(
        active_local_offsets, active_entry_bases, entry_rows,
        entry_coefficients, chi, active, local, component, n_eff, chi_len);
}

// One threadgroup cooperatively contracts each long original-coefficient list.
kernel void dec_build_parallel_original_forms(
    device const uint *active_local_offsets [[buffer(0)]],
    device const ulong *active_entry_bases [[buffer(1)]],
    device const uint *entry_rows [[buffer(3)]],
    device const ulong *entry_coefficients [[buffer(4)]],
    device const ulong *chi [[buffer(5)]],
    device const ulong *shape [[buffer(6)]],
    device ulong *forms [[buffer(7)]],
    device const uint *parallel_lists [[buffer(9)]],
    uint lane [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    threadgroup ulong partials[DEC_FORM_REDUCTION_THREADS];
    ulong component = group % 2;
    ulong encoded = parallel_lists[group / 2];
    ulong active = encoded / RING_DEGREE;
    ulong local = encoded % RING_DEGREE;
    ulong offset = active * (RING_DEGREE + 1) + local;
    ulong entry_base = active_entry_bases[active];
    ulong start = entry_base + (ulong)active_local_offsets[offset];
    ulong end = entry_base + (ulong)active_local_offsets[offset + 1];
    ulong n_eff = shape[2];
    ulong chi_len = shape[3];
    ulong value = 0;
    for (ulong entry = start + lane; entry < end; entry += DEC_FORM_REDUCTION_THREADS) {
        ulong row = (ulong)entry_rows[entry];
        if (row < n_eff && row < chi_len) {
            value = gl_add(
                value,
                gl_mul(
                    gl_from_word(chi[2 * row + component]),
                    gl_from_word(entry_coefficients[entry])));
        }
    }
    partials[lane] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = DEC_FORM_REDUCTION_THREADS / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            partials[lane] = gl_add(partials[lane], partials[lane + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        forms[(active * 2 + component) * RING_DEGREE + local] = partials[0];
    }
}

// Phi_81 bar is a permutation of thirteen four-value cycles plus coefficient
// 27. Owning each cycle in one thread makes this transform safe in place.
kernel void dec_bar_ring_forms_in_place(
    device ulong *forms [[buffer(0)]],
    uint index [[thread_position_in_grid]]) {
    ulong slot = index % 14;
    ulong output_base = (index / 14) * RING_DEGREE;
    if (slot == 13) {
        forms[output_base + 27] = gl_sub(0, forms[output_base + 27]);
        return;
    }

    ulong low = slot + 1;
    ulong reflected = 27 - low;
    ulong low_value = forms[output_base + low];
    ulong low_high_value = forms[output_base + 27 + low];
    ulong reflected_value = forms[output_base + reflected];
    ulong reflected_high_value = forms[output_base + 27 + reflected];
    forms[output_base + reflected] = gl_sub(0, gl_add(low_value, low_high_value));
    forms[output_base + 27 + reflected] = gl_sub(0, low_value);
    forms[output_base + low] = gl_sub(0, gl_add(reflected_value, reflected_high_value));
    forms[output_base + 27 + low] = gl_sub(0, reflected_value);
}

constant ulong DEC_SEEDED_OUTPUT_HEADER_WORDS = 9;

inline ulong dec_seeded_raw_rotation(
    device const ulong *rotation,
    ulong shift,
    ulong exponent) {
    if (exponent < shift || exponent - shift >= RING_DEGREE) {
        return 0;
    }
    return gl_from_word(rotation[exponent - shift]);
}

// Coefficient `coordinate` after multiplying the static rotation by X^shift
// modulo Phi_81 = X^54 + X^27 + 1.
inline ulong dec_seeded_rotated_coefficient(
    device const ulong *rotation,
    ulong shift,
    ulong coordinate) {
    ulong value = dec_seeded_raw_rotation(rotation, shift, coordinate);
    if (coordinate <= 26) {
        value = gl_sub(value, dec_seeded_raw_rotation(rotation, shift, coordinate + 54));
        if (coordinate <= 25) {
            value = gl_add(value, dec_seeded_raw_rotation(rotation, shift, coordinate + 81));
        }
    } else {
        value = gl_sub(value, dec_seeded_raw_rotation(rotation, shift, coordinate + 27));
    }
    return value;
}

// Contract the structure-static seeded stream directly against the resident
// row tensor. Segment ownership gives every output coefficient one writer.
kernel void dec_build_seeded_ring_forms(
    device const ulong *output_headers [[buffer(0)]],
    device const uint *word_starts [[buffer(1)]],
    device const ulong *rotations [[buffer(2)]],
    device const uint *active_segment_offsets [[buffer(3)]],
    device const uint *segments [[buffer(4)]],
    device const ulong *chi [[buffer(5)]],
    device const ulong *shape [[buffer(6)]],
    device ulong *seeded_forms [[buffer(7)]],
    device const uint *active_blocks [[buffer(8)]],
    device const uint *active_indices [[buffer(9)]],
    uint index [[thread_position_in_grid]]) {
    ulong local = (ulong)index % RING_DEGREE;
    ulong rest = (ulong)index / RING_DEGREE;
    ulong component = rest % 2;
    ulong group = rest / 2;
    ulong active = (ulong)active_indices[group];
    ulong column_block = (ulong)active_blocks[active] % shape[1];
    ulong column = column_block * RING_DEGREE + local;
    ulong row_limit = min(shape[2], shape[3]);
    ulong value = 0;
    ulong segment_end = (ulong)active_segment_offsets[group + 1];
    for (ulong segment = (ulong)active_segment_offsets[group]; segment < segment_end; ++segment) {
        ulong output = (ulong)segments[2 * segment];
        ulong word = (ulong)segments[2 * segment + 1];
        device const ulong *header = output_headers + output * DEC_SEEDED_OUTPUT_HEADER_WORDS;
        ulong word_start = (ulong)word_starts[word];
        ulong word_width = header[3];
        if (column < word_start || column - word_start >= word_width || header[1] >= row_limit) {
            continue;
        }
        ulong bit_index = (word - header[5]) * word_width + column - word_start;
        ulong message_row = bit_index / header[2];
        ulong message_col = bit_index % header[2];
        if (message_row >= RING_DEGREE) {
            continue;
        }
        device const ulong *rotation = rotations + header[6] + message_col * RING_DEGREE;
        ulong coordinate_count = min(RING_DEGREE, row_limit - header[1]);
        ulong weight = 0;
        for (ulong coordinate = 0; coordinate < coordinate_count; ++coordinate) {
            ulong coefficient = dec_seeded_rotated_coefficient(rotation, message_row, coordinate);
            if (coefficient != 0) {
                weight = gl_add(
                    weight,
                    gl_mul(gl_from_word(chi[2 * (header[1] + coordinate) + component]), coefficient));
            }
        }
        value = gl_add(value, weight);
    }
    seeded_forms[index] = value;
}

// Add bar(seed-only forms) to the already-barred explicit forms without a
// host patch or a second 41 MiB transformed buffer.
kernel void dec_add_bar_seeded_ring_forms(
    device const ulong *seeded_forms [[buffer(0)]],
    device ulong *forms [[buffer(1)]],
    device const uint *active_indices [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    ulong local = (ulong)index % RING_DEGREE;
    ulong base = (ulong)index - local;
    ulong rest = (ulong)index / RING_DEGREE;
    ulong component = rest % 2;
    ulong group = rest / 2;
    ulong value;
    if (local == 0) {
        value = seeded_forms[base];
    } else if (local == 27) {
        value = gl_sub(0, seeded_forms[base + 27]);
    } else if (local < 27) {
        value = gl_sub(
            0,
            gl_add(seeded_forms[base + 27 - local], seeded_forms[base + 54 - local]));
    } else {
        value = gl_sub(0, seeded_forms[base + 54 - local]);
    }
    ulong destination = ((ulong)active_indices[group] * 2 + component) * RING_DEGREE + local;
    forms[destination] = gl_add(forms[destination], value);
}

kernel void dec_add_sparse_ring_forms(
    device const ulong *bases [[buffer(0)]],
    device const ulong *coefficients [[buffer(1)]],
    device ulong *forms [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    ulong local = index % (2 * RING_DEGREE);
    ulong entry = index / (2 * RING_DEGREE);
    ulong destination = bases[entry] + local;
    forms[destination] = gl_add(forms[destination], gl_from_word(coefficients[index]));
}
