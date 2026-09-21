// Compact transposed SuperNeo forms used by the canonical PiCCS openings.

constant ulong DEC_PARALLEL_FORM_LIST_THRESHOLD = 128;
constant uint DEC_FORM_REDUCTION_THREADS = 256;

inline ulong dec_pow(ulong base, ulong exponent) {
    ulong value = 1;
    while (exponent != 0) {
        if ((exponent & 1) != 0) {
            value = gl_mul(value, base);
        }
        base = gl_mul(base, base);
        exponent >>= 1;
    }
    return value;
}

// The transpose carries one reference per row/block. Coefficient patterns are
// shared by all uses of the same block formula.
inline ulong dec_pattern_coefficient(
    uint descriptor,
    ulong local,
    device const uint *offsets,
    device const uchar *locals,
    device const ulong *coefficients) {
    if ((descriptor & 0x80000000u) == 0) {
        if (((descriptor >> 24) & 0x3fu) != local) {
            return 0;
        }
        return (descriptor & 0x40000000u) == 0 ? 1 : GOLDILOCKS_MODULUS - 1;
    }
    uint pattern = descriptor & 0x7fffffffu;
    uint start = offsets[pattern];
    uint end = offsets[pattern + 1];
    if (end - start == RING_DEGREE) {
        return gl_from_word(coefficients[start + local]);
    }
    uint limit = end;
    while (start < end) {
        uint middle = start + (end - start) / 2;
        if ((ulong)locals[middle] < local) {
            start = middle + 1;
        } else {
            end = middle;
        }
    }
    return start < limit && (ulong)locals[start] == local
        ? gl_from_word(coefficients[start]) : 0;
}

inline Kx dec_factored_weight(
    device const ulong *low,
    device const ulong *high,
    ulong low_bits,
    ulong row) {
    ulong a = 2 * (row & ((1ul << low_bits) - 1));
    ulong b = 2 * (row >> low_bits);
    return kx_mul(Kx{low[a], low[a + 1]}, Kx{high[b], high[b + 1]});
}

kernel void dec_build_row_weights(
    device const ulong *low [[buffer(0)]],
    device const ulong *high [[buffer(1)]],
    device const ulong *shape [[buffer(2)]],
    device ulong *chi [[buffer(3)]],
    uint row [[thread_position_in_grid]]) {
    Kx value = dec_factored_weight(low, high, shape[7], row);
    chi[2 * (ulong)row] = value.c0;
    chi[2 * (ulong)row + 1] = value.c1;
}

inline ulong dec_original_term(
    uint2 entry,
    ulong local,
    ulong component,
    device const uint *pattern_offsets,
    device const uchar *pattern_locals,
    device const ulong *pattern_coefficients,
    device const ulong *chi) {
    ulong coefficient = dec_pattern_coefficient(
        entry.y, local, pattern_offsets, pattern_locals, pattern_coefficients);
    return gl_mul(chi[2 * (ulong)entry.x + component], coefficient);
}

kernel void dec_build_ring_forms(
    device const ulong *active_offsets [[buffer(0)]],
    device const ulong *active_local_masks [[buffer(1)]],
    device const uint *matrix_identity [[buffer(2)]],
    device const uint2 *entries [[buffer(3)]],
    device const uint *pattern_offsets [[buffer(4)]],
    device const ulong *chi [[buffer(5)]],
    device const ulong *shape [[buffer(6)]],
    device ulong *forms [[buffer(7)]],
    device const uint *active_blocks [[buffer(8)]],
    device const uchar *pattern_locals [[buffer(11)]],
    device const ulong *pattern_coefficients [[buffer(12)]],
    device const ulong *chi_low [[buffer(13)]],
    device const ulong *chi_high [[buffer(14)]],
    uint index [[thread_position_in_grid]]) {
    ulong blocks = shape[1];
    ulong local = index % RING_DEGREE;
    ulong rest = index / RING_DEGREE;
    ulong component = rest % 2;
    ulong active = shape[5] + rest / 2;
    ulong encoded = active_blocks[active];
    ulong matrix = encoded / blocks;
    ulong block = encoded % blocks;
    if (matrix_identity[matrix] != 0) {
        ulong row = block * RING_DEGREE + local;
        // Pad includes the completion tail; application identities stop at n.
        ulong limit = matrix == 0 ? shape[4] : shape[2];
        Kx value = row < limit
            ? dec_factored_weight(chi_low, chi_high, shape[7], row) : Kx{0, 0};
        forms[index] = component == 0 ? value.c0 : value.c1;
        return;
    }
    if ((active_local_masks[active] & (1ul << local)) == 0) {
        forms[index] = 0;
        return;
    }
    ulong start = active_offsets[active];
    ulong end = active_offsets[active + 1];
    if (end - start >= DEC_PARALLEL_FORM_LIST_THRESHOLD) {
        return;
    }
    ulong value = 0;
    for (ulong entry = start; entry < end; ++entry) {
        value = gl_add(value, dec_original_term(entries[entry], local, component,
            pattern_offsets, pattern_locals, pattern_coefficients, chi));
    }
    forms[index] = value;
}

kernel void dec_build_parallel_original_forms(
    device const ulong *active_offsets [[buffer(0)]],
    device const uint2 *entries [[buffer(3)]],
    device const uint *pattern_offsets [[buffer(4)]],
    device const ulong *chi [[buffer(5)]],
    device const ulong *shape [[buffer(6)]],
    device ulong *forms [[buffer(7)]],
    device const uint *parallel_lists [[buffer(9)]],
    device const uchar *pattern_locals [[buffer(11)]],
    device const ulong *pattern_coefficients [[buffer(12)]],
    uint lane [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    threadgroup ulong partials[DEC_FORM_REDUCTION_THREADS];
    ulong component = group % 2;
    ulong encoded = parallel_lists[group / 2];
    ulong active = encoded / RING_DEGREE;
    if (active < shape[5] || active >= shape[6]) {
        return;
    }
    ulong local = encoded % RING_DEGREE;
    ulong start = active_offsets[active];
    ulong end = active_offsets[active + 1];
    ulong value = 0;
    for (ulong entry = start + lane; entry < end; entry += DEC_FORM_REDUCTION_THREADS) {
        value = gl_add(value, dec_original_term(entries[entry], local, component,
            pattern_offsets, pattern_locals, pattern_coefficients, chi));
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
        forms[((active - shape[5]) * 2 + component) * RING_DEGREE + local] = partials[0];
    }
}

kernel void dec_build_parallel_original_form_tiles(
    device const ulong *active_offsets [[buffer(0)]],
    device const uint2 *entries [[buffer(3)]],
    device const uint *pattern_offsets [[buffer(4)]],
    device const ulong *chi [[buffer(5)]],
    device const ulong *shape [[buffer(6)]],
    device const uint *tiles [[buffer(9)]],
    device ulong *tile_partials [[buffer(10)]],
    device const uchar *pattern_locals [[buffer(11)]],
    device const ulong *pattern_coefficients [[buffer(12)]],
    uint lane [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    threadgroup ulong partials[DEC_FORM_REDUCTION_THREADS];
    ulong component = group % 2;
    ulong tile = group / 2;
    ulong descriptor = 3 * tile;
    ulong encoded = tiles[descriptor];
    ulong active = encoded / RING_DEGREE;
    if (active < shape[5] || active >= shape[6]) {
        return;
    }
    ulong local = encoded % RING_DEGREE;
    ulong start = active_offsets[active] + (ulong)tiles[descriptor + 1];
    ulong end = start + (ulong)tiles[descriptor + 2];
    ulong value = 0;
    for (ulong entry = start + lane; entry < end; entry += DEC_FORM_REDUCTION_THREADS) {
        value = gl_add(value, dec_original_term(entries[entry], local, component,
            pattern_offsets, pattern_locals, pattern_coefficients, chi));
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
        tile_partials[group] = partials[0];
    }
}

kernel void dec_reduce_parallel_original_form_tiles(
    device ulong *forms [[buffer(0)]],
    device const uint *lists [[buffer(1)]],
    device const uint *tile_offsets [[buffer(2)]],
    device const ulong *tile_partials [[buffer(3)]],
    device const ulong *shape [[buffer(4)]],
    uint lane [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    threadgroup ulong partials[DEC_FORM_REDUCTION_THREADS];
    ulong component = group % 2;
    ulong list = group / 2;
    ulong active = (ulong)lists[list] / RING_DEGREE;
    if (active < shape[5] || active >= shape[6]) {
        return;
    }
    ulong start = tile_offsets[list];
    ulong end = tile_offsets[list + 1];
    ulong value = 0;
    for (ulong tile = start + lane; tile < end; tile += DEC_FORM_REDUCTION_THREADS) {
        value = gl_add(value, gl_from_word(tile_partials[2 * tile + component]));
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
        ulong encoded = lists[list];
        ulong active = encoded / RING_DEGREE;
        ulong local = encoded % RING_DEGREE;
        forms[((active - shape[5]) * 2 + component) * RING_DEGREE + local] = partials[0];
    }
}

kernel void dec_add_geometric_ring_forms(
    device const uint2 *groups [[buffer(0)]],
    device const uint2 *segments [[buffer(1)]],
    device const ulong *runs [[buffer(2)]],
    device const ulong *chi [[buffer(3)]],
    device const ulong *shape [[buffer(4)]],
    device ulong *forms [[buffer(5)]],
    device const uint *active_blocks [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    ulong local = index % RING_DEGREE;
    ulong rest = index / RING_DEGREE;
    ulong component = rest % 2;
    ulong group_index = rest / 2;
    uint2 group = groups[group_index];
    if (group.x < shape[5] || group.x >= shape[6]) {
        return;
    }
    ulong column = ((ulong)active_blocks[group.x] % shape[1]) * RING_DEGREE + local;
    ulong value = 0;
    for (ulong segment = group.y; segment < groups[group_index + 1].y; ++segment) {
        uint2 entry = segments[segment];
        ulong row = entry.x;
        if (row >= shape[2] || row >= shape[3]) {
            continue;
        }
        ulong packed = runs[3 * (ulong)entry.y];
        ulong start = packed & 0xfffffffful;
        ulong length = packed >> 32;
        if (column < start || column >= start + length) {
            continue;
        }
        ulong coefficient = gl_mul(
            gl_from_word(runs[3 * (ulong)entry.y + 1]),
            dec_pow(gl_from_word(runs[3 * (ulong)entry.y + 2]), column - start));
        value = gl_add(value, gl_mul(gl_from_word(chi[2 * row + component]), coefficient));
    }
    ulong output = (((ulong)group.x - shape[5]) * 2 + component) * RING_DEGREE + local;
    forms[output] = gl_add(gl_from_word(forms[output]), value);
}

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
    if (active < shape[5] || active >= shape[6]) {
        return;
    }
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

kernel void dec_add_bar_seeded_ring_forms(
    device const ulong *seeded_forms [[buffer(0)]],
    device ulong *forms [[buffer(1)]],
    device const uint *active_indices [[buffer(2)]],
    device const ulong *shape [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    ulong local = (ulong)index % RING_DEGREE;
    ulong base = (ulong)index - local;
    ulong rest = (ulong)index / RING_DEGREE;
    ulong component = rest % 2;
    ulong group = rest / 2;
    ulong active = active_indices[group];
    if (active < shape[5] || active >= shape[6]) {
        return;
    }
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
    ulong destination = ((active - shape[5]) * 2 + component) * RING_DEGREE + local;
    forms[destination] = gl_add(forms[destination], value);
}
