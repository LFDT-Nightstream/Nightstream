#include <metal_stdlib>

using namespace metal;

constant ulong GOLDILOCKS_MODULUS = 0xffffffff00000001ul;
constant ulong GOLDILOCKS_EPSILON = 0xfffffffful;
constant ulong LIMB_MASK = 0xfffffffful;
constant uint POSEIDON_WIDTH = 16;
constant uint POSEIDON_RATE = 12;
constant uint POSEIDON_DIGEST = 4;
constant uint POSEIDON_EXTERNAL_HALF_ROUNDS = 4;
constant uint POSEIDON_INTERNAL_ROUNDS = 22;
constant uint POSEIDON_RC_INTERNAL = POSEIDON_EXTERNAL_HALF_ROUNDS * POSEIDON_WIDTH;
constant uint POSEIDON_RC_TERMINAL = POSEIDON_RC_INTERNAL + POSEIDON_INTERNAL_ROUNDS;
constant uint POSEIDON_RC_DIAG = POSEIDON_RC_TERMINAL + POSEIDON_EXTERNAL_HALF_ROUNDS * POSEIDON_WIDTH;
constant ulong RING_DEGREE = 54;
constant ulong RING_PRODUCT_COEFFICIENTS = 107;
constant ulong DEC_CHUNK_COLUMNS = 512;

struct WideProduct {
    ulong lo;
    ulong hi;
};

inline ulong gl_from_word(ulong value) {
    return value >= GOLDILOCKS_MODULUS ? value - GOLDILOCKS_MODULUS : value;
}

inline ulong gl_add(ulong lhs, ulong rhs) {
    ulong sum = lhs + rhs;
    bool carry = sum < lhs;
    if (carry) {
        sum += GOLDILOCKS_EPSILON;
    }
    if (sum >= GOLDILOCKS_MODULUS) {
        sum -= GOLDILOCKS_MODULUS;
    }
    return sum;
}

inline ulong gl_sub(ulong lhs, ulong rhs) {
    ulong difference = lhs - rhs;
    if (lhs < rhs) {
        difference -= GOLDILOCKS_EPSILON;
    }
    return difference;
}

inline WideProduct mul_wide_32(ulong lhs, ulong rhs) {
    ulong lhs_lo = lhs & LIMB_MASK;
    ulong lhs_hi = lhs >> 32;
    ulong rhs_lo = rhs & LIMB_MASK;
    ulong rhs_hi = rhs >> 32;

    ulong p00 = lhs_lo * rhs_lo;
    ulong p01 = lhs_lo * rhs_hi;
    ulong p10 = lhs_hi * rhs_lo;
    ulong p11 = lhs_hi * rhs_hi;
    ulong middle = (p00 >> 32) + (p01 & LIMB_MASK) + (p10 & LIMB_MASK);

    WideProduct out;
    out.lo = (p00 & LIMB_MASK) | (middle << 32);
    out.hi = p11 + (p01 >> 32) + (p10 >> 32) + (middle >> 32);
    return out;
}

inline WideProduct mul_wide_native(ulong lhs, ulong rhs) {
    return WideProduct{lhs * rhs, mulhi(lhs, rhs)};
}

inline ulong gl_reduce_wide(WideProduct value) {
    ulong hi_hi = value.hi >> 32;
    ulong hi_lo = value.hi & GOLDILOCKS_EPSILON;
    ulong reduced_lo = value.lo - hi_hi;
    if (value.lo < hi_hi) {
        reduced_lo -= GOLDILOCKS_EPSILON;
    }
    ulong folded_hi = (hi_lo << 32) - hi_lo;
    return gl_add(reduced_lo, folded_hi);
}

inline ulong gl_mul(ulong lhs, ulong rhs) {
    return gl_reduce_wide(mul_wide_32(lhs, rhs));
}

inline ulong gl_mul_native(ulong lhs, ulong rhs) {
    return gl_reduce_wide(mul_wide_native(lhs, rhs));
}

inline ulong gl_reduce_sum(ulong lo, ulong hi) {
    return gl_add(gl_from_word(lo), hi * GOLDILOCKS_EPSILON);
}

inline ulong gl_sbox(ulong value) {
    ulong square = gl_mul(value, value);
    ulong cube = gl_mul(square, value);
    return gl_mul(gl_mul(cube, cube), value);
}

// Quadratic extension layout is interleaved c0, c1 with u^2 = 7.
struct Kx {
    ulong c0;
    ulong c1;
};

inline Kx kx_add(Kx lhs, Kx rhs) {
    return Kx{gl_add(lhs.c0, rhs.c0), gl_add(lhs.c1, rhs.c1)};
}

inline Kx kx_sub(Kx lhs, Kx rhs) {
    return Kx{gl_sub(lhs.c0, rhs.c0), gl_sub(lhs.c1, rhs.c1)};
}

inline ulong gl_mul_by_7(ulong value) {
    ulong twice = gl_add(value, value);
    ulong four_times = gl_add(twice, twice);
    return gl_add(gl_add(four_times, twice), value);
}

inline Kx kx_mul(Kx lhs, Kx rhs) {
    ulong c0_product = gl_mul(lhs.c0, rhs.c0);
    ulong c1_product = gl_mul(lhs.c1, rhs.c1);
    ulong cross_product = gl_sub(
        gl_sub(gl_mul(gl_add(lhs.c0, lhs.c1), gl_add(rhs.c0, rhs.c1)), c0_product),
        c1_product);
    ulong c0 = gl_add(c0_product, gl_mul_by_7(c1_product));
    ulong c1 = cross_product;
    return Kx{c0, c1};
}

struct PoseidonState {
    ulong s0;
    ulong s1;
    ulong s2;
    ulong s3;
    ulong s4;
    ulong s5;
    ulong s6;
    ulong s7;
    ulong s8;
    ulong s9;
    ulong s10;
    ulong s11;
    ulong s12;
    ulong s13;
    ulong s14;
    ulong s15;
};

inline ulong4 poseidon_mat4(ulong x0, ulong x1, ulong x2, ulong x3) {
    ulong t01 = gl_add(x0, x1);
    ulong t23 = gl_add(x2, x3);
    ulong t0123 = gl_add(t01, t23);
    ulong t01123 = gl_add(t0123, x1);
    ulong t01233 = gl_add(t0123, x3);
    return ulong4(
        gl_add(t01123, t01),
        gl_add(gl_add(t01123, x2), x2),
        gl_add(t01233, t23),
        gl_add(gl_add(t01233, x0), x0));
}

inline PoseidonState poseidon_mds_light(PoseidonState state) {
    ulong4 a = poseidon_mat4(state.s0, state.s1, state.s2, state.s3);
    ulong4 b = poseidon_mat4(state.s4, state.s5, state.s6, state.s7);
    ulong4 c = poseidon_mat4(state.s8, state.s9, state.s10, state.s11);
    ulong4 d = poseidon_mat4(state.s12, state.s13, state.s14, state.s15);
    ulong4 mixed = ulong4(
        gl_add(gl_add(a.x, b.x), gl_add(c.x, d.x)),
        gl_add(gl_add(a.y, b.y), gl_add(c.y, d.y)),
        gl_add(gl_add(a.z, b.z), gl_add(c.z, d.z)),
        gl_add(gl_add(a.w, b.w), gl_add(c.w, d.w)));
    return PoseidonState{
        gl_add(a.x, mixed.x),
        gl_add(a.y, mixed.y),
        gl_add(a.z, mixed.z),
        gl_add(a.w, mixed.w),
        gl_add(b.x, mixed.x),
        gl_add(b.y, mixed.y),
        gl_add(b.z, mixed.z),
        gl_add(b.w, mixed.w),
        gl_add(c.x, mixed.x),
        gl_add(c.y, mixed.y),
        gl_add(c.z, mixed.z),
        gl_add(c.w, mixed.w),
        gl_add(d.x, mixed.x),
        gl_add(d.y, mixed.y),
        gl_add(d.z, mixed.z),
        gl_add(d.w, mixed.w)};
}

inline PoseidonState poseidon_external_round(
    PoseidonState state,
    constant const ulong *round_constants,
    uint base) {
    state.s0 = gl_sbox(gl_add(state.s0, round_constants[base]));
    state.s1 = gl_sbox(gl_add(state.s1, round_constants[base + 1]));
    state.s2 = gl_sbox(gl_add(state.s2, round_constants[base + 2]));
    state.s3 = gl_sbox(gl_add(state.s3, round_constants[base + 3]));
    state.s4 = gl_sbox(gl_add(state.s4, round_constants[base + 4]));
    state.s5 = gl_sbox(gl_add(state.s5, round_constants[base + 5]));
    state.s6 = gl_sbox(gl_add(state.s6, round_constants[base + 6]));
    state.s7 = gl_sbox(gl_add(state.s7, round_constants[base + 7]));
    state.s8 = gl_sbox(gl_add(state.s8, round_constants[base + 8]));
    state.s9 = gl_sbox(gl_add(state.s9, round_constants[base + 9]));
    state.s10 = gl_sbox(gl_add(state.s10, round_constants[base + 10]));
    state.s11 = gl_sbox(gl_add(state.s11, round_constants[base + 11]));
    state.s12 = gl_sbox(gl_add(state.s12, round_constants[base + 12]));
    state.s13 = gl_sbox(gl_add(state.s13, round_constants[base + 13]));
    state.s14 = gl_sbox(gl_add(state.s14, round_constants[base + 14]));
    state.s15 = gl_sbox(gl_add(state.s15, round_constants[base + 15]));
    return poseidon_mds_light(state);
}

inline PoseidonState poseidon_permute(PoseidonState state, constant const ulong *round_constants) {
    state = poseidon_mds_light(state);
    for (uint round = 0; round < POSEIDON_EXTERNAL_HALF_ROUNDS; ++round) {
        state = poseidon_external_round(state, round_constants, round * POSEIDON_WIDTH);
    }
    for (uint round = 0; round < POSEIDON_INTERNAL_ROUNDS; ++round) {
        state.s0 = gl_sbox(gl_add(state.s0, round_constants[POSEIDON_RC_INTERNAL + round]));
        ulong sum = gl_add(
            gl_add(gl_add(gl_add(state.s0, state.s1), gl_add(state.s2, state.s3)),
                   gl_add(gl_add(state.s4, state.s5), gl_add(state.s6, state.s7))),
            gl_add(gl_add(gl_add(state.s8, state.s9), gl_add(state.s10, state.s11)),
                   gl_add(gl_add(state.s12, state.s13), gl_add(state.s14, state.s15))));
        state = PoseidonState{
            gl_add(gl_mul(state.s0, round_constants[POSEIDON_RC_DIAG]), sum),
            gl_add(gl_mul(state.s1, round_constants[POSEIDON_RC_DIAG + 1]), sum),
            gl_add(gl_mul(state.s2, round_constants[POSEIDON_RC_DIAG + 2]), sum),
            gl_add(gl_mul(state.s3, round_constants[POSEIDON_RC_DIAG + 3]), sum),
            gl_add(gl_mul(state.s4, round_constants[POSEIDON_RC_DIAG + 4]), sum),
            gl_add(gl_mul(state.s5, round_constants[POSEIDON_RC_DIAG + 5]), sum),
            gl_add(gl_mul(state.s6, round_constants[POSEIDON_RC_DIAG + 6]), sum),
            gl_add(gl_mul(state.s7, round_constants[POSEIDON_RC_DIAG + 7]), sum),
            gl_add(gl_mul(state.s8, round_constants[POSEIDON_RC_DIAG + 8]), sum),
            gl_add(gl_mul(state.s9, round_constants[POSEIDON_RC_DIAG + 9]), sum),
            gl_add(gl_mul(state.s10, round_constants[POSEIDON_RC_DIAG + 10]), sum),
            gl_add(gl_mul(state.s11, round_constants[POSEIDON_RC_DIAG + 11]), sum),
            gl_add(gl_mul(state.s12, round_constants[POSEIDON_RC_DIAG + 12]), sum),
            gl_add(gl_mul(state.s13, round_constants[POSEIDON_RC_DIAG + 13]), sum),
            gl_add(gl_mul(state.s14, round_constants[POSEIDON_RC_DIAG + 14]), sum),
            gl_add(gl_mul(state.s15, round_constants[POSEIDON_RC_DIAG + 15]), sum)};
    }
    for (uint round = 0; round < POSEIDON_EXTERNAL_HALF_ROUNDS; ++round) {
        state = poseidon_external_round(
            state,
            round_constants,
            POSEIDON_RC_TERMINAL + round * POSEIDON_WIDTH);
    }
    return state;
}

// Merkle leaf digests of rows: Plonky3's overwrite sponge (PaddingFreeSponge).
// Each row starts from the zero state; each block of up to POSEIDON_RATE words
// overwrites the first lanes and is followed by one permutation.
kernel void poseidon2_hash_rows_overwrite(
    device const ulong *fields [[buffer(0)]],
    device ulong *output [[buffer(1)]],
    constant const ulong *round_constants [[buffer(2)]],
    device const ulong *shape [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    PoseidonState state = PoseidonState{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    ulong width = shape[0];
    ulong offset = index * width;
    for (ulong position = 0; position < width; position += POSEIDON_RATE) {
        ulong take = min((ulong)POSEIDON_RATE, width - position);
        if (take > 0) state.s0 = gl_from_word(fields[offset + position]);
        if (take > 1) state.s1 = gl_from_word(fields[offset + position + 1]);
        if (take > 2) state.s2 = gl_from_word(fields[offset + position + 2]);
        if (take > 3) state.s3 = gl_from_word(fields[offset + position + 3]);
        if (take > 4) state.s4 = gl_from_word(fields[offset + position + 4]);
        if (take > 5) state.s5 = gl_from_word(fields[offset + position + 5]);
        if (take > 6) state.s6 = gl_from_word(fields[offset + position + 6]);
        if (take > 7) state.s7 = gl_from_word(fields[offset + position + 7]);
        if (take > 8) state.s8 = gl_from_word(fields[offset + position + 8]);
        if (take > 9) state.s9 = gl_from_word(fields[offset + position + 9]);
        if (take > 10) state.s10 = gl_from_word(fields[offset + position + 10]);
        if (take > 11) state.s11 = gl_from_word(fields[offset + position + 11]);
        state = poseidon_permute(state, round_constants);
    }
    uint base = POSEIDON_DIGEST * index;
    output[base] = state.s0;
    output[base + 1] = state.s1;
    output[base + 2] = state.s2;
    output[base + 3] = state.s3;
}

inline WideProduct wide_sum_add_small(WideProduct sum, ulong value, uint scale) {
    ulong low_product = (value & LIMB_MASK) * scale;
    ulong high_product = (value >> 32) * scale;
    ulong product_lo = low_product + (high_product << 32);
    ulong product_hi = (high_product >> 32) + (product_lo < low_product);
    ulong next = sum.lo + product_lo;
    sum.hi += product_hi + (next < sum.lo);
    sum.lo = next;
    return sum;
}

inline void accumulate_small_signed(
    ulong value,
    int coefficient,
    thread WideProduct &positive,
    thread WideProduct &negative) {
    uint magnitude = coefficient < 0 ? (uint)(-coefficient) : (uint)coefficient;
    if (coefficient > 0) {
        positive = wide_sum_add_small(positive, value, magnitude);
    } else if (coefficient < 0) {
        negative = wide_sum_add_small(negative, value, magnitude);
    }
}

constant uint SUMCHECK_MAX_COEFFS = 10;

inline Kx load_k(device const ulong *values, ulong index) {
    return Kx{gl_from_word(values[2 * index]), gl_from_word(values[2 * index + 1])};
}

kernel void tensor_point_expand_k(
    device const ulong *challenges [[buffer(0)]],
    device const ulong *stage_words [[buffer(1)]],
    device ulong *table [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    ulong stage = stage_words[0];
    ulong step = 1ul << stage;
    if ((ulong)index >= step) {
        return;
    }
    Kx value = stage == 0 ? Kx{1, 0} : load_k(table, index);
    Kx high = kx_mul(value, load_k(challenges, stage));
    Kx low = kx_sub(value, high);
    table[2 * index] = low.c0;
    table[2 * index + 1] = low.c1;
    table[2 * (index + step)] = high.c0;
    table[2 * (index + step) + 1] = high.c1;
}

inline ulong compact_row_offset(device const uchar *offsets, ulong row, ulong width) {
    if (width == 3) {
        ulong base = 3 * row;
        return (ulong)offsets[base] | ((ulong)offsets[base + 1] << 8) | ((ulong)offsets[base + 2] << 16);
    }
    return width == 0 ? 0 : (ulong)((device const uint *)offsets)[row];
}

constant uint COMPACT_DENSE_BLOCK_TAG = 1u << 31;
constant uint COMPACT_NEGATIVE_BLOCK_TAG = 1u << 30;
constant uint COMPACT_SINGLE_LOCAL_SHIFT = 24;
constant uint COMPACT_SINGLE_LOCAL_MASK = 0x3fu;
constant uint COMPACT_SINGLE_BLOCK_MASK = (1u << COMPACT_SINGLE_LOCAL_SHIFT) - 1;
constant uint COMPACT_DENSE_INDEX_MASK = COMPACT_DENSE_BLOCK_TAG - 1;

kernel void fe_weighted_row_table(
    device const uchar *row_offsets [[buffer(0)]],
    device const uint *row_blocks [[buffer(1)]],
    device const uint *dense_offsets [[buffer(2)]],
    device const uchar *dense_locals [[buffer(3)]],
    device const ulong *dense_coefficients [[buffer(4)]],
    device const uchar *geometric_row_offsets [[buffer(5)]],
    device const ulong *geometric_runs [[buffer(6)]],
    device const ulong *qk [[buffer(7)]],
    device const ulong *matrix_coefficient [[buffer(8)]],
    device const ulong *shape [[buffer(9)]],
    device ulong *output [[buffer(10)]],
    device const uint2 *dense_row_blocks [[buffer(11)]],
    uint local_row [[thread_position_in_grid]]) {
    ulong rows = shape[0];
    ulong n_eff = shape[1];
    ulong n_pad = shape[2];
    ulong offset_width = shape[3];
    bool identity = shape[4] != 0;
    ulong geometric_offset_width = shape[5];
    ulong row = shape[6] + (ulong)local_row;
    if ((ulong)local_row >= rows || row >= n_pad) {
        return;
    }
    Kx value = Kx{0, 0};
    if (row < n_eff) {
        if (identity) {
            value = load_k(qk, row);
        } else {
            ulong start = compact_row_offset(row_offsets, local_row, offset_width);
            ulong end = compact_row_offset(row_offsets, (ulong)local_row + 1, offset_width);
            for (ulong entry = start; entry < end; ++entry) {
                uint reference = row_blocks[entry];
                if ((reference & COMPACT_DENSE_BLOCK_TAG) == 0) {
                    ulong block = (ulong)(reference & COMPACT_SINGLE_BLOCK_MASK);
                    ulong local = (ulong)((reference >> COMPACT_SINGLE_LOCAL_SHIFT) & COMPACT_SINGLE_LOCAL_MASK);
                    ulong column = block * RING_DEGREE + local;
                    Kx input = load_k(qk, column);
                    value = (reference & COMPACT_NEGATIVE_BLOCK_TAG) == 0
                        ? kx_add(value, input)
                        : kx_sub(value, input);
                } else {
                    uint2 block = dense_row_blocks[reference & COMPACT_DENSE_INDEX_MASK];
                    uint dense = block.y;
                    for (uint coefficient = dense_offsets[dense]; coefficient < dense_offsets[dense + 1]; ++coefficient) {
                        ulong column = (ulong)block.x * RING_DEGREE + (ulong)dense_locals[coefficient];
                        ulong scalar = gl_from_word(dense_coefficients[coefficient]);
                        value.c0 = gl_add(value.c0, gl_mul(scalar, gl_from_word(qk[2 * column])));
                        value.c1 = gl_add(value.c1, gl_mul(scalar, gl_from_word(qk[2 * column + 1])));
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
                    ulong scalar = gl_from_word(geometric_runs[3 * run + 1]);
                    ulong ratio = gl_from_word(geometric_runs[3 * run + 2]);
                    for (; column < run_end; ++column) {
                        Kx input = load_k(qk, column);
                        value.c0 = gl_add(value.c0, gl_mul(scalar, input.c0));
                        value.c1 = gl_add(value.c1, gl_mul(scalar, input.c1));
                        scalar = gl_mul(scalar, ratio);
                    }
                }
            }
        }
    }
    Kx total = kx_add(load_k(output, row), kx_mul(load_k(matrix_coefficient, 0), value));
    output[2 * row] = total.c0;
    output[2 * row + 1] = total.c1;
}

inline void accumulate_signed_mask_rhos(
    device const char *rhos,
    ulong rho_base,
    ulong positive_mask,
    ulong negative_mask,
    thread WideProduct &positive,
    thread WideProduct &negative) {
    while (positive_mask != 0) {
        uint inner = (uint)ctz(positive_mask);
        positive_mask &= positive_mask - 1;
        accumulate_small_signed(1ul, (int)rhos[rho_base + inner], positive, negative);
    }
    while (negative_mask != 0) {
        uint inner = (uint)ctz(negative_mask);
        negative_mask &= negative_mask - 1;
        accumulate_small_signed(1ul, -(int)rhos[rho_base + inner], positive, negative);
    }
}

// Signed masks are planar: the positive columns, then the negative columns, of each input.
kernel void rlc_witness_mix_signed_masks(
    device const char *rhos [[buffer(0)]],
    device const ulong *masks [[buffer(1)]],
    device const ulong *shape [[buffer(2)]],
    device ulong *output [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    ulong input_count = shape[0];
    ulong cols = shape[1];
    ulong row = index / cols;
    ulong column = index % cols;
    WideProduct positive = WideProduct{0, 0};
    WideProduct negative = WideProduct{0, 0};
    for (ulong input = 0; input < input_count; ++input) {
        ulong rho_base = input * RING_DEGREE * RING_DEGREE + row * RING_DEGREE;
        ulong mask_base = 2 * input * cols + column;
        accumulate_signed_mask_rhos(
            rhos,
            rho_base,
            masks[mask_base],
            masks[mask_base + cols],
            positive,
            negative);
    }
    output[index] = gl_sub(
        gl_reduce_sum(positive.lo, positive.hi),
        gl_reduce_sum(negative.lo, negative.hi));
}

#include "decomposition.metal"

constant uint SUMCHECK_REDUCTION_THREADS = 64;

kernel void sumcheck_reduce_partials(
    device const ulong *partials [[buffer(0)]],
    device const ulong *shape [[buffer(1)]],
    device ulong *output [[buffer(2)]],
    uint coefficient [[thread_position_in_grid]]) {
    ulong rows = shape[0];
    ulong coefficient_count = shape[1];
    if (coefficient >= coefficient_count) {
        return;
    }
    Kx total = Kx{0, 0};
    for (ulong row = 0; row < rows; ++row) {
        total = kx_add(total, load_k(partials, row * coefficient_count + coefficient));
    }
    output[2 * coefficient] = total.c0;
    output[2 * coefficient + 1] = total.c1;
}

#include "dec_forms.metal"
#include "joint.metal"
#include "relation.metal"
#include "production_commitment.metal"
#include "dft.metal"
#include "gkr.metal"
