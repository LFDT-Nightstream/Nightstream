#include <metal_stdlib>

using namespace metal;

constant ulong GOLDILOCKS_MODULUS = 0xffffffff00000001ul;
constant ulong GOLDILOCKS_EPSILON = 0xfffffffful;
constant ulong LIMB_MASK = 0xfffffffful;
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

// A sum of full products as a 192-bit integer. It reduces one time:
// 2^128 = -2^32 in the field.
struct WideSum {
    ulong lo;
    ulong hi;
    ulong top;
};

inline void wide_sum_add(thread WideSum &sum, ulong lhs, ulong rhs) {
    WideProduct product = mul_wide_32(lhs, rhs);
    ulong lo = sum.lo + product.lo;
    ulong hi = sum.hi + product.hi;
    ulong carried = hi + (lo < sum.lo);
    sum.top += (hi < sum.hi) + (carried < hi);
    sum.lo = lo;
    sum.hi = carried;
}

inline ulong wide_sum_reduce(WideSum sum) {
    return gl_add(
        gl_reduce_wide(WideProduct{sum.lo, sum.hi}),
        gl_mul(sum.top, GOLDILOCKS_MODULUS - (1ul << 32)));
}

inline ulong gl_reduce_sum(ulong lo, ulong hi) {
    return gl_add(gl_from_word(lo), hi * GOLDILOCKS_EPSILON);
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

#include "dec_forms.metal"
#include "joint.metal"
#include "relation.metal"
#include "production_commitment.metal"
