#include <metal_stdlib>
#include "profile.metal"

using namespace metal;

// `build.rs` writes `profile.metal` from `neo-math`: FQ_MODULUS, RING_DEGREE
// and KX_NON_RESIDUE.

// 2^64 mod q.
constant ulong FQ_FOLD = 0ul - FQ_MODULUS;
constant ulong LIMB_MASK = 0xfffffffful;
constant ulong RING_PRODUCT_COEFFICIENTS = 2 * RING_DEGREE - 1;
constant ulong DEC_CHUNK_COLUMNS = 512;

static_assert(FQ_FOLD < (1ul << 16), "fq_reduce_wide folds the high word two times");
static_assert(RING_DEGREE == 64, "a mask word holds one bit for each coefficient");

struct WideProduct {
    ulong lo;
    ulong hi;
};

inline ulong fq_from_word(ulong value) {
    return value >= FQ_MODULUS ? value - FQ_MODULUS : value;
}

inline ulong fq_add(ulong lhs, ulong rhs) {
    ulong sum = lhs + rhs;
    return sum < lhs || sum >= FQ_MODULUS ? sum - FQ_MODULUS : sum;
}

inline ulong fq_sub(ulong lhs, ulong rhs) {
    ulong difference = lhs - rhs;
    return lhs < rhs ? difference + FQ_MODULUS : difference;
}

inline ulong fq_neg(ulong value) {
    return value == 0 ? 0 : FQ_MODULUS - value;
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

// The field value of any 128-bit integer. 2^64 = FQ_FOLD in the field, so
// hi * 2^64 + lo folds to hi * FQ_FOLD + lo, and the carry word of that sum
// folds one more time.
inline ulong fq_reduce_wide(WideProduct value) {
    ulong low_part = (value.hi & LIMB_MASK) * FQ_FOLD;
    ulong high_part = (value.hi >> 32) * FQ_FOLD + (low_part >> 32);
    ulong folded = value.lo + ((high_part << 32) | (low_part & LIMB_MASK));
    // At most FQ_FOLD: the high word of hi * FQ_FOLD is below FQ_FOLD.
    ulong top = (high_part >> 32) + (folded < value.lo);
    ulong reduced = folded + top * FQ_FOLD;
    // After a carry, `reduced` is below FQ_FOLD^2, and the dropped 2^64 adds FQ_FOLD.
    return reduced < folded || reduced >= FQ_MODULUS ? reduced + FQ_FOLD : reduced;
}

inline ulong fq_mul(ulong lhs, ulong rhs) {
    return fq_reduce_wide(mul_wide_32(lhs, rhs));
}

// The most terms that one `ProductSum` holds: a term adds less than
// 3 * 2^32 to a column.
constant ulong PRODUCT_SUM_TERMS = ~0ul / (3 * LIMB_MASK);

// A sum of full products of 64-bit words. The four columns hold the 32-bit
// limbs of the products at the weights 1, 2^32, 2^64 and 2^96, and no add
// needs a carry. The sum is exact for PRODUCT_SUM_TERMS terms, and it
// reduces one time.
struct ProductSum {
    ulong c0;
    ulong c1;
    ulong c2;
    ulong c3;
};

inline void product_sum_add(thread ProductSum &sum, ulong lhs, ulong rhs) {
    ulong lhs_lo = lhs & LIMB_MASK;
    ulong lhs_hi = lhs >> 32;
    ulong rhs_lo = rhs & LIMB_MASK;
    ulong rhs_hi = rhs >> 32;
    ulong p00 = lhs_lo * rhs_lo;
    ulong p01 = lhs_lo * rhs_hi;
    ulong p10 = lhs_hi * rhs_lo;
    ulong p11 = lhs_hi * rhs_hi;
    sum.c0 += p00 & LIMB_MASK;
    sum.c1 += (p00 >> 32) + (p01 & LIMB_MASK) + (p10 & LIMB_MASK);
    sum.c2 += (p01 >> 32) + (p10 >> 32) + (p11 & LIMB_MASK);
    sum.c3 += p11 >> 32;
}

// The columns join to a 192-bit integer; 2^128 = FQ_FOLD^2 in the field.
inline ulong product_sum_reduce(ProductSum sum) {
    ulong lo = sum.c0 + (sum.c1 << 32);
    ulong carried = (sum.c1 >> 32) + (lo < sum.c0);
    ulong middle = carried + sum.c2;
    ulong hi = middle + (sum.c3 << 32);
    ulong top = (sum.c3 >> 32) + (middle < carried) + (hi < middle);
    return fq_add(fq_reduce_wide(WideProduct{lo, hi}), fq_mul(top, FQ_FOLD * FQ_FOLD));
}

// A sum of 64-bit words as the sums of their 32-bit halves, so no add needs
// a carry. Each half is below 2^32, so the sum is exact for 2^32 terms.
struct HalfSums {
    ulong lo;
    ulong hi;
};

inline void half_sums_add(thread HalfSums &sum, ulong word) {
    sum.lo += word & LIMB_MASK;
    sum.hi += word >> 32;
}

inline ulong half_sums_reduce(HalfSums sum) {
    ulong low = sum.lo + (sum.hi << 32);
    return fq_reduce_wide(WideProduct{low, (sum.hi >> 32) + (low < sum.lo)});
}

// Quadratic extension layout is interleaved c0, c1 with u^2 = KX_NON_RESIDUE.
struct Kx {
    ulong c0;
    ulong c1;
};

inline Kx kx_add(Kx lhs, Kx rhs) {
    return Kx{fq_add(lhs.c0, rhs.c0), fq_add(lhs.c1, rhs.c1)};
}

inline Kx kx_sub(Kx lhs, Kx rhs) {
    return Kx{fq_sub(lhs.c0, rhs.c0), fq_sub(lhs.c1, rhs.c1)};
}

static_assert(KX_NON_RESIDUE == 7, "fq_mul_by_non_residue adds seven copies");

inline ulong fq_mul_by_non_residue(ulong value) {
    ulong twice = fq_add(value, value);
    ulong four_times = fq_add(twice, twice);
    return fq_add(fq_add(four_times, twice), value);
}

inline Kx kx_mul(Kx lhs, Kx rhs) {
    ulong c0_product = fq_mul(lhs.c0, rhs.c0);
    ulong c1_product = fq_mul(lhs.c1, rhs.c1);
    ulong cross_product = fq_sub(
        fq_sub(fq_mul(fq_add(lhs.c0, lhs.c1), fq_add(rhs.c0, rhs.c1)), c0_product),
        c1_product);
    ulong c0 = fq_add(c0_product, fq_mul_by_non_residue(c1_product));
    ulong c1 = cross_product;
    return Kx{c0, c1};
}

// A 128-bit sum of products of a 64-bit word and a 32-bit scale. Each term
// is below 2^96, so the sum is exact for 2^32 terms.
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
    return Kx{fq_from_word(values[2 * index]), fq_from_word(values[2 * index + 1])};
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
                        ulong scalar = fq_from_word(dense_coefficients[coefficient]);
                        value.c0 = fq_add(value.c0, fq_mul(scalar, fq_from_word(qk[2 * column])));
                        value.c1 = fq_add(value.c1, fq_mul(scalar, fq_from_word(qk[2 * column + 1])));
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
                    ulong scalar = fq_from_word(geometric_runs[3 * run + 1]);
                    ulong ratio = fq_from_word(geometric_runs[3 * run + 2]);
                    for (; column < run_end; ++column) {
                        Kx input = load_k(qk, column);
                        value.c0 = fq_add(value.c0, fq_mul(scalar, input.c0));
                        value.c1 = fq_add(value.c1, fq_mul(scalar, input.c1));
                        scalar = fq_mul(scalar, ratio);
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
    output[index] = fq_sub(fq_reduce_wide(positive), fq_reduce_wide(negative));
}

#include "decomposition.metal"

constant uint SUMCHECK_REDUCTION_THREADS = 64;

#include "dec_forms.metal"
#include "joint.metal"
#include "relation.metal"
#include "production_commitment.metal"
