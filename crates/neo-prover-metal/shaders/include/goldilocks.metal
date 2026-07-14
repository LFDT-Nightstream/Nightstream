#ifndef NIGHTSTREAM_GOLDILOCKS_METAL
#define NIGHTSTREAM_GOLDILOCKS_METAL

#include <metal_stdlib>
using namespace metal;

constant ulong GOLDILOCKS_MODULUS = 0xffffffff00000001UL;
constant ulong GOLDILOCKS_EPSILON = 0x00000000ffffffffUL;

inline ulong gl_from_word(ulong value) {
    return value >= GOLDILOCKS_MODULUS ? value - GOLDILOCKS_MODULUS : value;
}

inline ulong gl_add(ulong lhs, ulong rhs) {
    ulong sum = lhs + rhs;
    bool carry = sum < lhs;
    ulong out = carry ? sum + GOLDILOCKS_EPSILON : sum;
    return out >= GOLDILOCKS_MODULUS ? out - GOLDILOCKS_MODULUS : out;
}

inline ulong gl_sub(ulong lhs, ulong rhs) {
    ulong diff = lhs - rhs;
    return lhs < rhs ? diff - GOLDILOCKS_EPSILON : diff;
}

inline ulong gl_neg(ulong value) {
    return gl_sub(0, value);
}

inline ulong2 mul_wide(ulong lhs, ulong rhs) {
    ulong lhs_lo = lhs & GOLDILOCKS_EPSILON;
    ulong lhs_hi = lhs >> 32;
    ulong rhs_lo = rhs & GOLDILOCKS_EPSILON;
    ulong rhs_hi = rhs >> 32;

    ulong w0 = lhs_lo * rhs_lo;
    ulong t = lhs_hi * rhs_lo + (w0 >> 32);
    ulong w1 = t & GOLDILOCKS_EPSILON;
    ulong w2 = t >> 32;
    w1 = lhs_lo * rhs_hi + w1;

    ulong high = lhs_hi * rhs_hi + w2 + (w1 >> 32);
    ulong low = (w1 << 32) + (w0 & GOLDILOCKS_EPSILON);
    return ulong2(low, high);
}

inline ulong gl_reduce_wide(ulong low, ulong high) {
    ulong high_hi = high >> 32;
    ulong high_lo = high & GOLDILOCKS_EPSILON;
    ulong reduced_low = low - high_hi;
    if (low < high_hi) {
        reduced_low -= GOLDILOCKS_EPSILON;
    }
    ulong folded_high = (high_lo << 32) - high_lo;
    ulong sum = reduced_low + folded_high;
    ulong out = sum < reduced_low ? sum + GOLDILOCKS_EPSILON : sum;
    return out >= GOLDILOCKS_MODULUS ? out - GOLDILOCKS_MODULUS : out;
}

inline ulong gl_mul(ulong lhs, ulong rhs) {
    ulong2 product = mul_wide(lhs, rhs);
    return gl_reduce_wide(product.x, product.y);
}

inline ulong gl_mul_low_norm(ulong lhs, ulong rhs) {
    bool negative = rhs > GOLDILOCKS_MODULUS / 2;
    ulong magnitude = negative ? GOLDILOCKS_MODULUS - rhs : rhs;
    if ((magnitude >> 32) != 0) {
        return gl_mul(lhs, gl_from_word(rhs));
    }

    ulong p0 = (lhs & GOLDILOCKS_EPSILON) * magnitude;
    ulong p1 = (lhs >> 32) * magnitude;
    ulong shifted = p1 << 32;
    ulong low = p0 + shifted;
    ulong high = (p1 >> 32) + (low < p0 ? 1 : 0);
    ulong out = gl_reduce_wide(low, high);
    return negative ? gl_neg(out) : out;
}

#endif
