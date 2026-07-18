#include "include/goldilocks.metal"

kernel void goldilocks_add(
    device const ulong *lhs [[buffer(0)]],
    device const ulong *rhs [[buffer(1)]],
    device ulong *out [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    out[index] = gl_add(gl_from_word(lhs[index]), gl_from_word(rhs[index]));
}

kernel void goldilocks_sub(
    device const ulong *lhs [[buffer(0)]],
    device const ulong *rhs [[buffer(1)]],
    device ulong *out [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    out[index] = gl_sub(gl_from_word(lhs[index]), gl_from_word(rhs[index]));
}

kernel void goldilocks_mul(
    device const ulong *lhs [[buffer(0)]],
    device const ulong *rhs [[buffer(1)]],
    device ulong *out [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    out[index] = gl_mul(gl_from_word(lhs[index]), gl_from_word(rhs[index]));
}

kernel void goldilocks_mul_low_norm(
    device const ulong *lhs [[buffer(0)]],
    device const ulong *rhs [[buffer(1)]],
    device ulong *out [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    out[index] = gl_mul_low_norm(gl_from_word(lhs[index]), gl_from_word(rhs[index]));
}

kernel void goldilocks_extension_mul(
    device const ulong *lhs [[buffer(0)]],
    device const ulong *rhs [[buffer(1)]],
    device ulong *out [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    ulong lhs0 = gl_from_word(lhs[2 * index]);
    ulong lhs1 = gl_from_word(lhs[2 * index + 1]);
    ulong rhs0 = gl_from_word(rhs[2 * index]);
    ulong rhs1 = gl_from_word(rhs[2 * index + 1]);
    out[2 * index] = gl_add(gl_mul(lhs0, rhs0), gl_mul(gl_mul(lhs1, rhs1), 7));
    out[2 * index + 1] = gl_add(gl_mul(lhs0, rhs1), gl_mul(lhs1, rhs0));
}
