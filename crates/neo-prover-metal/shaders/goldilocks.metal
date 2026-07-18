#include <metal_stdlib>

// Root translation unit for every Nightstream Metal kernel.
// Host-visible field values are canonical unsigned Goldilocks words.

using namespace metal;

constant ulong GOLDILOCKS_MODULUS = 0xffffffff00000001ul;
constant ulong GOLDILOCKS_EPSILON = 0xfffffffful;
constant ulong LIMB_MASK = 0xfffffffful;
constant uint POSEIDON_WIDTH = 8;
constant uint POSEIDON_RATE = 4;
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

// Round constants come from the canonical Rust table embedded by build.rs.
struct PoseidonState {
    ulong s0;
    ulong s1;
    ulong s2;
    ulong s3;
    ulong s4;
    ulong s5;
    ulong s6;
    ulong s7;
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
    ulong4 mixed = ulong4(
        gl_add(a.x, b.x),
        gl_add(a.y, b.y),
        gl_add(a.z, b.z),
        gl_add(a.w, b.w));
    return PoseidonState{
        gl_add(a.x, mixed.x),
        gl_add(a.y, mixed.y),
        gl_add(a.z, mixed.z),
        gl_add(a.w, mixed.w),
        gl_add(b.x, mixed.x),
        gl_add(b.y, mixed.y),
        gl_add(b.z, mixed.z),
        gl_add(b.w, mixed.w)};
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
    return poseidon_mds_light(state);
}

inline PoseidonState poseidon_permute(PoseidonState state, constant const ulong *round_constants) {
    state = poseidon_mds_light(state);
    for (uint round = 0; round < POSEIDON_EXTERNAL_HALF_ROUNDS; ++round) {
        state = poseidon_external_round(state, round_constants, round * POSEIDON_WIDTH);
    }
    for (uint round = 0; round < POSEIDON_INTERNAL_ROUNDS; ++round) {
        state.s0 = gl_sbox(gl_add(state.s0, round_constants[POSEIDON_RC_INTERNAL + round]));
        ulong sum = gl_add(gl_add(gl_add(state.s0, state.s1), gl_add(state.s2, state.s3)),
                           gl_add(gl_add(state.s4, state.s5), gl_add(state.s6, state.s7)));
        state = PoseidonState{
            gl_add(gl_mul(state.s0, round_constants[POSEIDON_RC_DIAG]), sum),
            gl_add(gl_mul(state.s1, round_constants[POSEIDON_RC_DIAG + 1]), sum),
            gl_add(gl_mul(state.s2, round_constants[POSEIDON_RC_DIAG + 2]), sum),
            gl_add(gl_mul(state.s3, round_constants[POSEIDON_RC_DIAG + 3]), sum),
            gl_add(gl_mul(state.s4, round_constants[POSEIDON_RC_DIAG + 4]), sum),
            gl_add(gl_mul(state.s5, round_constants[POSEIDON_RC_DIAG + 5]), sum),
            gl_add(gl_mul(state.s6, round_constants[POSEIDON_RC_DIAG + 6]), sum),
            gl_add(gl_mul(state.s7, round_constants[POSEIDON_RC_DIAG + 7]), sum)};
    }
    for (uint round = 0; round < POSEIDON_EXTERNAL_HALF_ROUNDS; ++round) {
        state = poseidon_external_round(
            state,
            round_constants,
            POSEIDON_RC_TERMINAL + round * POSEIDON_WIDTH);
    }
    return state;
}

inline ulong poseidon_tile_shuffle(ulong value, ushort tile_base, ushort source_lane) {
    uint lo = simd_shuffle((uint)value, tile_base + source_lane);
    uint hi = simd_shuffle((uint)(value >> 32), tile_base + source_lane);
    return ((ulong)hi << 32) | lo;
}

inline ulong poseidon_shuffle_xor(ulong value, ushort mask) {
    uint lo = simd_shuffle_xor((uint)value, mask);
    uint hi = simd_shuffle_xor((uint)(value >> 32), mask);
    return ((ulong)hi << 32) | lo;
}

inline ulong poseidon_mds_light_simd(ulong state, ushort lane, ushort tile_base) {
    ushort half_base = lane & 4;
    ulong x0 = poseidon_tile_shuffle(state, tile_base, half_base);
    ulong x1 = poseidon_tile_shuffle(state, tile_base, half_base + 1);
    ulong x2 = poseidon_tile_shuffle(state, tile_base, half_base + 2);
    ulong x3 = poseidon_tile_shuffle(state, tile_base, half_base + 3);
    ulong t01 = gl_add(x0, x1);
    ulong t23 = gl_add(x2, x3);
    ulong t0123 = gl_add(t01, t23);
    ulong t01123 = gl_add(t0123, x1);
    ulong t01233 = gl_add(t0123, x3);
    ulong local;
    switch (lane & 3) {
        case 0: local = gl_add(t01123, t01); break;
        case 1: local = gl_add(gl_add(t01123, x2), x2); break;
        case 2: local = gl_add(t01233, t23); break;
        default: local = gl_add(gl_add(t01233, x0), x0); break;
    }
    ulong paired = poseidon_tile_shuffle(local, tile_base, lane ^ 4);
    return gl_add(gl_add(local, local), paired);
}

inline ulong poseidon_permute_simd(
    ulong state,
    ushort lane,
    ushort tile_base,
    constant const ulong *round_constants) {
    state = poseidon_mds_light_simd(state, lane, tile_base);
    for (uint round = 0; round < POSEIDON_EXTERNAL_HALF_ROUNDS; ++round) {
        state = gl_sbox(gl_add(state, round_constants[round * POSEIDON_WIDTH + lane]));
        state = poseidon_mds_light_simd(state, lane, tile_base);
    }
    for (uint round = 0; round < POSEIDON_INTERNAL_ROUNDS; ++round) {
        if (lane == 0) {
            state = gl_sbox(gl_add(state, round_constants[POSEIDON_RC_INTERNAL + round]));
        }
        ulong sum = state;
        sum = gl_add(sum, poseidon_shuffle_xor(sum, 1));
        sum = gl_add(sum, poseidon_shuffle_xor(sum, 2));
        sum = gl_add(sum, poseidon_shuffle_xor(sum, 4));
        state = gl_add(gl_mul(state, round_constants[POSEIDON_RC_DIAG + lane]), sum);
    }
    for (uint round = 0; round < POSEIDON_EXTERNAL_HALF_ROUNDS; ++round) {
        state = gl_sbox(gl_add(
            state,
            round_constants[POSEIDON_RC_TERMINAL + round * POSEIDON_WIDTH + lane]));
        state = poseidon_mds_light_simd(state, lane, tile_base);
    }
    return state;
}

inline PoseidonState poseidon_load(device const ulong *words, uint base) {
    return PoseidonState{
        gl_from_word(words[base]),
        gl_from_word(words[base + 1]),
        gl_from_word(words[base + 2]),
        gl_from_word(words[base + 3]),
        gl_from_word(words[base + 4]),
        gl_from_word(words[base + 5]),
        gl_from_word(words[base + 6]),
        gl_from_word(words[base + 7])};
}

inline void poseidon_store(device ulong *words, uint base, PoseidonState state) {
    words[base] = state.s0;
    words[base + 1] = state.s1;
    words[base + 2] = state.s2;
    words[base + 3] = state.s3;
    words[base + 4] = state.s4;
    words[base + 5] = state.s5;
    words[base + 6] = state.s6;
    words[base + 7] = state.s7;
}

// Device transcript state mirrors eight sponge words plus a rate cursor.
inline void transcript_set(thread PoseidonState &state, uint lane, ulong value) {
    switch (lane) {
        case 0: state.s0 = value; break;
        case 1: state.s1 = value; break;
        case 2: state.s2 = value; break;
        default: state.s3 = value; break;
    }
}

inline void transcript_absorb(
    thread PoseidonState &state,
    thread uint &cursor,
    ulong value,
    constant const ulong *round_constants) {
    if (cursor >= POSEIDON_RATE) {
        state = poseidon_permute(state, round_constants);
        cursor = 0;
    }
    transcript_set(state, cursor, gl_from_word(value));
    cursor += 1;
}

// Ring products are reduced modulo Phi_81 = X^54 + X^27 + 1.
inline ulong ring_convolution_coeff(
    device const ulong *matrix,
    device const ulong *message,
    ulong row,
    ulong cols,
    ulong exponent) {
    ulong accumulator = 0;
    ulong coefficient_start = exponent >= RING_DEGREE ? exponent - (RING_DEGREE - 1) : 0;
    ulong coefficient_end = min(exponent + 1, RING_DEGREE);
    for (ulong col = 0; col < cols; ++col) {
        ulong matrix_base = (row * cols + col) * RING_DEGREE;
        ulong message_base = col * RING_DEGREE;
        for (ulong coefficient = coefficient_start; coefficient < coefficient_end; ++coefficient) {
            accumulator = gl_add(
                accumulator,
                gl_mul(
                    gl_from_word(matrix[matrix_base + coefficient]),
                    gl_from_word(message[message_base + exponent - coefficient])));
        }
    }
    return accumulator;
}

kernel void goldilocks_ops(
    device const ulong *lhs [[buffer(0)]],
    device const ulong *rhs [[buffer(1)]],
    device ulong *output [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    ulong a = gl_from_word(lhs[index]);
    ulong b = gl_from_word(rhs[index]);
    output[3 * index] = gl_add(a, b);
    output[3 * index + 1] = gl_sub(a, b);
    output[3 * index + 2] = gl_mul(a, b);
}

kernel void goldilocks_ops_native(
    device const ulong *lhs [[buffer(0)]],
    device const ulong *rhs [[buffer(1)]],
    device ulong *output [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    ulong a = gl_from_word(lhs[index]);
    ulong b = gl_from_word(rhs[index]);
    output[3 * index] = gl_add(a, b);
    output[3 * index + 1] = gl_sub(a, b);
    output[3 * index + 2] = gl_mul_native(a, b);
}

kernel void copy_k_words(
    device const ulong *input [[buffer(0)]],
    device ulong *output [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    uint base = 2 * index;
    output[base] = input[base];
    output[base + 1] = input[base + 1];
}

kernel void kx_mul_add(
    device const ulong *state [[buffer(0)]],
    device const ulong *multiplier [[buffer(1)]],
    device ulong *output [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    uint base = 2 * index;
    Kx a = Kx{gl_from_word(state[base]), gl_from_word(state[base + 1])};
    Kx b = Kx{gl_from_word(multiplier[base]), gl_from_word(multiplier[base + 1])};
    Kx result = kx_add(kx_mul(a, b), a);
    output[base] = result.c0;
    output[base + 1] = result.c1;
}

kernel void poseidon2_permute_states(
    device ulong *states [[buffer(0)]],
    constant const ulong *round_constants [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    uint base = POSEIDON_WIDTH * index;
    PoseidonState state = poseidon_load(states, base);
    poseidon_store(states, base, poseidon_permute(state, round_constants));
}

kernel void poseidon2_hash_fields(
    device const ulong *fields [[buffer(0)]],
    device const ulong *offsets [[buffer(1)]],
    device const ulong *lengths [[buffer(2)]],
    device ulong *output [[buffer(3)]],
    constant const ulong *round_constants [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    PoseidonState state = PoseidonState{0, 0, 0, 0, 0, 0, 0, 0};
    ulong offset = offsets[index];
    ulong length = lengths[index];
    for (ulong position = 0; position < length; position += POSEIDON_RATE) {
        ulong take = min((ulong)POSEIDON_RATE, length - position);
        if (take > 0) state.s0 = gl_add(state.s0, gl_from_word(fields[offset + position]));
        if (take > 1) state.s1 = gl_add(state.s1, gl_from_word(fields[offset + position + 1]));
        if (take > 2) state.s2 = gl_add(state.s2, gl_from_word(fields[offset + position + 2]));
        if (take > 3) state.s3 = gl_add(state.s3, gl_from_word(fields[offset + position + 3]));
        state = poseidon_permute(state, round_constants);
    }
    state.s0 = gl_add(state.s0, 1);
    state = poseidon_permute(state, round_constants);
    uint base = POSEIDON_DIGEST * index;
    output[base] = state.s0;
    output[base + 1] = state.s1;
    output[base + 2] = state.s2;
    output[base + 3] = state.s3;
}

kernel void poseidon2_hash_fields_simd(
    device const ulong *fields [[buffer(0)]],
    device const ulong *offsets [[buffer(1)]],
    device const ulong *lengths [[buffer(2)]],
    device ulong *output [[buffer(3)]],
    constant const ulong *round_constants [[buffer(4)]],
    uint thread_index [[thread_position_in_grid]],
    ushort simd_lane [[thread_index_in_simdgroup]]) {
    ushort lane = thread_index & 7;
    ushort tile_base = simd_lane - lane;
    uint hash_index = thread_index / POSEIDON_WIDTH;
    ulong offset = offsets[hash_index];
    ulong length = lengths[hash_index];
    ulong state = 0;
    for (ulong position = 0; position < length; position += POSEIDON_RATE) {
        ulong take = min((ulong)POSEIDON_RATE, length - position);
        if (lane < take) {
            state = gl_add(state, gl_from_word(fields[offset + position + lane]));
        }
        state = poseidon_permute_simd(state, lane, tile_base, round_constants);
    }
    if (lane == 0) {
        state = gl_add(state, 1);
    }
    state = poseidon_permute_simd(state, lane, tile_base, round_constants);
    if (lane < POSEIDON_DIGEST) {
        output[POSEIDON_DIGEST * hash_index + lane] = state;
    }
}

kernel void poseidon2_hash_uniform(
    device const ulong *fields [[buffer(0)]],
    device ulong *output [[buffer(1)]],
    constant const ulong *round_constants [[buffer(2)]],
    device const ulong *shape [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    PoseidonState state = PoseidonState{0, 0, 0, 0, 0, 0, 0, 0};
    ulong length = shape[0];
    ulong offset = index * length;
    for (ulong position = 0; position < length; position += POSEIDON_RATE) {
        ulong take = min((ulong)POSEIDON_RATE, length - position);
        if (take > 0) state.s0 = gl_add(state.s0, gl_from_word(fields[offset + position]));
        if (take > 1) state.s1 = gl_add(state.s1, gl_from_word(fields[offset + position + 1]));
        if (take > 2) state.s2 = gl_add(state.s2, gl_from_word(fields[offset + position + 2]));
        if (take > 3) state.s3 = gl_add(state.s3, gl_from_word(fields[offset + position + 3]));
        state = poseidon_permute(state, round_constants);
    }
    state.s0 = gl_add(state.s0, 1);
    state = poseidon_permute(state, round_constants);
    uint base = POSEIDON_DIGEST * index;
    output[base] = state.s0;
    output[base + 1] = state.s1;
    output[base + 2] = state.s2;
    output[base + 3] = state.s3;
}

kernel void poseidon2_hash_uniform_simd(
    device const ulong *fields [[buffer(0)]],
    device ulong *output [[buffer(1)]],
    constant const ulong *round_constants [[buffer(2)]],
    device const ulong *shape [[buffer(3)]],
    uint thread_index [[thread_position_in_grid]],
    ushort simd_lane [[thread_index_in_simdgroup]]) {
    ushort lane = thread_index & 7;
    ushort tile_base = simd_lane - lane;
    uint hash_index = thread_index / POSEIDON_WIDTH;
    ulong length = shape[0];
    ulong offset = hash_index * length;
    ulong state = 0;
    for (ulong position = 0; position < length; position += POSEIDON_RATE) {
        ulong take = min((ulong)POSEIDON_RATE, length - position);
        if (lane < take) {
            state = gl_add(state, gl_from_word(fields[offset + position + lane]));
        }
        state = poseidon_permute_simd(state, lane, tile_base, round_constants);
    }
    if (lane == 0) {
        state = gl_add(state, 1);
    }
    state = poseidon_permute_simd(state, lane, tile_base, round_constants);
    if (lane < POSEIDON_DIGEST) {
        output[POSEIDON_DIGEST * hash_index + lane] = state;
    }
}

kernel void transcript_absorb_challenge2(
    device ulong *transcript_state [[buffer(0)]],
    device const ulong *fields [[buffer(1)]],
    device ulong *challenge [[buffer(2)]],
    constant const ulong *round_constants [[buffer(3)]],
    device const ulong *shape [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    if (index != 0) return;
    PoseidonState state = poseidon_load(transcript_state, 0);
    uint cursor = (uint)transcript_state[POSEIDON_WIDTH];
    ulong field_count = shape[0];
    transcript_absorb(state, cursor, field_count, round_constants);
    for (ulong field = 0; field < field_count; ++field) {
        transcript_absorb(state, cursor, fields[field], round_constants);
    }
    transcript_absorb(state, cursor, 1, round_constants);
    state = poseidon_permute(state, round_constants);
    cursor = 0;
    challenge[0] = state.s0;
    challenge[1] = state.s1;
    poseidon_store(transcript_state, 0, state);
    transcript_state[POSEIDON_WIDTH] = cursor;
}

kernel void ajtai_mat_vec(
    device const ulong *matrix [[buffer(0)]],
    device const ulong *message [[buffer(1)]],
    device ulong *output [[buffer(2)]],
    device const ulong *shape [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    ulong row = index / RING_DEGREE;
    ulong coefficient = index % RING_DEGREE;
    ulong cols = shape[1];
    ulong result = ring_convolution_coeff(matrix, message, row, cols, coefficient);
    if (coefficient <= 26) {
        result = gl_sub(
            result,
            ring_convolution_coeff(matrix, message, row, cols, coefficient + 54));
        if (coefficient <= 25) {
            result = gl_add(
                result,
                ring_convolution_coeff(matrix, message, row, cols, coefficient + 81));
        }
    } else {
        result = gl_sub(
            result,
            ring_convolution_coeff(matrix, message, row, cols, coefficient + 27));
    }
    output[index] = result;
}

inline WideProduct wide_sum_add(WideProduct sum, ulong value) {
    ulong next = sum.lo + value;
    sum.hi += next < sum.lo;
    sum.lo = next;
    return sum;
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

inline void accumulate_low_norm_mask(
    device const ulong *matrix,
    ulong matrix_base,
    ulong source,
    ulong positive_mask,
    ulong negative_mask,
    bool subtract,
    thread WideProduct &positive_sum,
    thread WideProduct &negative_sum) {
    ulong term_start = source >= RING_DEGREE - 1 ? source - (RING_DEGREE - 1) : 0;
    ulong term_end = min(source, RING_DEGREE - 1);
    ulong valid = (~0ul << term_start) & ((1ul << (term_end + 1)) - 1);
    ulong positive = (subtract ? negative_mask : positive_mask) & valid;
    ulong negative = (subtract ? positive_mask : negative_mask) & valid;
    while (positive != 0) {
        uint term = (uint)ctz(positive);
        positive &= positive - 1;
        positive_sum = wide_sum_add(positive_sum, matrix[matrix_base + source - term]);
    }
    while (negative != 0) {
        uint term = (uint)ctz(negative);
        negative &= negative - 1;
        negative_sum = wide_sum_add(negative_sum, matrix[matrix_base + source - term]);
    }
}

kernel void ajtai_low_norm_products(
    device const ulong *matrix [[buffer(0)]],
    device const ulong *message_masks [[buffer(1)]],
    device ulong *output [[buffer(2)]],
    device const ulong *shape [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    ulong row_col = index / RING_DEGREE;
    ulong row = row_col / shape[1];
    ulong col = row_col % shape[1];
    ulong coefficient = index % RING_DEGREE;
    ulong cols = shape[1];
    ulong matrix_base = (row * cols + col) * RING_DEGREE;
    ulong positive_mask = message_masks[2 * col];
    ulong negative_mask = message_masks[2 * col + 1];
    WideProduct positive_sum = WideProduct{0, 0};
    WideProduct negative_sum = WideProduct{0, 0};
    accumulate_low_norm_mask(
        matrix,
        matrix_base,
        coefficient,
        positive_mask,
        negative_mask,
        false,
        positive_sum,
        negative_sum);
    if (coefficient <= 26) {
        accumulate_low_norm_mask(
            matrix,
            matrix_base,
            coefficient + 54,
            positive_mask,
            negative_mask,
            true,
            positive_sum,
            negative_sum);
        if (coefficient <= 25) {
            accumulate_low_norm_mask(
                matrix,
                matrix_base,
                coefficient + 81,
                positive_mask,
                negative_mask,
                false,
                positive_sum,
                negative_sum);
        }
    } else {
        accumulate_low_norm_mask(
            matrix,
            matrix_base,
            coefficient + 27,
            positive_mask,
            negative_mask,
            true,
            positive_sum,
            negative_sum);
    }
    output[index] = gl_sub(
        gl_reduce_sum(positive_sum.lo, positive_sum.hi),
        gl_reduce_sum(negative_sum.lo, negative_sum.hi));
}

#include "seeded_ajtai.metal"

constant ulong SIS_BALANCED_TERNARY_SHIFT = 18236498188585393201ul;
constant ulong SIS_MODULUS_MINUS_SHIFT = 210245880829191120ul;
constant ulong SIS_BALANCED_TERNARY_DIGITS = 41ul;

kernel void sis_balanced_ternary_message(
    device const ulong *fields [[buffer(0)]],
    device char *message [[buffer(1)]],
    device const ulong *shape [[buffer(2)]],
    uint field [[thread_position_in_grid]]) {
    ulong field_count = shape[0];
    ulong message_cols = shape[1];
    if (field >= field_count || message_cols == 0) {
        return;
    }
    ulong value = fields[field];
    ulong remaining = value >= SIS_MODULUS_MINUS_SHIFT
        ? value - SIS_MODULUS_MINUS_SHIFT
        : value + SIS_BALANCED_TERNARY_SHIFT;
    for (ulong digit = 0; digit < SIS_BALANCED_TERNARY_DIGITS; ++digit) {
        ulong trit = remaining % 3;
        remaining /= 3;
        ulong logical = (ulong)field * SIS_BALANCED_TERNARY_DIGITS + digit;
        ulong row = logical / message_cols;
        ulong column = logical % message_cols;
        message[column * RING_DEGREE + row] = (char)((int)trit - 1);
    }
}

kernel void sis_pack_signed_masks(
    device const char *message [[buffer(0)]],
    device ulong *masks [[buffer(1)]],
    device const ulong *shape [[buffer(2)]],
    uint column [[thread_position_in_grid]]) {
    ulong field_count = shape[0];
    ulong message_cols = shape[1];
    if (column >= message_cols) {
        return;
    }
    ulong logical_len = field_count * SIS_BALANCED_TERNARY_DIGITS;
    ulong positive = 0;
    ulong negative = 0;
    for (ulong row = 0; row < RING_DEGREE; ++row) {
        ulong logical = row * message_cols + column;
        if (logical >= logical_len) {
            break;
        }
        char digit = message[(ulong)column * RING_DEGREE + row];
        if (digit > 0) {
            positive |= 1ul << row;
        } else if (digit < 0) {
            negative |= 1ul << row;
        }
    }
    masks[2 * column] = positive;
    masks[2 * column + 1] = negative;
}

kernel void ajtai_reduce_columns(
    device const ulong *input [[buffer(0)]],
    device ulong *output [[buffer(1)]],
    device const ulong *shape [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    ulong cols = shape[1];
    ulong next_cols = (cols + 1) / 2;
    ulong row_width = next_cols * RING_DEGREE;
    ulong row = index / row_width;
    ulong within_row = index % row_width;
    ulong out_col = within_row / RING_DEGREE;
    ulong coefficient = within_row % RING_DEGREE;
    ulong in_col = out_col * 2;
    ulong input_base = (row * cols + in_col) * RING_DEGREE + coefficient;
    ulong value = input[input_base];
    if (in_col + 1 < cols) {
        value = gl_add(value, input[input_base + RING_DEGREE]);
    }
    output[index] = value;
}

kernel void fold_k_table(
    device const ulong *table [[buffer(0)]],
    device const ulong *challenge_words [[buffer(1)]],
    device ulong *output [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    uint base = 4 * index;
    Kx left = Kx{gl_from_word(table[base]), gl_from_word(table[base + 1])};
    Kx right = Kx{gl_from_word(table[base + 2]), gl_from_word(table[base + 3])};
    Kx challenge = Kx{
        gl_from_word(challenge_words[0]),
        gl_from_word(challenge_words[1])};
    Kx folded = kx_add(left, kx_mul(challenge, kx_sub(right, left)));
    output[2 * index] = folded.c0;
    output[2 * index + 1] = folded.c1;
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

inline void poly_mul_affine(
    thread Kx *poly,
    Kx a,
    Kx b,
    uint current_degree) {
    Kx previous = Kx{0, 0};
    for (uint degree = 0; degree <= current_degree + 1; ++degree) {
        Kx old = poly[degree];
        poly[degree] = kx_add(kx_mul(a, old), kx_mul(b, previous));
        previous = old;
    }
}

kernel void fe_carried_mask_lin_comb(
    device const ulong *masks [[buffer(0)]],
    device const ulong *coeffs [[buffer(1)]],
    device const ulong *shape [[buffer(2)]],
    device ulong *z_re [[buffer(3)]],
    device ulong *z_im [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    ulong child_count = shape[0];
    ulong blocks = shape[1];
    ulong plane_len = blocks * RING_DEGREE;
    if (index >= plane_len) {
        return;
    }
    ulong block = index / RING_DEGREE;
    ulong lane = index % RING_DEGREE;
    ulong bit = 1ul << lane;
    ulong re = 0;
    ulong im = 0;
    for (ulong child = 0; child < child_count; ++child) {
        ulong mask_base = 2 * (child * blocks + block);
        ulong cr = gl_from_word(coeffs[2 * child]);
        ulong ci = gl_from_word(coeffs[2 * child + 1]);
        if ((masks[mask_base] & bit) != 0) {
            re = gl_add(re, cr);
            im = gl_add(im, ci);
        } else if ((masks[mask_base + 1] & bit) != 0) {
            re = gl_sub(re, cr);
            im = gl_sub(im, ci);
        }
    }
    z_re[index] = re;
    z_im[index] = im;
}

kernel void fe_weighted_basis_dots(
    device const ulong *basis_re [[buffer(0)]],
    device const ulong *basis_im [[buffer(1)]],
    device const ulong *z_re [[buffer(2)]],
    device const ulong *z_im [[buffer(3)]],
    device const ulong *shape [[buffer(4)]],
    device ulong *qk [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    ulong blocks = shape[1];
    if (index >= blocks * RING_DEGREE) {
        return;
    }
    ulong block = index / RING_DEGREE;
    ulong local = index % RING_DEGREE;
    ulong rr = 0;
    ulong ir = 0;
    ulong ri = 0;
    ulong ii = 0;
    for (ulong lane = 0; lane < RING_DEGREE; ++lane) {
        ulong fr = gl_from_word(basis_re[local * RING_DEGREE + lane]);
        ulong fi = gl_from_word(basis_im[local * RING_DEGREE + lane]);
        ulong zr = gl_from_word(z_re[block * RING_DEGREE + lane]);
        ulong zi = gl_from_word(z_im[block * RING_DEGREE + lane]);
        rr = gl_add(rr, gl_mul(fr, zr));
        ir = gl_add(ir, gl_mul(fi, zr));
        ri = gl_add(ri, gl_mul(fr, zi));
        ii = gl_add(ii, gl_mul(fi, zi));
    }
    qk[2 * index] = gl_add(rr, gl_mul(7, ii));
    qk[2 * index + 1] = gl_add(ir, ri);
}

kernel void fe_weighted_row_table(
    device const uint *matrix_row_offsets [[buffer(0)]],
    device const ulong *matrix_entry_bases [[buffer(1)]],
    device const uint *matrix_identity [[buffer(2)]],
    device const uint *entry_columns [[buffer(3)]],
    device const ulong *entry_coefficients [[buffer(4)]],
    device const ulong *qk [[buffer(5)]],
    device const ulong *mat_coeffs [[buffer(6)]],
    device const ulong *shape [[buffer(7)]],
    device ulong *output [[buffer(8)]],
    uint row [[thread_position_in_grid]]) {
    ulong matrix_count = shape[2];
    ulong rows = shape[3];
    ulong n_eff = shape[4];
    ulong n_pad = shape[5];
    if (row >= n_pad) {
        return;
    }
    Kx total = Kx{0, 0};
    if (row < n_eff) {
        for (ulong matrix = 0; matrix < matrix_count; ++matrix) {
            ulong offset = matrix * (rows + 1) + row;
            ulong entry_base = matrix_entry_bases[matrix];
            ulong start = entry_base + matrix_row_offsets[offset];
            ulong end = entry_base + matrix_row_offsets[offset + 1];
            Kx value = Kx{0, 0};
            if (matrix_identity[matrix] != 0) {
                value = load_k(qk, row);
            } else {
                for (ulong entry = start; entry < end; ++entry) {
                    ulong column = entry_columns[entry];
                    ulong coefficient = gl_from_word(entry_coefficients[entry]);
                    value.c0 = gl_add(value.c0, gl_mul(coefficient, gl_from_word(qk[2 * column])));
                    value.c1 = gl_add(value.c1, gl_mul(coefficient, gl_from_word(qk[2 * column + 1])));
                }
            }
            total = kx_add(
                total,
                kx_mul(Kx{mat_coeffs[2 * matrix], mat_coeffs[2 * matrix + 1]}, value));
        }
    }
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

// Pi_RLC consumes either signed masks, dense planes, or a resident mask tail.
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
        ulong mask_base = 2 * (input * cols + column);
        accumulate_signed_mask_rhos(
            rhos,
            rho_base,
            masks[mask_base],
            masks[mask_base + 1],
            positive,
            negative);
    }
    output[index] = gl_sub(
        gl_reduce_sum(positive.lo, positive.hi),
        gl_reduce_sum(negative.lo, negative.hi));
}

kernel void rlc_witness_mix(
    device const char *rhos [[buffer(0)]],
    device const ulong *witnesses [[buffer(1)]],
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
        ulong witness_base = input * RING_DEGREE * cols + column;
        for (ulong inner = 0; inner < RING_DEGREE; ++inner) {
            accumulate_small_signed(
                gl_from_word(witnesses[witness_base + inner * cols]),
                (int)rhos[rho_base + inner],
                positive,
                negative);
        }
    }
    output[index] = gl_sub(
        gl_reduce_sum(positive.lo, positive.hi),
        gl_reduce_sum(negative.lo, negative.hi));
}

kernel void rlc_witness_mix_dense_fresh_resident_masks(
    device const char *rhos [[buffer(0)]],
    device const ulong *fresh_witnesses [[buffer(1)]],
    device const ulong *resident_masks [[buffer(2)]],
    device const ulong *shape [[buffer(3)]],
    device ulong *output [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    ulong input_count = shape[0];
    ulong fresh_count = shape[1];
    ulong cols = shape[2];
    ulong row = index / cols;
    ulong column = index % cols;
    WideProduct positive = WideProduct{0, 0};
    WideProduct negative = WideProduct{0, 0};
    for (ulong input = 0; input < input_count; ++input) {
        ulong rho_base = input * RING_DEGREE * RING_DEGREE + row * RING_DEGREE;
        if (input >= fresh_count) {
            ulong mask_base = 2 * ((input - fresh_count) * cols + column);
            accumulate_signed_mask_rhos(
                rhos,
                rho_base,
                resident_masks[mask_base],
                resident_masks[mask_base + 1],
                positive,
                negative);
            continue;
        }
        ulong witness_base = input * RING_DEGREE * cols + column;
        for (ulong inner = 0; inner < RING_DEGREE; ++inner) {
            accumulate_small_signed(
                gl_from_word(fresh_witnesses[witness_base + inner * cols]),
                (int)rhos[rho_base + inner],
                positive,
                negative);
        }
    }
    output[index] = gl_sub(
        gl_reduce_sum(positive.lo, positive.hi),
        gl_reduce_sum(negative.lo, negative.hi));
}

#include "dec_forms.metal"
#include "lane_commitments.metal"
#include "dec_public.metal"

// Pi_DEC writes fourteen child masks per parent scan. The first group also checks
// that every centered coefficient fits in the fixed base-2 child count.
constant ushort DEC_SPLIT_CHILDREN_PER_THREAD = 14;

kernel void dec_split_base2_masks(
    device const ulong *parent [[buffer(0)]],
    device const ulong *shape [[buffer(1)]],
    device ulong *masks [[buffer(2)]],
    device atomic_uint *child_nonzero [[buffer(3)]],
    device atomic_uint *status [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    ulong child_count = shape[1];
    ulong cols = shape[3];
    ulong first_child = (index / cols) * DEC_SPLIT_CHILDREN_PER_THREAD;
    ulong column = index % cols;
    if (first_child >= child_count) {
        return;
    }

    ulong positive[DEC_SPLIT_CHILDREN_PER_THREAD];
    ulong negative[DEC_SPLIT_CHILDREN_PER_THREAD];
    for (ushort local = 0; local < DEC_SPLIT_CHILDREN_PER_THREAD; ++local) {
        positive[local] = 0;
        negative[local] = 0;
    }
    for (ulong coefficient = 0; coefficient < RING_DEGREE; ++coefficient) {
        ulong word = gl_from_word(parent[coefficient * cols + column]);
        bool is_negative = word > (GOLDILOCKS_MODULUS - 1) / 2;
        ulong magnitude = is_negative ? GOLDILOCKS_MODULUS - word : word;
        if (first_child == 0 && (magnitude >> child_count) != 0) {
            atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
        }
        for (ushort local = 0; local < DEC_SPLIT_CHILDREN_PER_THREAD; ++local) {
            ulong child = first_child + local;
            if (child < child_count && ((magnitude >> child) & 1ul) != 0) {
                if (is_negative) {
                    negative[local] |= 1ul << coefficient;
                } else {
                    positive[local] |= 1ul << coefficient;
                }
            }
        }
    }

    for (ushort local = 0; local < DEC_SPLIT_CHILDREN_PER_THREAD; ++local) {
        ulong child = first_child + local;
        if (child < child_count) {
            ulong mask_index = child * cols + column;
            masks[2 * mask_index] = positive[local];
            masks[2 * mask_index + 1] = negative[local];
            if ((positive[local] | negative[local]) != 0) {
                atomic_fetch_or_explicit(&child_nonzero[child], 1u, memory_order_relaxed);
            }
        }
    }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void dec_ring_partials(
    device const ulong *forms [[buffer(0)]],
    device const ulong *masks [[buffer(1)]],
    device const ulong *shape [[buffer(2)]],
    device ulong *partials [[buffer(3)]],
    device const uint *active_children [[buffer(4)]],
    device const uint *child_nonzero [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    ulong active_count = shape[1];
    ulong form_rows = shape[2];
    ulong cols = shape[3];
    ulong chunks = shape[4];
    ulong coefficient = index % RING_PRODUCT_COEFFICIENTS;
    ulong rest = index / RING_PRODUCT_COEFFICIENTS;
    ulong chunk = rest % chunks;
    ulong group = rest / chunks;
    ulong active_child = group / form_rows;
    ulong form_row = group % form_rows;
    if (active_child >= active_count) {
        return;
    }
    ulong child = active_children[active_child];
    if (shape[5] != 0 && child_nonzero[child] == 0) {
        partials[index] = 0;
        return;
    }

    ulong column_start = chunk * DEC_CHUNK_COLUMNS;
    ulong column_end = min(column_start + DEC_CHUNK_COLUMNS, cols);
    ulong term_start = coefficient >= RING_DEGREE ? coefficient - (RING_DEGREE - 1) : 0;
    ulong term_end = coefficient < RING_DEGREE ? coefficient : RING_DEGREE - 1;
    ulong valid = (~0ul << term_start) & ((1ul << (term_end + 1)) - 1);
    ulong positive_lo = 0;
    ulong positive_hi = 0;
    ulong negative_lo = 0;
    ulong negative_hi = 0;
    for (ulong column = column_start; column < column_end; ++column) {
        ulong mask_base = 2 * (child * cols + column);
        ulong positive = masks[mask_base] & valid;
        while (positive != 0) {
            uint term = (uint)ctz(positive);
            positive &= positive - 1;
            ulong value = forms[(form_row * cols + column) * RING_DEGREE + coefficient - term];
            ulong next = positive_lo + value;
            positive_hi += next < positive_lo;
            positive_lo = next;
        }
        ulong negative = masks[mask_base + 1] & valid;
        while (negative != 0) {
            uint term = (uint)ctz(negative);
            negative &= negative - 1;
            ulong value = forms[(form_row * cols + column) * RING_DEGREE + coefficient - term];
            ulong next = negative_lo + value;
            negative_hi += next < negative_lo;
            negative_lo = next;
        }
    }
    partials[index] = gl_sub(gl_reduce_sum(positive_lo, positive_hi), gl_reduce_sum(negative_lo, negative_hi));
}

[[max_total_threads_per_threadgroup(128)]]
kernel void dec_sparse_ring_partials(
    device const ulong *forms [[buffer(0)]],
    device const ulong *masks [[buffer(1)]],
    device const ulong *shape [[buffer(2)]],
    device ulong *partials [[buffer(3)]],
    device const uint *active_children [[buffer(4)]],
    device const uint *active_blocks [[buffer(5)]],
    device const uint *active_chunk_bases [[buffer(6)]],
    device const uint *active_chunk_matrices [[buffer(7)]],
    device const uint *matrix_active_offsets [[buffer(8)]],
    device const uint *child_nonzero [[buffer(9)]],
    uint index [[thread_position_in_grid]]) {
    ulong active_count = shape[1];
    ulong form_rows = shape[2];
    ulong blocks = shape[3];
    ulong chunk_count = shape[4];
    ulong coefficient = index % RING_PRODUCT_COEFFICIENTS;
    ulong rest = index / RING_PRODUCT_COEFFICIENTS;
    ulong component = rest % 2;
    rest /= 2;
    ulong chunk = rest % chunk_count;
    ulong active_child = rest / chunk_count;
    if (active_child >= active_count) {
        return;
    }
    ulong child = active_children[active_child];
    if (shape[5] != 0 && child_nonzero[child] == 0) {
        partials[index] = 0;
        return;
    }
    ulong matrix = active_chunk_matrices[chunk];
    if (2 * matrix + component >= form_rows) {
        return;
    }
    ulong start = active_chunk_bases[chunk];
    ulong end = min(start + DEC_CHUNK_COLUMNS, (ulong)matrix_active_offsets[matrix + 1]);
    ulong term_start = coefficient >= RING_DEGREE ? coefficient - (RING_DEGREE - 1) : 0;
    ulong term_end = coefficient < RING_DEGREE ? coefficient : RING_DEGREE - 1;
    ulong valid = (~0ul << term_start) & ((1ul << (term_end + 1)) - 1);
    ulong positive_lo = 0;
    ulong positive_hi = 0;
    ulong negative_lo = 0;
    ulong negative_hi = 0;
    for (ulong active = start; active < end; ++active) {
        ulong block = (ulong)active_blocks[active] % blocks;
        ulong mask_base = 2 * (child * blocks + block);
        ulong positive = masks[mask_base] & valid;
        while (positive != 0) {
            uint term = (uint)ctz(positive);
            positive &= positive - 1;
            ulong value = forms[(active * 2 + component) * RING_DEGREE + coefficient - term];
            ulong next = positive_lo + value;
            positive_hi += next < positive_lo;
            positive_lo = next;
        }
        ulong negative = masks[mask_base + 1] & valid;
        while (negative != 0) {
            uint term = (uint)ctz(negative);
            negative &= negative - 1;
            ulong value = forms[(active * 2 + component) * RING_DEGREE + coefficient - term];
            ulong next = negative_lo + value;
            negative_hi += next < negative_lo;
            negative_lo = next;
        }
    }
    partials[index] = gl_sub(gl_reduce_sum(positive_lo, positive_hi), gl_reduce_sum(negative_lo, negative_hi));
}

kernel void dec_ring_sum_chunks(
    device const ulong *partials [[buffer(0)]],
    device const ulong *shape [[buffer(1)]],
    device ulong *sums [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    ulong chunks = shape[4];
    ulong group = index / RING_PRODUCT_COEFFICIENTS;
    ulong coefficient = index % RING_PRODUCT_COEFFICIENTS;
    ulong value = 0;
    for (ulong chunk = 0; chunk < chunks; ++chunk) {
        value = gl_add(value, partials[(group * chunks + chunk) * RING_PRODUCT_COEFFICIENTS + coefficient]);
    }
    sums[index] = value;
}

kernel void dec_sparse_ring_sum_chunks(
    device const ulong *partials [[buffer(0)]],
    device const ulong *shape [[buffer(1)]],
    device ulong *sums [[buffer(2)]],
    device const uint *matrix_chunk_offsets [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    ulong form_rows = shape[2];
    ulong chunk_count = shape[4];
    ulong group = index / RING_PRODUCT_COEFFICIENTS;
    ulong coefficient = index % RING_PRODUCT_COEFFICIENTS;
    ulong child = group / form_rows;
    ulong form_row = group % form_rows;
    ulong matrix = form_row / 2;
    ulong component = form_row % 2;
    ulong value = 0;
    ulong start = matrix_chunk_offsets[matrix];
    ulong end = matrix_chunk_offsets[matrix + 1];
    for (ulong chunk = start; chunk < end; ++chunk) {
        ulong partial = ((child * chunk_count + chunk) * 2 + component) * RING_PRODUCT_COEFFICIENTS + coefficient;
        value = gl_add(value, partials[partial]);
    }
    sums[index] = value;
}

kernel void dec_ring_reduce_phi81(
    device const ulong *sums [[buffer(0)]],
    device const ulong *shape [[buffer(1)]],
    device ulong *output [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    ulong groups = shape[1] * shape[2];
    ulong group = index / RING_DEGREE;
    ulong coefficient = index % RING_DEGREE;
    if (group >= groups) {
        return;
    }
    ulong base = group * RING_PRODUCT_COEFFICIENTS;
    ulong value = gl_from_word(sums[base + coefficient]);
    if (coefficient <= 26) {
        value = gl_sub(value, gl_from_word(sums[base + coefficient + 54]));
        if (coefficient <= 25) {
            value = gl_add(value, gl_from_word(sums[base + coefficient + 81]));
        }
    } else {
        value = gl_sub(value, gl_from_word(sums[base + coefficient + 27]));
    }
    output[index] = value;
}

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

kernel void fe_round_partials(
    device const ulong *tables [[buffer(0)]],
    device const ulong *shape [[buffer(1)]],
    device const ulong *mcs_headers [[buffer(2)]],
    device const ulong *mcs_table_indices [[buffer(3)]],
    device const ulong *gammas [[buffer(4)]],
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
    ulong eq_table = shape[4];
    ulong eq_inputs_plus_one = shape[5];
    ulong eval_plus_one = shape[6];
    ulong mcs_count = shape[7];
    ulong term_count = shape[8];
    Kx f_at_zero = Kx{shape[9], shape[10]};
    Kx gamma_to_k = Kx{shape[11], shape[12]};
    threadgroup Kx shared[SUMCHECK_REDUCTION_THREADS * SUMCHECK_MAX_COEFFS];
    Kx local[SUMCHECK_MAX_COEFFS];
    for (uint degree = 0; degree < SUMCHECK_MAX_COEFFS; ++degree) {
        local[degree] = Kx{0, 0};
    }
    ulong pairs = (active_len + 1) / 2;
    if (pair < pairs) {
        ulong index = 2 * pair;
        Kx eq0 = load_k(tables, eq_table * table_len + index);
        Kx eq1 = kx_sub(load_k(tables, eq_table * table_len + index + 1), eq0);
        Kx inner[SUMCHECK_MAX_COEFFS];
        for (uint degree = 0; degree < SUMCHECK_MAX_COEFFS; ++degree) {
            inner[degree] = Kx{0, 0};
        }
        for (ulong mcs = 0; mcs < mcs_count; ++mcs) {
            ulong header = 3 * mcs;
            bool is_zero = mcs_headers[header] != 0;
            ulong table_start = mcs_headers[header + 1];
            ulong table_count = mcs_headers[header + 2];
            Kx gamma = load_k(gammas, mcs);
            if (is_zero) {
                inner[0] = kx_add(inner[0], kx_mul(f_at_zero, gamma));
                continue;
            }
            for (ulong term = 0; term < term_count; ++term) {
                ulong term_header = 4 * term;
                Kx term_coefficient = Kx{term_headers[term_header], term_headers[term_header + 1]};
                ulong variable_start = term_headers[term_header + 2];
                ulong variable_count = term_headers[term_header + 3];
                Kx polynomial[SUMCHECK_MAX_COEFFS];
                for (uint degree = 0; degree < SUMCHECK_MAX_COEFFS; ++degree) {
                    polynomial[degree] = Kx{0, 0};
                }
                polynomial[0] = kx_mul(term_coefficient, gamma);
                uint current_degree = 0;
                for (ulong variable = 0; variable < variable_count; ++variable) {
                    ulong variable_header = 2 * (variable_start + variable);
                    ulong variable_position = term_variables[variable_header];
                    uint exponent = (uint)term_variables[variable_header + 1];
                    if (variable_position >= table_count) {
                        continue;
                    }
                    ulong table_index = mcs_table_indices[table_start + variable_position];
                    Kx a = load_k(tables, table_index * table_len + index);
                    Kx b = kx_sub(load_k(tables, table_index * table_len + index + 1), a);
                    for (uint power = 0; power < exponent && current_degree < row_degree; ++power) {
                        poly_mul_affine(polynomial, a, b, current_degree);
                        current_degree += 1;
                    }
                }
                uint limit = min(current_degree, row_degree);
                for (uint degree = 0; degree <= limit; ++degree) {
                    inner[degree] = kx_add(inner[degree], polynomial[degree]);
                }
            }
        }
        local[0] = kx_mul(eq0, inner[0]);
        for (uint coefficient = 1; coefficient < active_coefficients; ++coefficient) {
            local[coefficient] = kx_add(
                kx_mul(eq0, inner[coefficient]),
                kx_mul(eq1, inner[coefficient - 1]));
        }
        if (eq_inputs_plus_one != 0 && eval_plus_one != 0) {
            ulong eq_inputs = eq_inputs_plus_one - 1;
            ulong eval = eval_plus_one - 1;
            Kx r0 = load_k(tables, eq_inputs * table_len + index);
            Kx r1 = kx_sub(load_k(tables, eq_inputs * table_len + index + 1), r0);
            Kx v0 = load_k(tables, eval * table_len + index);
            Kx v1 = kx_sub(load_k(tables, eval * table_len + index + 1), v0);
            local[0] = kx_add(local[0], kx_mul(gamma_to_k, kx_mul(r0, v0)));
            local[1] = kx_add(local[1], kx_mul(gamma_to_k, kx_add(kx_mul(r0, v1), kx_mul(r1, v0))));
            local[2] = kx_add(local[2], kx_mul(gamma_to_k, kx_mul(r1, v1)));
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
        for (uint coefficient = 0; coefficient < coefficient_count; ++coefficient) {
            Kx value = coefficient < active_coefficients ? shared[coefficient] : Kx{0, 0};
            ulong output_index = group * coefficient_count + coefficient;
            partials[2 * output_index] = value.c0;
            partials[2 * output_index + 1] = value.c1;
        }
    }
}

// Phase-specific kernels share the arithmetic and ABI helpers above.
#include "nc.metal"
#include "oracle.metal"
