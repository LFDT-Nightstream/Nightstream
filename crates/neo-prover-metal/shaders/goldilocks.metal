#include <metal_stdlib>

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
constant ulong DEC_CHUNK_COLUMNS = 64;

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

inline ulong gl_scale_small_signed(ulong value, int coefficient) {
    uint magnitude = coefficient < 0 ? (uint)(-coefficient) : (uint)coefficient;
    ulong scaled = 0;
    for (uint i = 0; i < magnitude; ++i) {
        scaled = gl_add(scaled, value);
    }
    return coefficient < 0 ? gl_sub(0, scaled) : scaled;
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

inline Kx kx_mul(Kx lhs, Kx rhs) {
    ulong c0 = gl_add(gl_mul(lhs.c0, rhs.c0), gl_mul(gl_mul(lhs.c1, rhs.c1), 7ul));
    ulong c1 = gl_add(gl_mul(lhs.c0, rhs.c1), gl_mul(lhs.c1, rhs.c0));
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

inline ulong add_low_norm_digit(ulong accumulator, ulong value, char digit) {
    if (digit > 0) return gl_add(accumulator, value);
    if (digit < 0) return gl_sub(accumulator, value);
    return accumulator;
}

inline ulong sub_low_norm_digit(ulong accumulator, ulong value, char digit) {
    if (digit > 0) return gl_sub(accumulator, value);
    if (digit < 0) return gl_add(accumulator, value);
    return accumulator;
}

kernel void ajtai_low_norm_products(
    device const ulong *matrix [[buffer(0)]],
    device const char *message [[buffer(1)]],
    device ulong *output [[buffer(2)]],
    device const ulong *shape [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    ulong row_col = index / RING_DEGREE;
    ulong row = row_col / shape[1];
    ulong col = row_col % shape[1];
    ulong coefficient = index % RING_DEGREE;
    ulong cols = shape[1];
    ulong accumulator = 0;
    ulong matrix_base = (row * cols + col) * RING_DEGREE;
    ulong message_base = col * RING_DEGREE;
    for (ulong shift = 0; shift < RING_DEGREE; ++shift) {
        char digit = message[message_base + shift];
        if (digit == 0) continue;
        if (coefficient >= shift) {
            accumulator = add_low_norm_digit(
                accumulator,
                gl_from_word(matrix[matrix_base + coefficient - shift]),
                digit);
        }
        if (coefficient <= 26) {
            ulong source = coefficient + 54;
            if (source >= shift && source - shift < RING_DEGREE) {
                accumulator = sub_low_norm_digit(
                    accumulator,
                    gl_from_word(matrix[matrix_base + source - shift]),
                    digit);
            }
            if (coefficient <= 25) {
                source = coefficient + 81;
                if (source >= shift && source - shift < RING_DEGREE) {
                    accumulator = add_low_norm_digit(
                        accumulator,
                        gl_from_word(matrix[matrix_base + source - shift]),
                        digit);
                }
            }
        } else {
            ulong source = coefficient + 27;
            if (source >= shift && source - shift < RING_DEGREE) {
                accumulator = add_low_norm_digit(
                    accumulator,
                    gl_from_word(matrix[matrix_base + source - shift]),
                    -digit);
            }
        }
    }
    output[index] = accumulator;
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

constant uint SUMCHECK_MAX_COEFFS = 9;

inline Kx load_k(device const ulong *values, ulong index) {
    return Kx{gl_from_word(values[2 * index]), gl_from_word(values[2 * index + 1])};
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

kernel void nc_fold_compact(
    device const ulong *input [[buffer(0)]],
    device const ulong *challenge_words [[buffer(1)]],
    device const ulong *shape [[buffer(2)]],
    device ulong *output [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    ulong witness_count = shape[0];
    ulong rows = shape[1];
    ulong width = shape[2];
    bool dense = shape[3] != 0;
    bool output_dense = dense || 2 * width > RING_DEGREE;
    ulong half_rows = (rows + 1) / 2;
    ulong input_per_witness = dense ? rows * RING_DEGREE : rows * width;
    ulong output_width = output_dense ? RING_DEGREE : 2 * width;
    ulong output_per_witness = half_rows * output_width;
    ulong witness = index / output_per_witness;
    ulong within = index % output_per_witness;
    ulong out_row = within / output_width;
    ulong slot = within % output_width;
    if (witness >= witness_count) {
        return;
    }
    ulong input_base = witness * input_per_witness;
    ulong lo_row = 2 * out_row;
    ulong hi_row = lo_row + 1;
    Kx challenge = Kx{challenge_words[0], challenge_words[1]};
    Kx lo = Kx{0, 0};
    Kx hi = Kx{0, 0};
    if (!output_dense) {
        if (slot < width) {
            lo = load_k(input, input_base + lo_row * width + slot);
        } else if (hi_row < rows) {
            hi = load_k(input, input_base + hi_row * width + slot - width);
        }
    } else if (dense) {
        lo = load_k(input, input_base + lo_row * RING_DEGREE + slot);
        if (hi_row < rows) {
            hi = load_k(input, input_base + hi_row * RING_DEGREE + slot);
        }
    } else {
        ulong start_lo = (lo_row * width) % RING_DEGREE;
        ulong lo_slot = (slot + RING_DEGREE - start_lo) % RING_DEGREE;
        if (lo_slot < width) {
            lo = load_k(input, input_base + lo_row * width + lo_slot);
        }
        if (hi_row < rows) {
            ulong start_hi = (hi_row * width) % RING_DEGREE;
            ulong hi_slot = (slot + RING_DEGREE - start_hi) % RING_DEGREE;
            if (hi_slot < width) {
                hi = load_k(input, input_base + hi_row * width + hi_slot);
            }
        }
    }
    Kx folded = kx_add(lo, kx_mul(challenge, kx_sub(hi, lo)));
    output[2 * index] = folded.c0;
    output[2 * index + 1] = folded.c1;
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
    ulong value = 0;
    for (ulong input = 0; input < input_count; ++input) {
        ulong rho_base = input * RING_DEGREE * RING_DEGREE + row * RING_DEGREE;
        ulong witness_base = input * RING_DEGREE * cols + column;
        for (ulong inner = 0; inner < RING_DEGREE; ++inner) {
            value = gl_add(
                value,
                gl_scale_small_signed(
                    gl_from_word(witnesses[witness_base + inner * cols]),
                    (int)rhos[rho_base + inner]));
        }
    }
    output[index] = value;
}

kernel void rlc_witness_mix_resident_tail(
    device const char *rhos [[buffer(0)]],
    device const ulong *fresh_witnesses [[buffer(1)]],
    device const ulong *resident_witnesses [[buffer(2)]],
    device const ulong *shape [[buffer(3)]],
    device ulong *output [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    ulong input_count = shape[0];
    ulong fresh_count = shape[1];
    ulong cols = shape[2];
    ulong row = index / cols;
    ulong column = index % cols;
    ulong value = 0;
    for (ulong input = 0; input < input_count; ++input) {
        ulong rho_base = input * RING_DEGREE * RING_DEGREE + row * RING_DEGREE;
        ulong witness_index = input < fresh_count ? input : input - fresh_count;
        device const ulong *witnesses = input < fresh_count ? fresh_witnesses : resident_witnesses;
        ulong witness_base = witness_index * RING_DEGREE * cols + column;
        for (ulong inner = 0; inner < RING_DEGREE; ++inner) {
            value = gl_add(
                value,
                gl_scale_small_signed(
                    gl_from_word(witnesses[witness_base + inner * cols]),
                    (int)rhos[rho_base + inner]));
        }
    }
    output[index] = value;
}

kernel void dec_split_base2(
    device const ulong *parent [[buffer(0)]],
    device const ulong *shape [[buffer(1)]],
    device ulong *children [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    ulong entries = shape[0];
    ulong child_count = shape[1];
    ulong word = gl_from_word(parent[index]);
    long value = word <= (GOLDILOCKS_MODULUS - 1) / 2
        ? (long)word
        : -((long)(GOLDILOCKS_MODULUS - word));
    for (ulong child = 0; child < child_count; ++child) {
        long digit;
        if ((value & 1l) == 0) {
            digit = 0;
            value >>= 1;
        } else if (value > 0) {
            digit = 1;
            value = (value - 1) >> 1;
        } else {
            digit = -1;
            value = (value + 1) >> 1;
        }
        children[child * entries + index] = digit < 0 ? GOLDILOCKS_MODULUS - 1 : (ulong)digit;
    }
}

kernel void dec_validate_split(
    device const ulong *parent [[buffer(0)]],
    device const ulong *children [[buffer(1)]],
    device const ulong *shape [[buffer(2)]],
    device atomic_uint *status [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    ulong entries = shape[0];
    ulong child_count = shape[1];
    if (index >= entries) {
        return;
    }
    ulong recomposed = 0;
    ulong power = 1;
    bool valid = true;
    for (ulong child = 0; child < child_count; ++child) {
        ulong digit = gl_from_word(children[child * entries + index]);
        valid &= digit == 0 || digit == 1 || digit == GOLDILOCKS_MODULUS - 1;
        recomposed = gl_add(recomposed, gl_mul(power, digit));
        power = gl_add(power, power);
    }
    valid &= recomposed == gl_from_word(parent[index]);
    if (!valid) {
        atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
    }
}

kernel void dec_build_ring_forms(
    device const ulong *matrix_block_offsets [[buffer(0)]],
    device const ulong *entry_rows [[buffer(1)]],
    device const ulong *entry_bars [[buffer(2)]],
    device const ulong *chi [[buffer(3)]],
    device const ulong *shape [[buffer(4)]],
    device ulong *forms [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    ulong matrix_count = shape[0];
    ulong blocks = shape[1];
    ulong n_eff = shape[2];
    ulong chi_len = shape[3];
    ulong coefficient = index % RING_DEGREE;
    ulong rest = index / RING_DEGREE;
    ulong block = rest % blocks;
    ulong form_row = rest / blocks;
    ulong matrix = form_row / 2;
    ulong component = form_row % 2;
    if (matrix >= matrix_count) {
        return;
    }
    ulong offset_base = matrix * (blocks + 1) + block;
    ulong start = matrix_block_offsets[offset_base];
    ulong end = matrix_block_offsets[offset_base + 1];
    ulong value = 0;
    for (ulong entry = start; entry < end; ++entry) {
        ulong row = entry_rows[entry];
        if (row >= n_eff || row >= chi_len) {
            continue;
        }
        value = gl_add(
            value,
            gl_mul(
                gl_from_word(chi[2 * row + component]),
                gl_from_word(entry_bars[entry * RING_DEGREE + coefficient])));
    }
    forms[index] = value;
}

kernel void dec_binary_masks(
    device const ulong *children [[buffer(0)]],
    device const ulong *shape [[buffer(1)]],
    device ulong *masks [[buffer(2)]],
    device atomic_uint *child_nonzero [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    ulong entries = shape[0];
    ulong child_count = shape[1];
    ulong cols = shape[3];
    ulong child = index / cols;
    ulong column = index % cols;
    if (child >= child_count) {
        return;
    }
    ulong positive = 0;
    ulong negative = 0;
    ulong base = child * entries + column;
    for (ulong coefficient = 0; coefficient < RING_DEGREE; ++coefficient) {
        ulong value = gl_from_word(children[base + coefficient * cols]);
        positive |= (ulong)(value == 1) << coefficient;
        negative |= (ulong)(value == GOLDILOCKS_MODULUS - 1) << coefficient;
    }
    masks[2 * index] = positive;
    masks[2 * index + 1] = negative;
    if ((positive | negative) != 0) {
        atomic_fetch_or_explicit(&child_nonzero[child], 1u, memory_order_relaxed);
    }
}

kernel void dec_ring_partials(
    device const ulong *forms [[buffer(0)]],
    device const ulong *masks [[buffer(1)]],
    device const ulong *shape [[buffer(2)]],
    device ulong *partials [[buffer(3)]],
    device const uint *active_children [[buffer(4)]],
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

kernel void dec_ring_reduce_phi81(
    device const ulong *sums [[buffer(0)]],
    device const ulong *shape [[buffer(1)]],
    device ulong *output [[buffer(2)]],
    uint group [[thread_position_in_grid]]) {
    ulong groups = shape[1] * shape[2];
    if (group >= groups) {
        return;
    }
    ulong values[RING_PRODUCT_COEFFICIENTS];
    ulong base = (ulong)group * RING_PRODUCT_COEFFICIENTS;
    for (ulong coefficient = 0; coefficient < RING_PRODUCT_COEFFICIENTS; ++coefficient) {
        values[coefficient] = gl_from_word(sums[base + coefficient]);
    }
    for (int coefficient = (int)RING_PRODUCT_COEFFICIENTS - 1; coefficient >= (int)RING_DEGREE; --coefficient) {
        ulong value = values[coefficient];
        values[coefficient] = 0;
        values[coefficient - (int)RING_DEGREE] = gl_sub(values[coefficient - (int)RING_DEGREE], value);
        int middle = coefficient - 27;
        if (middle < (int)RING_DEGREE) {
            values[middle] = gl_sub(values[middle], value);
        } else {
            values[middle - (int)RING_DEGREE] = gl_add(values[middle - (int)RING_DEGREE], value);
            if (middle - 27 < (int)RING_DEGREE) {
                values[middle - 27] = gl_add(values[middle - 27], value);
            }
        }
    }
    ulong output_base = (ulong)group * RING_DEGREE;
    for (ulong coefficient = 0; coefficient < RING_DEGREE; ++coefficient) {
        output[output_base + coefficient] = values[coefficient];
    }
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
        for (uint coefficient = 1; coefficient < coefficient_count; ++coefficient) {
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
    for (uint coefficient = 0; coefficient < coefficient_count; ++coefficient) {
        shared[lane * SUMCHECK_MAX_COEFFS + coefficient] = local[coefficient];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = SUMCHECK_REDUCTION_THREADS / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            for (uint coefficient = 0; coefficient < coefficient_count; ++coefficient) {
                uint dst = lane * SUMCHECK_MAX_COEFFS + coefficient;
                uint src = (lane + stride) * SUMCHECK_MAX_COEFFS + coefficient;
                shared[dst] = kx_add(shared[dst], shared[src]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        for (uint coefficient = 0; coefficient < coefficient_count; ++coefficient) {
            Kx value = shared[coefficient];
            ulong output_index = group * coefficient_count + coefficient;
            partials[2 * output_index] = value.c0;
            partials[2 * output_index + 1] = value.c1;
        }
    }
}

kernel void nc_round_partials(
    device const ulong *eq_table [[buffer(0)]],
    device const ulong *shape [[buffer(1)]],
    device const ulong *digit_values [[buffer(2)]],
    device const ulong *weights [[buffer(3)]],
    device ulong *partials [[buffer(4)]],
    uint pair [[thread_position_in_grid]],
    uint lane_index [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    ulong table_len = shape[0];
    ulong witness_count = shape[1];
    ulong width = shape[2];
    bool dense = shape[3] != 0;
    ulong values_per_witness = shape[4];
    threadgroup Kx shared[SUMCHECK_REDUCTION_THREADS * 5];
    Kx local[5] = {Kx{0, 0}, Kx{0, 0}, Kx{0, 0}, Kx{0, 0}, Kx{0, 0}};
    if (pair < table_len / 2) {
        ulong index = 2 * pair;
        Kx e0 = load_k(eq_table, index);
        Kx e1 = kx_sub(load_k(eq_table, index + 1), e0);
        Kx inner[4] = {Kx{0, 0}, Kx{0, 0}, Kx{0, 0}, Kx{0, 0}};
        for (ulong witness = 0; witness < witness_count; ++witness) {
            for (ulong ring_lane = 0; ring_lane < RING_DEGREE; ++ring_lane) {
                Kx weight = load_k(weights, witness * RING_DEGREE + ring_lane);
                ulong witness_base = witness * values_per_witness;
                Kx a = Kx{0, 0};
                Kx hi = Kx{0, 0};
                if (dense) {
                    a = load_k(digit_values, witness_base + index * RING_DEGREE + ring_lane);
                    hi = load_k(digit_values, witness_base + (index + 1) * RING_DEGREE + ring_lane);
                } else {
                    ulong start_lo = (index * width) % RING_DEGREE;
                    ulong slot_lo = (ring_lane + RING_DEGREE - start_lo) % RING_DEGREE;
                    if (slot_lo < width) {
                        a = load_k(digit_values, witness_base + index * width + slot_lo);
                    }
                    ulong start_hi = ((index + 1) * width) % RING_DEGREE;
                    ulong slot_hi = (ring_lane + RING_DEGREE - start_hi) % RING_DEGREE;
                    if (slot_hi < width) {
                        hi = load_k(digit_values, witness_base + (index + 1) * width + slot_hi);
                    }
                }
                Kx b = kx_sub(hi, a);
                Kx a2 = kx_mul(a, a);
                Kx b2 = kx_mul(b, b);
                inner[0] = kx_add(inner[0], kx_mul(weight, kx_sub(kx_mul(a2, a), a)));
                inner[1] = kx_add(inner[1], kx_mul(weight, kx_sub(kx_mul(kx_mul(a2, b), Kx{3, 0}), b)));
                inner[2] = kx_add(inner[2], kx_mul(weight, kx_mul(kx_mul(a, b2), Kx{3, 0})));
                inner[3] = kx_add(inner[3], kx_mul(weight, kx_mul(b2, b)));
            }
        }
        local[0] = kx_mul(e0, inner[0]);
        local[1] = kx_add(kx_mul(e0, inner[1]), kx_mul(e1, inner[0]));
        local[2] = kx_add(kx_mul(e0, inner[2]), kx_mul(e1, inner[1]));
        local[3] = kx_add(kx_mul(e0, inner[3]), kx_mul(e1, inner[2]));
        local[4] = kx_mul(e1, inner[3]);
    }
    for (uint coefficient = 0; coefficient < 5; ++coefficient) {
        shared[lane_index * 5 + coefficient] = local[coefficient];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = SUMCHECK_REDUCTION_THREADS / 2; stride > 0; stride >>= 1) {
        if (lane_index < stride) {
            for (uint coefficient = 0; coefficient < 5; ++coefficient) {
                uint dst = lane_index * 5 + coefficient;
                uint src = (lane_index + stride) * 5 + coefficient;
                shared[dst] = kx_add(shared[dst], shared[src]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane_index == 0) {
        for (uint coefficient = 0; coefficient < 5; ++coefficient) {
            Kx value = shared[coefficient];
            ulong output_index = group * 5 + coefficient;
            partials[2 * output_index] = value.c0;
            partials[2 * output_index + 1] = value.c1;
        }
    }
}
