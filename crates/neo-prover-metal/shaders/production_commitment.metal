// nightstream-ajtai-shake128-wide256-v1, with the verifier-owned setup ID
// and seed supplied by Rust. Key rows are expanded into device memory, either
// one row at a time for one call or once for every call of a prover. Every
// witness accumulates its signed ring products from the row it is given.

constant ulong PRODUCTION_KECCAK_RC[24] = {
    0x0000000000000001ul, 0x0000000000008082ul, 0x800000000000808aul, 0x8000000080008000ul,
    0x000000000000808bul, 0x0000000080000001ul, 0x8000000080008081ul, 0x8000000000008009ul,
    0x000000000000008aul, 0x0000000000000088ul, 0x0000000080008009ul, 0x000000008000000aul,
    0x000000008000808bul, 0x800000000000008bul, 0x8000000000008089ul, 0x8000000000008003ul,
    0x8000000000008002ul, 0x8000000000000080ul, 0x000000000000800aul, 0x800000008000000aul,
    0x8000000080008081ul, 0x8000000000008080ul, 0x0000000080000001ul, 0x8000000080008008ul,
};
// SHAKE128 absorbs and squeezes 168 bytes, 21 lanes, per permutation.
constant uint PRODUCTION_SHAKE128_RATE_LANES = 21;

inline ulong production_rotl64(ulong value, uint amount) {
    return (value << amount) | (value >> (64 - amount));
}

// One Keccak-f[1600] round on lanes a[5y + x]: theta, rho and pi, chi and
// iota (FIPS 202, Section 3.3), in the order of the Lean specification. Every
// lane index is fixed, so the state can stay in registers.
inline void production_keccak_round(thread ulong *a, ulong round_constant) {
    ulong c0 = a[0] ^ a[5] ^ a[10] ^ a[15] ^ a[20];
    ulong c1 = a[1] ^ a[6] ^ a[11] ^ a[16] ^ a[21];
    ulong c2 = a[2] ^ a[7] ^ a[12] ^ a[17] ^ a[22];
    ulong c3 = a[3] ^ a[8] ^ a[13] ^ a[18] ^ a[23];
    ulong c4 = a[4] ^ a[9] ^ a[14] ^ a[19] ^ a[24];
    ulong d0 = c4 ^ production_rotl64(c1, 1);
    ulong d1 = c0 ^ production_rotl64(c2, 1);
    ulong d2 = c1 ^ production_rotl64(c3, 1);
    ulong d3 = c2 ^ production_rotl64(c4, 1);
    ulong d4 = c3 ^ production_rotl64(c0, 1);
    ulong b00 = a[0] ^ d0;
    ulong b10 = production_rotl64(a[6] ^ d1, 44);
    ulong b20 = production_rotl64(a[12] ^ d2, 43);
    ulong b30 = production_rotl64(a[18] ^ d3, 21);
    ulong b40 = production_rotl64(a[24] ^ d4, 14);
    ulong b01 = production_rotl64(a[3] ^ d3, 28);
    ulong b11 = production_rotl64(a[9] ^ d4, 20);
    ulong b21 = production_rotl64(a[10] ^ d0, 3);
    ulong b31 = production_rotl64(a[16] ^ d1, 45);
    ulong b41 = production_rotl64(a[22] ^ d2, 61);
    ulong b02 = production_rotl64(a[1] ^ d1, 1);
    ulong b12 = production_rotl64(a[7] ^ d2, 6);
    ulong b22 = production_rotl64(a[13] ^ d3, 25);
    ulong b32 = production_rotl64(a[19] ^ d4, 8);
    ulong b42 = production_rotl64(a[20] ^ d0, 18);
    ulong b03 = production_rotl64(a[4] ^ d4, 27);
    ulong b13 = production_rotl64(a[5] ^ d0, 36);
    ulong b23 = production_rotl64(a[11] ^ d1, 10);
    ulong b33 = production_rotl64(a[17] ^ d2, 15);
    ulong b43 = production_rotl64(a[23] ^ d3, 56);
    ulong b04 = production_rotl64(a[2] ^ d2, 62);
    ulong b14 = production_rotl64(a[8] ^ d3, 55);
    ulong b24 = production_rotl64(a[14] ^ d4, 39);
    ulong b34 = production_rotl64(a[15] ^ d0, 41);
    ulong b44 = production_rotl64(a[21] ^ d1, 2);
    a[0] = b00 ^ (~b10 & b20) ^ round_constant;
    a[1] = b10 ^ (~b20 & b30);
    a[2] = b20 ^ (~b30 & b40);
    a[3] = b30 ^ (~b40 & b00);
    a[4] = b40 ^ (~b00 & b10);
    a[5] = b01 ^ (~b11 & b21);
    a[6] = b11 ^ (~b21 & b31);
    a[7] = b21 ^ (~b31 & b41);
    a[8] = b31 ^ (~b41 & b01);
    a[9] = b41 ^ (~b01 & b11);
    a[10] = b02 ^ (~b12 & b22);
    a[11] = b12 ^ (~b22 & b32);
    a[12] = b22 ^ (~b32 & b42);
    a[13] = b32 ^ (~b42 & b02);
    a[14] = b42 ^ (~b02 & b12);
    a[15] = b03 ^ (~b13 & b23);
    a[16] = b13 ^ (~b23 & b33);
    a[17] = b23 ^ (~b33 & b43);
    a[18] = b33 ^ (~b43 & b03);
    a[19] = b43 ^ (~b03 & b13);
    a[20] = b04 ^ (~b14 & b24);
    a[21] = b14 ^ (~b24 & b34);
    a[22] = b24 ^ (~b34 & b44);
    a[23] = b34 ^ (~b44 & b04);
    a[24] = b44 ^ (~b04 & b14);
}

inline void production_keccak_f(thread ulong *st) {
    for (uint round = 0; round < 24; ++round) production_keccak_round(st, PRODUCTION_KECCAK_RC[round]);
}

// One 32-byte little-endian chunk, four output lanes, reduced modulo
// Goldilocks. For x = 2^32, x^2 = x - 1 and x^6 = 1, so the value is a + b*x.
inline ulong production_wide_coefficient(ulong q0, ulong q1, ulong q2, ulong q3) {
    uint w0 = (uint)q0, w1 = (uint)(q0 >> 32), w2 = (uint)q1, w3 = (uint)(q1 >> 32);
    uint w4 = (uint)q2, w5 = (uint)(q2 >> 32), w6 = (uint)q3, w7 = (uint)(q3 >> 32);
    long a = (long)w0 - w2 - w3 + w5 + w6;
    long b = (long)w1 + w2 - w4 - w5 + w7;
    ulong a_field = a < 0 ? GOLDILOCKS_MODULUS - (ulong)(-a) : (ulong)a;
    ulong b_field = b < 0 ? GOLDILOCKS_MODULUS - (ulong)(-b) : (ulong)b;
    return gl_add(a_field, gl_mul_native(b_field, 1ul << 32));
}

// The 54 coefficients of key element (row, column):
// SHAKE128(setup ID ‖ seed ‖ row_u32_le ‖ column_u64_le), 81 input bytes.
// `prefix` holds input bytes 0..68 as nine little-endian lanes; bytes 69..71
// of the last lane are zero. Coefficient L is written to out[L * stride].
inline void production_ajtai_element(
    device const ulong *prefix, uint row, ulong column, device ulong *out, ulong stride) {
    ulong st[25];
    for (uint i = 0; i < 25; ++i) st[i] = 0;
    for (uint i = 0; i < 9; ++i) st[i] = prefix[i];
    st[8] |= ((ulong)row & 0xFFFFFFul) << 40;
    st[9] = ((ulong)row >> 24) | (column << 8);
    // Byte 80 is the last column byte; byte 81 starts the SHAKE padding.
    st[10] = (column >> 56) | (0x1Ful << 8);
    st[PRODUCTION_SHAKE128_RATE_LANES - 1] ^= 0x80ul << 56;
    // Squeeze the rate lanes in order. The last four lanes shift through
    // q0..q3, so every state index stays fixed.
    ulong q0 = 0, q1 = 0, q2 = 0, q3 = 0;
    uint words = 0, lane = 0;
    for (;;) {
        production_keccak_f(st);
        #pragma unroll
        for (uint i = 0; i < PRODUCTION_SHAKE128_RATE_LANES; ++i) {
            q0 = q1;
            q1 = q2;
            q2 = q3;
            q3 = st[i];
            if (++words == 4) {
                out[lane * stride] = production_wide_coefficient(q0, q1, q2, q3);
                words = 0;
                if (++lane == RING_DEGREE) return;
            }
        }
    }
}

// One key row: one thread per listed column. slab[position * 54 + L] holds
// coefficient L of key element (row, columns[position]), so the lanes of one
// witness read one column's coefficients from consecutive words.
kernel void production_key_row(
    device const ulong *prefix [[buffer(0)]],
    device const uint *columns [[buffer(1)]],
    device const ulong *shape [[buffer(2)]],
    device ulong *slab [[buffer(3)]],
    uint position [[thread_position_in_grid]]) {
    ulong count = shape[0];
    production_ajtai_element(prefix, (uint)shape[4], columns[position], slab + position * RING_DEGREE, 1);
}

// Positive or negative sums of the 32-bit key halves for raw coefficients L
// and L + 54. Each term is below 2^32, so a sum cannot overflow 64 bits before
// 2^32 terms; a threadgroup adds far fewer.
struct ProductionHalfSums {
    ulong low_lo;
    ulong low_hi;
    ulong high_lo;
    ulong high_hi;
};

// Add the key coefficients that the set bits of `digits` select. Bit s times
// key coefficient j lands on raw coefficient s + j. Lane L owns raw
// coefficients L (s <= L) and L + 54 (s > L), so every lane adds once per bit
// and all lanes of a witness follow the same bits.
inline void production_add_bits(
    uint bits, uint offset, device const uint2 *key, uint lane, thread ProductionHalfSums &sums) {
    while (bits != 0) {
        uint shift = offset + ctz(bits);
        bits &= bits - 1;
        bool low = lane >= shift;
        uint2 value = key[low ? lane - shift : lane + RING_DEGREE - shift];
        sums.low_lo += low ? value.x : 0u;
        sums.low_hi += low ? value.y : 0u;
        sums.high_lo += low ? 0u : value.x;
        sums.high_hi += low ? 0u : value.y;
    }
}

inline void production_add_digits(
    ulong digits, device const ulong *key, uint lane, thread ProductionHalfSums &sums) {
    device const uint2 *halves = (device const uint2 *)key;
    production_add_bits((uint)digits, 0, halves, lane, sums);
    production_add_bits((uint)(digits >> 32), 32, halves, lane, sums);
}

// lo + hi * 2^32 modulo Goldilocks. The high word of the 128-bit value is
// below 2^32, as gl_reduce_sum requires.
inline ulong production_join_halves(ulong lo, ulong hi) {
    ulong low = lo + (hi << 32);
    ulong high = (hi >> 32) + (low < lo ? 1ul : 0ul);
    return gl_reduce_sum(low, high);
}

// One threadgroup sums the signed key products of a block of witnesses over a
// range of occupied columns; 64 lanes serve one witness. Occupied position p
// reads key element key_columns[p] of the row. Each group writes one
// Phi81-reduced partial per witness. The integer work is 32-bit where it can
// be: the GPU emulates 64-bit operations, and this loop is ALU-bound.
kernel void production_ajtai_accumulate(
    device const ulong *key [[buffer(0)]],
    device const ulong2 *masks [[buffer(1)]],
    device const ulong *shape [[buffer(2)]],
    device ulong *partials [[buffer(3)]],
    device const uint *key_columns [[buffer(4)]],
    threadgroup ulong *scratch [[threadgroup(0)]],
    uint group_index [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    ulong count = shape[0];
    ulong witnesses = shape[1];
    ulong groups = shape[2];
    ulong columns_per_group = shape[3];
    ulong witnesses_per_group = shape[5];
    ulong group = group_index % groups;
    uint lane = thread_index % 64;
    ulong witness = (group_index / groups) * witnesses_per_group + thread_index / 64;
    bool active = witness < witnesses && lane < RING_DEGREE;
    ProductionHalfSums positive = {0, 0, 0, 0};
    ProductionHalfSums negative = {0, 0, 0, 0};
    ulong first = group * columns_per_group;
    ulong end = min(first + columns_per_group, count);
    if (active) {
        for (ulong position = first; position < end; ++position) {
            ulong2 digits = masks[witness * count + position];
            device const ulong *element = key + (ulong)key_columns[position] * RING_DEGREE;
            production_add_digits(digits.x, element, lane, positive);
            production_add_digits(digits.y, element, lane, negative);
        }
    }
    // Raw-coefficient scratch, one row of 107 per witness.
    threadgroup ulong *raw = scratch + (thread_index / 64) * RING_PRODUCT_COEFFICIENTS;
    if (active) {
        raw[lane] = gl_sub(
            production_join_halves(positive.low_lo, positive.low_hi),
            production_join_halves(negative.low_lo, negative.low_hi));
        if (lane + RING_DEGREE < RING_PRODUCT_COEFFICIENTS) {
            raw[lane + RING_DEGREE] = gl_sub(
                production_join_halves(positive.high_lo, positive.high_hi),
                production_join_halves(negative.high_lo, negative.high_hi));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active) {
        // X^54 = -X^27 - 1 and X^81 = 1.
        ulong value;
        if (lane < RING_DEGREE / 2) {
            value = gl_sub(raw[lane], raw[lane + RING_DEGREE]);
            if (lane + 81 < RING_PRODUCT_COEFFICIENTS) value = gl_add(value, raw[lane + 81]);
        } else {
            value = gl_sub(raw[lane], raw[lane + RING_DEGREE / 2]);
        }
        partials[(witness * groups + group) * RING_DEGREE + lane] = value;
    }
}

// One thread per (witness, coefficient): the sum of the group partials.
kernel void production_ajtai_sum_groups(
    device const ulong *partials [[buffer(0)]],
    device const ulong *shape [[buffer(1)]],
    device ulong *output [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    ulong groups = shape[2];
    ulong witness = index / RING_DEGREE;
    ulong coefficient = index % RING_DEGREE;
    ulong value = 0;
    for (ulong group = 0; group < groups; ++group) {
        value = gl_add(value, partials[(witness * groups + group) * RING_DEGREE + coefficient]);
    }
    output[index] = value;
}
