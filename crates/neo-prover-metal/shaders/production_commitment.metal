// nightstream-ajtai-chacha20-wide256-v1, with the verifier-owned seed supplied
// by Rust. Key tiles exist only in threadgroup memory and are shared by all
// witnesses. The output is a partial Phi81 ring product for each witness.

inline ulong production_ajtai_coefficient(
    device const uint *seed, uint row, uint column, uint lane) {
    uint initial[16] = {
        0x61707865u, 0x3320646eu, 0x79622d32u, 0x6b206574u,
        seed[0], seed[1], seed[2], seed[3], seed[4], seed[5], seed[6], seed[7],
        lane, row, column, 0u,
    };
    uint words[16];
    for (uint i = 0; i < 16; ++i) words[i] = initial[i];
    for (uint round = 0; round < 10; ++round) {
        seeded_ajtai_quarter_round(words, 0, 4, 8, 12);
        seeded_ajtai_quarter_round(words, 1, 5, 9, 13);
        seeded_ajtai_quarter_round(words, 2, 6, 10, 14);
        seeded_ajtai_quarter_round(words, 3, 7, 11, 15);
        seeded_ajtai_quarter_round(words, 0, 5, 10, 15);
        seeded_ajtai_quarter_round(words, 1, 6, 11, 12);
        seeded_ajtai_quarter_round(words, 2, 7, 8, 13);
        seeded_ajtai_quarter_round(words, 3, 4, 9, 14);
    }
    for (uint i = 0; i < 8; ++i) words[i] += initial[i];
    // For x = 2^32, x^2 = x - 1 and x^6 = 1 modulo Goldilocks.
    // Thus the first eight little-endian words reduce to a + b*x.
    long a = (long)words[0] - words[2] - words[3] + words[5] + words[6];
    long b = (long)words[1] + words[2] - words[4] - words[5] + words[7];
    ulong a_field = a < 0 ? GOLDILOCKS_MODULUS - (ulong)(-a) : (ulong)a;
    ulong b_field = b < 0 ? GOLDILOCKS_MODULUS - (ulong)(-b) : (ulong)b;
    return gl_add(a_field, gl_mul_native(b_field, 1ul << 32));
}

kernel void production_ajtai_partials(
    device const uint *seed [[buffer(0)]],
    device const uint *columns [[buffer(1)]],
    device const ulong2 *masks [[buffer(2)]],
    device const ulong *shape [[buffer(3)]],
    device ulong *partials [[buffer(4)]],
    threadgroup ulong *key [[threadgroup(0)]],
    uint chunk [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    ulong column_count = shape[0];
    ulong count = shape[1];
    ulong chunks = shape[2];
    ulong tile = shape[3];
    uint row = (uint)shape[4];
    ulong threads = shape[5];
    ulong start = (ulong)chunk * tile;
    ulong end = min(start + tile, column_count);
    ulong width = end - start;
    threadgroup ulong raw[RING_PRODUCT_COEFFICIENTS];
    for (ulong index = lane; index < width * RING_DEGREE; index += threads) {
        key[index] = production_ajtai_coefficient(
            seed, row, columns[start + index / RING_DEGREE], (uint)(index % RING_DEGREE));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (ulong witness = 0; witness < count; ++witness) {
        if (lane < RING_PRODUCT_COEFFICIENTS) {
            uint low = lane >= RING_DEGREE ? lane - (RING_DEGREE - 1) : 0;
            uint high = min(lane, (uint)RING_DEGREE - 1);
            ulong valid = (~0ul << low) & ((1ul << (high + 1)) - 1);
            ulong positive_lo = 0, positive_hi = 0, negative_lo = 0, negative_hi = 0;
            for (ulong column = 0; column < width; ++column) {
                ulong2 digits = masks[witness * column_count + start + column];
                ulong positive = digits.x & valid;
                ulong negative = digits.y & valid;
                while (positive != 0) {
                    uint shift = (uint)ctz(positive);
                    positive &= positive - 1;
                    ulong value = key[column * RING_DEGREE + lane - shift];
                    ulong next = positive_lo + value;
                    positive_hi += next < positive_lo;
                    positive_lo = next;
                }
                while (negative != 0) {
                    uint shift = (uint)ctz(negative);
                    negative &= negative - 1;
                    ulong value = key[column * RING_DEGREE + lane - shift];
                    ulong next = negative_lo + value;
                    negative_hi += next < negative_lo;
                    negative_lo = next;
                }
            }
            raw[lane] = gl_sub(gl_reduce_sum(positive_lo, positive_hi), gl_reduce_sum(negative_lo, negative_hi));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < RING_DEGREE) {
            // X^54 = -X^27 - 1 and X^81 = 1.
            ulong value;
            if (lane < RING_DEGREE / 2) {
                value = gl_sub(raw[lane], raw[lane + RING_DEGREE]);
                if (lane + 81 < RING_PRODUCT_COEFFICIENTS) value = gl_add(value, raw[lane + 81]);
            } else {
                value = gl_sub(raw[lane], raw[lane + RING_DEGREE / 2]);
            }
            partials[(witness * chunks + chunk) * RING_DEGREE + lane] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
