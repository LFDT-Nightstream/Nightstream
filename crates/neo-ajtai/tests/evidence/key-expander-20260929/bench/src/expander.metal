// One kernel per expander. A thread expands every key element (row, column)
// with column = tid + k * threads, and folds the 54 reduced coefficients
// into the checksum that the CPU code also computes.

#include <metal_stdlib>
using namespace metal;

constant ulong P = 0xFFFFFFFF00000001ul;
constant uint RING_DEGREE = 54;

inline ulong gl_add(ulong x, ulong y) {
    // x, y < P. A carry adds 2^64 = 2^32 - 1 (mod P).
    ulong s = x + y;
    if (s < x) {
        s += 0xFFFFFFFFul;
    }
    if (s >= P) {
        s -= P;
    }
    return s;
}

// Eight little-endian 32-bit words, reduced modulo P. With x = 2^32,
// x^2 = x - 1 and x^6 = 1, so the value is a + b * x.
inline ulong reduce_words(uint w0, uint w1, uint w2, uint w3, uint w4, uint w5, uint w6, uint w7) {
    long a = (long)w0 - w2 - w3 + w5 + w6;
    long b = (long)w1 + w2 - w4 - w5 + w7;
    ulong af = a < 0 ? P - (ulong)(-a) : (ulong)a;
    ulong bf = b < 0 ? P - (ulong)(-b) : (ulong)b;
    // bf * 2^32 = high * 2^64 + low * 2^32 = high * (2^32 - 1) + low * 2^32.
    ulong high = bf >> 32;
    ulong low = bf & 0xFFFFFFFFul;
    return gl_add(af, gl_add(low << 32, (high << 32) - high));
}

inline ulong fold(ulong sum, ulong coefficient, uint lane) {
    return sum + coefficient * (2ul * lane + 1ul);
}

inline ulong element_start(ulong index) {
    return index * 0x9E3779B97F4A7C15ul;
}

// ---- ChaCha20 (RFC 8439 block function) ----

inline uint rotl32(uint value, uint amount) {
    return (value << amount) | (value >> (32 - amount));
}

inline void quarter_round(thread uint *s, uint a, uint b, uint c, uint d) {
    s[a] += s[b]; s[d] = rotl32(s[d] ^ s[a], 16);
    s[c] += s[d]; s[b] = rotl32(s[b] ^ s[c], 12);
    s[a] += s[b]; s[d] = rotl32(s[d] ^ s[a], 8);
    s[c] += s[d]; s[b] = rotl32(s[b] ^ s[c], 7);
}

// Counter = `counter`, nonce = row || column (little-endian words).
inline void chacha_block(device const uint *seed, uint counter, uint row, ulong column, thread uint *out) {
    uint initial[16] = {
        0x61707865u, 0x3320646eu, 0x79622d32u, 0x6b206574u,
        seed[0], seed[1], seed[2], seed[3], seed[4], seed[5], seed[6], seed[7],
        counter, row, (uint)column, (uint)(column >> 32),
    };
    for (uint i = 0; i < 16; ++i) out[i] = initial[i];
    for (uint round = 0; round < 10; ++round) {
        quarter_round(out, 0, 4, 8, 12);
        quarter_round(out, 1, 5, 9, 13);
        quarter_round(out, 2, 6, 10, 14);
        quarter_round(out, 3, 7, 11, 15);
        quarter_round(out, 0, 5, 10, 15);
        quarter_round(out, 1, 6, 11, 12);
        quarter_round(out, 2, 7, 8, 13);
        quarter_round(out, 3, 4, 9, 14);
    }
    for (uint i = 0; i < 16; ++i) out[i] += initial[i];
}

// Current setup: lane L uses the first 32 bytes of block L.
kernel void expand_chacha20(
    device const uint *seed [[buffer(0)]],
    device const ulong *prefix [[buffer(1)]],
    device const ulong *shape [[buffer(2)]],
    device ulong *out [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
    uint row = (uint)shape[0];
    ulong columns = shape[1];
    ulong threads = shape[2];
    ulong acc = 0;
    for (ulong column = tid; column < columns; column += threads) {
        ulong sum = element_start((ulong)row * columns + column);
        for (uint lane = 0; lane < RING_DEGREE; ++lane) {
            uint w[16];
            chacha_block(seed, lane, row, column, w);
            sum = fold(sum, reduce_words(w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]), lane);
        }
        acc ^= sum;
    }
    out[tid] ^= acc;
}

// Contiguous keystream: lane L uses bytes 32L..32L+31, so block k gives
// lanes 2k and 2k+1.
kernel void expand_chacha20_contiguous(
    device const uint *seed [[buffer(0)]],
    device const ulong *prefix [[buffer(1)]],
    device const ulong *shape [[buffer(2)]],
    device ulong *out [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
    uint row = (uint)shape[0];
    ulong columns = shape[1];
    ulong threads = shape[2];
    ulong acc = 0;
    for (ulong column = tid; column < columns; column += threads) {
        ulong sum = element_start((ulong)row * columns + column);
        for (uint block = 0; block < RING_DEGREE / 2; ++block) {
            uint w[16];
            chacha_block(seed, block, row, column, w);
            sum = fold(sum, reduce_words(w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]), 2 * block);
            sum = fold(sum, reduce_words(w[8], w[9], w[10], w[11], w[12], w[13], w[14], w[15]), 2 * block + 1);
        }
        acc ^= sum;
    }
    out[tid] ^= acc;
}

// ---- SHAKE (Keccak-f[1600]) ----

constant ulong KECCAK_RC[24] = {
    0x0000000000000001ul, 0x0000000000008082ul, 0x800000000000808aul, 0x8000000080008000ul,
    0x000000000000808bul, 0x0000000080000001ul, 0x8000000080008081ul, 0x8000000000008009ul,
    0x000000000000008aul, 0x0000000000000088ul, 0x0000000080008009ul, 0x000000008000000aul,
    0x000000008000808bul, 0x800000000000008bul, 0x8000000000008089ul, 0x8000000000008003ul,
    0x8000000000008002ul, 0x8000000000000080ul, 0x000000000000800aul, 0x800000008000000aul,
    0x8000000080008081ul, 0x8000000000008080ul, 0x0000000080000001ul, 0x8000000080008008ul,
};
constant uint KECCAK_ROTC[24] = {1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44};
constant uint KECCAK_PILN[24] = {10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1};

inline ulong rotl64(ulong value, uint amount) {
    return (value << amount) | (value >> (64 - amount));
}

inline void keccak_f(thread ulong *st) {
    for (uint round = 0; round < 24; ++round) {
        ulong bc[5];
        for (uint i = 0; i < 5; ++i) bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
        for (uint i = 0; i < 5; ++i) {
            ulong t = bc[(i + 4) % 5] ^ rotl64(bc[(i + 1) % 5], 1);
            for (uint j = 0; j < 25; j += 5) st[j + i] ^= t;
        }
        ulong t = st[1];
        for (uint i = 0; i < 24; ++i) {
            uint j = KECCAK_PILN[i];
            ulong b = st[j];
            st[j] = rotl64(t, KECCAK_ROTC[i]);
            t = b;
        }
        for (uint j = 0; j < 25; j += 5) {
            for (uint i = 0; i < 5; ++i) bc[i] = st[j + i];
            for (uint i = 0; i < 5; ++i) st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
        }
        st[0] ^= KECCAK_RC[round];
    }
}

// Input: 32-byte tag || 32-byte seed || row (u32) || column (u64), then
// SHAKE padding 0x1F ... 0x80. The first eight lanes come from `prefix`.
template <uint RATE_LANES>
inline ulong shake_element(device const ulong *prefix, uint row, ulong column, ulong sum) {
    ulong st[25];
    for (uint i = 0; i < 25; ++i) st[i] = 0;
    for (uint i = 0; i < 8; ++i) st[i] = prefix[i];
    st[8] = (ulong)row | ((column & 0xFFFFFFFFul) << 32);
    st[9] = (column >> 32) | (0x1Ful << 32);
    st[RATE_LANES - 1] ^= 0x80ul << 56;
    keccak_f(st);
    uint position = 0;
    for (uint lane = 0; lane < RING_DEGREE; ++lane) {
        ulong quad[4];
        for (uint k = 0; k < 4; ++k) {
            if (position == RATE_LANES) {
                keccak_f(st);
                position = 0;
            }
            quad[k] = st[position++];
        }
        ulong c = reduce_words(
            (uint)quad[0], (uint)(quad[0] >> 32), (uint)quad[1], (uint)(quad[1] >> 32),
            (uint)quad[2], (uint)(quad[2] >> 32), (uint)quad[3], (uint)(quad[3] >> 32));
        sum = fold(sum, c, lane);
    }
    return sum;
}

template <uint RATE_LANES>
inline void expand_shake(device const ulong *prefix, device const ulong *shape, device ulong *out, uint tid) {
    uint row = (uint)shape[0];
    ulong columns = shape[1];
    ulong threads = shape[2];
    ulong acc = 0;
    for (ulong column = tid; column < columns; column += threads) {
        acc ^= shake_element<RATE_LANES>(prefix, row, column, element_start((ulong)row * columns + column));
    }
    out[tid] ^= acc;
}

// SHAKE128: rate 168 bytes = 21 lanes.
kernel void expand_shake128(
    device const uint *seed [[buffer(0)]],
    device const ulong *prefix [[buffer(1)]],
    device const ulong *shape [[buffer(2)]],
    device ulong *out [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
    expand_shake<21>(prefix, shape, out, tid);
}

// SHAKE256: rate 136 bytes = 17 lanes.
kernel void expand_shake256(
    device const uint *seed [[buffer(0)]],
    device const ulong *prefix [[buffer(1)]],
    device const ulong *shape [[buffer(2)]],
    device ulong *out [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
    expand_shake<17>(prefix, shape, out, tid);
}
