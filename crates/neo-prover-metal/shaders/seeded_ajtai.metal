// Canonical ChaCha8 expansion for fixed-seed Ajtai matrices.

inline uint seeded_ajtai_rotl(uint value, uint amount) {
    return (value << amount) | (value >> (32 - amount));
}

inline void seeded_ajtai_quarter_round(
    thread uint *state,
    uint a,
    uint b,
    uint c,
    uint d) {
    state[a] += state[b];
    state[d] = seeded_ajtai_rotl(state[d] ^ state[a], 16);
    state[c] += state[d];
    state[b] = seeded_ajtai_rotl(state[b] ^ state[c], 12);
    state[a] += state[b];
    state[d] = seeded_ajtai_rotl(state[d] ^ state[a], 8);
    state[c] += state[d];
    state[b] = seeded_ajtai_rotl(state[b] ^ state[c], 7);
}

inline void seeded_ajtai_chacha8_block(
    device const uint *seed,
    ulong counter,
    thread uint *output) {
    uint initial[16] = {
        0x61707865u,
        0x3320646eu,
        0x79622d32u,
        0x6b206574u,
        seed[0],
        seed[1],
        seed[2],
        seed[3],
        seed[4],
        seed[5],
        seed[6],
        seed[7],
        (uint)counter,
        (uint)(counter >> 32),
        0u,
        0u,
    };
    for (uint word = 0; word < 16; ++word) {
        output[word] = initial[word];
    }
    for (uint round = 0; round < 4; ++round) {
        seeded_ajtai_quarter_round(output, 0, 4, 8, 12);
        seeded_ajtai_quarter_round(output, 1, 5, 9, 13);
        seeded_ajtai_quarter_round(output, 2, 6, 10, 14);
        seeded_ajtai_quarter_round(output, 3, 7, 11, 15);
        seeded_ajtai_quarter_round(output, 0, 5, 10, 15);
        seeded_ajtai_quarter_round(output, 1, 6, 11, 12);
        seeded_ajtai_quarter_round(output, 2, 7, 8, 13);
        seeded_ajtai_quarter_round(output, 3, 4, 9, 14);
    }
    for (uint word = 0; word < 16; ++word) {
        output[word] += initial[word];
    }
}

// One thread expands one 64-byte ChaCha block (eight field candidates).
// Full PP chunks contain 32,768 ring columns, so chunk boundaries are block
// aligned. The final short chunk is also the final block range of its row.
kernel void seeded_ajtai_matrix(
    device const uint *seeds [[buffer(0)]],
    device const ulong *shape [[buffer(1)]],
    device ulong *matrix [[buffer(2)]],
    device atomic_uint *rejected [[buffer(3)]],
    uint group [[thread_position_in_grid]]) {
    ulong rows = shape[0];
    ulong cols = shape[1];
    ulong chunk_size = shape[2];
    ulong chunks_per_row = shape[3];
    ulong groups_per_row = shape[4];
    ulong row = (ulong)group / groups_per_row;
    ulong group_in_row = (ulong)group % groups_per_row;
    if (row >= rows) {
        return;
    }

    ulong row_words = cols * RING_DEGREE;
    ulong first_word = group_in_row * 8;
    if (first_word >= row_words) {
        return;
    }
    ulong chunk_words = chunk_size * RING_DEGREE;
    ulong chunk = first_word / chunk_words;
    ulong local_word = first_word - chunk * chunk_words;
    ulong seed_base = (row * chunks_per_row + chunk) * 8;
    uint block[16];
    seeded_ajtai_chacha8_block(seeds + seed_base, local_word / 8, block);

    ulong output_base = row * row_words + first_word;
    for (uint slot = 0; slot < 8 && first_word + slot < row_words; ++slot) {
        ulong candidate = (ulong)block[2 * slot] | ((ulong)block[2 * slot + 1] << 32);
        matrix[output_base + slot] = candidate;
        if (candidate >= GOLDILOCKS_MODULUS) {
            atomic_fetch_or_explicit(rejected, 1u, memory_order_relaxed);
        }
    }
}
