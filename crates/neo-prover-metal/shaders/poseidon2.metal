#include "include/goldilocks.metal"

constant uint P2_WIDTH = 8;
constant uint P2_RATE = 4;
constant uint P2_DIGEST_LEN = 4;
constant uint P2_EXTERNAL_HALF_ROUNDS = 4;
constant uint P2_INTERNAL_ROUNDS = 22;
constant uint P2_RC_INITIAL = 0;
constant uint P2_RC_INTERNAL = 32;
constant uint P2_RC_TERMINAL = 54;
constant uint P2_RC_DIAG = 86;

struct P2State {
    ulong s0;
    ulong s1;
    ulong s2;
    ulong s3;
    ulong s4;
    ulong s5;
    ulong s6;
    ulong s7;
};

struct P2Half {
    ulong s0;
    ulong s1;
    ulong s2;
    ulong s3;
};

inline ulong p2_sbox(ulong value) {
    ulong square = gl_mul(value, value);
    ulong cube = gl_mul(square, value);
    return gl_mul(gl_mul(cube, cube), value);
}

inline P2Half p2_mat4(ulong x0, ulong x1, ulong x2, ulong x3) {
    ulong t01 = gl_add(x0, x1);
    ulong t23 = gl_add(x2, x3);
    ulong t0123 = gl_add(t01, t23);
    ulong t01123 = gl_add(t0123, x1);
    ulong t01233 = gl_add(t0123, x3);
    return P2Half {
        gl_add(t01123, t01),
        gl_add(gl_add(t01123, x2), x2),
        gl_add(t01233, t23),
        gl_add(gl_add(t01233, x0), x0)
    };
}

inline P2State p2_mds_light(P2State state) {
    P2Half a = p2_mat4(state.s0, state.s1, state.s2, state.s3);
    P2Half b = p2_mat4(state.s4, state.s5, state.s6, state.s7);
    ulong m0 = gl_add(a.s0, b.s0);
    ulong m1 = gl_add(a.s1, b.s1);
    ulong m2 = gl_add(a.s2, b.s2);
    ulong m3 = gl_add(a.s3, b.s3);
    return P2State {
        gl_add(a.s0, m0),
        gl_add(a.s1, m1),
        gl_add(a.s2, m2),
        gl_add(a.s3, m3),
        gl_add(b.s0, m0),
        gl_add(b.s1, m1),
        gl_add(b.s2, m2),
        gl_add(b.s3, m3)
    };
}

inline P2State p2_external_round(P2State state, device const ulong *constants, uint base) {
    state.s0 = p2_sbox(gl_add(state.s0, constants[base]));
    state.s1 = p2_sbox(gl_add(state.s1, constants[base + 1]));
    state.s2 = p2_sbox(gl_add(state.s2, constants[base + 2]));
    state.s3 = p2_sbox(gl_add(state.s3, constants[base + 3]));
    state.s4 = p2_sbox(gl_add(state.s4, constants[base + 4]));
    state.s5 = p2_sbox(gl_add(state.s5, constants[base + 5]));
    state.s6 = p2_sbox(gl_add(state.s6, constants[base + 6]));
    state.s7 = p2_sbox(gl_add(state.s7, constants[base + 7]));
    return p2_mds_light(state);
}

inline P2State p2_permute(P2State state, device const ulong *constants) {
    state = p2_mds_light(state);
    for (uint round = 0; round < P2_EXTERNAL_HALF_ROUNDS; ++round) {
        state = p2_external_round(state, constants, P2_RC_INITIAL + P2_WIDTH * round);
    }

    ulong d0 = constants[P2_RC_DIAG];
    ulong d1 = constants[P2_RC_DIAG + 1];
    ulong d2 = constants[P2_RC_DIAG + 2];
    ulong d3 = constants[P2_RC_DIAG + 3];
    ulong d4 = constants[P2_RC_DIAG + 4];
    ulong d5 = constants[P2_RC_DIAG + 5];
    ulong d6 = constants[P2_RC_DIAG + 6];
    ulong d7 = constants[P2_RC_DIAG + 7];
    for (uint round = 0; round < P2_INTERNAL_ROUNDS; ++round) {
        state.s0 = p2_sbox(gl_add(state.s0, constants[P2_RC_INTERNAL + round]));
        ulong sum = gl_add(gl_add(gl_add(state.s0, state.s1), gl_add(state.s2, state.s3)),
                           gl_add(gl_add(state.s4, state.s5), gl_add(state.s6, state.s7)));
        state = P2State {
            gl_add(gl_mul(state.s0, d0), sum),
            gl_add(gl_mul(state.s1, d1), sum),
            gl_add(gl_mul(state.s2, d2), sum),
            gl_add(gl_mul(state.s3, d3), sum),
            gl_add(gl_mul(state.s4, d4), sum),
            gl_add(gl_mul(state.s5, d5), sum),
            gl_add(gl_mul(state.s6, d6), sum),
            gl_add(gl_mul(state.s7, d7), sum)
        };
    }
    for (uint round = 0; round < P2_EXTERNAL_HALF_ROUNDS; ++round) {
        state = p2_external_round(state, constants, P2_RC_TERMINAL + P2_WIDTH * round);
    }
    return state;
}

inline P2State p2_load(device const ulong *words, ulong base) {
    return P2State {
        gl_from_word(words[base]),
        gl_from_word(words[base + 1]),
        gl_from_word(words[base + 2]),
        gl_from_word(words[base + 3]),
        gl_from_word(words[base + 4]),
        gl_from_word(words[base + 5]),
        gl_from_word(words[base + 6]),
        gl_from_word(words[base + 7])
    };
}

inline void p2_store(device ulong *words, ulong base, P2State state) {
    words[base] = state.s0;
    words[base + 1] = state.s1;
    words[base + 2] = state.s2;
    words[base + 3] = state.s3;
    words[base + 4] = state.s4;
    words[base + 5] = state.s5;
    words[base + 6] = state.s6;
    words[base + 7] = state.s7;
}

inline void p2_set_rate_lane(thread P2State &state, uint lane, ulong word) {
    ulong value = gl_from_word(word);
    switch (lane) {
        case 0: state.s0 = value; break;
        case 1: state.s1 = value; break;
        case 2: state.s2 = value; break;
        default: state.s3 = value; break;
    }
}

inline void p2_absorb_word(
    thread P2State &state,
    thread uint &cursor,
    ulong word,
    device const ulong *constants) {
    if (cursor >= P2_RATE) {
        state = p2_permute(state, constants);
        cursor = 0;
    }
    p2_set_rate_lane(state, cursor, word);
    ++cursor;
}

inline void p2_write_digest(device ulong *out, ulong offset, uint count, P2State state) {
    out[offset] = state.s0;
    if (count > 1) out[offset + 1] = state.s1;
    if (count > 2) out[offset + 2] = state.s2;
    if (count > 3) out[offset + 3] = state.s3;
}

kernel void poseidon2_permute_states(
    device ulong *states [[buffer(0)]],
    device const ulong *constants [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    ulong base = ulong(index) * P2_WIDTH;
    p2_store(states, base, p2_permute(p2_load(states, base), constants));
}

kernel void poseidon2_hash_fields(
    device const ulong *fields [[buffer(0)]],
    device const ulong *offsets [[buffer(1)]],
    device const ulong *lengths [[buffer(2)]],
    device ulong *out [[buffer(3)]],
    device const ulong *constants [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    ulong offset = offsets[index];
    ulong length = lengths[index];
    P2State state = P2State { 0, 0, 0, 0, 0, 0, 0, 0 };
    ulong position = 0;
    while (position < length) {
        ulong remaining = length - position;
        ulong take = remaining < P2_RATE ? remaining : P2_RATE;
        if (take > 0) state.s0 = gl_add(state.s0, gl_from_word(fields[offset + position]));
        if (take > 1) state.s1 = gl_add(state.s1, gl_from_word(fields[offset + position + 1]));
        if (take > 2) state.s2 = gl_add(state.s2, gl_from_word(fields[offset + position + 2]));
        if (take > 3) state.s3 = gl_add(state.s3, gl_from_word(fields[offset + position + 3]));
        state = p2_permute(state, constants);
        position += take;
    }
    state.s0 = gl_add(state.s0, 1);
    state = p2_permute(state, constants);
    ulong base = ulong(index) * P2_DIGEST_LEN;
    out[base] = state.s0;
    out[base + 1] = state.s1;
    out[base + 2] = state.s2;
    out[base + 3] = state.s3;
}

kernel void poseidon2_transcript_ops(
    device ulong *state_words [[buffer(0)]],
    device const ulong *ops [[buffer(1)]],
    device const ulong *payload [[buffer(2)]],
    device ulong *out [[buffer(3)]],
    device const ulong *meta [[buffer(4)]],
    device const ulong *constants [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    if (index != 0) return;

    P2State state = p2_load(state_words, 0);
    uint cursor = uint(state_words[P2_WIDTH]);
    ulong payload_position = 0;
    ulong output_position = 0;
    ulong op_count = meta[0];
    for (ulong op = 0; op < op_count; ++op) {
        ulong code = ops[2 * op];
        ulong argument = ops[2 * op + 1];
        if (code == 0) {
            for (ulong i = 0; i < argument; ++i) {
                p2_absorb_word(state, cursor, payload[payload_position], constants);
                ++payload_position;
            }
        } else {
            ulong produced = 0;
            while (produced < argument) {
                if (cursor >= P2_RATE) {
                    state = p2_permute(state, constants);
                    cursor = 0;
                }
                p2_set_rate_lane(state, cursor, 1);
                state = p2_permute(state, constants);
                cursor = 0;
                ulong remaining = argument - produced;
                uint take = uint(remaining < P2_DIGEST_LEN ? remaining : P2_DIGEST_LEN);
                p2_write_digest(out, output_position, take, state);
                output_position += take;
                produced += take;
            }
        }
    }
    p2_store(state_words, 0, state);
    state_words[P2_WIDTH] = cursor;
}
