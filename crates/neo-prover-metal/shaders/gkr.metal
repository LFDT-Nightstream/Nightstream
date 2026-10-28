// The rounds of a batched fraction-sum GKR layer over the cubic extension
// Gl[X] / (X^3 - X - 1). Values are three words per element. Tables are
// reached through a table of GPU addresses; a folded table keeps its entries
// in place with a stride that doubles every round.

struct GkrExt {
    ulong c0;
    ulong c1;
    ulong c2;
};

constant uint GKR_PARTS = 16;
constant uint GKR_NODES = 4;
constant uint GKR_THREADS = 256;

struct GkrTables {
    device ulong *part[GKR_PARTS];
};

inline GkrExt gkr_add(GkrExt a, GkrExt b) {
    return GkrExt{gl_add(a.c0, b.c0), gl_add(a.c1, b.c1), gl_add(a.c2, b.c2)};
}

inline GkrExt gkr_sub(GkrExt a, GkrExt b) {
    return GkrExt{gl_sub(a.c0, b.c0), gl_sub(a.c1, b.c1), gl_sub(a.c2, b.c2)};
}

// X^3 = X + 1.
inline GkrExt gkr_mul(GkrExt a, GkrExt b) {
    ulong b02 = gl_add(b.c0, b.c2);
    ulong b12 = gl_add(b.c1, b.c2);
    return GkrExt{
        gl_add(gl_add(gl_mul(a.c0, b.c0), gl_mul(a.c1, b.c2)), gl_mul(a.c2, b.c1)),
        gl_add(gl_add(gl_mul(a.c0, b.c1), gl_mul(a.c1, b02)), gl_mul(a.c2, b12)),
        gl_add(gl_add(gl_mul(a.c0, b.c2), gl_mul(a.c1, b.c1)), gl_mul(a.c2, b02))};
}

inline GkrExt gkr_load(device const ulong *table, ulong index) {
    return GkrExt{gl_from_word(table[3 * index]), gl_from_word(table[3 * index + 1]), gl_from_word(table[3 * index + 2])};
}

inline void gkr_store(device ulong *table, ulong index, GkrExt value) {
    table[3 * index] = value.c0;
    table[3 * index + 1] = value.c1;
    table[3 * index + 2] = value.c2;
}

inline GkrExt gkr_param(constant ulong *words, uint at) {
    return GkrExt{words[at], words[at + 1], words[at + 2]};
}

// Bind the lowest variable to r: t[2k·s] = t[2k·s] + r·(t[(2k+1)·s] - t[2k·s])
// for k < count. shape: [count, stride, r].
kernel void gkr_fold(
    device ulong *table [[buffer(0)]],
    constant ulong *shape [[buffer(1)]],
    uint k [[thread_position_in_grid]]) {
    ulong count = shape[0];
    if (k >= count) {
        return;
    }
    ulong stride = shape[1];
    ulong low = 2 * (ulong)k * stride;
    GkrExt a = gkr_load(table, low);
    GkrExt b = gkr_load(table, low + stride);
    gkr_store(table, low, gkr_add(a, gkr_mul(gkr_param(shape, 2), gkr_sub(b, a))));
}

// Move each line in value from node j - 1 to node j of 0, 2, 3, ...
inline void gkr_step(thread GkrExt *value, thread const GkrExt *slope, uint count, uint j) {
    if (j == 0) {
        return;
    }
    for (uint i = 0; i < count; ++i) {
        value[i] = gkr_add(value[i], slope[i]);
        if (j == 1) {
            value[i] = gkr_add(value[i], slope[i]);
        }
    }
}

// The round message, as partial sums per threadgroup: for pair m < half,
// Σ_j over nodes 0, 2, ..., degree of eq_m(x_j)·Σ_t w_t·(p_t0 q_t1 + p_t1
// q_t0 + μ q_t0 q_t1)(x_j). Every table has 4·half logical entries at the
// stride; tree t reads children b at b·2·half + 2m and 2m + 1. shape:
// [half, stride, trees, degree, μ (3)], then per tree [factors, first part,
// weight (3)]. The eq table has 2·half logical entries.
kernel void gkr_message(
    constant GkrTables &tables [[buffer(0)]],
    device const ulong *eq [[buffer(1)]],
    constant ulong *shape [[buffer(2)]],
    device ulong *partials [[buffer(3)]],
    uint m [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    threadgroup ulong shared[GKR_THREADS * GKR_NODES * 3];
    ulong half_size = shape[0];
    ulong stride = shape[1];
    uint trees = (uint)shape[2];
    uint degree = (uint)shape[3];
    GkrExt mu = gkr_param(shape, 4);
    GkrExt sums[GKR_NODES];
    for (uint j = 0; j < GKR_NODES; ++j) {
        sums[j] = GkrExt{0, 0, 0};
    }
    if (m < half_size) {
        GkrExt totals[GKR_NODES];
        for (uint j = 0; j < GKR_NODES; ++j) {
            totals[j] = GkrExt{0, 0, 0};
        }
        for (uint t = 0; t < trees; ++t) {
            uint factors = (uint)shape[7 + 5 * t];
            uint first = (uint)shape[8 + 5 * t];
            GkrExt weight = gkr_param(shape, 9 + 5 * t);
            // value[2·part + b], slope[2·part + b] for parts ≤ 3.
            GkrExt value[6];
            GkrExt slope[6];
            for (uint part = 0; part <= factors; ++part) {
                device const ulong *table = tables.part[first + part];
                for (uint b = 0; b < 2; ++b) {
                    ulong at = (b * 2 * half_size + 2 * (ulong)m) * stride;
                    GkrExt low = gkr_load(table, at);
                    value[2 * part + b] = low;
                    slope[2 * part + b] = gkr_sub(gkr_load(table, at + stride), low);
                }
            }
            for (uint j = 0; j < degree; ++j) {
                gkr_step(value, slope, 2 * (factors + 1), j);
                GkrExt q0 = value[2 * factors];
                GkrExt q1 = value[2 * factors + 1];
                GkrExt cross;
                if (factors == 0) {
                    cross = gkr_add(q0, q1);
                } else if (factors == 1) {
                    cross = gkr_add(gkr_mul(value[0], q1), gkr_mul(value[1], q0));
                } else {
                    cross = gkr_add(
                        gkr_mul(gkr_mul(value[0], value[2]), q1),
                        gkr_mul(gkr_mul(value[1], value[3]), q0));
                }
                GkrExt term = gkr_add(cross, gkr_mul(gkr_mul(mu, q0), q1));
                totals[j] = gkr_add(totals[j], gkr_mul(weight, term));
            }
        }
        GkrExt e[1] = {gkr_load(eq, 2 * (ulong)m * stride)};
        GkrExt de[1] = {gkr_sub(gkr_load(eq, (2 * (ulong)m + 1) * stride), e[0])};
        for (uint j = 0; j < degree; ++j) {
            gkr_step(e, de, 1, j);
            sums[j] = gkr_mul(e[0], totals[j]);
        }
    }
    for (uint j = 0; j < GKR_NODES; ++j) {
        uint at = (tid * GKR_NODES + j) * 3;
        shared[at] = sums[j].c0;
        shared[at + 1] = sums[j].c1;
        shared[at + 2] = sums[j].c2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = GKR_THREADS / 2; width > 0; width /= 2) {
        if (tid < width) {
            for (uint w = 0; w < GKR_NODES * 3; ++w) {
                uint at = tid * GKR_NODES * 3 + w;
                shared[at] = gl_add(shared[at], shared[at + width * GKR_NODES * 3]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        for (uint w = 0; w < GKR_NODES * 3; ++w) {
            partials[group * GKR_NODES * 3 + w] = shared[w];
        }
    }
}
