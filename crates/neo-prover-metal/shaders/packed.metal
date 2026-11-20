namespace packed_oracle {
using E = ulong2;
inline ulong a64(ulong a, ulong b) {
    ulong s = a + b;
    s += s < a ? 59ul : 0ul;
    return min(s, s + 59ul);
}
inline ulong s64(ulong a, ulong b) {
    return (a - b) - (a < b ? 59ul : 0ul);
}
inline ulong m64(ulong a, ulong b) {
    ulong l = a * b, h = mulhi(a, b), t = l + 59ul * h, f = t + 59ul * (mulhi(h, 59ul) + (t < l));
    f += f < t ? 59ul : 0ul;
    return min(f, f + 59ul);
}
inline E add(E a, E b) {
    return E(a64(a.x, b.x), a64(a.y, b.y));
}
inline E sub(E a, E b) {
    return E(s64(a.x, b.x), s64(a.y, b.y));
}
inline E mul(E a, E b) {
    ulong ac = m64(a.x, b.x), bd = m64(a.y, b.y);
    return E(a64(ac, a64(bd, bd)), a64(m64(a.x, b.y), m64(a.y, b.x)));
}
inline E scaled(E a, ulong n) {
    return E(m64(a.x, n), m64(a.y, n));
}
inline E fold(E a, E b, E r) {
    return add(a, mul(sub(b, a), r));
}
struct Shape {
    uint count, norm, rows, r, low, carry, unused[2];
    E challenge;
};
template <uint N> struct Coeff {
    E v[N];
};
template <uint N> inline void reduce(Coeff<N> v, threadgroup Coeff<N> *shared, uint lane, uint group, device E *out) {
    shared[lane] = v;
    for (uint d = 64; d; d /= 2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < d)
            for (uint j = 0; j < N; j++)
                shared[lane].v[j] = add(shared[lane].v[j], shared[lane + d].v[j]);
    }
    if (!lane)
        for (uint j = 0; j < N; j++)
            out[group * N + j] = shared[0].v[j];
}
inline E wt(uint i, device const E *low, device const E *high, uint width) {
    return mul(low[i & (width - 1)], high[i / width]);
}
inline E gate_row(constant Shape &s, device const ulong4 *checked, device E *table, uint i, uint m) {
    uint n = (s.rows + ((1u << s.r) - 1)) >> s.r;
    if (i >= n)
        return 0;
    if (!s.r)
        return E(checked[i][m], 0);
    uint j = i << (s.r - 1);
    E a, b;
    if (s.r == 1) {
        a = E(checked[2 * i][m], 0);
        b = 2 * i + 1 < s.rows ? E(checked[2 * i + 1][m], 0) : E(0);
    } else {
        uint prev = (s.rows + ((1u << (s.r - 1)) - 1)) >> (s.r - 1);
        a = table[4 * j + m];
        b = 2 * i + 1 < prev ? table[4 * (j + (1u << (s.r - 2))) + m] : E(0);
    }
    E bound = fold(a, b, s.challenge);
    table[4 * j + m] = bound;
    return bound;
}
kernel void packed_gate(device const ulong4 *checked [[buffer(0)]], device E *table [[buffer(1)]],
                        device const E *lo [[buffer(2)]], device const E *hi [[buffer(3)]], device E *out [[buffer(4)]],
                        constant Shape &s [[buffer(5)]], uint group [[threadgroup_position_in_grid]],
                        uint lane [[thread_position_in_threadgroup]]) {
    uint i = 128 * group + lane;
    Coeff<8> t = {};
    if (i < s.count) {
        E x[4], y[4];
        for (uint j = 0; j < 4; j++) {
            x[j] = gate_row(s, checked, table, 2 * i, j);
            y[j] = gate_row(s, checked, table, 2 * i + 1, j);
        }
        E da = sub(y[0], x[0]), db = sub(y[1], x[1]), ds = sub(y[3], x[3]);
        E w = wt(i, lo, hi, s.low);
        E low = sub(mul(x[0], x[1]), x[2]), quad = mul(da, db);
        t.v[0] = low;
        t.v[1] = sub(sub(sub(mul(y[0], y[1]), y[2]), low), quad);
        t.v[2] = quad;
        E sx = x[3];
        if (any(sx != 0) || any(ds != 0)) {
            E sp[8], dp[8];
            sp[0] = dp[0] = E(1, 0);
            for (uint k = 1; k < 8; k++) {
                sp[k] = mul(sp[k - 1], sx);
                dp[k] = mul(dp[k - 1], ds);
            }
            const ulong bin[] = {1, 7, 21, 35, 35, 21, 7, 1};
            for (uint k = 0; k < 8; k++)
                t.v[k] = add(t.v[k], scaled(mul(sp[7 - k], dp[k]), bin[k]));
        }
        for (uint k = 0; k < 8; k++)
            t.v[k] = mul(w, t.v[k]);
    }
    threadgroup Coeff<8> shared[128];
    reduce(t, shared, lane, group, out);
}
inline uint code(device const ulong *positive, device const ulong *negative, constant ushort *tern, uint bit,
                 uint width, uint zero) {
    uint offset = bit % 64;
    ulong2 word = ulong2(positive[bit / 64], negative[bit / 64]);
    uint mask = (1u << width) - 1;
    return zero + tern[(word.x >> offset) & mask] - tern[(word.y >> offset) & mask];
}
inline E normrow(constant Shape &s, device const ulong *positive, device const ulong *negative, device const E *lut,
                 constant ushort *tern, device E *rows, uint i) {
    if (i >= ((s.norm + (1u << s.r) - 1) >> s.r))
        return 0;
    if (s.r <= 3) {
        uint zeros[] = {1, 4, 40, 3280};
        return lut[code(positive, negative, tern, i << s.r, 1u << s.r, zeros[s.r])];
    }
    uint j = i << (s.r - 4);
    E x, y;
    uint prev = (s.norm + (1u << (s.r - 1)) - 1) >> (s.r - 1);
    if (s.r == 4) {
        x = lut[code(positive, negative, tern, 16 * i, 8, 3280)];
        y = 2 * i + 1 < prev ? lut[code(positive, negative, tern, 16 * i + 8, 8, 3280)] : E(0);
    } else {
        x = rows[j];
        y = 2 * i + 1 < prev ? rows[j + (1u << (s.r - 5))] : E(0);
    }
    E v = fold(x, y, s.challenge);
    rows[j] = v;
    return v;
}
kernel void packed_norm(device const ulong *positive [[buffer(0)]], device const ulong *negative [[buffer(8)]],
                        device E *rows [[buffer(1)]], device const E *lo [[buffer(2)]],
                        device const E *hi [[buffer(3)]], device E *out [[buffer(4)]], constant Shape &s [[buffer(5)]],
                        device const E *lut [[buffer(6)]], constant ushort *tern [[buffer(7)]],
                        uint group [[threadgroup_position_in_grid]], uint lane [[thread_position_in_threadgroup]]) {
    uint i = group * 128 + lane;
    Coeff<4> t = {};
    if (i < s.count) {
        E a = normrow(s, positive, negative, lut, tern, rows, 2 * i),
          b = normrow(s, positive, negative, lut, tern, rows, 2 * i + 1);
        if (any(a != 0) || any(b != 0)) {
            E w = wt(i, lo, hi, s.low), d = sub(b, a), sq = mul(a, a), ds = mul(d, d);
            t.v[0] = mul(w, mul(a, sub(sq, E(1, 0))));
            t.v[1] = mul(w, mul(d, sub(scaled(sq, 3), E(1, 0))));
            t.v[2] = mul(w, scaled(mul(a, ds), 3));
            t.v[3] = mul(w, mul(d, ds));
        }
    }
    threadgroup Coeff<4> shared[128];
    reduce(t, shared, lane, group, out);
}
template <uint N>
inline void tail(device const E *v, device E *out, uint n, uint g, uint lane, threadgroup Coeff<N> *shared) {
    uint i = 128 * g + lane;
    Coeff<N> t = {};
    if (i < n)
        for (uint j = 0; j < N; j++)
            t.v[j] = v[N * i + j];
    reduce(t, shared, lane, g, out);
}
kernel void packed_tail4(device const E *v [[buffer(0)]], device E *out [[buffer(1)]], constant uint &n [[buffer(2)]],
                         uint g [[threadgroup_position_in_grid]], uint lane [[thread_position_in_threadgroup]]) {
    threadgroup Coeff<4> shared[128];
    tail(v, out, n, g, lane, shared);
}
kernel void packed_tail8(device const E *v [[buffer(0)]], device E *out [[buffer(1)]], constant uint &n [[buffer(2)]],
                         uint g [[threadgroup_position_in_grid]], uint lane [[thread_position_in_threadgroup]]) {
    threadgroup Coeff<8> shared[128];
    tail(v, out, n, g, lane, shared);
}

} // namespace packed_oracle
