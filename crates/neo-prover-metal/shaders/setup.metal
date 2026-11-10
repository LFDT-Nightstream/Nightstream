// Layer-1 setup oracles (neo-spartan `setup.rs`): a key row on the cube
// x = 64·b + lane, the Möbius transform, the coset shift, a single-column
// DIF DFT, and the gather of a codeword's leaves from its bit-reversed
// output.

constant uint SETUP_LANES = 64;

// Block `block` of key row shape[0]: the RING_DEGREE key coefficients for
// block < shape[1], and zero lanes after them and in every later block. The
// threads cover the whole rate-1/2 buffer, so its upper half is zero too.
kernel void setup_key_table(
    device const ulong *prefix [[buffer(0)]],
    device const ulong *shape [[buffer(1)]],
    device ulong *table [[buffer(2)]],
    uint block [[thread_position_in_grid]]) {
    device ulong *out = table + (ulong)block * SETUP_LANES;
    uint lane = 0;
    if (block < shape[1]) {
        production_ajtai_element(prefix, (uint)shape[0], block, out, 1);
        lane = RING_DEGREE;
    }
    for (; lane < SETUP_LANES; ++lane) {
        out[lane] = 0;
    }
}

// values[shape[0] + i] = 0.
kernel void setup_zero(
    device ulong *values [[buffer(0)]],
    device const ulong *shape [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
    values[shape[0] + i] = 0;
}

// One pass of a single-column transform over 2^shape[0] values: stages
// shape[2] .. shape[2] + shape[3], which mix the rows base + e·stride. A
// threadgroup owns SETUP_COLUMNS neighboring `low` offsets (fewer when the
// stride is smaller), so its reads and writes are coalesced. `twiddles` is
// null for the Möbius transform (v[x + 2^t] -= v[x] for every x without bit
// t; the stages commute) and holds ω^k for the DIF DFT of `dft_dif_pass`.
// That DFT's twiddle ω^((j·stride + low)·2^t) is the product of a row factor
// ω^(j·stride·2^t) and a column factor ω^(low·2^t), which the threadgroup
// loads once per stage.
constant uint SETUP_COLUMNS = 8;

inline void setup_pass(
    device ulong *values,
    device const ulong *twiddles,
    device const ulong *shape,
    bool mobius,
    uint group_index,
    uint tid,
    uint threads,
    threadgroup ulong *tile,
    threadgroup ulong *row_factors,
    threadgroup ulong *column_factors) {
    uint log_rows = (uint)shape[0];
    uint first = (uint)shape[2];
    uint stages = (uint)shape[3];
    ulong stride = 1ul << (log_rows - first - stages);
    // Every count is a power of two: shifts and masks, not division.
    uint log_lanes = min(log_rows - first - stages, (uint)ctz(SETUP_COLUMNS));
    uint lanes = 1u << log_lanes;
    uint log_blocks = log_rows - first - stages - log_lanes;
    ulong low0 = ((ulong)group_index & ((1ul << log_blocks) - 1)) << log_lanes;
    ulong base = (((ulong)group_index >> log_blocks) << (log_rows - first)) | low0;
    uint elements = (1u << stages) << log_lanes;
    for (uint index = tid; index < elements; index += threads) {
        tile[index] = gl_from_word(values[base + ((ulong)(index >> log_lanes) << (log_rows - first - stages)) + (index & (lanes - 1))]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint pairs = elements / 2;
    for (uint u = 0; u < stages; ++u) {
        uint log_half = stages - 1 - u;
        uint half_size = 1u << log_half;
        if (!mobius) {
            uint t = first + u;
            for (uint j = tid; j < half_size; j += threads) {
                row_factors[j] = twiddles[((ulong)j * stride) << t];
            }
            for (uint lane = tid; lane < lanes; lane += threads) {
                column_factors[lane] = twiddles[(low0 + lane) << t];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint p = tid; p < pairs; p += threads) {
            uint lane = p & (lanes - 1);
            uint q = p >> log_lanes;
            uint j = q & (half_size - 1);
            uint e0 = ((q >> log_half) << (log_half + 1)) + j;
            uint at0 = (e0 << log_lanes) + lane;
            uint at1 = at0 + (half_size << log_lanes);
            ulong a = tile[at0];
            ulong b = tile[at1];
            if (mobius) {
                tile[at1] = gl_sub(b, a);
            } else {
                ulong twiddle = gl_mul(row_factors[j], column_factors[lane]);
                tile[at0] = gl_add(a, b);
                tile[at1] = gl_mul(gl_sub(a, b), twiddle);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint index = tid; index < elements; index += threads) {
        values[base + ((ulong)(index >> log_lanes) << (log_rows - first - stages)) + (index & (lanes - 1))] = tile[index];
    }
}

kernel void setup_mobius_pass(
    device ulong *values [[buffer(0)]],
    device const ulong *shape [[buffer(1)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    threadgroup ulong tile[(1u << DFT_PASS_BITS) * SETUP_COLUMNS];
    setup_pass(values, nullptr, shape, true, group, tid, threads, tile, nullptr, nullptr);
}

kernel void setup_dif_pass(
    device ulong *values [[buffer(0)]],
    device const ulong *twiddles [[buffer(1)]],
    device const ulong *shape [[buffer(2)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    threadgroup ulong tile[(1u << DFT_PASS_BITS) * SETUP_COLUMNS];
    threadgroup ulong row_factors[1u << (DFT_PASS_BITS - 1)];
    threadgroup ulong column_factors[SETUP_COLUMNS];
    setup_pass(values, twiddles, shape, false, group, tid, threads, tile, row_factors, column_factors);
}

// values[i] = g^i with g = shape[0], for the SETUP_LANES values of chunk
// `chunk`: the twiddles of `setup_dif_pass`.
kernel void setup_powers(
    device ulong *values [[buffer(0)]],
    device const ulong *shape [[buffer(1)]],
    uint chunk [[thread_position_in_grid]]) {
    ulong g = shape[0];
    ulong start = (ulong)chunk * SETUP_LANES;
    ulong power = 1;
    ulong base = g;
    for (ulong exponent = start; exponent != 0; exponent >>= 1) {
        if (exponent & 1) {
            power = gl_mul(power, base);
        }
        base = gl_mul(base, base);
    }
    for (uint i = 0; i < SETUP_LANES; ++i) {
        values[start + i] = power;
        power = gl_mul(power, g);
    }
}

// values[i] = values[i]·g^i with g = shape[0], for the SETUP_LANES values
// of chunk `chunk`.
kernel void setup_coset(
    device ulong *values [[buffer(0)]],
    device const ulong *shape [[buffer(1)]],
    uint chunk [[thread_position_in_grid]]) {
    ulong g = shape[0];
    ulong start = (ulong)chunk * SETUP_LANES;
    ulong power = 1;
    ulong base = g;
    for (ulong exponent = start; exponent != 0; exponent >>= 1) {
        if (exponent & 1) {
            power = gl_mul(power, base);
        }
        base = gl_mul(base, base);
    }
    for (uint i = 0; i < SETUP_LANES; ++i) {
        values[start + i] = gl_mul(values[start + i], power);
        power = gl_mul(power, g);
    }
}

// out[i] = codeword position leaves[i / 8] + (i % 8)·shape[1], read from
// the bit-reversed output of a 2^shape[0]-row `setup_dif_pass` run.
kernel void setup_gather(
    device const ulong *values [[buffer(0)]],
    device const uint *leaves [[buffer(1)]],
    device const ulong *shape [[buffer(2)]],
    device ulong *out [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    uint log_rows = (uint)shape[0];
    uint position = leaves[i / 8] + (i % 8) * (uint)shape[1];
    out[i] = values[reverse_bits(position) >> (32 - log_rows)];
}
