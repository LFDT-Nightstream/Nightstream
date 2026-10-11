// Batched DFT over the columns of a row-major Goldilocks matrix: row k of
// the result is Σ_i row_i·ω^{ik}. Radix-2 decimation-in-frequency stages
// run in passes of up to DFT_PASS_BITS stages in threadgroup memory, then a
// bit reversal puts the rows in natural order.

constant uint DFT_PASS_BITS = 8;
constant uint DFT_COLUMNS = 16;

// twiddles[k] = ω^k with ω = shape[0].
kernel void dft_twiddles(
    device ulong *twiddles [[buffer(0)]],
    device const ulong *shape [[buffer(1)]],
    uint k [[thread_position_in_grid]]) {
    ulong base = shape[0];
    ulong power = 1;
    for (uint exponent = k; exponent != 0; exponent >>= 1) {
        if (exponent & 1) {
            power = gl_mul(power, base);
        }
        base = gl_mul(base, base);
    }
    twiddles[k] = power;
}

// Stages shape[2] .. shape[2] + shape[3] of a 2^shape[0]-row DFT over
// shape[1] columns. A threadgroup owns the 2^stages rows that these stages
// mix, for DFT_COLUMNS columns (or all columns of a narrower matrix).
kernel void dft_dif_pass(
    device ulong *values [[buffer(0)]],
    device const ulong *twiddles [[buffer(1)]],
    device const ulong *shape [[buffer(2)]],
    uint group_index [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    threadgroup ulong tile[(1u << DFT_PASS_BITS) * DFT_COLUMNS];
    uint log_rows = (uint)shape[0];
    ulong width = shape[1];
    uint first = (uint)shape[2];
    uint stages = (uint)shape[3];
    uint columns = (uint)min(width, (ulong)DFT_COLUMNS);
    ulong groups = 1ul << (log_rows - stages);
    ulong group = group_index % groups;
    ulong column0 = (group_index / groups) * columns;
    // The mixed rows are `base + e·stride` for e < 2^stages.
    ulong stride = 1ul << (log_rows - first - stages);
    ulong low = group % stride;
    ulong base = ((group / stride) << (log_rows - first)) | low;
    uint elements = (1u << stages) * columns;
    for (uint index = tid; index < elements; index += threads) {
        ulong row = base + (ulong)(index / columns) * stride;
        tile[index] = gl_from_word(values[row * width + column0 + index % columns]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint pairs = elements / 2;
    for (uint u = 0; u < stages; ++u) {
        uint half_size = 1u << (stages - 1 - u);
        for (uint p = tid; p < pairs; p += threads) {
            uint column = p % columns;
            uint q = p / columns;
            uint j = q % half_size;
            uint e0 = (q / half_size) * 2 * half_size + j;
            uint at0 = e0 * columns + column;
            uint at1 = at0 + half_size * columns;
            ulong a = tile[at0];
            ulong b = tile[at1];
            // ω^{(i mod h)·2^t} for the row i of e0 at global stage t.
            ulong twiddle = twiddles[((ulong)j * stride + low) << (first + u)];
            tile[at0] = gl_add(a, b);
            tile[at1] = gl_mul(gl_sub(a, b), twiddle);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint index = tid; index < elements; index += threads) {
        ulong row = base + (ulong)(index / columns) * stride;
        values[row * width + column0 + index % columns] = tile[index];
    }
}

// Swap rows i and bitrev(i) of a 2^shape[0]-row matrix of shape[1] columns.
kernel void dft_bit_reverse(
    device ulong *values [[buffer(0)]],
    device const ulong *shape [[buffer(1)]],
    uint row [[thread_position_in_grid]]) {
    uint log_rows = (uint)shape[0];
    ulong width = shape[1];
    uint other = reverse_bits(row) >> (32 - log_rows);
    if (row < other) {
        for (ulong column = 0; column < width; ++column) {
            ulong value = values[(ulong)row * width + column];
            values[(ulong)row * width + column] = values[(ulong)other * width + column];
            values[(ulong)other * width + column] = value;
        }
    }
}
