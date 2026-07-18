// Compact public Pi_DEC surfaces derived directly from the resident parent.
// Partials are child-major, ring-row-major, then column chunk.

constant ulong DEC_PUBLIC_CHUNK_BLOCKS = 512;
constant ushort DEC_PUBLIC_CHILDREN_PER_THREAD = 7;

kernel void dec_y_zcol_partials(
    device const ulong *parent [[buffer(0)]],
    device const ulong *chi_s [[buffer(1)]],
    device const ulong *shape [[buffer(2)]],
    device ulong *partials [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    ulong active_rows = shape[0];
    ulong blocks = shape[1];
    ulong child_count = shape[2];
    ulong chunks = shape[3];
    ulong chunk = (ulong)index % chunks;
    ulong packed = (ulong)index / chunks;
    ulong row = packed % RING_DEGREE;
    ulong first_child = (packed / RING_DEGREE) * DEC_PUBLIC_CHILDREN_PER_THREAD;
    if (first_child >= child_count) {
        return;
    }

    Kx sums[DEC_PUBLIC_CHILDREN_PER_THREAD];
    for (ushort local = 0; local < DEC_PUBLIC_CHILDREN_PER_THREAD; ++local) {
        sums[local] = Kx{0, 0};
    }
    ulong start = chunk * DEC_PUBLIC_CHUNK_BLOCKS;
    ulong end = min(start + DEC_PUBLIC_CHUNK_BLOCKS, blocks);
    for (ulong block = start; block < end; ++block) {
        ulong logical = block * RING_DEGREE + row;
        if (logical >= active_rows) {
            break;
        }
        ulong word = gl_from_word(parent[row * blocks + block]);
        bool negative = word > (GOLDILOCKS_MODULUS - 1) / 2;
        ulong magnitude = negative ? GOLDILOCKS_MODULUS - word : word;
        Kx weight = load_k(chi_s, logical);
        for (ushort local = 0; local < DEC_PUBLIC_CHILDREN_PER_THREAD; ++local) {
            ulong child = first_child + local;
            if (child < child_count && ((magnitude >> child) & 1ul) != 0) {
                sums[local] = negative ? kx_sub(sums[local], weight) : kx_add(sums[local], weight);
            }
        }
    }
    for (ushort local = 0; local < DEC_PUBLIC_CHILDREN_PER_THREAD; ++local) {
        ulong child = first_child + local;
        if (child < child_count) {
            ulong output = (child * RING_DEGREE + row) * chunks + chunk;
            partials[2 * output] = sums[local].c0;
            partials[2 * output + 1] = sums[local].c1;
        }
    }
}

// The second stage reduces chunks to one K value per child and ring row.
kernel void dec_y_zcol_reduce(
    device const ulong *partials [[buffer(0)]],
    device const ulong *shape [[buffer(1)]],
    device ulong *output [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    ulong child_count = shape[2];
    ulong chunks = shape[3];
    ulong child = (ulong)index / RING_DEGREE;
    if (child >= child_count) {
        return;
    }
    Kx sum = Kx{0, 0};
    ulong partial_base = (ulong)index * chunks;
    for (ulong chunk = 0; chunk < chunks; ++chunk) {
        sum = kx_add(sum, load_k(partials, partial_base + chunk));
    }
    output[2 * index] = sum.c0;
    output[2 * index + 1] = sum.c1;
}
