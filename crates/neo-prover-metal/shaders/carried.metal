// The carried table of the running sources: each block's weighted lane sum
// `combined`, projected through the basis forms, plus the row sums.
inline void joint_carried_project(
    threadgroup Kx *combined,
    device const ulong *shape,
    device const ulong *basis_re,
    device const ulong *basis_im,
    device const ulong *row_sums,
    device ulong *output,
    uint block,
    uint lane) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ulong blocks = shape[1];
    ulong column = (ulong)block * RING_DEGREE + lane;
    if (lane >= RING_DEGREE || column >= shape[7]) return;
    // `kx_mul` term by term, with the three products of each term added
    // unreduced: RING_DEGREE terms for each sum, far below PRODUCT_SUM_TERMS.
    ProductSum real = ProductSum{0, 0, 0, 0};
    ProductSum imaginary = ProductSum{0, 0, 0, 0};
    ProductSum mixed = ProductSum{0, 0, 0, 0};
    for (ulong input = 0; block < blocks && input < RING_DEGREE; ++input) {
        ulong entry = lane * RING_DEGREE + input;
        Kx basis = Kx{fq_from_word(basis_re[entry]), fq_from_word(basis_im[entry])};
        Kx source = combined[input];
        product_sum_add(real, basis.c0, source.c0);
        product_sum_add(imaginary, basis.c1, source.c1);
        product_sum_add(mixed, fq_add(basis.c0, basis.c1), fq_add(source.c0, source.c1));
    }
    ulong c0_product = product_sum_reduce(real);
    ulong c1_product = product_sum_reduce(imaginary);
    Kx value = Kx{
        fq_add(c0_product, fq_mul_by_non_residue(c1_product)),
        fq_sub(fq_sub(product_sum_reduce(mixed), c0_product), c1_product)};
    if (column < shape[8]) value = kx_add(value, load_k(row_sums, column));
    output[2 * column] = value.c0;
    output[2 * column + 1] = value.c1;
}

kernel void joint_carried_projection(
    device const ulong *masks [[buffer(0)]],
    device const ulong *coefficients [[buffer(1)]],
    device const ulong *shape [[buffer(2)]],
    device const ulong *basis_re [[buffer(3)]],
    device const ulong *basis_im [[buffer(4)]],
    device const ulong *row_sums [[buffer(5)]],
    device ulong *output [[buffer(6)]],
    uint block [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    ulong count = shape[0];
    ulong blocks = shape[1];
    ulong magnitudes = shape[6];
    threadgroup Kx combined[RING_DEGREE];
    if (lane < RING_DEGREE) {
        Kx value = Kx{0, 0};
        ulong bit = 1ul << lane;
        for (ulong source = 0; block < blocks && source < count; ++source) {
            Kx coefficient = load_k(coefficients, source);
            ulong base = 2 * magnitudes * (source * blocks + block);
            for (ulong magnitude = 1; magnitude <= magnitudes; ++magnitude) {
                Kx scaled = Kx{fq_mul(coefficient.c0, magnitude), fq_mul(coefficient.c1, magnitude)};
                if ((masks[base + 2 * (magnitude - 1)] & bit) != 0) value = kx_add(value, scaled);
                else if ((masks[base + 2 * (magnitude - 1) + 1] & bit) != 0) value = kx_sub(value, scaled);
            }
        }
        combined[lane] = value;
    }
    joint_carried_project(combined, shape, basis_re, basis_im, row_sums, output, block, lane);
}

// The same table over running sources without norm terms, whose field values
// are stored source by source, block by block and lane by lane.
kernel void joint_carried_projection_dense(
    device const ulong *values [[buffer(0)]],
    device const ulong *coefficients [[buffer(1)]],
    device const ulong *shape [[buffer(2)]],
    device const ulong *basis_re [[buffer(3)]],
    device const ulong *basis_im [[buffer(4)]],
    device const ulong *row_sums [[buffer(5)]],
    device ulong *output [[buffer(6)]],
    uint block [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    ulong count = shape[0];
    ulong blocks = shape[1];
    threadgroup Kx combined[RING_DEGREE];
    if (lane < RING_DEGREE) {
        Kx value = Kx{0, 0};
        for (ulong source = 0; block < blocks && source < count; ++source) {
            Kx coefficient = load_k(coefficients, source);
            ulong entry = fq_from_word(values[(source * blocks + block) * RING_DEGREE + lane]);
            value = kx_add(value, Kx{fq_mul(coefficient.c0, entry), fq_mul(coefficient.c1, entry)});
        }
        combined[lane] = value;
    }
    joint_carried_project(combined, shape, basis_re, basis_im, row_sums, output, block, lane);
}
