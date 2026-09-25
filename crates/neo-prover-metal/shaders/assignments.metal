// Witness-prefix encoding: masks, byte/ushort indices, then dense K values.
// All tables have an implicit zero suffix. Indices are shared across witnesses.

inline ulong joint_assignment_code(
    device const ulong *data, device const ulong *sources,
    ulong source, ulong index, ulong length, ulong encoding,
    ulong blocks, ulong magnitudes, ulong zero) {
    if (index >= length) return zero;
    ulong position = source * length + index;
    if (encoding == 1) return ((device const uchar *)data)[position];
    if (encoding == 2) return ((device const ushort *)data)[position];
    ulong value = joint_mask_value(data, blocks, sources[source], index, magnitudes);
    return value <= magnitudes ? zero + value : zero - gl_sub(0, value);
}

inline Kx joint_assignment_value(
    device const ulong *data, device const ulong *sources, device const ulong *values,
    ulong source, ulong index, ulong encoding,
    ulong blocks, ulong magnitudes, ulong length) {
    if (index >= length) return Kx{0, 0};
    if (encoding == 0) return Kx{joint_mask_value(data, blocks, sources[source], index, magnitudes), 0};
    if (encoding == 16) return load_k(data, source * length + index);
    ulong position = source * length + index;
    ulong code = encoding == 1 ? ((device const uchar *)data)[position] : ((device const ushort *)data)[position];
    return load_k(values, code);
}

kernel void joint_fold_assignment_values(
    device const ulong *values [[buffer(0)]],
    device const ulong *challenge_words [[buffer(1)]],
    device const ulong *shape [[buffer(2)]],
    device ulong *output [[buffer(3)]],
    uint code [[thread_position_in_grid]]) {
    ulong alphabet = shape[5];
    if ((ulong)code >= alphabet * alphabet) return;
    Kx low = load_k(values, code % alphabet);
    Kx high = load_k(values, code / alphabet);
    Kx value = kx_add(low, kx_mul(load_k(challenge_words, 0), kx_sub(high, low)));
    output[2 * code] = value.c0;
    output[2 * code + 1] = value.c1;
}

kernel void joint_fold_assignments(
    device const ulong *data [[buffer(0)]],
    device const ulong *challenge_words [[buffer(1)]],
    device const ulong *shape [[buffer(2)]],
    device ulong *output [[buffer(3)]],
    device const ulong *sources [[buffer(4)]],
    device const ulong *values [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    ulong length = shape[0];
    ulong count = shape[1];
    ulong blocks = shape[2];
    ulong magnitudes = shape[3];
    ulong encoding = shape[4];
    ulong alphabet = shape[5];
    ulong zero = shape[6];
    ulong next_encoding = shape[7];
    ulong next_len = (length + 1) / 2;
    if ((ulong)index >= count * next_len) return;
    ulong source = index / next_len;
    ulong low_index = 2 * (index % next_len);
    if (next_encoding != 16) {
        ulong low = joint_assignment_code(data, sources, source, low_index, length, encoding, blocks, magnitudes, zero);
        ulong high = joint_assignment_code(data, sources, source, low_index + 1, length, encoding, blocks, magnitudes, zero);
        ulong code = low + alphabet * high;
        if (next_encoding == 1) ((device uchar *)output)[index] = (uchar)code;
        else ((device ushort *)output)[index] = (ushort)code;
    } else {
        Kx low = joint_assignment_value(data, sources, values, source, low_index, encoding, blocks, magnitudes, length);
        Kx high = joint_assignment_value(data, sources, values, source, low_index + 1, encoding, blocks, magnitudes, length);
        Kx value = kx_add(low, kx_mul(load_k(challenge_words, 0), kx_sub(high, low)));
        output[2 * index] = value.c0;
        output[2 * index + 1] = value.c1;
    }
}
