// Check the exact CCS polynomial on every row of the supplied matrix tables.
// The host owns statement binding and validates table and polynomial dimensions.
kernel void ccs_first_unsatisfied_row(
    device const ulong *tables [[buffer(0)]],
    device const ulong *term_headers [[buffer(1)]],
    device const ulong *term_variables [[buffer(2)]],
    device const ulong *shape [[buffer(3)]],
    device atomic_uint *first_failure [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
    ulong rows = shape[0];
    if ((ulong)row >= rows) {
        return;
    }
    ulong sum = 0;
    for (ulong term = 0; term < shape[1]; ++term) {
        ulong value = gl_from_word(term_headers[3 * term]);
        ulong start = term_headers[3 * term + 1];
        ulong end = start + term_headers[3 * term + 2];
        for (ulong variable = start; variable < end; ++variable) {
            ulong matrix = term_variables[2 * variable];
            ulong exponent = term_variables[2 * variable + 1];
            value = gl_mul(value, dec_pow(gl_from_word(tables[matrix * rows + row]), exponent));
        }
        sum = gl_add(sum, value);
    }
    if (sum != 0) {
        atomic_fetch_min_explicit(first_failure, row, memory_order_relaxed);
    }
}
