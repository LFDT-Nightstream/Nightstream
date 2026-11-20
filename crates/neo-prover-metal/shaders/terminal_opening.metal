#include <metal_stdlib>
using namespace metal;

using E = ulong2;
constant ulong modulus = 0xffffffffffffffc5ul;

inline ulong add_fp(ulong a, ulong b) {
    ulong sum = a + b;
    sum += sum < a ? 59ul : 0ul;
    return min(sum, sum + 59ul);
}
inline ulong sub_fp(ulong a, ulong b) {
    return (a - b) - (a < b ? 59ul : 0ul);
}
inline ulong mul_fp(ulong a, ulong b) {
    ulong lo = a * b;
    ulong hi = mulhi(a, b);
    ulong folded = lo + 59ul * hi;
    ulong value = folded + 59ul * (mulhi(hi, 59ul) + (folded < lo));
    value += value < folded ? 59ul : 0ul;
    return min(value, value + 59ul);
}
inline E add(E a, E b) { return E(add_fp(a.x, b.x), add_fp(a.y, b.y)); }
inline E sub(E a, E b) { return E(sub_fp(a.x, b.x), sub_fp(a.y, b.y)); }
inline E scale(E a, ulong factor) { return E(mul_fp(a.x, factor), mul_fp(a.y, factor)); }

struct Config {
    uint slots, blocks, stride, tile, workers, signed_witness;
    uint offsets_head, rows_head, tags_head, factors_head, eq_head;
    uint slots_head, starts_head, widths_head, witness_head, negative_head;
    uint first_head, tasks_head, heavy_head;
};

template <typename T>
inline T read_split(device const T *head, device const T *tail, uint span, uint index) {
    return index < span ? head[index] : tail[index - span];
}

inline E weighted(uint from, uint to, uint step, constant Config &c,
                  device const uchar *rows, device const uchar *rows_tail,
                  device const ushort *tags, device const ushort *tags_tail,
                  device const ulong *factors, device const ulong *factors_tail,
                  device const E *eq, device const E *eq_tail) {
    E result = 0;
    for (uint entry = from; entry < to; entry += step) {
        uint offset = entry * c.stride;
        uint row = 0;
        for (uint digit = 0; digit < c.stride; ++digit)
            row |= uint(read_split(rows, rows_tail, c.rows_head, offset + digit)) << (8 * digit);
        ushort tag = read_split(tags, tags_tail, c.tags_head, entry);
        E weight = read_split(eq, eq_tail, c.eq_head, row);
        if (tag == 0)
            result = add(result, weight);
        else if (tag == 1)
            result = sub(result, weight);
        else
            result = add(result, scale(weight, read_split(factors, factors_tail, c.factors_head, tag)));
    }
    return result;
}

inline E reduce_simd(E value) {
    for (ushort step = 16; step > 0; step >>= 1)
        value = add(value, as_type<E>(simd_shuffle_down(as_type<uint4>(value), step)));
    return value;
}

struct LayoutView {
    device const uint *slots;
    device const uint *slots_tail;
    device const uint *starts;
    device const uint *starts_tail;
    device const uchar *widths;
    device const uchar *widths_tail;

    uint2 interval(uint index, constant Config &c) const {
        uint slot = read_split(slots, slots_tail, c.slots_head, index);
        uint start = read_split(starts, starts_tail, c.starts_head, slot);
        uint width = read_split(widths, widths_tail, c.widths_head, slot);
        return uint2(start, start + width);
    }

    uint lower_bound(uint value, uint first, uint end, constant Config &c) const {
        while (first < end) {
            uint mid = first + (end - first) / 2;
            if (interval(mid, c).y <= value)
                first = mid + 1;
            else
                end = mid;
        }
        return first;
    }

    uint first_slot(uint guess, uint boundary, constant Config &c) const {
        if ((guess == 0 || interval(guess - 1, c).y <= boundary)
                && (guess == c.slots || interval(guess, c).y > boundary))
            return guess;
        return lower_bound(boundary, 0, c.slots, c);
    }
};

kernel void terminal_weights(device const uint *offsets [[buffer(0)]],
                    device const uchar *rows [[buffer(1)]],
                    device const ushort *tags [[buffer(2)]],
                    device const ulong *factors [[buffer(3)]],
                    device const E *eq [[buffer(4)]],
                    device E *out [[buffer(5)]],
                    constant Config &c [[buffer(6)]],
                    device const uint *offsets_tail [[buffer(7)]],
                    device const uchar *rows_tail [[buffer(8)]],
                    device const ushort *tags_tail [[buffer(9)]],
                    device const ulong *factors_tail [[buffer(10)]],
                    device const E *eq_tail [[buffer(11)]],
                    uint slot [[thread_position_in_grid]]) {
    if (slot >= c.slots)
        return;
    uint start = read_split(offsets, offsets_tail, c.offsets_head, slot);
    uint end = read_split(offsets, offsets_tail, c.offsets_head, slot + 1);
    if (end - start <= 128)
        out[slot] = weighted(start, end, 1, c, rows, rows_tail, tags, tags_tail, factors, factors_tail, eq, eq_tail);
}

kernel void terminal_heavy_weights(device const uint2 *tasks [[buffer(0)]],
                          device const uint2 *tasks_tail [[buffer(7)]],
                          device const uchar *rows [[buffer(1)]],
                          device const ushort *tags [[buffer(2)]],
                          device const ulong *factors [[buffer(3)]],
                          device const E *eq [[buffer(4)]],
                          device E *partial [[buffer(5)]],
                          constant Config &c [[buffer(6)]],
                          device const uchar *rows_tail [[buffer(8)]],
                          device const ushort *tags_tail [[buffer(9)]],
                          device const ulong *factors_tail [[buffer(10)]],
                          device const E *eq_tail [[buffer(11)]],
                          uint group [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_threadgroup]]) {
    uint2 task = read_split(tasks, tasks_tail, c.tasks_head, group);
    E value = weighted(task.x + lane, task.y, 32, c, rows, rows_tail, tags, tags_tail, factors, factors_tail, eq, eq_tail);
    value = reduce_simd(value);
    if (lane == 0)
        partial[group] = value;
}

kernel void terminal_heavy_merge(device const uint4 *tasks [[buffer(0)]],
                        device const uint4 *tasks_tail [[buffer(3)]],
                        constant Config &c [[buffer(4)]],
                        device const E *partial [[buffer(1)]],
                        device E *out [[buffer(2)]],
                        uint group [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_threadgroup]]) {
    uint4 task = read_split(tasks, tasks_tail, c.heavy_head, group);
    E value = 0;
    for (uint index = task.y + lane; index < task.z; index += 32)
        value = add(value, partial[index]);
    value = reduce_simd(value);
    if (lane == 0)
        out[task.x] = value;
}

kernel void terminal_opening(device const uint *first [[buffer(0)]],
                    device const uint *first_tail [[buffer(15)]],
                    device const uint *slots [[buffer(1)]],
                    device const E *weights [[buffer(2)]],
                    device const uint *starts [[buffer(3)]],
                    device const uchar *widths [[buffer(4)]],
                    device const ulong *witness [[buffer(5)]],
                    device const ulong *negative [[buffer(6)]],
                    device const ulong *powers [[buffer(7)]],
                    device E *partial [[buffer(8)]],
                    constant Config &c [[buffer(9)]],
                    device const uint *slots_tail [[buffer(10)]],
                    device const uint *starts_tail [[buffer(11)]],
                    device const uchar *widths_tail [[buffer(12)]],
                    device const ulong *witness_tail [[buffer(13)]],
                    device const ulong *negative_tail [[buffer(14)]],
                    uint tile [[threadgroup_position_in_grid]],
                    uint coordinate [[thread_index_in_threadgroup]],
                    uint simd_id [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
    LayoutView layout{slots, slots_tail, starts, starts_tail, widths, widths_tail};
    threadgroup E forms[64];
    threadgroup ulong digits[64];
    threadgroup ulong nonzero_form[2];
    threadgroup uint live_witness[2];
    E accumulator = 0;
    for (uint block = tile * c.tile; block < min(c.blocks, (tile + 1) * c.tile); ++block) {
        ulong pos = 0, neg = 0;
        if (c.signed_witness) {
            pos = read_split(witness, witness_tail, c.witness_head, block);
            neg = read_split(negative, negative_tail, c.negative_head, block);
            if ((pos | neg) == 0)
                continue;
        }
        uint boundary = block * 64;
        uint first_slot = layout.first_slot(read_split(first, first_tail, c.first_head, block), boundary, c);
        if (first_slot >= c.slots || layout.interval(first_slot, c).x >= boundary + 64)
            continue;
        uint end = layout.first_slot(read_split(first, first_tail, c.first_head, block + 1), boundary + 64, c);
        uint column = boundary + coordinate;
        uint slot = layout.lower_bound(column, first_slot, min(end + 1, c.slots), c);
        E value = 0;
        if (slot < c.slots) {
            uint2 interval = layout.interval(slot, c);
            if (interval.x <= column && column < interval.y)
                value = scale(weights[slot], powers[column - interval.x]);
        }
        forms[coordinate] = value;
        ulong digit = c.signed_witness
            ? (((pos >> coordinate) & 1ul) ? 1ul : ((neg >> coordinate) & 1ul) ? modulus - 1 : 0ul)
            : read_split(witness, witness_tail, c.witness_head, column);
        digits[coordinate] = digit;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        E carry = scale(forms[(coordinate + 63) & 63], powers[1]);
        value = coordinate == 0 ? add(value, carry) : sub(value, carry);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        forms[coordinate] = value;
        ulong ballot = simd_sum(any(value != E(0)) ? (1u << lane) : 0u);
        bool live = simd_any(digit != 0);
        if (lane == 0) {
            nonzero_form[simd_id] = ballot;
            live_witness[simd_id] = live;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ulong mask = nonzero_form[0] | (nonzero_form[1] << 32);
        if ((live_witness[0] | live_witness[1]) == 0)
            mask = 0;
        while (mask != 0) {
            uint offset = ctz(mask);
            mask &= mask - 1;
            ulong d = digits[(coordinate + offset) & 63];
            E form = forms[offset];
            if (d == 0 || all(form == E(0)))
                continue;
            bool subtract = coordinate + offset >= 64;
            if (d == modulus - 1) {
                subtract = !subtract;
                d = 1;
            }
            E term = d == 1 ? form : scale(form, d);
            accumulator = subtract ? sub(accumulator, term) : add(accumulator, term);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    partial[tile * 64 + coordinate] = accumulator;
}

kernel void terminal_partial_openings(device const E *source [[buffer(0)]],
                             device E *out [[buffer(1)]],
                             constant Config &c [[buffer(2)]],
                             uint group [[threadgroup_position_in_grid]],
                             uint coordinate [[thread_index_in_threadgroup]]) {
    E sum = 0;
    for (uint index = group * 64; index < min(c.workers, (group + 1) * 64); ++index)
        sum = add(sum, source[index * 64 + coordinate]);
    out[group * 64 + coordinate] = sum;
}

kernel void terminal_finish_opening(device const E *partial [[buffer(0)]],
                           device E *out [[buffer(1)]],
                           constant Config &c [[buffer(2)]],
                           device const ulong *powers [[buffer(3)]],
                           uint coordinate [[thread_position_in_grid]]) {
    E folded = 0;
    for (uint group = 0; group < (c.workers + 63) / 64; ++group)
        folded = add(folded, partial[group * 64 + coordinate]);
    threadgroup E delta[64];
    delta[coordinate] = folded;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    E result = 0;
    for (uint offset = 0; offset < 64; ++offset) {
        E term = scale(delta[(coordinate + offset) & 63], powers[offset]);
        result = coordinate + offset < 64 ? add(result, term) : sub(result, term);
    }
    out[coordinate] = scale(result, powers[256]);
}
