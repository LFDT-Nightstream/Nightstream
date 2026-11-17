//! The kernels of `dec_forms.metal`: the row weights and the ring forms of
//! the openings, `Σ_rows χ(row)·M[row, column]` per active column block.

use cuda_device::{cuda_module, kernel, thread, SharedArray};

use crate::kernels::field::*;

pub(crate) const DEC_PARALLEL_FORM_LIST_THRESHOLD: u64 = 128;
pub(crate) const DEC_FORM_REDUCTION_THREADS: usize = 256;

#[inline(always)]
pub(crate) fn dec_pow(mut base: u64, mut exponent: u64) -> u64 {
    let mut value = 1;
    while exponent != 0 {
        if exponent & 1 != 0 {
            value = gl_mul(value, base);
        }
        base = gl_mul(base, base);
        exponent >>= 1;
    }
    value
}

/// Lane `local` of a row-block pattern: a signed unit, or a dense pattern
/// with its locals in order.
///
/// # Safety
///
/// The pattern tables hold the descriptor's pattern.
#[inline(always)]
unsafe fn dec_pattern_coefficient(
    descriptor: u32,
    local: u64,
    offsets: *const u32,
    locals: *const u8,
    coefficients: *const u64,
) -> u64 {
    if descriptor & 0x8000_0000 == 0 {
        if u64::from((descriptor >> 24) & 0x3f) != local {
            return 0;
        }
        return if descriptor & 0x4000_0000 == 0 {
            1
        } else {
            GOLDILOCKS_MODULUS - 1
        };
    }
    let pattern = (descriptor & 0x7fff_ffff) as usize;
    // SAFETY: the caller's pattern tables.
    unsafe {
        let mut start = *offsets.add(pattern);
        let mut end = *offsets.add(pattern + 1);
        if u64::from(end - start) == RING_DEGREE {
            return gl_from_word(*coefficients.add((u64::from(start) + local) as usize));
        }
        let limit = end;
        while start < end {
            let middle = start + (end - start) / 2;
            if u64::from(*locals.add(middle as usize)) < local {
                start = middle + 1;
            } else {
                end = middle;
            }
        }
        if start < limit && u64::from(*locals.add(start as usize)) == local {
            gl_from_word(*coefficients.add(start as usize))
        } else {
            0
        }
    }
}

/// `χ(row)` from the two halves of the tensor point.
///
/// # Safety
///
/// `low` and `high` hold the halves' tables.
#[inline(always)]
unsafe fn dec_factored_weight(low: *const u64, high: *const u64, low_bits: u64, row: u64) -> Kx {
    let a = (2 * (row & ((1 << low_bits) - 1))) as usize;
    let b = (2 * (row >> low_bits)) as usize;
    // SAFETY: the caller's tables.
    unsafe {
        kx_mul(
            Kx {
                c0: *low.add(a),
                c1: *low.add(a + 1),
            },
            Kx {
                c0: *high.add(b),
                c1: *high.add(b + 1),
            },
        )
    }
}

/// # Safety
///
/// The pattern tables hold the entry's pattern and `chi` its row.
#[inline(always)]
unsafe fn dec_original_term(
    entry: [u32; 2],
    local: u64,
    component: u64,
    pattern_offsets: *const u32,
    pattern_locals: *const u8,
    pattern_coefficients: *const u64,
    chi: *const u64,
) -> u64 {
    // SAFETY: the caller's tables.
    unsafe {
        let coefficient =
            dec_pattern_coefficient(entry[1], local, pattern_offsets, pattern_locals, pattern_coefficients);
        gl_mul(*chi.add((2 * u64::from(entry[0]) + component) as usize), coefficient)
    }
}

/// Sum the block's `DEC_FORM_REDUCTION_THREADS` values in `partials`;
/// lane 0 returns the sum.
///
/// # Safety
///
/// Every thread of the block calls it with its own lane.
#[inline(always)]
unsafe fn dec_reduce(partials: *mut u64, lane: usize, value: u64) -> u64 {
    // SAFETY: `lane` and `lane + stride` index the block's shared words.
    unsafe {
        *partials.add(lane) = value;
        thread::sync_threads();
        let mut stride = DEC_FORM_REDUCTION_THREADS / 2;
        while stride > 0 {
            if lane < stride {
                *partials.add(lane) = gl_add(*partials.add(lane), *partials.add(lane + stride));
            }
            thread::sync_threads();
            stride >>= 1;
        }
        *partials
    }
}

/// # Safety
///
/// `starts` and `ranks` hold the span layer of `column`.
#[inline(always)]
unsafe fn dec_span_rank(starts: *const u64, ranks: *const u32, column: u64) -> u64 {
    let below = (1u64 << (column % 64)) - 1;
    let word = (column / 64) as usize;
    // SAFETY: the caller's span layer.
    unsafe { u64::from(*ranks.add(word)) + u64::from((*starts.add(word) & below).count_ones()) }
}

#[cuda_module]
pub mod dec_form_kernels {
    use super::*;

    /// `chi[row] = χ(shape[8] + row)` for `row < count`.
    #[kernel]
    pub fn dec_build_row_weights(low: *const u64, high: *const u64, shape: [u64; 9], chi: *mut u64, count: u64) {
        let row = thread::index_1d().get() as u64;
        if row >= count {
            return;
        }
        // SAFETY: the host gives both tensor halves and `count` weights.
        unsafe { store_k(chi, row, dec_factored_weight(low, high, shape[7], shape[8] + row)) };
    }

    /// The ring forms of the active blocks `shape[5] ..` with fewer than
    /// `DEC_PARALLEL_FORM_LIST_THRESHOLD` entries, and of the identity
    /// matrices, at form word `index < count`.
    #[kernel]
    pub fn dec_build_ring_forms(
        active_offsets: *const u64,
        active_local_masks: *const u64,
        matrix_identity: *const u32,
        entries: *const [u32; 2],
        pattern_offsets: *const u32,
        chi: *const u64,
        shape: [u64; 9],
        forms: *mut u64,
        active_blocks: *const u32,
        pattern_locals: *const u8,
        pattern_coefficients: *const u64,
        chi_low: *const u64,
        chi_high: *const u64,
        count: u64,
    ) {
        let index = thread::index_1d().get() as u64;
        if index >= count {
            return;
        }
        let blocks = shape[1];
        let local = index % RING_DEGREE;
        let rest = index / RING_DEGREE;
        let component = rest % 2;
        let active = shape[5] + rest / 2;
        // SAFETY: the host gives the opening plan and `count` form words.
        unsafe {
            let encoded = u64::from(*active_blocks.add(active as usize));
            let matrix = encoded / blocks;
            let block = encoded % blocks;
            let out = forms.add(index as usize);
            if *matrix_identity.add(matrix as usize) != 0 {
                let row = block * RING_DEGREE + local;
                let valid = if matrix == 0 {
                    row < shape[4]
                } else {
                    row >= shape[8] && row - shape[8] < shape[2]
                };
                let value = if valid {
                    dec_factored_weight(chi_low, chi_high, shape[7], row)
                } else {
                    KX_ZERO
                };
                *out = if component == 0 { value.c0 } else { value.c1 };
                return;
            }
            if *active_local_masks.add(active as usize) & (1 << local) == 0 {
                *out = 0;
                return;
            }
            let start = *active_offsets.add(active as usize);
            let end = *active_offsets.add(active as usize + 1);
            if end - start >= DEC_PARALLEL_FORM_LIST_THRESHOLD {
                return;
            }
            let mut value = 0;
            for entry in start..end {
                value = gl_add(
                    value,
                    dec_original_term(
                        *entries.add(entry as usize),
                        local,
                        component,
                        pattern_offsets,
                        pattern_locals,
                        pattern_coefficients,
                        chi,
                    ),
                );
            }
            *out = value;
        }
    }

    /// The ring form lane of each long list, one block per list and
    /// component.
    #[kernel]
    pub fn dec_build_parallel_original_forms(
        active_offsets: *const u64,
        entries: *const [u32; 2],
        pattern_offsets: *const u32,
        chi: *const u64,
        shape: [u64; 9],
        forms: *mut u64,
        parallel_lists: *const u32,
        pattern_locals: *const u8,
        pattern_coefficients: *const u64,
    ) {
        static mut PARTIALS: SharedArray<u64, DEC_FORM_REDUCTION_THREADS> = SharedArray::UNINIT;
        let lane = thread::threadIdx_x() as usize;
        let group = u64::from(thread::blockIdx_x());
        let component = group % 2;
        // SAFETY: the host gives the opening plan, one list per block pair
        // and the forms of the active blocks `shape[5] .. shape[6]`.
        unsafe {
            let encoded = u64::from(*parallel_lists.add((group / 2) as usize));
            let active = encoded / RING_DEGREE;
            if active < shape[5] || active >= shape[6] {
                return;
            }
            let local = encoded % RING_DEGREE;
            let start = *active_offsets.add(active as usize);
            let end = *active_offsets.add(active as usize + 1);
            let mut value = 0;
            let mut entry = start + lane as u64;
            while entry < end {
                value = gl_add(
                    value,
                    dec_original_term(
                        *entries.add(entry as usize),
                        local,
                        component,
                        pattern_offsets,
                        pattern_locals,
                        pattern_coefficients,
                        chi,
                    ),
                );
                entry += DEC_FORM_REDUCTION_THREADS as u64;
            }
            let sum = dec_reduce(&raw mut PARTIALS as *mut u64, lane, value);
            if lane == 0 {
                *forms.add((((active - shape[5]) * 2 + component) * RING_DEGREE + local) as usize) = sum;
            }
        }
    }

    /// The partial sum of one tile of a very long list, one block per tile
    /// and component.
    #[kernel]
    pub fn dec_build_parallel_original_form_tiles(
        active_offsets: *const u64,
        entries: *const [u32; 2],
        pattern_offsets: *const u32,
        chi: *const u64,
        shape: [u64; 9],
        tiles: *const u32,
        tile_partials: *mut u64,
        pattern_locals: *const u8,
        pattern_coefficients: *const u64,
    ) {
        static mut PARTIALS: SharedArray<u64, DEC_FORM_REDUCTION_THREADS> = SharedArray::UNINIT;
        let lane = thread::threadIdx_x() as usize;
        let group = u64::from(thread::blockIdx_x());
        let component = group % 2;
        let descriptor = (3 * (group / 2)) as usize;
        // SAFETY: the host gives the opening plan and one partial per block.
        unsafe {
            let encoded = u64::from(*tiles.add(descriptor));
            let active = encoded / RING_DEGREE;
            if active < shape[5] || active >= shape[6] {
                return;
            }
            let local = encoded % RING_DEGREE;
            let start = *active_offsets.add(active as usize) + u64::from(*tiles.add(descriptor + 1));
            let end = start + u64::from(*tiles.add(descriptor + 2));
            let mut value = 0;
            let mut entry = start + lane as u64;
            while entry < end {
                value = gl_add(
                    value,
                    dec_original_term(
                        *entries.add(entry as usize),
                        local,
                        component,
                        pattern_offsets,
                        pattern_locals,
                        pattern_coefficients,
                        chi,
                    ),
                );
                entry += DEC_FORM_REDUCTION_THREADS as u64;
            }
            let sum = dec_reduce(&raw mut PARTIALS as *mut u64, lane, value);
            if lane == 0 {
                *tile_partials.add(group as usize) = sum;
            }
        }
    }

    /// The ring form lane of each very long list: the sum of its tiles.
    #[kernel]
    pub fn dec_reduce_parallel_original_form_tiles(
        forms: *mut u64,
        lists: *const u32,
        tile_offsets: *const u32,
        tile_partials: *const u64,
        shape: [u64; 9],
    ) {
        static mut PARTIALS: SharedArray<u64, DEC_FORM_REDUCTION_THREADS> = SharedArray::UNINIT;
        let lane = thread::threadIdx_x() as usize;
        let group = u64::from(thread::blockIdx_x());
        let component = group % 2;
        let list = (group / 2) as usize;
        // SAFETY: the host gives the tile lists and their partials.
        unsafe {
            let encoded = u64::from(*lists.add(list));
            let active = encoded / RING_DEGREE;
            if active < shape[5] || active >= shape[6] {
                return;
            }
            let start = u64::from(*tile_offsets.add(list));
            let end = u64::from(*tile_offsets.add(list + 1));
            let mut value = 0;
            let mut tile = start + lane as u64;
            while tile < end {
                value = gl_add(value, gl_from_word(*tile_partials.add((2 * tile + component) as usize)));
                tile += DEC_FORM_REDUCTION_THREADS as u64;
            }
            let sum = dec_reduce(&raw mut PARTIALS as *mut u64, lane, value);
            if lane == 0 {
                let local = encoded % RING_DEGREE;
                *forms.add((((active - shape[5]) * 2 + component) * RING_DEGREE + local) as usize) = sum;
            }
        }
    }

    /// The weight `Σ_entries χ(row)·c` of each geometric span `< count`.
    #[kernel]
    pub fn dec_geometric_span_weights(
        offsets: *const u32,
        rows: *const u32,
        coefficients: *const u64,
        chi: *const u64,
        shape: [u64; 9],
        weights: *mut u64,
        count: u64,
    ) {
        let span = thread::index_1d().get();
        if span as u64 >= count {
            return;
        }
        let mut value = KX_ZERO;
        // SAFETY: the host gives the span layer, the row weights and
        // `count` span weights.
        unsafe {
            for entry in *offsets.add(span)..*offsets.add(span + 1) {
                let row = u64::from(*rows.add(entry as usize));
                if row >= shape[2] || row >= shape[3] {
                    continue;
                }
                let coefficient = gl_from_word(*coefficients.add(entry as usize));
                value.c0 = gl_add(value.c0, gl_mul(gl_from_word(*chi.add(2 * row as usize)), coefficient));
                value.c1 = gl_add(
                    value.c1,
                    gl_mul(gl_from_word(*chi.add(2 * row as usize + 1)), coefficient),
                );
            }
            store_k(weights, span as u64, value);
        }
    }

    /// Add the geometric spans that cover each lane `< count` of the active
    /// blocks `shape[5] ..` to its form.
    #[kernel]
    pub fn dec_add_geometric_span_forms(
        starts: *const u64,
        ranks: *const u32,
        shapes: *const u64,
        weights: *const u64,
        shape: [u64; 9],
        forms: *mut u64,
        active_blocks: *const u32,
        longest: u64,
        count: u64,
    ) {
        let index = thread::index_1d().get() as u64;
        if index >= count {
            return;
        }
        let active = shape[5] + index / RING_DEGREE;
        let local = index % RING_DEGREE;
        // SAFETY: the host gives the span layer, the span weights and the
        // forms of `count / RING_DEGREE` active blocks.
        unsafe {
            let column = (u64::from(*active_blocks.add(active as usize)) % shape[1]) * RING_DEGREE + local;
            let first = (column + 1).saturating_sub(longest);
            let end = dec_span_rank(starts, ranks, column + 1);
            let mut value = KX_ZERO;
            for span in dec_span_rank(starts, ranks, first)..end {
                let packed = *shapes.add(2 * span as usize);
                let start = packed & 0xffff_ffff;
                if column >= start + (packed >> 32) {
                    continue;
                }
                let power = dec_pow(gl_from_word(*shapes.add(2 * span as usize + 1)), column - start);
                value.c0 = gl_add(value.c0, gl_mul(gl_from_word(*weights.add(2 * span as usize)), power));
                value.c1 = gl_add(
                    value.c1,
                    gl_mul(gl_from_word(*weights.add(2 * span as usize + 1)), power),
                );
            }
            let output = ((active - shape[5]) * 2 * RING_DEGREE + local) as usize;
            *forms.add(output) = gl_add(gl_from_word(*forms.add(output)), value.c0);
            let imaginary = output + RING_DEGREE as usize;
            *forms.add(imaginary) = gl_add(gl_from_word(*forms.add(imaginary)), value.c1);
        }
    }

    /// The conjugate `bar` of each ring form, in place, at slot `index <
    /// count` (14 slots per form).
    #[kernel]
    pub fn dec_bar_ring_forms_in_place(forms: *mut u64, count: u64) {
        let index = thread::index_1d().get() as u64;
        if index >= count {
            return;
        }
        let slot = index % 14;
        let base = ((index / 14) * RING_DEGREE) as usize;
        // SAFETY: the host gives `count / 14` forms.
        unsafe {
            if slot == 13 {
                *forms.add(base + 27) = gl_sub(0, *forms.add(base + 27));
                return;
            }
            let low = slot as usize + 1;
            let reflected = 27 - low;
            let low_value = *forms.add(base + low);
            let low_high_value = *forms.add(base + 27 + low);
            let reflected_value = *forms.add(base + reflected);
            let reflected_high_value = *forms.add(base + 27 + reflected);
            *forms.add(base + reflected) = gl_sub(0, gl_add(low_value, low_high_value));
            *forms.add(base + 27 + reflected) = gl_sub(0, low_value);
            *forms.add(base + low) = gl_sub(0, gl_add(reflected_value, reflected_high_value));
            *forms.add(base + 27 + low) = gl_sub(0, reflected_value);
        }
    }
}
