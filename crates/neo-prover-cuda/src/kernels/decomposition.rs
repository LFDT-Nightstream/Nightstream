//! The kernels of `decomposition.metal`: the base-2 digit split of a Π_RLC
//! parent, and the ring products of the openings.

use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::{cuda_module, kernel, thread};

use crate::kernels::field::*;

/// Children of one split thread; child masks are planar: the positive
/// columns, then the negative columns, of each child.
pub(crate) const DEC_SPLIT_CHILDREN_PER_THREAD: usize = 14;
pub(crate) const DEC_CHUNK_COLUMNS: u64 = 512;

/// Add `magnitude·forms[form - term]` for each set `term` of `bits` to the
/// 128-bit sum `(lo, hi)`.
///
/// # Safety
///
/// `forms` holds `form - term` for each set bit.
#[inline(always)]
unsafe fn add_terms(mut bits: u64, forms: *const u64, form: u64, magnitude: u64, lo: &mut u64, hi: &mut u64) {
    while bits != 0 {
        let term = u64::from(bits.trailing_zeros());
        bits &= bits - 1;
        // SAFETY: the caller's form.
        let mut value = unsafe { *forms.add((form - term) as usize) };
        if magnitude != 1 {
            value = gl_mul(value, magnitude);
        }
        let next = lo.wrapping_add(value);
        *hi += u64::from(next < *lo);
        *lo = next;
    }
}

#[cuda_module]
pub mod decomposition_kernels {
    use super::*;

    /// The digit masks of children `14·(index / cols) ..` of column
    /// `index % cols`, for `index < count`. Shape: `[_, children, _, cols]`.
    /// A parent coefficient of `children` bits or more sets `status`.
    #[kernel]
    pub fn dec_split_base2_masks(
        parent: *const u64,
        shape: [u64; 4],
        masks: *mut u64,
        child_nonzero: *mut u32,
        status: *mut u32,
        count: u64,
    ) {
        let index = thread::index_1d().get() as u64;
        if index >= count {
            return;
        }
        let (child_count, cols) = (shape[1], shape[3]);
        let first_child = (index / cols) * DEC_SPLIT_CHILDREN_PER_THREAD as u64;
        let column = index % cols;
        let mut positive = [0u64; DEC_SPLIT_CHILDREN_PER_THREAD];
        let mut negative = [0u64; DEC_SPLIT_CHILDREN_PER_THREAD];
        // SAFETY: the host gives the parent, the planes of every child, one
        // flag per child and the status word.
        unsafe {
            for coefficient in 0..RING_DEGREE {
                let word = gl_from_word(*parent.add((coefficient * cols + column) as usize));
                let is_negative = word > (GOLDILOCKS_MODULUS - 1) / 2;
                let magnitude = if is_negative { GOLDILOCKS_MODULUS - word } else { word };
                if first_child == 0 && magnitude >> child_count != 0 {
                    DeviceAtomicU32::from_ptr(status).fetch_or(1, AtomicOrdering::Relaxed);
                }
                for local in 0..DEC_SPLIT_CHILDREN_PER_THREAD {
                    let child = first_child + local as u64;
                    if child < child_count && (magnitude >> child) & 1 != 0 {
                        if is_negative {
                            negative[local] |= 1 << coefficient;
                        } else {
                            positive[local] |= 1 << coefficient;
                        }
                    }
                }
            }
            for local in 0..DEC_SPLIT_CHILDREN_PER_THREAD {
                let child = first_child + local as u64;
                if child < child_count {
                    *masks.add((2 * child * cols + column) as usize) = positive[local];
                    *masks.add(((2 * child + 1) * cols + column) as usize) = negative[local];
                    if positive[local] | negative[local] != 0 {
                        DeviceAtomicU32::from_ptr(child_nonzero.add(child as usize))
                            .fetch_or(1, AtomicOrdering::Relaxed);
                    }
                }
            }
        }
    }

    /// The ring-product partial of masked child `index / (2·107·chunks)`,
    /// for `index < count`, over one chunk of active blocks. Shape:
    /// `[_, active, form rows, blocks, chunks, gate, magnitudes, first
    /// active, masked, dense]`.
    #[kernel]
    pub fn dec_sparse_ring_partials(
        forms: *const u64,
        masks: *const u64,
        shape: [u64; 10],
        partials: *mut u64,
        active_children: *const u32,
        active_blocks: *const u32,
        active_chunk_bases: *const u32,
        active_chunk_matrices: *const u32,
        matrix_active_offsets: *const u32,
        child_nonzero: *const u32,
        count: u64,
    ) {
        let index = thread::index_1d().get() as u64;
        if index >= count {
            return;
        }
        let (active_count, form_rows, blocks, chunk_count, magnitudes) =
            (shape[1], shape[2], shape[3], shape[4], shape[6]);
        let coefficient = index % RING_PRODUCT_COEFFICIENTS;
        let mut rest = index / RING_PRODUCT_COEFFICIENTS;
        let component = rest % 2;
        rest /= 2;
        let chunk = rest % chunk_count;
        let active_child = rest / chunk_count;
        if active_child >= active_count {
            return;
        }
        // SAFETY: the host gives the forms, masks and opening plan of the
        // chunk, and `count` partials.
        unsafe {
            let child = u64::from(*active_children.add(active_child as usize));
            if shape[5] != 0 && *child_nonzero.add(child as usize) == 0 {
                *partials.add(index as usize) = 0;
                return;
            }
            let matrix = u64::from(*active_chunk_matrices.add(chunk as usize));
            if 2 * matrix + component >= form_rows {
                return;
            }
            let start = u64::from(*active_chunk_bases.add(chunk as usize));
            let end = (start + DEC_CHUNK_COLUMNS).min(u64::from(*matrix_active_offsets.add(matrix as usize + 1)));
            let term_start = if coefficient >= RING_DEGREE {
                coefficient - (RING_DEGREE - 1)
            } else {
                0
            };
            let term_end = if coefficient < RING_DEGREE {
                coefficient
            } else {
                RING_DEGREE - 1
            };
            let valid = (!0u64 << term_start) & ((1u64 << (term_end + 1)) - 1);
            let (mut positive_lo, mut positive_hi, mut negative_lo, mut negative_hi) = (0u64, 0u64, 0u64, 0u64);
            for active in start..end {
                let block = u64::from(*active_blocks.add(active as usize)) % blocks;
                let mask_base = 2 * magnitudes * (child * blocks + block);
                let form = ((active - shape[7]) * 2 + component) * RING_DEGREE + coefficient;
                for magnitude in 1..=magnitudes {
                    let at = (mask_base + 2 * (magnitude - 1)) as usize;
                    add_terms(
                        *masks.add(at) & valid,
                        forms,
                        form,
                        magnitude,
                        &mut positive_lo,
                        &mut positive_hi,
                    );
                    add_terms(
                        *masks.add(at + 1) & valid,
                        forms,
                        form,
                        magnitude,
                        &mut negative_lo,
                        &mut negative_hi,
                    );
                }
            }
            *partials.add(index as usize) = gl_sub(
                gl_reduce_sum(positive_lo, positive_hi),
                gl_reduce_sum(negative_lo, negative_hi),
            );
        }
    }

    /// `dec_sparse_ring_partials` over running sources without norm terms,
    /// whose field values are stored source by source, block by block and
    /// lane by lane. Source `s` writes the partials of opening `shape[8] +
    /// s`.
    #[kernel]
    pub fn dec_dense_ring_partials(
        forms: *const u64,
        values: *const u64,
        shape: [u64; 10],
        partials: *mut u64,
        sources: *const u32,
        active_blocks: *const u32,
        active_chunk_bases: *const u32,
        active_chunk_matrices: *const u32,
        matrix_active_offsets: *const u32,
        count: u64,
    ) {
        let index = thread::index_1d().get() as u64;
        if index >= count {
            return;
        }
        let (form_rows, blocks, chunk_count) = (shape[2], shape[3], shape[4]);
        let coefficient = index % RING_PRODUCT_COEFFICIENTS;
        let mut rest = index / RING_PRODUCT_COEFFICIENTS;
        let component = rest % 2;
        rest /= 2;
        let chunk = rest % chunk_count;
        let dense = rest / chunk_count;
        if dense >= shape[9] {
            return;
        }
        // SAFETY: the host gives the forms, the dense values and opening
        // plan of the chunk, and the partials of every opening.
        unsafe {
            let source = u64::from(*sources.add(dense as usize));
            let matrix = u64::from(*active_chunk_matrices.add(chunk as usize));
            if 2 * matrix + component >= form_rows {
                return;
            }
            let start = u64::from(*active_chunk_bases.add(chunk as usize));
            let end = (start + DEC_CHUNK_COLUMNS).min(u64::from(*matrix_active_offsets.add(matrix as usize + 1)));
            let term_start = if coefficient >= RING_DEGREE {
                coefficient - (RING_DEGREE - 1)
            } else {
                0
            };
            let term_end = if coefficient < RING_DEGREE {
                coefficient
            } else {
                RING_DEGREE - 1
            };
            let (mut lo, mut hi) = (0u64, 0u64);
            for active in start..end {
                let block = u64::from(*active_blocks.add(active as usize)) % blocks;
                let lanes = (source * blocks + block) * RING_DEGREE;
                let form = ((active - shape[7]) * 2 + component) * RING_DEGREE + coefficient;
                for term in term_start..=term_end {
                    let value = gl_mul(*forms.add((form - term) as usize), *values.add((lanes + term) as usize));
                    let next = lo.wrapping_add(value);
                    hi += u64::from(next < lo);
                    lo = next;
                }
            }
            let partial = ((shape[8] + dense) * chunk_count + chunk) * 2 + component;
            *partials.add((partial * RING_PRODUCT_COEFFICIENTS + coefficient) as usize) = gl_reduce_sum(lo, hi);
        }
    }

    /// The sum over chunks of each opening's partials, for `index < count`.
    #[kernel]
    pub fn dec_sparse_ring_sum_chunks(
        partials: *const u64,
        shape: [u64; 10],
        sums: *mut u64,
        matrix_chunk_offsets: *const u32,
        count: u64,
    ) {
        let index = thread::index_1d().get() as u64;
        if index >= count {
            return;
        }
        let (form_rows, chunk_count) = (shape[2], shape[4]);
        let group = index / RING_PRODUCT_COEFFICIENTS;
        let coefficient = index % RING_PRODUCT_COEFFICIENTS;
        let child = group / form_rows;
        let form_row = group % form_rows;
        let matrix = (form_row / 2) as usize;
        let component = form_row % 2;
        let mut value = 0;
        // SAFETY: the host gives the chunk offsets and every partial.
        unsafe {
            for chunk in u64::from(*matrix_chunk_offsets.add(matrix))..u64::from(*matrix_chunk_offsets.add(matrix + 1))
            {
                let partial = ((child * chunk_count + chunk) * 2 + component) * RING_PRODUCT_COEFFICIENTS + coefficient;
                value = gl_add(value, *partials.add(partial as usize));
            }
            *sums.add(index as usize) = value;
        }
    }

    /// Reduce each product of `2·D - 1` coefficients modulo `Φ_81`.
    #[kernel]
    pub fn dec_ring_reduce_phi81(sums: *const u64, shape: [u64; 10], output: *mut u64) {
        let index = thread::index_1d().get() as u64;
        let groups = shape[1] * shape[2];
        let group = index / RING_DEGREE;
        let coefficient = index % RING_DEGREE;
        if group >= groups {
            return;
        }
        let base = group * RING_PRODUCT_COEFFICIENTS;
        // SAFETY: the host gives `groups` products of `2·D - 1` words.
        let sum = |at: u64| gl_from_word(unsafe { *sums.add((base + at) as usize) });
        let mut value = sum(coefficient);
        if coefficient <= 26 {
            value = gl_sub(value, sum(coefficient + 54));
            if coefficient <= 25 {
                value = gl_add(value, sum(coefficient + 81));
            }
        } else {
            value = gl_sub(value, sum(coefficient + 27));
        }
        // SAFETY: the host gives `groups·D` outputs.
        unsafe { *output.add(index as usize) = value };
    }
}
