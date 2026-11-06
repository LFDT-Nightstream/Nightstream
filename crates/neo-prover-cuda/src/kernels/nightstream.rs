//! The kernels of `nightstream.metal`, and the helpers that the later
//! Metal files take from it.

use cuda_device::{cuda_module, kernel, thread};

use crate::kernels::field::{LIMB_MASK, RING_DEGREE};

pub(crate) const COMPACT_DENSE_BLOCK_TAG: u32 = 1 << 31;
pub(crate) const COMPACT_NEGATIVE_BLOCK_TAG: u32 = 1 << 30;
pub(crate) const COMPACT_SINGLE_LOCAL_SHIFT: u32 = 24;
pub(crate) const COMPACT_SINGLE_LOCAL_MASK: u32 = 0x3f;
pub(crate) const COMPACT_SINGLE_BLOCK_MASK: u32 = (1 << COMPACT_SINGLE_LOCAL_SHIFT) - 1;
pub(crate) const COMPACT_DENSE_INDEX_MASK: u32 = COMPACT_DENSE_BLOCK_TAG - 1;

/// A 128-bit sum as two words.
#[derive(Clone, Copy)]
pub(crate) struct WideProduct {
    pub(crate) lo: u64,
    pub(crate) hi: u64,
}

pub(crate) const WIDE_ZERO: WideProduct = WideProduct { lo: 0, hi: 0 };

/// `sum + value·scale` for a 32-bit `scale`.
#[inline(always)]
pub(crate) fn wide_sum_add_small(mut sum: WideProduct, value: u64, scale: u32) -> WideProduct {
    let low_product = (value & LIMB_MASK) * u64::from(scale);
    let high_product = (value >> 32) * u64::from(scale);
    let product_lo = low_product.wrapping_add(high_product << 32);
    let product_hi = (high_product >> 32) + u64::from(product_lo < low_product);
    let next = sum.lo.wrapping_add(product_lo);
    sum.hi = sum.hi.wrapping_add(product_hi + u64::from(next < sum.lo));
    sum.lo = next;
    sum
}

/// Add `value·|coefficient|` to the sum of the coefficient's sign.
#[inline(always)]
pub(crate) fn accumulate_small_signed(
    value: u64,
    coefficient: i32,
    positive: &mut WideProduct,
    negative: &mut WideProduct,
) {
    let magnitude = coefficient.unsigned_abs();
    if coefficient > 0 {
        *positive = wide_sum_add_small(*positive, value, magnitude);
    } else if coefficient < 0 {
        *negative = wide_sum_add_small(*negative, value, magnitude);
    }
}

/// Row `row` of a compact offset table of `width` bytes per row (3, or 4 as
/// `u32`, or 0 for no table).
///
/// # Safety
///
/// `offsets` holds entry `row`.
#[inline(always)]
pub(crate) unsafe fn compact_row_offset(offsets: *const u8, row: u64, width: u64) -> u64 {
    // SAFETY: the caller's entry.
    unsafe {
        if width == 3 {
            let base = 3 * row as usize;
            u64::from(*offsets.add(base))
                | (u64::from(*offsets.add(base + 1)) << 8)
                | (u64::from(*offsets.add(base + 2)) << 16)
        } else if width == 0 {
            0
        } else {
            u64::from(*(offsets as *const u32).add(row as usize))
        }
    }
}

/// # Safety
///
/// `rhos` holds `rho_base + 64` entries for every set mask bit.
#[inline(always)]
unsafe fn accumulate_signed_mask_rhos(
    rhos: *const i8,
    rho_base: u64,
    mut positive_mask: u64,
    mut negative_mask: u64,
    positive: &mut WideProduct,
    negative: &mut WideProduct,
) {
    while positive_mask != 0 {
        let inner = positive_mask.trailing_zeros() as u64;
        positive_mask &= positive_mask - 1;
        // SAFETY: the caller's rho row.
        let rho = i32::from(unsafe { *rhos.add((rho_base + inner) as usize) });
        accumulate_small_signed(1, rho, positive, negative);
    }
    while negative_mask != 0 {
        let inner = negative_mask.trailing_zeros() as u64;
        negative_mask &= negative_mask - 1;
        // SAFETY: as above.
        let rho = i32::from(unsafe { *rhos.add((rho_base + inner) as usize) });
        accumulate_small_signed(1, -rho, positive, negative);
    }
}

#[cuda_module]
pub mod nightstream_kernels {
    use super::*;
    use crate::kernels::field::*;
    /// Merkle leaf digests of `rows` rows of `width` words: Plonky3's
    /// overwrite sponge (`PaddingFreeSponge`). Each row starts from the zero
    /// state; each block of up to `POSEIDON_RATE` words overwrites the first
    /// lanes and is followed by one permutation.
    #[kernel]
    pub fn poseidon2_hash_rows_overwrite(
        fields: *const u64,
        output: *mut u64,
        round_constants: *const u64,
        width: u64,
        rows: u64,
    ) {
        let index = thread::index_1d().get();
        if index as u64 >= rows {
            return;
        }
        let width = width as usize;
        let offset = index * width;
        let mut state = [0u64; POSEIDON_WIDTH];
        let mut position = 0;
        while position < width {
            let take = if width - position < POSEIDON_RATE {
                width - position
            } else {
                POSEIDON_RATE
            };
            for (lane, word) in state.iter_mut().enumerate().take(take) {
                // SAFETY: the row holds `width` words from `offset`.
                *word = gl_from_word(unsafe { *fields.add(offset + position + lane) });
            }
            // SAFETY: the session uploads every round constant.
            unsafe { poseidon_permute(&mut state, round_constants) };
            position += POSEIDON_RATE;
        }
        for (lane, &word) in state.iter().enumerate().take(POSEIDON_DIGEST) {
            // SAFETY: the output holds `POSEIDON_DIGEST` words per row.
            unsafe { *output.add(POSEIDON_DIGEST * index + lane) = word };
        }
    }

    /// Stage `stage` of the tensor expansion of `challenges` into `table`:
    /// entries `index < 2^stage` split into `(1 - r)·v` and `r·v`.
    #[kernel]
    pub fn tensor_point_expand_k(challenges: *const u64, table: *mut u64, stage: u64) {
        let index = thread::index_1d().get() as u64;
        let step = 1u64 << stage;
        if index >= step {
            return;
        }
        // SAFETY: the host gives every challenge and `2^(stage + 1)` entries.
        unsafe {
            let value = if stage == 0 { KX_ONE } else { load_k(table, index) };
            let high = kx_mul(value, load_k(challenges, stage));
            let low = kx_sub(value, high);
            store_k(table, index, low);
            store_k(table, index + step, high);
        }
    }

    /// `output[row] += c·Σ_entries M[row, column]·qk[column]` for the rows
    /// `shape[6] ..` of one matrix window. Shape: `[rows, n_eff, n_pad,
    /// offset width, identity, geometric offset width, first row]`.
    #[kernel]
    pub fn fe_weighted_row_table(
        row_offsets: *const u8,
        row_blocks: *const u32,
        dense_offsets: *const u32,
        dense_locals: *const u8,
        dense_coefficients: *const u64,
        geometric_row_offsets: *const u8,
        geometric_runs: *const u64,
        qk: *const u64,
        matrix_coefficient: *const u64,
        shape: [u64; 7],
        output: *mut u64,
        dense_row_blocks: *const [u32; 2],
    ) {
        let local_row = thread::index_1d().get() as u64;
        let (rows, n_eff, n_pad, offset_width) = (shape[0], shape[1], shape[2], shape[3]);
        let identity = shape[4] != 0;
        let geometric_offset_width = shape[5];
        let row = shape[6] + local_row;
        if local_row >= rows || row >= n_pad {
            return;
        }
        let mut value = KX_ZERO;
        // SAFETY: the host gives the window's compact matrix, a `qk` entry
        // for every column and an output entry for every row.
        unsafe {
            if row < n_eff {
                if identity {
                    value = load_k(qk, row);
                } else {
                    let start = compact_row_offset(row_offsets, local_row, offset_width);
                    let end = compact_row_offset(row_offsets, local_row + 1, offset_width);
                    for entry in start..end {
                        let reference = *row_blocks.add(entry as usize);
                        if reference & COMPACT_DENSE_BLOCK_TAG == 0 {
                            let block = u64::from(reference & COMPACT_SINGLE_BLOCK_MASK);
                            let local =
                                u64::from((reference >> COMPACT_SINGLE_LOCAL_SHIFT) & COMPACT_SINGLE_LOCAL_MASK);
                            let input = load_k(qk, block * RING_DEGREE + local);
                            value = if reference & COMPACT_NEGATIVE_BLOCK_TAG == 0 {
                                kx_add(value, input)
                            } else {
                                kx_sub(value, input)
                            };
                        } else {
                            let [block, dense] = *dense_row_blocks.add((reference & COMPACT_DENSE_INDEX_MASK) as usize);
                            let first = *dense_offsets.add(dense as usize);
                            let last = *dense_offsets.add(dense as usize + 1);
                            for coefficient in first..last {
                                let column =
                                    u64::from(block) * RING_DEGREE + u64::from(*dense_locals.add(coefficient as usize));
                                let scalar = gl_from_word(*dense_coefficients.add(coefficient as usize));
                                let input = load_k(qk, column);
                                value.c0 = gl_add(value.c0, gl_mul(scalar, input.c0));
                                value.c1 = gl_add(value.c1, gl_mul(scalar, input.c1));
                            }
                        }
                    }
                    if geometric_offset_width != 0 {
                        let geometric_start =
                            compact_row_offset(geometric_row_offsets, local_row, geometric_offset_width);
                        let geometric_end =
                            compact_row_offset(geometric_row_offsets, local_row + 1, geometric_offset_width);
                        for run in geometric_start..geometric_end {
                            let packed = *geometric_runs.add(3 * run as usize);
                            let mut column = packed & 0xffff_ffff;
                            let run_end = column + (packed >> 32);
                            let mut scalar = gl_from_word(*geometric_runs.add(3 * run as usize + 1));
                            let ratio = gl_from_word(*geometric_runs.add(3 * run as usize + 2));
                            while column < run_end {
                                let input = load_k(qk, column);
                                value.c0 = gl_add(value.c0, gl_mul(scalar, input.c0));
                                value.c1 = gl_add(value.c1, gl_mul(scalar, input.c1));
                                scalar = gl_mul(scalar, ratio);
                                column += 1;
                            }
                        }
                    }
                }
            }
            let total = kx_add(load_k(output, row), kx_mul(load_k(matrix_coefficient, 0), value));
            store_k(output, row, total);
        }
    }

    /// The Π_RLC parent `Σ_i ρ_i·Z_i` at entry `index < count` (row-major,
    /// `columns` per row). Signed masks are planar: the positive columns,
    /// then the negative columns, of each input.
    #[kernel]
    pub fn rlc_witness_mix_signed_masks(
        rhos: *const i8,
        masks: *const u64,
        output: *mut u64,
        input_count: u64,
        columns: u64,
        count: u64,
    ) {
        let index = thread::index_1d().get() as u64;
        if index >= count {
            return;
        }
        let row = index / columns;
        let column = index % columns;
        let mut positive = WIDE_ZERO;
        let mut negative = WIDE_ZERO;
        for input in 0..input_count {
            let rho_base = input * RING_DEGREE * RING_DEGREE + row * RING_DEGREE;
            let mask_base = (2 * input * columns + column) as usize;
            // SAFETY: the host gives every input's rho and masks.
            unsafe {
                accumulate_signed_mask_rhos(
                    rhos,
                    rho_base,
                    *masks.add(mask_base),
                    *masks.add(mask_base + columns as usize),
                    &mut positive,
                    &mut negative,
                );
            }
        }
        // SAFETY: the output holds `count` words.
        unsafe {
            *output.add(index as usize) = gl_sub(
                gl_reduce_sum(positive.lo, positive.hi),
                gl_reduce_sum(negative.lo, negative.hi),
            );
        }
    }

    /// `output[c] = Σ_row partials[row, c]` for `c < coefficient_count`.
    #[kernel]
    pub fn sumcheck_reduce_partials(partials: *const u64, output: *mut u64, rows: u64, coefficient_count: u64) {
        let coefficient = thread::index_1d().get() as u64;
        if coefficient >= coefficient_count {
            return;
        }
        let mut total = KX_ZERO;
        for row in 0..rows {
            // SAFETY: the host gives `rows·coefficient_count` partials.
            total = kx_add(total, unsafe {
                load_k(partials, row * coefficient_count + coefficient)
            });
        }
        // SAFETY: the output holds `coefficient_count` values.
        unsafe { store_k(output, coefficient, total) };
    }
}
