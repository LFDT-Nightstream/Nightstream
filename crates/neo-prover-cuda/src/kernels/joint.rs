//! The kernels of `joint.metal`: the Π_CCS joint oracle's application
//! tables, table folds and round partials.

use cuda_device::{cuda_module, kernel, thread, SharedArray};

use crate::kernels::assignments::joint_assignment_value;
use crate::kernels::field::*;
use crate::kernels::nightstream::{
    compact_row_offset, COMPACT_DENSE_BLOCK_TAG, COMPACT_DENSE_INDEX_MASK, COMPACT_NEGATIVE_BLOCK_TAG,
    COMPACT_SINGLE_BLOCK_MASK, COMPACT_SINGLE_LOCAL_MASK, COMPACT_SINGLE_LOCAL_SHIFT,
};

pub(crate) const SUMCHECK_MAX_COEFFS: usize = 10;
pub(crate) const SUMCHECK_REDUCTION_THREADS: usize = 64;

/// The signed digit of `witness` at `bit` of `block`: `m` or `-m` for the
/// first magnitude `m` whose positive or negative mask holds the bit.
///
/// # Safety
///
/// `masks` holds the masks of `witness`.
#[inline(always)]
pub(crate) unsafe fn joint_mask_digit(
    masks: *const u64,
    blocks: u64,
    witness: u64,
    block: u64,
    bit: u64,
    magnitudes: u64,
) -> u64 {
    let base = 2 * magnitudes * (witness * blocks + block);
    for magnitude in 1..=magnitudes {
        let at = (base + 2 * (magnitude - 1)) as usize;
        // SAFETY: the caller's masks.
        unsafe {
            if *masks.add(at) & bit != 0 {
                return magnitude;
            }
            if *masks.add(at + 1) & bit != 0 {
                return gl_sub(0, magnitude);
            }
        }
    }
    0
}

/// # Safety
///
/// As `joint_mask_digit`.
#[inline(always)]
pub(crate) unsafe fn joint_mask_value(
    masks: *const u64,
    blocks: u64,
    witness: u64,
    column: u64,
    magnitudes: u64,
) -> u64 {
    // SAFETY: the caller's masks.
    unsafe {
        joint_mask_digit(
            masks,
            blocks,
            witness,
            column / RING_DEGREE,
            1 << (column % RING_DEGREE),
            magnitudes,
        )
    }
}

/// # Safety
///
/// `tables` holds `local_len` entries of each table.
#[inline(always)]
unsafe fn joint_load_table(
    tables: *const u64,
    table_len: u64,
    table: u64,
    index: u64,
    base_field: bool,
    row_offset: u64,
    local_len: u64,
) -> Kx {
    if index >= table_len || index < row_offset || index - row_offset >= local_len {
        return KX_ZERO;
    }
    let position = table * local_len + index - row_offset;
    // SAFETY: the position lies in the caller's tables.
    unsafe {
        if base_field {
            Kx {
                c0: gl_from_word(*tables.add(position as usize)),
                c1: 0,
            }
        } else {
            load_k(tables, position)
        }
    }
}

#[inline(always)]
fn joint_signed_root_k(root: i32) -> Kx {
    let c0 = if root < 0 {
        gl_sub(0, u64::from(root.unsigned_abs()))
    } else {
        root as u64
    };
    Kx { c0, c1: 0 }
}

/// # Safety
///
/// `tables` holds `chunks_per_round·256` values of round `round`.
#[inline(always)]
unsafe fn joint_equality_suffix(tables: *const u64, chunks_per_round: u64, round: u64, pair: u64) -> Kx {
    let mut value = KX_ONE;
    let base = round * chunks_per_round * 256;
    for chunk in 0..chunks_per_round {
        let index = (pair >> (8 * chunk)) & 255;
        // SAFETY: the caller's chunk tables.
        value = kx_mul(value, unsafe { load_k(tables, base + chunk * 256 + index) });
    }
    value
}

#[inline(always)]
fn joint_poly_mul_affine(polynomial: &mut [Kx; SUMCHECK_MAX_COEFFS], constant_term: Kx, slope: Kx, degree: &mut usize) {
    let mut previous = KX_ZERO;
    for coefficient in 0..=*degree + 1 {
        let old = polynomial[coefficient];
        polynomial[coefficient] = kx_add(kx_mul(constant_term, old), kx_mul(slope, previous));
        previous = old;
    }
    *degree += 1;
}

/// -1, 0 or 1 for a base-field digit, else 2.
#[inline(always)]
fn joint_trit(value: Kx) -> i32 {
    if value.c1 != 0 {
        2
    } else if value.c0 == 0 {
        0
    } else if value.c0 == 1 {
        1
    } else if value.c0 == GOLDILOCKS_MODULUS - 1 {
        -1
    } else {
        2
    }
}

#[cuda_module]
pub mod joint_kernels {
    use super::*;

    /// One matrix's application table of one fresh source over the rows of
    /// a matrix window. Shape: `[rows, blocks, n_eff, table_len, witness,
    /// output table, offset width, identity, magnitudes, geometric offset
    /// width, first row, output offset]`.
    #[kernel]
    pub fn joint_build_application_tables(
        row_offsets: *const u8,
        row_blocks: *const u32,
        dense_offsets: *const u32,
        dense_locals: *const u8,
        dense_coefficients: *const u64,
        geometric_row_offsets: *const u8,
        geometric_runs: *const u64,
        masks: *const u64,
        shape: [u64; 12],
        output: *mut u64,
        dense_row_blocks: *const [u32; 2],
    ) {
        let local_row = thread::index_1d().get() as u64;
        let (rows, blocks, n_eff, table_len, witness, output_table) =
            (shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]);
        let (offset_width, identity, magnitudes, geometric_offset_width) =
            (shape[6], shape[7] != 0, shape[8], shape[9]);
        let row = shape[10] + local_row;
        if local_row >= rows {
            return;
        }
        let mut value = 0;
        // SAFETY: the host gives the window's compact matrix, the masks of
        // `witness` and the output tables.
        unsafe {
            if row < n_eff {
                if identity {
                    value = joint_mask_value(masks, blocks, witness, row, magnitudes);
                } else {
                    let start = compact_row_offset(row_offsets, local_row, offset_width);
                    let end = compact_row_offset(row_offsets, local_row + 1, offset_width);
                    for entry in start..end {
                        let reference = *row_blocks.add(entry as usize);
                        if reference & COMPACT_DENSE_BLOCK_TAG == 0 {
                            let block = u64::from(reference & COMPACT_SINGLE_BLOCK_MASK);
                            let local =
                                u64::from((reference >> COMPACT_SINGLE_LOCAL_SHIFT) & COMPACT_SINGLE_LOCAL_MASK);
                            let input = joint_mask_digit(masks, blocks, witness, block, 1 << local, magnitudes);
                            value = if reference & COMPACT_NEGATIVE_BLOCK_TAG == 0 {
                                gl_add(value, input)
                            } else {
                                gl_sub(value, input)
                            };
                        } else {
                            let [block, dense] = *dense_row_blocks.add((reference & COMPACT_DENSE_INDEX_MASK) as usize);
                            let first = *dense_offsets.add(dense as usize);
                            let last = *dense_offsets.add(dense as usize + 1);
                            for coefficient in first..last {
                                let bit = 1u64 << *dense_locals.add(coefficient as usize);
                                let input = joint_mask_digit(masks, blocks, witness, u64::from(block), bit, magnitudes);
                                if input != 0 {
                                    value = gl_add(
                                        value,
                                        gl_mul(gl_from_word(*dense_coefficients.add(coefficient as usize)), input),
                                    );
                                }
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
                            let mut coefficient = gl_from_word(*geometric_runs.add(3 * run as usize + 1));
                            let ratio = gl_from_word(*geometric_runs.add(3 * run as usize + 2));
                            let mut block = column / RING_DEGREE;
                            let mut bit = 1u64 << (column % RING_DEGREE);
                            while column < run_end {
                                let input = joint_mask_digit(masks, blocks, witness, block, bit, magnitudes);
                                if input != 0 && coefficient != 0 {
                                    value = gl_add(value, gl_mul(coefficient, input));
                                }
                                coefficient = gl_mul(coefficient, ratio);
                                bit <<= 1;
                                if bit == 1 << RING_DEGREE {
                                    bit = 1;
                                    block += 1;
                                }
                                column += 1;
                            }
                        }
                    }
                }
            }
            *output.add((output_table * table_len + shape[11] + local_row) as usize) = value;
        }
    }

    /// `words[i] = 0` for `i < count`.
    #[kernel]
    pub fn joint_zero_words(words: *mut u64, count: u64) {
        let index = thread::index_1d().get();
        if index as u64 >= count {
            return;
        }
        // SAFETY: the host gives `count` words.
        unsafe { *words.add(index) = 0 };
    }

    /// Fold base-field tables of `shape[0]` entries (`shape[1]` tables) at
    /// the challenge into `K` tables of half the length.
    #[kernel]
    pub fn joint_fold_base_tables(tables: *const u64, challenge: [u64; 2], shape: [u64; 2], output: *mut u64) {
        let index = thread::index_1d().get() as u64;
        let (table_len, table_count) = (shape[0], shape[1]);
        let folded_len = table_len.div_ceil(2);
        if index >= table_count * folded_len {
            return;
        }
        let table = index / folded_len;
        let pair = index % folded_len;
        let input = (table * table_len + 2 * pair) as usize;
        // SAFETY: the host gives `table_count` tables and their folds.
        unsafe {
            let left = Kx {
                c0: gl_from_word(*tables.add(input)),
                c1: 0,
            };
            let right = if 2 * pair + 1 < table_len {
                Kx {
                    c0: gl_from_word(*tables.add(input + 1)),
                    c1: 0,
                }
            } else {
                KX_ZERO
            };
            let challenge = Kx {
                c0: gl_from_word(challenge[0]),
                c1: gl_from_word(challenge[1]),
            };
            store_k(output, index, kx_add(left, kx_mul(challenge, kx_sub(right, left))));
        }
    }

    /// `joint_fold_base_tables` over `K` tables.
    #[kernel]
    pub fn joint_fold_k_tables(tables: *const u64, challenge: [u64; 2], shape: [u64; 2], output: *mut u64) {
        let index = thread::index_1d().get() as u64;
        let (table_len, table_count) = (shape[0], shape[1]);
        let folded_len = table_len.div_ceil(2);
        if index >= table_count * folded_len {
            return;
        }
        let table = index / folded_len;
        let pair = index % folded_len;
        let input = table * table_len + 2 * pair;
        // SAFETY: as above.
        unsafe {
            let left = load_k(tables, input);
            let right = if 2 * pair + 1 < table_len {
                load_k(tables, input + 1)
            } else {
                KX_ZERO
            };
            let challenge = Kx {
                c0: gl_from_word(challenge[0]),
                c1: gl_from_word(challenge[1]),
            };
            store_k(output, index, kx_add(left, kx_mul(challenge, kx_sub(right, left))));
        }
    }

    /// `output[table·shape[1] + shape[2]] += weight·input[table]` for
    /// `table < shape[0]`, with the weight `(shape[3], shape[4])`.
    #[kernel]
    pub fn joint_accumulate_application(input: *const u64, shape: [u64; 5], output: *mut u64) {
        let table = thread::index_1d().get() as u64;
        if table >= shape[0] {
            return;
        }
        let destination = table * shape[1] + shape[2];
        let weight = Kx {
            c0: gl_from_word(shape[3]),
            c1: gl_from_word(shape[4]),
        };
        // SAFETY: the host gives `shape[0]` inputs and their destinations.
        unsafe {
            let value = kx_add(load_k(output, destination), kx_mul(load_k(input, table), weight));
            store_k(output, destination, value);
        }
    }

    /// One block's partial round polynomial over pairs `shape[27] ..` of the
    /// joint oracle: the application terms, the norm terms of the
    /// assignments, the equality weights and the carried table.
    #[kernel]
    pub fn joint_round_partials(
        application_tables: *const u64,
        assignments_or_masks: *const u64,
        common_tables: *const u64,
        shape: [u64; 31],
        weights: *const u64,
        term_headers: *const u64,
        term_variables: *const u64,
        partials: *mut u64,
        equality_chunks: *const u64,
        prior_equality_chunks: *const u64,
        assignment_sources: *const u64,
        assignment_values: *const u64,
    ) {
        static mut SHARED: SharedArray<u64, { 2 * SUMCHECK_REDUCTION_THREADS * SUMCHECK_MAX_COEFFS }> =
            SharedArray::UNINIT;
        let lane = thread::threadIdx_x() as usize;
        let group = u64::from(thread::blockIdx_x());
        let local_pair = group * SUMCHECK_REDUCTION_THREADS as u64 + lane as u64;
        let fresh_count = shape[1];
        let matrix_count = shape[2];
        let assignment_count = shape[3];
        let coefficient_count = shape[4] as usize;
        let term_count = shape[5];
        let base_round = shape[6] != 0;
        let blocks = shape[7];
        let active_len = shape[9];
        let application_len = shape[10];
        let assignment_len = shape[11];
        let has_prior = shape[12] != 0;
        let common_len = shape[13];
        let chunks_per_round = shape[14];
        let round = shape[15];
        let k = |at: usize| Kx {
            c0: gl_from_word(shape[at]),
            c1: gl_from_word(shape[at + 1]),
        };
        let (alpha_low_factor, alpha_slope_factor) = (k(16), k(18));
        let (prior_low_factor, prior_slope_factor) = (k(20), k(22));
        let range_base = shape[24];
        let f_vanishes_at_zero = shape[25] != 0;
        let encoding = shape[26];
        let pair = shape[27] + local_pair;
        let pair_count = shape[28];
        let application_row_offset = shape[29];
        let application_local_len = shape[30];
        let mut local = [KX_ZERO; SUMCHECK_MAX_COEFFS];

        let pairs = active_len.div_ceil(2);
        if local_pair < pair_count && pair < pairs {
            let low_index = 2 * pair;
            let high_index = low_index + 1;
            let mut inner = [KX_ZERO; SUMCHECK_MAX_COEFFS];
            // SAFETY: the host gives every table, weight and term that the
            // shape names.
            unsafe {
                let mut source = 0;
                while source < fresh_count && (low_index < application_len || !f_vanishes_at_zero) {
                    let source_weight = load_k(weights, source);
                    for term in 0..term_count {
                        let header = 3 * term as usize;
                        let mut polynomial = [KX_ZERO; SUMCHECK_MAX_COEFFS];
                        polynomial[0] = kx_mul(
                            Kx {
                                c0: gl_from_word(*term_headers.add(header)),
                                c1: 0,
                            },
                            source_weight,
                        );
                        let variable_start = *term_headers.add(header + 1);
                        let variable_count = *term_headers.add(header + 2);
                        let mut degree = 0;
                        for variable in 0..variable_count {
                            let variable_header = 2 * (variable_start + variable) as usize;
                            let matrix = *term_variables.add(variable_header);
                            let exponent = *term_variables.add(variable_header + 1);
                            let table = source * matrix_count + matrix;
                            let low = joint_load_table(
                                application_tables,
                                application_len,
                                table,
                                low_index,
                                base_round,
                                application_row_offset,
                                application_local_len,
                            );
                            let high = joint_load_table(
                                application_tables,
                                application_len,
                                table,
                                high_index,
                                base_round,
                                application_row_offset,
                                application_local_len,
                            );
                            let slope = kx_sub(high, low);
                            for _ in 0..exponent {
                                joint_poly_mul_affine(&mut polynomial, low, slope, &mut degree);
                            }
                        }
                        for coefficient in 0..=degree {
                            inner[coefficient] = kx_add(inner[coefficient], polynomial[coefficient]);
                        }
                    }
                    source += 1;
                }

                // In the base round with b = 2, each unequal digit pair adds
                // one of six fixed cubics times its weight. Three signed
                // weight sums give the same coefficients as the per-source
                // products below (Lean `PiCCSNormTritSums.finish_sums`).
                let trit_round = base_round && range_base == 2;
                let (mut acc_u, mut acc_v, mut acc_w) = (KX_ZERO, KX_ZERO, KX_ZERO);
                for source in 0..assignment_count {
                    let value = |index| {
                        joint_assignment_value(
                            assignments_or_masks,
                            assignment_sources,
                            assignment_values,
                            source,
                            index,
                            encoding,
                            blocks,
                            range_base - 1,
                            assignment_len,
                        )
                    };
                    let (low, high) = (value(low_index), value(high_index));
                    if (base_round && low.c0 == high.c0 && low.c1 == high.c1)
                        || (low.c0 | low.c1 | high.c0 | high.c1) == 0
                    {
                        continue;
                    }
                    if trit_round {
                        let (l, h) = (joint_trit(low), joint_trit(high));
                        if l != 2 && h != 2 {
                            let weight = load_k(weights, fresh_count + source);
                            if h == 0 {
                                acc_u = if l < 0 {
                                    kx_add(acc_u, weight)
                                } else {
                                    kx_sub(acc_u, weight)
                                };
                            } else if l == 0 {
                                acc_w = if h < 0 {
                                    kx_add(acc_w, weight)
                                } else {
                                    kx_sub(acc_w, weight)
                                };
                            } else {
                                acc_v = if h > 0 {
                                    kx_add(acc_v, weight)
                                } else {
                                    kx_sub(acc_v, weight)
                                };
                            }
                            continue;
                        }
                    }
                    let slope = kx_sub(high, low);
                    let mut polynomial = [KX_ZERO; SUMCHECK_MAX_COEFFS];
                    polynomial[0] = load_k(weights, fresh_count + source);
                    let mut degree = 0;
                    let bound = range_base as i32 - 1;
                    for root in -bound..=bound {
                        joint_poly_mul_affine(
                            &mut polynomial,
                            kx_sub(low, joint_signed_root_k(root)),
                            slope,
                            &mut degree,
                        );
                    }
                    for coefficient in 0..=degree {
                        inner[coefficient] = kx_add(inner[coefficient], polynomial[coefficient]);
                    }
                }
                if trit_round {
                    // (-1,0)=X^3-3X^2+2X, (-1,1)=8X^3-12X^2+4X, (0,-1)=X-X^3,
                    // and the negated mirrors: inner += (2u+4v+w)X +
                    // (-3u-12v)X^2 + (u+8v-w)X^3.
                    let two_u = kx_add(acc_u, acc_u);
                    let two_v = kx_add(acc_v, acc_v);
                    let four_v = kx_add(two_v, two_v);
                    let eight_v = kx_add(four_v, four_v);
                    inner[1] = kx_add(inner[1], kx_add(kx_add(two_u, four_v), acc_w));
                    inner[2] = kx_sub(inner[2], kx_add(kx_add(two_u, acc_u), kx_add(eight_v, four_v)));
                    inner[3] = kx_add(inner[3], kx_sub(kx_add(acc_u, eight_v), acc_w));
                }

                let equality_suffix = joint_equality_suffix(equality_chunks, chunks_per_round, round, pair);
                let eq_low = kx_mul(alpha_low_factor, equality_suffix);
                let eq_slope = kx_mul(alpha_slope_factor, equality_suffix);
                local[0] = kx_mul(eq_low, inner[0]);
                for coefficient in 1..coefficient_count {
                    local[coefficient] = kx_add(
                        kx_mul(eq_low, inner[coefficient]),
                        kx_mul(eq_slope, inner[coefficient - 1]),
                    );
                }

                if has_prior {
                    let prior_suffix = joint_equality_suffix(prior_equality_chunks, chunks_per_round, round, pair);
                    let prior_low = kx_mul(prior_low_factor, prior_suffix);
                    let prior_slope = kx_mul(prior_slope_factor, prior_suffix);
                    let carried_low = if low_index < common_len {
                        load_k(common_tables, low_index)
                    } else {
                        KX_ZERO
                    };
                    let carried_high = if high_index < common_len {
                        load_k(common_tables, high_index)
                    } else {
                        KX_ZERO
                    };
                    let carried_slope = kx_sub(carried_high, carried_low);
                    local[0] = kx_add(local[0], kx_mul(prior_low, carried_low));
                    local[1] = kx_add(
                        local[1],
                        kx_add(kx_mul(prior_low, carried_slope), kx_mul(prior_slope, carried_low)),
                    );
                    local[2] = kx_add(local[2], kx_mul(prior_slope, carried_slope));
                }
            }
        }

        let shared = |lane: usize, coefficient: usize| 2 * (lane * SUMCHECK_MAX_COEFFS + coefficient);
        for (coefficient, value) in local.iter().enumerate().take(coefficient_count) {
            // SAFETY: the index lies in the block's shared words.
            unsafe {
                SHARED[shared(lane, coefficient)] = value.c0;
                SHARED[shared(lane, coefficient) + 1] = value.c1;
            }
        }
        thread::sync_threads();
        let mut stride = SUMCHECK_REDUCTION_THREADS / 2;
        while stride > 0 {
            if lane < stride {
                for coefficient in 0..coefficient_count {
                    let (destination, source) = (shared(lane, coefficient), shared(lane + stride, coefficient));
                    // SAFETY: both indices lie in the shared words.
                    unsafe {
                        let sum = kx_add(
                            Kx {
                                c0: SHARED[destination],
                                c1: SHARED[destination + 1],
                            },
                            Kx {
                                c0: SHARED[source],
                                c1: SHARED[source + 1],
                            },
                        );
                        SHARED[destination] = sum.c0;
                        SHARED[destination + 1] = sum.c1;
                    }
                }
            }
            thread::sync_threads();
            stride >>= 1;
        }
        if lane == 0 {
            for coefficient in 0..coefficient_count {
                // SAFETY: the host gives one partial per block and
                // coefficient.
                unsafe {
                    store_k(
                        partials,
                        group * coefficient_count as u64 + coefficient as u64,
                        Kx {
                            c0: SHARED[shared(0, coefficient)],
                            c1: SHARED[shared(0, coefficient) + 1],
                        },
                    );
                }
            }
        }
    }
}
