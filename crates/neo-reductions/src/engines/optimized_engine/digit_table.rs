#![allow(non_snake_case)]

use neo_ccs::Mat;
use neo_math::{Fq, D, K};
use neo_params::NeoParams;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::error::PiCcsError;

/// NC coefficient rows for one SuperNeo packed witness.
///
/// The unfolded table is always *diagonal*: logical column `col`'s single
/// live lane is `col % D` (that is where the SuperNeo packing places the
/// digit), so one `K` per column suffices. Folding merges pairs of rows;
/// after `k` folds, row `idx` covers `width = 2^k` original columns whose
/// lanes form one contiguous window `(idx·width .. idx·width+width) % D`.
/// The two operand windows of a merge stay lane-disjoint while
/// `2·width <= D`, so the fold is an in-place slot transform on a flat
/// vector and no dense rows exist at all until `width` would exceed the
/// disjointness bound.
///
/// - `Lane0`: every live column sits in ring lane 0 (`col % D == 0`); kept
///   as its own variant because `lane(idx, rho)` never needs the modulus.
/// - `Strided`: compact windowed rows. `width == 1` is the unfolded
///   diagonal table; `values[idx·width + j]` is the value at lane
///   `(idx·width + j) % D`. Invariant: `values.len() == len() · width`.
/// - `Dense`: full `[K; D]` rows, materialized only when a merge's lane
///   windows would collide (`2·width > D`), i.e. at ~1/64 of the
///   original row count.
#[derive(Debug)]
pub enum NcDigitTable {
    Zero {
        len: usize,
    },
    Lane0(Vec<K>),
    Strided {
        width: usize,
        values: Vec<K>,
    },
    Dense(Vec<[K; D]>),
    /// Placeholder when a device backend owns the column phase: the host
    /// never built (and must never read) the values. Any host access
    /// panics; `NcOracle::materialize_digit_tables` converts back to a
    /// built table if the backend declines.
    Deferred {
        len: usize,
    },
}

#[derive(Debug)]
pub enum NcDigitMasks {
    Zero { len: usize },
    Dense(Vec<u64>),
}

impl NcDigitMasks {
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::Zero { len } => *len,
            Self::Dense(values) => values.len(),
        }
    }

    #[inline]
    pub fn get(&self, index: usize) -> u64 {
        match self {
            Self::Zero { len } => {
                debug_assert!(index < *len);
                0
            }
            Self::Dense(values) => values[index],
        }
    }

    pub fn to_dense(&self) -> Vec<u64> {
        match self {
            Self::Zero { len } => vec![0; *len],
            Self::Dense(values) => values.clone(),
        }
    }

    fn dense_mut(&mut self) -> &mut Vec<u64> {
        match self {
            Self::Dense(values) => values,
            Self::Zero { .. } => panic!("nonzero NC table has zero masks"),
        }
    }
}

impl NcDigitTable {
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::Zero { len } => *len,
            Self::Lane0(values) => values.len(),
            Self::Strided { width, values } => values.len() / width,
            Self::Dense(rows) => rows.len(),
            Self::Deferred { len } => *len,
        }
    }

    #[inline]
    pub fn lane(&self, idx: usize, rho: usize) -> K {
        match self {
            Self::Zero { len } => {
                debug_assert!(idx < *len);
                let _ = rho;
                K::ZERO
            }
            Self::Lane0(values) => {
                if rho == 0 {
                    values[idx]
                } else {
                    K::ZERO
                }
            }
            Self::Strided { width, values } => {
                let start = (idx * width) % D;
                let j = (rho + D - start) % D;
                if j < *width {
                    values[idx * width + j]
                } else {
                    K::ZERO
                }
            }
            Self::Dense(rows) => rows[idx][rho],
            Self::Deferred { .. } => panic!("deferred NC digit table read on host"),
        }
    }

    #[inline]
    pub fn lane_real(&self, idx: usize, rho: usize) -> Fq {
        self.lane(idx, rho).real()
    }

    #[inline]
    pub fn row(&self, idx: usize) -> [K; D] {
        match self {
            Self::Zero { len } => {
                debug_assert!(idx < *len);
                [K::ZERO; D]
            }
            Self::Lane0(values) => {
                let mut out = [K::ZERO; D];
                out[0] = values[idx];
                out
            }
            Self::Strided { width, values } => {
                let mut out = [K::ZERO; D];
                for j in 0..*width {
                    let flat = idx * width + j;
                    out[flat % D] = values[flat];
                }
                out
            }
            Self::Dense(rows) => rows[idx],
            Self::Deferred { .. } => panic!("deferred NC digit table read on host"),
        }
    }

    #[inline]
    pub fn fold_inplace(&mut self, masks: &mut NcDigitMasks, r: K) {
        if let Self::Zero { len } = self {
            let NcDigitMasks::Zero { len: mask_len } = masks else {
                panic!("zero NC table has dense masks");
            };
            debug_assert_eq!(*len, *mask_len);
            *len = len.div_ceil(2);
            *mask_len = *len;
            return;
        }
        let masks = masks.dense_mut();
        match self {
            Self::Zero { .. } => unreachable!(),
            Self::Lane0(values) => fold_lane0_table_inplace(values, masks, r),
            Self::Strided { width, values } => {
                if 2 * *width <= D {
                    fold_strided_table_inplace(values, masks, *width, r);
                    *width *= 2;
                } else {
                    let folded = fold_strided_table_to_dense(values, masks, *width, r);
                    *self = Self::Dense(folded);
                }
            }
            Self::Dense(rows) => fold_dense_table_inplace(rows, masks, r),
            Self::Deferred { .. } => panic!("deferred NC digit table folded on host"),
        }
    }
}

pub fn build_nc_digit_table_compact<Ff>(
    params: &NeoParams,
    Z: &Mat<Ff>,
    expected_m: usize,
) -> Result<(NcDigitTable, NcDigitMasks), PiCcsError>
where
    Ff: PrimeField64 + PrimeCharacteristicRing + Copy + Send + Sync,
    K: From<Ff>,
{
    crate::common::validate_superneo_witness_mat(Z, expected_m)?;
    if params.b < 2 {
        return Err(PiCcsError::InvalidInput(format!(
            "NC witness table: invalid b={} (must be >= 2)",
            params.b
        )));
    }

    if Z.virtual_constant_value()
        .is_some_and(|value| *value == Ff::ZERO)
    {
        return Ok((
            NcDigitTable::Zero { len: expected_m },
            NcDigitMasks::Zero { len: expected_m },
        ));
    }

    let active_cols = expected_m.div_ceil(D);
    let all_zero =
        (0..D).all(|rho| (0..active_cols).all(|block| block * D + rho >= expected_m || Z[(rho, block)] == Ff::ZERO));
    if all_zero {
        return Ok((
            NcDigitTable::Zero { len: expected_m },
            NcDigitMasks::Zero { len: expected_m },
        ));
    }

    let mut values = vec![K::ZERO; expected_m];
    let mut masks = vec![0u64; expected_m];

    // The unfolded table is diagonal by construction: column `col`'s digit
    // lives in lane `col % D`, so one pass fills the compact value/mask
    // vectors. Track whether any live column sits outside lane 0 only to
    // pick the cheaper `Lane0` accessor when possible.
    let saw_nonzero_lane = AtomicBool::new(false);
    let process_block = |blk: usize, value_chunk: &mut [K], mask_chunk: &mut [u64]| {
        if blk >= active_cols {
            return;
        }
        for (rho, (dst, mask_slot)) in value_chunk
            .iter_mut()
            .zip(mask_chunk.iter_mut())
            .enumerate()
        {
            let col = blk * D + rho;
            if col >= expected_m {
                break;
            }
            let raw = Z[(rho, blk)];
            if raw == Ff::ZERO {
                continue;
            }
            *dst = K::from(raw);
            *mask_slot = 1u64 << rho;
            if rho != 0 {
                saw_nonzero_lane.store(true, Ordering::Relaxed);
            }
        }
    };

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    {
        if rayon::current_thread_index().is_none() {
            values
                .par_chunks_mut(D)
                .zip(masks.par_chunks_mut(D))
                .enumerate()
                .for_each(|(blk, (value_chunk, mask_chunk))| process_block(blk, value_chunk, mask_chunk));
        } else {
            for (blk, (value_chunk, mask_chunk)) in values.chunks_mut(D).zip(masks.chunks_mut(D)).enumerate() {
                process_block(blk, value_chunk, mask_chunk);
            }
        }
    }
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    {
        for (blk, (value_chunk, mask_chunk)) in values.chunks_mut(D).zip(masks.chunks_mut(D)).enumerate() {
            process_block(blk, value_chunk, mask_chunk);
        }
    }

    if saw_nonzero_lane.load(Ordering::Relaxed) {
        Ok((NcDigitTable::Strided { width: 1, values }, NcDigitMasks::Dense(masks)))
    } else {
        Ok((NcDigitTable::Lane0(values), NcDigitMasks::Dense(masks)))
    }
}

/// In-place fold of a strided table while merge windows are lane-disjoint
/// (`2·width <= D`). Row `i`'s output occupies exactly the flat range of its
/// two source rows (`i·2w == (2i)·w`), so the transform is per-slot: a live
/// lo-window lane sees `(lo, hi=0)` and a live hi-window lane `(lo=0, hi)`,
/// giving `v·(1-r)` and `v·r` respectively — exactly
/// `fold_dense_table_inplace`'s arithmetic on the implicit dense rows
/// (non-live lanes hold zero by invariant and stay zero under either form).
fn fold_strided_table_inplace(values: &mut Vec<K>, masks: &mut Vec<u64>, width: usize, r: K) {
    debug_assert!(!values.is_empty());
    debug_assert_eq!(values.len() % width, 0, "strided table length must be a width multiple");
    let rows = values.len() / width;
    debug_assert_eq!(rows, masks.len(), "NC digit table/mask length mismatch");
    let half = rows.div_ceil(2);
    let new_width = 2 * width;
    // A ragged tail (odd row count) needs one extra zero half-row.
    values.resize(half * new_width, K::ZERO);
    let one_minus_r = K::ONE - r;
    for i in 0..half {
        let base = 2 * i;
        let lo_mask = masks[base];
        let hi_mask = if base + 1 < rows { masks[base + 1] } else { 0 };
        masks[i] = lo_mask | hi_mask;
        let off = i * new_width;
        if lo_mask != 0 {
            for slot in &mut values[off..off + width] {
                if *slot != K::ZERO {
                    *slot *= one_minus_r;
                }
            }
        }
        if hi_mask != 0 {
            for slot in &mut values[off + width..off + new_width] {
                if *slot != K::ZERO {
                    *slot *= r;
                }
            }
        }
    }
    values.truncate(half * new_width);
    masks.truncate(half);
}

/// Terminal strided fold: merge windows would collide (`2·width > D`), so
/// materialize the half-size dense rows with the general per-lane formula —
/// identical to folding the implicit dense form with
/// `fold_dense_table_inplace`.
fn fold_strided_table_to_dense(values: &mut Vec<K>, masks: &mut Vec<u64>, width: usize, r: K) -> Vec<[K; D]> {
    debug_assert!(!values.is_empty());
    debug_assert_eq!(values.len() % width, 0, "strided table length must be a width multiple");
    let rows = values.len() / width;
    debug_assert_eq!(rows, masks.len(), "NC digit table/mask length mismatch");
    let lane_value = |row: usize, rho: usize| -> K {
        let start = (row * width) % D;
        let j = (rho + D - start) % D;
        if j < width {
            values[row * width + j]
        } else {
            K::ZERO
        }
    };
    let half = rows.div_ceil(2);
    let mut folded = Vec::with_capacity(half);
    for i in 0..half {
        let base = 2 * i;
        let lo_mask = masks[base];
        let hi_mask = if base + 1 < rows { masks[base + 1] } else { 0 };
        let active_mask = lo_mask | hi_mask;
        masks[i] = active_mask;
        let mut out = [K::ZERO; D];
        let mut lanes = active_mask;
        while lanes != 0 {
            let rho = lanes.trailing_zeros() as usize;
            lanes &= lanes - 1;
            let lo = if lo_mask & (1u64 << rho) != 0 {
                lane_value(base, rho)
            } else {
                K::ZERO
            };
            let hi = if hi_mask & (1u64 << rho) != 0 {
                lane_value(base + 1, rho)
            } else {
                K::ZERO
            };
            out[rho] = if hi == lo { lo } else { lo + (hi - lo) * r };
        }
        folded.push(out);
    }
    masks.truncate(half);
    values.clear();
    values.shrink_to_fit();
    folded
}

#[inline]
fn fold_lane0_table_inplace(values: &mut Vec<K>, masks: &mut Vec<u64>, r: K) {
    debug_assert!(!values.is_empty());
    debug_assert_eq!(values.len(), masks.len(), "NC digit table/mask length mismatch");
    let half = values.len().div_ceil(2);
    for i in 0..half {
        let base = 2 * i;
        let active_mask = masks[base] | if base + 1 < masks.len() { masks[base + 1] } else { 0 };
        let was_zero = masks[i] == 0;
        masks[i] = active_mask;
        if active_mask == 0 {
            if !was_zero {
                values[i] = K::ZERO;
            }
            continue;
        }

        let lo = values[base];
        let hi = if base + 1 < values.len() {
            values[base + 1]
        } else {
            K::ZERO
        };
        values[i] = if hi == lo { lo } else { lo + (hi - lo) * r };
    }
    values.truncate(half);
    masks.truncate(half);
}

#[inline]
fn fold_dense_table_inplace(table: &mut Vec<[K; D]>, masks: &mut Vec<u64>, r: K) {
    debug_assert!(!table.is_empty());
    debug_assert_eq!(table.len(), masks.len(), "NC digit table/mask length mismatch");
    let half = table.len().div_ceil(2);
    for i in 0..half {
        let base = 2 * i;
        let active_mask = masks[base] | if base + 1 < masks.len() { masks[base + 1] } else { 0 };
        let was_zero = masks[i] == 0;
        masks[i] = active_mask;
        if active_mask == 0 {
            if !was_zero {
                table[i] = [K::ZERO; D];
            }
            continue;
        }

        let lo_row = table[base];
        let hi_row = if base + 1 < table.len() {
            table[base + 1]
        } else {
            [K::ZERO; D]
        };
        if active_mask == 1 {
            let lo = lo_row[0];
            let hi = hi_row[0];
            let mut out = [K::ZERO; D];
            out[0] = if hi == lo { lo } else { lo + (hi - lo) * r };
            table[i] = out;
            continue;
        }
        let mut out = [K::ZERO; D];
        let mut lanes = active_mask;
        while lanes != 0 {
            let rho = lanes.trailing_zeros() as usize;
            lanes &= lanes - 1;
            let lo = lo_row[rho];
            let hi = hi_row[rho];
            out[rho] = if hi == lo { lo } else { lo + (hi - lo) * r };
        }
        table[i] = out;
    }
    table.truncate(half);
    masks.truncate(half);
}
