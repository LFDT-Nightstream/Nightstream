#![allow(non_snake_case)]

use neo_ccs::Mat;
use neo_math::{Fq, D, K};
use neo_params::NeoParams;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::error::PiCcsError;

/// NC digit rows for one SuperNeo packed witness.
///
/// F' witnesses are commonly bit-valued in the packed layout, so each live
/// logical column occupies only digit lane 0. Keeping those tables dense as
/// `[K; D]` rows creates huge zero-heavy allocations. `Lane0` keeps the same
/// folding semantics with one `K` per logical column and falls back to `Dense`
/// whenever a witness actually needs higher digit lanes.
#[derive(Debug)]
pub enum NcDigitTable {
    Lane0(Vec<K>),
    Dense(Vec<[K; D]>),
}

impl NcDigitTable {
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::Lane0(values) => values.len(),
            Self::Dense(rows) => rows.len(),
        }
    }

    #[inline]
    pub fn lane(&self, idx: usize, rho: usize) -> K {
        match self {
            Self::Lane0(values) => {
                if rho == 0 {
                    values[idx]
                } else {
                    K::ZERO
                }
            }
            Self::Dense(rows) => rows[idx][rho],
        }
    }

    #[inline]
    pub fn lane_real(&self, idx: usize, rho: usize) -> Fq {
        self.lane(idx, rho).real()
    }

    #[inline]
    pub fn row(&self, idx: usize) -> [K; D] {
        match self {
            Self::Lane0(values) => {
                let mut out = [K::ZERO; D];
                out[0] = values[idx];
                out
            }
            Self::Dense(rows) => rows[idx],
        }
    }

    #[inline]
    pub fn fold_inplace(&mut self, masks: &mut Vec<u64>, r: K) {
        match self {
            Self::Lane0(values) => fold_lane0_table_inplace(values, masks, r),
            Self::Dense(rows) => fold_dense_table_inplace(rows, masks, r),
        }
    }
}

pub fn build_nc_digit_table_compact<Ff>(
    params: &NeoParams,
    Z: &Mat<Ff>,
    expected_m: usize,
) -> Result<(NcDigitTable, Vec<u64>), PiCcsError>
where
    Ff: PrimeField64 + PrimeCharacteristicRing + Copy + Send + Sync,
    K: From<Ff>,
{
    crate::common::validate_superneo_witness_mat(Z, expected_m)?;

    let mut lane0 = vec![K::ZERO; expected_m];
    let mut masks = vec![0u64; expected_m];
    let active_cols = expected_m.div_ceil(D);
    let rows: [&[Ff]; D] = {
        let mut tmp: [&[Ff]; D] = [&[]; D];
        for (rho, slot) in tmp.iter_mut().enumerate() {
            *slot = Z.row(rho);
        }
        tmp
    };

    let needs_dense = AtomicBool::new(false);
    let process_block = |blk: usize, lane_chunk: &mut [K], mask_chunk: &mut [u64]| {
        if blk >= active_cols {
            return;
        }
        for (rho, (dst, mask_slot)) in lane_chunk.iter_mut().zip(mask_chunk.iter_mut()).enumerate() {
            let col = blk * D + rho;
            if col >= expected_m {
                break;
            }
            let raw = rows[rho].get(blk).copied().unwrap_or(Ff::ZERO);
            if raw == Ff::ZERO {
                continue;
            }
            if raw == Ff::ONE {
                *dst = K::ONE;
                *mask_slot = 1;
                continue;
            }
            match crate::common::decompose_balanced_fixed_d_digits_k(raw, params.b) {
                Ok(digits) if digits[1..].iter().all(|&digit| digit == K::ZERO) => {
                    *dst = digits[0];
                    *mask_slot = 1;
                }
                _ => {
                    needs_dense.store(true, Ordering::Relaxed);
                }
            }
        }
    };

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    {
        if rayon::current_thread_index().is_none() {
            lane0
                .par_chunks_mut(D)
                .zip(masks.par_chunks_mut(D))
                .enumerate()
                .for_each(|(blk, (lane_chunk, mask_chunk))| process_block(blk, lane_chunk, mask_chunk));
        } else {
            for (blk, (lane_chunk, mask_chunk)) in lane0.chunks_mut(D).zip(masks.chunks_mut(D)).enumerate() {
                process_block(blk, lane_chunk, mask_chunk);
            }
        }
    }
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    {
        for (blk, (lane_chunk, mask_chunk)) in lane0.chunks_mut(D).zip(masks.chunks_mut(D)).enumerate() {
            process_block(blk, lane_chunk, mask_chunk);
        }
    }

    if needs_dense.load(Ordering::Relaxed) {
        let (rows, masks) = crate::common::build_witness_nc_digit_table_with_masks(params, Z, expected_m)?;
        Ok((NcDigitTable::Dense(rows), masks))
    } else {
        Ok((NcDigitTable::Lane0(lane0), masks))
    }
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
