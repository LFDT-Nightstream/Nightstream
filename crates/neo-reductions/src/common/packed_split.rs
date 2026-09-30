//! Parallel balanced base-2 split into packed signed-unit digit planes.
//! It returns the serial split's digits for valid input; any entry outside
//! the DEC range returns `None`, and the caller's serial split reports it.

use neo_ccs::Mat;
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use rayon::prelude::*;

use super::balanced_divrem_i64_base2;

/// Witness columns per task; each task reads its columns row by row.
const CHUNK_COLUMNS: usize = 1 << 14;

type Planes = (Vec<Mat<F>>, Vec<bool>);

/// Split `z` into `k` balanced base-2 digit planes, one worker per column range.
pub(super) fn split_base2_packed(z: &Mat<F>, k: usize) -> Option<Planes> {
    let (rows, columns) = (z.rows(), z.cols());
    debug_assert!(rows <= u64::BITS as usize && k < i64::BITS as usize - 1);
    let data = z.as_slice();
    let bound = 1_u64 << k;
    let mut planes: Vec<(Vec<u64>, Vec<u64>)> = (0..k)
        .map(|_| (vec![0; columns], vec![0; columns]))
        .collect();
    let mut chunks: Vec<Vec<(&mut [u64], &mut [u64])>> = (0..columns.div_ceil(CHUNK_COLUMNS))
        .map(|_| Vec::with_capacity(k))
        .collect();
    for (positive, negative) in &mut planes {
        for (chunk, pair) in positive
            .chunks_mut(CHUNK_COLUMNS)
            .zip(negative.chunks_mut(CHUNK_COLUMNS))
            .enumerate()
        {
            chunks[chunk].push(pair);
        }
    }
    let nonzero = chunks
        .into_par_iter()
        .enumerate()
        .map(|(chunk, mut masks)| {
            let first = chunk * CHUNK_COLUMNS;
            let width = masks.first().map_or(0, |(positive, _)| positive.len());
            let mut nonzero = 0_u64;
            for row in 0..rows {
                for local in 0..width {
                    let entry = data[row * columns + first + local];
                    if entry == F::ZERO {
                        continue;
                    }
                    let mut value = balanced_value(entry.as_canonical_u64(), bound)?;
                    for (plane, (positive, negative)) in masks.iter_mut().enumerate() {
                        if value == 0 {
                            break;
                        }
                        let (digit, quotient) = balanced_divrem_i64_base2(value);
                        match digit {
                            1 => positive[local] |= 1 << row,
                            -1 => negative[local] |= 1 << row,
                            _ => {}
                        }
                        nonzero |= u64::from(digit != 0) << plane;
                        value = quotient;
                    }
                    if value != 0 {
                        return None;
                    }
                }
            }
            Some(nonzero)
        })
        .try_reduce(|| 0, |left, right| Some(left | right))?;
    let flags: Vec<bool> = (0..k).map(|plane| nonzero >> plane & 1 == 1).collect();
    let digits = planes
        .into_iter()
        .zip(&flags)
        .map(|((positive, negative), &used)| {
            if used {
                Mat::compact_signed_unit_from_column_masks(rows, columns, &positive, &negative)
                    .expect("binary split writes disjoint in-range row masks")
            } else {
                Mat::virtual_constant(rows, columns, F::ZERO)
            }
        })
        .collect();
    Some((digits, flags))
}

/// The balanced representative of a canonical field value in `(-bound, bound)`.
fn balanced_value(canonical: u64, bound: u64) -> Option<i64> {
    let negative_magnitude = F::ORDER_U64 - canonical;
    match (canonical < bound, negative_magnitude < bound) {
        (false, false) => None,
        (true, false) => Some(canonical as i64),
        (false, true) => Some(-(negative_magnitude as i64)),
        (true, true) if canonical <= negative_magnitude => Some(canonical as i64),
        (true, true) => Some(-(negative_magnitude as i64)),
    }
}

#[cfg(test)]
#[path = "../../tests/unit/packed_split.rs"]
mod tests;
