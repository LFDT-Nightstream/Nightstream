//! Complete openings and row checks from original matrix windows. Global row
//! weights and columns stay fixed when local cache storage is released.

use neo_ccs::{SparsePoly, V1_1Evaluations};
use neo_math::{D, F, K};
use p3_field::PrimeCharacteristicRing;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

use super::{
    check_ccs_relation_zero_cached_with_blocks, eval_ring_scratch_real_z_blocks, openings::pad_opening,
    EqualityWeights, MatrixRows, MatrixWindow, RingEvalScratch, SuperneoCachedRelationError, SuperneoZBlocks,
};
use crate::PiCcsError;

/// The workspace bounds matrix-window construction; witness and opening-form
/// storage belong to the caller's separate live-allocation accounting.
pub fn eval_real_v1_1_openings_from_rows(
    source: &dyn MatrixRows,
    point: &[K],
    witnesses: &[SuperneoZBlocks],
    workspace_bytes: usize,
) -> Result<Vec<V1_1Evaluations<K>>, PiCcsError> {
    eval_real_v1_1_openings_from_rows_reusing(source, point, witnesses, workspace_bytes, Vec::new())
}

pub(crate) fn eval_real_v1_1_openings_from_rows_reusing(
    source: &dyn MatrixRows,
    point: &[K],
    witnesses: &[SuperneoZBlocks],
    workspace_bytes: usize,
    storage: Vec<K>,
) -> Result<Vec<V1_1Evaluations<K>>, PiCcsError> {
    let shape = source.shape();
    shape.validate(&(0..shape.rows))?;
    let variables = (usize::BITS
        - shape
            .rows
            .max(shape.columns)
            .saturating_sub(1)
            .leading_zeros()) as usize;
    if shape.columns == 0
        || shape.columns % D != 0
        || shape.matrices == 0
        || point.len() != variables
        || witnesses
            .iter()
            .any(|w| w.block_len() != shape.columns / D || !w.imag_all_zero)
    {
        return Err(PiCcsError::InvalidInput(
            "streamed opening point or witness shape".into(),
        ));
    }
    let weights = EqualityWeights::new(point);
    let mut result = witnesses
        .iter()
        .map(|witness| V1_1Evaluations {
            eval_k: pad_opening(witness, &weights).to_vec(),
            eval_a: vec![vec![K::ZERO; D]; shape.matrices],
        })
        .collect::<Vec<_>>();
    let active = witnesses
        .iter()
        .enumerate()
        .filter_map(|(index, witness)| (!witness.real_is_zero()).then_some(index))
        .collect::<Vec<_>>();
    if active.is_empty() {
        return Ok(result);
    }
    let mut scratch = RingEvalScratch::reuse(storage, shape.columns / D);
    let mut next = 0;
    while next < shape.rows {
        let window = MatrixWindow::load_next_with_payload(source, next..shape.rows, workspace_bytes, size_of::<K>())?;
        let range = window.rows();
        let row_weights: Vec<_> = range.clone().map(|row| weights.at(row)).collect();
        for (matrix, cache) in window.cache().matrix_caches().iter().enumerate() {
            // Each local cache contains original coefficients. Reuse the same
            // global row weights across its matrices, then release both owners.
            cache.accumulate_original_ring_form_with(range.len(), &mut scratch, |row| row_weights[row]);
            scratch.bar_active();
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            let values: Vec<_> = active
                .par_iter()
                .map(|&index| eval_ring_scratch_real_z_blocks(&scratch, &witnesses[index]))
                .collect();
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            let values: Vec<_> = active
                .iter()
                .map(|&index| eval_ring_scratch_real_z_blocks(&scratch, &witnesses[index]))
                .collect();
            for (&index, value) in active.iter().zip(values) {
                for (total, part) in result[index].eval_a[matrix].iter_mut().zip(value) {
                    *total += part;
                }
            }
            scratch.clear_active();
        }
        next = range.end;
    }
    Ok(result)
}

pub(crate) fn fill_weighted_rows_from_source(
    source: &dyn MatrixRows,
    projection: &[K],
    matrix_coefficients: &[K],
    output: &mut [K],
    workspace_bytes: usize,
) -> Result<(), PiCcsError> {
    let shape = source.shape();
    shape.validate(&(0..shape.rows))?;
    if output.len() != shape.rows || projection.len() != shape.columns || matrix_coefficients.len() != shape.matrices {
        return Err(PiCcsError::InvalidInput("streamed carried projection shape".into()));
    }
    let mut next = 0;
    while next < shape.rows {
        let window = MatrixWindow::load_next(source, next..shape.rows, workspace_bytes)?;
        let range = window.rows();
        window
            .cache()
            .fill_weighted_rows_from_projection(projection, matrix_coefficients, &mut output[range.clone()]);
        next = range.end;
    }
    Ok(())
}

pub fn first_unsatisfied_row_from_rows(
    source: &dyn MatrixRows,
    polynomial: &SparsePoly<F>,
    assignment: &SuperneoZBlocks,
    workspace_bytes: usize,
) -> Result<Option<usize>, PiCcsError> {
    let shape = source.shape();
    shape.validate(&(0..shape.rows))?;
    if assignment.block_len().checked_mul(D) != Some(shape.columns) || polynomial.arity() != shape.matrices {
        return Err(PiCcsError::InvalidInput("streamed relation shape".into()));
    }
    let row_bytes = shape
        .matrices
        .checked_mul(core::mem::size_of::<F>())
        .ok_or_else(|| PiCcsError::InvalidInput("streamed relation row size overflow".into()))?;
    let mut next = 0;
    while next < shape.rows {
        let window = MatrixWindow::load_next_with_payload(source, next..shape.rows, workspace_bytes, row_bytes)?;
        let range = window.rows();
        match check_ccs_relation_zero_cached_with_blocks(window.cache(), polynomial, assignment) {
            Ok(()) => {}
            Err(SuperneoCachedRelationError::UnsatisfiedRow { row }) => return Ok(Some(range.start + row)),
            Err(error) => return Err(PiCcsError::InvalidInput(error.to_string())),
        }
        next = range.end;
    }
    Ok(None)
}
