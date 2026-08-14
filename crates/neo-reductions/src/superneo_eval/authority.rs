//! Exact CCS relation checks over the verifier-owned compact matrix cache.

use neo_ccs::SparsePoly;
use neo_math::{F, K};
use p3_field::PrimeCharacteristicRing;
use thiserror::Error;

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

use super::{SuperneoEvalCache, SuperneoZBlocks};

/// Failure of an exact row-wise CCS check over a compact relation.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum SuperneoCachedRelationError {
    #[error("compact relation cache has no matrices")]
    Empty,
    #[error("compact relation matrices do not share one shape")]
    MatrixShape,
    #[error("compact relation polynomial arity is {got}, expected {expected}")]
    PolynomialArity { expected: usize, got: usize },
    #[error("compact relation assignment width is {got}, expected {expected}")]
    AssignmentWidth { expected: usize, got: usize },
    #[error("compact relation assignment has a nonzero extension component")]
    NonRealAssignment,
    #[error("compact CCS relation fails at row {row}")]
    UnsatisfiedRow { row: usize },
}

/// Check `f(M₁z, …, Mₜz) = 0` on every row using the exact compact matrices
/// selected by the verifier. The assignment must contain real base-field
/// values represented in `K` and include the cache's complete padded width.
pub fn check_ccs_relation_zero_cached(
    cache: &SuperneoEvalCache,
    polynomial: &SparsePoly<F>,
    assignment: &[K],
) -> Result<(), SuperneoCachedRelationError> {
    let (rows, columns, matrix_count) = cache
        .relation_shape()
        .ok_or(SuperneoCachedRelationError::Empty)?;
    if cache
        .mats
        .iter()
        .any(|matrix| matrix.rows != rows || matrix.cols != columns)
    {
        return Err(SuperneoCachedRelationError::MatrixShape);
    }
    if polynomial.arity() != matrix_count {
        return Err(SuperneoCachedRelationError::PolynomialArity {
            expected: matrix_count,
            got: polynomial.arity(),
        });
    }
    if assignment.len() != columns {
        return Err(SuperneoCachedRelationError::AssignmentWidth {
            expected: columns,
            got: assignment.len(),
        });
    }
    let assignment = SuperneoZBlocks::from_z(assignment);
    if !assignment.imag_all_zero {
        return Err(SuperneoCachedRelationError::NonRealAssignment);
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    let matrix_rows = cache
        .mats
        .par_iter()
        .map(|matrix| {
            let mut values = vec![F::ZERO; rows];
            matrix.fill_row_dots_base_with_blocks(&mut values, &assignment);
            values
        })
        .collect::<Vec<_>>();
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    let matrix_rows = cache
        .mats
        .iter()
        .map(|matrix| {
            let mut values = vec![F::ZERO; rows];
            matrix.fill_row_dots_base_with_blocks(&mut values, &assignment);
            values
        })
        .collect::<Vec<_>>();

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    let failed_row = if rows >= 4_096 && rayon::current_num_threads() > 1 {
        (0..rows)
            .into_par_iter()
            .map_init(
                || vec![F::ZERO; matrix_count],
                |point, row| {
                    for (matrix, values) in matrix_rows.iter().enumerate() {
                        point[matrix] = values[row];
                    }
                    (row, polynomial.eval(point) != F::ZERO)
                },
            )
            .find_first(|(_, failed)| *failed)
            .map(|(row, _)| row)
    } else {
        first_unsatisfied_row(polynomial, &matrix_rows, rows)
    };
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    let failed_row = first_unsatisfied_row(polynomial, &matrix_rows, rows);

    failed_row.map_or(Ok(()), |row| Err(SuperneoCachedRelationError::UnsatisfiedRow { row }))
}

fn first_unsatisfied_row(polynomial: &SparsePoly<F>, matrix_rows: &[Vec<F>], rows: usize) -> Option<usize> {
    let mut point = vec![F::ZERO; matrix_rows.len()];
    (0..rows).find(|&row| {
        for (matrix, values) in matrix_rows.iter().enumerate() {
            point[matrix] = values[row];
        }
        polynomial.eval(&point) != F::ZERO
    })
}
