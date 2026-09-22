//! Borrowed original matrix rows. Sources own coefficients; sinks own temporary storage.
//! Every row/matrix slot is explicit, including empty slots. No digest grants authority.

use std::{ops::ControlFlow, ops::Range};

use neo_ccs::GeometricRowRun;
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

use super::SuperneoEvalCache;
use crate::PiCcsError;

/// Global source dimensions. Columns include the complete padded carrier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatrixShape {
    pub rows: usize,
    pub columns: usize,
    pub matrices: usize,
}

impl MatrixShape {
    pub(super) fn validate(self, rows: &Range<usize>) -> Result<(), PiCcsError> {
        if self.rows == 0
            || self.columns == 0
            || !self.columns.is_multiple_of(D)
            || self.matrices == 0
            || self.columns / D > super::COMPACT_SINGLE_BLOCK_MASK as usize + 1
            || rows.start > rows.end
            || rows.end > self.rows
        {
            return Err(invalid("matrix source shape or requested row range"));
        }
        Ok(())
    }
}

/// Immutable original coefficients, visited in `(global row, matrix)` order.
/// Runs within a slot may overlap and need not be sorted. Each slot must end
/// once, even when empty. A source must stop immediately when its sink breaks.
/// Repeated visits to a range must produce the same matrix coefficients.
pub trait MatrixRows: Sync {
    fn shape(&self) -> MatrixShape;

    fn visit_rows(&self, rows: Range<usize>, sink: &mut dyn MatrixRowSink) -> Result<(), PiCcsError>;
}

pub trait MatrixRowSink {
    fn push_run(&mut self, row: usize, matrix: usize, run: GeometricRowRun<F>) -> Result<(), PiCcsError>;

    fn finish_matrix_row(&mut self, row: usize, matrix: usize) -> Result<ControlFlow<()>, PiCcsError>;
}

/// Borrow a cache without copying its rows. Plain seeded terms are emitted as
/// scalar runs. Pretransformed seeded columns are not original one-joint input.
pub struct CachedMatrixRows<'a> {
    cache: &'a SuperneoEvalCache,
    shape: MatrixShape,
}

impl<'a> CachedMatrixRows<'a> {
    pub fn new(cache: &'a SuperneoEvalCache) -> Result<Self, PiCcsError> {
        let (rows, columns, matrices) = cache
            .relation_shape()
            .ok_or_else(|| invalid("cached matrix source has inconsistent shape"))?;
        let shape = MatrixShape {
            rows,
            columns,
            matrices,
        };
        shape.validate(&(0..rows))?;
        if cache.matrix_caches().iter().any(|matrix| {
            matrix
                .compact_seeded_phi81_blocks()
                .iter()
                .any(|block| block.has_superneo_transformed_columns())
        }) {
            return Err(invalid("matrix source cannot use pretransformed seeded columns"));
        }
        Ok(Self { cache, shape })
    }
}

impl MatrixRows for CachedMatrixRows<'_> {
    fn shape(&self) -> MatrixShape {
        self.shape
    }

    fn visit_rows(&self, rows: Range<usize>, sink: &mut dyn MatrixRowSink) -> Result<(), PiCcsError> {
        self.shape.validate(&rows)?;
        for row in rows {
            for (index, matrix) in self.cache.matrix_caches().iter().enumerate() {
                if matrix.identity {
                    sink.push_run(row, index, GeometricRowRun::new(row, row, 1, F::ONE, F::ONE))?;
                } else {
                    let mut result = Ok(());
                    matrix.for_each_compact_explicit_row_coefficient(row, |block, local, coefficient| {
                        if result.is_ok() {
                            result = sink.push_run(
                                row,
                                index,
                                GeometricRowRun::new(row, block as usize * D + local as usize, 1, coefficient, F::ONE),
                            );
                        }
                    });
                    result?;
                    for &[packed, coefficient, ratio] in matrix.geometric_runs_for(row) {
                        sink.push_run(
                            row,
                            index,
                            GeometricRowRun::new(
                                row,
                                packed as u32 as usize,
                                (packed >> 32) as usize,
                                F::from_u64(coefficient),
                                F::from_u64(ratio),
                            ),
                        )?;
                    }
                    for block in matrix.compact_seeded_phi81_blocks() {
                        let mut result = Ok(());
                        block.for_each_row_term::<F, _>(row, |column, coefficient| {
                            if result.is_ok() {
                                result = sink.push_run(
                                    row,
                                    index,
                                    GeometricRowRun::new(row, column, 1, coefficient, F::ONE),
                                );
                            }
                        });
                        result?;
                    }
                }
                if sink.finish_matrix_row(row, index)?.is_break() {
                    return Ok(());
                }
            }
        }
        Ok(())
    }
}

pub(super) fn invalid(message: &str) -> PiCcsError {
    PiCcsError::InvalidInput(message.to_owned())
}
