//! Original package rows and selected working-storage accounting. Production
//! consumers request windows from the unchanged relation.

use super::PreparedLifecycle;
use neo_ccs::GeometricRowRun;
use neo_math::{D, F};
use neo_reductions::superneo_eval::{MatrixRowSink, MatrixRows, MatrixShape};
use neo_reductions::PiCcsError;
use nightstream_fprime::{LoadedPerApplicationPackage, PackageError};
use p3_field::PrimeCharacteristicRing;
use std::ops::{ControlFlow, Range};

pub(crate) struct PackageRows<'a>(&'a LoadedPerApplicationPackage);

impl MatrixRows for PackageRows<'_> {
    fn shape(&self) -> MatrixShape {
        MatrixShape {
            rows: self.0.row_count(),
            columns: self.0.logical_column_count().div_ceil(D) * D,
            matrices: self.0.ccs_relation().matrix_sources().len(),
        }
    }

    fn visit_rows(&self, rows: Range<usize>, sink: &mut dyn MatrixRowSink) -> Result<(), PiCcsError> {
        let mut sink_error = None;
        let _ = self
            .0
            .visit_matrix_runs_until(rows, |ordinal, row| {
                for (matrix, terms) in row.into_iter().enumerate() {
                    for term in terms {
                        if let Err(error) = sink.push_run(
                            ordinal,
                            matrix,
                            GeometricRowRun::new(
                                ordinal,
                                term.column(),
                                term.column_count(),
                                F::from_u64(term.coefficient()),
                                F::from_u64(term.ratio()),
                            ),
                        ) {
                            sink_error = Some(error);
                            return Ok(ControlFlow::Break(()));
                        }
                    }
                    match sink.finish_matrix_row(ordinal, matrix) {
                        Ok(ControlFlow::Continue(())) => {}
                        Ok(ControlFlow::Break(())) => return Ok(ControlFlow::Break(())),
                        Err(error) => {
                            sink_error = Some(error);
                            return Ok(ControlFlow::Break(()));
                        }
                    }
                }
                Ok(ControlFlow::Continue(()))
            })
            .map_err(|error| PiCcsError::InvalidInput(error.to_string()))?;
        sink_error.map_or(Ok(()), Err)
    }
}

impl PreparedLifecycle {
    pub(crate) fn matrix_rows(&self) -> PackageRows<'_> {
        PackageRows(&self.package)
    }

    /// Reserve the selected profile's packed witness copies, carried table,
    /// and largest assignment-fold overlap before giving rows working space.
    /// This accounts for engine payload, not package/allocator/driver RSS.
    pub(crate) fn matrix_workspace_bytes(&self) -> Result<usize, PackageError> {
        let width = self.structure.m.div_ceil(D) as u128 * D as u128;
        let rows = self.structure.n as u128;
        let sources = nightstream_fprime::PI_CCS_V1_1_SOURCE_COUNT as u128;
        let blocks = width / D as u128;
        let masks = sources * blocks * (4 * size_of::<u64>() + size_of::<bool>()) as u128;
        let common = width.max(rows) * size_of::<neo_math::K>() as u128;
        // Signed-unit prefixes initially use two-byte pair codes. Conversion
        // to K can overlap one old code vector with its new field vector.
        let codes = width.div_ceil(2) * size_of::<u16>() as u128;
        let values = (u16::MAX as u128 + 1) * size_of::<neo_math::K>() as u128;
        let assignments = (sources + 1) * (codes + values);
        let matrix_projection = rows * size_of::<neo_math::K>() as u128;
        let reserved = masks + common + assignments.max(matrix_projection);
        // The owner requested 16 GB. The existing Metal buffer policy uses
        // the same decimal-byte ceiling; RSS is still checked separately.
        let remaining = 16_000_000_000u128
            .checked_sub(reserved)
            .ok_or(PackageError::Invalid("selected row workspace exceeds memory allowance"))?;
        usize::try_from(remaining).map_err(|_| PackageError::Invalid("selected row workspace exceeds address space"))
    }
}
