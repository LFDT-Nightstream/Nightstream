//! Derive the native compact cache from the selected package's actual rows.

use super::Poseidon2HashChainV1Package;
use neo_math::{D, F};
use neo_reductions::superneo_eval::{SuperneoEvalCache, SuperneoEvalCacheBuilder};
use nightstream_fprime::{LogicalMatrixEntry, PackageError};
use p3_field::PrimeCharacteristicRing;

impl Poseidon2HashChainV1Package {
    /// Evaluate all active matrix rows and store their original coefficients
    /// in the native compact cache. Boolean rows beyond this prefix stay zero.
    pub fn build_superneo_cache(&self) -> Result<SuperneoEvalCache, PackageError> {
        let rows = self.package.row_count();
        let matrices = self.structure.t();
        let mut capacities = (0..matrices)
            .map(|_| MatrixCapacity::default())
            .collect::<Vec<_>>();
        self.package.visit_matrix_rows(0..rows, |_, row| {
            for (matrix, capacity) in capacities.iter_mut().enumerate() {
                capacity.add_row(
                    row.matrix(matrix)
                        .ok_or(PackageError::Invalid("selected SuperNeo matrix slot"))?,
                )?;
            }
            Ok(())
        })?;
        let mut builder = SuperneoEvalCacheBuilder::new(rows, self.package.logical_column_count(), matrices)
            .map_err(|_| PackageError::Invalid("selected SuperNeo cache shape"))?;
        for (matrix, capacity) in capacities.into_iter().enumerate() {
            builder
                .reserve_matrix(matrix, capacity.blocks, capacity.dense, capacity.coefficients)
                .map_err(|_| PackageError::Invalid("selected SuperNeo cache allocation"))?;
        }
        self.package.visit_matrix_rows(0..rows, |ordinal, row| {
            for matrix in 0..matrices {
                let entries = row
                    .matrix(matrix)
                    .ok_or(PackageError::Invalid("selected SuperNeo matrix slot"))?;
                builder
                    .push_row(
                        matrix,
                        ordinal,
                        entries
                            .iter()
                            .map(|entry| (entry.column(), F::from_u64(entry.coefficient()))),
                    )
                    .map_err(|_| PackageError::Invalid("selected SuperNeo cache row"))?;
            }
            Ok(())
        })?;
        builder
            .finish()
            .map_err(|_| PackageError::Invalid("selected SuperNeo cache row coverage"))
    }
}

#[derive(Default)]
struct MatrixCapacity {
    blocks: usize,
    dense: usize,
    coefficients: usize,
}

impl MatrixCapacity {
    fn add_row(&mut self, entries: &[LogicalMatrixEntry]) -> Result<(), PackageError> {
        let overflow = || PackageError::Invalid("selected SuperNeo cache count overflow");
        let mut start = 0;
        while start < entries.len() {
            let block = entries[start].column() / D;
            let mut end = start + 1;
            while end < entries.len() && entries[end].column() / D == block {
                end += 1;
            }
            self.blocks = self.blocks.checked_add(1).ok_or_else(overflow)?;
            let coefficient = F::from_u64(entries[start].coefficient());
            if end != start + 1 || (coefficient != F::ONE && coefficient != -F::ONE) {
                self.dense = self.dense.checked_add(1).ok_or_else(overflow)?;
                self.coefficients = self
                    .coefficients
                    .checked_add(end - start)
                    .ok_or_else(overflow)?;
            }
            start = end;
        }
        Ok(())
    }
}
