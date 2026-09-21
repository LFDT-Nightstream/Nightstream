//! Derive the native compact cache from the selected package's actual rows.

use super::PreparedLifecycle;
use neo_math::F;
use neo_reductions::superneo_eval::{SuperneoEvalCache, SuperneoEvalCacheBuilder};
use nightstream_fprime::PackageError;
use p3_field::PrimeCharacteristicRing;
use std::sync::Arc;

impl PreparedLifecycle {
    /// Evaluate all active matrix rows and store their original coefficients
    /// in the native compact cache. Boolean rows beyond this prefix stay zero.
    pub(crate) fn build_superneo_cache(&self) -> Result<&Arc<SuperneoEvalCache>, PackageError> {
        if let Some(cache) = self.cache.get() {
            return Ok(cache);
        }
        let rows = self.package.row_count();
        let matrices = self.structure.t();
        let mut builder = SuperneoEvalCacheBuilder::new(rows, self.package.logical_column_count(), matrices)
            .map_err(|_| PackageError::Invalid("selected SuperNeo cache shape"))?;
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
        let cache = builder
            .finish()
            .map_err(|_| PackageError::Invalid("selected SuperNeo cache row coverage"))?;
        let _ = self.cache.set(Arc::new(cache));
        Ok(self.cache.get().expect("cache was initialized"))
    }
}
