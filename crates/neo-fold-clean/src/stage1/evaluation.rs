//! Derive the native compact cache from the selected package's actual rows.

use super::Poseidon2HashChainV1Package;
use neo_math::F;
use neo_reductions::superneo_eval::{SuperneoEvalCache, SuperneoEvalCacheBuilder};
use nightstream_fprime::PackageError;
use p3_field::PrimeCharacteristicRing;

// Allocation hints from the full selected-package census.
// Package identity: [9705822157724451396, 520958727644325895, 9285622073986934000, 874020794279380938].
// These counts do not supply or validate any matrix coefficient.
const SELECTED_CACHE_CAPACITIES: [(usize, usize, usize); 14] = [
    (2_138_583, 2_086_257, 33_564_222),
    (4_650_801, 0, 0),
    (5_063_460, 4_494_661, 139_130_631),
    (5_483_392, 5_194_724, 117_185_066),
    (11_858_988, 11_629_792, 315_252_607),
    (34_909_868, 34_893_821, 1_535_199_455),
    (2_225_793, 2_225_793, 33_006_078),
    (1_726_758, 0, 0),
    (1_368_299, 1_354_119, 32_206_039),
    (2_062_032, 2_062_032, 30_971_178),
    (1_283_866, 1_270_589, 30_220_492),
    (1_979_667, 1_956_411, 31_110_714),
    (1_290_588, 1_277_551, 30_379_648),
    (0, 0, 0),
];

impl Poseidon2HashChainV1Package {
    /// Evaluate all active matrix rows and store their original coefficients
    /// in the native compact cache. Boolean rows beyond this prefix stay zero.
    pub fn build_superneo_cache(&self) -> Result<SuperneoEvalCache, PackageError> {
        let rows = self.package.row_count();
        let matrices = self.structure.t();
        if matrices != SELECTED_CACHE_CAPACITIES.len() {
            return Err(PackageError::Invalid("selected SuperNeo matrix count"));
        }
        let mut builder = SuperneoEvalCacheBuilder::new(rows, self.package.logical_column_count(), matrices)
            .map_err(|_| PackageError::Invalid("selected SuperNeo cache shape"))?;
        for (matrix, (blocks, dense, coefficients)) in SELECTED_CACHE_CAPACITIES.into_iter().enumerate() {
            builder
                .reserve_matrix(matrix, blocks, dense, coefficients)
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
