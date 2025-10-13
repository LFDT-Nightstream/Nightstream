//! Binding and witness extraction utilities for NIVC

use crate::F;
use p3_field::PrimeCharacteristicRing;

/// Helper to extract y_step from a witness vector given indices
pub struct IndexExtractor {
    pub indices: Vec<usize>,
}

impl IndexExtractor {
    /// Extract values at the specified indices from the witness
    pub fn extract_y_step(&self, witness: &[F]) -> Vec<F> {
        self.indices.iter().map(|&i| witness.get(i).copied().unwrap_or(F::ZERO)).collect()
    }
}

