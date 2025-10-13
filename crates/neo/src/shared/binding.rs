//! Shared binding utilities for extracting step outputs
//!
//! These extractors define how step relations expose their outputs for Nova folding.

use crate::F;

/// Trait for extracting y_step values from step computations
/// 
/// This allows different step relations to define how their outputs
/// should be extracted for Nova folding, avoiding placeholder values.
pub trait StepOutputExtractor {
    /// Extract the compact output values (y_step) from a step witness
    /// These values represent what the step "produces" for Nova folding
    fn extract_y_step(&self, step_witness: &[F]) -> Vec<F>;
}

/// Simple extractor that takes the last N elements as y_step
pub struct LastNExtractor {
    pub n: usize,
}

impl StepOutputExtractor for LastNExtractor {
    fn extract_y_step(&self, step_witness: &[F]) -> Vec<F> {
        if step_witness.len() >= self.n {
            step_witness[step_witness.len() - self.n..].to_vec()
        } else {
            step_witness.to_vec()
        }
    }
}

/// Extractor that takes specific indices from the witness
pub struct IndexExtractor {
    pub indices: Vec<usize>,
}

impl StepOutputExtractor for IndexExtractor {
    fn extract_y_step(&self, step_witness: &[F]) -> Vec<F> {
        self.indices
            .iter()
            .filter_map(|&i| step_witness.get(i).copied())
            .collect()
    }
}

