//! Block modules for Q polynomial evaluation
//!
//! This module contains the three main blocks that constitute the Q polynomial:
//! - F block: CCS constraint polynomial evaluation
//! - NC block: Norm/decomposition constraints  
//! - Eval block: Evaluation ties

use neo_math::K;
use crate::pi_ccs::oracle::gate::PairGate;

/// Trait for univariate block contributions to Q polynomial
pub trait UnivariateBlock {
    /// Fold the block in-place with challenge r: g(X) -> (1-r)g(0)+r g(1)
    fn fold(&mut self, r: K);
}

/// Row-phase block evaluation
pub trait RowBlock: UnivariateBlock {
    /// Evaluate contribution at point x with given row gate
    fn eval_at(&self, x: K, row_gate: PairGate) -> K;
}

/// Ajtai-phase block evaluation  
pub trait AjtaiBlock: UnivariateBlock {
    /// Evaluate contribution at point x with given Ajtai gate and row scalar
    fn eval_at(&self, x: K, ajtai_gate: PairGate, row_scalar: K) -> K;
}

pub mod f_block;
pub mod nc_block;
pub mod eval_block;

pub use f_block::*;
pub use nc_block::*;
pub use eval_block::*;