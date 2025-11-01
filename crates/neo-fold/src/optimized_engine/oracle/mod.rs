//! Modular oracle implementation for Generic CCS sum-check
//!
//! This module provides a phase-structured, block-modular implementation
//! of the Q polynomial oracle that closely mirrors Paper Section 4.4.
//!
//! ## Architecture
//!
//! The oracle is split into:
//! - **Blocks**: F (constraints), NC (norm constraints), and Eval (evaluation ties)
//! - **Engine**: Manages the row phase (rounds 0..ell_n-1) and Ajtai phase 
//!   (rounds ell_n..ell_n+ell_d-1) directly within the oracle implementation
//!
//! ## Paper Reference
//!
//! Section 4.4, Q polynomial:
//! ```text
//! Q(X) = eq(X,β)·[F + Σ γ^i·NC_i] + γ^k·Σ_{i≥2,j} γ^{i+(j-1)k-1}·Eval_{(i,j)}(X)
//! ```

use neo_math::K;

pub mod gate;
pub mod blocks;
pub mod engine;

#[cfg(test)]
pub mod tests;

// Re-export main types
pub use engine::GenericCcsOracle;

// NcState definition (moved from original oracle.rs)
/// NC state after row rounds: y_{i,1}(r') Ajtai partials & γ weights
pub struct NcState {
    pub y_partials: Vec<Vec<K>>,
    pub gamma_pows: Vec<K>,
    pub f_at_rprime: Option<K>,
}
