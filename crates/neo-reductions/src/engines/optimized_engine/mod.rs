//! Optimized engine implementation for Π_CCS
//!
//! This module contains the optimized implementation with factored algebra
//! and precomputed terms for efficient sumcheck proving.

#![allow(non_snake_case)]

use neo_math::K;
use p3_field::PrimeCharacteristicRing;

// Common types and utility functions shared across engines
mod common;
mod sparse;
mod terminal_identities;

pub mod oracle;
pub mod prove;
pub mod verify;

// Re-export commonly used items
pub use common::Challenges;

/// Proof format variant for Π_CCS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PiCcsProofVariant {
    /// Split-NC proof with two sumchecks: FE-only + NC-only.
    SplitNcV1,
}

// Re-export core functions for building proofs and cross-checking
pub use common::{
    chi_ajtai_at_bool_point,

    chi_row_at_bool_point,
    // Public claimed sum for sumcheck
    claimed_initial_sum_from_inputs_with_k_mcs,

    dec_reduction_paper_exact,
    dec_reduction_paper_exact_with_commit_check,
    dec_reduction_paper_exact_with_sparse_cache,
    // Core equalities & helpers
    eq_points,
    // Q(X) and sums
    q_at_point_paper_exact,
    q_eval_at_ext_point_paper_exact,
    q_eval_at_ext_point_paper_exact_with_inputs,

    // Utilities
    recomposed_z_from_Z,

    rlc_reduction_optimized,
    rlc_reduction_optimized_with_commit_mix,
    // Paper-exact RLC/DEC
    rlc_reduction_paper_exact,
    rlc_reduction_paper_exact_with_commit_mix,
    sum_q_over_hypercube_paper_exact,
};
pub use terminal_identities::{
    rhs_terminal_identity_fe, rhs_terminal_identity_fe_with_k_mcs, rhs_terminal_identity_nc,
};

/// Proof structure for the Π_CCS protocol
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PiCcsProof {
    /// Proof format variant.
    pub variant: PiCcsProofVariant,

    /// Sumcheck rounds (each round is a vector of polynomial coefficients)
    pub sumcheck_rounds: Vec<Vec<K>>,

    /// Initial sum over the Boolean hypercube (optional, can be derived from round 0)
    pub sc_initial_sum: Option<K>,

    /// Sumcheck challenges (r' || α' from the sumcheck protocol)
    pub sumcheck_challenges: Vec<K>,

    /// NC-only sumcheck rounds (digit-range / norm-check).
    pub sumcheck_rounds_nc: Vec<Vec<K>>,

    /// Initial sum for the NC sumcheck (optional; typically 0).
    pub sc_initial_sum_nc: Option<K>,

    /// NC sumcheck challenges (s' || α'_nc from the sumcheck protocol)
    pub sumcheck_challenges_nc: Vec<K>,

    /// Public challenges (α, β, γ)
    pub challenges_public: Challenges,

    /// Final running sum after all sumcheck rounds
    pub sumcheck_final: K,

    /// Final running sum after all NC sumcheck rounds
    pub sumcheck_final_nc: K,

    /// Header digest for binding
    pub header_digest: Vec<u8>,

    /// Additional proof data (if needed)
    pub _extra: Option<Vec<u8>>,
}

impl PiCcsProof {
    /// Create a new proof
    pub fn new(sumcheck_rounds: Vec<Vec<K>>, sc_initial_sum: Option<K>) -> Self {
        Self {
            variant: PiCcsProofVariant::SplitNcV1,
            sumcheck_rounds,
            sc_initial_sum,
            sumcheck_challenges: Vec::new(),
            sumcheck_rounds_nc: Vec::new(),
            sc_initial_sum_nc: None,
            sumcheck_challenges_nc: Vec::new(),
            challenges_public: Challenges {
                alpha: Vec::new(),
                beta_a: Vec::new(),
                beta_r: Vec::new(),
                beta_m: Vec::new(),
                gamma: K::ZERO,
            },
            sumcheck_final: K::ZERO,
            sumcheck_final_nc: K::ZERO,
            header_digest: Vec::new(),
            _extra: None,
        }
    }
}

// Re-export optimized prove/verify entrypoints as the main interface
pub use prove::optimized_prove as pi_ccs_prove;
pub use verify::optimized_verify as pi_ccs_verify;

/// Wrapper for simple case (k=1, no ME inputs)
pub use prove::optimized_prove_simple as pi_ccs_prove_simple;

// Re-export the oracle for Route A integration
pub use oracle::OptimizedOracle as CcsOracle;
