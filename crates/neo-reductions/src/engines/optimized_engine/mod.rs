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

pub mod oracle;
pub mod prove;
pub mod verify;

// Re-export commonly used items
pub use common::Challenges;

// Re-export core functions for building proofs and cross-checking
pub use common::{
    // Step 3 outputs
    build_me_outputs_paper_exact,

    chi_ajtai_at_bool_point,

    chi_row_at_bool_point,
    // Public claimed sum for sumcheck
    claimed_initial_sum_from_inputs,

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

    // Terminal identity (verifier RHS)
    rhs_terminal_identity_paper_exact,

    rlc_reduction_optimized,
    // Paper-exact RLC/DEC
    rlc_reduction_paper_exact,
    rlc_reduction_paper_exact_with_commit_mix,
    sum_q_over_hypercube_paper_exact,
};

/// Proof structure for the Π_CCS protocol
#[derive(Debug, Clone)]
pub struct PiCcsProof {
    /// Sumcheck rounds (each round is a vector of polynomial coefficients)
    pub sumcheck_rounds: Vec<Vec<K>>,

    /// Initial sum over the Boolean hypercube (optional, can be derived from round 0)
    pub sc_initial_sum: Option<K>,

    /// Sumcheck challenges (r' || α' from the sumcheck protocol)
    pub sumcheck_challenges: Vec<K>,

    /// Public challenges (α, β, γ)
    pub challenges_public: Challenges,

    /// Final running sum after all sumcheck rounds
    pub sumcheck_final: K,

    /// Header digest for binding
    pub header_digest: Vec<u8>,

    /// Additional proof data (if needed)
    pub _extra: Option<Vec<u8>>,
}

impl PiCcsProof {
    /// Create a new proof
    pub fn new(sumcheck_rounds: Vec<Vec<K>>, sc_initial_sum: Option<K>) -> Self {
        Self {
            sumcheck_rounds,
            sc_initial_sum,
            sumcheck_challenges: Vec::new(),
            challenges_public: Challenges {
                alpha: Vec::new(),
                beta_a: Vec::new(),
                beta_r: Vec::new(),
                gamma: K::ZERO,
            },
            sumcheck_final: K::ZERO,
            header_digest: Vec::new(),
            _extra: None,
        }
    }
}

// Re-export the paper-exact prove/verify functions as the main interface
pub use prove::optimized_prove as pi_ccs_prove;
pub use verify::paper_exact_verify as pi_ccs_verify;

/// Wrapper for simple case (k=1, no ME inputs)
pub use prove::optimized_prove_simple as pi_ccs_prove_simple;

// Route A: Split CCS prover for batched sum-check with Twist/Shout
pub use prove::{finalize_ccs_after_batch, prepare_ccs_for_batch, CcsBatchContext};

// Re-export the oracle for Route A integration
pub use oracle::OptimizedOracle as CcsOracle;
