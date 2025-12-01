//! Optimized engine implementation for Π_CCS
//!
//! This module contains the optimized implementation with factored algebra
//! and precomputed terms for efficient sumcheck proving.

#![allow(non_snake_case)]

use neo_math::{K, KExtensions};
use p3_field::PrimeCharacteristicRing;

// Common types and utility functions shared across engines
mod common;

pub mod oracle;
pub mod prove;
pub mod verify;

// Re-export commonly used items
pub use common::Challenges;

// Re-export core functions for building proofs and cross-checking
pub use common::{
    // Core equalities & helpers
    eq_points, chi_row_at_bool_point, chi_ajtai_at_bool_point,

    // Q(X) and sums
    q_at_point_paper_exact,
    sum_q_over_hypercube_paper_exact,
    q_eval_at_ext_point_paper_exact,
    q_eval_at_ext_point_paper_exact_with_inputs,

    // Terminal identity (verifier RHS)
    rhs_terminal_identity_paper_exact,

    // Public claimed sum for sumcheck
    claimed_initial_sum_from_inputs,

    // Step 3 outputs
    build_me_outputs_paper_exact,

    // Utilities
    recomposed_z_from_Z,

    // Paper-exact RLC/DEC
    rlc_reduction_paper_exact,
    rlc_reduction_paper_exact_with_commit_mix,
    dec_reduction_paper_exact,
    dec_reduction_paper_exact_with_commit_check,
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

/// Naive Lagrange interpolation over K to monomial coefficients.
/// Returns coefficients c such that p(x) = Σ c[i] x^i matches (xs, ys).
pub(crate) fn interpolate_univariate(xs: &[K], ys: &[K]) -> Vec<K> {
    assert_eq!(xs.len(), ys.len());
    let n = xs.len();
    let mut coeffs = vec![K::ZERO; n];
    // Build Lagrange basis polynomials ℓ_i(x) and accumulate y_i·ℓ_i(x)
    for i in 0..n {
        // Numerator: prod_{j≠i} (x - x_j)
        let mut numer = vec![K::ZERO; n];
        numer[0] = K::ONE; // degree 0 poly = 1
        let mut cur_deg = 0usize;
        for j in 0..n {
            if i == j { continue; }
            // Multiply numer by (x - x_j)
            let xj = xs[j];
            let mut next = vec![K::ZERO; n];
            for d in 0..=cur_deg {
                // x * numer[d]
                next[d + 1] += numer[d];
                // -x_j * numer[d]
                next[d] += -xj * numer[d];
            }
            numer = next;
            cur_deg += 1;
        }
        // Denominator: prod_{j≠i} (x_i - x_j)
        let mut denom = K::ONE;
        for j in 0..n { if i != j { denom *= xs[i] - xs[j]; } }
        let scale = ys[i] * denom.inv();
        for d in 0..=cur_deg { coeffs[d] += scale * numer[d]; }
    }
    coeffs
}

// Re-export the paper-exact prove/verify functions as the main interface
pub use prove::optimized_prove as pi_ccs_prove;
pub use verify::paper_exact_verify as pi_ccs_verify;

/// Wrapper for simple case (k=1, no ME inputs)
pub use prove::optimized_prove_simple as pi_ccs_prove_simple;

// Re-export the paper-exact oracle
#[cfg(feature = "paper-exact")]
pub use oracle::OptimizedOracle as GenericCcsOracle;

