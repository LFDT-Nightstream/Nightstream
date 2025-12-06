//! Paper-exact reference implementations (CCS, RLC, DEC).
//!
//! This module mirrors the equations in docs/neo-paper §§4.4–4.6 as literally as possible.
//! All computations are done with direct loops over the Boolean hypercube and dense matrices.
//!
//! Goals
//! - Clarity and 1:1 parity with the paper formulas.
//! - No CSR, no half-table eq, no partial folding caches.
//! - Suitable as a cross-check oracle for tests and debugging.

#![allow(non_snake_case)]

pub mod oracle;
mod paper_exact;
pub mod prove;
pub mod verify;

pub use paper_exact::{
    // Step 3 outputs
    build_me_outputs_paper_exact,

    chi_ajtai_at_bool_point,

    chi_row_at_bool_point,
    // Public claimed sum for sumcheck
    claimed_initial_sum_from_inputs,

    dec_reduction_paper_exact,
    dec_reduction_paper_exact_with_commit_check,
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

    // Paper-exact RLC/DEC
    rlc_reduction_paper_exact,
    rlc_reduction_paper_exact_with_commit_mix,
    sum_q_over_hypercube_paper_exact,
};

pub use prove::paper_exact_prove;
pub use verify::paper_exact_verify;
