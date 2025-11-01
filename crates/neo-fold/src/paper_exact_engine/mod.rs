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

mod paper_exact;
pub mod oracle;

pub use paper_exact::{
    // Core equalities & helpers
    eq_points, chi_row_at_bool_point, chi_ajtai_at_bool_point,

    // Q(X) and sums
    q_at_point_paper_exact,
    sum_q_over_hypercube_paper_exact,
    q_eval_at_ext_point_paper_exact,

    // Terminal identity (verifier RHS)
    rhs_terminal_identity_paper_exact,

    // Step 3 outputs
    build_me_outputs_paper_exact,

    // Utilities
    recomposed_z_from_Z,

    // Paper-exact RLC/DEC
    rlc_reduction_paper_exact,
    dec_reduction_paper_exact,
};
