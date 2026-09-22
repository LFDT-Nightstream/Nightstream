//! Shared Π_RLC projection evaluation phase.
//!
//! **Owns:** the one beta-power ladder and the evaluations of every rho
//! polynomial at that beta. **Does not own:** transcript sampling, quotient
//! binding, or claim identities. **Emits constraints:** beta-ladder products
//! and rho-evaluation product sums. **Authority boundary:** beta comes from the
//! binding transcript and the rho coefficients come from the NIFS transcript;
//! identities consume the exact values returned here.
//!
//! | Stage child | Mathematical obligation | Arithmetic owner |
//! | --- | --- | --- |
//! | `projection_shared.beta_ladder` | `powers[j] = beta^j` | `ring_action::enforce_beta_ladder` |
//! | `projection_shared.rho_evaluations` | `rho_i(beta) = sum_j rho_i[j] beta^j` | `ring_action::enforce_polynomial_evaluations_at_beta` |

use neo_math::ring::D;

use crate::engine::r1cs_circuit::field_ext::KVar;
use crate::engine::r1cs_circuit::ring_action::{
    enforce_beta_ladder, enforce_polynomial_evaluations_at_beta, PolynomialEvaluationsAtBeta,
};
use crate::engine::r1cs_circuit::{R1csBuilder, Var};
use crate::paper::reductions::pi_rlc_circuit::stage;

pub(super) struct SharedProjection {
    pub(super) powers: Vec<KVar>,
    pub(super) rho_evaluations: PolynomialEvaluationsAtBeta,
}

pub(super) fn enforce(builder: &mut R1csBuilder, beta: [Var; 2], rho_wires: &[[Var; D]]) -> SharedProjection {
    let shared_start = builder.rows();
    builder.begin_encoding_stage(stage::PROJECTION_SHARED);
    builder.begin_encoding_stage(stage::PROJECTION_SHARED_BETA_LADDER);
    let powers = enforce_beta_ladder(builder, KVar::new(beta[0], beta[1]), D);
    builder.begin_encoding_stage(stage::PROJECTION_SHARED_RHO_EVALUATIONS);
    let rho_evaluations = enforce_polynomial_evaluations_at_beta(builder, rho_wires, &powers);
    builder.record_row_family("nifs.pi_rlc.projection_shared", shared_start);
    SharedProjection {
        powers,
        rho_evaluations,
    }
}
