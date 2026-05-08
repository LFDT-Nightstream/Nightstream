//! Owns FE/NC terminal-identity gadgets for the RV32IM main relation circuit.
//!
//! These gadgets mirror the native optimized-engine RHS formulas over
//! authoritative claim fields. They do not own transcript binding, sumcheck
//! replay, or CE witness-opening checks.

mod fe;
mod helpers;
mod nc;

pub use fe::enforce_terminal_identity_fe;
#[allow(unused_imports)]
pub use fe::rhs_terminal_identity_fe;
use helpers::{compute_eval_sum, compute_f_prime};
pub use nc::enforce_terminal_identity_nc;
#[allow(unused_imports)]
pub use nc::rhs_terminal_identity_nc;

use crate::spartan_backend::SpartanF;
use bellpepper_core::{ConstraintSystem, SynthesisError};
use neo_ajtai::Commitment;
use neo_ccs::{CcsStructure, CeClaim};
use neo_math::{F, K};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;

use super::claim::CircuitCeClaim;
use super::k_field::{alloc_constant_k, enforce_k_eq, k_add, k_mul, KNum, KNumVar};
use super::terminal_common::{
    chi_table_var, dot_k_var_rows, eq_points, eval_sparse_poly_in_k, pow_k_var, range_product,
};

fn claim_has_zero_y_ring(claim: &CircuitCeClaim, t: usize) -> bool {
    claim
        .openings
        .y_ring_values
        .iter()
        .take(t)
        .all(|row| row.iter().all(|value| *value == K::ZERO))
}

fn claim_has_zero_y_zcol(claim: &CircuitCeClaim) -> bool {
    claim
        .norm_check
        .y_zcol_values
        .iter()
        .all(|value| *value == K::ZERO)
}

pub fn dummy_claim(
    y_ring: Vec<Vec<K>>,
    ct: Vec<K>,
    y_zcol: Vec<K>,
    r: Vec<K>,
    s_col: Vec<K>,
) -> CeClaim<Commitment, F, K> {
    CeClaim {
        c: Commitment::zeros(neo_math::D, 1),
        X: neo_ccs::Mat::zero(neo_math::D, 1, F::ZERO),
        r,
        s_col,
        y_ring,
        ct,
        aux_openings: Vec::new(),
        y_zcol,
        m_in: 1,
        fold_digest: [0; 32],
        c_step_coords: Vec::new(),
        u_offset: 0,
        u_len: 0,
    }
}
