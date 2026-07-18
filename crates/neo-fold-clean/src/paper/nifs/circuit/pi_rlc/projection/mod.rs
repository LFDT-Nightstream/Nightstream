//! Π_RLC projection schedule orchestration.
//!
//! **Owns:** the binding → shared evaluation → identity/padding lifecycle.
//! **Does not own:** fold-input construction, transcript rho sampling, or
//! arithmetic leaf implementations. **Emits constraints:** only through its
//! three phase children. **Authority boundary:** binding produces the sole beta
//! and quotient wires; shared evaluation and every identity consume those exact
//! wires without substitution.
//!
//! | Child | Responsibility |
//! | --- | --- |
//! | [`binding`] | Bind combined values/advice, SIS-compress, sample beta |
//! | [`shared`] | Build one beta ladder and evaluate all rho polynomials |
//! | [`identities`] | Enforce paper-public commitment/X/y_ring identities and separately classified adv/y_zcol extensions |
//! | [`super::padding`] | Enforce implementation-only inactive X and padded y tails as zero |

use crate::engine::r1cs_circuit::ring_action::PROJECTION_QUOTIENT_LEN;
use crate::engine::r1cs_circuit::transcript::TranscriptGadget;
use crate::engine::r1cs_circuit::{PiRlcYZcolBoundaryAudit, R1csBuilder, Var};
use crate::paper::reductions::pi_dec_circuit::DecInputWires;
use neo_ccs::LaneCommitments;
use neo_math::ring::D;

use super::super::Error;
use super::fold_wires::FoldWires;

mod binding;
mod identities;
mod shared;

pub(super) struct Outputs {
    pub(super) beta: [Var; 2],
    pub(super) commitment_q: Vec<[Var; PROJECTION_QUOTIENT_LEN]>,
    pub(super) adv_q: Option<LaneCommitments<Vec<[Var; PROJECTION_QUOTIENT_LEN]>>>,
    pub(super) x_q: Vec<[Var; PROJECTION_QUOTIENT_LEN]>,
    pub(super) y_ring_q: Vec<[[Var; PROJECTION_QUOTIENT_LEN]; 2]>,
    pub(super) y_zcol_q: [[Var; PROJECTION_QUOTIENT_LEN]; 2],
}

pub(super) fn enforce(
    builder: &mut R1csBuilder,
    transcript: &mut TranscriptGadget,
    dec_wires: &DecInputWires,
    rho_wires: &[[Var; D]],
    folds: &FoldWires,
    kappa: usize,
    m_in: usize,
) -> Result<Outputs, Error> {
    let binding = binding::enforce(builder, transcript, dec_wires, folds, kappa)?;
    let shared = shared::enforce(builder, binding.beta, rho_wires);
    identities::enforce(builder, folds, &binding, &shared, kappa, m_in)?;
    builder.record_pi_rlc_y_zcol_boundary(PiRlcYZcolBoundaryAudit::new(
        [
            folds.y_zcol.combined_c0[..D]
                .iter()
                .map(|wire| wire.col())
                .collect(),
            folds.y_zcol.combined_c1[..D]
                .iter()
                .map(|wire| wire.col())
                .collect(),
        ],
        binding
            .y_zcol_q
            .each_ref()
            .map(|quotient| quotient.iter().map(|wire| wire.col()).collect()),
        binding.beta.map(|wire| wire.col()),
    ));
    Ok(Outputs {
        beta: binding.beta,
        commitment_q: binding.commitment_q,
        adv_q: binding.adv_q,
        x_q: binding.x_q,
        y_ring_q: binding.y_ring_q,
        y_zcol_q: binding.y_zcol_q,
    })
}
