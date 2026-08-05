//! Π_RLC projection identities and their explicit padding glue.
//!
//! Owns: branch order, audit-role assignment, and the interleaving of each
//! identity with its zero-padding rows.
//!
//! Does not own: quotient allocation, beta evaluation, or low-level polynomial
//! equations.
//!
//! Emits constraints: yes, by invoking the arithmetic claim and padding leaves.
//!
//! Authority boundary: identity inputs must be transcript-bound quotient wires
//! plus the shared beta/rho evaluation.
//!
//! | Child path | Mathematical obligation | Emits constraints? | Rust owner | Lean owner |
//! |---|---|---|---|---|
//! | `identities.{commitment,x,y_ring}` | Paper-public one-point projection equations over the packed carrier | yes | `pi_rlc_circuit::{commitment,x,padded_k}` | `NifsPaper.PiRlc` (conditional) |
//! | `identities.adv` | Nebula product-commitment extension projection | yes | `pi_rlc_circuit::commitment` | separate Nebula refinement open |
//! | `padding.x` | Inactive X columns equal zero | yes | `pi_rlc_circuit/x.rs` | `Claims/X.lean` |
//! | `padding.y_ring` | Every padded y_ring tail equals zero | yes | `pi_rlc_circuit/padded_k.rs` | `Claims/Padding.lean` |

use crate::engine::r1cs_circuit::builder::{ProjectionIdentityRole, ProjectionNebulaCoordinate};
use crate::engine::r1cs_circuit::R1csBuilder;
use crate::paper::reductions::pi_rlc_circuit::{
    enforce_rlc_commitment_combination_projection_with_quotient_wires_and_stages,
    enforce_rlc_padded_k_projection_identities_with_quotient_wires_and_stages,
    enforce_rlc_x_projection_identities_with_quotient_wires_and_stages, stage,
};
use crate::paper::relations::superneo_public_x_cols;

use super::super::super::Error;
use super::super::fold_wires::FoldWires;
use super::super::padding;
use super::binding::BindingOutputs;
use super::shared::SharedProjection;

pub(super) fn enforce(
    builder: &mut R1csBuilder,
    folds: &FoldWires,
    binding: &BindingOutputs,
    shared: &SharedProjection,
    kappa: usize,
    m_in: usize,
) -> Result<(), Error> {
    let identities_start = builder.rows();
    let audit_start = builder.projection_identity_audits().len();

    builder.begin_encoding_stage(stage::IDENTITIES);
    for &phase in stage::IDENTITY_PHASE_NODES {
        builder.begin_encoding_stage(phase);
    }
    builder.begin_encoding_stage(stage::IDENTITIES_COMMITMENT);
    enforce_rlc_commitment_combination_projection_with_quotient_wires_and_stages(
        builder,
        &shared.powers,
        &shared.rho_evaluations,
        &folds.commitment,
        &binding.commitment_q,
        Some(stage::COMMITMENT_IDENTITY_STAGES),
    )?;

    builder.begin_encoding_stage(stage::IDENTITIES_ADV);
    if let (Some(adv), Some(quotients)) = (&folds.adv, &binding.adv_q) {
        for (coordinate, coordinate_q) in [
            (&adv.ops, &quotients.ops),
            (&adv.is, &quotients.is),
            (&adv.fs, &quotients.fs),
        ] {
            enforce_rlc_commitment_combination_projection_with_quotient_wires_and_stages(
                builder,
                &shared.powers,
                &shared.rho_evaluations,
                coordinate,
                coordinate_q,
                Some(stage::ADV_IDENTITY_STAGES),
            )?;
        }
    }

    builder.begin_encoding_stage(stage::IDENTITIES_X);
    enforce_rlc_x_projection_identities_with_quotient_wires_and_stages(
        builder,
        &shared.powers,
        &shared.rho_evaluations,
        &folds.x,
        &binding.x_q,
        Some(stage::X_IDENTITY_STAGES),
    )?;

    padding::enforce_x(builder, &folds.x)?;

    builder.begin_encoding_stage(stage::IDENTITIES_Y_RING);
    for (row, (wires, quotients)) in folds.y_ring.iter().zip(&binding.y_ring_q).enumerate() {
        builder.begin_encoding_stage(stage::IDENTITIES_Y_RING);
        enforce_rlc_padded_k_projection_identities_with_quotient_wires_and_stages(
            builder,
            &shared.powers,
            &shared.rho_evaluations,
            wires,
            &quotients[0],
            &quotients[1],
            Some([stage::Y_RING_IDENTITY_STAGES; 2]),
        )?;
        padding::enforce_y_ring(builder, wires, row)?;
    }

    let mut roles = Vec::with_capacity(builder.projection_identity_audits().len());
    roles.extend((0..kappa).map(|lane| ProjectionIdentityRole::CommitmentLane { lane }));
    if binding.adv_q.is_some() {
        for coordinate in [
            ProjectionNebulaCoordinate::Ops,
            ProjectionNebulaCoordinate::Is,
            ProjectionNebulaCoordinate::Fs,
        ] {
            roles.extend((0..kappa).map(|lane| ProjectionIdentityRole::NebulaCommitmentLane { coordinate, lane }));
        }
    }
    roles.extend((0..superneo_public_x_cols(m_in)).map(|column| ProjectionIdentityRole::ActiveXColumn { column }));
    for row in 0..folds.y_ring.len() {
        roles.extend((0..2).map(|limb| ProjectionIdentityRole::YRingLimb { row, limb }));
    }
    builder.assign_projection_identity_roles(audit_start, &roles);
    builder.record_row_family("nifs.pi_rlc.projection_identities", identities_start);
    Ok(())
}
