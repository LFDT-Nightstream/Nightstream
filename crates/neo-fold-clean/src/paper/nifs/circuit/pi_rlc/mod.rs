//! In-circuit Π_RLC lifecycle orchestration inside NIFS.V.
//!
//! **Owns:** output-count validation, rho sampling, Π_DEC parent allocation and
//! shape binding, fold-input preparation, consistency, and projection phases.
//! **Does not own:** Π_CCS verification, Π_DEC child verification, or any
//! arithmetic leaf equations.
//! **Emits constraints:** transcript sampling, shape equalities, shared-field
//! consistency, projection binding/evaluation, identities, and padding.
//! **Authority boundary:** only Π_CCS-derived output wires may be folded into
//! the allocated Π_DEC parent; every later phase consumes those same wires.
//!
//! | Child path | Mathematical obligation | Emits constraints? | Rust owner | Lean owner |
//! | --- | --- | --- | --- | --- |
//! | `challenge` | Bind the output digest and derive all rho values | yes | `alphabet_sampling` | `ChallengeWiringArtifact` proves static sharing only; terminal source binding is conditional and recursive source binding is open |
//! | `shape` | Allocate and pin parent/child dimensions | yes | this file | concrete refinement open |
//! | [`fold_wires`] | Build typed branch views | no | `fold_wires.rs` | claim parameters |
//! | [`consistency`] | Bind non-CE `s_col` and `fold_digest` sidecars across the fold | yes | `consistency.rs` | transcript/NC authority proof open |
//! | [`projection`] | Bind advice, share beta evaluation, and enforce paper-public plus extension identities | yes | `projection/` | `NifsPaper.PiRlc` and separate sidecars |
//! | [`padding`] | Canonically zero inactive X and padded y tails | yes | `padding.rs` | encoding/sidecar refinement open |

use neo_ccs::LaneCommitments;
use neo_math::ring::D;

use crate::engine::r1cs_circuit::alphabet_sampling::{enforce_pi_rlc_rhos_from_transcript, pi_rlc_challenge_stage};
use crate::engine::r1cs_circuit::ring_action::PROJECTION_QUOTIENT_LEN;
use crate::engine::r1cs_circuit::transcript::TranscriptGadget;
use crate::engine::r1cs_circuit::{Lc, R1csBuilder, Var};
use crate::paper::params::Params;
use crate::paper::reductions::pi_ccs_split_nc_circuit::{
    SplitNcPiCcsOutputWires, SplitNcPiCcsVConfig, SplitNcPiCcsVDerived,
};
use crate::paper::reductions::pi_dec_circuit::{alloc_dec_inputs, enforce_split_nc_d_pad_shape, DecInputWires};
use crate::paper::reductions::pi_rlc;
use crate::paper::reductions::pi_rlc_circuit::stage;
use crate::paper::relations::CeClaim;

use super::Error;

mod consistency;
mod fold_wires;
mod padding;
mod projection;

pub(super) struct Outputs {
    pub(super) dec_wires: DecInputWires,
    pub(super) projection_beta: [Var; 2],
    pub(super) projection_q_lanes: Vec<[Var; PROJECTION_QUOTIENT_LEN]>,
    pub(super) projection_adv_q_lanes: Option<LaneCommitments<Vec<[Var; PROJECTION_QUOTIENT_LEN]>>>,
    pub(super) projection_x_q_lanes: Vec<[Var; PROJECTION_QUOTIENT_LEN]>,
    pub(super) projection_y_ring_q_lanes: Vec<[[Var; PROJECTION_QUOTIENT_LEN]; 2]>,
    pub(super) projection_y_zcol_q_lanes: [[Var; PROJECTION_QUOTIENT_LEN]; 2],
}

pub(super) fn enforce(
    builder: &mut R1csBuilder,
    pp: &Params,
    cfg: &SplitNcPiCcsVConfig<'_>,
    transcript: &mut TranscriptGadget,
    ccs: &SplitNcPiCcsVDerived,
    combined: &CeClaim,
    children: &[CeClaim],
) -> Result<Outputs, Error> {
    let pi_rlc_start = builder.rows();
    let pi_rlc_first_column = builder.cols();
    builder.begin_encoding_stage(stage::ROOT);
    let d_pad = 1usize << cfg.ell_d;
    let k_total = ccs.outputs.len();
    if k_total == 0 {
        return Err(Error::Inner("Π_CCS.V emitted zero outputs".into()));
    }
    let kappa = ccs.outputs[0].c_kappa;
    let m_in = ccs.outputs[0].x_cols;

    // Definition 14 fixes this bound structurally at circuit-emission time.
    crate::paper::sampling::check_rlc_bound(pp, k_total, pp.T() as u128)
        .map_err(|error| Error::Inner(format!("Π_RLC bound: {error}")))?;

    let transcript_start = builder.rows();
    builder.begin_encoding_stage(pi_rlc_challenge_stage::CHALLENGE);
    builder.begin_encoding_stage(pi_rlc_challenge_stage::TRANSCRIPT);
    builder.begin_encoding_stage(pi_rlc_challenge_stage::BIND_OUTPUTS_DIGEST);
    transcript.append_fields(
        builder,
        pi_rlc::PI_RLC_INPUT_CLAIMS_DIGEST_LABEL,
        &ccs.output_claims_digest,
    );
    let rho_wires = enforce_pi_rlc_rhos_from_transcript(builder, transcript, k_total);
    builder.record_row_family("nifs.pi_rlc.transcript_rhos", transcript_start);

    let shape_start = builder.rows();
    builder.begin_encoding_stage(stage::SHAPE);
    builder.begin_encoding_stage(stage::SHAPE_ALLOCATE);
    let dec_wires = alloc_dec_inputs(builder, combined, children);
    builder.begin_encoding_stage(stage::SHAPE_OUTPUT_PARITY);
    enforce_output_shape_parity(builder, &ccs.outputs)?;
    builder.begin_encoding_stage(stage::SHAPE_PARENT);
    enforce_parent_shape(builder, &dec_wires, &ccs.outputs, kappa, m_in)?;
    builder.begin_encoding_stage(stage::SHAPE_D_PAD);
    enforce_split_nc_d_pad_shape(&dec_wires, cfg.structure.t(), d_pad)?;
    builder.record_row_family("nifs.pi_rlc.shape", shape_start);

    let folds_start = builder.rows();
    builder.begin_encoding_stage(stage::VERIFY);
    let folds = fold_wires::prepare(
        builder,
        &rho_wires,
        &ccs.outputs,
        &dec_wires,
        kappa,
        m_in,
        cfg.structure.t(),
        d_pad,
    )?;
    consistency::enforce(builder, &ccs.outputs, &dec_wires)?;
    builder.record_row_family("nifs.pi_rlc.linear_folds", folds_start);

    let projection = projection::enforce(builder, transcript, &dec_wires, &rho_wires, &folds, kappa, m_in)?;
    builder.record_row_family("nifs.pi_rlc", pi_rlc_start);
    builder.record_program_range("nifs.pi_rlc", pi_rlc_start, pi_rlc_first_column);

    Ok(Outputs {
        dec_wires,
        projection_beta: projection.beta,
        projection_q_lanes: projection.commitment_q,
        projection_adv_q_lanes: projection.adv_q,
        projection_x_q_lanes: projection.x_q,
        projection_y_ring_q_lanes: projection.y_ring_q,
        projection_y_zcol_q_lanes: projection.y_zcol_q,
    })
}

fn enforce_parent_shape(
    builder: &mut R1csBuilder,
    dec_wires: &DecInputWires,
    outputs: &[SplitNcPiCcsOutputWires],
    kappa: usize,
    m_in: usize,
) -> Result<(), Error> {
    let first = outputs
        .first()
        .ok_or_else(|| Error::Inner("Π_RLC parent shape requires at least one Π_CCS output".into()))?;
    let parent = &dec_wires.parent;
    let expected_c_len = D * kappa;
    if parent.c_data.len() < expected_c_len {
        return Err(Error::Inner(format!(
            "Π_RLC parent commitment lane count {} < D*kappa {expected_c_len}",
            parent.c_data.len()
        )));
    }
    let expected_x_len = D * m_in;
    if parent.x.len() < expected_x_len {
        return Err(Error::Inner(format!(
            "Π_RLC parent X lane count {} < D*m_in {expected_x_len}",
            parent.x.len()
        )));
    }
    enforce_var_eq(builder, parent.c_d_var, first.c_d_var);
    enforce_var_eq(builder, parent.c_kappa_var, first.c_kappa_var);
    enforce_var_eq(builder, parent.x_rows_var, first.x_rows_var);
    enforce_var_eq(builder, parent.x_cols_var, first.x_cols_var);
    enforce_var_eq(builder, parent.m_in_var, first.m_in_var);
    Ok(())
}

fn enforce_output_shape_parity(builder: &mut R1csBuilder, outputs: &[SplitNcPiCcsOutputWires]) -> Result<(), Error> {
    let first = outputs
        .first()
        .ok_or_else(|| Error::Inner("Π_RLC output shape parity requires at least one Π_CCS output".into()))?;
    for output in outputs.iter().skip(1) {
        enforce_var_eq(builder, output.c_d_var, first.c_d_var);
        enforce_var_eq(builder, output.c_kappa_var, first.c_kappa_var);
        enforce_var_eq(builder, output.x_rows_var, first.x_rows_var);
        enforce_var_eq(builder, output.x_cols_var, first.x_cols_var);
        enforce_var_eq(builder, output.m_in_var, first.m_in_var);
    }
    Ok(())
}

fn enforce_var_eq(builder: &mut R1csBuilder, left: Var, right: Var) {
    builder.enforce_eq(&Lc::from_var(left), &Lc::from_var(right));
}
