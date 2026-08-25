//! NIFS.V circuit composition: Π_CCS → Π_RLC → Π_DEC → point binding.
//!
//! **Owns:** the public circuit API, top-level verifier order, and the wires
//! surfaced to recursive F'. **Does not own:** child verifier algebra or
//! transcript internals. **Emits constraints:** by invoking each verifier and
//! finally equating the Π_DEC parent point with Π_CCS `r_prime`.
//! **Authority boundary:** Π_CCS-derived output wires are the sole Π_RLC input;
//! Π_DEC checks their parent cache, while the exact ordered paper-level
//! children are the outgoing Construction-2 accumulator. Incoming running
//! claims use the same strict Pi_DEC check before they enter Pi_CCS.
//!
//! | Child phase | Mathematical obligation | Emits constraints? | Rust owner | Lean owner |
//! |---|---|---|---|---|
//! | Pi_CCS | Derive output wires and `r_prime` from fresh/running claims | yes | `pi_ccs_circuit` | concrete bridge open |
//! | Pi_RLC | Fold outputs into a shape-bound parent and children | yes | [`pi_rlc`] | algebra model partial |
//! | Pi_DEC | Strictly recompose the parent from claimed radix children | yes | `pi_dec_circuit` | PiDEC bridge partial |
//! | point binding | Equate the Pi_DEC parent point with Pi_CCS `r_prime` | yes | this file | NIFS bridge open |

use neo_ccs::LaneCommitments;

use crate::engine::r1cs_circuit::field_ext::KVar;
use crate::engine::r1cs_circuit::ring_action::PROJECTION_QUOTIENT_LEN;
use crate::engine::r1cs_circuit::transcript::TranscriptGadget;
use crate::engine::r1cs_circuit::{Lc, R1csBuilder, Var};
use crate::paper::params::Params;
use crate::paper::reductions::pi_ccs_circuit::{
    enforce_pi_ccs, enforce_pi_ccs_with_matrix_digest_wires, PiCcsOutputWires, PiCcsVerifierConfig,
    PiCcsVerifierMessages,
};
use crate::paper::reductions::pi_dec_circuit::{self, enforce_dec_v_strict};
use crate::paper::reductions::{pi_ccs, pi_ccs_circuit};
use crate::paper::relations::product_commitment_circuit::AdvCommitmentWires;
use crate::paper::relations::{CcsClaim, CeClaim};

mod pi_rlc;
pub mod stage;

/// Configuration for one NIFS.V step.
pub struct NifsVCircuitConfig<'a> {
    pub pi_ccs: PiCcsVerifierConfig<'a>,
}

/// Witness/protocol messages from a real native `nifs::prove` proof.
pub struct NifsVCircuitMessages<'a> {
    pub fresh: &'a [CcsClaim],
    pub running: &'a [CeClaim],
    pub running_parent_authority: Option<&'a CeClaim>,
    pub pi_ccs: &'a pi_ccs::Proof,
    pub combined: &'a CeClaim,
    pub children: &'a [CeClaim],
}

/// Output wires from one NIFS.V step that F' R1CS composition consumes.
pub struct NifsVOutputs {
    /// Π_DEC parent commitment data.
    pub parent_c_data: Vec<Var>,
    /// Fresh CCS instances' public-input wires `[fresh_idx][x_lane]`.
    pub fresh_x: Vec<Vec<Var>>,
    /// Product-commitment coordinates paired with [`Self::fresh_x`].
    pub fresh_adv: Vec<Option<AdvCommitmentWires>>,
    /// Per-running-claim commitment data wires `[running_idx][lane]`.
    pub running_c_data: Vec<Vec<Var>>,
    /// Four-lane handle of the exact ordered incoming child accumulator.
    pub running_acc_digest: [Var; 4],
    /// Incoming running accumulator wires used by the continuity gate.
    pub running: Vec<PiCcsOutputWires>,
    /// Checked Π_RLC recomposition cache for [`Self::running`], when non-empty.
    pub running_parent_authority: Option<PiCcsOutputWires>,
    /// Current Π_RLC parent, checked by Π_DEC and retained as a cache.
    pub parent: pi_dec_circuit::CeClaimWires,
    /// Exact ordered Π_DEC children that become the next accumulator.
    pub children: Vec<pi_dec_circuit::CeClaimWires>,
    /// Exact compiler receipt for the outer strict Π_DEC public-X rows.
    pub pi_dec_canonical_x_receipt: crate::engine::r1cs_circuit::PiDecCanonicalXReceipt,
    /// Transcript-owned Π_RLC projection beta.
    pub projection_beta: [Var; 2],
    /// Per-commitment-lane projection quotient advice.
    pub projection_q_lanes: Vec<[Var; PROJECTION_QUOTIENT_LEN]>,
    /// Product-commitment `(ops, is, fs)` quotient advice for Nebula folds.
    pub projection_adv_q_lanes: Option<LaneCommitments<Vec<[Var; PROJECTION_QUOTIENT_LEN]>>>,
    /// One quotient per active X ring column.
    pub projection_x_q_lanes: Vec<[Var; PROJECTION_QUOTIENT_LEN]>,
    /// Two quotients per y_ring row.
    pub projection_y_ring_q_lanes: Vec<[[Var; PROJECTION_QUOTIENT_LEN]; 2]>,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("NIFS.V circuit: {0}")]
    Inner(String),
    #[error(transparent)]
    PiCcs(#[from] pi_ccs_circuit::Error),
    #[error(transparent)]
    PiDec(#[from] crate::paper::reductions::pi_dec_circuit::Error),
    #[error(transparent)]
    PiRlc(#[from] crate::paper::reductions::pi_rlc_circuit::Error),
}

/// Enforce NIFS.V on top of `transcript`.
pub fn enforce_nifs_v_circuit_with_transcript(
    builder: &mut R1csBuilder,
    pp: &Params,
    cfg: &NifsVCircuitConfig<'_>,
    transcript: &mut TranscriptGadget,
    msg: &NifsVCircuitMessages<'_>,
) -> Result<NifsVOutputs, Error> {
    enforce_nifs_v_circuit_with_transcript_inner(builder, pp, cfg, transcript, msg, None)
}

/// Folded-F' entrypoint using the verifier header carried by F' state.
pub fn enforce_nifs_v_circuit_with_transcript_and_header_bundle(
    builder: &mut R1csBuilder,
    pp: &Params,
    cfg: &NifsVCircuitConfig<'_>,
    transcript: &mut TranscriptGadget,
    msg: &NifsVCircuitMessages<'_>,
    header_bundle: [Var; 4],
) -> Result<NifsVOutputs, Error> {
    enforce_nifs_v_circuit_with_transcript_inner(builder, pp, cfg, transcript, msg, Some(header_bundle))
}

/// Fixed-relation alias with the header-first argument order used by F'.
pub fn enforce_nifs_v_circuit_with_transcript_and_header_bundle_wires(
    builder: &mut R1csBuilder,
    pp: &Params,
    cfg: &NifsVCircuitConfig<'_>,
    transcript: &mut TranscriptGadget,
    header_bundle: [Var; 4],
    msg: &NifsVCircuitMessages<'_>,
) -> Result<NifsVOutputs, Error> {
    enforce_nifs_v_circuit_with_transcript_inner(builder, pp, cfg, transcript, msg, Some(header_bundle))
}

fn enforce_nifs_v_circuit_with_transcript_inner(
    builder: &mut R1csBuilder,
    pp: &Params,
    cfg: &NifsVCircuitConfig<'_>,
    transcript: &mut TranscriptGadget,
    msg: &NifsVCircuitMessages<'_>,
    header_bundle: Option<[Var; 4]>,
) -> Result<NifsVOutputs, Error> {
    let nifs_start = builder.rows();
    let pi_ccs_start = builder.rows();
    let pi_ccs_first_column = builder.cols();
    let pi_ccs_messages = PiCcsVerifierMessages {
        fresh: msg.fresh,
        running: msg.running,
        running_parent_authority: msg.running_parent_authority,
        outputs: &msg.pi_ccs.outputs,
        sumcheck_rounds: &msg.pi_ccs.sumcheck.sumcheck_rounds,
    };
    let ccs = match header_bundle {
        Some(header_bundle) => {
            enforce_pi_ccs_with_matrix_digest_wires(builder, transcript, &cfg.pi_ccs, &pi_ccs_messages, header_bundle)?
        }
        None => enforce_pi_ccs(builder, transcript, &cfg.pi_ccs, &pi_ccs_messages)?,
    };
    builder.record_row_family("nifs.pi_ccs", pi_ccs_start);
    builder.record_program_range("nifs.pi_ccs", pi_ccs_start, pi_ccs_first_column);

    enforce_running_parent_authority(builder, pp, &ccs)?;

    let rlc = pi_rlc::enforce(builder, pp, &cfg.pi_ccs, transcript, &ccs, msg.combined, msg.children)?;

    let pi_dec_start = builder.rows();
    let pi_dec_first_column = builder.cols();
    builder.begin_encoding_stage(stage::PI_DEC);
    builder.begin_encoding_stage(stage::PI_DEC_VERIFY);
    let pi_dec_canonical_x_receipt = enforce_dec_v_strict(builder, pp, &rlc.dec_wires)?;
    builder.record_row_family("nifs.pi_dec", pi_dec_start);
    builder.record_program_range("nifs.pi_dec", pi_dec_start, pi_dec_first_column);

    let point_binding_start = builder.rows();
    let point_binding_first_column = builder.cols();
    builder.begin_encoding_stage(stage::POINT_BINDING);
    enforce_kvar_vec_eq(builder, &rlc.dec_wires.parent.r, &ccs.r_prime)?;
    builder.record_row_family("nifs.point_binding", point_binding_start);
    builder.record_program_range("nifs.point_binding", point_binding_start, point_binding_first_column);
    builder.record_row_family("nifs.total", nifs_start);

    let parent = rlc.dec_wires.parent.clone();
    let children = rlc.dec_wires.children.clone();
    Ok(NifsVOutputs {
        parent_c_data: rlc.dec_wires.parent.c_data.clone(),
        fresh_x: ccs.fresh_x,
        fresh_adv: ccs.fresh_adv,
        running_c_data: ccs.running_c_data,
        running_acc_digest: ccs.running_acc_digest,
        running: ccs.running.clone(),
        running_parent_authority: ccs.running_parent_authority.clone(),
        parent,
        children,
        pi_dec_canonical_x_receipt,
        projection_beta: rlc.projection_beta,
        projection_q_lanes: rlc.projection_q_lanes,
        projection_adv_q_lanes: rlc.projection_adv_q_lanes,
        projection_x_q_lanes: rlc.projection_x_q_lanes,
        projection_y_ring_q_lanes: rlc.projection_y_ring_q_lanes,
    })
}

fn enforce_running_parent_authority(
    builder: &mut R1csBuilder,
    pp: &Params,
    ccs: &crate::paper::reductions::pi_ccs_circuit::PiCcsVerifierResult,
) -> Result<(), Error> {
    match (ccs.running.as_slice(), ccs.running_parent_authority.as_ref()) {
        ([], None) => Ok(()),
        ([], Some(_)) => Err(Error::Inner("empty running accumulator has a parent authority".into())),
        (_, None) => Err(Error::Inner(
            "nonempty running accumulator has no parent authority".into(),
        )),
        (children, Some(parent)) => {
            let row_start = builder.rows();
            let first_column = builder.cols();
            builder.begin_encoding_stage(stage::RUNNING_PARENT_PI_DEC);
            let wires = pi_dec_circuit::DecInputWires {
                parent: dec_claim_wires(parent),
                children: children.iter().map(dec_claim_wires).collect(),
            };
            enforce_dec_v_strict(builder, pp, &wires)?;
            builder.record_row_family(stage::RUNNING_PARENT_PI_DEC, row_start);
            builder.record_program_range(stage::RUNNING_PARENT_PI_DEC, row_start, first_column);
            Ok(())
        }
    }
}

fn dec_claim_wires(claim: &PiCcsOutputWires) -> pi_dec_circuit::CeClaimWires {
    pi_dec_circuit::CeClaimWires {
        c_data: claim.c_data.clone(),
        c_d: claim.c_d,
        c_d_var: claim.c_d_var,
        c_kappa: claim.c_kappa,
        c_kappa_var: claim.c_kappa_var,
        adv: claim.adv.clone(),
        x: claim.x.clone(),
        x_rows: claim.x_rows,
        x_rows_var: claim.x_rows_var,
        x_cols: claim.x_cols,
        x_cols_var: claim.x_cols_var,
        m_in: claim.m_in,
        m_in_var: claim.m_in_var,
        y_ring: claim
            .y_ring
            .iter()
            .map(|row| row.iter().flat_map(|value| [value.c0, value.c1]).collect())
            .collect(),
        y_ring_lanes: claim.y_ring.first().map(Vec::len).unwrap_or(0),
        ct: claim.ct.clone(),
        r: claim.r.clone(),
        fold_digest_fields: claim.fold_digest_fields,
    }
}

fn enforce_kvar_vec_eq(builder: &mut R1csBuilder, left: &[KVar], right: &[KVar]) -> Result<(), Error> {
    if left.len() != right.len() {
        return Err(Error::Inner(format!(
            "KVar vector length mismatch: {} vs {}",
            left.len(),
            right.len()
        )));
    }
    for (left, right) in left.iter().zip(right) {
        builder.enforce_eq(&Lc::from_var(left.c0), &Lc::from_var(right.c0));
        builder.enforce_eq(&Lc::from_var(left.c1), &Lc::from_var(right.c1));
    }
    Ok(())
}
