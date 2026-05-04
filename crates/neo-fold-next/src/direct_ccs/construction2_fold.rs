//! Owns the in-circuit Construction-2 accumulator update for direct F'.

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use neo_ajtai::Commitment;
use neo_ccs::{CcsStructure, CeClaim};
use neo_math::{F, K};
use neo_params::NeoParams;

use super::circuit_util::{
    digest32_as_spartan_fields, direct_accumulator_digest_circuit_from_claims,
    direct_terminal_construction2_accumulator_digest_range, enforce_digest_eq_constant,
    enforce_digest_fields_public_io,
};
use super::ivc::{DirectCcsChunkCircuitSurface, DirectCcsFPrimeSnarkError};
use super::ivc_helpers::{alloc_initial_claim_bundle, alloc_initial_transcript};
use crate::ivc::SuperNeoIvcTranscriptSnapshot;
use crate::spartan_backend::{NeoFoldDeciderEngine, ShapeCS, SpartanF};
use crate::superneo_nifs_circuit::synthesize_superneo_nifs_chunk;
use neo_reductions::engines::utils::Dims;

#[derive(Clone)]
pub(crate) struct DirectCcsConstruction2FoldContext {
    pub(crate) params: NeoParams,
    pub(crate) structure: CcsStructure<F>,
    pub(crate) dims: Dims,
    pub(crate) mat_digest: [F; 4],
    pub(crate) initial_claims: Vec<CeClaim<Commitment, F, K>>,
    pub(crate) initial_transcript: Option<SuperNeoIvcTranscriptSnapshot>,
    pub(crate) surface: DirectCcsChunkCircuitSurface,
    pub(crate) accumulator_in_digest: [u8; 32],
    pub(crate) accumulator_out_digest: [u8; 32],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DirectCcsConstruction2FoldShapeDelta {
    pub(crate) rows: usize,
    pub(crate) public_cols: usize,
    pub(crate) aux_cols: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct DirectCcsConstruction2FoldBreakdown {
    pub(crate) initial_transcript: DirectCcsConstruction2FoldShapeDelta,
    pub(crate) initial_claims: DirectCcsConstruction2FoldShapeDelta,
    pub(crate) accumulator_in_digest: DirectCcsConstruction2FoldShapeDelta,
    pub(crate) accumulator_in_digest_check: DirectCcsConstruction2FoldShapeDelta,
    pub(crate) nifs_v: DirectCcsConstruction2FoldShapeDelta,
    pub(crate) accumulator_out_digest: DirectCcsConstruction2FoldShapeDelta,
    pub(crate) accumulator_out_digest_check: DirectCcsConstruction2FoldShapeDelta,
    pub(crate) accumulator_out_public_link: DirectCcsConstruction2FoldShapeDelta,
    pub(crate) total: DirectCcsConstruction2FoldShapeDelta,
}

impl DirectCcsConstruction2FoldBreakdown {
    pub(crate) fn log_lines(&self) -> Vec<String> {
        let mut lines =
            vec!["direct_ccs_ivc.construction2_fold_breakdown stage|rows|public_cols|aux_cols|primitive".to_owned()];
        push_log(
            &mut lines,
            "initial_transcript",
            self.initial_transcript,
            "allocate prior F' transcript state",
        );
        push_log(
            &mut lines,
            "initial_claims",
            self.initial_claims,
            "allocate prior F' carried CE claims",
        );
        push_log(
            &mut lines,
            "accumulator_in_digest",
            self.accumulator_in_digest,
            "Poseidon2 digest of incoming prior F' CE accumulator",
        );
        push_log(
            &mut lines,
            "accumulator_in_digest_check",
            self.accumulator_in_digest_check,
            "check incoming prior F' accumulator digest boundary",
        );
        push_log(
            &mut lines,
            "nested_nifs_v",
            self.nifs_v,
            "SuperNeo NIFS.V replay for the prior F' step",
        );
        push_log(
            &mut lines,
            "accumulator_out_digest",
            self.accumulator_out_digest,
            "Poseidon2 digest of outgoing prior F' CE accumulator",
        );
        push_log(
            &mut lines,
            "accumulator_out_digest_check",
            self.accumulator_out_digest_check,
            "check outgoing prior F' accumulator digest boundary",
        );
        push_log(
            &mut lines,
            "accumulator_out_public_link",
            self.accumulator_out_public_link,
            "link outgoing prior F' accumulator digest to terminal public image",
        );
        push_log(
            &mut lines,
            "total",
            self.total,
            "Construction-2 folded prior F' accumulator update",
        );
        lines
    }
}

fn push_log(lines: &mut Vec<String>, stage: &str, shape: DirectCcsConstruction2FoldShapeDelta, primitive: &str) {
    lines.push(format!(
        "direct_ccs_ivc.construction2_fold_breakdown {stage}|{}|{}|{}|{primitive}",
        shape.rows, shape.public_cols, shape.aux_cols
    ));
}

fn shape_point(cs: &ShapeCS<NeoFoldDeciderEngine>) -> DirectCcsConstruction2FoldShapeDelta {
    DirectCcsConstruction2FoldShapeDelta {
        rows: cs.num_constraints(),
        public_cols: cs.num_inputs(),
        aux_cols: cs.num_aux(),
    }
}

fn shape_delta(
    start: DirectCcsConstruction2FoldShapeDelta,
    cs: &ShapeCS<NeoFoldDeciderEngine>,
) -> DirectCcsConstruction2FoldShapeDelta {
    let end = shape_point(cs);
    DirectCcsConstruction2FoldShapeDelta {
        rows: end.rows.saturating_sub(start.rows),
        public_cols: end.public_cols.saturating_sub(start.public_cols),
        aux_cols: end.aux_cols.saturating_sub(start.aux_cols),
    }
}

pub(crate) fn synthesize_direct_construction2_fold<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    context: Option<&DirectCcsConstruction2FoldContext>,
    public_inputs: &[AllocatedNum<SpartanF>],
    accumulator_in_digest: [u8; 32],
) -> Result<(), SynthesisError> {
    match context {
        Some(context) => {
            synthesize_direct_construction2_fold_context(cs, context, public_inputs, accumulator_in_digest)
        }
        None => Ok(()),
    }
}

pub(crate) fn measure_direct_construction2_fold(
    cs: &mut ShapeCS<NeoFoldDeciderEngine>,
    context: Option<&DirectCcsConstruction2FoldContext>,
    public_inputs: &[AllocatedNum<SpartanF>],
    accumulator_in_digest: [u8; 32],
) -> Result<DirectCcsConstruction2FoldBreakdown, SynthesisError> {
    match context {
        Some(context) => measure_direct_construction2_fold_context(cs, context, public_inputs, accumulator_in_digest),
        None => Ok(DirectCcsConstruction2FoldBreakdown::default()),
    }
}

fn synthesize_direct_construction2_fold_context<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    context: &DirectCcsConstruction2FoldContext,
    public_inputs: &[AllocatedNum<SpartanF>],
    accumulator_in_digest: [u8; 32],
) -> Result<(), SynthesisError> {
    if context.accumulator_in_digest != accumulator_in_digest {
        return Err(SynthesisError::Unsatisfiable);
    }
    let mut transcript = alloc_initial_transcript(
        &mut cs.namespace(|| "direct_terminal_construction2_fold_transcript"),
        context.initial_transcript.as_ref(),
    )?;
    let carried = alloc_initial_claim_bundle(
        &mut cs.namespace(|| "direct_terminal_construction2_fold_initial_claims"),
        &context.initial_claims,
    )?;
    let accumulator_in = direct_accumulator_digest_circuit_from_claims(
        &mut cs.namespace(|| "direct_terminal_construction2_fold_accumulator_in"),
        &context.params,
        carried.effective_claims(),
    )?;
    enforce_digest_eq_constant(
        &mut cs.namespace(|| "direct_terminal_construction2_fold_accumulator_in_private"),
        &accumulator_in,
        context.accumulator_in_digest,
        "direct_terminal_construction2_fold_accumulator_in_private",
    )?;
    let (next, _) = synthesize_superneo_nifs_chunk(
        &context.params,
        &context.structure,
        context.dims,
        &context.mat_digest,
        &mut cs.namespace(|| "direct_terminal_construction2_fold_nifs_v"),
        0,
        &context.surface.cover,
        &context.surface.replay,
        &mut transcript,
        carried,
        Some((
            &accumulator_in,
            digest32_as_spartan_fields(context.accumulator_in_digest),
        )),
    )?;
    let accumulator_out = direct_accumulator_digest_circuit_from_claims(
        &mut cs.namespace(|| "direct_terminal_construction2_fold_accumulator_out"),
        &context.params,
        next.effective_claims(),
    )?;
    enforce_digest_eq_constant(
        &mut cs.namespace(|| "direct_terminal_construction2_fold_accumulator_out_private"),
        &accumulator_out,
        context.accumulator_out_digest,
        "direct_terminal_construction2_fold_accumulator_out_private",
    )?;
    enforce_digest_fields_public_io(
        &mut cs.namespace(|| "direct_terminal_construction2_fold_accumulator_out_public"),
        &accumulator_out,
        public_inputs,
        direct_terminal_construction2_accumulator_digest_range(),
        "direct_terminal_construction2_fold_accumulator_out_public",
    )
}

fn measure_direct_construction2_fold_context(
    cs: &mut ShapeCS<NeoFoldDeciderEngine>,
    context: &DirectCcsConstruction2FoldContext,
    public_inputs: &[AllocatedNum<SpartanF>],
    accumulator_in_digest: [u8; 32],
) -> Result<DirectCcsConstruction2FoldBreakdown, SynthesisError> {
    if context.accumulator_in_digest != accumulator_in_digest {
        return Err(SynthesisError::Unsatisfiable);
    }
    let start = shape_point(cs);
    let mut out = DirectCcsConstruction2FoldBreakdown::default();

    let before = shape_point(cs);
    let mut transcript = alloc_initial_transcript(
        &mut cs.namespace(|| "direct_terminal_construction2_fold_transcript"),
        context.initial_transcript.as_ref(),
    )?;
    out.initial_transcript = shape_delta(before, cs);

    let before = shape_point(cs);
    let carried = alloc_initial_claim_bundle(
        &mut cs.namespace(|| "direct_terminal_construction2_fold_initial_claims"),
        &context.initial_claims,
    )?;
    out.initial_claims = shape_delta(before, cs);

    let before = shape_point(cs);
    let accumulator_in = direct_accumulator_digest_circuit_from_claims(
        &mut cs.namespace(|| "direct_terminal_construction2_fold_accumulator_in"),
        &context.params,
        carried.effective_claims(),
    )?;
    out.accumulator_in_digest = shape_delta(before, cs);

    let before = shape_point(cs);
    enforce_digest_eq_constant(
        &mut cs.namespace(|| "direct_terminal_construction2_fold_accumulator_in_private"),
        &accumulator_in,
        context.accumulator_in_digest,
        "direct_terminal_construction2_fold_accumulator_in_private",
    )?;
    out.accumulator_in_digest_check = shape_delta(before, cs);

    let before = shape_point(cs);
    let (next, _) = synthesize_superneo_nifs_chunk(
        &context.params,
        &context.structure,
        context.dims,
        &context.mat_digest,
        &mut cs.namespace(|| "direct_terminal_construction2_fold_nifs_v"),
        0,
        &context.surface.cover,
        &context.surface.replay,
        &mut transcript,
        carried,
        Some((
            &accumulator_in,
            digest32_as_spartan_fields(context.accumulator_in_digest),
        )),
    )?;
    out.nifs_v = shape_delta(before, cs);

    let before = shape_point(cs);
    let accumulator_out = direct_accumulator_digest_circuit_from_claims(
        &mut cs.namespace(|| "direct_terminal_construction2_fold_accumulator_out"),
        &context.params,
        next.effective_claims(),
    )?;
    out.accumulator_out_digest = shape_delta(before, cs);

    let before = shape_point(cs);
    enforce_digest_eq_constant(
        &mut cs.namespace(|| "direct_terminal_construction2_fold_accumulator_out_private"),
        &accumulator_out,
        context.accumulator_out_digest,
        "direct_terminal_construction2_fold_accumulator_out_private",
    )?;
    out.accumulator_out_digest_check = shape_delta(before, cs);

    let before = shape_point(cs);
    enforce_digest_fields_public_io(
        &mut cs.namespace(|| "direct_terminal_construction2_fold_accumulator_out_public"),
        &accumulator_out,
        public_inputs,
        direct_terminal_construction2_accumulator_digest_range(),
        "direct_terminal_construction2_fold_accumulator_out_public",
    )?;
    out.accumulator_out_public_link = shape_delta(before, cs);

    out.total = shape_delta(start, cs);
    Ok(out)
}

impl DirectCcsConstruction2FoldContext {
    pub(crate) fn validate_digest_linkage(
        &self,
        in_digest: [u8; 32],
        out_digest: [u8; 32],
    ) -> Result<(), DirectCcsFPrimeSnarkError> {
        if self.accumulator_in_digest != in_digest || self.accumulator_out_digest != out_digest {
            return Err(DirectCcsFPrimeSnarkError::Input(
                "direct F' Construction-2 fold context does not match terminal accumulator digest boundary".into(),
            ));
        }
        Ok(())
    }
}
