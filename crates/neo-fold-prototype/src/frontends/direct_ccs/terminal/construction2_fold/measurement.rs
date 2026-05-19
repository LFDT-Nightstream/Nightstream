//! Shape measurement for the direct F' Construction-2 fold.

use super::*;

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
