//! Synthesis for the direct F' Construction-2 fold.

use super::*;

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
