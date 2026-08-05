//! Independent prover for the one-joint padded-row paper protocol.

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, CeClaim, Mat};
use neo_math::{F, K};
use neo_params::NeoParams;
use neo_transcript::Poseidon2Transcript;

use crate::engines::pi_ccs_joint::ProtocolTrace;
use crate::engines::pi_ccs_protocol::PiCcsProof;
use crate::error::PiCcsError;

use super::paper_joint::{
    build_outputs, dimensions, initial_claim, paper_prior_point, terminal, validate_fresh_assignment,
    validate_public_instances, PaperJointOracle,
};
use super::transcript::{absorb_outputs, assemble_proof, bind_and_sample, prove_sumcheck, PaperTranscriptBinding};

#[allow(clippy::too_many_arguments)]
pub(crate) fn paper_exact_prove_with_trace<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    fresh_witnesses: &[CcsWitness<F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    running_witnesses: &[Mat<F>],
    commitment: &L,
) -> Result<(Vec<CeClaim<Cmt, F, K>>, PiCcsProof, ProtocolTrace), PiCcsError> {
    paper_exact_prove_with_trace_and_binding(
        transcript,
        params,
        structure,
        fresh_claims,
        fresh_witnesses,
        running_claims,
        running_witnesses,
        commitment,
        PaperTranscriptBinding::claims(),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn paper_exact_prove_with_trace_and_binding<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    fresh_witnesses: &[CcsWitness<F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    running_witnesses: &[Mat<F>],
    commitment: &L,
    binding: PaperTranscriptBinding,
) -> Result<(Vec<CeClaim<Cmt, F, K>>, PiCcsProof, ProtocolTrace), PiCcsError> {
    if fresh_claims.len() != fresh_witnesses.len() {
        return Err(PiCcsError::InvalidInput(
            "PaperExact fresh claim/witness count mismatch".into(),
        ));
    }
    if running_claims.len() != running_witnesses.len() {
        return Err(PiCcsError::InvalidInput(
            "PaperExact running claim/witness count mismatch".into(),
        ));
    }

    validate_public_instances(structure, fresh_claims, running_claims)?;
    let dims = dimensions(params, structure, fresh_claims.len(), running_claims.len())?;
    for witness in fresh_witnesses {
        validate_fresh_assignment(&witness.Z, structure.m, dims)?;
    }
    let prior_point = paper_prior_point(running_claims, dims.variables)?;
    let mut trace = ProtocolTrace::default();
    let challenges = bind_and_sample(
        transcript,
        &mut trace,
        structure,
        fresh_claims,
        running_claims,
        dims,
        binding,
    )?;
    let initial = initial_claim(structure, &challenges, fresh_claims.len(), running_claims)?;
    let mut oracle = PaperJointOracle::new(
        structure,
        params,
        fresh_witnesses,
        running_witnesses,
        challenges.clone(),
        prior_point,
        dims,
    )?;
    let (rounds, round_challenges, final_claim) = prove_sumcheck(transcript, &mut trace, initial, &mut oracle)?;
    let mut outputs = build_outputs(
        structure,
        fresh_claims,
        fresh_witnesses,
        running_claims,
        running_witnesses,
        &round_challenges,
        dims,
        commitment,
    )?;
    let expected = terminal::<F>(
        structure,
        params,
        &challenges,
        fresh_claims.len(),
        prior_point,
        &round_challenges,
        &outputs,
    )?;
    if final_claim != expected {
        return Err(PiCcsError::SumcheckError(
            "PaperExact terminal value does not match the paper output message".into(),
        ));
    }
    let digest = absorb_outputs(transcript, &mut trace, &outputs, fresh_claims.len(), dims)?;
    for output in &mut outputs {
        output.fold_digest = digest;
    }
    let proof = assemble_proof(rounds);
    Ok((outputs, proof, trace))
}

pub fn paper_exact_prove<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    fresh_witnesses: &[CcsWitness<F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    running_witnesses: &[Mat<F>],
    commitment: &L,
) -> Result<(Vec<CeClaim<Cmt, F, K>>, PiCcsProof), PiCcsError> {
    let (outputs, proof, _) = paper_exact_prove_with_trace(
        transcript,
        params,
        structure,
        fresh_claims,
        fresh_witnesses,
        running_claims,
        running_witnesses,
        commitment,
    )?;
    Ok((outputs, proof))
}

/// PaperExact prover entrypoint for the compact NIFS statement binding.
///
/// The two digests are recomputed by the paper layer from authoritative
/// claims. This function only selects the matching independent transcript
/// encoding; it does not accept them as proof authority.
#[allow(clippy::too_many_arguments)]
pub fn paper_exact_prove_with_instance_digest_and_me_input_handle<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    fresh_witnesses: &[CcsWitness<F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    running_witnesses: &[Mat<F>],
    public_instance_digest: [F; 4],
    running_accumulator_handle: [F; 4],
    commitment: &L,
) -> Result<(Vec<CeClaim<Cmt, F, K>>, PiCcsProof), PiCcsError> {
    let (outputs, proof, _) = paper_exact_prove_with_trace_and_binding(
        transcript,
        params,
        structure,
        fresh_claims,
        fresh_witnesses,
        running_claims,
        running_witnesses,
        commitment,
        PaperTranscriptBinding::digests(public_instance_digest, Some(running_accumulator_handle)),
    )?;
    Ok((outputs, proof))
}
