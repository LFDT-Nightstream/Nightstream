//! Independent verifier for the one-joint padded-row paper protocol.

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsClaim, CcsStructure, CeClaim};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;

use crate::engines::pi_ccs_joint::{JointDims, ProtocolTrace};
use crate::engines::pi_ccs_protocol::PiCcsProof;
use crate::error::PiCcsError;

use super::paper_joint::{
    dimensions, initial_claim, paper_prior_point, terminal_components, validate_public_instances,
};
use super::transcript::{absorb_outputs, bind_and_sample, verify_sumcheck, PaperTranscriptBinding};

fn validate_outputs(
    fresh: &[CcsClaim<Cmt, F>],
    running: &[CeClaim<Cmt, F, K>],
    outputs: &[CeClaim<Cmt, F, K>],
    point: &[K],
    dims: JointDims,
) -> Result<(), PiCcsError> {
    if outputs.len() != fresh.len() + running.len() {
        return Err(PiCcsError::InvalidInput(
            "PaperExact output source count mismatch".into(),
        ));
    }
    for (index, output) in outputs.iter().enumerate() {
        if output.r != point
            || output.X.rows() != D
            || output.X.cols() != neo_ccs::superneo_public_x_cols(output.m_in)
            || output.eval_k.len() != D.next_power_of_two()
            || output.eval_a.len() != dims.matrix_count
        {
            return Err(PiCcsError::InvalidInput(format!(
                "PaperExact output {index} does not have the canonical v1_1 shape"
            )));
        }
        if output.eval_k.iter().skip(D).any(|&value| value != K::ZERO) {
            return Err(PiCcsError::InvalidInput(format!(
                "PaperExact output {index} Eval_K is not canonical"
            )));
        }
        for (matrix, coefficients) in output.eval_a.iter().enumerate() {
            if coefficients.len() != D.next_power_of_two() || coefficients.iter().skip(D).any(|&value| value != K::ZERO)
            {
                return Err(PiCcsError::InvalidInput(format!(
                    "PaperExact output {index} Eval_A matrix {matrix} is not canonical"
                )));
            }
        }
    }
    for (claim, output) in fresh.iter().zip(outputs) {
        if claim.c != output.c || claim.m_in != output.m_in || claim.adv != output.adv {
            return Err(PiCcsError::ProtocolError(
                "PaperExact fresh output changed its public instance".into(),
            ));
        }
        if claim.m_in % D != 0 || claim.x.len() != claim.m_in {
            return Err(PiCcsError::InvalidInput(
                "PaperExact fresh public input is not whole-ring aligned".into(),
            ));
        }
        for (coordinate, &value) in claim.x.iter().enumerate() {
            if output.X[(coordinate % D, coordinate / D)] != value {
                return Err(PiCcsError::ProtocolError(
                    "PaperExact fresh output changed a public input coordinate".into(),
                ));
            }
        }
    }
    for (claim, output) in running.iter().zip(outputs.iter().skip(fresh.len())) {
        if claim.c != output.c || claim.X != output.X || claim.m_in != output.m_in || claim.adv != output.adv {
            return Err(PiCcsError::ProtocolError(
                "PaperExact carried output changed its public instance".into(),
            ));
        }
    }
    Ok(())
}

pub fn paper_exact_verify_with_trace(
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    outputs: &[CeClaim<Cmt, F, K>],
    proof: &PiCcsProof,
) -> Result<(bool, ProtocolTrace), PiCcsError> {
    paper_exact_verify_with_trace_and_binding(
        transcript,
        params,
        structure,
        fresh_claims,
        running_claims,
        outputs,
        proof,
        PaperTranscriptBinding::digest_only(),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn paper_exact_verify_with_trace_and_binding(
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    outputs: &[CeClaim<Cmt, F, K>],
    proof: &PiCcsProof,
    binding: PaperTranscriptBinding,
) -> Result<(bool, ProtocolTrace), PiCcsError> {
    validate_public_instances(structure, fresh_claims, running_claims)?;
    let dims = dimensions(params, structure, fresh_claims.len(), running_claims.len())?;
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
    let (point, final_claim) = verify_sumcheck(transcript, &mut trace, dims, initial, &proof.sumcheck_rounds)?;
    validate_outputs(fresh_claims, running_claims, outputs, &point, dims)?;
    let terminal = terminal_components::<F>(
        structure,
        params,
        &challenges,
        fresh_claims.len(),
        prior_point,
        &point,
        outputs,
    )?;
    let expected = terminal.terminal;
    trace.terminal_components = terminal;
    let digest = absorb_outputs(transcript, &mut trace, outputs, fresh_claims.len(), dims)?;
    if outputs.iter().any(|output| output.fold_digest != digest) {
        return Err(PiCcsError::ProtocolError(
            "PaperExact output digest does not match transcript replay".into(),
        ));
    }
    Ok((final_claim == expected, trace))
}

pub fn paper_exact_verify(
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    outputs: &[CeClaim<Cmt, F, K>],
    proof: &PiCcsProof,
) -> Result<bool, PiCcsError> {
    Ok(paper_exact_verify_with_trace(
        transcript,
        params,
        structure,
        fresh_claims,
        running_claims,
        outputs,
        proof,
    )?
    .0)
}
