//! Independent formula verifier for corrected rectangular-paper `Pi_CCS`.

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsClaim, CcsStructure, CeClaim};
use neo_math::{F, K};
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

use crate::engines::pi_ccs_protocol::{PiCcsProof, PiCcsProofVariant};
use crate::engines::{pi_ccs_rectangular, utils};
use crate::error::PiCcsError;

use super::paper_rectangular::{paper_fe_initial, paper_fe_terminal, paper_nc_terminal};

pub fn paper_exact_verify(
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    outputs: &[CeClaim<Cmt, F, K>],
    proof: &PiCcsProof,
) -> Result<bool, PiCcsError> {
    if proof.variant != PiCcsProofVariant::PaperRectangularV1 {
        return Err(PiCcsError::ProtocolError(
            "PaperExact expected a PaperRectangularV1 proof".into(),
        ));
    }
    if proof._extra.is_some() {
        return Err(PiCcsError::ProtocolError(
            "PaperRectangularV1 does not permit extra proof data".into(),
        ));
    }

    let (dims, challenges) =
        pi_ccs_rectangular::bind_and_sample(transcript, params, structure, fresh_claims, running_claims)?;
    if proof.challenges_public != challenges {
        return Err(PiCcsError::ProtocolError(
            "PaperExact public challenges do not match transcript replay".into(),
        ));
    }

    let initial_fe = paper_fe_initial(structure, &challenges, fresh_claims.len(), running_claims)?;
    if proof.sc_initial_sum != Some(initial_fe) || proof.sc_initial_sum_nc != Some(K::ZERO) {
        return Err(PiCcsError::SumcheckError("PaperExact initial claim mismatch".into()));
    }

    let (row_point, final_fe) = pi_ccs_rectangular::verify_phase(
        transcript,
        utils::PI_CCS_SUMCHECK_FE_RAW_DOMAIN_TAG,
        dims.d_sc,
        dims.ell_n,
        initial_fe,
        &proof.sumcheck_rounds,
    )?;
    let (column_point, final_nc) = pi_ccs_rectangular::verify_phase(
        transcript,
        utils::PI_CCS_SUMCHECK_NC_RAW_DOMAIN_TAG,
        dims.d_sc,
        dims.ell_m,
        K::ZERO,
        &proof.sumcheck_rounds_nc,
    )?;
    if proof.sumcheck_challenges != row_point
        || proof.sumcheck_challenges_nc != column_point
        || proof.sumcheck_final != final_fe
        || proof.sumcheck_final_nc != final_nc
    {
        return Err(PiCcsError::ProtocolError(
            "PaperExact stored SumCheck state does not match transcript replay".into(),
        ));
    }

    let fold_digest = transcript.digest32();
    if proof.header_digest.as_slice() != fold_digest {
        return Err(PiCcsError::ProtocolError(
            "PaperExact proof digest does not match transcript replay".into(),
        ));
    }
    if outputs
        .iter()
        .any(|output| output.fold_digest != fold_digest)
    {
        return Err(PiCcsError::ProtocolError(
            "PaperExact output digest does not match transcript replay".into(),
        ));
    }
    utils::validate_me_outputs_against_inputs(
        structure,
        params,
        fresh_claims,
        running_claims,
        outputs,
        &row_point,
        &column_point,
    )?;

    let prior_point = utils::shared_me_input_r(running_claims, dims.ell_n)?;
    let expected_fe = paper_fe_terminal(
        structure,
        &challenges,
        fresh_claims.len(),
        prior_point,
        &row_point,
        outputs,
    )?;
    let expected_nc = paper_nc_terminal::<F>(params, &challenges, fresh_claims.len(), &column_point, outputs)?;
    Ok(final_fe == expected_fe && final_nc == expected_nc)
}
