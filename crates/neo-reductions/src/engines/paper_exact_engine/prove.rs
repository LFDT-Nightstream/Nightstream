//! Independent prover for corrected rectangular-paper `Pi_CCS`.

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, CeClaim, Mat};
use neo_math::{F, K};
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

use crate::engines::pi_ccs_protocol::PiCcsProof;
use crate::engines::pi_ccs_rectangular;
use crate::error::PiCcsError;

use super::paper_rectangular::{
    build_outputs, paper_fe_initial, paper_fe_terminal, paper_nc_terminal, PaperJointSquareOracle,
    PaperRectangularFeOracle, PaperRectangularNcOracle,
};

/// Execute the paper's original one-polynomial square-domain SumCheck. This
/// function is a conformance baseline and is not a transport proof variant.
#[allow(clippy::too_many_arguments)]
pub fn paper_joint_square_prove_phase(
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    fresh_witnesses: &[CcsWitness<F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    running_witnesses: &[Mat<F>],
    challenges: crate::engines::pi_ccs_protocol::Challenges,
) -> Result<(K, pi_ccs_rectangular::PhaseProof), PiCcsError> {
    if fresh_claims.len() != fresh_witnesses.len() || running_claims.len() != running_witnesses.len() {
        return Err(PiCcsError::InvalidInput(
            "paper joint square claim/witness count mismatch".into(),
        ));
    }
    let dims = crate::engines::utils::build_dims_and_policy(params, structure)?;
    if dims.ell_n != dims.ell_m {
        return Err(PiCcsError::InvalidInput(
            "paper joint square requires equal padded row and column domains".into(),
        ));
    }
    let prior_point = crate::engines::utils::shared_me_input_r(running_claims, dims.ell_n)?;
    let initial = paper_fe_initial(structure, &challenges, fresh_claims.len(), running_claims)?;
    let mut oracle = PaperJointSquareOracle::new(
        structure,
        params,
        fresh_witnesses,
        running_witnesses,
        challenges,
        prior_point,
        dims.ell_n,
        dims.d_sc,
    )?;
    let phase = pi_ccs_rectangular::prove_phase(
        transcript,
        crate::engines::utils::PI_CCS_SUMCHECK_JOINT_RAW_DOMAIN_TAG,
        initial,
        &mut oracle,
    )?;
    Ok((initial, phase))
}

/// Prove the corrected paper algebra with one row-domain FE SumCheck and one
/// column-domain NC SumCheck.
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
    if fresh_claims.is_empty() {
        return Err(PiCcsError::InvalidInput(
            "paper_exact_prove: empty fresh claim list".into(),
        ));
    }
    if fresh_claims.len() != fresh_witnesses.len() {
        return Err(PiCcsError::InvalidInput(
            "paper_exact_prove: fresh claim/witness count mismatch".into(),
        ));
    }
    if running_claims.len() != running_witnesses.len() {
        return Err(PiCcsError::InvalidInput(
            "paper_exact_prove: running claim/witness count mismatch".into(),
        ));
    }

    let (dims, challenges) =
        pi_ccs_rectangular::bind_and_sample(transcript, params, structure, fresh_claims, running_claims)?;
    let prior_point = crate::engines::utils::shared_me_input_r(running_claims, dims.ell_n)?;
    let initial_fe = paper_fe_initial(structure, &challenges, fresh_claims.len(), running_claims)?;

    let mut fe_oracle = PaperRectangularFeOracle::new(
        structure,
        fresh_witnesses,
        running_witnesses,
        challenges.clone(),
        prior_point,
        dims.ell_n,
        dims.d_sc,
    )?;
    let fe = pi_ccs_rectangular::prove_phase(
        transcript,
        crate::engines::utils::PI_CCS_SUMCHECK_FE_RAW_DOMAIN_TAG,
        initial_fe,
        &mut fe_oracle,
    )?;

    let mut nc_oracle = PaperRectangularNcOracle::new(
        structure,
        params,
        fresh_witnesses,
        running_witnesses,
        challenges.clone(),
        dims.ell_m,
        dims.d_sc,
    )?;
    let nc = pi_ccs_rectangular::prove_phase(
        transcript,
        crate::engines::utils::PI_CCS_SUMCHECK_NC_RAW_DOMAIN_TAG,
        K::ZERO,
        &mut nc_oracle,
    )?;

    let fold_digest = transcript.digest32();
    let outputs = build_outputs(
        structure,
        fresh_claims,
        fresh_witnesses,
        running_claims,
        running_witnesses,
        &fe.challenges,
        &nc.challenges,
        fold_digest,
        commitment,
    )?;
    let expected_fe = paper_fe_terminal(
        structure,
        &challenges,
        fresh_claims.len(),
        prior_point,
        &fe.challenges,
        &outputs,
    )?;
    let expected_nc = paper_nc_terminal::<F>(params, &challenges, fresh_claims.len(), &nc.challenges, &outputs)?;
    if fe.final_claim != expected_fe || nc.final_claim != expected_nc {
        return Err(PiCcsError::SumcheckError(
            "paper exact terminal evaluation does not match direct output openings".into(),
        ));
    }

    let proof = pi_ccs_rectangular::assemble_proof(challenges, initial_fe, fe, nc, fold_digest);
    Ok((outputs, proof))
}
