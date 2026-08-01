//! Canonical transcript and proof flow for rectangular-paper `Pi_CCS`.
//!
//! Provers provide FE and NC round oracles. This module owns the shared
//! Poseidon2 transcript order, fixed round encoding, proof assembly, and
//! verifier replay. It does not own an oracle implementation.

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsClaim, CcsStructure, CeClaim};
use neo_math::{KExtensions, F, K};
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

use crate::engines::pi_ccs_protocol::{
    fe_initial_claim, fe_terminal, nc_terminal, Challenges, PiCcsProof, PiCcsProofVariant,
};
use crate::engines::utils::{self, Dims, PiCcsTranscriptVariant};
use crate::error::PiCcsError;
use crate::sumcheck::RoundOracle;

#[derive(Debug)]
pub struct PhaseProof {
    pub rounds: Vec<Vec<K>>,
    pub challenges: Vec<K>,
    pub final_claim: K,
}

/// Alternative public transcript bindings used by recursive callers. The
/// caller must recompute both values from authoritative public inputs.
#[derive(Clone, Copy, Debug, Default)]
pub struct TranscriptBinding {
    pub public_instance_digest: Option<[F; 4]>,
    pub running_accumulator_handle: Option<[F; 4]>,
}

impl TranscriptBinding {
    pub const fn claims() -> Self {
        Self {
            public_instance_digest: None,
            running_accumulator_handle: None,
        }
    }

    pub const fn digest(public_instance_digest: [F; 4]) -> Self {
        Self {
            public_instance_digest: Some(public_instance_digest),
            running_accumulator_handle: None,
        }
    }

    pub const fn digest_and_handle(public_instance_digest: [F; 4], running_accumulator_handle: [F; 4]) -> Self {
        Self {
            public_instance_digest: Some(public_instance_digest),
            running_accumulator_handle: Some(running_accumulator_handle),
        }
    }
}

/// Bind one public statement and sample the canonical row, column, and gamma
/// challenges. Both engines call this exact function.
pub fn bind_and_sample(
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    running_claims: &[CeClaim<Cmt, F, K>],
) -> Result<(Dims, Challenges), PiCcsError> {
    bind_and_sample_with_binding(
        transcript,
        params,
        structure,
        fresh_claims,
        running_claims,
        TranscriptBinding::claims(),
    )
}

/// Bind a statement through either its public claims or caller-recomputed
/// digests, then sample the canonical rectangular challenges.
pub fn bind_and_sample_with_binding(
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    binding: TranscriptBinding,
) -> Result<(Dims, Challenges), PiCcsError> {
    if fresh_claims.len() > neo_params::goldilocks_paper_b2::MAX_FRESH_K as usize {
        return Err(PiCcsError::InvalidInput(format!(
            "PaperRectangularV1 fresh source count {} exceeds the parameter profile maximum {}",
            fresh_claims.len(),
            neo_params::goldilocks_paper_b2::MAX_FRESH_K
        )));
    }
    if running_claims.len() > params.k_rho as usize {
        return Err(PiCcsError::InvalidInput(format!(
            "PaperRectangularV1 running source count {} exceeds k_rho={}",
            running_claims.len(),
            params.k_rho
        )));
    }
    for (index, claim) in running_claims.iter().enumerate() {
        crate::engines::pi_ccs_protocol::validate_inactive_x_zero(
            &format!("PaperRectangularV1 running_claims[{index}]"),
            claim,
        )?;
    }
    let dims = utils::build_dims_and_policy(params, structure)?;
    let matrix_digest = utils::digest_ccs_matrices(structure);
    if let Some(public_instance_digest) = binding.public_instance_digest {
        utils::bind_header_and_instance_digest_with_digest_for_variant(
            transcript,
            params,
            structure,
            dims,
            &matrix_digest,
            &public_instance_digest,
            PiCcsTranscriptVariant::PaperRectangularV1,
        )?;
    } else {
        utils::bind_header_and_instances_with_digest_for_variant(
            transcript,
            params,
            structure,
            fresh_claims,
            dims,
            &matrix_digest,
            PiCcsTranscriptVariant::PaperRectangularV1,
        )?;
    }
    if let Some(handle) = binding.running_accumulator_handle {
        utils::bind_me_inputs_accumulator_handle(transcript, running_claims.len(), &handle)?;
    } else {
        utils::bind_me_inputs(transcript, running_claims)?;
    }
    let challenges = utils::sample_paper_rectangular_challenges(transcript, dims.ell_n, dims.ell_m)?;
    Ok((dims, challenges))
}

fn append_phase_prolog(transcript: &mut Poseidon2Transcript, domain_tag: u64, initial_claim: K) {
    transcript.append_fields_raw(&[F::from_u64(domain_tag)]);
    transcript.append_fields_raw(&[F::from_u64(utils::PI_CCS_SUMCHECK_INITIAL_RAW_TAG)]);
    transcript.append_fields_raw(&initial_claim.as_coeffs());
    transcript.append_fields_raw(&[F::from_u64(crate::sumcheck::SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG)]);
}

/// Run one canonical raw-transcript SumCheck phase.
pub fn prove_phase<O: RoundOracle>(
    transcript: &mut Poseidon2Transcript,
    domain_tag: u64,
    initial_claim: K,
    oracle: &mut O,
) -> Result<PhaseProof, PiCcsError> {
    append_phase_prolog(transcript, domain_tag, initial_claim);
    let mut running_claim = initial_claim;
    let mut rounds = Vec::with_capacity(oracle.num_rounds());
    let mut challenges = Vec::with_capacity(oracle.num_rounds());

    for round in 0..oracle.num_rounds() {
        let degree = oracle.degree_bound();
        let points: Vec<K> = (0..=degree)
            .map(|value| K::from(F::from_u64(value as u64)))
            .collect();
        let evaluations = oracle.evals_at(&points);
        if evaluations.len() != degree + 1 {
            return Err(PiCcsError::ProtocolError(format!(
                "rectangular SumCheck round {round} returned {} evaluations, expected {}",
                evaluations.len(),
                degree + 1
            )));
        }
        let actual = evaluations[0] + evaluations[1];
        if actual != running_claim {
            return Err(PiCcsError::SumcheckError(format!(
                "rectangular SumCheck round {round} invariant failed"
            )));
        }
        let coefficients = crate::sumcheck::interpolate_from_evals(&points, &evaluations);
        if coefficients.len() != degree + 1 {
            return Err(PiCcsError::ProtocolError(
                "rectangular SumCheck coefficient width changed".into(),
            ));
        }
        transcript.append_fields_raw(&crate::sumcheck::round_coeff_fields(&coefficients));
        let sampled = transcript.challenge_fields_raw(2);
        let challenge = neo_math::from_complex(sampled[0], sampled[1]);
        running_claim = crate::sumcheck::poly_eval_k(&coefficients, challenge);
        oracle.fold(challenge);
        rounds.push(coefficients);
        challenges.push(challenge);
    }

    Ok(PhaseProof {
        rounds,
        challenges,
        final_claim: running_claim,
    })
}

pub(crate) fn verify_phase(
    transcript: &mut Poseidon2Transcript,
    domain_tag: u64,
    degree: usize,
    expected_rounds: usize,
    initial_claim: K,
    rounds: &[Vec<K>],
) -> Result<(Vec<K>, K), PiCcsError> {
    if rounds.len() != expected_rounds {
        return Err(PiCcsError::InvalidInput(format!(
            "rectangular SumCheck has {} rounds, expected {expected_rounds}",
            rounds.len()
        )));
    }
    if rounds.iter().any(|round| round.len() != degree + 1) {
        return Err(PiCcsError::InvalidInput(format!(
            "rectangular SumCheck rounds must each have {} coefficients",
            degree + 1
        )));
    }
    append_phase_prolog(transcript, domain_tag, initial_claim);
    let (challenges, final_claim, valid) =
        crate::sumcheck::verify_sumcheck_rounds_poseidon_v3(transcript, degree, initial_claim, rounds);
    if !valid {
        return Err(PiCcsError::SumcheckError(
            "rectangular SumCheck verifier rejected a round".into(),
        ));
    }
    Ok((challenges, final_claim))
}

pub fn assemble_proof(
    challenges: Challenges,
    fe_initial: K,
    fe: PhaseProof,
    nc: PhaseProof,
    fold_digest: [u8; 32],
) -> PiCcsProof {
    let mut proof = PiCcsProof::new(fe.rounds, Some(fe_initial));
    proof.variant = PiCcsProofVariant::PaperRectangularV1;
    proof.sumcheck_challenges = fe.challenges;
    proof.sumcheck_rounds_nc = nc.rounds;
    proof.sc_initial_sum_nc = Some(K::ZERO);
    proof.sumcheck_challenges_nc = nc.challenges;
    proof.challenges_public = challenges;
    proof.sumcheck_final = fe.final_claim;
    proof.sumcheck_final_nc = nc.final_claim;
    proof.header_digest = fold_digest.to_vec();
    proof.canonicalize();
    proof
}

/// Shared verifier for the canonical rectangular proof. It derives every
/// challenge from the transcript and checks all redundant wire fields.
pub fn verify(
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    outputs: &[CeClaim<Cmt, F, K>],
    proof: &PiCcsProof,
) -> Result<bool, PiCcsError> {
    verify_with_binding(
        transcript,
        params,
        structure,
        fresh_claims,
        running_claims,
        outputs,
        proof,
        TranscriptBinding::claims(),
    )
}

/// Verify a canonical proof under a caller-recomputed public transcript
/// binding.
#[allow(clippy::too_many_arguments)]
pub fn verify_with_binding(
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    outputs: &[CeClaim<Cmt, F, K>],
    proof: &PiCcsProof,
    binding: TranscriptBinding,
) -> Result<bool, PiCcsError> {
    if proof.variant != PiCcsProofVariant::PaperRectangularV1 {
        return Err(PiCcsError::ProtocolError("expected PaperRectangularV1 proof".into()));
    }
    if proof._extra.is_some() {
        return Err(PiCcsError::ProtocolError(
            "PaperRectangularV1 does not permit extra proof data".into(),
        ));
    }
    let (dims, challenges) =
        bind_and_sample_with_binding(transcript, params, structure, fresh_claims, running_claims, binding)?;
    if proof.challenges_public != challenges {
        return Err(PiCcsError::ProtocolError(
            "stored rectangular public challenges do not match transcript replay".into(),
        ));
    }
    let initial_fe = fe_initial_claim(structure, &challenges, fresh_claims.len(), running_claims)?;
    if proof.sc_initial_sum != Some(initial_fe) || proof.sc_initial_sum_nc != Some(K::ZERO) {
        return Err(PiCcsError::SumcheckError("rectangular initial claim mismatch".into()));
    }

    let (row_point, final_fe) = verify_phase(
        transcript,
        utils::PI_CCS_SUMCHECK_FE_RAW_DOMAIN_TAG,
        dims.d_sc,
        dims.ell_n,
        initial_fe,
        &proof.sumcheck_rounds,
    )?;
    let (column_point, final_nc) = verify_phase(
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
            "stored rectangular SumCheck state does not match transcript replay".into(),
        ));
    }

    let fold_digest = transcript.digest32();
    if proof.header_digest.as_slice() != fold_digest {
        return Err(PiCcsError::ProtocolError(
            "rectangular proof digest does not match transcript replay".into(),
        ));
    }
    if outputs
        .iter()
        .any(|output| output.fold_digest != fold_digest)
    {
        return Err(PiCcsError::ProtocolError(
            "rectangular output digest does not match transcript replay".into(),
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
    let expected_fe = fe_terminal(
        structure,
        &challenges,
        fresh_claims.len(),
        prior_point,
        &row_point,
        outputs,
    )?;
    let expected_nc = nc_terminal::<F>(params, &challenges, fresh_claims.len(), &column_point, outputs)?;
    Ok(final_fe == expected_fe && final_nc == expected_nc)
}
