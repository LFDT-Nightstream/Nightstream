//! Optimized-side protocol flow for SuperNeo v1.2 PiCCS.
//!
//! This file independently implements the Lean-owned transcript schedule. It
//! does not call the PaperExact transcript, SumCheck driver, or proof assembly.

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsClaim, CcsStructure, CeClaim};
use neo_math::{KExtensions, D, F, K};
use neo_params::NeoParams;
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;

use crate::engines::pi_ccs_joint::{
    build_joint_dims, JointDims, ProtocolTrace, TraceEvent, ALPHA_TAG, GAMMA_TAG, ROUND_CHALLENGE_TAG,
};
use crate::engines::pi_ccs_protocol::{Challenges, PiCcsProof};
use crate::error::PiCcsError;

/// One SuperNeo v1.2 output opening. Pad (`Eval_K`) is separate from the
/// genuine CCS-matrix family (`Eval_A`).
pub type V1_2OutputOpening = neo_ccs::V1_2Evaluations<K>;

/// Fallible evaluator boundary for the selected one-joint SumCheck.
///
/// The canonical driver below owns every transcript action and checks each
/// returned round message. A device backend can only evaluate and fold its
/// private multilinear tables.
pub trait PaperJointRoundOracle {
    fn evals_at(&mut self, points: &[K]) -> Result<Vec<K>, PiCcsError>;
    fn num_rounds(&self) -> usize;
    fn degree_bound(&self) -> usize;
    fn fold(&mut self, challenge: K) -> Result<(), PiCcsError>;

    /// Return canonical ring openings at the completed SumCheck point when
    /// the evaluator can produce them without rebuilding its private state.
    /// The outer prover still validates the terminal claim and owns every
    /// transcript action. `None` selects the canonical host computation.
    fn output_openings(&mut self, _point: &[K]) -> Result<Option<Vec<V1_2OutputOpening>>, PiCcsError> {
        Ok(None)
    }
}

const DOMAIN_TAG: &[u64] = &[
    78, 105, 103, 104, 116, 115, 116, 114, 101, 97, 109, 47, 83, 117, 112, 101, 114, 78, 101, 111, 47, 80, 105, 67, 67,
    83, 47, 100, 105, 103, 101, 115, 116, 45, 111, 110, 108, 121, 47, 118, 49, 95, 49,
];

fn k_fields(output: &mut Vec<F>, value: K) {
    output.extend_from_slice(&value.as_coeffs());
}

fn append(transcript: &mut Poseidon2Transcript, trace: &mut ProtocolTrace, fields: Vec<F>) {
    transcript.absorb_v1_2(&fields);
    trace.events.push(TraceEvent::Absorb(fields));
}

fn append_block(transcript: &mut Poseidon2Transcript, trace: &mut ProtocolTrace, fields: Vec<F>) {
    transcript.absorb_block_v1_2(&fields);
    let mut framed = Vec::with_capacity(fields.len() + 1);
    framed.push(F::from_u64(fields.len() as u64));
    framed.extend(fields);
    trace.events.push(TraceEvent::Absorb(framed));
}

fn squeeze(transcript: &mut Poseidon2Transcript, trace: &mut ProtocolTrace, label: u64, index: Option<usize>) -> K {
    let fields = match index {
        Some(index) => vec![F::from_u64(label), F::from_u64(index as u64)],
        None => vec![F::from_u64(label)],
    };
    append(transcript, trace, fields);
    let sampled = transcript.squeeze_extension_v1_2();
    let value = neo_math::from_complex(sampled[0], sampled[1]);
    trace
        .events
        .push(TraceEvent::Challenge { label, index, value });
    value
}

fn check_commitment(commitment: &Cmt, params: &NeoParams) -> Result<(), PiCcsError> {
    if commitment.d != D
        || commitment.kappa != params.kappa as usize
        || commitment.data.len() != D * params.kappa as usize
    {
        return Err(PiCcsError::InvalidInput(
            "PiCCS v1_2 commitment does not have the fixed Ajtai shape".into(),
        ));
    }
    Ok(())
}

fn commitment_fields(commitment: &Cmt, params: &NeoParams) -> Result<Vec<F>, PiCcsError> {
    check_commitment(commitment, params)?;
    Ok(commitment.data.clone())
}

fn validate_selected_inputs(
    structure: &CcsStructure<F>,
    fresh: &[CcsClaim<Cmt, F>],
    running: &[CeClaim<Cmt, F, K>],
    dims: JointDims,
) -> Result<(), PiCcsError> {
    for (index, claim) in fresh.iter().enumerate() {
        if claim.m_in > structure.m || claim.x.len() != claim.m_in || claim.m_in % D != 0 {
            return Err(PiCcsError::InvalidInput(format!(
                "optimized fresh claim {index} is not a complete whole-ring public input"
            )));
        }
    }
    for (index, claim) in running.iter().enumerate() {
        if claim.m_in > structure.m
            || claim.m_in % D != 0
            || claim.X.rows() != D
            || claim.X.cols() != neo_ccs::superneo_public_x_cols(claim.m_in)
            || claim.eval_k.len() != D.next_power_of_two()
            || claim.eval_a.len() != dims.matrix_count
        {
            return Err(PiCcsError::InvalidInput(format!(
                "optimized running claim {index} does not have the selected paper shape"
            )));
        }
        if claim.eval_k.iter().skip(D).any(|&value| value != K::ZERO) {
            return Err(PiCcsError::InvalidInput(format!(
                "optimized running claim {index} Eval_K is not canonical"
            )));
        }
        for (matrix, coefficients) in claim.eval_a.iter().enumerate() {
            if coefficients.len() != D.next_power_of_two() || coefficients.iter().skip(D).any(|&value| value != K::ZERO)
            {
                return Err(PiCcsError::InvalidInput(format!(
                    "optimized running claim {index} Eval_A matrix {matrix} is not canonical"
                )));
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn bind_and_sample_with_trace(
    transcript: &mut Poseidon2Transcript,
    trace: &mut ProtocolTrace,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh: &[CcsClaim<Cmt, F>],
    running: &[CeClaim<Cmt, F, K>],
) -> Result<(JointDims, Challenges), PiCcsError> {
    let dims = build_joint_dims(params, structure, fresh.len(), running.len())?;
    validate_selected_inputs(structure, fresh, running, dims)?;
    // The fresh public input carries the prior digest that names the running
    // claims; their commitments still need the shape fresh commitments get.
    for claim in running {
        check_commitment(&claim.c, params)?;
    }
    if fresh.iter().any(|claim| claim.adv.is_some()) || running.iter().any(|claim| claim.adv.is_some()) {
        return Err(PiCcsError::InvalidInput(
            "PiCCS v1_2 does not bind auxiliary lane commitments".into(),
        ));
    }
    if fresh.is_empty() {
        return Err(PiCcsError::InvalidInput(
            "PiCCS v1_2 digest-only statement requires a fresh claim".into(),
        ));
    }
    let prior_point = running
        .first()
        .map_or_else(|| vec![K::ZERO; dims.variables], |claim| claim.r.clone());
    if prior_point.len() != dims.variables || running.iter().any(|claim| claim.r != prior_point) {
        return Err(PiCcsError::InvalidInput(
            "PiCCS v1_2 running claims must share the complete prior point".into(),
        ));
    }

    transcript.reset_v1_2();
    append(
        transcript,
        trace,
        DOMAIN_TAG.iter().map(|&word| F::from_u64(word)).collect(),
    );

    append_block(
        transcript,
        trace,
        neo_transcript::prior_digest_v1_2(&fresh[0].x).to_vec(),
    );
    for claim in fresh {
        append_block(transcript, trace, commitment_fields(&claim.c, params)?);
        append_block(transcript, trace, claim.x.clone());
    }

    let alpha: Vec<K> = (0..dims.variables)
        .map(|index| squeeze(transcript, trace, ALPHA_TAG, Some(index)))
        .collect();
    let gamma = squeeze(transcript, trace, GAMMA_TAG, None);
    trace.alpha = alpha.clone();
    trace.gamma = gamma;
    trace.pre_sumcheck_state = transcript.state();
    Ok((dims, Challenges::new(alpha, gamma)))
}

pub fn prove_phase<O: PaperJointRoundOracle + ?Sized>(
    transcript: &mut Poseidon2Transcript,
    trace: &mut ProtocolTrace,
    initial_claim: K,
    oracle: &mut O,
) -> Result<(Vec<Vec<K>>, Vec<K>, K), PiCcsError> {
    trace.initial_claim = initial_claim;
    let mut claim = initial_claim;
    let mut rounds = Vec::with_capacity(oracle.num_rounds());
    let mut challenges = Vec::with_capacity(oracle.num_rounds());
    for round in 0..oracle.num_rounds() {
        let points: Vec<K> = (0..=oracle.degree_bound())
            .map(|value| K::from(F::from_u64(value as u64)))
            .collect();
        let evaluations = oracle.evals_at(&points)?;
        if evaluations.len() != points.len() || evaluations[0] + evaluations[1] != claim {
            let actual =
                evaluations.first().copied().unwrap_or(K::ZERO) + evaluations.get(1).copied().unwrap_or(K::ZERO);
            return Err(PiCcsError::SumcheckError(format!(
                "optimized joint SumCheck invariant failed at round {round}: expected {claim:?}, got {actual:?}"
            )));
        }
        let coefficients = crate::sumcheck::interpolate_from_evals(&points, &evaluations);
        let mut fields = vec![F::from_u64(round as u64)];
        for &coefficient in &coefficients {
            k_fields(&mut fields, coefficient);
        }
        append_block(transcript, trace, fields);
        let challenge = squeeze(transcript, trace, ROUND_CHALLENGE_TAG, Some(round));
        claim = crate::sumcheck::poly_eval_k(&coefficients, challenge);
        oracle.fold(challenge)?;
        rounds.push(coefficients);
        challenges.push(challenge);
        trace.round_states.push(transcript.state());
        trace.round_claims.push(claim);
    }
    trace.rounds = rounds.clone();
    trace.round_challenges = challenges.clone();
    trace.terminal_claim = claim;
    Ok((rounds, challenges, claim))
}

fn verify_phase(
    transcript: &mut Poseidon2Transcript,
    trace: &mut ProtocolTrace,
    dims: JointDims,
    initial_claim: K,
    rounds: &[Vec<K>],
) -> Result<(Vec<K>, K), PiCcsError> {
    if rounds.len() != dims.variables || rounds.iter().any(|round| round.len() != dims.degree + 1) {
        return Err(PiCcsError::InvalidInput(
            "optimized joint SumCheck message shape mismatch".into(),
        ));
    }
    trace.initial_claim = initial_claim;
    let mut claim = initial_claim;
    let mut challenges = Vec::with_capacity(dims.variables);
    for (round_index, coefficients) in rounds.iter().enumerate() {
        if coefficients[0] + crate::sumcheck::poly_eval_k(coefficients, K::ONE) != claim {
            return Err(PiCcsError::SumcheckError(format!(
                "optimized verifier rejected SumCheck round {round_index}"
            )));
        }
        let mut fields = vec![F::from_u64(round_index as u64)];
        for &coefficient in coefficients {
            k_fields(&mut fields, coefficient);
        }
        append_block(transcript, trace, fields);
        let challenge = squeeze(transcript, trace, ROUND_CHALLENGE_TAG, Some(round_index));
        claim = crate::sumcheck::poly_eval_k(coefficients, challenge);
        challenges.push(challenge);
        trace.round_states.push(transcript.state());
        trace.round_claims.push(claim);
    }
    trace.rounds = rounds.to_vec();
    trace.round_challenges = challenges.clone();
    trace.terminal_claim = claim;
    Ok((challenges, claim))
}

/// Encode the complete Section 7.3 output message in
/// source/`Eval_K`/`Eval_A` order. Each quadratic-extension value is low limb
/// first.
pub fn output_message_fields(outputs: &[CeClaim<Cmt, F, K>], dims: JointDims) -> Result<Vec<F>, PiCcsError> {
    let mut fields = Vec::with_capacity(outputs.len() * (dims.matrix_count + 1) * D * 2);
    for output in outputs {
        if output.eval_k.len() < D || output.eval_a.len() != dims.matrix_count {
            return Err(PiCcsError::InvalidInput(
                "optimized output v1_2 families are incomplete".into(),
            ));
        }
        for coefficient in 0..D {
            k_fields(&mut fields, output.eval_k[coefficient]);
        }
        for matrix in 0..dims.matrix_count {
            for coefficient in 0..D {
                k_fields(
                    &mut fields,
                    *output.eval_a[matrix]
                        .get(coefficient)
                        .ok_or_else(|| PiCcsError::InvalidInput("optimized ring output is incomplete".into()))?,
                );
            }
        }
    }
    Ok(fields)
}

pub fn absorb_outputs(
    transcript: &mut Poseidon2Transcript,
    trace: &mut ProtocolTrace,
    outputs: &[CeClaim<Cmt, F, K>],
    fresh_count: usize,
    dims: JointDims,
) -> Result<[u8; 32], PiCcsError> {
    let _ = fresh_count;
    let fields = output_message_fields(outputs, dims)?;
    append_block(transcript, trace, fields);
    trace.outgoing_state = transcript.state();
    let digest = transcript.state_prefix_v1_2();
    trace.final_digest = digest;
    Ok(digest)
}

pub fn assemble_proof(rounds: Vec<Vec<K>>) -> PiCcsProof {
    let mut proof = PiCcsProof::new(rounds);
    proof.canonicalize();
    proof
}

fn validate_outputs(
    structure: &CcsStructure<F>,
    fresh: &[CcsClaim<Cmt, F>],
    running: &[CeClaim<Cmt, F, K>],
    outputs: &[CeClaim<Cmt, F, K>],
    point: &[K],
    dims: JointDims,
) -> Result<(), PiCcsError> {
    if outputs.len() != fresh.len() + running.len() {
        return Err(PiCcsError::InvalidInput(
            "optimized output source count mismatch".into(),
        ));
    }
    for output in outputs {
        if output.r != point
            || output.X.rows() != D
            || output.X.cols() != neo_ccs::superneo_public_x_cols(output.m_in)
            || output.eval_k.len() != D.next_power_of_two()
            || output.eval_a.len() != dims.matrix_count
        {
            return Err(PiCcsError::InvalidInput(
                "optimized output does not have the one-joint shape".into(),
            ));
        }
        if output.eval_k.iter().skip(D).any(|&value| value != K::ZERO) {
            return Err(PiCcsError::InvalidInput(
                "optimized Eval_K output is not canonical".into(),
            ));
        }
        for row in &output.eval_a {
            if row.len() != D.next_power_of_two() || row.iter().skip(D).any(|&value| value != K::ZERO) {
                return Err(PiCcsError::InvalidInput("optimized output is not canonical".into()));
            }
        }
    }
    for (claim, output) in fresh.iter().zip(outputs) {
        if claim.c != output.c || claim.m_in != output.m_in || claim.adv != output.adv {
            return Err(PiCcsError::ProtocolError(
                "fresh output changed its public instance".into(),
            ));
        }
        if claim.m_in % D != 0 || claim.x.len() != claim.m_in {
            return Err(PiCcsError::InvalidInput(
                "fresh public input is not whole-ring aligned".into(),
            ));
        }
        for (coordinate, &value) in claim.x.iter().enumerate() {
            if output.X[(coordinate % D, coordinate / D)] != value {
                return Err(PiCcsError::ProtocolError(
                    "fresh output changed a public input coordinate".into(),
                ));
            }
        }
    }
    for (claim, output) in running.iter().zip(outputs.iter().skip(fresh.len())) {
        if claim.c != output.c || claim.X != output.X || claim.m_in != output.m_in || claim.adv != output.adv {
            return Err(PiCcsError::ProtocolError(
                "carried output changed its public instance".into(),
            ));
        }
    }
    let _ = structure;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_with_trace(
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh: &[CcsClaim<Cmt, F>],
    running: &[CeClaim<Cmt, F, K>],
    outputs: &[CeClaim<Cmt, F, K>],
    proof: &PiCcsProof,
) -> Result<(bool, ProtocolTrace), PiCcsError> {
    let mut trace = ProtocolTrace::default();
    let (dims, challenges) = bind_and_sample_with_trace(transcript, &mut trace, params, structure, fresh, running)?;
    let prior_point = crate::engines::utils::shared_me_input_r(running, dims.variables)?;
    let initial =
        crate::engines::optimized_engine::paper_joint::initial_claim(structure, &challenges, fresh.len(), running)?;
    let (point, final_claim) = verify_phase(transcript, &mut trace, dims, initial, &proof.sumcheck_rounds)?;
    validate_outputs(structure, fresh, running, outputs, &point, dims)?;
    let terminal = crate::engines::optimized_engine::paper_joint::terminal_components::<F>(
        structure,
        params,
        &challenges,
        fresh.len(),
        prior_point,
        &point,
        outputs,
    )?;
    let expected = terminal.terminal;
    trace.terminal_components = terminal;
    let digest = absorb_outputs(transcript, &mut trace, outputs, fresh.len(), dims)?;
    if outputs.iter().any(|output| output.fold_digest != digest) {
        return Err(PiCcsError::ProtocolError(
            "optimized output digest does not match transcript replay".into(),
        ));
    }
    Ok((final_claim == expected, trace))
}

/// Verify PiCCS with the digest-only transcript binding.
///
/// The [caller contract](crate::engines::PiCcsEngine::verify) applies.
pub fn verify(
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh: &[CcsClaim<Cmt, F>],
    running: &[CeClaim<Cmt, F, K>],
    outputs: &[CeClaim<Cmt, F, K>],
    proof: &PiCcsProof,
) -> Result<bool, PiCcsError> {
    Ok(verify_with_trace(transcript, params, structure, fresh, running, outputs, proof)?.0)
}
