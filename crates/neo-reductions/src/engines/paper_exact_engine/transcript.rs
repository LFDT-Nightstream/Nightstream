//! Independent PaperExact Fiat--Shamir and SumCheck schedule.
//!
//! This is a direct Rust transcription of the Lean-owned v1_1 schedule. It
//! does not call the optimized transcript or proof-assembly implementation.
//! The canonical statement serializer is shared because it is protocol input,
//! not an alternative prover computation.

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsClaim, CcsStructure, CeClaim};
use neo_math::{KExtensions, D, F, K};
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;

use crate::engines::pi_ccs_joint::{JointDims, ProtocolTrace, TraceEvent};
use crate::engines::pi_ccs_protocol::{Challenges, PiCcsProof};
use crate::error::PiCcsError;
use crate::sumcheck::RoundOracle;

const ALPHA_TAG: u64 = 1;
const GAMMA_TAG: u64 = 2;
const ROUND_CHALLENGE_TAG: u64 = 3;
const DOMAIN_TAG: &[u64] = &[
    78, 105, 103, 104, 116, 115, 116, 114, 101, 97, 109, 47, 83, 117, 112, 101, 114, 78, 101, 111, 47, 80, 105, 67, 67,
    83, 47, 100, 105, 103, 101, 115, 116, 45, 111, 110, 108, 121, 47, 118, 49, 95, 49,
];

/// The one Lean-owned PaperExact statement binding.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PaperTranscriptBinding;

impl PaperTranscriptBinding {
    pub(crate) const fn digest_only() -> Self {
        Self
    }
}

fn k_fields(output: &mut Vec<F>, value: K) {
    output.extend_from_slice(&value.as_coeffs());
}

fn append(transcript: &mut Poseidon2Transcript, trace: &mut ProtocolTrace, fields: Vec<F>) {
    transcript.absorb_v1_1(&fields);
    trace.events.push(TraceEvent::Absorb(fields));
}

fn append_block(transcript: &mut Poseidon2Transcript, trace: &mut ProtocolTrace, fields: Vec<F>) {
    transcript.absorb_block_v1_1(&fields);
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
    let sampled = transcript.squeeze_extension_v1_1();
    let value = neo_math::from_complex(sampled[0], sampled[1]);
    trace
        .events
        .push(TraceEvent::Challenge { label, index, value });
    value
}

fn commitment_fields(commitment: &Cmt) -> Result<Vec<F>, PiCcsError> {
    let kappa = neo_params::nightstream_goldilocks_k16::KAPPA as usize;
    if commitment.d != D || commitment.kappa != kappa || commitment.data.len() != D * kappa {
        return Err(PiCcsError::InvalidInput(
            "PaperExact v1_1 commitment does not have the fixed Ajtai shape".into(),
        ));
    }
    Ok(commitment.data.clone())
}

fn paper_poly_eval(coefficients: &[K], point: K) -> K {
    coefficients
        .iter()
        .rev()
        .copied()
        .fold(K::ZERO, |value, coefficient| value * point + coefficient)
}

fn paper_interpolate(points: &[K], evaluations: &[K]) -> Vec<K> {
    assert_eq!(
        points.len(),
        evaluations.len(),
        "PaperExact interpolation shape mismatch"
    );
    let count = points.len();
    let mut coefficients = vec![K::ZERO; count];
    for index in 0..count {
        let mut numerator = vec![K::ZERO; count];
        numerator[0] = K::ONE;
        let mut degree = 0;
        for other in 0..count {
            if index == other {
                continue;
            }
            let mut next = vec![K::ZERO; count];
            for coefficient in 0..=degree {
                next[coefficient] -= points[other] * numerator[coefficient];
                next[coefficient + 1] += numerator[coefficient];
            }
            numerator = next;
            degree += 1;
        }
        let mut denominator = K::ONE;
        for other in 0..count {
            if index != other {
                denominator *= points[index] - points[other];
            }
        }
        let scale = evaluations[index] * denominator.inv();
        for coefficient in 0..=degree {
            coefficients[coefficient] += scale * numerator[coefficient];
        }
    }
    coefficients
}

pub(super) fn bind_and_sample(
    transcript: &mut Poseidon2Transcript,
    trace: &mut ProtocolTrace,
    _structure: &CcsStructure<F>,
    fresh: &[CcsClaim<Cmt, F>],
    running: &[CeClaim<Cmt, F, K>],
    dims: JointDims,
    _binding: PaperTranscriptBinding,
) -> Result<Challenges, PiCcsError> {
    if fresh.is_empty() || fresh[0].x.len() < 5 {
        return Err(PiCcsError::InvalidInput(
            "PaperExact v1_1 digest-only statement requires prior-digest slots in the first fresh claim".into(),
        ));
    }
    let prior_point = running
        .first()
        .map_or_else(|| vec![K::ZERO; dims.variables], |claim| claim.r.clone());
    if prior_point.len() != dims.variables || running.iter().any(|claim| claim.r != prior_point) {
        return Err(PiCcsError::InvalidInput(
            "PaperExact v1_1 running claims must share the complete prior point".into(),
        ));
    }

    transcript.reset_v1_1();
    append(
        transcript,
        trace,
        DOMAIN_TAG.iter().map(|&word| F::from_u64(word)).collect(),
    );

    append_block(transcript, trace, fresh[0].x[1..5].to_vec());
    for claim in fresh {
        append_block(transcript, trace, commitment_fields(&claim.c)?);
        append_block(transcript, trace, claim.x.clone());
    }

    let alpha: Vec<K> = (0..dims.variables)
        .map(|index| squeeze(transcript, trace, ALPHA_TAG, Some(index)))
        .collect();
    let gamma = squeeze(transcript, trace, GAMMA_TAG, None);
    trace.alpha = alpha.clone();
    trace.gamma = gamma;
    trace.pre_sumcheck_state = transcript.state();
    Ok(Challenges::new(alpha, gamma))
}

pub(super) fn prove_sumcheck<O: RoundOracle>(
    transcript: &mut Poseidon2Transcript,
    trace: &mut ProtocolTrace,
    initial_claim: K,
    oracle: &mut O,
) -> Result<(Vec<Vec<K>>, Vec<K>, K), PiCcsError> {
    trace.initial_claim = initial_claim;
    let mut running_claim = initial_claim;
    let mut rounds = Vec::with_capacity(oracle.num_rounds());
    let mut challenges = Vec::with_capacity(oracle.num_rounds());
    for round in 0..oracle.num_rounds() {
        let degree = oracle.degree_bound();
        let points: Vec<K> = (0..=degree)
            .map(|value| K::from(F::from_u64(value as u64)))
            .collect();
        let evaluations = oracle.evals_at(&points);
        if evaluations.len() != degree + 1 || evaluations[0] + evaluations[1] != running_claim {
            return Err(PiCcsError::SumcheckError(format!(
                "PaperExact joint SumCheck invariant failed at round {round}"
            )));
        }
        let coefficients = paper_interpolate(&points, &evaluations);
        let mut fields = vec![F::from_u64(round as u64)];
        for &coefficient in &coefficients {
            k_fields(&mut fields, coefficient);
        }
        append_block(transcript, trace, fields);
        let challenge = squeeze(transcript, trace, ROUND_CHALLENGE_TAG, Some(round));
        running_claim = paper_poly_eval(&coefficients, challenge);
        oracle.fold(challenge);
        rounds.push(coefficients);
        challenges.push(challenge);
        trace.round_states.push(transcript.state());
        trace.round_claims.push(running_claim);
    }
    trace.rounds = rounds.clone();
    trace.round_challenges = challenges.clone();
    trace.terminal_claim = running_claim;
    Ok((rounds, challenges, running_claim))
}

pub(super) fn verify_sumcheck(
    transcript: &mut Poseidon2Transcript,
    trace: &mut ProtocolTrace,
    dims: JointDims,
    initial_claim: K,
    rounds: &[Vec<K>],
) -> Result<(Vec<K>, K), PiCcsError> {
    if rounds.len() != dims.variables || rounds.iter().any(|round| round.len() != dims.degree + 1) {
        return Err(PiCcsError::InvalidInput(
            "PaperExact joint SumCheck message shape mismatch".into(),
        ));
    }
    trace.initial_claim = initial_claim;
    let mut claim = initial_claim;
    let mut challenges = Vec::with_capacity(dims.variables);
    for (round_index, coefficients) in rounds.iter().enumerate() {
        if coefficients[0] + paper_poly_eval(coefficients, K::ONE) != claim {
            return Err(PiCcsError::SumcheckError(format!(
                "PaperExact verifier rejected SumCheck round {round_index}"
            )));
        }
        let mut fields = vec![F::from_u64(round_index as u64)];
        for &coefficient in coefficients {
            k_fields(&mut fields, coefficient);
        }
        append_block(transcript, trace, fields);
        let challenge = squeeze(transcript, trace, ROUND_CHALLENGE_TAG, Some(round_index));
        claim = paper_poly_eval(coefficients, challenge);
        challenges.push(challenge);
        trace.round_states.push(transcript.state());
        trace.round_claims.push(claim);
    }
    trace.rounds = rounds.to_vec();
    trace.round_challenges = challenges.clone();
    trace.terminal_claim = claim;
    Ok((challenges, claim))
}

pub(super) fn absorb_outputs(
    transcript: &mut Poseidon2Transcript,
    trace: &mut ProtocolTrace,
    outputs: &[CeClaim<Cmt, F, K>],
    fresh_count: usize,
    dims: JointDims,
) -> Result<[u8; 32], PiCcsError> {
    if outputs.len() < fresh_count {
        return Err(PiCcsError::InvalidInput(
            "PaperExact output source count mismatch".into(),
        ));
    }
    let _ = fresh_count;
    let mut fields = Vec::new();
    // SuperNeo Section 7.3 sends y' after the SumCheck. Validate the complete
    // message shape here. PiRLC binds its canonical digest before sampling
    // rho; PiCCS adds no extra non-paper output equation.
    for output in outputs {
        if output.eval_k.len() < neo_math::D || output.eval_a.len() != dims.matrix_count {
            return Err(PiCcsError::InvalidInput(
                "PaperExact output v1_1 families are incomplete".into(),
            ));
        }
        for coefficient in 0..neo_math::D {
            k_fields(&mut fields, output.eval_k[coefficient]);
        }
        for matrix in 0..dims.matrix_count {
            for coefficient in 0..neo_math::D {
                k_fields(
                    &mut fields,
                    *output.eval_a[matrix]
                        .get(coefficient)
                        .ok_or_else(|| PiCcsError::InvalidInput("PaperExact ring output is incomplete".into()))?,
                );
            }
        }
    }
    append_block(transcript, trace, fields);
    trace.outgoing_state = transcript.state();
    let digest = transcript.state_prefix_v1_1();
    trace.final_digest = digest;
    Ok(digest)
}

pub(super) fn assemble_proof(rounds: Vec<Vec<K>>) -> PiCcsProof {
    PiCcsProof::new(rounds)
}

/// Encode the one-joint proof without using the production codec.
///
/// The duplicate constants and field order are intentional. PaperExact must
/// detect a production codec change that is not made in the reference codec.
#[doc(hidden)]
pub fn encode_proof(proof: &PiCcsProof) -> Result<Vec<u8>, PiCcsError> {
    const PROOF_TAG: u64 = 1102;
    const CODEC_VERSION: u64 = 1;

    let coefficient_count = proof.sumcheck_rounds.first().map_or(0, Vec::len);
    if proof
        .sumcheck_rounds
        .iter()
        .any(|round| round.len() != coefficient_count)
    {
        return Err(PiCcsError::InvalidInput(
            "PaperExact codec requires one fixed round degree".into(),
        ));
    }

    fn push_u64(output: &mut Vec<u8>, value: u64) {
        output.extend_from_slice(&value.to_le_bytes());
    }

    let mut output = Vec::with_capacity(4 * 8 + proof.sumcheck_rounds.len() * coefficient_count * 16);
    push_u64(&mut output, PROOF_TAG);
    push_u64(&mut output, CODEC_VERSION);
    push_u64(&mut output, proof.sumcheck_rounds.len() as u64);
    push_u64(&mut output, coefficient_count as u64);
    for round in &proof.sumcheck_rounds {
        for &coefficient in round {
            let (low, high) = coefficient.to_limbs_u64();
            push_u64(&mut output, low);
            push_u64(&mut output, high);
        }
    }
    Ok(output)
}
