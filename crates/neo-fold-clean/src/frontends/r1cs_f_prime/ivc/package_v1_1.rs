//! PiCCS proof-message bridge to the Lean-emitted v1_1 package.
//!
//! Owns only the field-for-field conversion from native prover messages to
//! the package input types, canonical lifecycle serialization, and the
//! verifier-owned package input construction. It does not define, load, prove,
//! or verify a relation.

use neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash;
use neo_math::{KExtensions, F, K};
use nightstream_fprime::{
    PackageError, PiCcsV1_1OutputEvaluations, PiCcsV1_1PackageInputs, PiCcsV1_1VerifierContext,
    PI_CCS_V1_1_COEFFICIENT_COUNT, PI_CCS_V1_1_FRESH_COMMITMENT_WORDS, PI_CCS_V1_1_MATRIX_COUNT,
    PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS, PI_CCS_V1_1_ROUND_COEFFICIENT_COUNT, PI_CCS_V1_1_ROUND_COUNT,
    PI_CCS_V1_1_SOURCE_COUNT, PI_CCS_V1_1_STATE_PREIMAGE_WORDS,
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use thiserror::Error;

use crate::paper::pi_ccs;
use crate::paper::relations::{CcsClaim, CeClaim};

const STATE_DOMAIN_TAG: [u64; 23] = [
    72, 121, 112, 101, 114, 78, 111, 118, 97, 47, 78, 73, 86, 67, 47, 115, 116, 97, 116, 101, 47, 118, 49,
];
const COMMITMENT_WIDTH: usize = PI_CCS_V1_1_FRESH_COMMITMENT_WORDS / PI_CCS_V1_1_COEFFICIENT_COUNT;

#[derive(Debug, Error)]
pub enum PiCcsV1_1PackageBridgeError {
    #[error("PiCCS v1_1 package bridge: {0}")]
    Shape(&'static str),
    #[error(transparent)]
    Package(#[from] PackageError),
}

/// Exact PiCCS-owned part of one Lean package assignment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiCcsV1_1ProofInputs {
    fresh_commitment: Vec<u64>,
    round_messages: Vec<Vec<[u64; 2]>>,
    output_evaluations: PiCcsV1_1OutputEvaluations,
}

impl PiCcsV1_1ProofInputs {
    /// Convert one native v1_1 PiCCS proof without changing its order.
    pub fn from_proof(fresh: &[CcsClaim], proof: &pi_ccs::Proof) -> Result<Self, PiCcsV1_1PackageBridgeError> {
        if fresh.len() != 1 {
            return Err(PiCcsV1_1PackageBridgeError::Shape("fresh source count"));
        }
        if fresh[0].c.d != PI_CCS_V1_1_COEFFICIENT_COUNT
            || fresh[0].c.kappa != COMMITMENT_WIDTH
            || fresh[0].c.data.len() != PI_CCS_V1_1_FRESH_COMMITMENT_WORDS
        {
            return Err(PiCcsV1_1PackageBridgeError::Shape("fresh commitment width"));
        }
        if proof.sumcheck.sumcheck_rounds.len() != PI_CCS_V1_1_ROUND_COUNT
            || proof
                .sumcheck
                .sumcheck_rounds
                .iter()
                .any(|round| round.len() != PI_CCS_V1_1_ROUND_COEFFICIENT_COUNT)
        {
            return Err(PiCcsV1_1PackageBridgeError::Shape("round messages"));
        }
        if proof.outputs.len() != PI_CCS_V1_1_SOURCE_COUNT {
            return Err(PiCcsV1_1PackageBridgeError::Shape("output source count"));
        }

        let fresh_commitment = fresh[0]
            .c
            .data
            .iter()
            .map(|value| value.as_canonical_u64())
            .collect();
        let round_messages = proof
            .sumcheck
            .sumcheck_rounds
            .iter()
            .map(|round| round.iter().copied().map(extension_words).collect())
            .collect();

        let mut eval_k = Vec::with_capacity(PI_CCS_V1_1_SOURCE_COUNT);
        let mut eval_a = Vec::with_capacity(PI_CCS_V1_1_SOURCE_COUNT);
        for output in &proof.outputs {
            validate_family(&output.eval_k)?;
            if output.eval_a.len() != PI_CCS_V1_1_MATRIX_COUNT {
                return Err(PiCcsV1_1PackageBridgeError::Shape("Eval_A matrix count"));
            }
            let source_eval_k = output.eval_k[..PI_CCS_V1_1_COEFFICIENT_COUNT]
                .iter()
                .copied()
                .map(extension_words)
                .collect();
            let mut source_eval_a = Vec::with_capacity(PI_CCS_V1_1_MATRIX_COUNT);
            for matrix in &output.eval_a {
                validate_family(matrix)?;
                source_eval_a.push(
                    matrix[..PI_CCS_V1_1_COEFFICIENT_COUNT]
                        .iter()
                        .copied()
                        .map(extension_words)
                        .collect(),
                );
            }
            eval_k.push(source_eval_k);
            eval_a.push(source_eval_a);
        }

        Ok(Self {
            fresh_commitment,
            round_messages,
            output_evaluations: PiCcsV1_1OutputEvaluations::new(eval_k, eval_a)?,
        })
    }

    pub fn fresh_commitment(&self) -> &[u64] {
        &self.fresh_commitment
    }

    pub fn round_messages(&self) -> &[Vec<[u64; 2]>] {
        &self.round_messages
    }

    pub fn output_evaluations(&self) -> &PiCcsV1_1OutputEvaluations {
        &self.output_evaluations
    }

    /// Add the lifecycle-owned fields after their canonical serializer has
    /// produced them.
    pub fn into_package_inputs(
        self,
        prior_preimage: Vec<u64>,
        output_preimage: Vec<u64>,
        prior_public_input: Vec<u64>,
        output_digest: [u64; 4],
        verifier_context: PiCcsV1_1VerifierContext,
    ) -> Result<PiCcsV1_1PackageInputs, PiCcsV1_1PackageBridgeError> {
        Ok(PiCcsV1_1PackageInputs::new(
            prior_preimage,
            output_preimage,
            self.fresh_commitment,
            self.round_messages,
            self.output_evaluations,
            prior_public_input,
            output_digest,
            verifier_context,
        )?)
    }
}

/// Serialize the exact Lean `HashPreimage` for the fixed v1_1 profile.
#[allow(clippy::too_many_arguments)]
pub fn serialize_pi_ccs_v1_1_state_preimage(
    verifier_key_digest: [F; 4],
    iteration: u64,
    z0: [F; 4],
    current: [F; 4],
    running: &[CeClaim],
    pc: u64,
) -> Result<Vec<u64>, PiCcsV1_1PackageBridgeError> {
    if iteration >= F::ORDER_U64 || pc >= F::ORDER_U64 {
        return Err(PiCcsV1_1PackageBridgeError::Shape(
            "iteration or program counter is not canonical",
        ));
    }
    if running.len() != 16 {
        return Err(PiCcsV1_1PackageBridgeError::Shape("running source count"));
    }
    let point = &running[0].r;
    if point.len() != PI_CCS_V1_1_ROUND_COUNT
        || running
            .iter()
            .any(|claim| claim.r.as_slice() != point.as_slice())
    {
        return Err(PiCcsV1_1PackageBridgeError::Shape("shared running point"));
    }

    let mut words = Vec::with_capacity(PI_CCS_V1_1_STATE_PREIMAGE_WORDS);
    words.extend_from_slice(&STATE_DOMAIN_TAG);
    push_block(&mut words, &verifier_key_digest.map(|value| value.as_canonical_u64()));
    words.push(iteration);
    push_block(&mut words, &z0.map(|value| value.as_canonical_u64()));
    push_block(&mut words, &current.map(|value| value.as_canonical_u64()));

    let mut point_words = Vec::with_capacity(2 * PI_CCS_V1_1_ROUND_COUNT);
    for value in point {
        point_words.extend_from_slice(&extension_words(*value));
    }
    push_block(&mut words, &point_words);

    for claim in running {
        if claim.c.d != PI_CCS_V1_1_COEFFICIENT_COUNT
            || claim.c.kappa != COMMITMENT_WIDTH
            || claim.c.data.len() != PI_CCS_V1_1_FRESH_COMMITMENT_WORDS
        {
            return Err(PiCcsV1_1PackageBridgeError::Shape("running commitment"));
        }
        let commitment: Vec<_> = claim
            .c
            .data
            .iter()
            .map(|value| value.as_canonical_u64())
            .collect();
        push_block(&mut words, &commitment);

        if claim.m_in != PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS
            || claim.X.rows() != PI_CCS_V1_1_COEFFICIENT_COUNT
            || claim.X.cols() != PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS / PI_CCS_V1_1_COEFFICIENT_COUNT
        {
            return Err(PiCcsV1_1PackageBridgeError::Shape("running public input"));
        }
        let public_input: Vec<_> = (0..PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS)
            .map(|index| {
                claim.X[(
                    index % PI_CCS_V1_1_COEFFICIENT_COUNT,
                    index / PI_CCS_V1_1_COEFFICIENT_COUNT,
                )]
                    .as_canonical_u64()
            })
            .collect();
        push_block(&mut words, &public_input);

        validate_family(&claim.eval_k)?;
        if claim.eval_a.len() != PI_CCS_V1_1_MATRIX_COUNT {
            return Err(PiCcsV1_1PackageBridgeError::Shape("running Eval_A matrix count"));
        }
        let mut evaluations = Vec::with_capacity((PI_CCS_V1_1_MATRIX_COUNT + 1) * PI_CCS_V1_1_COEFFICIENT_COUNT * 2);
        for value in &claim.eval_k[..PI_CCS_V1_1_COEFFICIENT_COUNT] {
            evaluations.extend_from_slice(&extension_words(*value));
        }
        for matrix in &claim.eval_a {
            validate_family(matrix)?;
            for value in &matrix[..PI_CCS_V1_1_COEFFICIENT_COUNT] {
                evaluations.extend_from_slice(&extension_words(*value));
            }
        }
        push_block(&mut words, &evaluations);
    }
    words.push(pc);
    if words.len() != PI_CCS_V1_1_STATE_PREIMAGE_WORDS {
        return Err(PiCcsV1_1PackageBridgeError::Shape("serialized state preimage length"));
    }
    Ok(words)
}

/// Recompute the Lean `stateHash` from its canonical serialized preimage.
pub fn pi_ccs_v1_1_state_hash(preimage: &[u64]) -> Result<[u64; 4], PiCcsV1_1PackageBridgeError> {
    if preimage.len() != PI_CCS_V1_1_STATE_PREIMAGE_WORDS || preimage.iter().any(|word| *word >= F::ORDER_U64) {
        return Err(PiCcsV1_1PackageBridgeError::Shape("state preimage words"));
    }
    let fields: Vec<_> = preimage.iter().map(|word| F::from_u64(*word)).collect();
    Ok(poseidon2_hash(&fields).map(|value| value.as_canonical_u64()))
}

/// Exact Lean `encHash`: marker, 256 little-endian digest bits, then zero padding.
pub fn encode_pi_ccs_v1_1_public_input(digest: [u64; 4]) -> Result<Vec<u64>, PiCcsV1_1PackageBridgeError> {
    if digest.iter().any(|word| *word >= F::ORDER_U64) {
        return Err(PiCcsV1_1PackageBridgeError::Shape("state digest words"));
    }
    let mut output = vec![0; PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS];
    output[0] = 1;
    for (word, value) in digest.into_iter().enumerate() {
        for bit in 0..64 {
            output[1 + word * 64 + bit] = (value >> bit) & 1;
        }
    }
    Ok(output)
}

fn push_block(output: &mut Vec<u64>, block: &[u64]) {
    output.push(block.len() as u64);
    output.extend_from_slice(block);
}

fn validate_family(values: &[K]) -> Result<(), PiCcsV1_1PackageBridgeError> {
    if values.len() < PI_CCS_V1_1_COEFFICIENT_COUNT
        || values[PI_CCS_V1_1_COEFFICIENT_COUNT..]
            .iter()
            .any(|value| *value != K::ZERO)
    {
        return Err(PiCcsV1_1PackageBridgeError::Shape("evaluation family width or padding"));
    }
    Ok(())
}

fn extension_words(value: K) -> [u64; 2] {
    let [low, high] = value.as_coeffs();
    [low.as_canonical_u64(), high.as_canonical_u64()]
}
