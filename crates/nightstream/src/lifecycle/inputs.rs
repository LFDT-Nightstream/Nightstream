//! PiCCS proof-message bridge to the Lean-emitted v1_1 package.
//!
//! Owns only the field-for-field conversion from native prover messages to
//! the package input types, canonical lifecycle serialization, and the
//! verifier-owned package input construction. It does not define, load, prove,
//! or verify a relation.

use neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash;
use neo_ccs::Mat;
use neo_math::{KExtensions, F, K};
use neo_reductions::common::split_b_matrix_k;
use nightstream_fprime::{
    PackageError, PiCcsV1_1OutputEvaluations, PiCcsV1_1PackageInputs, PiCcsV1_1VerifierContext,
    PI_CCS_V1_1_COEFFICIENT_COUNT, PI_CCS_V1_1_FRESH_COMMITMENT_WORDS, PI_CCS_V1_1_MATRIX_COUNT,
    PI_CCS_V1_1_PRIOR_CHILDREN_WORDS, PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS, PI_CCS_V1_1_ROUND_COEFFICIENT_COUNT,
    PI_CCS_V1_1_ROUND_COUNT, PI_CCS_V1_1_SOURCE_COUNT, PI_CCS_V1_1_STATE_PREIMAGE_WORDS, PI_DEC_V1_1_CHILD_COUNT,
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use thiserror::Error;

use crate::folding::pi_ccs;
use crate::folding::{CcsClaim, CeClaim};

/// `HyperNova/NIVC/state/v2`, eight little-endian bytes per word, then zero
/// words up to one full sponge rate.
const STATE_DOMAIN_TAG: &[u8; 23] = b"HyperNova/NIVC/state/v2";
const STATE_DOMAIN_WORDS: usize = 12;
/// Lean `stateDomainChunk`: the tag bytes, little-endian, zero-padded to one
/// full sponge rate.
const STATE_DOMAIN_CHUNK: [u64; STATE_DOMAIN_WORDS] = {
    let mut words = [0; STATE_DOMAIN_WORDS];
    let mut index = 0;
    while index < STATE_DOMAIN_TAG.len() {
        words[index / 8] |= (STATE_DOMAIN_TAG[index] as u64) << (8 * (index % 8));
        index += 1;
    }
    words
};
const COMMITMENT_WIDTH: usize = PI_CCS_V1_1_FRESH_COMMITMENT_WORDS / PI_CCS_V1_1_COEFFICIENT_COUNT;
/// Three parent coordinates share one state word in radix `2^17`.
const PACK_RADIX: u64 = 1 << 17;
const PUBLIC_COLUMNS: usize = PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS / PI_CCS_V1_1_COEFFICIENT_COUNT;

#[derive(Debug, Error)]
pub enum PiCcsV1_1PackageBridgeError {
    #[error("PiCCS v1_1 package bridge: {0}")]
    Shape(&'static str),
    #[error("PiCCS v1_1 package bridge: running children are not the canonical split of their parent")]
    NonCanonicalChildren,
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

    /// Add the lifecycle-owned fields after their canonical serializer has
    /// produced them.
    pub fn into_package_inputs(
        self,
        prior_preimage: Vec<u64>,
        output_preimage: Vec<u64>,
        prior_children: Vec<u64>,
        prior_public_input: Vec<u64>,
        output_digest: [u64; 4],
        verifier_context: PiCcsV1_1VerifierContext,
    ) -> Result<PiCcsV1_1PackageInputs, PiCcsV1_1PackageBridgeError> {
        Ok(PiCcsV1_1PackageInputs::new(
            prior_preimage,
            output_preimage,
            prior_children,
            self.fresh_commitment,
            self.round_messages,
            self.output_evaluations,
            prior_public_input,
            output_digest,
            verifier_context,
        )?)
    }
}

/// Serialize the exact Lean `HashPreimage` for the fixed v1_1 profile: the
/// domain chunk, the running commitments, `Eval_K`, `Eval_A` and point, the
/// packed parent public input, then `vk, i, z0, zi`. The children's public
/// inputs enter only through their parent, so this rejects any running
/// instance whose children are not the canonical split of that parent
/// (SuperNeo Π_DEC verifier step 2); a second split cannot reuse the hash.
pub fn serialize_pi_ccs_v1_1_state_preimage(
    verifier_context_digest: [F; 4],
    iteration: u64,
    z0: [F; 4],
    current: [F; 4],
    running: &[CeClaim],
) -> Result<Vec<u64>, PiCcsV1_1PackageBridgeError> {
    if iteration >= F::ORDER_U64 {
        return Err(PiCcsV1_1PackageBridgeError::Shape("iteration is not canonical"));
    }
    let parent = canonical_parent(running)?;
    let point = &running[0].r;
    let mut words = Vec::with_capacity(PI_CCS_V1_1_STATE_PREIMAGE_WORDS);
    words.extend_from_slice(&STATE_DOMAIN_CHUNK);
    for claim in running {
        if claim.c.d != PI_CCS_V1_1_COEFFICIENT_COUNT
            || claim.c.kappa != COMMITMENT_WIDTH
            || claim.c.data.len() != PI_CCS_V1_1_FRESH_COMMITMENT_WORDS
        {
            return Err(PiCcsV1_1PackageBridgeError::Shape("running commitment"));
        }
        words.extend(claim.c.data.iter().map(|value| value.as_canonical_u64()));
    }
    for claim in running {
        validate_family(&claim.eval_k)?;
        for value in &claim.eval_k[..PI_CCS_V1_1_COEFFICIENT_COUNT] {
            words.extend_from_slice(&extension_words(*value));
        }
    }
    for claim in running {
        if claim.eval_a.len() != PI_CCS_V1_1_MATRIX_COUNT {
            return Err(PiCcsV1_1PackageBridgeError::Shape("running Eval_A matrix count"));
        }
        for matrix in &claim.eval_a {
            validate_family(matrix)?;
            for value in &matrix[..PI_CCS_V1_1_COEFFICIENT_COUNT] {
                words.extend_from_slice(&extension_words(*value));
            }
        }
    }
    for value in point {
        words.extend_from_slice(&extension_words(*value));
    }
    let radix = F::from_u64(PACK_RADIX);
    for lanes in parent.chunks_exact(3) {
        words.push((lanes[0] + radix * lanes[1] + radix * radix * lanes[2]).as_canonical_u64());
    }
    words.extend(verifier_context_digest.map(|value| value.as_canonical_u64()));
    words.push(iteration);
    words.extend(z0.map(|value| value.as_canonical_u64()));
    words.extend(current.map(|value| value.as_canonical_u64()));
    if words.len() != PI_CCS_V1_1_STATE_PREIMAGE_WORDS {
        return Err(PiCcsV1_1PackageBridgeError::Shape("serialized state preimage length"));
    }
    Ok(words)
}

/// The PiCCS prior child region: the sixteen child public inputs, child-major.
/// The children must be the canonical split of their parent; the circuit
/// derives each lane's sign from its own digits.
pub fn pi_ccs_v1_1_prior_children(running: &[CeClaim]) -> Result<Vec<u64>, PiCcsV1_1PackageBridgeError> {
    canonical_parent(running)?;
    let mut words = Vec::with_capacity(PI_CCS_V1_1_PRIOR_CHILDREN_WORDS);
    for claim in running {
        words.extend((0..PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS).map(|column| public_input(claim, column)));
    }
    Ok(words)
}

/// The parent public input of a running instance with the shared point,
/// after checking that its children are the parent's canonical split.
fn canonical_parent(running: &[CeClaim]) -> Result<Vec<F>, PiCcsV1_1PackageBridgeError> {
    checked_running_point(running)?;
    let parent = parent_public_input(running)?;
    let mut matrix = Mat::zero(PI_CCS_V1_1_COEFFICIENT_COUNT, PUBLIC_COLUMNS, F::ZERO);
    for (column, value) in parent.iter().enumerate() {
        matrix[(
            column % PI_CCS_V1_1_COEFFICIENT_COUNT,
            column / PI_CCS_V1_1_COEFFICIENT_COUNT,
        )] = *value;
    }
    // Canonical digits always recompose below the split bound, so a parent
    // past the bound also means non-canonical children.
    let split =
        split_b_matrix_k(&matrix, running.len(), 2).map_err(|_| PiCcsV1_1PackageBridgeError::NonCanonicalChildren)?;
    if split
        .iter()
        .zip(running)
        .any(|(digits, claim)| *digits != claim.X)
    {
        return Err(PiCcsV1_1PackageBridgeError::NonCanonicalChildren);
    }
    Ok(parent)
}

/// Lean `parentPublic`: `Σ_j 2^j x_j` in Lean column order.
fn parent_public_input(running: &[CeClaim]) -> Result<Vec<F>, PiCcsV1_1PackageBridgeError> {
    let mut parent = vec![F::ZERO; PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS];
    let mut power = F::ONE;
    for claim in running {
        checked_public_shape(claim)?;
        for (column, value) in parent.iter_mut().enumerate() {
            *value += power * F::from_u64(public_input(claim, column));
        }
        power = power.double();
    }
    Ok(parent)
}

fn checked_running_point(running: &[CeClaim]) -> Result<&[K], PiCcsV1_1PackageBridgeError> {
    if running.len() != PI_DEC_V1_1_CHILD_COUNT {
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
    Ok(point)
}

fn checked_public_shape(claim: &CeClaim) -> Result<(), PiCcsV1_1PackageBridgeError> {
    if claim.m_in != PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS
        || claim.X.rows() != PI_CCS_V1_1_COEFFICIENT_COUNT
        || claim.X.cols() != PUBLIC_COLUMNS
    {
        return Err(PiCcsV1_1PackageBridgeError::Shape("running public input"));
    }
    Ok(())
}

/// Lean public-input column `j` is the ring coefficient `j mod 54` of column `j / 54`.
fn public_input(claim: &CeClaim, column: usize) -> u64 {
    claim.X[(
        column % PI_CCS_V1_1_COEFFICIENT_COUNT,
        column / PI_CCS_V1_1_COEFFICIENT_COUNT,
    )]
        .as_canonical_u64()
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
