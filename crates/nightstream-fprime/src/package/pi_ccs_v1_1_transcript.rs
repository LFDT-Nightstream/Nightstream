//! Exact additive Poseidon2 transcript replay for SuperNeo v1.1 PiCCS.
//!
//! Lean emits differential vectors for this implementation. This module
//! does not select a circuit or accept a proof; it derives verifier-owned
//! challenges and the post-output state from fixed-profile public messages.

use neo_transcript::Poseidon2Transcript;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;

use super::{canonical_field, PackageError, PI_CCS_V1_1_ROUND_COEFFICIENT_COUNT, PI_CCS_V1_1_ROUND_COUNT};

const WIDTH: usize = 8;
const CUBE_VARIABLES: usize = PI_CCS_V1_1_ROUND_COUNT;
const RUNNING_SOURCES: usize = 16;
const SOURCE_COUNT: usize = 17;
const MATRIX_COUNT: usize = 14;
const COEFFICIENT_COUNT: usize = 54;
const COMMITMENT_WORDS: usize = 1_188;
const PUBLIC_INPUT_WORDS: usize = 270;
const EVALUATION_WORDS: usize = (MATRIX_COUNT + 1) * COEFFICIENT_COUNT * 2;
const OUTPUT_WORDS: usize = SOURCE_COUNT * EVALUATION_WORDS;

const DOMAIN_TAG: &[u64] = &[
    78, 105, 103, 104, 116, 115, 116, 114, 101, 97, 109, 47, 83, 117, 112, 101, 114, 78, 101, 111, 47, 80, 105, 67, 67,
    83, 47, 100, 105, 103, 101, 115, 116, 45, 111, 110, 108, 121, 47, 118, 49, 95, 49,
];

/// Verifier-derived PiCCS values in exact Lean order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiCcsV1_1Transcript {
    alpha: Vec<[u64; 2]>,
    gamma: [u64; 2],
    round_point: Vec<[u64; 2]>,
    outgoing_state: [u64; WIDTH],
}

impl PiCcsV1_1Transcript {
    /// The pre-SumCheck challenges.
    pub fn alpha(&self) -> &[[u64; 2]] {
        &self.alpha
    }

    /// The joint-polynomial mixing challenge.
    pub fn gamma(&self) -> [u64; 2] {
        self.gamma
    }

    /// The SumCheck round challenges.
    pub fn round_point(&self) -> &[[u64; 2]] {
        &self.round_point
    }

    /// State after the complete separate `Eval_K` and `Eval_A` output.
    pub fn outgoing_state(&self) -> [u64; WIDTH] {
        self.outgoing_state
    }
}

/// Derive the fixed-profile v1.1 PiCCS transcript.
///
/// Public statement blocks are: the pilot-recomputed prior-state digest, the
/// fresh commitment, and the fresh public input. The semantic verifier blocks
/// are validated but are not absorbed again because the prior digest already
/// binds them.
pub fn derive_pi_ccs_v1_1_transcript(
    public_statement_blocks: &[Vec<u64>],
    verifier_input_blocks: &[Vec<u64>],
    rounds: &[Vec<[u64; 2]>],
    output_words: &[u64],
) -> Result<PiCcsV1_1Transcript, PackageError> {
    validate_shapes(public_statement_blocks, verifier_input_blocks, rounds, output_words)?;
    let mut transcript = Poseidon2Transcript::new_v1_1();
    transcript.absorb_v1_1(&canonical_words(DOMAIN_TAG)?);
    for block in public_statement_blocks {
        transcript.absorb_block_v1_1(&canonical_words(block)?);
    }

    let mut alpha = Vec::with_capacity(CUBE_VARIABLES);
    for coordinate in 0..CUBE_VARIABLES {
        transcript.absorb_v1_1(&canonical_words(&[1, coordinate as u64])?);
        alpha.push(extension_words(transcript.squeeze_extension_v1_1()));
    }
    transcript.absorb_v1_1(&canonical_words(&[2])?);
    let gamma = extension_words(transcript.squeeze_extension_v1_1());

    let mut round_point = Vec::with_capacity(CUBE_VARIABLES);
    for (round_index, message) in rounds.iter().enumerate() {
        let mut words = Vec::with_capacity(1 + PI_CCS_V1_1_ROUND_COEFFICIENT_COUNT * 2);
        words.push(round_index as u64);
        for coefficient in message {
            words.extend(coefficient);
        }
        transcript.absorb_block_v1_1(&canonical_words(&words)?);
        transcript.absorb_v1_1(&canonical_words(&[3, round_index as u64])?);
        round_point.push(extension_words(transcript.squeeze_extension_v1_1()));
    }
    transcript.absorb_block_v1_1(&canonical_words(output_words)?);

    Ok(PiCcsV1_1Transcript {
        alpha,
        gamma,
        round_point,
        outgoing_state: transcript.state().map(|value| value.as_canonical_u64()),
    })
}

fn validate_shapes(
    public: &[Vec<u64>],
    verifier: &[Vec<u64>],
    rounds: &[Vec<[u64; 2]>],
    output: &[u64],
) -> Result<(), PackageError> {
    if public.len() != 3
        || public[0].len() != 4
        || public[1].len() != COMMITMENT_WORDS
        || public[2].len() != PUBLIC_INPUT_WORDS
        || verifier.len() != 2
        || verifier[0].len() != CUBE_VARIABLES * 2
        || verifier[1].len() != RUNNING_SOURCES * EVALUATION_WORDS
        || rounds.len() != CUBE_VARIABLES
        || rounds
            .iter()
            .any(|message| message.len() != PI_CCS_V1_1_ROUND_COEFFICIENT_COUNT)
        || output.len() != OUTPUT_WORDS
    {
        return Err(PackageError::Invalid("PiCCS v1_1 transcript shape"));
    }
    Ok(())
}

fn canonical_words(words: &[u64]) -> Result<Vec<Goldilocks>, PackageError> {
    words
        .iter()
        .map(|&value| {
            canonical_field(value, "PiCCS v1_1 transcript word")?;
            Ok(Goldilocks::from_u64(value))
        })
        .collect()
}

fn extension_words(value: [Goldilocks; 2]) -> [u64; 2] {
    value.map(|coefficient| coefficient.as_canonical_u64())
}
