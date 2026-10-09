//! Exact additive Poseidon2 transcript replay for SuperNeo v1.2 PiCCS.
//!
//! Lean emits differential vectors for this implementation. This module
//! does not select a circuit or accept a proof; it derives verifier-owned
//! challenges and the post-output state from fixed-profile public messages.

use neo_transcript::{fold_domain_chunk_v1_2, Poseidon2Transcript};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;

use super::{canonical_field, PackageError, PI_CCS_V1_2_ROUND_COEFFICIENT_COUNT, PI_CCS_V1_2_ROUND_COUNT};

const WIDTH: usize = neo_ccs::crypto::poseidon2_goldilocks::WIDTH;
const RATE: usize = neo_ccs::crypto::poseidon2_goldilocks::RATE;
const COINS_PER_CHUNK: usize = RATE / 2;
const CUBE_VARIABLES: usize = PI_CCS_V1_2_ROUND_COUNT;
const RUNNING_SOURCES: usize = 16;
const SOURCE_COUNT: usize = 17;
const MATRIX_COUNT: usize = super::PI_CCS_V1_2_MATRIX_COUNT;
const COEFFICIENT_COUNT: usize = 54;
const COMMITMENT_WORDS: usize = 1_188;
const PUBLIC_INPUT_WORDS: usize = 270;
const EVALUATION_WORDS: usize = (MATRIX_COUNT + 1) * COEFFICIENT_COUNT * 2;
const OUTPUT_WORDS: usize = SOURCE_COUNT * EVALUATION_WORDS;

/// Verifier-derived PiCCS values in exact Lean order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiCcsV1_2Transcript {
    alpha: Vec<[u64; 2]>,
    gamma: [u64; 2],
    round_point: Vec<[u64; 2]>,
    outgoing_state: [u64; WIDTH],
}

impl PiCcsV1_2Transcript {
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

/// Derive the fixed-profile v1.2 PiCCS transcript.
///
/// After the constant domain chunk, the public statement blocks are absorbed
/// as one stream: the pilot-recomputed prior-state digest, the fresh
/// commitment, and the fresh public input. The semantic verifier blocks are
/// validated but are not absorbed again because the prior digest already
/// binds them. Coins are read from rate-lane pairs; nothing is labeled or
/// length-prefixed.
pub fn derive_pi_ccs_v1_2_transcript(
    public_statement_blocks: &[Vec<u64>],
    verifier_input_blocks: &[Vec<u64>],
    rounds: &[Vec<[u64; 2]>],
    output_words: &[u64],
) -> Result<PiCcsV1_2Transcript, PackageError> {
    validate_shapes(public_statement_blocks, verifier_input_blocks, rounds, output_words)?;
    let mut transcript = Poseidon2Transcript::new_v1_2();
    transcript.absorb_v1_2(&fold_domain_chunk_v1_2());
    transcript.absorb_v1_2(&canonical_words(&public_statement_blocks.concat())?);

    let mut alpha = Vec::with_capacity(CUBE_VARIABLES);
    for position in 0..CUBE_VARIABLES {
        alpha.push(read_coin(&mut transcript, position));
    }
    let gamma = read_coin(&mut transcript, CUBE_VARIABLES);

    let mut round_point = Vec::with_capacity(CUBE_VARIABLES);
    for message in rounds {
        let words: Vec<u64> = message.iter().flatten().copied().collect();
        transcript.absorb_v1_2(&canonical_words(&words)?);
        round_point.push(read_coin(&mut transcript, 0));
    }
    transcript.absorb_v1_2(&canonical_words(output_words)?);

    Ok(PiCcsV1_2Transcript {
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
            .any(|message| message.len() != PI_CCS_V1_2_ROUND_COEFFICIENT_COUNT)
        || output.len() != OUTPUT_WORDS
    {
        return Err(PackageError::Invalid("PiCCS v1_2 transcript shape"));
    }
    Ok(())
}

fn canonical_words(words: &[u64]) -> Result<Vec<Goldilocks>, PackageError> {
    words
        .iter()
        .map(|&value| {
            canonical_field(value, "PiCCS v1_2 transcript word")?;
            Ok(Goldilocks::from_u64(value))
        })
        .collect()
}

/// Read the coin at `position` from rate-lane pair `position % 6`. After the
/// sixth pair of a state, absorb one zero chunk.
fn read_coin(transcript: &mut Poseidon2Transcript, position: usize) -> [u64; 2] {
    let pair = position % COINS_PER_CHUNK;
    let value = transcript
        .read_pair_v1_2(pair)
        .map(|coefficient| coefficient.as_canonical_u64());
    if pair == COINS_PER_CHUNK - 1 {
        transcript.absorb_v1_2(&[Goldilocks::ZERO; RATE]);
    }
    value
}
