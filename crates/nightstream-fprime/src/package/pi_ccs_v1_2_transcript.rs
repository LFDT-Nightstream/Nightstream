//! Exact additive Poseidon2 transcript replay for SuperNeo v1.2 PiCCS.
//!
//! This module does not select a circuit or accept a proof; it derives
//! verifier-owned challenges and the post-output state from fixed-profile public messages.

use neo_transcript::{domain_chunk_v1_2, fold_domain_chunk_v1_2, Poseidon2Transcript};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;

use super::{canonical_field, PackageError, PI_CCS_V1_2_ROUND_COEFFICIENT_COUNT, PI_CCS_V1_2_ROUND_COUNT};

const WIDTH: usize = neo_ccs::crypto::poseidon2_goldilocks::WIDTH;
const CUBE_VARIABLES: usize = PI_CCS_V1_2_ROUND_COUNT;
const SOURCE_COUNT: usize = 17;
const LIGHT_SOURCE_COUNT: usize = 2;
/// The fold domains of option B's light and refresh PiCCS (Lean `OptionB.Keys`).
pub const LIGHT_DOMAIN: &[u8] = b"Nightstream/SuperNeo/fold/light/v1";
pub const REFRESH_DOMAIN: &[u8] = b"Nightstream/SuperNeo/fold/refresh/v1";
const CHILD_WORDS: usize = 16 * (COMMITMENT_WORDS + EVALUATION_WORDS);
const MATRIX_COUNT: usize = super::PI_CCS_V1_2_MATRIX_COUNT;
const COEFFICIENT_COUNT: usize = 54;
const COMMITMENT_WORDS: usize = 1_188;
const PUBLIC_INPUT_WORDS: usize = 270;
const EVALUATION_WORDS: usize = (MATRIX_COUNT + 1) * COEFFICIENT_COUNT * 2;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiCcsV1_2Transcript {
    alpha: Vec<[u64; 2]>,
    gamma: [u64; 2],
    round_point: Vec<[u64; 2]>,
    outgoing_state: [u64; WIDTH],
}

impl PiCcsV1_2Transcript {
    pub fn alpha(&self) -> &[[u64; 2]] {
        &self.alpha
    }

    pub fn gamma(&self) -> [u64; 2] {
        self.gamma
    }

    pub fn round_point(&self) -> &[[u64; 2]] {
        &self.round_point
    }

    pub fn outgoing_state(&self) -> [u64; WIDTH] {
        self.outgoing_state
    }
}

/// After the constant domain chunk, the statement is absorbed as one stream
/// with no length prefixes: the prior-state digest, the fresh commitment and
/// the fresh public input. The digest is Lean's `decodeHash` of that input, as
/// `ProductionKey.priorDigest` reads it, so it is not a separate argument.
/// Coins are read from rate-lane pairs; nothing is labeled.
pub fn derive_pi_ccs_v1_2_transcript(
    fresh_commitment: &[u64],
    fresh_public_input: &[u64],
    rounds: &[Vec<[u64; 2]>],
    output_words: &[u64],
) -> Result<PiCcsV1_2Transcript, PackageError> {
    derive(
        fold_domain_chunk_v1_2(),
        fresh_commitment,
        fresh_public_input,
        &[],
        rounds,
        output_words,
        SOURCE_COUNT,
    )
}

/// The light fold's transcript (`OptionB.lightPiCcsOracle`): the production
/// transcript under the light fold's domain, with the two sources' outputs.
pub fn derive_light_transcript(
    fresh_commitment: &[u64],
    fresh_public_input: &[u64],
    rounds: &[Vec<[u64; 2]>],
    output_words: &[u64],
) -> Result<PiCcsV1_2Transcript, PackageError> {
    derive(
        domain_chunk_v1_2(LIGHT_DOMAIN),
        fresh_commitment,
        fresh_public_input,
        &[],
        rounds,
        output_words,
        LIGHT_SOURCE_COUNT,
    )
}

/// The refresh fold's transcript (`OptionB.refreshWords`): the 16 Π_DEC
/// children's commitments and evaluation families follow the statement.
pub fn derive_refresh_transcript(
    fresh_commitment: &[u64],
    fresh_public_input: &[u64],
    child_words: &[u64],
    rounds: &[Vec<[u64; 2]>],
    output_words: &[u64],
) -> Result<PiCcsV1_2Transcript, PackageError> {
    if child_words.len() != CHILD_WORDS {
        return Err(PackageError::Malformed("refresh transcript child words"));
    }
    derive(
        domain_chunk_v1_2(REFRESH_DOMAIN),
        fresh_commitment,
        fresh_public_input,
        child_words,
        rounds,
        output_words,
        SOURCE_COUNT,
    )
}

fn derive(
    domain: [Goldilocks; neo_ccs::crypto::poseidon2_goldilocks::RATE],
    fresh_commitment: &[u64],
    fresh_public_input: &[u64],
    extra_words: &[u64],
    rounds: &[Vec<[u64; 2]>],
    output_words: &[u64],
    source_count: usize,
) -> Result<PiCcsV1_2Transcript, PackageError> {
    validate_shapes(fresh_commitment, fresh_public_input, rounds, output_words, source_count)?;
    let public_input = canonical_words(fresh_public_input)?;
    let mut transcript = Poseidon2Transcript::new_v1_2();
    transcript.absorb_v1_2(&domain);
    let mut statement = neo_transcript::prior_digest_v1_2(&public_input).to_vec();
    statement.extend(canonical_words(fresh_commitment)?);
    statement.extend_from_slice(&public_input);
    statement.extend(canonical_words(extra_words)?);
    transcript.absorb_v1_2(&statement);
    let mut alpha = Vec::with_capacity(CUBE_VARIABLES);
    for _ in 0..CUBE_VARIABLES {
        alpha.push(read_coin(&mut transcript));
    }
    let gamma = read_coin(&mut transcript);
    let mut round_point = Vec::with_capacity(CUBE_VARIABLES);
    for message in rounds {
        let words: Vec<u64> = message.iter().flatten().copied().collect();
        transcript.absorb_v1_2(&canonical_words(&words)?);
        round_point.push(read_coin(&mut transcript));
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
    commitment: &[u64],
    public_input: &[u64],
    rounds: &[Vec<[u64; 2]>],
    output: &[u64],
    source_count: usize,
) -> Result<(), PackageError> {
    if commitment.len() != COMMITMENT_WORDS
        || public_input.len() != PUBLIC_INPUT_WORDS
        || rounds.len() != CUBE_VARIABLES
        || rounds
            .iter()
            .any(|message| message.len() != PI_CCS_V1_2_ROUND_COEFFICIENT_COUNT)
        || output.len() != source_count * EVALUATION_WORDS
    {
        return Err(PackageError::Malformed("PiCCS v1_2 transcript shape"));
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

fn read_coin(transcript: &mut Poseidon2Transcript) -> [u64; 2] {
    transcript
        .read_coin_v1_2()
        .value
        .map(|coefficient| coefficient.as_canonical_u64())
}
