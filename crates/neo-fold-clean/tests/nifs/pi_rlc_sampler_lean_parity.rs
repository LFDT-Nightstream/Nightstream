//! Exact PiRLC sampler parity against the Lean-emitted Stage 1 vectors.

use std::{fs, path::PathBuf};

use neo_fold_clean::engine::{optimized, paper_exact};
use neo_fold_clean::paper::params::Params;
use neo_math::{D, F};
use neo_params::goldilocks_paper_b2::CHALLENGE_ALPHABET;
use neo_reductions::common::{
    decode_pi_rlc_v1_1_coefficients, PI_RLC_V1_1_DIGEST_ROUNDS, PI_RLC_V1_1_RATE_LANES, PI_RLC_V1_1_REJECTION_BUCKET,
};
use neo_transcript::Poseidon2Transcript;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde::Deserialize;

const MODULUS: u64 = 0xffff_ffff_0000_0001;

#[derive(Deserialize)]
struct Artifact(
    u64,
    [u64; 7],
    Vec<DecoderCase>,
    InjectedCase,
    InjectedCase,
    TranscriptCase,
);

#[derive(Deserialize)]
struct DecoderCase(u64, u64, Vec<u64>);

#[derive(Deserialize)]
struct InjectedCase(Vec<u64>, Vec<[u64; 4]>, TaggedWords);

#[derive(Deserialize)]
#[serde(untagged)]
enum TaggedWords {
    None((u64,)),
    Some((u64, Vec<u64>)),
}

impl TaggedWords {
    fn words(&self) -> Option<&[u64]> {
        match self {
            Self::None((tag,)) => {
                assert_eq!(*tag, 0, "canonical none tag");
                None
            }
            Self::Some((tag, words)) => {
                assert_eq!(*tag, 1, "canonical some tag");
                Some(words)
            }
        }
    }
}

#[derive(Deserialize)]
struct TranscriptCase([u64; 8], u64, Vec<TranscriptEntry>, [u64; 8], u64);

#[derive(Deserialize)]
struct TranscriptEntry(
    u64,
    [u64; 8],
    [u64; 8],
    Vec<DigestBlock>,
    Vec<u64>,
    TaggedWords,
    TaggedWords,
    [u64; 8],
);

#[derive(Deserialize)]
struct DigestBlock([u64; 8], [u64; 4], [u64; 8]);

fn artifact_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-pi-rlc-sampler-v1.json")
}

fn artifact() -> Artifact {
    serde_json::from_slice(&fs::read(artifact_path()).expect("Lean PiRLC sampler artifact"))
        .expect("canonical Lean PiRLC sampler JSON")
}

fn field(word: u64) -> F {
    assert!(word < MODULUS, "canonical Goldilocks word");
    F::from_u64(word)
}

fn state_words(transcript: &Poseidon2Transcript) -> [u64; 8] {
    transcript.state().map(|value| value.as_canonical_u64())
}

fn digest_array(words: &[[u64; 4]]) -> [[F; PI_RLC_V1_1_RATE_LANES]; PI_RLC_V1_1_DIGEST_ROUNDS] {
    words
        .iter()
        .map(|digest| digest.map(field))
        .collect::<Vec<_>>()
        .try_into()
        .expect("exactly eight digest windows")
}

fn unpack_candidates(digests: &[[u64; 4]]) -> Vec<u64> {
    digests
        .iter()
        .flat_map(|digest| {
            digest
                .iter()
                .flat_map(|word| [word & 0xffff, (word >> 16) & 0xffff])
        })
        .collect()
}

fn expected_centered(indices: &[u64]) -> Vec<i8> {
    indices
        .iter()
        .map(|index| CHALLENGE_ALPHABET[usize::try_from(*index).expect("alphabet index")])
        .collect()
}

#[test]
fn direct_decoder_matches_lean_boundaries_and_fails_closed() {
    let Artifact(schema, parameters, decoder_cases, success, shortfall, _) = artifact();
    assert_eq!(schema, 1);
    assert_eq!(parameters, [65_536, 65_535, 5, D as u64, 8, 8, 64]);
    assert_eq!(PI_RLC_V1_1_REJECTION_BUCKET, 65_535);

    for DecoderCase(candidate, accepted, decoded) in decoder_cases {
        assert_eq!(accepted, u64::from(candidate != 65_535));
        if accepted == 0 {
            assert_eq!(decoded, [0]);
        } else {
            let index = candidate % 5;
            let centered = CHALLENGE_ALPHABET[index as usize];
            let field_word = if centered < 0 {
                MODULUS - centered.unsigned_abs() as u64
            } else {
                centered as u64
            };
            assert_eq!(decoded, [1, index, field_word]);
        }
    }

    assert_eq!(success.0.len(), 64);
    assert_eq!(success.1.len(), PI_RLC_V1_1_DIGEST_ROUNDS);
    assert_eq!(unpack_candidates(&success.1), success.0);
    let expected = expected_centered(success.2.words().expect("Lean success output"));
    assert_eq!(expected.len(), D);
    assert_eq!(
        decode_pi_rlc_v1_1_coefficients(&digest_array(&success.1)).unwrap(),
        expected.as_slice()
    );

    assert_eq!(shortfall.0.len(), 64);
    assert_eq!(unpack_candidates(&shortfall.1), shortfall.0);
    assert!(shortfall.2.words().is_none(), "Lean shortfall must be none");
    let error = decode_pi_rlc_v1_1_coefficients(&digest_array(&shortfall.1)).expect_err("shortfall must reject");
    assert!(error.to_string().contains("sampler shortfall"));
}

#[test]
fn transcript_windows_and_both_engines_match_lean() {
    let Artifact(_, _, _, _, _, TranscriptCase(initial, count, entries, final_state, accepted)) = artifact();
    assert_eq!(initial, [0; 8]);
    assert_eq!(count, entries.len() as u64);
    assert_eq!(accepted, 1);

    let mut replay = Poseidon2Transcript::new_v1_1();
    for TranscriptEntry(coordinate, before, entered, blocks, candidates, scalar, ring, next) in &entries {
        assert_eq!(state_words(&replay), *before);
        replay.absorb_v1_1(&[F::from_u64(4), F::from_u64(*coordinate)]);
        assert_eq!(state_words(&replay), *entered);
        assert_eq!(blocks.len(), PI_RLC_V1_1_DIGEST_ROUNDS);

        let mut replay_candidates = Vec::with_capacity(64);
        let mut replay_digests = Vec::with_capacity(PI_RLC_V1_1_DIGEST_ROUNDS);
        for DigestBlock(block_state, digest, block_candidates) in blocks {
            assert_eq!(state_words(&replay), *block_state);
            let actual = replay
                .squeeze_digest_v1_1()
                .map(|value| value.as_canonical_u64());
            assert_eq!(actual, *digest);
            let actual_candidates = unpack_candidates(std::slice::from_ref(digest));
            assert_eq!(actual_candidates, block_candidates);
            replay_candidates.extend(actual_candidates);
            replay_digests.push(*digest);
        }
        assert_eq!(replay_candidates, *candidates);
        let decoded = decode_pi_rlc_v1_1_coefficients(&digest_array(&replay_digests)).unwrap();
        let scalar_indices = scalar.words().expect("Lean transcript scalar");
        assert_eq!(decoded.as_slice(), expected_centered(scalar_indices));
        let ring_words = ring.words().expect("Lean transcript ring challenge");
        assert_eq!(ring_words.len(), D);
        for (actual, expected) in decoded.iter().zip(ring_words) {
            assert_eq!(field(*expected), F::from_i8(*actual));
        }
        assert_eq!(state_words(&replay), *next);
    }
    assert_eq!(state_words(&replay), final_state);

    let params = Params::production();
    let mut optimized_transcript = Poseidon2Transcript::new_v1_1();
    let optimized_rhos =
        optimized::sample_rho_n(&mut optimized_transcript, &params, entries.len()).expect("optimized PiRLC sampler");
    let mut paper_transcript = Poseidon2Transcript::new_v1_1();
    let paper_rhos =
        paper_exact::sample_rho_n(&mut paper_transcript, &params, entries.len()).expect("PaperExact PiRLC sampler");

    assert_eq!(state_words(&optimized_transcript), final_state);
    assert_eq!(state_words(&paper_transcript), final_state);
    assert_eq!(optimized_rhos.len(), entries.len());
    assert_eq!(paper_rhos.len(), entries.len());
    for ((optimized, paper), entry) in optimized_rhos.iter().zip(&paper_rhos).zip(&entries) {
        assert_eq!(optimized.as_mat(), paper.as_mat());
        let expected = entry.6.words().expect("Lean ring challenge");
        for (row, word) in expected.iter().enumerate() {
            assert_eq!(optimized.as_mat()[(row, 0)], field(*word));
        }
    }
}
