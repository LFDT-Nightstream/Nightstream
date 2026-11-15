//! The words and bytes of the compression transcript against Lean
//! (`Export.BoundedDepth.CompressionSponge`, SPEC.md section 5.4). Lean
//! emits the domain, the session label, the words of byte strings and of a
//! claim, the session bytes of a sponge state, and the pieces of a sample
//! run with its session.

use std::fs;
use std::path::PathBuf;

use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_math::{from_complex, D, F, K};
use neo_transcript::{domain_chunk_v1_2, Poseidon2Transcript};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde_json::Value;

use super::{byte_words, challenges, session};
use crate::support::{Rows, KAPPA};
use crate::{claim_words, CeClaim, Norm, Statement, DOMAIN, SESSION};

fn lean() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/nightstream-fprime/artifacts/nightstream-fprime-encoding-parity-v1.json");
    let file: Value = serde_json::from_slice(&fs::read(path).expect("Lean encoding file")).expect("JSON");
    assert_eq!(file[0].as_u64(), Some(1), "Lean encoding schema");
    file[3].clone()
}

fn numbers(value: &Value) -> Vec<u64> {
    let numbers = value.as_array().expect("numbers").iter();
    numbers.map(|number| number.as_u64().expect("u64")).collect()
}

fn bytes(value: &Value) -> Vec<u8> {
    let bytes = numbers(value).into_iter();
    bytes.map(|byte| u8::try_from(byte).expect("a byte")).collect()
}

fn fields(value: &Value) -> Vec<F> {
    numbers(value).into_iter().map(F::from_u64).collect()
}

fn extension(value: &Value) -> Vec<K> {
    let pairs = fields(value);
    pairs.chunks(2).map(|pair| from_complex(pair[0], pair[1])).collect()
}

fn canonical(words: &[F]) -> Vec<u64> {
    words.iter().map(|word| word.as_canonical_u64()).collect()
}

#[test]
fn the_domain_and_the_session_label_are_the_lean_ones() {
    let lean = lean();
    assert_eq!(canonical(&domain_chunk_v1_2(DOMAIN)), numbers(&lean[0][1]));
    assert_eq!(SESSION, bytes(&lean[0][2]));
}

#[test]
fn byte_words_are_the_lean_words() {
    let lean = lean();
    let cases = lean[1].as_array().expect("byte strings");
    let lengths: Vec<usize> = cases.iter().map(|case| bytes(&case[0]).len()).collect();
    assert_eq!(lengths, [0, 1, 6, 7, 8, 13, 14, 15, 50]);
    for case in cases {
        assert_eq!(canonical(&byte_words(&bytes(&case[0]))), numbers(&case[1]));
    }
}

#[test]
fn claim_words_are_the_lean_words() {
    let lean = lean();
    let claim = &lean[2];
    let blocks = claim[0].as_u64().expect("blocks") as usize;
    let cells = fields(&claim[3]);
    let evaluations = extension(&claim[6]);
    let rows = Rows::random(&mut ChaCha8Rng::seed_from_u64(1), 4, blocks);
    let statement = Statement {
        claim: &CeClaim {
            c: Commitment::new(KAPPA, fields(&claim[2])).expect("commitment"),
            m_in: cells.len(),
            X: Mat::from_row_major(D, cells.len() / D, cells),
            r: extension(&claim[4]),
            eval_k: extension(&claim[5]),
            eval_a: evaluations.chunks(D).map(<[K]>::to_vec).collect(),
            fold_digest: [0; 32],
            adv: None,
        },
        rows: &rows,
        norm: Norm::Table {
            bound: claim[1].as_u64().expect("the norm word") - 1,
        },
    };
    assert_eq!(canonical(&claim_words(&statement)), numbers(&claim[7]));
}

fn transcript_at(state: &[F]) -> Poseidon2Transcript {
    Poseidon2Transcript::from_state_and_absorbed(state.try_into().expect("a sponge state"), 0)
}

#[test]
fn the_session_is_the_lean_session() {
    let lean = lean();
    let transcript = transcript_at(&fields(&lean[3][0]));
    assert_eq!(session(&transcript, SESSION), bytes(&lean[3][1]));
}

/// Lean's piece list (`Sponge.calls`) has the zero blocks of
/// `read_coin_v1_2` at the same places as the Rust transcript.
#[test]
fn a_replay_of_the_lean_pieces_ends_in_the_lean_session() {
    let lean = lean();
    let mut transcript = Poseidon2Transcript::new_v1_2();
    let pieces = lean[4][0].as_array().expect("pieces");
    assert!(pieces.iter().any(|piece| piece[1].as_u64().is_some_and(|count| count > 12)));
    for piece in pieces {
        match piece[0].as_u64() {
            Some(0) => transcript.absorb_v1_2(&fields(&piece[1])),
            Some(1) => drop(challenges(&mut transcript, piece[1].as_u64().expect("count") as usize)),
            other => panic!("piece {other:?}"),
        }
    }
    assert_eq!(session(&transcript, SESSION), bytes(&lean[4][1]));
}
