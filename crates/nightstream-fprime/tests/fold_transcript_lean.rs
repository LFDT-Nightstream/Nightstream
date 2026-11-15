//! The light and the refresh fold transcript against Lean
//! (`Lifecycle.RandomOracle.Schedule.pieces`). `absorb_v1_2` writes no
//! length, so each message of a transcript must have the length that Lean
//! proves fixed (`Schedule.pieces_alike`). Lean emits the pieces of each
//! transcript on sample data, the first and the last round challenge and the
//! sponge state after the output. Rust must absorb the same words at the
//! same lengths and reach the same state, and it must reject each other
//! length.

use std::fs;
use std::path::PathBuf;

use neo_math::Fq;
use nightstream_fprime::{derive_light_transcript, derive_refresh_transcript, COMMITMENT_WORDS, PUBLIC_WORDS};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde_json::Value;

/// The messages of one Lean transcript, split as the Rust transcript takes them.
struct Transcript {
    commitment: Vec<u64>,
    public_input: Vec<u64>,
    children: Vec<u64>,
    rounds: Vec<Vec<[u64; 2]>>,
    output: Vec<u64>,
    challenges: Vec<Vec<u64>>,
    state: Vec<u64>,
}

fn words(value: &Value) -> Vec<u64> {
    let words = value.as_array().expect("words").iter();
    words.map(|word| word.as_u64().expect("u64")).collect()
}

fn lean(index: usize, domain: &[u8]) -> Transcript {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/nightstream-fprime/artifacts/nightstream-fprime-encoding-parity-v1.json");
    let file: Value = serde_json::from_slice(&fs::read(path).expect("Lean encoding file")).expect("JSON");
    assert_eq!(file[0].as_u64(), Some(1), "Lean encoding schema");
    let transcript = &file[2][index];
    let pieces = transcript[0].as_array().expect("pieces");
    let absorb = |piece: &Value| {
        assert_eq!(piece[0].as_u64(), Some(0), "an absorb");
        words(&piece[1])
    };
    let reads = |piece: &Value| {
        assert_eq!(piece[0].as_u64(), Some(1), "reads");
        piece[1].as_u64().expect("count") as usize
    };
    let domain_chunk = neo_transcript::domain_chunk_v1_2(domain).map(|word| word.as_canonical_u64());
    assert_eq!(absorb(&pieces[0]), domain_chunk);
    let statement = absorb(&pieces[1]);
    let round_count = reads(&pieces[2]) - 1;
    assert_eq!(pieces.len(), 3 + 2 * round_count + 1);
    let rounds = (0..round_count)
        .map(|round| {
            assert_eq!(reads(&pieces[4 + 2 * round]), 1);
            let coefficients = absorb(&pieces[3 + 2 * round]);
            coefficients
                .chunks(2)
                .map(|pair| [pair[0], pair[1]])
                .collect()
        })
        .collect();
    let public_input = statement[4 + COMMITMENT_WORDS..4 + COMMITMENT_WORDS + PUBLIC_WORDS].to_vec();
    let fields: Vec<Fq> = public_input
        .iter()
        .map(|&word| Fq::from_u64(word))
        .collect();
    let prior = neo_transcript::prior_digest_v1_2(&fields).map(|word| word.as_canonical_u64());
    assert_eq!(statement[..4], prior, "the prior digest of the public input");
    Transcript {
        commitment: statement[4..4 + COMMITMENT_WORDS].to_vec(),
        public_input,
        children: statement[4 + COMMITMENT_WORDS + PUBLIC_WORDS..].to_vec(),
        rounds,
        output: absorb(pieces.last().expect("the output")),
        challenges: transcript[1]
            .as_array()
            .expect("challenges")
            .iter()
            .map(words)
            .collect(),
        state: words(&transcript[2]),
    }
}

/// Each list with one word less and with one word more.
fn other_lengths(words: &[u64]) -> [Vec<u64>; 2] {
    [words[..words.len() - 1].to_vec(), [words, &[0]].concat()]
}

/// `derive` reaches the Lean state on the Lean messages, and it rejects each
/// message with another length.
fn check(
    transcript: &Transcript,
    derive: impl Fn(&[u64], &[u64], &[u64], &[Vec<[u64; 2]>], &[u64]) -> Option<Derived>,
) {
    let Transcript {
        commitment,
        public_input,
        children,
        rounds,
        output,
        ..
    } = transcript;
    let (round_point, state) = derive(commitment, public_input, children, rounds, output).expect("the Lean lengths");
    assert_eq!(state.to_vec(), transcript.state);
    assert_eq!(transcript.challenges.len(), 2);
    assert_eq!(round_point[0].to_vec(), transcript.challenges[0]);
    assert_eq!(round_point[rounds.len() - 1].to_vec(), transcript.challenges[1]);

    for other in other_lengths(commitment) {
        assert!(derive(&other, public_input, children, rounds, output).is_none());
    }
    for other in other_lengths(public_input) {
        assert!(derive(commitment, &other, children, rounds, output).is_none());
    }
    for other in other_lengths(output) {
        assert!(derive(commitment, public_input, children, rounds, &other).is_none());
    }
    let mut fewer = rounds.clone();
    fewer.pop();
    let mut more = rounds.clone();
    more.push(rounds[0].clone());
    let mut shorter = rounds.clone();
    shorter[0].pop();
    let mut longer = rounds.clone();
    longer[0].push([0, 0]);
    for other in [fewer, more, shorter, longer] {
        assert!(derive(commitment, public_input, children, &other, output).is_none());
    }
}

type Derived = (Vec<[u64; 2]>, [u64; 16]);

#[test]
fn the_light_transcript_absorbs_the_lean_pieces() {
    let transcript = lean(0, nightstream_fprime::LIGHT_DOMAIN);
    assert!(transcript.children.is_empty());
    check(&transcript, |commitment, public_input, _, rounds, output| {
        let derived = derive_light_transcript(commitment, public_input, rounds, output).ok()?;
        Some((derived.round_point().to_vec(), derived.outgoing_state()))
    });
}

#[test]
fn the_refresh_transcript_absorbs_the_lean_pieces() {
    let transcript = lean(1, nightstream_fprime::REFRESH_DOMAIN);
    check(&transcript, |commitment, public_input, children, rounds, output| {
        let derived = derive_refresh_transcript(commitment, public_input, children, rounds, output).ok()?;
        Some((derived.round_point().to_vec(), derived.outgoing_state()))
    });
    for other in other_lengths(&transcript.children) {
        let Transcript {
            commitment,
            public_input,
            rounds,
            output,
            ..
        } = &transcript;
        assert!(derive_refresh_transcript(commitment, public_input, &other, rounds, output).is_err());
    }
}
