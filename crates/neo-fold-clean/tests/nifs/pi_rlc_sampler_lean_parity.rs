//! Exact wide PiRLC sampler parity against the shared Lean vectors.

use std::{fs, path::PathBuf};

use neo_fold_clean::engine::{optimized, paper_exact};
use neo_fold_clean::paper::params::Params;
use neo_math::{D, F};
use neo_reductions::common::decode_pi_rlc_wide_coefficients;
use neo_transcript::Poseidon2Transcript;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde::Deserialize;

const MODULUS: u64 = 0xffff_ffff_0000_0001;

#[derive(Deserialize)]
struct Fixture {
    schema: u64,
    modulus: u64,
    degree: usize,
    boundaries: Vec<Boundary>,
    transcripts: Vec<Transcript>,
}

#[derive(Deserialize)]
struct Boundary {
    integer: String,
    draw: [u64; 4],
    coefficients: Vec<i8>,
}

#[derive(Deserialize)]
struct Transcript {
    initial: [u64; 8],
    steps: Vec<Step>,
    r#final: [u64; 8],
}

#[derive(Deserialize)]
struct Step {
    source: u64,
    entered: [u64; 8],
    draw: [u64; 4],
    coefficients: Vec<i8>,
    outgoing: [u64; 8],
}

fn fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../neo-reductions/tests/fixtures/pi-rlc-wide-lean.json");
    let fixture: Fixture = serde_json::from_slice(&fs::read(path).expect("shared Lean PiRLC wide fixture"))
        .expect("canonical Lean PiRLC wide JSON");
    assert_eq!((fixture.schema, fixture.modulus, fixture.degree), (1, MODULUS, D));
    fixture
}

fn field(word: u64) -> F {
    assert!(word < MODULUS, "canonical Goldilocks word");
    F::from_u64(word)
}

fn state_words(transcript: &Poseidon2Transcript) -> [u64; 8] {
    transcript.state().map(|value| value.as_canonical_u64())
}

#[test]
fn whole_vector_decoder_matches_lean_modular_boundaries() {
    for case in fixture().boundaries {
        let actual = decode_pi_rlc_wide_coefficients(&case.draw.map(field));
        assert_eq!(actual.as_slice(), case.coefficients, "X={}", case.integer);
        assert!(actual
            .into_iter()
            .all(|coefficient| (-2..=2).contains(&coefficient)));
    }
}

#[test]
fn one_window_transcript_and_both_engines_match_lean() {
    let params = Params::production();
    for case in fixture().transcripts {
        let mut replay = Poseidon2Transcript::from_state_and_absorbed(case.initial.map(field), 0);
        for (source, step) in case.steps.iter().enumerate() {
            assert_eq!(step.source, source as u64, "exact source domain index");
            replay.absorb_v1_1(&[F::from_u64(4), F::from_u64(step.source)]);
            assert_eq!(state_words(&replay), step.entered);
            let draw = replay.squeeze_digest_v1_1();
            assert_eq!(draw.map(|value| value.as_canonical_u64()), step.draw);
            assert_eq!(decode_pi_rlc_wide_coefficients(&draw).as_slice(), step.coefficients);
            assert_eq!(state_words(&replay), step.outgoing);
            assert_eq!(replay.absorbed(), 0);
        }
        assert_eq!(state_words(&replay), case.r#final);

        let mut optimized_transcript = Poseidon2Transcript::from_state_and_absorbed(case.initial.map(field), 0);
        let optimized_rhos = optimized::sample_rho_n(&mut optimized_transcript, &params, case.steps.len())
            .expect("total optimized PiRLC wide sampler");
        let mut paper_transcript = Poseidon2Transcript::from_state_and_absorbed(case.initial.map(field), 0);
        let paper_rhos = paper_exact::sample_rho_n(&mut paper_transcript, &params, case.steps.len())
            .expect("total reference PiRLC wide sampler");

        assert_eq!(state_words(&optimized_transcript), case.r#final);
        assert_eq!(state_words(&paper_transcript), case.r#final);
        assert_eq!(optimized_rhos.len(), case.steps.len());
        assert_eq!(paper_rhos.len(), case.steps.len());
        for ((optimized, paper), step) in optimized_rhos.iter().zip(&paper_rhos).zip(&case.steps) {
            assert_eq!(optimized.as_mat(), paper.as_mat());
            assert_eq!(step.coefficients.len(), D);
            for (lane, coefficient) in step.coefficients.iter().enumerate() {
                assert_eq!(optimized.as_mat()[(lane, 0)], F::from_i8(*coefficient));
            }
        }
    }
}
