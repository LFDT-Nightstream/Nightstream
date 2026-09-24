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

fn field(word: u64) -> F {
    assert!(word < MODULUS, "fixture field value must be canonical");
    F::from_u64(word)
}

fn fixture() -> Fixture {
    let fixture: Fixture = serde_json::from_str(include_str!("fixtures/pi-rlc-wide-lean.json"))
        .expect("Lean whole-vector sampler fixture");
    assert_eq!((fixture.schema, fixture.modulus, fixture.degree), (1, MODULUS, D));
    fixture
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
fn one_window_schedule_matches_lean_transcript_states() {
    for case in fixture().transcripts {
        let mut transcript = Poseidon2Transcript::from_state_and_absorbed(case.initial.map(field), 0);
        for (index, step) in case.steps.into_iter().enumerate() {
            assert_eq!(step.source, index as u64);
            transcript.absorb_v1_1(&[F::from_u64(4), F::from_u64(step.source)]);
            assert_eq!(transcript.state().map(|value| value.as_canonical_u64()), step.entered);
            let draw = transcript.squeeze_digest_v1_1();
            assert_eq!(draw.map(|value| value.as_canonical_u64()), step.draw);
            assert_eq!(decode_pi_rlc_wide_coefficients(&draw).as_slice(), step.coefficients);
            assert_eq!(transcript.state().map(|value| value.as_canonical_u64()), step.outgoing);
            assert_eq!(transcript.absorbed(), 0);
        }
        assert_eq!(transcript.state().map(|value| value.as_canonical_u64()), case.r#final);
    }
}
