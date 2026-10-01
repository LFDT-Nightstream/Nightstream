//! Recompute child commitments and public inputs from all 17 checked sources.
//! Raw child openings and parent rows have separate independent checks.

use std::{fs, path::Path, time::Instant};

use crate::folding::{kernels, Params};
use neo_ajtai::nightstream_fprime_setup::{
    production_authority_words, PRODUCTION_CARRIER_WIDTH, PRODUCTION_VERIFIER_ROWS,
};
use neo_math::{Rq, D, F};
use neo_transcript::Poseidon2Transcript;
use nightstream_fprime::load_per_application_package;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde_json::{json, Value};

const DIGITS: usize = 16;
const PARENT_BOUND: u64 = 1 << DIGITS;

const MODULUS: u64 = 0xffff_ffff_0000_0001;
const PUBLIC: usize = 270;
const COMMITMENT: usize = PRODUCTION_VERIFIER_ROWS as usize * D;

fn read(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).expect("fixture input")).expect("fixture JSON")
}

fn field(value: u64) -> F {
    assert!(value < MODULUS, "canonical field word");
    F::from_u64(value)
}

fn signed(value: i32) -> F {
    let magnitude = F::from_u64(u64::from(value.unsigned_abs()));
    if value < 0 {
        -magnitude
    } else {
        magnitude
    }
}

fn product(rho: &[i8; D], values: &[u64]) -> Vec<F> {
    assert_eq!(values.len() % D, 0);
    let scalar = Rq(rho.map(|value| signed(i32::from(value))));
    values
        .chunks_exact(D)
        .flat_map(|block| {
            let mut result = [F::ZERO; D];
            for (column, &value) in block.iter().enumerate() {
                let shifted = scalar.mul_by_monomial(column);
                for (target, coefficient) in result.iter_mut().zip(shifted.0) {
                    *target += field(value) * coefficient;
                }
            }
            result
        })
        .collect()
}

pub(super) fn check(
    candidate: &Path,
    expected: [u64; 4],
    input_path: &Path,
    lean_path: &Path,
    children_path: &Path,
    output: &Path,
) {
    let started = Instant::now();
    assert!(!output.exists(), "use a fresh external recomposition record");
    let bytes = fs::read(candidate).expect("canonical candidate package");
    let package = load_per_application_package(&bytes, expected).expect("selected package identity");
    let binding = package
        .production_verifier_binding()
        .expect("selected verifier binding");
    assert_eq!(
        binding.verifier_context().commitment_key_words(),
        production_authority_words()
    );
    assert_eq!(package.logical_column_count().div_ceil(D) * D, PRODUCTION_CARRIER_WIDTH);
    drop((package, bytes));
    let input = read(input_path);
    let lean = read(lean_path);
    let children = read(children_path);
    assert_eq!(input.as_array().unwrap().len(), 7);
    assert_eq!(input[0], 2);
    assert_eq!(lean.as_array().unwrap().len(), 6);
    assert_eq!(lean[0], 1);
    assert_eq!(lean[1], input, "complete PiCCS input");
    assert_eq!(lean[5][0], 1, "accepted PiCCS result");
    assert_eq!(children.as_array().unwrap().len(), 5);
    assert_eq!(children[0], lean[5][6], "all children use the checked PiCCS point");
    assert_eq!(children[1].as_array().unwrap().len(), DIGITS);
    assert_eq!(children[2].as_array().unwrap().len(), DIGITS);
    let state: [u64; 16] = serde_json::from_value(lean[5][14].clone()).expect("PiCCS outgoing state");
    let mut transcript = Poseidon2Transcript::from_state_and_absorbed(state.map(field), 0);
    let sampled = kernels::sample_rho_n(&mut transcript, &Params::production(), 17).expect("wide sampler");
    let mut expected_commitment = vec![F::ZERO; COMMITMENT];
    let mut expected_public = vec![F::ZERO; PUBLIC];
    for (source, challenge) in sampled.iter().enumerate() {
        let rho: [i8; D] = std::array::from_fn(|lane| {
            let value = challenge.as_mat()[(lane, 0)];
            (-2_i8..=2)
                .find(|&digit| value == signed(i32::from(digit)))
                .expect("centered digit")
        });
        let (commitment, public) = if source == 0 {
            (&input[1], &input[2])
        } else {
            (&input[6][1][source - 1], &input[6][2][source - 1])
        };
        let commitment: Vec<u64> = serde_json::from_value(commitment.clone()).unwrap();
        let public: Vec<u64> = serde_json::from_value(public.clone()).unwrap();
        assert_eq!((commitment.len(), public.len()), (COMMITMENT, PUBLIC));
        for (sum, value) in expected_commitment
            .iter_mut()
            .zip(product(&rho, &commitment))
        {
            *sum += value;
        }
        for (sum, value) in expected_public.iter_mut().zip(product(&rho, &public)) {
            *sum += value;
        }
    }
    let mut commitment_sum = vec![F::ZERO; COMMITMENT];
    let mut public_sum = vec![F::ZERO; PUBLIC];
    for child in 0..DIGITS {
        let commitment: Vec<u64> = serde_json::from_value(children[1][child].clone()).unwrap();
        let public: Vec<u64> = serde_json::from_value(children[2][child].clone()).unwrap();
        assert_eq!((commitment.len(), public.len()), (COMMITMENT, PUBLIC));
        for (word, parent) in public.iter().zip(&expected_public) {
            let parent = parent.as_canonical_u64();
            let negative = parent > MODULUS / 2;
            let magnitude = if negative { MODULUS - parent } else { parent };
            assert!(magnitude < PARENT_BOUND, "strict public parent bound");
            let bit = F::from_u64((magnitude >> child) & 1);
            assert_eq!(
                field(*word),
                if negative { -bit } else { bit },
                "canonical child public digit"
            );
        }
        let weight = F::from_u64(1u64 << child);
        for (target, value) in commitment_sum.iter_mut().zip(commitment) {
            *target += weight * field(value);
        }
        for (target, value) in public_sum.iter_mut().zip(public) {
            *target += weight * field(value);
        }
    }
    assert_eq!(
        commitment_sum, expected_commitment,
        "all child commitments recompose from all 17 sources"
    );
    assert_eq!(public_sum, expected_public, "all child public inputs recompose");
    let result = json!([
        1,
        expected,
        binding.verifier_context().digest(),
        children[0],
        expected_public
            .iter()
            .map(|value| value.as_canonical_u64())
            .collect::<Vec<_>>(),
        expected_commitment
            .iter()
            .map(|value| value.as_canonical_u64())
            .collect::<Vec<_>>()
    ]);
    let mut encoded = serde_json::to_vec(&result).expect("recomposition JSON");
    encoded.push(b'\n');
    fs::write(output, encoded).expect("recomposition sink");
    println!(
        "child_commitment_recomposition=passed children={DIGITS} commitments={COMMITMENT} public={PUBLIC} elapsed={:?}",
        started.elapsed()
    );
}
