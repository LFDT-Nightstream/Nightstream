//! Child commitment data for the next PiCCS parent fixture.
//! The selected package fixes the key. Full child evaluations and the
//! independent parent-assignment check remain separate obligations.

use std::{fs, path::Path, time::Instant};

use neo_ajtai::nightstream_fprime_setup::{
    commit_production_signed_units, production_authority_words, PRODUCTION_CARRIER_WIDTH, PRODUCTION_VERIFIER_ROWS,
};
use neo_fold_clean::{engine::optimized, paper::params::Params};
use neo_math::{Rq, D, F};
use neo_transcript::Poseidon2Transcript;
use nightstream_fprime::load_per_application_package;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use rayon::prelude::*;
use serde_json::{json, Value};

use super::folded_opening::{signed_digits, DIGITS, PARENT_BOUND};

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

fn selected_metadata(candidate: &Path, expected: [u64; 4], cache: &Path) -> Value {
    let bytes = fs::read(candidate).expect("canonical candidate package");
    let package = load_per_application_package(&bytes, expected).expect("selected package identity");
    drop(bytes);
    let binding = package
        .production_verifier_binding()
        .expect("selected verifier binding");
    assert_eq!(
        binding.verifier_context().commitment_key_words(),
        production_authority_words()
    );
    assert_eq!(package.logical_column_count().div_ceil(D) * D, PRODUCTION_CARRIER_WIDTH);
    let meta = read(&cache.join("folded.json"));
    assert_eq!(meta.as_array().expect("folded metadata").len(), 13);
    assert_eq!(meta[0], 1);
    assert_eq!(meta[1], json!(expected));
    assert_eq!(meta[2], json!(binding.verifier_context().digest()));
    assert_eq!(meta[3], json!(package.logical_column_count()));
    assert_eq!(meta[4], json!(PRODUCTION_CARRIER_WIDTH));
    let bound = meta[8].as_u64().expect("folded bound");
    assert!(bound < PARENT_BOUND as u64);
    let point: Vec<[u64; 2]> = serde_json::from_value(meta[12].clone()).expect("child point");
    assert_eq!(point.len(), 28);
    assert!(point.iter().flatten().all(|&word| word < MODULUS));
    meta
}

pub fn generate(candidate: &Path, expected: [u64; 4], cache: &Path, child: usize, output: &Path) {
    let started = Instant::now();
    assert!(child < DIGITS);
    assert!(!output.exists(), "use a fresh external child commitment file");
    let meta = selected_metadata(candidate, expected, cache);
    let bound = meta[8].as_u64().expect("folded bound");
    let bytes = fs::read(cache.join("folded.i16")).expect("complete folded carrier");
    assert_eq!(bytes.len(), PRODUCTION_CARRIER_WIDTH * 2);
    let carrier: Vec<i8> = bytes
        .par_chunks_exact(2)
        .map(|word| {
            let value = i16::from_le_bytes(word.try_into().expect("signed coefficient"));
            assert!(u64::from(value.unsigned_abs()) <= bound);
            signed_digits(i32::from(value)).expect("strict parent bound")[child]
        })
        .collect();
    drop(bytes);
    let public: Vec<i8> = serde_json::from_value(meta[11][child].clone()).expect("child public digits");
    assert_eq!(public.len(), PUBLIC);
    assert_eq!(&carrier[..PUBLIC], public);
    let nonzero = carrier.par_iter().filter(|&&value| value != 0).count();
    println!(
        "child={child} signed coordinates={nonzero} input_time={:?}",
        started.elapsed()
    );
    let commitment = commit_production_signed_units(&carrier).expect("selected-key child commitment");
    assert_eq!((commitment.d, commitment.kappa), (D, PRODUCTION_VERIFIER_ROWS as usize));
    let commitment: Vec<u64> = commitment
        .data
        .iter()
        .map(|value| value.as_canonical_u64())
        .collect();
    let public: Vec<u64> = public
        .into_iter()
        .map(|value| signed(i32::from(value)).as_canonical_u64())
        .collect();
    let result = json!([1, expected, meta[2], child, meta[12], public, commitment]);
    let mut encoded = serde_json::to_vec(&result).expect("canonical child commitment JSON");
    encoded.push(b'\n');
    fs::write(output, encoded).expect("child commitment sink");
    println!(
        "child_commitment={child} coefficients={COMMITMENT} elapsed={:?}",
        started.elapsed()
    );
    println!("independent_child_openings_and_parent_assignment=still_required");
}

/// Check every native coefficient once before deriving the zero high digits.
pub fn generate_zero_children(candidate: &Path, expected: [u64; 4], cache: &Path, output: &Path) {
    let started = Instant::now();
    let meta = selected_metadata(candidate, expected, cache);
    let bound = meta[8].as_u64().expect("folded bound");
    let first_zero = (u64::BITS - bound.leading_zeros()) as usize;
    assert!(first_zero <= DIGITS);
    let bytes = fs::read(cache.join("folded.i16")).expect("complete folded carrier");
    assert_eq!(bytes.len(), PRODUCTION_CARRIER_WIDTH * 2);
    bytes.par_chunks_exact(2).for_each(|word| {
        let value = i16::from_le_bytes(word.try_into().expect("signed coefficient"));
        assert!(u64::from(value.unsigned_abs()) <= bound);
        assert!(
            signed_digits(i32::from(value)).expect("strict parent bound")[first_zero..]
                .iter()
                .all(|&digit| digit == 0)
        );
    });
    for child in first_zero..DIGITS {
        assert!(
            !output.join(format!("child-{child}.json")).exists(),
            "use fresh zero-child sinks"
        );
        let public: Vec<i8> = serde_json::from_value(meta[11][child].clone()).expect("child public digits");
        assert_eq!(public, vec![0; PUBLIC]);
    }
    for child in first_zero..DIGITS {
        let result = json!([
            1,
            expected,
            meta[2],
            child,
            meta[12],
            vec![0u64; PUBLIC],
            vec![0u64; COMMITMENT]
        ]);
        let mut encoded = serde_json::to_vec(&result).expect("zero child JSON");
        encoded.push(b'\n');
        fs::write(output.join(format!("child-{child}.json")), encoded).expect("zero child sink");
    }
    println!("zero_child_commitments=passed first={first_zero} count={} carrier={PRODUCTION_CARRIER_WIDTH} bound={bound} elapsed={:?}", DIGITS - first_zero, started.elapsed());
    println!("independent_child_openings_and_parent_assignment=still_required");
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

pub fn check(candidate: &Path, expected: [u64; 4], base: &Path, cache: &Path, children: &Path, output: &Path) {
    let started = Instant::now();
    assert!(!output.exists(), "use a fresh external recomposition record");
    let meta = selected_metadata(candidate, expected, cache);
    let opening = read(&base.join("opening.json"));
    assert_eq!(opening[0], 1);
    assert_eq!(opening[1], json!(expected));
    assert_eq!(opening[2], meta[2]);
    assert_eq!(opening[3], meta[3]);
    assert_eq!(opening[4], meta[4]);
    let state: [u64; 8] = serde_json::from_value(meta[5].clone()).expect("PiCCS outgoing state");
    let mut transcript = Poseidon2Transcript::from_state_and_absorbed(state.map(field), 0);
    let sampled = optimized::sample_rho_n(&mut transcript, &Params::production(), 17).expect("existing sampler");
    let rhos: Vec<Vec<i8>> = serde_json::from_value(meta[6].clone()).expect("recorded coefficients");
    assert_eq!(rhos.len(), 17);
    for (rho, recorded) in sampled.iter().zip(&rhos) {
        assert_eq!(recorded.len(), D);
        for (row, &coefficient) in recorded.iter().enumerate() {
            assert!((-2..=2).contains(&coefficient));
            assert_eq!(rho.as_mat()[(row, 0)], signed(i32::from(coefficient)));
        }
    }
    assert_eq!(json!(transcript.state().map(|value| value.as_canonical_u64())), meta[7]);
    let rho: [i8; D] = rhos[0].clone().try_into().expect("first ring challenge");
    let base_commitment: Vec<u64> = serde_json::from_value(opening[10].clone()).expect("checked base commitment");
    let base_public: Vec<u64> = serde_json::from_value(opening[9].clone()).expect("checked base public input");
    assert_eq!((base_commitment.len(), base_public.len()), (COMMITMENT, PUBLIC));
    let expected_commitment = product(&rho, &base_commitment);
    let expected_public = product(&rho, &base_public);
    let parent_public: Vec<i32> = serde_json::from_value(meta[10].clone()).expect("folded public input");
    assert_eq!(
        parent_public
            .iter()
            .copied()
            .map(signed)
            .collect::<Vec<_>>(),
        expected_public
    );
    let mut commitment_sum = vec![F::ZERO; COMMITMENT];
    let mut public_sum = vec![F::ZERO; PUBLIC];
    for child in 0..DIGITS {
        let data = read(&children.join(format!("child-{child}.json")));
        assert_eq!(data.as_array().expect("child record").len(), 7);
        assert_eq!(data[0], 1);
        assert_eq!(data[1], json!(expected));
        assert_eq!(data[2], meta[2]);
        assert_eq!(data[3], json!(child));
        assert_eq!(data[4], meta[12]);
        let public: Vec<u64> = serde_json::from_value(data[5].clone()).expect("child public input");
        let commitment: Vec<u64> = serde_json::from_value(data[6].clone()).expect("child commitment");
        assert_eq!((public.len(), commitment.len()), (PUBLIC, COMMITMENT));
        assert!(public
            .iter()
            .all(|&value| matches!(value, 0 | 1) || value == MODULUS - 1));
        let expected_digits: Vec<i8> = serde_json::from_value(meta[11][child].clone()).expect("public digits");
        assert_eq!(
            public.iter().copied().map(field).collect::<Vec<_>>(),
            expected_digits
                .into_iter()
                .map(|value| signed(i32::from(value)))
                .collect::<Vec<_>>()
        );
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
        "all child commitments recompose to rho times the fresh commitment"
    );
    assert_eq!(public_sum, expected_public, "all child public inputs recompose");
    let result = json!([
        1,
        expected,
        meta[2],
        meta[12],
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
