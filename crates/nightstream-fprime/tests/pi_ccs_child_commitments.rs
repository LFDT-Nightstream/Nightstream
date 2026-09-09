//! Independent commitment openings for the sixteen actual folded digits.
//! Each row shares indexed key expansion across the children. The full check
//! decodes the raw parent once, without the fixture generator or its ring code.

use std::{fs, path::Path, path::PathBuf, time::Instant};

use nightstream_fprime::load_per_application_package;
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};

#[allow(dead_code)]
#[path = "support/pi_ccs_commitment.rs"]
mod commitment;

const DEGREE: usize = 54;
const DIGITS: usize = 16;
const PUBLIC: usize = 270;
const MODULUS: u128 = 18_446_744_069_414_584_321;
type Products = [[i128; 2 * DEGREE - 1]; DIGITS];

#[derive(Deserialize)]
struct ChildInputs {
    package: PathBuf,
    structural_identity: [u64; 4],
    folded_cache: PathBuf,
    children_dir: PathBuf,
    setup_fixture: PathBuf,
}

fn read(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).expect("child opening input")).expect("child opening JSON")
}

fn evaluate_row(seed: &[u8; 32], row: u32, bytes: &[u8], bound: u64) -> [[u64; DEGREE]; DIGITS] {
    assert!(bound < 1 << DIGITS);
    assert_eq!(bytes.len() % (2 * DEGREE), 0);
    let native_width = bytes.len() / 2;
    assert!((native_width as u128)
        .checked_mul(MODULUS - 1)
        .and_then(|value| value.checked_mul(3))
        .is_some_and(|value| value < i128::MAX as u128));
    let mut products = bytes
        .par_chunks_exact(2 * DEGREE)
        .enumerate()
        .fold(
            || -> Box<Products> { Box::new([[0; 2 * DEGREE - 1]; DIGITS]) },
            |mut sums, (block, words)| {
                let parent: [i32; DEGREE] = std::array::from_fn(|lane| {
                    let value = i16::from_le_bytes(words[2 * lane..2 * lane + 2].try_into().unwrap());
                    assert!(
                        u64::from(value.unsigned_abs()) <= bound,
                        "actual parent coefficient bound"
                    );
                    i32::from(value)
                });
                if parent.iter().all(|&value| value == 0) {
                    return sums;
                }
                for lane in 0..DEGREE {
                    let key = i128::from(commitment::coefficient(seed, row, block as u64, lane as u32));
                    for (power, &value) in parent.iter().enumerate() {
                        let coefficient = if value < 0 { -key } else { key };
                        let mut digits = value.unsigned_abs();
                        while digits != 0 {
                            let child = digits.trailing_zeros() as usize;
                            assert!(child < DIGITS);
                            sums[child][lane + power] += coefficient;
                            digits &= digits - 1;
                        }
                    }
                }
                sums
            },
        )
        .reduce(
            || Box::new([[0; 2 * DEGREE - 1]; DIGITS]),
            |mut left, right| {
                for (left, right) in left.iter_mut().zip(right.iter()) {
                    for (left, right) in left.iter_mut().zip(right) {
                        *left += right;
                    }
                }
                left
            },
        );
    for child in products.iter_mut() {
        for power in (DEGREE..child.len()).rev() {
            let value = child[power];
            child[power - DEGREE] -= value;
            child[power - DEGREE / 2] -= value;
        }
    }
    std::array::from_fn(|child| std::array::from_fn(|lane| products[child][lane].rem_euclid(MODULUS as i128) as u64))
}

#[test]
#[ignore = "requires external child-opening paths as JSON on stdin; run under the 300-second cap"]
fn independent_actual_child_commitments() {
    let started = Instant::now();
    let inputs: ChildInputs = serde_json::from_reader(std::io::stdin().lock()).expect("child opening paths");
    let bytes = fs::read(&inputs.package).expect("selected canonical package");
    let package = load_per_application_package(&bytes, inputs.structural_identity).expect("selected package identity");
    let binding = package
        .production_verifier_binding()
        .expect("selected binding");
    let context = binding.verifier_context().digest();
    let meta = read(&inputs.folded_cache.join("folded.json"));
    assert_eq!(meta.as_array().expect("folded metadata").len(), 13);
    assert_eq!(meta[0], 1);
    assert_eq!(meta[1], json!(inputs.structural_identity));
    assert_eq!(meta[2], json!(context));
    assert_eq!(meta[3], json!(package.logical_column_count()));
    let native_width = package.logical_column_count().div_ceil(DEGREE) * DEGREE;
    assert_eq!(meta[4], json!(native_width));
    let point: Vec<[u64; 2]> = serde_json::from_value(meta[12].clone()).expect("child point");
    assert_eq!(point.len(), 28);
    assert!(point
        .iter()
        .flatten()
        .all(|&word| u128::from(word) < MODULUS));
    let bound = meta[8].as_u64().expect("parent bound");
    assert!(bound < 1 << DIGITS);

    let setup = read(&inputs.setup_fixture);
    assert_eq!(setup[0], 3, "Lean wide256 setup schema");
    let authority: Vec<u64> = serde_json::from_value(setup[6].clone()).expect("Lean setup authority");
    assert_eq!(authority.len(), 73);
    assert_eq!(authority[0], 37);
    assert_eq!(
        &authority[1..38],
        b"nightstream-ajtai-chacha20-wide256-v1"
            .iter()
            .map(|&byte| u64::from(byte))
            .collect::<Vec<_>>()
    );
    assert_eq!(&authority[38..41], &[22, (native_width / DEGREE) as u64, 32]);
    assert_eq!(binding.verifier_context().commitment_key_words(), authority);
    let seed: [u8; 32] = authority[41..]
        .iter()
        .map(|&word| u8::try_from(word).expect("seed byte"))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    assert_eq!(setup[4], json!(seed));
    let test_seed: [u8; 32] = serde_json::from_value(setup[2].clone()).expect("Lean RFC seed");
    let expected_block: [u32; 16] = serde_json::from_value(setup[3].clone()).expect("Lean RFC block");
    assert_eq!(
        commitment::block_words(&test_seed, 0x09000000, 0x4a000000, 1),
        expected_block
    );
    let samples: Vec<[u64; 4]> = serde_json::from_value(setup[5].clone()).expect("Lean indexed setup samples");
    for [row, block, lane, expected] in samples {
        assert_eq!(
            commitment::coefficient(&seed, row.try_into().unwrap(), block, lane.try_into().unwrap()),
            expected
        );
    }
    drop(binding);
    drop(package);
    drop(bytes);

    let parent = fs::read(inputs.folded_cache.join("folded.i16")).expect("raw complete folded carrier");
    assert_eq!(parent.len(), native_width * 2);
    let mut expected = vec![vec![0u64; 22 * DEGREE]; DIGITS];
    for (child, expected) in expected.iter_mut().enumerate() {
        let record = read(&inputs.children_dir.join(format!("child-{child}.json")));
        assert_eq!(record.as_array().expect("child record").len(), 7);
        assert_eq!(record[0], 1);
        assert_eq!(record[1], json!(inputs.structural_identity));
        assert_eq!(record[2], json!(context));
        assert_eq!(record[3], json!(child));
        assert_eq!(record[4], json!(point));
        let public: Vec<u64> = serde_json::from_value(record[5].clone()).expect("child public input");
        assert_eq!(public.len(), PUBLIC);
        for (column, &word) in public.iter().enumerate() {
            let value = i16::from_le_bytes(parent[2 * column..2 * column + 2].try_into().unwrap());
            let digit = (u64::from(value.unsigned_abs()) / (1u64 << child)) % 2;
            let canonical = if value < 0 && digit != 0 {
                MODULUS as u64 - digit
            } else {
                digit
            };
            assert_eq!(word, canonical, "actual child {child} public coordinate {column}");
        }
        let commitment: Vec<u64> = serde_json::from_value(record[6].clone()).expect("complete child commitment");
        assert_eq!(commitment.len(), 22 * DEGREE);
        assert!(commitment.iter().all(|&word| u128::from(word) < MODULUS));
        *expected = commitment;
    }
    println!(
        "independent_child_commitments rows=22 children={DIGITS} input_time={:?}",
        started.elapsed()
    );
    for row in 0..22 {
        let actual = evaluate_row(&seed, row as u32, &parent, bound);
        for (child, value) in actual.iter().enumerate() {
            assert_eq!(
                value.as_slice(),
                &expected[child][row * DEGREE..(row + 1) * DEGREE],
                "every coefficient of child {child}, commitment row {row}"
            );
        }
        println!(
            "independent_child_commitment_row={row} passed children={DIGITS} coefficients={} elapsed={:?}",
            DIGITS * DEGREE,
            started.elapsed()
        );
    }
    println!("independent_child_commitments=passed rows=22 children={DIGITS} coefficients={} carrier={native_width} elapsed={:?}", 22 * DIGITS * DEGREE, started.elapsed());
}
