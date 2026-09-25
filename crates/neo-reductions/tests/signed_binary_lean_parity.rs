//! Exhaustive strict-B scalar parity, with matrix order and norm rejection.

use std::io::{BufRead, BufReader};

use neo_ccs::Mat;
use neo_math::{D, F};
use neo_reductions::common::split_b_matrix_k_with_nonzero_flags;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

type SplitVector = (u64, Vec<u64>);

fn check_block(vectors: &[SplitVector], digit_count: usize, base: u32) {
    // A ring coefficient column is the natural block size for this primitive.
    // Two columns also expose accidental row/column transposition.
    let rows = vectors.len().div_ceil(2);
    let mut values: Vec<F> = vectors
        .iter()
        .map(|(value, _)| F::from_u64(*value))
        .collect();
    values.resize(rows * 2, F::ZERO);
    let input = Mat::from_row_major(rows, 2, values);
    let (digits, nonzero) = split_b_matrix_k_with_nonzero_flags(&input, digit_count, base).expect("strict-B input");
    assert_eq!(digits.len(), digit_count);
    assert_eq!(nonzero.len(), digit_count);
    for index in 0..digit_count {
        let mut any_nonzero = false;
        for (position, (value, expected)) in vectors.iter().enumerate() {
            assert_eq!(expected.len(), digit_count);
            let expected_digit = expected[index];
            assert!(expected_digit < F::ORDER_U64, "canonical Lean digit");
            let actual = digits[index][(position / 2, position % 2)].as_canonical_u64();
            assert_eq!(actual, expected_digit, "value={value}, digit={index}");
            any_nonzero |= expected_digit != 0;
        }
        if vectors.len() % 2 != 0 {
            assert_eq!(digits[index][(rows - 1, 1)], F::ZERO, "padding stays zero");
        }
        assert_eq!(nonzero[index], any_nonzero, "digit {index} nonzero flag");
    }
}

#[test]
#[ignore = "fresh Lean vectors required; run scripts/check_fprime_foundation_parity.sh"]
fn active_lean_signed_binary_matches_runtime() {
    let path: std::path::PathBuf = serde_json::from_reader(std::io::stdin()).expect("Lean fixture path as JSON");
    let mut lines = BufReader::new(std::fs::File::open(path).expect("open Lean split vectors")).lines();
    let header: Vec<u64> =
        serde_json::from_str(&lines.next().expect("header").expect("read header")).expect("decode header");
    // The owner-approved Nightstream Goldilocks profile.
    assert_eq!(header, [F::ORDER_U64, 2, 16, 1 << 16]);
    let [_, base, digit_count, bound]: [u64; 4] = header.try_into().unwrap();
    let mut vectors = Vec::with_capacity(D * 2);
    for magnitude in 0..bound {
        for value in [magnitude, F::ORDER_U64 - magnitude] {
            if value == F::ORDER_U64 {
                continue; // Zero has one canonical encoding.
            }
            let vector: SplitVector = serde_json::from_str(
                &lines
                    .next()
                    .expect("complete strict-B domain")
                    .expect("read vector"),
            )
            .expect("decode split vector");
            assert_eq!(vector.0, value, "canonical input order and coverage");
            vectors.push(vector);
            if vectors.len() == D * 2 {
                check_block(&vectors, digit_count as usize, base as u32);
                vectors.clear();
            }
        }
    }
    check_block(&vectors, digit_count as usize, base as u32);

    for magnitude in [bound, bound + 1, F::ORDER_U64 / 2] {
        for value in [magnitude, F::ORDER_U64 - magnitude] {
            let vector: SplitVector = serde_json::from_str(
                &lines
                    .next()
                    .expect("rejection case")
                    .expect("read rejection"),
            )
            .expect("decode rejection");
            assert_eq!(vector, (value, vec![]), "Lean rejects the norm boundary");
            // Place the bad value after a valid value, to check whole-input rejection.
            let input = Mat::from_row_major(1, 2, vec![F::ONE, F::from_u64(value)]);
            assert!(
                split_b_matrix_k_with_nonzero_flags(&input, digit_count as usize, base as u32).is_err(),
                "Rust accepted {value}"
            );
        }
    }
    assert!(lines.next().is_none(), "unexpected extra split vector");

    // The virtual-zero optimization must have the same result as Lean's zero.
    let zero = Mat::virtual_constant(D, 1, F::ZERO);
    let (digits, flags) = split_b_matrix_k_with_nonzero_flags(&zero, digit_count as usize, base as u32).unwrap();
    assert_eq!(digits.len(), digit_count as usize);
    assert_eq!(flags, vec![false; digit_count as usize]);
    for digit in digits {
        assert_eq!(digit, zero);
    }
}
