use super::{coefficient_pair, expanded_key_bytes, split_pair};
use crate::nightstream_fprime_setup::{
    coefficient_block, production_key_pair, MAX_MESSAGE_COLUMNS, PRODUCTION_SEED, PRODUCTION_VERIFIER_ROWS,
};
use neo_math::{signed_sums::SignedShiftSums, F};
use p3_field::PrimeCharacteristicRing;

#[test]
fn paired_key_coefficients_match_independent_shake128() {
    for seed in [PRODUCTION_SEED, [0; 32], core::array::from_fn(|i| i as u8)] {
        for row in [0, 1, PRODUCTION_VERIFIER_ROWS as u32 - 1, u32::MAX] {
            for columns in [
                [0, 1],
                [65_535, 65_536],
                [0, MAX_MESSAGE_COLUMNS - 1],
                [u64::MAX, u64::MAX],
            ] {
                for (key, column) in split_pair(&seed, row, columns).iter().zip(columns) {
                    let mut sum = SignedShiftSums::zero();
                    sum.add_signed_units(key, 1, 0);
                    assert_eq!(sum.reduce(), coefficient_block(&seed, row, column).map(F::from_u64));
                }
                assert_eq!(
                    coefficient_pair(&seed, row, columns),
                    columns.map(|column| coefficient_block(&seed, row, column)),
                    "row {row}, columns {columns:?}"
                );
            }
        }
    }
}

/// Each path to a key element adds the 2,048 output bytes of the element,
/// from each thread. Other tests of this binary expand elements too, so the
/// count has a lower bound only.
#[test]
fn each_expansion_adds_its_output_bytes_to_the_count() {
    const ELEMENT_BYTES: u64 = 2048;
    let before = expanded_key_bytes();
    coefficient_block(&PRODUCTION_SEED, 0, 0);
    assert!(expanded_key_bytes() - before >= ELEMENT_BYTES);
    let before = expanded_key_bytes();
    production_key_pair(1, [2, 3]);
    assert!(expanded_key_bytes() - before >= 2 * ELEMENT_BYTES);
    let before = expanded_key_bytes();
    std::thread::scope(|scope| {
        for row in 0..4 {
            scope.spawn(move || split_pair(&PRODUCTION_SEED, row, [5, 6]));
        }
    });
    assert!(expanded_key_bytes() - before >= 8 * ELEMENT_BYTES);
}
