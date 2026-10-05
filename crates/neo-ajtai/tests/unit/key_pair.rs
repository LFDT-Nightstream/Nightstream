use super::{coefficient_pair, split_pair};
use crate::nightstream_fprime_setup::{coefficient_block, MAX_MESSAGE_COLUMNS, PRODUCTION_SEED};
use neo_math::{signed_sums::SignedShiftSums, F};
use p3_field::PrimeCharacteristicRing;

#[test]
fn paired_key_coefficients_match_independent_shake128() {
    for seed in [PRODUCTION_SEED, [0; 32], core::array::from_fn(|i| i as u8)] {
        for row in [0, 1, 21, u32::MAX] {
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
