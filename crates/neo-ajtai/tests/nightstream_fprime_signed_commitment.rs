//! Exact indexed setup and signed-unit commitment conformance.
//!
//! The short ordinary-reference sum removes only zero message blocks. Every
//! retained key element keeps its original production row and column index.

use neo_ajtai::{
    nightstream_fprime_setup::{
        coefficient, coefficient_block, commit_production_signed_units, PRODUCTION_CARRIER_WIDTH,
        PRODUCTION_MESSAGE_COLUMNS, PRODUCTION_SEED, PRODUCTION_VERIFIER_ROWS,
    },
    try_commit, AjtaiError, Commitment, PP,
};
use neo_math::ring::{Rq, D};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

#[test]
fn streamed_key_blocks_match_all_lanes_of_authoritative_setup_cases() {
    // Unique row/column pairs from nightstream-fprime-ajtai-setup-v1-parity.
    for (row, block) in [(0, 0), (1, 32_768), (21, PRODUCTION_MESSAGE_COLUMNS - 1)] {
        let streamed = coefficient_block(&PRODUCTION_SEED, row, block);
        for lane in 0..D {
            assert_eq!(
                streamed[lane],
                coefficient(&PRODUCTION_SEED, row, block, lane as u32),
                "indexed row {row}, block {block}, lane {lane}"
            );
        }
    }
    // The existing RFC-8439 fixture uses this seed and nonce. Checking all
    // lanes also crosses every four-block SIMD buffer boundary up to lane 53.
    let seed = core::array::from_fn(|index| index as u8);
    let streamed = coefficient_block(&seed, 0x0900_0000, 0x4a00_0000);
    for lane in 0..D {
        assert_eq!(
            streamed[lane],
            coefficient(&seed, 0x0900_0000, 0x4a00_0000, lane as u32)
        );
    }
}

#[test]
fn signed_commitment_matches_the_ordinary_sum_under_the_same_key() {
    let selected = [0, 32_768, PRODUCTION_MESSAGE_COLUMNS - 1];
    let mut carrier = vec![0_i8; PRODUCTION_CARRIER_WIDTH];
    // X^0, -X^27 and X^53 exercise both signs and reduction across 54 and 81.
    // The final carrier coordinate participates; this is a ring test, not an
    // application witness whose alignment coordinates must be zero.
    for ((block, lane), value) in selected.into_iter().zip([0, 27, 53]).zip([1, -1, 1]) {
        carrier[block as usize * D + lane] = value;
    }
    let key = PP {
        d: D,
        kappa: PRODUCTION_VERIFIER_ROWS as usize,
        m: selected.len(),
        m_rows: (0..PRODUCTION_VERIFIER_ROWS)
            .map(|row| {
                selected
                    .iter()
                    .copied()
                    .map(|block| {
                        Rq(core::array::from_fn(|lane| {
                            Goldilocks::from_u64(coefficient(&PRODUCTION_SEED, row as u32, block, lane as u32))
                        }))
                    })
                    .collect()
            })
            .collect(),
    };
    let message = selected
        .iter()
        .copied()
        .flat_map(|block| {
            carrier[block as usize * D..(block as usize + 1) * D]
                .iter()
                .map(|value| match value {
                    -1 => -Goldilocks::ONE,
                    0 => Goldilocks::ZERO,
                    1 => Goldilocks::ONE,
                    _ => unreachable!("test signed unit"),
                })
        })
        .collect::<Vec<_>>();
    let expected = try_commit(&key, &message).expect("ordinary exact-key ring sum");
    let actual = commit_production_signed_units(&carrier).expect("complete production carrier commitment");
    assert_eq!(actual, expected);
    assert_ne!(actual, Commitment::zeros(D, PRODUCTION_VERIFIER_ROWS as usize));

    carrier.fill(0);
    assert_eq!(
        commit_production_signed_units(&carrier).expect("zero carrier"),
        Commitment::zeros(D, PRODUCTION_VERIFIER_ROWS as usize)
    );
}

#[test]
fn signed_commitment_rejects_wrong_lengths_and_out_of_range_units() {
    assert_eq!(
        commit_production_signed_units(&[]),
        Err(AjtaiError::SizeMismatch {
            expected: PRODUCTION_CARRIER_WIDTH,
            actual: 0,
        })
    );
    let mut carrier = vec![0_i8; PRODUCTION_CARRIER_WIDTH];
    assert!(matches!(
        commit_production_signed_units(&carrier[..carrier.len() - 1]),
        Err(AjtaiError::SizeMismatch { .. })
    ));
    carrier.push(0);
    assert!(matches!(
        commit_production_signed_units(&carrier),
        Err(AjtaiError::SizeMismatch { .. })
    ));
    carrier.pop();
    carrier[0] = 2;
    assert_eq!(
        commit_production_signed_units(&carrier),
        Err(AjtaiError::RangeViolation { value: 2, bound: 2 })
    );
    carrier[0] = 0;
    carrier[PRODUCTION_CARRIER_WIDTH - 1] = -2;
    assert_eq!(
        commit_production_signed_units(&carrier),
        Err(AjtaiError::RangeViolation { value: -2, bound: 2 })
    );
}
