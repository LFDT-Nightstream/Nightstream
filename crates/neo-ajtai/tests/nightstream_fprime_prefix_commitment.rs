use neo_ajtai::{
    nightstream_fprime_setup::{
        coefficient, commit_production_signed_unit_matrix, commit_production_signed_unit_prefix_matrix,
        PRODUCTION_MESSAGE_COLUMNS, PRODUCTION_SEED, PRODUCTION_VERIFIER_ROWS,
    },
    AjtaiError, Commitment,
};
use neo_ccs::Mat;
use neo_math::{Rq, D};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

#[test]
fn prefix_commitment_equals_exact_suffix_zero_extension() {
    // Two blocks distinguish the first key address from its successor.
    // Both end lanes and both signs exercise the complete smaller carrier.
    let support = [
        (0, [(0, 1i8), (D / 2, -1), (D - 1, 1)]),
        (1, [(0, -1), (1, 1), (D - 1, -1)]),
    ];
    let columns = PRODUCTION_MESSAGE_COLUMNS as usize;
    let mut positive = vec![0u64; columns];
    let mut negative = vec![0u64; columns];
    let mut dense = Mat::zero(D, support.len(), Goldilocks::ZERO);
    for (block, entries) in support {
        for (lane, sign) in entries {
            if sign > 0 {
                positive[block] |= 1 << lane;
                dense[(lane, block)] = Goldilocks::ONE;
            } else {
                negative[block] |= 1 << lane;
                dense[(lane, block)] = -Goldilocks::ONE;
            }
        }
    }
    let prefix = Mat::compact_signed_unit_from_column_masks(
        D,
        support.len(),
        &positive[..support.len()],
        &negative[..support.len()],
    )
    .unwrap();
    let extended = Mat::compact_signed_unit_from_column_masks(D, columns, &positive, &negative).unwrap();
    drop((positive, negative));

    let actual = commit_production_signed_unit_prefix_matrix(&prefix).unwrap();
    assert_eq!(actual, commit_production_signed_unit_prefix_matrix(&dense).unwrap());
    assert_eq!(actual, commit_production_signed_unit_matrix(&extended).unwrap());
    assert_eq!(actual, commit_production_signed_unit_prefix_matrix(&extended).unwrap());

    // Independent scalar setup coefficients and ordinary ring products.
    let mut expected = Commitment::zeros(D, PRODUCTION_VERIFIER_ROWS as usize);
    for row in 0..PRODUCTION_VERIFIER_ROWS as usize {
        for (block, entries) in support {
            let key = Rq(std::array::from_fn(|lane| {
                Goldilocks::from_u64(coefficient(&PRODUCTION_SEED, row as u32, block as u64, lane as u32))
            }));
            let mut value = [Goldilocks::ZERO; D];
            for (lane, sign) in entries {
                value[lane] = if sign > 0 { Goldilocks::ONE } else { -Goldilocks::ONE };
            }
            let product = key.mul(&Rq(value));
            for lane in 0..D {
                expected.data[row * D + lane] += product.0[lane];
            }
        }
    }
    assert_eq!(actual, expected);
    assert_ne!(actual, Commitment::zeros(D, PRODUCTION_VERIFIER_ROWS as usize));
    assert!(matches!(
        commit_production_signed_unit_matrix(&prefix),
        Err(AjtaiError::InvalidDimensions(_))
    ));
}

#[test]
fn prefix_commitment_checks_bounds_before_zero_or_key_expansion() {
    let columns = PRODUCTION_MESSAGE_COLUMNS as usize;
    for (rows, width) in [(D - 1, 1), (D, 0), (D, columns + 1)] {
        let invalid = Mat::virtual_constant(rows, width, Goldilocks::ZERO);
        assert!(matches!(
            commit_production_signed_unit_prefix_matrix(&invalid),
            Err(AjtaiError::InvalidDimensions(_))
        ));
    }
    for width in [1, columns] {
        let zero = Mat::virtual_constant(D, width, Goldilocks::ZERO);
        assert_eq!(
            commit_production_signed_unit_prefix_matrix(&zero).unwrap(),
            Commitment::zeros(D, PRODUCTION_VERIFIER_ROWS as usize)
        );
    }
    for value in [2i128, -2] {
        let field = if value > 0 {
            Goldilocks::from_u64(value as u64)
        } else {
            -Goldilocks::from_u64((-value) as u64)
        };
        let invalid = Mat::virtual_constant(D, 1, field);
        assert_eq!(
            commit_production_signed_unit_prefix_matrix(&invalid),
            Err(AjtaiError::RangeViolation { value, bound: 2 })
        );
        let mut final_coordinate = Mat::zero(D, 1, Goldilocks::ZERO);
        final_coordinate[(D - 1, 0)] = field;
        assert_eq!(
            commit_production_signed_unit_prefix_matrix(&final_coordinate),
            Err(AjtaiError::RangeViolation { value, bound: 2 })
        );
    }
}
