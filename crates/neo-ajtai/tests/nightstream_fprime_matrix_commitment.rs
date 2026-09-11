use neo_ajtai::{
    nightstream_fprime_setup::{
        coefficient, commit_production_signed_unit_matrix, PRODUCTION_MESSAGE_COLUMNS, PRODUCTION_SEED,
        PRODUCTION_VERIFIER_ROWS,
    },
    AjtaiError, Commitment,
};
use neo_ccs::Mat;
use neo_math::{Rq, D};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

#[test]
fn production_matrix_commitment_matches_sparse_endpoint_key_products() {
    let columns = PRODUCTION_MESSAGE_COLUMNS as usize;
    let mut positive = vec![0_u64; columns];
    let mut negative = vec![0_u64; columns];
    let support = [
        (0, [(0, 1i8), (D / 2, -1), (D - 1, 1)]),
        (columns - 1, [(0, -1), (1, 1), (D - 1, -1)]),
    ];
    for (block, entries) in support {
        for (lane, sign) in entries {
            if sign > 0 {
                positive[block] |= 1 << lane;
            } else {
                negative[block] |= 1 << lane;
            }
        }
    }
    let witness = Mat::compact_signed_unit_from_column_masks(D, columns, &positive, &negative).unwrap();
    drop((positive, negative));
    let actual = commit_production_signed_unit_matrix(&witness).unwrap();
    let mut expected = Commitment::zeros(D, PRODUCTION_VERIFIER_ROWS as usize);
    for row in 0..PRODUCTION_VERIFIER_ROWS as usize {
        for (block, entries) in support {
            // Use the scalar indexed setup definition and ordinary ring
            // products, independent of the streamed convolution accumulator.
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
    assert_eq!(actual.data.len(), PRODUCTION_VERIFIER_ROWS as usize * D);
    assert_eq!(actual, expected);
    assert_ne!(actual, Commitment::zeros(D, PRODUCTION_VERIFIER_ROWS as usize));
    assert_eq!(
        commit_production_signed_unit_matrix(&Mat::virtual_constant(D, columns, Goldilocks::ZERO)).unwrap(),
        Commitment::zeros(D, PRODUCTION_VERIFIER_ROWS as usize)
    );
}

#[test]
fn production_matrix_commitment_rejects_shape_and_nonunit() {
    let columns = PRODUCTION_MESSAGE_COLUMNS as usize;
    for (rows, width) in [(D - 1, columns), (D, columns - 1), (D, columns + 1), (1, D * columns)] {
        let wrong = Mat::virtual_constant(rows, width, Goldilocks::ZERO);
        assert!(matches!(
            commit_production_signed_unit_matrix(&wrong),
            Err(AjtaiError::InvalidDimensions(_))
        ));
    }
    // Constant storage reaches the checked fallback without allocating a
    // dense carrier. Both signs must fail before any key expansion.
    for value in [2i128, -2] {
        let magnitude = Goldilocks::from_u64(value.unsigned_abs() as u64);
        let field = if value > 0 { magnitude } else { -magnitude };
        let wrong = Mat::virtual_constant(D, columns, field);
        assert_eq!(
            commit_production_signed_unit_matrix(&wrong),
            Err(AjtaiError::RangeViolation { value, bound: 2 })
        );
    }
}
