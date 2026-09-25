use neo_ajtai::{
    nightstream_fprime_setup::{
        coefficient, commit_production_signed_unit_prefix_matrices, commit_production_signed_unit_prefix_matrix,
        MAX_MESSAGE_COLUMNS, PRODUCTION_MESSAGE_COLUMNS, PRODUCTION_SEED, PRODUCTION_VERIFIER_ROWS,
    },
    AjtaiError, Commitment,
};
use neo_ccs::Mat;
use neo_math::{Rq, D};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

fn dense(columns: usize, entries: &[(usize, usize, i8)]) -> Mat<F> {
    let mut witness = Mat::zero(D, columns, F::ZERO);
    for &(lane, column, sign) in entries {
        witness[(lane, column)] = if sign > 0 { F::ONE } else { -F::ONE };
    }
    witness
}

fn column_packed(witness: &Mat<F>) -> Mat<F> {
    let mut positive = vec![0u64; witness.cols()];
    let mut negative = vec![0u64; witness.cols()];
    for column in 0..witness.cols() {
        for lane in 0..D {
            match witness[(lane, column)] {
                value if value == F::ONE => positive[column] |= 1 << lane,
                value if value == -F::ONE => negative[column] |= 1 << lane,
                value => assert_eq!(value, F::ZERO),
            }
        }
    }
    Mat::compact_signed_unit_from_column_masks(D, witness.cols(), &positive, &negative).unwrap()
}

fn independent_commitment(witness: &Mat<F>) -> Commitment {
    let mut commitment = Commitment::zeros(D, PRODUCTION_VERIFIER_ROWS as usize);
    if witness.virtual_constant_value() == Some(&F::ZERO) {
        return commitment;
    }
    for column in 0..witness.cols() {
        let message = Rq(std::array::from_fn(|lane| witness[(lane, column)]));
        if message.0.iter().all(|value| *value == F::ZERO) {
            continue;
        }
        for row in 0..PRODUCTION_VERIFIER_ROWS as usize {
            // Scalar ChaCha words and ordinary ring multiplication are independent
            // of coefficient_block, mask merging and signed convolution sums.
            let key = Rq(std::array::from_fn(|lane| {
                F::from_u64(coefficient(&PRODUCTION_SEED, row as u32, column as u64, lane as u32))
            }));
            let product = key.mul(&message);
            for lane in 0..D {
                commitment.data[row * D + lane] += product.0[lane];
            }
        }
    }
    commitment
}

#[test]
fn batched_prefix_commitments_match_scalar_and_independent_ring_products() {
    let first = dense(3, &[(0, 0, 1), (D / 2, 0, -1), (D - 1, 2, 1)]);
    let overlap = dense(4, &[(D - 1, 2, -1), (1, 2, 1), (D / 2 - 1, 3, 1)]);
    let disjoint = dense(2, &[(0, 1, -1), (D - 1, 1, -1)]);
    let mut row_values = Vec::with_capacity(D * disjoint.cols());
    for lane in 0..D {
        for column in 0..disjoint.cols() {
            row_values.push(disjoint[(lane, column)]);
        }
    }
    let row_packed = Mat::compact_signed_unit(D, disjoint.cols(), row_values);
    assert!(row_packed.is_packed_signed_unit());
    assert!(row_packed.packed_signed_unit_column_masks().is_none());
    let witnesses = vec![
        first.clone(),
        column_packed(&overlap),
        row_packed,
        Mat::virtual_constant(D, PRODUCTION_MESSAGE_COLUMNS as usize, F::ZERO),
        Mat::virtual_constant(D, 1, F::ONE),
        Mat::virtual_constant(D, 1, -F::ONE),
        column_packed(&first),
    ];
    let actual = commit_production_signed_unit_prefix_matrices(&witnesses).unwrap();
    assert_eq!(actual.len(), witnesses.len());
    for (index, witness) in witnesses.iter().enumerate() {
        assert_eq!(
            actual[index],
            commit_production_signed_unit_prefix_matrix(witness).unwrap(),
            "scalar witness {index}"
        );
        assert_eq!(
            actual[index],
            independent_commitment(witness),
            "independent witness {index}"
        );
    }
    assert_eq!(actual[0], actual[6], "dense and column-packed copies");
    assert_ne!(actual[0], actual[1], "different witnesses retain distinct commitments");
    let reversed: Vec<_> = witnesses.into_iter().rev().collect();
    let expected: Vec<_> = actual.into_iter().rev().collect();
    assert_eq!(
        commit_production_signed_unit_prefix_matrices(&reversed).unwrap(),
        expected
    );
}

#[test]
fn empty_and_zero_batches_preserve_order_and_prefix_dimensions() {
    assert!(commit_production_signed_unit_prefix_matrices(&[])
        .unwrap()
        .is_empty());
    let witnesses = [
        Mat::virtual_constant(D, PRODUCTION_MESSAGE_COLUMNS as usize, F::ZERO),
        Mat::zero(D, 1, F::ZERO),
        Mat::compact_signed_unit_from_column_masks(D, 2, &[0, 0], &[0, 0]).unwrap(),
    ];
    assert_eq!(
        commit_production_signed_unit_prefix_matrices(&witnesses).unwrap(),
        vec![Commitment::zeros(D, PRODUCTION_VERIFIER_ROWS as usize); witnesses.len()]
    );
}

#[test]
fn batch_rejects_dimensions_and_nonunit_values_at_their_input_index() {
    let columns = PRODUCTION_MESSAGE_COLUMNS as usize;
    let valid = column_packed(&dense(2, &[(0, 0, 1), (D - 1, 1, -1)]));
    let mut malformed = vec![
        Mat::virtual_constant(D - 1, 1, F::ZERO),
        Mat::virtual_constant(D, 0, F::ZERO),
        Mat::virtual_constant(D, MAX_MESSAGE_COLUMNS as usize + 1, F::ZERO),
        Mat::virtual_constant(D, 1, F::from_u64(2)),
        Mat::virtual_constant(D, 1, -F::from_u64(2)),
    ];
    let mut final_coordinate = Mat::zero(D, 2, F::ZERO);
    final_coordinate[(D - 1, 1)] = -F::from_u64(2);
    malformed.push(final_coordinate);
    for invalid in malformed {
        let expected = commit_production_signed_unit_prefix_matrix(&invalid).unwrap_err();
        let error =
            commit_production_signed_unit_prefix_matrices(&[valid.clone(), invalid, valid.clone()]).unwrap_err();
        assert_eq!(error.witness_index(), 1);
        assert_eq!(error.error(), &expected);
        assert_eq!(error.into_error(), expected);
    }
    let invalid_value = Mat::virtual_constant(D, 1, F::from_u64(2));
    let invalid_shape = Mat::virtual_constant(D, 0, F::ZERO);
    let error = commit_production_signed_unit_prefix_matrices(&[invalid_value, invalid_shape]).unwrap_err();
    assert_eq!(error.witness_index(), 0);
    assert_eq!(error.into_error(), AjtaiError::RangeViolation { value: 2, bound: 2 });

    let error = commit_production_signed_unit_prefix_matrices(&[
        valid,
        Mat::virtual_constant(D, columns, F::ZERO),
        Mat::virtual_constant(D, 0, F::ZERO),
    ])
    .unwrap_err();
    assert_eq!(error.witness_index(), 2);
    assert!(matches!(error.into_error(), AjtaiError::InvalidDimensions(_)));
}
