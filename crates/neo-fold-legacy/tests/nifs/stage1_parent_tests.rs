use super::*;

#[test]
fn child_batch_requires_distinct_in_range_indices() {
    let count = Params::production().k_rho() as usize;
    validate_children(&[0, count - 1], count);
    validate_children(&[count - 1, 0], count);
    for invalid in [vec![], vec![0, 0], vec![count], vec![0, count]] {
        assert!(std::panic::catch_unwind(|| validate_children(&invalid, count)).is_err());
    }
}

#[test]
fn saved_digit_must_equal_canonical_parent_split() {
    let params = Params::production();
    let mut values = vec![F::ZERO; D];
    values[0] = F::from_u64(5);
    let parent = Mat::from_row_major(D, 1, values);
    let (digits, flags) = split_b_matrix_k_with_nonzero_flags(&parent, params.k_rho() as usize, params.b()).unwrap();
    assert!(check_digit(&digits, &flags, &digits[0], 0));
    assert!(!check_digit(&digits, &flags, &digits[1], 1));
    assert!(check_digit(&digits, &flags, &digits[2], 2));
    assert!(std::panic::catch_unwind(|| check_digit(&digits, &flags, &digits[1], 0)).is_err());
    assert!(std::panic::catch_unwind(|| check_digit(&digits, &flags, &digits[0], 1)).is_err());
}

#[test]
fn saved_witness_must_open_parent_commitment() {
    use neo_ajtai::nightstream_fprime_setup::PRODUCTION_MESSAGE_COLUMNS;

    let columns = PRODUCTION_MESSAGE_COLUMNS as usize;
    let mut positive = vec![0u64; columns];
    let negative = vec![0u64; columns];
    positive[0] = 1;
    let original = Mat::<F>::compact_signed_unit_from_column_masks(D, columns, &positive, &negative).unwrap();
    let expected = commit_production_signed_unit_matrix(&original).unwrap();
    drop((original, positive, negative));
    let altered = Mat::virtual_constant(D, columns, F::ZERO);
    let params = Params::production();
    let zero = Commitment::zeros(D, params.inner().kappa as usize);
    assert_ne!(expected, zero, "the original saved witness has a nonzero commitment");
    let checked = commit_parent(&params, &altered, &zero);
    assert!(checked.nonzero.iter().all(|flag| !flag));
    assert_eq!(checked.digits.len(), params.k_rho() as usize);
    assert!(std::panic::catch_unwind(|| commit_parent(&params, &altered, &expected)).is_err());
}
