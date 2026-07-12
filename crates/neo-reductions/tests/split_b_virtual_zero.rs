use neo_ccs::Mat;
use neo_math::F;
use neo_reductions::split_b_matrix_k_with_nonzero_flags;
use p3_field::PrimeCharacteristicRing;

#[test]
fn split_b_does_not_allocate_structurally_zero_digit_planes() {
    let input = Mat::from_row_major(2, 3, vec![F::ZERO, F::ONE, -F::ONE, F::ZERO, F::ONE, F::ZERO]);
    let (digits, nonzero) = split_b_matrix_k_with_nonzero_flags(&input, 14, 2).expect("balanced split");

    assert!(nonzero[0]);
    assert!(!digits[0].is_virtual_constant());
    assert!(digits[0].is_packed_signed_unit());
    assert_eq!(digits[0].packed_signed_unit_nonzero_count(), Some(3));
    assert_eq!(digits[0].to_dense_vec(), input.to_dense_vec());
    for (digit, &is_nonzero) in digits.iter().zip(&nonzero).skip(1) {
        assert!(!is_nonzero);
        assert!(digit.is_virtual_constant());
        assert_eq!(digit.virtual_constant_value(), Some(&F::ZERO));
        for row in 0..digit.rows() {
            for column in 0..digit.cols() {
                assert_eq!(digit[(row, column)], F::ZERO);
            }
        }
    }
}
