use neo_ccs::Mat;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

#[test]
fn virtual_zero_matrix_materializes_only_on_mutation() {
    let mut matrix = Mat::virtual_constant(3, 5, F::ZERO);
    assert!(matrix.is_virtual_constant());
    assert_eq!(matrix[(2, 4)], F::ZERO);
    assert_eq!(matrix, Mat::zero(3, 5, F::ZERO));

    matrix[(1, 3)] = F::ONE;
    assert!(!matrix.is_virtual_constant());
    assert_eq!(matrix[(1, 3)], F::ONE);
    assert_eq!(matrix[(0, 0)], F::ZERO);
}
