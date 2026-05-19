use neo_ccs::{check_ccs_rowwise_zero, r1cs_to_ccs, sparse_r1cs_to_ccs, CcsMatrix, CscMat, Mat};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

#[test]
fn square_r1cs_uses_three_matrix_embedding() {
    let n = 4usize;
    let m = 4usize;

    let mut a = Mat::zero(n, m, F::ZERO);
    let mut b = Mat::zero(n, m, F::ZERO);
    let mut c = Mat::zero(n, m, F::ZERO);
    a[(0, 0)] = F::ONE;
    b[(0, 1)] = F::ONE;
    c[(0, 2)] = F::ONE;

    let ccs = r1cs_to_ccs(a, b, c);
    assert_eq!(ccs.t(), 3, "square R1CS must not auto-insert identity matrix");
    assert!(!ccs.matrices[0].is_identity(), "M0 should be A, not identity");
}

#[test]
fn rectangular_r1cs_uses_three_matrix_embedding() {
    let n = 2usize;
    let m = 5usize;

    let mut a = Mat::zero(n, m, F::ZERO);
    let mut b = Mat::zero(n, m, F::ZERO);
    let mut c = Mat::zero(n, m, F::ZERO);
    a[(0, 0)] = F::ONE;
    b[(0, 1)] = F::ONE;
    c[(0, 2)] = F::ONE;

    let ccs = r1cs_to_ccs(a, b, c);
    assert_eq!(ccs.t(), 3);
    assert_eq!(ccs.n, n);
    assert_eq!(ccs.m, m);
}

#[test]
fn sparse_r1cs_embedding_preserves_rowwise_relation_without_dense_matrices() {
    let n = 2usize;
    let m = 4usize;
    let a = CcsMatrix::Csc(CscMat::from_triplets(vec![(0, 0, F::ONE), (1, 1, F::ONE)], n, m));
    let b = CcsMatrix::Csc(CscMat::from_triplets(vec![(0, 1, F::ONE), (1, 2, F::ONE)], n, m));
    let c = CcsMatrix::Csc(CscMat::from_triplets(vec![(0, 2, F::ONE), (1, 3, F::ONE)], n, m));

    let ccs = sparse_r1cs_to_ccs(a, b, c).expect("sparse R1CS embedding");
    assert_eq!(ccs.t(), 3);
    assert_eq!(ccs.n, n);
    assert_eq!(ccs.m, m);
    assert!(
        ccs.matrices.iter().all(|matrix| matrix.as_csc().is_some()),
        "sparse R1CS embedding should preserve CSC matrices"
    );

    let x = [F::from_u64(2)];
    let w = [F::from_u64(3), F::from_u64(6), F::from_u64(18)];
    check_ccs_rowwise_zero(&ccs, &x, &w).expect("R1CS row relation should hold");

    let bad_w = [F::from_u64(3), F::from_u64(7), F::from_u64(18)];
    assert!(
        check_ccs_rowwise_zero(&ccs, &x, &bad_w).is_err(),
        "R1CS row relation must reject an unsatisfied witness"
    );
}
