use neo_ccs::{CcsMatrix, CcsStructure, SparsePoly};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

#[test]
fn verifier_artifact_header_is_shape_only_and_cannot_enter_sparse_constructor() {
    let header: CcsStructure<F> = CcsStructure::new_verifier_artifact_header(4, 8, 2, SparsePoly::new(2, Vec::new()))
        .expect("valid verifier-artifact header");
    header.validate().expect("valid header shape");
    assert!(header.is_verifier_artifact_header());
    assert_eq!((header.n, header.m, header.t()), (4, 8, 2));

    let marker = CcsMatrix::<F>::VerifierArtifact { rows: 4, cols: 8 };
    assert!(CcsStructure::new_sparse(vec![marker; 2], SparsePoly::new(2, Vec::new())).is_err());
}

#[test]
#[should_panic(expected = "raw multiplication is unavailable")]
fn verifier_artifact_header_fails_closed_on_raw_matrix_evaluation() {
    let matrix = CcsMatrix::<F>::VerifierArtifact { rows: 4, cols: 8 };
    let mut output = vec![F::ZERO; 4];
    matrix.add_mul_into(&[F::ZERO; 8], &mut output, 4);
}
