use neo_ccs::{check_ccs_rowwise_zero, sparse_selected_r1cs_to_ccs, CcsMatrix, CscMat};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

fn matrix(entries: Vec<(usize, usize, F)>, rows: usize, columns: usize) -> CcsMatrix<F> {
    CcsMatrix::Csc(CscMat::from_triplets(entries, rows, columns))
}

fn selected_product_relation(selector_column: usize) -> neo_ccs::CcsStructure<F> {
    let rows = 1;
    let columns = 5;
    let a = matrix(vec![(0, 1, F::ONE)], rows, columns);
    let b = matrix(vec![(0, 2, F::ONE)], rows, columns);
    let c = matrix(vec![(0, 3, F::ONE)], rows, columns);
    let selector = matrix(vec![(0, selector_column, F::ONE)], rows, columns);
    sparse_selected_r1cs_to_ccs(a, b, c, selector).expect("valid selected relation")
}

#[test]
fn native_selector_has_the_lean_owned_shape() {
    let relation = selected_product_relation(4);
    assert_eq!(relation.t(), 4);
    assert_eq!(relation.max_degree(), 3);

    let terms = relation.f.terms();
    assert_eq!(terms.len(), 2);
    assert_eq!(terms[0].coeff, F::ONE);
    assert_eq!(terms[0].exps, [1, 1, 0, 1]);
    assert_eq!(terms[1].coeff, -F::ONE);
    assert_eq!(terms[1].exps, [0, 0, 1, 1]);
}

#[test]
fn active_selector_enforces_the_source_row() {
    let relation = selected_product_relation(4);

    let public = [F::ONE];
    let valid_witness = [F::from_u64(3), F::from_u64(4), F::from_u64(12), F::ONE];
    check_ccs_rowwise_zero(&relation, &public, &valid_witness).expect("active source equation must hold");

    let invalid_witness = [F::from_u64(3), F::from_u64(4), F::from_u64(11), F::ONE];
    assert!(
        check_ccs_rowwise_zero(&relation, &public, &invalid_witness).is_err(),
        "active selector must reject an invalid source equation"
    );
}

#[test]
fn inactive_selector_accepts_an_arbitrary_source_residual() {
    let relation = selected_product_relation(4);
    let public = [F::ONE];
    let witness = [F::from_u64(3), F::from_u64(4), F::from_u64(11), F::ZERO];
    check_ccs_rowwise_zero(&relation, &public, &witness).expect("zero selector must disable the source equation");
}

#[test]
fn selector_matrix_position_is_load_bearing() {
    let correct_relation = selected_product_relation(4);
    let wrong_relation = selected_product_relation(3);
    let public = [F::ONE];
    let witness = [F::from_u64(3), F::from_u64(4), F::ZERO, F::ONE];
    assert!(
        check_ccs_rowwise_zero(&correct_relation, &public, &witness).is_err(),
        "the correct activation column must reject the invalid source row"
    );
    assert!(
        check_ccs_rowwise_zero(&wrong_relation, &public, &witness).is_ok(),
        "moving S to the zero C coordinate must incorrectly disable the row"
    );
}

#[test]
fn shape_mismatch_fails_closed() {
    let a = matrix(vec![(0, 0, F::ONE)], 1, 2);
    let b = matrix(vec![(0, 0, F::ONE)], 1, 2);
    let c = matrix(vec![(0, 0, F::ONE)], 1, 2);
    let selector = matrix(vec![(0, 0, F::ONE)], 2, 2);
    assert!(
        sparse_selected_r1cs_to_ccs(a, b, c, selector).is_err(),
        "all four matrices must have one exact shape"
    );
}
