use neo_ccs::CscMat;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

#[test]
fn duplicate_csc_entries_are_rejected_at_the_relation_boundary() {
    let duplicated = CscMat::new(1, 1, vec![0, 2], vec![0, 0], vec![F::from_u64(2), F::from_u64(4)]);
    assert!(duplicated.is_err(), "duplicate sparse coordinates must be rejected");
}

#[test]
fn unsorted_csc_rows_are_rejected_at_the_relation_boundary() {
    let unsorted = CscMat::new(2, 1, vec![0, 2], vec![1, 0], vec![F::from_u64(5), F::from_u64(7)]);
    assert!(unsorted.is_err(), "unsorted sparse rows must be rejected");
}
