use neo_ajtai::nightstream_fprime_setup::{
    commit_production_signed_unit_prefix_matrices, commit_production_signed_unit_prefix_matrix,
    commit_signed_unit_prefix_matrices, commit_signed_unit_prefix_matrix, expanded_key_bytes, KeyTable,
};
use neo_ccs::Mat;
use neo_math::{Fq as F, D};

fn witness(positive: &[u64], negative: &[u64]) -> Mat<F> {
    Mat::compact_signed_unit_from_column_masks(D, positive.len(), positive, negative).expect("masks")
}

/// A commitment with a key table is the production commitment. It expands no
/// key element of a block that the table holds, and each element of a block
/// that the table does not hold. This file has one test, because the count
/// of the expanded bytes is one for the process.
#[test]
fn a_commitment_with_a_key_table_is_the_production_commitment() {
    let inside = witness(&[0b1011, 0, 1 << 63, 1 << 20], &[1 << 40, 0b110, 0, 1]);
    let across = witness(&[1, 0, 0, 1 << 7, 0, 1 << 63, 0b11], &[0, 1 << 5, 0, 0, 0, 1, 1 << 9]);
    let expected =
        [&inside, &across].map(|witness| commit_production_signed_unit_prefix_matrix(witness).expect("commitment"));
    let batch = commit_production_signed_unit_prefix_matrices(&[&inside, &across]).expect("batch");

    let table = KeyTable::production(4);
    let before = expanded_key_bytes();
    assert_eq!(
        commit_signed_unit_prefix_matrix(Some(&table), &inside).expect("commitment"),
        expected[0]
    );
    assert_eq!(
        commit_signed_unit_prefix_matrices(Some(&table), &[&inside, &inside]).expect("batch"),
        [expected[0].clone(), expected[0].clone()]
    );
    assert_eq!(
        expanded_key_bytes(),
        before,
        "the table holds each block of the witness"
    );

    assert_eq!(
        commit_signed_unit_prefix_matrix(Some(&table), &across).expect("commitment"),
        expected[1]
    );
    assert_eq!(
        commit_signed_unit_prefix_matrices(Some(&table), &[&inside, &across]).expect("batch"),
        batch
    );
    assert!(
        expanded_key_bytes() > before,
        "a block from the table's size is expanded"
    );
}
