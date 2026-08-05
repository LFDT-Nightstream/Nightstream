use neo_ccs::{CcsMatrix, CcsStructure, CscMat, SparsePoly};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

#[test]
fn duplicate_csc_entries_are_rejected_at_the_relation_boundary() {
    // This is the single-entry matrix [2 + 4], represented with two entries
    // at the same coordinate. `CcsStructure::new_sparse` accepts this public
    // representation and ordinary CSC multiplication sums both entries.
    let duplicated = CscMat {
        nrows: 1,
        ncols: 1,
        col_ptr: vec![0, 2],
        row_idx: vec![0, 0],
        vals: vec![F::from_u64(2), F::from_u64(4)],
    };
    let result = CcsStructure::new_sparse(vec![CcsMatrix::Csc(duplicated)], SparsePoly::new(1, vec![]));
    assert!(result.is_err(), "duplicate sparse coordinates must be rejected");
}

#[test]
fn unsorted_csc_rows_are_rejected_at_the_relation_boundary() {
    // Both coordinates are in bounds, but the row indices in this column are
    // descending. Ordinary CSC multiplication remains well-defined and sees
    // row 1, while the SuperNeo evaluator performs a binary search that assumes
    // ascending row indices.
    let unsorted = CscMat {
        nrows: 2,
        ncols: 1,
        col_ptr: vec![0, 2],
        row_idx: vec![1, 0],
        vals: vec![F::from_u64(5), F::from_u64(7)],
    };
    let result = CcsStructure::new_sparse(vec![CcsMatrix::Csc(unsorted)], SparsePoly::new(1, vec![]));
    assert!(result.is_err(), "unsorted sparse rows must be rejected");
}
