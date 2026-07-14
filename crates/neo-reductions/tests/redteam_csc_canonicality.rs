use neo_ccs::{CcsMatrix, CcsStructure, CscMat, SparsePoly};
use neo_math::{F, K};
use neo_reductions::superneo_eval::{eval_all_mats_direct, eval_all_mats_superneo};
use p3_field::PrimeCharacteristicRing;

#[test]
fn accepted_duplicate_csc_entries_preserve_superneo_evaluator_parity() {
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
    let structure = CcsStructure::new_sparse(vec![CcsMatrix::Csc(duplicated)], SparsePoly::new(1, vec![]))
        .expect("the public sparse-structure boundary accepts duplicate coordinates");

    let z = vec![K::ONE];
    let chi_r = vec![K::ONE];
    let direct = eval_all_mats_direct(&structure, &z, &chi_r, 1);
    let superneo = eval_all_mats_superneo(&structure, &z, &chi_r, 1);

    assert_eq!(
        superneo, direct,
        "an accepted CCS matrix must have one meaning across the direct and SuperNeo evaluators"
    );
}

#[test]
fn accepted_unsorted_csc_rows_preserve_superneo_evaluator_parity() {
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
    let structure = CcsStructure::new_sparse(vec![CcsMatrix::Csc(unsorted)], SparsePoly::new(1, vec![]))
        .expect("the public sparse-structure boundary accepts unsorted row indices");

    let z = vec![K::ONE];
    let chi_r = vec![K::ZERO, K::ONE];
    let direct = eval_all_mats_direct(&structure, &z, &chi_r, 2);
    let superneo = eval_all_mats_superneo(&structure, &z, &chi_r, 2);

    assert_eq!(
        superneo, direct,
        "an accepted CCS matrix must have one meaning across the direct and SuperNeo evaluators"
    );
}
