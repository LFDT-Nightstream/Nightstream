use neo_ccs::{CscMat, GeometricRowRun};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

#[test]
fn counted_constructor_matches_canonical_csc_bytes() {
    let mut triplets = Vec::new();
    for row in 0..37usize {
        for offset in 0..7usize {
            let column = (row * 11 + offset * 5) % 19;
            let value = F::from_u64((row * 13 + offset + 1) as u64);
            triplets.push((row, column, value));
            if offset % 3 == 0 {
                triplets.push((row, column, -value));
                triplets.push((row, column, value));
            }
        }
        triplets.push((row, row % 19, F::ZERO));
    }

    let canonical = CscMat::from_triplets(triplets.clone(), 37, 19);
    triplets.reverse();
    let direct = CscMat::from_counted_triplets(triplets, 37, 19);
    assert_eq!(direct.nrows, canonical.nrows);
    assert_eq!(direct.ncols, canonical.ncols);
    assert_eq!(direct.col_ptr, canonical.col_ptr);
    assert_eq!(direct.row_idx, canonical.row_idx);
    assert_eq!(direct.vals, canonical.vals);
}

#[test]
fn direct_geometric_constructor_matches_expanded_canonical_csc_bytes() {
    let runs = vec![
        GeometricRowRun::new(3, 4, 11, F::from_u64(7), F::from_u64(3)),
        GeometricRowRun::new(3, 9, 7, -F::from_u64(2), F::from_u64(2)),
        GeometricRowRun::new(17, 1, 19, F::ONE, F::from_u64(3)),
    ];
    let explicit = vec![(3, 9, F::from_u64(5)), (0, 0, F::ONE), (17, 1, -F::ONE)];
    let mut expanded = explicit.clone();
    for run in &runs {
        run.for_each_term(|row, column, value| expanded.push((row, column, value)));
    }

    let canonical = CscMat::from_triplets(expanded, 23, 29);
    let direct = CscMat::from_triplets_and_geometric_runs(explicit, &runs, 23, 29);
    assert_eq!(direct.nrows, canonical.nrows);
    assert_eq!(direct.ncols, canonical.ncols);
    assert_eq!(direct.col_ptr, canonical.col_ptr);
    assert_eq!(direct.row_idx, canonical.row_idx);
    assert_eq!(direct.vals, canonical.vals);
}
