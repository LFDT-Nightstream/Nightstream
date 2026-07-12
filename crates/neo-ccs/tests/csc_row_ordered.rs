use neo_ccs::CscMat;
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
