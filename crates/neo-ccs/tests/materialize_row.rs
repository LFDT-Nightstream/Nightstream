//! Exact row-materialization parity for every compact matrix representation.
//!
//! | Case | Components | Oracle |
//! |---|---|---|
//! | identity/CSC | direct stored terms | `CcsMatrix::add_mul_into` |
//! | compact overlap | CSC + seeded Phi81 + geometric run | field-summed row action |
//! | transformed seeded block | SuperNeo bar-transformed columns | matrix action |

use neo_ccs::{CcsMatrix, CscMat, GeometricRowRun, SeededPhi81LinearBlock};
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

fn assert_row_action(matrix: &CcsMatrix<F>, row: usize, assignments: &[Vec<F>]) {
    let terms = matrix.materialize_row(row).expect("in-range row");
    assert!(terms.windows(2).all(|pair| pair[0].0 < pair[1].0));
    assert!(terms.iter().all(|(_, coefficient)| *coefficient != F::ZERO));
    for assignment in assignments {
        let mut image = vec![F::ZERO; matrix.rows()];
        matrix.add_mul_into(assignment, &mut image, matrix.rows());
        let row_action = terms.iter().fold(F::ZERO, |sum, &(column, coefficient)| {
            sum + coefficient * assignment[column]
        });
        assert_eq!(row_action, image[row]);
    }
}

fn seeded_block(transformed: bool) -> SeededPhi81LinearBlock {
    let block = SeededPhi81LinearBlock::new_with_word_width(0, vec![0], 1, 1, 1, 1, vec![vec![[0xa5; 32]]])
        .expect("tiny seeded block");
    if transformed {
        block.with_superneo_transformed_columns()
    } else {
        block
    }
}

#[test]
fn materialized_rows_match_identity_and_csc_actions() {
    let identity = CcsMatrix::<F>::Identity { n: 4 };
    assert_eq!(identity.materialize_row(2), Some(vec![(2, F::ONE)]));
    assert_eq!(identity.materialize_row(4), None);

    let csc = CcsMatrix::Csc(CscMat::from_triplets(
        vec![(1, 3, F::from_u64(7)), (1, 0, -F::ONE), (2, 1, F::ONE)],
        3,
        4,
    ));
    assert_eq!(csc.materialize_row(1), Some(vec![(0, -F::ONE), (3, F::from_u64(7))]));
    assert_row_action(
        &csc,
        1,
        &[
            vec![F::ZERO; 4],
            vec![F::ONE; 4],
            (0..4).map(|value| F::from_u64(value as u64 + 2)).collect(),
        ],
    );
}

#[test]
fn compact_row_sums_csc_seeded_and_geometric_overlaps() {
    let block = seeded_block(false);
    let seeded_only: CcsMatrix<F> =
        CcsMatrix::csc_with_seeded_phi81(CscMat::from_triplets(vec![], D, D), vec![block.clone()])
            .expect("seeded matrix");
    let (overlap_column, seeded_coefficient) = seeded_only
        .materialize_row(0)
        .expect("seeded row")
        .into_iter()
        .next()
        .expect("nonzero seeded row");
    let csc = CscMat::from_triplets(
        vec![(0, overlap_column, -seeded_coefficient), (0, D - 1, F::from_u64(11))],
        D,
        D,
    );
    let geometric = GeometricRowRun::new(0, overlap_column, 1, F::from_u64(3), F::from_u64(5));
    let compact = CcsMatrix::csc_with_compact_rows(csc, vec![block], vec![geometric]).expect("compact matrix");
    let terms = compact.materialize_row(0).expect("compact row");
    assert_eq!(
        terms.iter().find(|(column, _)| *column == overlap_column),
        Some(&(overlap_column, F::from_u64(3)))
    );
    assert_row_action(
        &compact,
        0,
        &[
            vec![F::ONE; D],
            (0..D).map(|value| F::from_u64(value as u64 + 1)).collect(),
        ],
    );
}

#[test]
fn transformed_seeded_row_matches_matrix_action() {
    let transformed: CcsMatrix<F> =
        CcsMatrix::csc_with_seeded_phi81(CscMat::from_triplets(vec![], D, D), vec![seeded_block(true)])
            .expect("transformed seeded matrix");
    assert_row_action(
        &transformed,
        0,
        &[
            vec![F::ONE; D],
            (0..D)
                .map(|value| F::from_u64((value * 7 + 3) as u64))
                .collect(),
        ],
    );
}
