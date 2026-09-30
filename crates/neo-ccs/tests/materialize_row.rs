//! Exact row-materialization parity for every compact matrix representation.
//!
//! | Case | Components | Oracle |
//! |---|---|---|
//! | identity/CSC | direct stored terms | `CcsMatrix::add_mul_into` |
//! | compact overlap | CSC + geometric run | field-summed row action |

use neo_ccs::{CcsMatrix, CscMat, GeometricRowRun};
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
fn compact_row_sums_csc_and_geometric_overlaps() {
    // The run has coefficients 7, 35, 175 in columns 5, 6, 7. The CSC terms
    // change column 5 to 3 and cancel column 6.
    let csc = CscMat::from_triplets(
        vec![
            (0, 5, -F::from_u64(4)),
            (0, 6, -F::from_u64(35)),
            (0, D - 1, F::from_u64(11)),
        ],
        D,
        D,
    );
    let geometric = GeometricRowRun::new(0, 5, 3, F::from_u64(7), F::from_u64(5));
    let compact = CcsMatrix::csc_with_geometric_runs(csc, vec![geometric]).expect("compact matrix");
    let terms = compact.materialize_row(0).expect("compact row");
    assert_eq!(
        terms,
        vec![(5, F::from_u64(3)), (7, F::from_u64(175)), (D - 1, F::from_u64(11))]
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
