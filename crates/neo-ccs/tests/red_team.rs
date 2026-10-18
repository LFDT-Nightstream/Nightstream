use neo_ccs::{
    matrix::Mat,
    poly::{SparsePoly, Term},
    relations::{check_ccs_rowwise_relaxed, check_ccs_rowwise_zero},
    CcsStructure,
};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

#[test]
fn ccs_rowwise_zero_rejects_bad_row() {
    let f = SparsePoly::new(
        2,
        vec![
            Term {
                coeff: F::ONE,
                exps: vec![1, 0],
            },
            Term {
                coeff: -F::ONE,
                exps: vec![0, 1],
            },
        ],
    );
    let a = Mat::from_row_major(1, 2, vec![F::ONE, F::ZERO]);
    let b = Mat::from_row_major(1, 2, vec![F::ONE, F::ONE]);
    let s = CcsStructure::new(vec![a, b], f).unwrap();

    let x = vec![];
    let w = vec![F::from_u64(7), F::ONE];
    assert!(check_ccs_rowwise_zero(&s, &x, &w).is_err());
    println!("✅ RED TEAM: CCS rowwise zero correctly rejects invalid relation");
}

#[test]
fn ccs_relaxed_rejects_wrong_rhs() {
    let f = SparsePoly::new(
        1,
        vec![Term {
            coeff: F::ONE,
            exps: vec![1],
        }],
    );
    let m = Mat::from_row_major(1, 2, vec![F::ONE, F::from_u64(2)]);
    let s = CcsStructure::new(vec![m], f).unwrap();
    let x = vec![F::from_u64(3)];
    let w = vec![F::from_u64(4)];
    let u = vec![F::from_u64(6)];
    let e = F::from_u64(2);
    assert!(check_ccs_rowwise_relaxed(&s, &x, &w, Some(&u), Some(e)).is_err());
    println!("✅ RED TEAM: CCS relaxed correctly rejects wrong RHS");
}

#[test]
fn ccs_relaxed_accepts_correct_rhs() {
    let f = SparsePoly::new(
        1,
        vec![Term {
            coeff: F::ONE,
            exps: vec![1],
        }],
    );
    let m = Mat::from_row_major(1, 2, vec![F::ONE, F::from_u64(2)]);
    let s = CcsStructure::new(vec![m], f).unwrap();
    let x = vec![F::from_u64(3)];
    let w = vec![F::from_u64(4)];
    let u = vec![F::ONE];
    let e = F::from_u64(11);
    assert!(check_ccs_rowwise_relaxed(&s, &x, &w, Some(&u), Some(e)).is_ok());
    println!("✅ RED TEAM: CCS relaxed correctly accepts valid relation");
}

#[test]
fn ccs_structure_validation_edge_cases() {
    let f = SparsePoly::new(
        1,
        vec![Term {
            coeff: F::ONE,
            exps: vec![1],
        }],
    );
    let empty_matrices = vec![];
    assert!(CcsStructure::new(empty_matrices, f.clone()).is_err());

    let m1 = Mat::from_row_major(2, 2, vec![F::ONE, F::ZERO, F::ZERO, F::ONE]);
    let m2 = Mat::from_row_major(3, 2, vec![F::ONE; 6]);
    assert!(CcsStructure::new(vec![m1, m2], f.clone()).is_err());

    let f2 = SparsePoly::new(
        3,
        vec![Term {
            coeff: F::ONE,
            exps: vec![1, 0, 0],
        }],
    );
    let m = Mat::from_row_major(2, 2, vec![F::ONE, F::ZERO, F::ZERO, F::ONE]);
    assert!(CcsStructure::new(vec![m.clone(), m], f2).is_err());

    println!("✅ RED TEAM: CCS structure validation correctly rejects malformed inputs");
}

#[test]
fn polynomial_evaluation_tampering_detection() {
    let f = SparsePoly::new(
        2,
        vec![
            Term {
                coeff: F::ONE,
                exps: vec![2, 0],
            },
            Term {
                coeff: F::ONE,
                exps: vec![0, 2],
            },
            Term {
                coeff: -F::from_u64(5),
                exps: vec![0, 0],
            },
        ],
    );

    let m1 = Mat::from_row_major(1, 2, vec![F::from_u64(2), F::ZERO]);
    let m2 = Mat::from_row_major(1, 2, vec![F::ZERO, F::ONE]);
    let s = CcsStructure::new(vec![m1, m2], f).unwrap();

    let x = vec![];
    let w = vec![F::ONE, F::ONE];
    assert!(check_ccs_rowwise_zero(&s, &x, &w).is_ok());

    let w_bad = vec![F::ONE, F::from_u64(2)];
    assert!(check_ccs_rowwise_zero(&s, &x, &w_bad).is_err());

    println!("✅ RED TEAM: Polynomial evaluation tampering correctly detected");
}

#[test]
fn witness_length_validation() {
    let f = SparsePoly::new(
        1,
        vec![Term {
            coeff: F::ONE,
            exps: vec![1],
        }],
    );
    let m = Mat::from_row_major(2, 3, vec![F::ONE, F::ZERO, F::ZERO, F::ZERO, F::ONE, F::ZERO]);
    let s = CcsStructure::new(vec![m], f).unwrap();

    let x = vec![];
    let w = vec![F::ZERO, F::ZERO, F::ZERO];
    assert!(check_ccs_rowwise_zero(&s, &x, &w).is_ok());

    let w_short = vec![F::ZERO];
    assert!(check_ccs_rowwise_zero(&s, &x, &w_short).is_err());

    let w_long = vec![F::ZERO, F::ZERO, F::ZERO, F::ZERO];
    assert!(check_ccs_rowwise_zero(&s, &x, &w_long).is_err());

    println!("✅ RED TEAM: Witness length validation works correctly");
}

#[test]
fn relaxed_slack_vector_validation() {
    let f = SparsePoly::new(
        1,
        vec![Term {
            coeff: F::ONE,
            exps: vec![1],
        }],
    );
    let m = Mat::from_row_major(2, 2, vec![F::ONE, F::ZERO, F::ZERO, F::ONE]);
    let s = CcsStructure::new(vec![m], f).unwrap();

    let x = vec![F::ONE];
    let w = vec![F::ONE];
    let e = F::from_u64(2);

    let u = vec![F::from_u64(2), F::from_u64(2)];
    assert!(check_ccs_rowwise_relaxed(&s, &x, &w, Some(&u), Some(e)).is_err());

    let u_short = vec![F::from_u64(2)];
    assert!(check_ccs_rowwise_relaxed(&s, &x, &w, Some(&u_short), Some(e)).is_err());

    let e_correct = F::ONE;
    let u_correct = vec![F::ONE, F::ONE];
    assert!(check_ccs_rowwise_relaxed(&s, &x, &w, Some(&u_correct), Some(e_correct)).is_ok());

    println!("✅ RED TEAM: Relaxed slack vector validation works correctly");
}

#[test]
fn matrix_polynomial_consistency_attack() {
    let f_single = SparsePoly::new(
        1,
        vec![Term {
            coeff: F::ONE,
            exps: vec![1],
        }],
    );
    let f_double = SparsePoly::new(
        2,
        vec![Term {
            coeff: F::ONE,
            exps: vec![1, 1],
        }],
    );

    let m1 = Mat::from_row_major(1, 2, vec![F::ONE, F::ZERO]);
    let m2 = Mat::from_row_major(1, 2, vec![F::ZERO, F::ONE]);

    assert!(CcsStructure::new(vec![m1.clone(), m2.clone()], f_double.clone()).is_ok());

    assert!(CcsStructure::new(vec![m1.clone(), m2.clone()], f_single.clone()).is_err());

    assert!(CcsStructure::new(vec![m1], f_double).is_err());

    println!("✅ RED TEAM: Matrix-polynomial consistency correctly enforced");
}

#[test]
fn zero_dimension_matrix_rejection() {
    let f = SparsePoly::new(
        1,
        vec![Term {
            coeff: F::ONE,
            exps: vec![1],
        }],
    );

    let m_zero_rows = Mat::from_row_major(0, 5, vec![]);
    assert!(CcsStructure::new(vec![m_zero_rows], f.clone()).is_err());

    let m_zero_cols = Mat::from_row_major(5, 0, vec![]);
    assert!(CcsStructure::new(vec![m_zero_cols], f.clone()).is_err());

    println!("✅ RED TEAM: Zero-dimension matrices correctly rejected");
}

#[test]
fn large_polynomial_degree_handling() {
    let f = SparsePoly::new(
        2,
        vec![
            Term {
                coeff: F::ONE,
                exps: vec![10, 0],
            },
            Term {
                coeff: F::ONE,
                exps: vec![0, 10],
            },
            Term {
                coeff: -F::from_u64(2),
                exps: vec![5, 5],
            },
        ],
    );

    let m1 = Mat::from_row_major(1, 2, vec![F::from_u64(2), F::ZERO]);
    let m2 = Mat::from_row_major(1, 2, vec![F::ZERO, F::ONE]);
    let s = CcsStructure::new(vec![m1, m2], f).unwrap();

    let x = vec![];
    let w = vec![F::ONE, F::ONE];

    assert!(check_ccs_rowwise_zero(&s, &x, &w).is_err());

    let u = vec![F::from_u64(961)];
    let e = F::ONE;
    assert!(check_ccs_rowwise_relaxed(&s, &x, &w, Some(&u), Some(e)).is_ok());

    println!("✅ RED TEAM: High-degree polynomial evaluation handles correctly");
}
