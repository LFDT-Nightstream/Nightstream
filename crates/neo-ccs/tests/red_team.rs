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
    // f(y0,y1)=y0 - y1
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
    // A,B with one row: A z = [z0], B z = [z0 + 1] to force a mismatch
    let a = Mat::from_row_major(1, 2, vec![F::ONE, F::ZERO]);
    let b = Mat::from_row_major(1, 2, vec![F::ONE, F::ONE]); // sums z0 + z1; pick z1=1 later
    let s = CcsStructure::new(vec![a, b], f).unwrap();

    let x = vec![]; // all witness
    let w = vec![F::from_u64(7), F::ONE]; // z0=7, z1=1 → A z = 7, B z = 8 → f= -1 ≠ 0
    assert!(check_ccs_rowwise_zero(&s, &x, &w).is_err());
    println!("✅ RED TEAM: CCS rowwise zero correctly rejects invalid relation");
}

#[test]
fn ccs_relaxed_rejects_wrong_rhs() {
    // f(y)=y, one matrix M = [1,2]
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
    let w = vec![F::from_u64(4)]; // M z = 11
                                  // Claim e*u = 12, but actual is 11 → reject
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
    let w = vec![F::from_u64(4)]; // M z = 11
    let u = vec![F::ONE];
    let e = F::from_u64(11);
    assert!(check_ccs_rowwise_relaxed(&s, &x, &w, Some(&u), Some(e)).is_ok());
    println!("✅ RED TEAM: CCS relaxed correctly accepts valid relation");
}

#[test]
fn ccs_structure_validation_edge_cases() {
    // Test empty matrices rejection
    let f = SparsePoly::new(
        1,
        vec![Term {
            coeff: F::ONE,
            exps: vec![1],
        }],
    );
    let empty_matrices = vec![];
    assert!(CcsStructure::new(empty_matrices, f.clone()).is_err());

    // Test matrix dimension mismatches
    let m1 = Mat::from_row_major(2, 2, vec![F::ONE, F::ZERO, F::ZERO, F::ONE]);
    let m2 = Mat::from_row_major(3, 2, vec![F::ONE; 6]); // Different row count
    assert!(CcsStructure::new(vec![m1, m2], f.clone()).is_err());

    // Test polynomial arity mismatch
    let f2 = SparsePoly::new(
        3,
        vec![Term {
            coeff: F::ONE,
            exps: vec![1, 0, 0],
        }],
    ); // arity 3
    let m = Mat::from_row_major(2, 2, vec![F::ONE, F::ZERO, F::ZERO, F::ONE]);
    assert!(CcsStructure::new(vec![m.clone(), m], f2).is_err()); // 2 matrices vs arity 3

    println!("✅ RED TEAM: CCS structure validation correctly rejects malformed inputs");
}

#[test]
fn polynomial_evaluation_tampering_detection() {
    // Create a quadratic polynomial f(x,y) = x^2 + y^2 - 5
    let f = SparsePoly::new(
        2,
        vec![
            Term {
                coeff: F::ONE,
                exps: vec![2, 0],
            }, // x^2
            Term {
                coeff: F::ONE,
                exps: vec![0, 2],
            }, // y^2
            Term {
                coeff: -F::from_u64(5),
                exps: vec![0, 0],
            }, // -5
        ],
    );

    // Matrices that should produce x=2, y=1 when multiplied by witness
    let m1 = Mat::from_row_major(1, 2, vec![F::from_u64(2), F::ZERO]); // 2*z0 + 0*z1
    let m2 = Mat::from_row_major(1, 2, vec![F::ZERO, F::ONE]); // 0*z0 + 1*z1
    let s = CcsStructure::new(vec![m1, m2], f).unwrap();

    // Witness that produces the right evaluation: z = [1, 1] → x=2, y=1 → f = 4+1-5 = 0
    let x = vec![]; // all witness
    let w = vec![F::ONE, F::ONE];
    assert!(check_ccs_rowwise_zero(&s, &x, &w).is_ok());

    // Tampered witness: z = [1, 2] → x=2, y=2 → f = 4+4-5 = 3 ≠ 0
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
    // Create a matrix that when multiplied by [0,0,0] gives [0,0] to satisfy f(y)=y=0
    let m = Mat::from_row_major(2, 3, vec![F::ONE, F::ZERO, F::ZERO, F::ZERO, F::ONE, F::ZERO]); // 2x3 matrix
    let s = CcsStructure::new(vec![m], f).unwrap();

    // Correct lengths - z=[0,0,0] gives M*z=[0,0], so f([0,0])=[0,0] satisfies rowwise zero
    let x = vec![]; // no public input
    let w = vec![F::ZERO, F::ZERO, F::ZERO]; // witness length 3, total z length 3
    assert!(check_ccs_rowwise_zero(&s, &x, &w).is_ok());

    // Wrong total length
    let w_short = vec![F::ZERO]; // witness too short, total z length 1
    assert!(check_ccs_rowwise_zero(&s, &x, &w_short).is_err());

    let w_long = vec![F::ZERO, F::ZERO, F::ZERO, F::ZERO]; // witness too long, total z length 4
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
    let m = Mat::from_row_major(2, 2, vec![F::ONE, F::ZERO, F::ZERO, F::ONE]); // Identity matrix
    let s = CcsStructure::new(vec![m], f).unwrap();

    let x = vec![F::ONE];
    let w = vec![F::ONE]; // z = [1, 1], M*z = [1, 1]
    let e = F::from_u64(2);

    // Correct slack vector length
    let u = vec![F::from_u64(2), F::from_u64(2)]; // u = [2, 2], e*u = [4, 4] ≠ [1, 1]
    assert!(check_ccs_rowwise_relaxed(&s, &x, &w, Some(&u), Some(e)).is_err());

    // Wrong slack vector length
    let u_short = vec![F::from_u64(2)]; // length 1, should be 2
    assert!(check_ccs_rowwise_relaxed(&s, &x, &w, Some(&u_short), Some(e)).is_err());

    // Correct relation: M*z = [1, 1], e*u = [1, 1] with e=1, u=[1, 1]
    let e_correct = F::ONE;
    let u_correct = vec![F::ONE, F::ONE];
    assert!(check_ccs_rowwise_relaxed(&s, &x, &w, Some(&u_correct), Some(e_correct)).is_ok());

    println!("✅ RED TEAM: Relaxed slack vector validation works correctly");
}

#[test]
fn matrix_polynomial_consistency_attack() {
    // Create inconsistent setup where polynomial arity doesn't match matrix count
    // This should be caught during CcsStructure creation, not during relation check

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

    // This should work: 2 matrices, arity 2
    assert!(CcsStructure::new(vec![m1.clone(), m2.clone()], f_double.clone()).is_ok());

    // This should fail: 2 matrices, arity 1
    assert!(CcsStructure::new(vec![m1.clone(), m2.clone()], f_single.clone()).is_err());

    // This should fail: 1 matrix, arity 2
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

    // Zero rows
    let m_zero_rows = Mat::from_row_major(0, 5, vec![]);
    assert!(CcsStructure::new(vec![m_zero_rows], f.clone()).is_err());

    // Zero columns
    let m_zero_cols = Mat::from_row_major(5, 0, vec![]);
    assert!(CcsStructure::new(vec![m_zero_cols], f.clone()).is_err());

    println!("✅ RED TEAM: Zero-dimension matrices correctly rejected");
}

#[test]
fn large_polynomial_degree_handling() {
    // Create a high-degree polynomial to test evaluation robustness
    let f = SparsePoly::new(
        2,
        vec![
            Term {
                coeff: F::ONE,
                exps: vec![10, 0],
            }, // x^10
            Term {
                coeff: F::ONE,
                exps: vec![0, 10],
            }, // y^10
            Term {
                coeff: -F::from_u64(2),
                exps: vec![5, 5],
            }, // -2*x^5*y^5
        ],
    );

    let m1 = Mat::from_row_major(1, 2, vec![F::from_u64(2), F::ZERO]); // x = 2
    let m2 = Mat::from_row_major(1, 2, vec![F::ZERO, F::ONE]); // y = 1
    let s = CcsStructure::new(vec![m1, m2], f).unwrap();

    // z = [1, 1] gives x=2, y=1
    // f(2,1) = 2^10 + 1^10 - 2*(2^5)*(1^5) = 1024 + 1 - 2*32*1 = 1025 - 64 = 961
    let x = vec![];
    let w = vec![F::ONE, F::ONE];

    // This should not equal zero, so check should fail
    assert!(check_ccs_rowwise_zero(&s, &x, &w).is_err());

    // For relaxed version with e=1, u=[961], should pass
    let u = vec![F::from_u64(961)];
    let e = F::ONE;
    assert!(check_ccs_rowwise_relaxed(&s, &x, &w, Some(&u), Some(e)).is_ok());

    println!("✅ RED TEAM: High-degree polynomial evaluation handles correctly");
}
