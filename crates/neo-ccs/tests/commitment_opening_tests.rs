//! Tests for Ajtai commitment opening and linear combination gadgets

use neo_ccs::gadgets::commitment_opening::*;
use neo_ccs::relations::check_ccs_rowwise_zero;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

type F = Goldilocks;

#[test]
fn test_commitment_lincomb_production() {
    let l = 3;
    let rho = F::from_u64(7);
    let c_prev = vec![F::from_u64(10), F::from_u64(20), F::from_u64(30)];
    let c_step = vec![F::from_u64(1), F::from_u64(2), F::from_u64(3)];

    let (witness, c_next) = build_commitment_lincomb_witness(rho, &c_prev, &c_step);
    let public_input = build_commitment_lincomb_public_input(rho, &c_prev, &c_step, &c_next);
    let ccs = commitment_lincomb_ccs(l);

    // Verify the computation
    assert_eq!(c_next[0], c_prev[0] + rho * c_step[0]); // 10 + 7*1 = 17
    assert_eq!(c_next[1], c_prev[1] + rho * c_step[1]); // 20 + 7*2 = 34
    assert_eq!(c_next[2], c_prev[2] + rho * c_step[2]); // 30 + 7*3 = 51

    // Should satisfy the CCS constraints
    assert!(check_ccs_rowwise_zero(&ccs, &public_input, &witness).is_ok());
}

#[test]
fn test_commitment_opening_basic() {
    let msg_len = 4;
    let _commit_len = 2;

    // Simple test rows (identity-like for easy verification)
    let rows = vec![
        vec![F::ONE, F::ZERO, F::ZERO, F::ZERO], // c_step[0] = Z[0]
        vec![F::ZERO, F::ONE, F::ONE, F::ZERO],  // c_step[1] = Z[1] + Z[2]
    ];

    let z_digits = vec![F::from_u64(5), F::from_u64(3), F::from_u64(2), F::from_u64(1)];
    let witness = build_opening_witness(&z_digits);

    // Expected c_step values
    let c_step = vec![
        F::from_u64(5),     // Z[0] = 5
        F::from_u64(3 + 2), // Z[1] + Z[2] = 3 + 2 = 5
    ];

    let ccs = commitment_opening_from_rows_ccs(&rows, msg_len);

    // Should satisfy the opening constraints
    assert!(check_ccs_rowwise_zero(&ccs, &c_step, &witness).is_ok());
}

#[test]
fn test_commitment_lincomb_security() {
    let l = 2;
    let rho = F::from_u64(5);
    let c_prev = vec![F::from_u64(100), F::from_u64(200)];
    let c_step = vec![F::from_u64(1), F::from_u64(2)];

    let (witness, c_next) = build_commitment_lincomb_witness(rho, &c_prev, &c_step);
    let public_input = build_commitment_lincomb_public_input(rho, &c_prev, &c_step, &c_next);
    let ccs = commitment_lincomb_ccs(l);

    // Valid case should pass
    assert!(check_ccs_rowwise_zero(&ccs, &public_input, &witness).is_ok());

    // Tamper with ρ - should fail
    let mut bad_public = public_input.clone();
    bad_public[0] = F::from_u64(999); // tamper with ρ
    assert!(check_ccs_rowwise_zero(&ccs, &bad_public, &witness).is_err());

    // Tamper with c_next - should fail
    let mut bad_public2 = public_input.clone();
    bad_public2[1 + 2 * l] = F::from_u64(999); // tamper with c_next[0]
    assert!(check_ccs_rowwise_zero(&ccs, &bad_public2, &witness).is_err());
}

#[test]
fn test_commitment_opening_large_rows() {
    let msg_len = 8;
    let _commit_len = 3;

    // Create some non-trivial rows
    let rows = vec![
        vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
            F::from_u64(0),
            F::from_u64(0),
            F::from_u64(0),
            F::from_u64(0),
        ],
        vec![
            F::from_u64(0),
            F::from_u64(0),
            F::from_u64(1),
            F::from_u64(1),
            F::from_u64(1),
            F::from_u64(1),
            F::from_u64(0),
            F::from_u64(0),
        ],
        vec![
            F::from_u64(5),
            F::from_u64(0),
            F::from_u64(0),
            F::from_u64(0),
            F::from_u64(0),
            F::from_u64(0),
            F::from_u64(0),
            F::from_u64(7),
        ],
    ];

    let z_digits = vec![
        F::from_u64(1),
        F::from_u64(1),
        F::from_u64(2),
        F::from_u64(2),
        F::from_u64(3),
        F::from_u64(3),
        F::from_u64(4),
        F::from_u64(4),
    ];
    let witness = build_opening_witness(&z_digits);

    // Compute expected c_step values
    let c_step = vec![
        F::from_u64(1 * 1 + 2 * 1 + 3 * 2 + 4 * 2), // 1+2+6+8 = 17
        F::from_u64(1 * 2 + 1 * 2 + 1 * 3 + 1 * 3), // 2+2+3+3 = 10
        F::from_u64(5 * 1 + 7 * 4),                 // 5+28 = 33
    ];

    let ccs = commitment_opening_from_rows_ccs(&rows, msg_len);

    // Should satisfy the opening constraints
    assert!(check_ccs_rowwise_zero(&ccs, &c_step, &witness).is_ok());
}

#[test]
fn test_commitment_lincomb_large_vectors() {
    let l = 5;
    let rho = F::from_u64(13);
    let c_prev = vec![
        F::from_u64(10),
        F::from_u64(20),
        F::from_u64(30),
        F::from_u64(40),
        F::from_u64(50),
    ];
    let c_step = vec![
        F::from_u64(1),
        F::from_u64(2),
        F::from_u64(3),
        F::from_u64(4),
        F::from_u64(5),
    ];

    let (witness, c_next) = build_commitment_lincomb_witness(rho, &c_prev, &c_step);
    let public_input = build_commitment_lincomb_public_input(rho, &c_prev, &c_step, &c_next);
    let ccs = commitment_lincomb_ccs(l);

    // Verify the computation for each coordinate
    for i in 0..l {
        let expected = c_prev[i] + rho * c_step[i];
        assert_eq!(c_next[i], expected);
    }

    // Should satisfy the CCS constraints
    assert!(check_ccs_rowwise_zero(&ccs, &public_input, &witness).is_ok());
}
