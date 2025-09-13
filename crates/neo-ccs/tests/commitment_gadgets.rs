//! Commitment gadgets (opening + lincomb) tests

use neo_ccs::gadgets::commitment_opening::{commitment_opening_from_rows_ccs, commitment_lincomb_ccs};
use neo_ccs::check_ccs_rowwise_zero;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

fn fe(x: u64) -> F { F::from_u64(x) }

#[test]
fn commitment_opening_happy_and_tamper() {
    // msg_len = 5, l = 3
    let msg_len = 5;
    let l = 3;

    // Choose Z and rows L_i
    let z = vec![fe(2), fe(4), fe(6), fe(8), fe(10)];
    let rows = vec![
        vec![fe(1), fe(0), fe(1), fe(0), fe(1)], // L0
        vec![fe(2), fe(1), fe(0), fe(1), fe(0)], // L1
        vec![fe(0), fe(3), fe(0), fe(0), fe(1)], // L2
    ];

    // c_coords[i] = <L_i, Z>
    let mut c_coords = Vec::with_capacity(l);
    for i in 0..l {
        let mut acc = F::ZERO;
        for j in 0..msg_len { acc += rows[i][j] * z[j]; }
        c_coords.push(acc);
    }

    let ccs = commitment_opening_from_rows_ccs(&rows, msg_len);

    // witness = [1 | Z]
    let mut witness = Vec::with_capacity(1 + msg_len);
    witness.push(F::ONE); // const
    witness.extend_from_slice(&z);

    assert!(check_ccs_rowwise_zero(&ccs, &c_coords, &witness).is_ok());

    // tamper c_coords
    let mut bad = c_coords.clone();
    bad[0] = bad[0] + fe(1);
    assert!(check_ccs_rowwise_zero(&ccs, &bad, &witness).is_err());

    // tamper Z (witness)
    let mut witness_bad = witness.clone();
    let last = witness_bad.len()-1;
    witness_bad[last] = witness_bad[last] + fe(1);
    assert!(check_ccs_rowwise_zero(&ccs, &c_coords, &witness_bad).is_err());
}

#[test]
fn commitment_lincomb_happy_and_tamper() {
    let len = 4;
    let rho = fe(7);
    let c_prev = vec![fe(1), fe(2), fe(3), fe(4)];
    let c_step = vec![fe(5), fe(6), fe(7), fe(8)];
    let u: Vec<F> = c_step.iter().map(|&x| rho * x).collect();
    let c_next: Vec<F> = c_prev.iter().zip(&u).map(|(a,b)| *a + *b).collect();

    let ccs = commitment_lincomb_ccs(len);

    // witness = [1 | u]
    let mut witness = Vec::with_capacity(1 + len);
    witness.push(F::ONE);
    witness.extend_from_slice(&u);

    // public = [rho | c_prev | c_step | c_next]
    let mut public = vec![rho];
    public.extend_from_slice(&c_prev);
    public.extend_from_slice(&c_step);
    public.extend_from_slice(&c_next);

    assert!(check_ccs_rowwise_zero(&ccs, &public, &witness).is_ok());

    // tamper rho
    let mut public_bad = public.clone();
    public_bad[0] = public_bad[0] + fe(1);
    assert!(check_ccs_rowwise_zero(&ccs, &public_bad, &witness).is_err());

    // tamper c_next
    let mut public_bad2 = public.clone();
    let last = public_bad2.len() - 1;
    public_bad2[last] = public_bad2[last] + fe(1);
    assert!(check_ccs_rowwise_zero(&ccs, &public_bad2, &witness).is_err());
}

#[test]
fn commitment_opening_zero_case() {
    // Test with all-zero Z
    let msg_len = 3;
    let z = vec![fe(0), fe(0), fe(0)];
    let rows = vec![
        vec![fe(1), fe(2), fe(3)],
        vec![fe(4), fe(5), fe(6)],
    ];

    // c_coords should all be zero
    let c_coords = vec![fe(0), fe(0)];

    let ccs = commitment_opening_from_rows_ccs(&rows, msg_len);
    let mut witness = vec![F::ONE];
    witness.extend_from_slice(&z);

    assert!(check_ccs_rowwise_zero(&ccs, &c_coords, &witness).is_ok());
}

#[test]
fn commitment_lincomb_zero_case() {
    // Test with zero rho
    let len = 3;
    let rho = fe(0);
    let c_prev = vec![fe(10), fe(20), fe(30)];
    let c_step = vec![fe(1), fe(2), fe(3)];
    
    // With rho=0, u should be all zeros and c_next = c_prev
    let u = vec![fe(0), fe(0), fe(0)];
    let c_next = c_prev.clone();

    let ccs = commitment_lincomb_ccs(len);
    let mut witness = vec![F::ONE];
    witness.extend_from_slice(&u);

    let mut public = vec![rho];
    public.extend_from_slice(&c_prev);
    public.extend_from_slice(&c_step);
    public.extend_from_slice(&c_next);

    assert!(check_ccs_rowwise_zero(&ccs, &public, &witness).is_ok());
}

#[test]
fn commitment_opening_single_element() {
    // Test minimal case: msg_len=1, l=1
    let _z = vec![fe(42)];
    let rows = vec![vec![fe(5)]]; // Single row, single element
    let c_coords = vec![fe(42 * 5)]; // 42 * 5 = 210

    let ccs = commitment_opening_from_rows_ccs(&rows, 1);
    let witness = vec![F::ONE, fe(42)];

    assert!(check_ccs_rowwise_zero(&ccs, &c_coords, &witness).is_ok());
}

#[test]
fn commitment_lincomb_single_element() {
    // Test minimal case: len=1
    let len = 1;
    let rho = fe(3);
    let c_prev = vec![fe(10)];
    let c_step = vec![fe(4)];
    let _u = vec![fe(3 * 4)]; // rho * c_step
    let c_next = vec![fe(10 + 12)]; // c_prev + u

    let ccs = commitment_lincomb_ccs(len);
    let witness = vec![F::ONE, fe(12)]; // [1, u[0]]

    let mut public = vec![rho];
    public.extend_from_slice(&c_prev);
    public.extend_from_slice(&c_step);
    public.extend_from_slice(&c_next);

    assert!(check_ccs_rowwise_zero(&ccs, &public, &witness).is_ok());
}
