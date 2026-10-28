#![allow(non_snake_case)]
use neo_ajtai::{commit, s_lincomb, setup, verify_open, Commitment, PP};
use neo_math::ring::{Rq as RqEl, D};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as Fq;
use rand::{Rng, SeedableRng};
use std::convert::TryInto;
use std::panic::catch_unwind;

#[test]
fn ajtai_s_lincomb_rejects_cross_codomain_commitments() {
    let narrow = Commitment::zeros(1);
    let mut wide_a = Commitment::zeros(2);
    let mut wide_b = Commitment::zeros(2);
    wide_a.data_mut()[D] = Fq::ONE;
    wide_b.data_mut()[D] = Fq::from_u64(2);

    assert_eq!(narrow.data().len(), narrow.d() * narrow.kappa());
    assert_eq!(wide_a.data().len(), wide_a.d() * wide_a.kappa());
    assert_eq!(wide_b.data().len(), wide_b.d() * wide_b.kappa());

    let zero = RqEl::from_field_scalar(Fq::ZERO);
    let one = RqEl::from_field_scalar(Fq::ONE);
    let mixed_a = s_lincomb(&[zero, one], &[narrow.clone(), wide_a]);
    let mixed_b = s_lincomb(&[zero, one], &[narrow, wide_b]);
    let accepted_and_collided = matches!((&mixed_a, &mixed_b), (Ok(a), Ok(b)) if a == b);

    assert!(
        !accepted_and_collided,
        "Ajtai S-linear combination accepted distinct canonical commitments from a wider codomain and discarded their extra column"
    );
}

#[test]
fn ajtai_verify_open_rejects_wrong_length_without_panicking() {
    let mut rng = rand::rngs::StdRng::from_seed([0xA5; 32]);
    let pp: PP<neo_math::ring::Rq> = setup(&mut rng, D, 2, 1).expect("setup");
    let z = vec![Fq::ZERO; D];
    let commitment = commit(&pp, &z);
    let malformed = vec![Fq::ZERO; D - 1];

    let result = catch_unwind(|| verify_open(&pp, &commitment, &malformed));
    let accepted = result.expect("public opening verifier must return false, not panic");
    assert!(!accepted, "wrong-length Ajtai opening must be rejected");
}

#[test]
fn ajtai_setup_rejects_zero_module_rank() {
    let mut serial_rng = rand::rngs::StdRng::from_seed([0xC1; 32]);
    let serial = setup(&mut serial_rng, D, 0, 1);
    let false_opening_accepted = serial
        .as_ref()
        .map(|pp| {
            let zero = vec![Fq::ZERO; D];
            let mut one = zero.clone();
            one[0] = Fq::ONE;
            let commitment = commit(pp, &zero);
            assert_eq!(commitment, commit(pp, &one));
            verify_open(pp, &commitment, &one)
        })
        .unwrap_or(false);

    assert!(
        serial.is_err() && !false_opening_accepted,
        "Ajtai setup accepted kappa=0 (serial_ok={}) and false opening={false_opening_accepted}",
        serial.is_ok(),
    );
}

#[test]
fn ajtai_opening_rejects_one_digit_flip() {
    let mut rng = rand::rngs::StdRng::from_seed([7u8; 32]);
    let m = 8usize;
    let pp: PP<neo_math::ring::Rq> = setup(&mut rng, D, 8, m).expect("Setup should succeed");

    let Z: Vec<Fq> = (0..D * m)
        .map(|_| Fq::from_u64(rng.random::<bool>() as u64))
        .collect();
    let c = commit(&pp, &Z);

    let mut Z_bad = Z.clone();
    Z_bad[0] += Fq::ONE;

    assert!(
        !verify_open(&pp, &c, &Z_bad),
        "Ajtai opening MUST fail on any digit tamper"
    );
}

#[test]
fn ajtai_s_linearity_positive_control() {
    use neo_ajtai::s_lincomb;
    use neo_math::{ring::Rq, s_action::SAction};

    let mut rng = rand::rngs::StdRng::from_seed([9u8; 32]);
    let m = 4usize;
    let pp: PP<Rq> = setup(&mut rng, D, 8, m).expect("Setup should succeed");

    let Z1: Vec<Fq> = (0..D * m).map(|i| Fq::from_u64(i as u64 % 3)).collect();
    let Z2: Vec<Fq> = (0..D * m).map(|i| Fq::from_u64(i as u64 % 5)).collect();
    let c1 = commit(&pp, &Z1);
    let c2 = commit(&pp, &Z2);

    let mut coeffs1 = [Fq::ZERO; D];
    let mut coeffs2 = [Fq::ZERO; D];
    for i in 0..D {
        coeffs1[i] = Fq::from_u64(rng.random::<u8>() as u64);
        coeffs2[i] = Fq::from_u64(rng.random::<u8>() as u64);
    }
    let rho1 = neo_math::cf_inv(coeffs1);
    let rho2 = neo_math::cf_inv(coeffs2);

    let lhs = { s_lincomb(&[rho1, rho2], &[c1.clone(), c2.clone()]).expect("S-lincomb should succeed") };
    let s1 = SAction::from_ring(rho1);
    let s2 = SAction::from_ring(rho2);
    let mut Z_lin = vec![Fq::ZERO; D * m];
    for col in 0..m {
        let z1_col: [Fq; D] = Z1[col * D..(col + 1) * D].try_into().unwrap();
        let z2_col: [Fq; D] = Z2[col * D..(col + 1) * D].try_into().unwrap();
        let a = s1.apply_vec(&z1_col);
        let b = s2.apply_vec(&z2_col);
        for r in 0..D {
            Z_lin[col * D + r] = a[r] + b[r];
        }
    }
    let rhs = commit(&pp, &Z_lin);
    assert_eq!(lhs, rhs, "S-linearity must hold (positive control)");
}

#[test]
fn ajtai_public_parameters_reject_a_short_row() {
    let element = RqEl(core::array::from_fn(|_| Fq::ONE));
    assert!(PP::new(D, 1, 2, vec![vec![element]]).is_err());
    assert!(PP::new(D, 1, 2, vec![vec![element, element]]).is_ok());
}

#[test]
fn ajtai_public_parameters_reject_zero_rank() {
    assert!(PP::<RqEl>::new(D, 0, 1, Vec::new()).is_err());
}
