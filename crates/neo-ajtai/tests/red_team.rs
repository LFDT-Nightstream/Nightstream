// crates/neo-ajtai/tests/red_team.rs
#![allow(non_snake_case)] // Allow Z, Z_bad, etc. for matrix notation consistency
use neo_ajtai::{
    commit, decomp_b, s_lincomb, setup, setup_par, verify_open, verify_split_open, Commitment, DecompStyle, PP,
};
use neo_math::ring::{Rq as RqEl, D};
use neo_params::NeoParams;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks as Fq;
use rand::{Rng, SeedableRng};
use std::convert::TryInto;
use std::panic::catch_unwind;

#[test]
fn ajtai_s_lincomb_rejects_cross_codomain_commitments() {
    let narrow = Commitment::zeros(D, 1);
    let mut wide_a = Commitment::zeros(D, 2);
    let mut wide_b = Commitment::zeros(D, 2);
    wide_a.data[D] = Fq::ONE;
    wide_b.data[D] = Fq::from_u64(2);

    assert_eq!(narrow.data.len(), narrow.d * narrow.kappa);
    assert_eq!(wide_a.data.len(), wide_a.d * wide_a.kappa);
    assert_eq!(wide_b.data.len(), wide_b.d * wide_b.kappa);

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
fn decomposition_preserves_digits_for_every_valid_u32_base() {
    let b = (1u32 << 31) + 1;
    assert!(
        NeoParams::new(Fq::ORDER_U64, 81, 54, 18, 1, b, 2, 1, 2, 1).is_ok(),
        "the public parameter validator admits this base"
    );
    let value = Fq::from_u64(1u64 << 31);
    let digits = decomp_b(&[value], b, 1, DecompStyle::NonNegative);

    assert_eq!(digits, vec![value], "one base-b digit must round-trip");
}

#[test]
fn ajtai_verify_split_open_checks_each_child_opening() {
    let mut rng = rand::rngs::StdRng::from_seed([0x5A; 32]);
    let pp: PP<neo_math::ring::Rq> = setup(&mut rng, D, 2, 1).expect("setup");
    let zero_witness = vec![Fq::ZERO; D];
    let parent = commit(&pp, &zero_witness);

    let mut child_zero = parent.clone();
    let mut child_one = parent.clone();
    child_zero.data[0] = Fq::from_u64(2);
    child_one.data[0] = Fq::ZERO - Fq::ONE;
    assert!(!verify_open(&pp, &child_zero, &zero_witness));
    assert!(!verify_open(&pp, &child_one, &zero_witness));
    let canceled_children_accepted = verify_split_open(
        &pp,
        &parent,
        2,
        &[child_zero, child_one],
        &[zero_witness.clone(), zero_witness],
    );

    assert!(
        !canceled_children_accepted,
        "split-opening verifier accepted individually false child openings that canceled only in aggregate"
    );
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

    let mut parallel_rng = rand::rngs::StdRng::from_seed([0xC2; 32]);
    let parallel = setup_par(&mut parallel_rng, D, 0, 1);

    assert!(
        serial.is_err() && parallel.is_err() && !false_opening_accepted,
        "Ajtai setup accepted kappa=0 (serial_ok={}, parallel_ok={}) and false opening={false_opening_accepted}",
        serial.is_ok(),
        parallel.is_ok(),
    );
}

#[test]
fn ajtai_opening_rejects_one_digit_flip() {
    let mut rng = rand::rngs::StdRng::from_seed([7u8; 32]);
    let m = 8usize;
    let pp: PP<neo_math::ring::Rq> = setup(&mut rng, D, 8, m).expect("Setup should succeed");

    // make a small "witness" z (base-field entries)
    let mut z = vec![Fq::ZERO; m];
    for x in &mut z {
        *x = Fq::from_u64(rng.random::<u16>() as u64);
    }

    // decompose to Z (d×m), col-major for commit()
    let Z = decomp_b(&z, 2, D, DecompStyle::Balanced);
    let c = commit(&pp, &Z);

    // tamper one coefficient
    let mut Z_bad = Z.clone();
    Z_bad[0] += Fq::ONE;

    assert!(
        !verify_open(&pp, &c, &Z_bad),
        "Ajtai opening MUST fail on any digit tamper"
    );
}

#[test]
fn ajtai_verify_split_open_rejects_tampered_ci() {
    let mut rng = rand::rngs::StdRng::from_seed([8u8; 32]);
    let m = 6usize;
    let pp: PP<neo_math::ring::Rq> = setup(&mut rng, D, 8, m).expect("Setup should succeed");

    // random z, decompose at base b=2, split into k slices
    let z = (0..m)
        .map(|_| Fq::from_u64(rng.random::<u16>() as u64))
        .collect::<Vec<_>>();
    let Z = decomp_b(&z, 2, D, DecompStyle::Balanced);
    let c = commit(&pp, &Z);

    let k = 8usize;
    let Zis = neo_ajtai::split_b(&Z, 2, D, m, k, DecompStyle::Balanced);
    let mut cis: Vec<Commitment> = Zis.iter().map(|Zi| commit(&pp, Zi)).collect();

    // baseline: correct split passes
    assert!(verify_split_open(&pp, &c, 2, &cis, &Zis));

    // red-team: flip one limb in c_0
    cis[0].data[0] += Fq::ONE;
    assert!(
        !verify_split_open(&pp, &c, 2, &cis, &Zis),
        "Split opening MUST reject tampered c_i"
    );
}

#[test]
fn ajtai_s_linearity_positive_control() {
    // L(ρ1 Z1 + ρ2 Z2) == ρ1 L(Z1) + ρ2 L(Z2)
    use neo_ajtai::s_lincomb;
    use neo_math::{ring::Rq, s_action::SAction};

    let mut rng = rand::rngs::StdRng::from_seed([9u8; 32]);
    let m = 4usize;
    let pp: PP<Rq> = setup(&mut rng, D, 8, m).expect("Setup should succeed");

    let Z1 = decomp_b(&vec![Fq::from_u64(3); m], 2, D, DecompStyle::Balanced);
    let Z2 = decomp_b(&vec![Fq::from_u64(5); m], 2, D, DecompStyle::Balanced);
    let c1 = commit(&pp, &Z1);
    let c2 = commit(&pp, &Z2);

    // choose two random ring elements via random coeffs → SAction
    let mut coeffs1 = [Fq::ZERO; D];
    let mut coeffs2 = [Fq::ZERO; D];
    for i in 0..D {
        coeffs1[i] = Fq::from_u64(rng.random::<u8>() as u64);
        coeffs2[i] = Fq::from_u64(rng.random::<u8>() as u64);
    }
    let rho1 = neo_math::cf_inv(coeffs1);
    let rho2 = neo_math::cf_inv(coeffs2);

    // compute both sides
    let lhs = {
        // ρ1 Z1 + ρ2 Z2 in the commitment domain: act column-wise on commitments then add
        s_lincomb(&[rho1, rho2], &[c1.clone(), c2.clone()]).expect("S-lincomb should succeed")
    };
    // For a ground-truth check, recompute via linearity on Z then commit
    // Z' = ρ1·Z1 + ρ2·Z2 (apply S-action to each column of Z col-major)
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
