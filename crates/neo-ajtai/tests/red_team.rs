// crates/neo-ajtai/tests/red_team.rs
#![allow(non_snake_case)] // Allow Z, Z_bad, etc. for matrix notation consistency
use neo_ajtai::{commit, decomp_b, setup, verify_open, verify_split_open, Commitment, DecompStyle, PP};
use neo_math::ring::D;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as Fq;
use rand::{Rng, SeedableRng};
use std::convert::TryInto;

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
