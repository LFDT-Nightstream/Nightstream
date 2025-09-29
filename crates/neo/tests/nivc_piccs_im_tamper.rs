//! NIVC end-to-end Pi-CCS Im-lane tamper: ensure final proof rejects.
//!
//! This test exercises the Pi-CCS embed used in the IVC verifier circuit path by
//! constructing a minimal ME instance with a single CCS matrix and weight vectors
//! for both Re and Im lanes. It proves once honestly, then tampers one Im digit
//! and expects proving to fail (or, if a proof is produced, verification to reject).

#![allow(deprecated)]

use p3_field::PrimeCharacteristicRing;
use neo_math::F;

// Compute χ_r(row) over K given a single (re, im) pair (ℓ=1) and row index ∈ {0,1}
fn chi_pair(re: F, im: F, row: usize) -> (F, F) {
    if row & 1 == 1 { (re, im) } else { (F::ONE - re, -im) }
}

#[test]
fn nivc_piccs_im_lane_tamper() {
    use neo_spartan_bridge::pi_ccs_embed::{CcsCsr, PiCcsEmbed};

    // Minimal setup: d = D, 1 column, 2 rows in CCS matrix
    let d = neo_math::D;
    let base_b = 2u64;

    // Witness digits in small range {-1,0,1}
    let mut z_digits = vec![0i64; d];
    for (i, zi) in z_digits.iter_mut().enumerate().take(d) {
        *zi = match i % 3 { 0 => -1, 1 => 0, _ => 1 };
    }

    // Ajtai binding: one row selects first digit (for simplicity)
    let mut row0 = vec![F::ZERO; d];
    row0[0] = F::ONE;
    let c0 = if z_digits[0] >= 0 { F::from_u64(z_digits[0] as u64) } else { -F::from_u64((-z_digits[0]) as u64) };

    // Pi-CCS: 1 matrix, rows=2, cols=1 with entries {(0,0)=1, (1,0)=2}
    let mj = CcsCsr { rows: 2, cols: 1, entries: vec![(0, 0, F::ONE), (1, 0, F::from_u64(2))] };
    let pi = PiCcsEmbed { matrices: vec![mj.clone()] };

    // r_point (ℓ=1)
    let r_re = F::from_u64(3);
    let r_im = F::from_u64(5);

    // Compute v_re/v_im and expand to weights
    let (chi0_re, chi0_im) = chi_pair(r_re, r_im, 0);
    let (chi1_re, chi1_im) = chi_pair(r_re, r_im, 1);
    let v_re = F::ONE * chi0_re + F::from_u64(2) * chi1_re;
    let v_im = F::ONE * chi0_im + F::from_u64(2) * chi1_im;
    let b = F::from_u64(base_b);
    let mut pow_b = vec![F::ONE; d];
    for i in 1..d { pow_b[i] = pow_b[i - 1] * b; }
    let mut w_re = vec![F::ZERO; d];
    let mut w_im = vec![F::ZERO; d];
    for r in 0..d { w_re[r] = v_re * pow_b[r]; w_im[r] = v_im * pow_b[r]; }

    // Legacy ME instance/witness expected by the bridge
    let mut legacy_wit = neo_ccs::MEWitness { z_digits, weight_vectors: vec![w_re.clone(), w_im.clone()], ajtai_rows: Some(vec![row0]) };
    let legacy_me = neo_ccs::MEInstance {
        c_coords: vec![c0],
        y_outputs: vec![],
        r_point: vec![r_re, r_im],
        base_b,
        header_digest: [0u8; 32],
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
    };

    // Honest proof verifies
    let proof_ok = neo_spartan_bridge::compress_ivc_verifier_to_lean_proof_with_linkage_and_pi_ccs(
        &legacy_me, &legacy_wit, None, None, None, None, Some(pi.clone())
    ).expect("prove ok");
    assert!(neo_spartan_bridge::verify_lean_proof(&proof_ok).expect("verify runs"));

    // Tamper: flip one Im digit
    legacy_wit.weight_vectors[1][0] += F::ONE;
    let res_bad = neo_spartan_bridge::compress_ivc_verifier_to_lean_proof_with_linkage_and_pi_ccs(
        &legacy_me, &legacy_wit, None, None, None, None, Some(pi)
    );
    match res_bad {
        Ok(proof_bad) => {
            // If the prover produced a proof, verification must reject
            let res = neo_spartan_bridge::verify_lean_proof(&proof_bad);
            assert!(res.is_err() || !res.unwrap(), "tampered Im-lane weight must be rejected by final verifier");
        }
        Err(_) => {
            // Also acceptable: proving failed due to violated in-circuit constraints
        }
    }
}
