#![allow(deprecated)]

use neo_spartan_bridge::{pi_ccs_embed::{CcsCsr, PiCcsEmbed}, verify_lean_proof, compress_ivc_verifier_to_lean_proof_with_linkage_and_pi_ccs};
use neo_ccs::{MEInstance, MEWitness};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

// Compute χ_r(row) over K given a single (re, im) pair (ℓ=1) and row index ∈ {0,1}
fn chi_pair(re: F, im: F, row: usize) -> (F, F) {
    // χ(0) = (1-re, -im); χ(1) = (re, im)
    if row & 1 == 1 { (re, im) } else { (F::ONE - re, -im) }
}

// Build a minimal ME instance and witness with Pi‑CCS embed and 2-lane weights (Re/Im)
fn build_me_with_piccs() -> (MEInstance, MEWitness, PiCcsEmbed) {
    let d = neo_math::D; // 54
    let base_b = 2u64;   // keep range constraints simple

    // Witness digits in range {−1,0,1}
    let mut z_digits = vec![0i64; d];
    for (i, zi) in z_digits.iter_mut().enumerate().take(d) {
        *zi = match i % 3 { 0 => -1, 1 => 0, _ => 1 };
    }

    // Ajtai rows: single row selecting the first limb
    let mut row0 = vec![F::ZERO; d];
    row0[0] = F::ONE;
    let c0 = if z_digits[0] >= 0 { F::from_u64(z_digits[0] as u64) } else { -F::from_u64((-z_digits[0]) as u64) };

    // Pi‑CCS matrices: one matrix (rows=2, cols=1) with entries {(0,0)=1, (1,0)=2}
    let mj = CcsCsr { rows: 2, cols: 1, entries: vec![(0, 0, F::ONE), (1, 0, F::from_u64(2))] };
    let pi = PiCcsEmbed { matrices: vec![mj.clone()] };

    // r_point = [re0, im0] (ℓ=1)
    let r_re = F::from_u64(3);
    let r_im = F::from_u64(5);

    // Compute v_re/v_im for column 0, then expand to digit weights w = v * b^r
    let (chi0_re, chi0_im) = chi_pair(r_re, r_im, 0);
    let (chi1_re, chi1_im) = chi_pair(r_re, r_im, 1);
    let v_re = F::ONE * chi0_re + F::from_u64(2) * chi1_re; // 1*(1-re) + 2*(re)
    let v_im = F::ONE * chi0_im + F::from_u64(2) * chi1_im; // 1*(-im) + 2*(im)
    let b = F::from_u64(base_b as u64);
    let mut pow_b = vec![F::ONE; d];
    for i in 1..d { pow_b[i] = pow_b[i - 1] * b; }
    let mut w_re = vec![F::ZERO; d];
    let mut w_im = vec![F::ZERO; d];
    for r in 0..d { w_re[r] = v_re * pow_b[r]; w_im[r] = v_im * pow_b[r]; }

    // y_outputs = <w, z>
    let dot = |w: &[F]| -> F {
        let mut acc = F::ZERO;
        for (i, wi) in w.iter().enumerate() {
            let zi = z_digits[i];
            let zf = if zi >= 0 { F::from_u64(zi as u64) } else { -F::from_u64((-zi) as u64) };
            acc += *wi * zf;
        }
        acc
    };
    let y_re = dot(&w_re);
    let y_im = dot(&w_im);

    let me = MEInstance {
        c_coords: vec![c0],
        y_outputs: vec![y_re, y_im],
        r_point: vec![r_re, r_im],
        base_b,
        header_digest: [0u8; 32],
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
    };
    let wit = MEWitness { z_digits, weight_vectors: vec![w_re, w_im], ajtai_rows: Some(vec![row0]) };
    (me, wit, pi)
}

#[test]
fn piccs_im_lane_tamper_im_digit_rejected() {
    let (me, mut wit, pi) = build_me_with_piccs();

    // Honest proof verifies
    let proof_ok = compress_ivc_verifier_to_lean_proof_with_linkage_and_pi_ccs(&me, &wit, None, None, None, None, Some(pi.clone()))
        .expect("prove");
    assert!(verify_lean_proof(&proof_ok).expect("verify runs"));

    // Tamper: flip one Im digit
    wit.weight_vectors[1][0] += F::ONE;
    let proof_bad = compress_ivc_verifier_to_lean_proof_with_linkage_and_pi_ccs(&me, &wit, None, None, None, None, Some(pi.clone()))
        .expect("prove tampered");
    let ok = verify_lean_proof(&proof_bad);
    assert!(ok.is_err() || !ok.unwrap(), "tampered Im-lane weight must be rejected");
}

#[test]
fn piccs_im_lane_tamper_re_digit_rejected() {
    let (me, mut wit, pi) = build_me_with_piccs();
    let proof_ok = compress_ivc_verifier_to_lean_proof_with_linkage_and_pi_ccs(&me, &wit, None, None, None, None, Some(pi.clone()))
        .expect("prove");
    assert!(verify_lean_proof(&proof_ok).expect("verify runs"));

    // Tamper: flip one Re digit
    wit.weight_vectors[0][0] += F::ONE;
    let proof_bad = compress_ivc_verifier_to_lean_proof_with_linkage_and_pi_ccs(&me, &wit, None, None, None, None, Some(pi.clone()))
        .expect("prove tampered");
    let ok = verify_lean_proof(&proof_bad);
    assert!(ok.is_err() || !ok.unwrap(), "tampered Re-lane weight must be rejected");
}

#[test]
fn piccs_im_lane_tamper_r_point_rejected() {
    let (mut me, wit, pi) = build_me_with_piccs();
    let proof_ok = compress_ivc_verifier_to_lean_proof_with_linkage_and_pi_ccs(&me, &wit, None, None, None, None, Some(pi.clone()))
        .expect("prove");
    assert!(verify_lean_proof(&proof_ok).expect("verify runs"));

    // Tamper: flip r_point (Re lane)
    me.r_point[0] += F::ONE;
    let proof_bad = compress_ivc_verifier_to_lean_proof_with_linkage_and_pi_ccs(&me, &wit, None, None, None, None, Some(pi))
        .expect("prove tampered");
    let ok = verify_lean_proof(&proof_bad);
    assert!(ok.is_err() || !ok.unwrap(), "tampered r_point must be rejected");
}
