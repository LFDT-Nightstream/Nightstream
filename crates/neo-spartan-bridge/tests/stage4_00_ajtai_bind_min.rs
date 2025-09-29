//! Stage 4: Spartan minimal (Ajtai binding only, no EV)

use std::sync::Arc;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

#[test]
fn ajtai_bind_honest_proof_verifies() {
    // Build tiny Ajtai instance: m=1 for speed, kappa=2
    let d = neo_math::ring::D; let m = 1usize; let kappa = 2usize;
    // Balanced digits in {-1,0,1}
    let mut z_digits: Vec<i64> = Vec::with_capacity(d*m);
    for i in 0..(d*m) { z_digits.push(match i % 3 { 0 => -1, 1 => 0, _ => 1 }); }
    // Map to field for commitment
    let to_f = |zi: i64| if zi >= 0 { F::from_u64(zi as u64) } else { -F::from_u64((-zi) as u64) };
    let z_f: Vec<F> = z_digits.iter().copied().map(to_f).collect();

    // Ajtai PP and commitment
    use rand::SeedableRng; use rand::rngs::StdRng;
    let mut rng = StdRng::from_seed([42u8; 32]);
    let pp = neo_ajtai::setup(&mut rng, d, kappa, m).expect("setup");
    let c = neo_ajtai::commit(&pp, &z_f);

    // Legacy ME instance/witness (minimal)
    #[allow(deprecated)]
    let me = neo_ccs::MEInstance {
        c_coords: c.data.clone(),
        y_outputs: vec![],
        r_point: vec![],
        base_b: 2,
        header_digest: [0u8; 32],
        c_step_coords: vec![], u_offset: 0, u_len: 0,
    };
    #[allow(deprecated)]
    let wit = neo_ccs::MEWitness { z_digits: z_digits.clone(), weight_vectors: vec![], ajtai_rows: None };

    let proof = neo_spartan_bridge::compress_me_to_lean_proof_with_pp(&me, &wit, Some(Arc::new(pp))).expect("prove");
    let ok = neo_spartan_bridge::verify_lean_proof(&proof).expect("verify path");
    assert!(ok, "Lean proof should verify for honest Ajtai binding");
}

#[test]
fn ajtai_bind_tamper_digit_fails() {
    let d = neo_math::ring::D; let m = 1usize; let kappa = 2usize;
    let mut z_digits: Vec<i64> = Vec::with_capacity(d*m);
    for i in 0..(d*m) { z_digits.push(match i % 3 { 0 => -1, 1 => 0, _ => 1 }); }
    let to_f = |zi: i64| if zi >= 0 { F::from_u64(zi as u64) } else { -F::from_u64((-zi) as u64) };
    let z_f: Vec<F> = z_digits.iter().copied().map(to_f).collect();

    use rand::SeedableRng; use rand::rngs::StdRng;
    let mut rng = StdRng::from_seed([7u8; 32]);
    let pp = neo_ajtai::setup(&mut rng, d, kappa, m).expect("setup");
    let c = neo_ajtai::commit(&pp, &z_f);

    #[allow(deprecated)]
    let me = neo_ccs::MEInstance {
        c_coords: c.data.clone(), y_outputs: vec![], r_point: vec![], base_b: 2,
        header_digest: [0u8; 32], c_step_coords: vec![], u_offset: 0, u_len: 0,
    };
    // Tamper one digit (stay in range)
    z_digits[0] = -z_digits[0];
    #[allow(deprecated)]
    let wit_tampered = neo_ccs::MEWitness { z_digits, weight_vectors: vec![], ajtai_rows: None };
    // Proving should fail since Ajtai binding cannot be satisfied
    let res = neo_spartan_bridge::compress_me_to_lean_proof_with_pp(&me, &wit_tampered, Some(Arc::new(pp)));
    assert!(res.is_err(), "Tampered digits should cause proving to fail");
}

#[test]
fn rlc_guard_disabled_with_pp_succeeds() {
    // With PP present and non-empty c_step_coords, the RLC guard must stay OFF and proving should succeed
    let d = neo_math::ring::D; let m = 1usize; let kappa = 2usize;
    // Simple small-range digits
    let mut z_digits: Vec<i64> = Vec::with_capacity(d*m);
    for i in 0..(d*m) { z_digits.push(match i % 3 { 0 => -1, 1 => 0, _ => 1 }); }
    let to_f = |zi: i64| if zi >= 0 { F::from_u64(zi as u64) } else { -F::from_u64((-zi) as u64) };
    let z_f: Vec<F> = z_digits.iter().copied().map(to_f).collect();

    use rand::SeedableRng; use rand::rngs::StdRng;
    let mut rng = StdRng::from_seed([23u8; 32]);
    let pp = neo_ajtai::setup(&mut rng, d, kappa, m).expect("setup");
    let c = neo_ajtai::commit(&pp, &z_f);

    #[allow(deprecated)]
    let me = neo_ccs::MEInstance {
        c_coords: c.data.clone(),
        y_outputs: vec![],
        r_point: vec![],
        base_b: 2,
        header_digest: [0u8; 32],
        // Non-empty c_step_coords would previously activate the guard; with PP it must be ignored
        c_step_coords: c.data.clone(),
        u_offset: 0,
        u_len: 0,
    };
    #[allow(deprecated)]
    let wit = neo_ccs::MEWitness { z_digits, weight_vectors: vec![], ajtai_rows: None };

    // Guard stays off (pp present); proof should verify
    let proof = neo_spartan_bridge::compress_me_to_lean_proof_with_pp(&me, &wit, Some(Arc::new(pp))).expect("prove");
    let ok = neo_spartan_bridge::verify_lean_proof(&proof).expect("verify path");
    assert!(ok, "Lean proof should verify with PP present and guard disabled");
}
