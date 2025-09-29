//! RLC guard seed hardening: flipping the fold/context digest must cause rejection.
#![allow(deprecated)]

use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks as F;

#[test]
fn rlc_guard_seed_flip_rejected() {
    use neo_ccs::{MEInstance, MEWitness};

    // Enable PRG-derived Ajtai rows and rely on RLC guard linkage
    std::env::set_var("NEO_ENABLE_PRG_ROWS", "1");

    let seed = [21u8; 32];
    let base_b = 3u64;
    let z_len = 8usize;

    let mut z_digits = vec![0i64; z_len];
    for (i, zi) in z_digits.iter_mut().enumerate() { *zi = match i % 5 { 0 => -2, 1 => -1, 2 => 0, 3 => 1, _ => 2 }; }

    // Build c_coords from PRG rows
    let rows = 4usize;
    let mut c_coords: Vec<neo_math::F> = Vec::with_capacity(rows);
    for i in 0..rows {
        let row = neo_spartan_bridge::ajtai_prg::expand_row_from_seed(seed, i as u32, z_len);
        let mut acc = F::ZERO;
        for (a, &zv) in row.iter().zip(z_digits.iter()) {
            let zf = if zv >= 0 { F::from_u64(zv as u64) } else { -F::from_u64((-zv) as u64) };
            acc += *a * zf;
        }
        c_coords.push(neo_math::F::from_u64(acc.as_canonical_u64()));
    }

    // Honest instance in PRG+RLC-guard mode
    let me = MEInstance {
        c_coords: c_coords.clone(),
        y_outputs: vec![],
        r_point: vec![],
        base_b,
        header_digest: seed,              // used for PRG rows and also as fold/context digest in circuit
        c_step_coords: c_coords.clone(),  // RLC guard ties these
        u_offset: 0,
        u_len: 0,
    };
    let wit = MEWitness { z_digits, weight_vectors: vec![], ajtai_rows: None };

    // Honest proof verifies
    let proof = neo_spartan_bridge::compress_me_to_lean_proof_with_pp(&me, &wit, None).expect("prove");
    assert!(neo_spartan_bridge::verify_lean_proof(&proof).expect("verify runs"));

    // Flip the seed (context/fold digest) -> RLC coefficients differ, guard breaks
    let mut me_bad = me.clone();
    me_bad.header_digest[0] ^= 1;
    let res_bad = neo_spartan_bridge::compress_me_to_lean_proof_with_pp(&me_bad, &wit, None);
    assert!(res_bad.is_err(), "tampering fold/context digest must be rejected at prove-time by RLC guard parity");
}

