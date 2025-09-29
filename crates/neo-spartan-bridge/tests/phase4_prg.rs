// crates/neo-spartan-bridge/tests/phase4_prg.rs
#![allow(deprecated)]
use p3_field::{PrimeField64, PrimeCharacteristicRing};
use p3_goldilocks::Goldilocks as F;

use neo_spartan_bridge::ajtai_prg::expand_row_from_seed;

#[test]
fn prg_rows_deterministic_and_ip_defined() {
    let seed = [42u8; 32];
    let row_len = 64usize;
    let rows = 7usize;

    // Balanced small digits in range for b=3 (±2)
    let mut z = vec![0i64; row_len];
    for (i, zi) in z.iter_mut().enumerate() {
        let v = ((i * 31 + 7) % 5) as i64 - 2; // in [-2,2]
        *zi = v;
    }

    for i in 0..rows {
        let r1 = expand_row_from_seed(seed, i as u32, row_len);
        let r2 = expand_row_from_seed(seed, i as u32, row_len);
        assert_eq!(r1, r2, "PRG must be deterministic for row {}", i);

        // Host-side inner product sanity
        let mut acc = F::ZERO;
        for (a, &zv) in r1.iter().zip(z.iter()) {
            let zf = if zv >= 0 { F::from_u64(zv as u64) } else { -F::from_u64((-zv) as u64) };
            acc += *a * zf;
        }
        // Not a strict invariant, but should be a concrete field element
        assert!(acc != F::ZERO, "unexpected zero inner product (sanity)");
    }
}

#[test]
fn prg_mode_end_to_end_lean_proof() {
    use neo_ccs::{MEInstance, MEWitness};

    // Enable PRG-derived Ajtai rows inside the circuit
    std::env::set_var("NEO_ENABLE_PRG_ROWS", "1");
    // Enable the in-circuit RLC guard for this test
    // RLC guard is enabled by default now; no env needed

    // Set up a small instance in PRG mode: no Ajtai rows, no PP
    let seed = [7u8; 32];
    let base_b = 3u64; // allow digits in [-2,2]
    let z_len = 8usize; // already power-of-two to avoid padding surprises

    let mut z_digits = vec![0i64; z_len];
    for (i, zi) in z_digits.iter_mut().enumerate() {
        *zi = match i % 5 { 0 => -2, 1 => -1, 2 => 0, 3 => 1, _ => 2 };
    }

    // Derive c_coords from PRG rows so the circuit can bind ⟨row_i, z⟩ = c_i
    let rows = 4usize;
    let mut c_coords: Vec<neo_math::F> = Vec::with_capacity(rows);
    for i in 0..rows {
        let row = expand_row_from_seed(seed, i as u32, z_len);
        let mut acc = F::ZERO;
        for (a, &zv) in row.iter().zip(z_digits.iter()) {
            let zf = if zv >= 0 { F::from_u64(zv as u64) } else { -F::from_u64((-zv) as u64) };
            acc += *a * zf;
        }
        c_coords.push(neo_math::F::from_u64(acc.as_canonical_u64()));
    }

    // Empty ME evals for this smoke test (Ajtai binding via PRG is sufficient)
    let me = MEInstance {
        c_coords: c_coords.clone(),
        y_outputs: vec![],
        r_point: vec![],
        base_b,
        header_digest: seed,
        // For the RLC guard to hold, use the same coords as step coords
        c_step_coords: c_coords.clone(),
        u_offset: 0,
        u_len: 0,
    };
    let wit = MEWitness { z_digits, weight_vectors: vec![], ajtai_rows: None };

    // Prove in lean mode (no PP, no Ajtai rows)
    let proof = neo_spartan_bridge::compress_me_to_lean_proof_with_pp(&me, &wit, None)
        .expect("PRG-mode lean proof should succeed");
    let ok = neo_spartan_bridge::verify_lean_proof(&proof).expect("verification runs");
    assert!(ok, "lean proof must verify in PRG mode");

    // Tamper c_step_coords → guard must fail proving
    let mut me_bad = me.clone();
    me_bad.c_step_coords[0] = me_bad.c_step_coords[0] + neo_math::F::from_u64(1);
    let res_bad = neo_spartan_bridge::compress_me_to_lean_proof_with_pp(&me_bad, &wit, None)
        .expect("proving may still produce a proof; verification should fail");
    let ok_bad = neo_spartan_bridge::verify_lean_proof(&res_bad);
    assert!(ok_bad.is_err() || !ok_bad.unwrap(), "tampering c_step_coords should make verification fail");
}

#[test]
fn prg_mode_tamper_header_digest_fails() {
    use neo_ccs::{MEInstance, MEWitness};

    std::env::set_var("NEO_ENABLE_PRG_ROWS", "1");

    let seed = [9u8; 32];
    let base_b = 3u64;
    let z_len = 8usize;

    let mut z_digits = vec![0i64; z_len];
    for (i, zi) in z_digits.iter_mut().enumerate() { *zi = match i % 5 { 0 => -2, 1 => -1, 2 => 0, 3 => 1, _ => 2 }; }

    let rows = 4usize;
    let mut c_coords: Vec<neo_math::F> = Vec::with_capacity(rows);
    for i in 0..rows {
        let row = expand_row_from_seed(seed, i as u32, z_len);
        let mut acc = F::ZERO;
        for (a, &zv) in row.iter().zip(z_digits.iter()) {
            let zf = if zv >= 0 { F::from_u64(zv as u64) } else { -F::from_u64((-zv) as u64) };
            acc += *a * zf;
        }
        c_coords.push(neo_math::F::from_u64(acc.as_canonical_u64()));
    }

    let me = MEInstance {
        c_coords: c_coords.clone(),
        y_outputs: vec![],
        r_point: vec![],
        base_b,
        header_digest: seed,
        c_step_coords: c_coords.clone(),
        u_offset: 0,
        u_len: 0,
    };
    let wit = MEWitness { z_digits, weight_vectors: vec![], ajtai_rows: None };

    // Honest
    let proof = neo_spartan_bridge::compress_me_to_lean_proof_with_pp(&me, &wit, None).expect("prove ok");
    assert!(neo_spartan_bridge::verify_lean_proof(&proof).expect("verify runs"));

    // Tamper header_digest only (seed flip) -> host PRG parity must reject at prove-time
    let mut me_bad = me.clone();
    me_bad.header_digest[0] ^= 1;
    let res_bad = neo_spartan_bridge::compress_me_to_lean_proof_with_pp(&me_bad, &wit, None);
    assert!(res_bad.is_err(), "tampering header_digest should be rejected at prove-time");
}

#[test]
fn prg_mode_tamper_c_coords_fails() {
    use neo_ccs::{MEInstance, MEWitness};

    std::env::set_var("NEO_ENABLE_PRG_ROWS", "1");

    let seed = [11u8; 32];
    let base_b = 3u64;
    let z_len = 8usize;

    let mut z_digits = vec![0i64; z_len];
    for (i, zi) in z_digits.iter_mut().enumerate() { *zi = match i % 5 { 0 => -2, 1 => -1, 2 => 0, 3 => 1, _ => 2 }; }

    let rows = 4usize;
    let mut c_coords: Vec<neo_math::F> = Vec::with_capacity(rows);
    for i in 0..rows {
        let row = expand_row_from_seed(seed, i as u32, z_len);
        let mut acc = F::ZERO;
        for (a, &zv) in row.iter().zip(z_digits.iter()) {
            let zf = if zv >= 0 { F::from_u64(zv as u64) } else { -F::from_u64((-zv) as u64) };
            acc += *a * zf;
        }
        c_coords.push(neo_math::F::from_u64(acc.as_canonical_u64()));
    }

    let me = MEInstance {
        c_coords: c_coords.clone(),
        y_outputs: vec![],
        r_point: vec![],
        base_b,
        header_digest: seed,
        c_step_coords: c_coords.clone(),
        u_offset: 0,
        u_len: 0,
    };
    let wit = MEWitness { z_digits, weight_vectors: vec![], ajtai_rows: None };

    // Honest
    let proof = neo_spartan_bridge::compress_me_to_lean_proof_with_pp(&me, &wit, None).expect("prove ok");
    assert!(neo_spartan_bridge::verify_lean_proof(&proof).expect("verify runs"));

    // Tamper c_coords
    let mut me_bad = me.clone();
    me_bad.c_coords[0] += neo_math::F::ONE;
    // In PRG mode, proving should fail fast due to host parity check
    let res_bad = neo_spartan_bridge::compress_me_to_lean_proof_with_pp(&me_bad, &wit, None);
    assert!(res_bad.is_err(), "tampering c_coords should be rejected at prove-time");
}
