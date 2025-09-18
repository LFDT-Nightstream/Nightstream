//! Red-team bridge tests against Hash‑MLE Spartan2 compression.
//!
//! Targets: 
//! - #7 (range violation) - Range‑violation with correct recomposition
//! - #23 (public‑input replay) - Public‑input replay under different `(c,X,y,r)`
//! - #25 (hash‑MLE IO ordering) - Hash‑MLE IO ordering / header mismatch 
//! - #1/#24 (Ajtai binding missing) - Missing Ajtai binding inside SNARK

#![cfg(feature = "redteam")]
#![allow(deprecated)] // Allow use of legacy bridge types for compatibility testing

use neo_spartan_bridge::{compress_me_to_spartan, verify_me_spartan, encode_bridge_io_header};
use neo_ccs::{MEInstance, MEWitness};
use p3_goldilocks::Goldilocks as F;
use p3_field::PrimeCharacteristicRing;

fn dot_f_z(row: &[F], z: &[i64]) -> F {
    row.iter().zip(z.iter()).fold(F::ZERO, |acc, (a, &zi)| {
        let zi_f = if zi >= 0 { F::from_u64(zi as u64) } else { -F::from_u64((-zi) as u64) };
        acc + (*a) * zi_f
    })
}

fn tiny_me_instance() -> (MEInstance, MEWitness) {
    // Witness digits (len = 8 = 2^3, nice for Hash‑MLE)
    let z = vec![1i64, 2, 3, 0, -1, 1, 0, 2];

    // Two Ajtai rows (unit vectors) so constraints become z0=c0 and z1=c1
    let ajtai_rows = vec![
        vec![F::ONE, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO],
        vec![F::ZERO, F::ONE, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO],
    ];
    let c0 = dot_f_z(&ajtai_rows[0], &z);
    let c1 = dot_f_z(&ajtai_rows[1], &z);
    let c_coords = vec![c0, c1]; // Match the number of Ajtai rows

    // ME weights: sum first 4, and z5+z7
    let w0 = vec![F::ONE, F::ONE, F::ONE, F::ONE, F::ZERO, F::ZERO, F::ZERO, F::ZERO];
    let w1 = vec![F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ONE, F::ZERO, F::ONE];
    let y0 = dot_f_z(&w0, &z);
    let y1 = dot_f_z(&w1, &z);
    let y_outputs = vec![y0, y1]; // Match the number of weight vectors

    let me = MEInstance {
        c_step_coords: vec![], // Pattern B: Populated by IVC layer, not folding
        u_offset: 0,
        u_len: 0,
        c_coords,
        y_outputs,
        r_point: vec![F::from_u64(3); 2], // not used by this tiny circuit
        base_b: 4,                        // => digits must be in [-3, 3]
        header_digest: [0u8; 32],
    };

    let wit = MEWitness {
        z_digits: z,
        weight_vectors: vec![w0, w1],
        ajtai_rows: Some(ajtai_rows),
    };

    (me, wit)
}

#[test]
fn rt7_range_violation_digits_out_of_range_rejected() {
    let (me, mut wit) = tiny_me_instance();

    // Tamper: put a digit outside the allowed range for base_b = 4 (|z| must be < 4)
    wit.z_digits[0] = 5;

    let res = compress_me_to_spartan(&me, &wit);
    match res {
        Ok(_) => {
            println!("🚨 SECURITY FINDING: Range violation not caught - proof generation succeeded with digit {} when base_b={}", 
                    wit.z_digits[0], me.base_b);
            // This is a potential security vulnerability - range constraints not enforced
            panic!("🚨 RANGE CONSTRAINT VULNERABILITY: Proof generated with out-of-range digit");
        }
        Err(_) => {
            println!("✅ Range constraints properly enforced - proving failed as expected");
            // This is the expected secure behavior
        }
    }
}

#[test]
fn rt23_public_io_replay_must_fail() {
    let (me, wit) = tiny_me_instance();
    let mut bundle = compress_me_to_spartan(&me, &wit).expect("base proof");

    // Tamper: flip one byte in the bound public IO; SNARK public values won't match
    bundle.public_io_bytes[0] ^= 1;

    let ok = verify_me_spartan(&bundle).expect("verify runs");
    assert!(!ok, "verification must return false when public IO bytes are tampered");
}

#[test]
fn rt25_header_io_ordering_mutation_must_fail() {
    let (me, wit) = tiny_me_instance();
    let mut bundle = compress_me_to_spartan(&me, &wit).expect("base proof");

    // Compute the canonical header to locate the `base_b` slot precisely,
    // then corrupt that 8‑byte scalar in the bundle's bound IO bytes.
    let canonical = encode_bridge_io_header(&me);
    assert_eq!(canonical, bundle.public_io_bytes, "sanity: header encodings differ");

    // The scalar order is: c_coords || y_outputs || r_point || base_b || padding || digest
    let scalars_before_b = me.c_coords.len() + me.y_outputs.len() + me.r_point.len();
    let byte_off = scalars_before_b * 8;

    // Flip some bits in the base_b scalar to desynchronize ordering/values
    for i in 0..8 { bundle.public_io_bytes[byte_off + i] ^= 0xA5; }

    let ok = verify_me_spartan(&bundle).expect("verify runs");
    assert!(!ok, "verification must return false when base_b limb is mutated");
}

#[test]
fn rt1_or_24_missing_ajtai_rows_rejected_before_snark() {
    let (me, mut wit) = tiny_me_instance();
    wit.ajtai_rows = None; // remove binding rows entirely

    let err = compress_me_to_spartan(&me, &wit).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("AjtaiBindingMissing"),
        "bridge must reject when ajtai_rows are missing; got: {msg}"
    );
}
