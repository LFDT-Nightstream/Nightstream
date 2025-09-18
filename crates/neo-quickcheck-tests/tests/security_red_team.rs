#![cfg(feature = "quickcheck")]
//! Security guard rails that should fail closed.

#![allow(deprecated)]

use neo_spartan_bridge::compress_me_to_spartan;
#[allow(deprecated)]
use neo_ccs::{MEInstance, MEWitness};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

fn dot_f_z(row: &[F], z: &[i64]) -> F {
    row.iter().zip(z.iter()).fold(F::ZERO, |acc, (a, &zi)| {
        let zi_f = if zi >= 0 { 
            F::from_u64(zi as u64) 
        } else { 
            -F::from_u64((-zi) as u64) 
        };
        acc + (*a) * zi_f
    })
}

// A tiny consistent instance (mirrors your bridge_smoke.rs pattern).
fn tiny_me_and_witness() -> (MEInstance, MEWitness) {
    let z = vec![1i64, 2, 3, 0, -1, 1, 0, 2];

    let ajtai_rows = vec![
        vec![F::ONE, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO],
        vec![F::ZERO, F::ONE, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO],
    ];

    let c0 = dot_f_z(&ajtai_rows[0], &z);
    let c1 = dot_f_z(&ajtai_rows[1], &z);
    let c_coords = vec![c0, c1, F::ZERO, F::ZERO];

    let w0 = vec![F::ONE, F::ONE, F::ONE, F::ONE, F::ZERO, F::ZERO, F::ZERO, F::ZERO];
    let w1 = vec![F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ONE, F::ZERO, F::ONE];

    let me = MEInstance {
        c_step_coords: vec![], // TODO: Pattern A - populate with actual pre-commit coords
        u_offset: 0,
        u_len: 0,
        c_coords,
        y_outputs: vec![dot_f_z(&w0, &z), dot_f_z(&w1, &z), F::ZERO, F::ZERO],
        r_point: vec![F::from_u64(3); 2],
        base_b: 4,
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
fn bridge_rejects_missing_ajtai_rows() {
    let (me, mut wit) = tiny_me_and_witness();
    wit.ajtai_rows = None; // remove binding rows
    let err = compress_me_to_spartan(&me, &wit)
        .expect_err("Ajtai rows missing must fail");
    let msg = format!("{err:?}");
    assert!(msg.contains("AjtaiBindingMissing"), "error: {msg}");
}

#[test]
fn bridge_rejects_empty_ajtai_rows() {
    let (me, mut wit) = tiny_me_and_witness();
    wit.ajtai_rows = Some(vec![]); // empty binding rows
    let err = compress_me_to_spartan(&me, &wit)
        .expect_err("Empty Ajtai rows must fail");
    let msg = format!("{err:?}");
    assert!(msg.contains("AjtaiBindingMissing"), "error: {msg}");
}
