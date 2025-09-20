//! Spartan2 bridge binding: parent vs digit mismatch must fail binding (#22).
//!
//! Target: #22 - Parent vs digit binding mismatch - use a digit's Spartan bundle 
//! while verifier expects the parent → fail on public‑IO binding

#![cfg(feature = "redteam")]
#![allow(deprecated)] // Allow use of legacy bridge types for compatibility testing

use neo_fold::{fold_ccs_instances, verify_folding_proof_with_spartan};
use neo_ccs::{CcsStructure, Mat, SparsePoly, Term, McsInstance, McsWitness};
use neo_params::NeoParams;
use neo_math::{F, ring::D};
use p3_field::PrimeCharacteristicRing;
use neo_ajtai::{setup as ajtai_setup, set_global_pp, commit as ajtai_commit};
use rand::{rngs::StdRng, SeedableRng};
use serial_test::serial;

fn dummy_ccs() -> CcsStructure<F> {
    let matrices = vec![Mat::zero(4, 3, F::ZERO)];
    let terms = vec![Term { coeff: F::ONE, exps: vec![1] }];
    let f = SparsePoly::new(1, terms);
    CcsStructure::new(matrices, f).expect("valid CCS")
}

fn ensure_global_ajtai_pp(m: usize) {
    if neo_ajtai::get_global_pp().is_ok() { return; }
    let mut rng = StdRng::from_seed([11u8; 32]);
    let pp = ajtai_setup(&mut rng, D, 16, m).expect("ajtai setup");
    let _ = set_global_pp(pp);
}

#[test]
#[serial]
fn rt22_digit_bundle_as_parent_must_fail_binding() {
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let ccs = dummy_ccs();
    ensure_global_ajtai_pp(ccs.m);

    // Build 2 identical MCS instances/witnesses (k+1 = 2) following pipeline.rs pattern
    let mut instances = Vec::new();
    let mut witnesses = Vec::new();
    
    for _i in 0..2 {
        let z = vec![F::ZERO; ccs.m]; // witness vector
        let z_cols = neo_ajtai::decomp_b(&z, params.b, D, neo_ajtai::DecompStyle::Balanced);
        let pp = neo_ajtai::get_global_pp().unwrap();
        let c = ajtai_commit(&pp, &z_cols);

        // Convert to row-major matrix for McsWitness
        let mut z_mat = Mat::zero(D, ccs.m, F::ZERO);
        for col in 0..ccs.m { 
            for row in 0..D { 
                z_mat[(row, col)] = z_cols[col * D + row]; 
            } 
        }

        instances.push(McsInstance { c, x: vec![], m_in: 0 });
        witnesses.push(McsWitness { w: z, Z: z_mat });
    }

    // Run folding → obtain DEC digits (terminal ME(b,L)) and proof
    let (digits, digit_wits, proof) = fold_ccs_instances(&params, &ccs, &instances, &witnesses).expect("fold ok");

    // Build a Spartan bundle for *a digit* …
    let legacy_digit = neo_fold::bridge_adapter::modern_to_legacy_instance(&digits[0], &params);
    let mut legacy_wit = neo_fold::bridge_adapter::modern_to_legacy_witness(&digit_wits[0], &params).expect("legacy wit");

    // The bridge insists on Ajtai rows present; fetch rows from PP  
    // Calculate z_len from original witness dimensions like neo::prove does
    let pp = neo_ajtai::get_global_pp().unwrap();
    let z_len = digit_wits[0].Z.rows() * digit_wits[0].Z.cols(); // D * m from original folding
    let rows = neo_ajtai::rows_for_coords(&pp, z_len, legacy_digit.c_coords.len()).expect("rows");
    legacy_wit.ajtai_rows = Some(rows);

    let digit_bundle = neo_spartan_bridge::compress_me_to_spartan(&legacy_digit, &legacy_wit)
        .expect("digit bundle");

    // … but verifier expects the *parent* ME(B,L) bundle inside verify_folding_proof.
    let ok = verify_folding_proof_with_spartan(
        &params, &ccs, &instances, &digits, &proof, &digit_bundle
    ).expect("verify path runs");
    assert!(
        !ok,
        "Bridge binding must fail: digit bundle presented where parent bundle is expected"
    );
}
