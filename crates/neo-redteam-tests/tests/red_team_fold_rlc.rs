//! Π_RLC red-team: non-invertible ρ differences (duplicate ρ) must fail verification.
//!
//! Target: #13 - Non-invertible ρ differences should cause RLC verification to fail

#![cfg(feature = "redteam")]

use neo_fold::{fold_ccs_instances, verify_folding_proof_with_spartan};
use neo_ccs::{CcsStructure, Mat, SparsePoly, Term, McsInstance, McsWitness};
use neo_params::NeoParams;
use neo_math::{F, ring::D};
use p3_field::PrimeCharacteristicRing;
use neo_spartan_bridge::ProofBundle;
use neo_ajtai::{setup as ajtai_setup, set_global_pp, commit as ajtai_commit};
use rand::{rngs::StdRng, SeedableRng};
use serial_test::serial;

fn dummy_ccs() -> CcsStructure<F> {
    // Single 4x3 zero matrix, trivial arity-1 poly => easy satisfiable CCS.
    let matrices = vec![Mat::zero(4, 3, F::ZERO)];
    let terms = vec![Term { coeff: F::ONE, exps: vec![1] }];
    let f = SparsePoly::new(1, terms);
    CcsStructure::new(matrices, f).expect("valid CCS")
}

fn ensure_global_ajtai_pp(m: usize) {
    if neo_ajtai::get_global_pp().is_ok() { return; }
    let mut rng = StdRng::from_seed([7u8; 32]);
    // κ=16 matches folding usage; m is CCS witness length
    let pp = ajtai_setup(&mut rng, D, 16, m).expect("ajtai setup");
    let _ = set_global_pp(pp);
}

/// Create a dummy ProofBundle for testing purposes
/// SECURITY NOTE: This is only for testing - real verification requires actual Spartan2 proofs
fn dummy_proof_bundle() -> ProofBundle {
    ProofBundle::new_with_vk(
        vec![0u8; 32],  // dummy proof
        vec![0u8; 32],  // dummy verifier key  
        vec![0u8; 16],  // dummy public IO
    )
}

#[test]
#[serial]
fn rt13_non_invertible_rhos_fail() {
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    let ccs = dummy_ccs();
    ensure_global_ajtai_pp(ccs.m);

    // Build 2 identical MCS instances/witnesses (k+1 = 2) following pipeline.rs pattern
    let mut inputs = Vec::new();
    let mut witnesses = Vec::new();
    
    for _ in 0..2 {
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

        inputs.push(McsInstance { c, x: vec![], m_in: 0 });
        witnesses.push(McsWitness { w: z, Z: z_mat });
    }

    // Produce digits and proof via full folding pipeline
    let (digits, _digit_wits, mut fold_proof) = fold_ccs_instances(
        &params, &ccs, &inputs, &witnesses
    ).expect("folding prove ok");

    // TAMPER: duplicate ρ values → pairwise differences non-invertible
    // (ρ_elems is public in the proof; duplicate first element)
    assert!(fold_proof.pi_rlc_proof.rho_elems.len() >= 2);
    let dup = fold_proof.pi_rlc_proof.rho_elems[0];
    fold_proof.pi_rlc_proof.rho_elems[1] = dup;

    // We don't need a real Spartan bundle because verification should fail in Π_RLC first.
    let dummy_bundle = dummy_proof_bundle();

    // Verify end-to-end: should fail at Π_RLC step with error
    let result = verify_folding_proof_with_spartan(
        &params, &ccs, &inputs, &digits, &fold_proof, &dummy_bundle
    );
    assert!(result.is_err(), "Π_RLC must reject duplicate/non-invertible ρ differences");
}
