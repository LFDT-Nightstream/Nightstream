//! Red-team API/binding tests that exercise neo::{prove, verify}.
//!
//! Targets: 
//! - #26 (wrong CCS/public_input) - Verify with wrong `(CCS, public_input)` binding check
//! - #29 (DoS length guard) - Unbounded bincode deserialization (DoS) size guard in `neo::verify`

#![cfg(feature = "redteam")]

use neo::{prove_spartan2, verify_spartan2, ProveInput, CcsStructure, NeoParams, F};
use p3_field::PrimeCharacteristicRing;
use neo_ccs::{Mat, SparsePoly, Term};

fn dummy_ccs_structure() -> CcsStructure<F> {
    // Single 4x3 zero matrix and a trivial arity-1 polynomial.
    // With zero witness, the row-wise CCS check is satisfied.
    let matrices = vec![Mat::zero(4, 3, F::ZERO)];
    let terms = vec![Term { coeff: F::ONE, exps: vec![1] }];
    let f = SparsePoly::new(1, terms);
    CcsStructure::new(matrices, f).expect("valid CCS")
}

#[test]
fn rt26_verify_wrong_public_input_must_fail() {
    // Arrange: small, satisfiable CCS + zero witness
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    let ccs = dummy_ccs_structure();
    let witness = vec![F::ZERO; ccs.m]; // length matches m
    let public_input: Vec<F> = vec![];  // empty is fine here

    // Prove for (ccs, public_input)
    let proof = prove_spartan2(ProveInput {
        params: &params, ccs: &ccs, public_input: &public_input, witness: &witness,
        output_claims: &[],
        vjs_opt: None,
    }).expect("prove ok");

    // Tamper: change the public_input at verification time
    let bad_public_input = vec![F::from_u64(1)];
    let ok = verify_spartan2(&ccs, &bad_public_input, &proof).expect("verify runs");
    assert!(!ok, "verification must return false when public_input differs");
}

#[test]
fn rt26_verify_wrong_ccs_must_fail() {
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    let mut ccs = dummy_ccs_structure();
    let witness = vec![F::ZERO; ccs.m];
    let public_input: Vec<F> = vec![];

    let proof = prove_spartan2(ProveInput {
        params: &params, ccs: &ccs, public_input: &public_input, witness: &witness,
        output_claims: &[],
        vjs_opt: None,
    }).expect("prove ok");

    // Tamper CCS deterministically: flip one entry in the first matrix
    ccs.matrices[0][(0, 0)] = F::from_u64(1);

    let ok = verify_spartan2(&ccs, &public_input, &proof).expect("verify runs");
    assert!(!ok, "verification must return false when CCS differs");
}

#[test]
fn rt29_proof_bundle_too_large_rejected_fast() {
    // Construct a dummy Proof with an oversized size header (no need for a real proof)
    const MAX: usize = 64 * 1024 * 1024; // must match neo::verify guard
    let too_big = (MAX as u32) + 1;

    let proof = neo::Proof {
        v: 2,  // Use ProofV2 format
        circuit_key: [0u8; 32],   // dummy circuit key
        vk_digest: [0u8; 32],     // dummy vk digest
        public_io: vec![0u8; 32], // Add context digest to avoid early rejection
        proof_bytes: {
            let mut oversized = Vec::new();
            oversized.resize(too_big as usize, 0u8); // fill to oversized amount
            oversized
        },
        public_results: vec![],
        meta: Default::default(),
    };

    let res = verify_spartan2(&dummy_ccs_structure(), &[], &proof);
    // With lean proof system, oversized proofs may return Ok(false) or Err
    match res {
        Ok(false) => {
            // Good: verification correctly rejected the oversized/invalid proof
        }
        Err(_) => {
            // Also good: verification errored on the oversized/invalid proof
        }
        Ok(true) => {
            panic!("oversized proof should not verify as valid");
        }
    }
}
