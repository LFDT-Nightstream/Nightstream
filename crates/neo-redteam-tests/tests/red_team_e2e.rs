// neo-tests/tests/red_team_e2e.rs
#![cfg(feature = "redteam")]
#![cfg_attr(debug_assertions, allow(unused))]
use neo::{prove, verify, ProveInput, NeoParams, F};
use neo_ccs::{Mat, r1cs::r1cs_to_ccs};
use p3_field::PrimeCharacteristicRing;

fn tiny_ccs() -> neo_ccs::CcsStructure<F> {
    // 1-row: (z0 - z1) * 1 = 0  → forces z0 = z1
    let a = Mat::from_row_major(1, 2, vec![F::ONE, -F::ONE]);
    let b = Mat::from_row_major(1, 2, vec![F::ONE, F::ZERO]);
    let c = Mat::from_row_major(1, 2, vec![F::ZERO, F::ZERO]);
    r1cs_to_ccs(a, b, c)
}

#[test]
fn e2e_rejects_tampered_proof() {
    let ccs = tiny_ccs();

    // witness: [5, 5], choose z0=z1=5 → satisfies (z0 - z1) = 0 with z0=z1
    let witness = vec![F::from_u64(5), F::from_u64(5)];
    let public_input = vec![]; // no public inputs for this test
    
    // Use minimal parameters suitable for testing
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);

    let prove_input = ProveInput {
        params: &params,
        ccs: &ccs,
        public_input: &public_input,
        witness: &witness,
    };

    // produce proof
    let proof = prove(prove_input).expect("prove should succeed");

    // tamper: flip one byte in the proof_bytes (main proof data)
    let mut forged_proof = proof.clone();
    if !forged_proof.proof_bytes.is_empty() { 
        let len = forged_proof.proof_bytes.len();
        forged_proof.proof_bytes[len/2] ^= 1; 
    }

    // Tampered proof should either return false or fail with error
    let verification_result = verify(&ccs, &public_input, &forged_proof);
    match verification_result {
        Ok(false) => {
            // Good: verification correctly rejected the tampered proof
        }
        Err(_) => {
            // Also good: tampered proof caused verification to fail with error
        }
        Ok(true) => {
            panic!("tampered proof was incorrectly accepted as valid");
        }
    }
}
