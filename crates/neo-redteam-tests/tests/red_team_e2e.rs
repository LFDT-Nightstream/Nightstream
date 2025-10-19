// neo-tests/tests/red_team_e2e.rs
#![cfg(feature = "redteam")]
#![cfg_attr(debug_assertions, allow(unused))]
use neo::{prove_spartan2, verify_spartan2, ProveInput, NeoParams, F};
use neo_ccs::{Mat, r1cs::r1cs_to_ccs};
use p3_field::PrimeCharacteristicRing;

fn tiny_ccs() -> neo_ccs::CcsStructure<F> {
    // CCS with 3 rows (pads to 4, ℓ=2):
    // Row 0: (z0 - z1) * 1 = 0  → forces z0 = z1
    // Row 1-2: 0 * 0 = 0 (padding)
    let a = Mat::from_row_major(3, 2, vec![
        F::ONE, -F::ONE,   // Row 0: z0 - z1
        F::ZERO, F::ZERO,  // Row 1: padding
        F::ZERO, F::ZERO,  // Row 2: padding
    ]);
    let b = Mat::from_row_major(3, 2, vec![
        F::ONE, F::ZERO,   // Row 0: * 1
        F::ZERO, F::ZERO,  // Row 1: padding
        F::ZERO, F::ZERO,  // Row 2: padding
    ]);
    let c = Mat::from_row_major(3, 2, vec![
        F::ZERO, F::ZERO,  // Row 0: = 0
        F::ZERO, F::ZERO,  // Row 1: padding
        F::ZERO, F::ZERO,  // Row 2: padding
    ]);
    r1cs_to_ccs(a, b, c)
}

#[test]
fn e2e_rejects_tampered_proof() {
    let ccs = tiny_ccs();

    // witness: [5, 5], choose z0=z1=5 → satisfies (z0 - z1) = 0 with z0=z1
    let witness = vec![F::from_u64(5), F::from_u64(5)];
    let public_input = vec![]; // no public inputs for this test
    
    // Use minimal parameters suitable for testing
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);

    let prove_input = ProveInput {
        params: &params,
        ccs: &ccs,
        public_input: &public_input,
        witness: &witness,
        output_claims: &[],
        vjs_opt: None,
    };

    // produce proof
    let proof = prove_spartan2(prove_input).expect("prove should succeed");

    // tamper: flip one byte in the proof_bytes (main proof data)
    let mut forged_proof = proof.clone();
    if !forged_proof.proof_bytes.is_empty() { 
        let len = forged_proof.proof_bytes.len();
        forged_proof.proof_bytes[len/2] ^= 1; 
    }

    // Tampered proof should either return false or fail with error
    let verification_result = verify_spartan2(&ccs, &public_input, &forged_proof);
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
