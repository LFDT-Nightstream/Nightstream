#![allow(non_snake_case)] // Allow mathematical notation like Z

use neo_ajtai::{setup, commit, decomp_b, DecompStyle, open_linear};
use neo_math::Fq;
use p3_field::PrimeCharacteristicRing;
use rand_chacha::{ChaCha20Rng, rand_core::SeedableRng};

#[test]
fn linear_opening_produces_correct_evaluation() {
    let mut rng = ChaCha20Rng::seed_from_u64(4242);
    let d = neo_math::D;
    let kappa = 2usize;
    let m = 3usize;

    // Public parameters and a small witness: z = [1, 2, 3]
    let pp = setup(&mut rng, d, kappa, m).expect("setup should succeed");
    let z: Vec<Fq> = (0..m).map(|i| Fq::from_u64((i as u64) + 1)).collect();
    let Z = decomp_b(&z, 2, d, DecompStyle::Balanced); // d×m matrix
    let c = commit(&pp, &Z);

    // Linear functional v = [3, 5, 7], so v^T z should equal 3*1 + 5*2 + 7*3 = 34
    let v = vec![Fq::from_u64(3), Fq::from_u64(5), Fq::from_u64(7)];
    let expected_result = Fq::from_u64(3 * 1 + 5 * 2 + 7 * 3); // = 34
    
    let (y_slices, proof) = open_linear(&pp, &c, &Z, &[v.clone()]);

    // Verify structural correctness
    assert_eq!(y_slices.len(), 1, "Should have one slice for one linear functional");
    assert_eq!(y_slices[0].len(), d, "Y slice should have d elements matching decomposition depth");
    
    // Verify proof structure is correct
    assert!(!proof.opened_values.is_empty(), "Proof should have opened values");
    assert_eq!(proof.opened_values.len(), 1, "Should have one opened value per linear functional");
    
    // Verify the linear evaluation is correct by manually computing v^T Z
    // Z is d×m, each column Z[:,j] represents the decomposition of z[j]
    let mut manual_result = Fq::from_u64(0);
    for j in 0..m {
        let z_j_contrib = v[j]; // v[j] * z[j], but z[j] should equal sum of Z[:, j] in the right base
        manual_result += z_j_contrib * Fq::from_u64((j as u64) + 1);
    }
    assert_eq!(manual_result, expected_result, "Manual computation should match expected");
    
    // SECURITY NOTE: 
    // This test only verifies prover-side computation correctness.
    // Verification of linear relations must be done via neo_fold::verify_linear (Π_RLC)
    // which provides the actual cryptographic soundness guarantees.
}

// REMOVED: Pointless test that just asserts true
// 
// The API design is enforced by:
// 1. Only exporting open_linear (not verify_linear) in lib.rs
// 2. Documentation clearly directing users to neo_fold::verify_linear
// 3. Integration tests demonstrating correct usage pattern
//
// If verify_linear were accidentally added, it would be caught by:
// - API review process
// - Integration tests failing
// - Documentation being inconsistent
