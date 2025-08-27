// neo-orchestrator/tests/spartan2_integration_test.rs
use neo_orchestrator::neutronnova_integration::NeutronNovaFoldState;
use neo_ccs::{CcsInstance, CcsStructure, CcsWitness, mv_poly};
use neo_fields::{from_base, ExtF, F};
use neo_commit::AjtaiCommitter;
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;

#[test]
fn test_neutronnova_integration() {
    println!("🧪 Testing NeutronNova SNARK integration");
    
    // Create a simple CCS: a + b = c
    let mats = vec![
        RowMajorMatrix::new(vec![F::ONE, F::ZERO, F::ZERO], 3),   // Matrix A: selects a
        RowMajorMatrix::new(vec![F::ZERO, F::ONE, F::ZERO], 3),   // Matrix B: selects b  
        RowMajorMatrix::new(vec![F::ZERO, F::ZERO, F::ONE], 3),   // Matrix C: selects c
    ];

    let f = mv_poly(|inputs: &[ExtF]| {
        if inputs.len() != 3 {
            ExtF::ZERO
        } else {
            inputs[0] + inputs[1] - inputs[2] // a + b - c = 0
        }
    }, 1);

    let structure = CcsStructure::new(mats, f);
    let mut fold_state = NeutronNovaFoldState::new(structure);

    // Create test instances and witnesses
    let instance1 = CcsInstance {
        commitment: vec![],
        public_input: vec![],
        u: F::ZERO,
        e: F::ONE,
    };
    
    let witness1 = CcsWitness {
        z: vec![
            from_base(F::from_u64(2)), // a = 2
            from_base(F::from_u64(3)), // b = 3
            from_base(F::from_u64(5)), // c = 5
        ],
    };

    let instance2 = instance1.clone();
    let witness2 = witness1.clone();

    let committer = AjtaiCommitter::new();
    
    // Generate proof (should use SNARK mode when feature is enabled)
    let proof = fold_state.generate_proof_snark(
        (instance1, witness1),
        (instance2, witness2),
        &committer
    );
    
    // Verify proof
    let is_valid = fold_state.verify_snark(&proof.transcript, &committer);
    assert!(is_valid, "Proof should be valid in SNARK mode");
    
    // Check that proof contains SNARK marker
    let transcript_str = String::from_utf8_lossy(&proof.transcript);
    assert!(transcript_str.contains("neo_spartan2_snark"), "Proof should contain SNARK marker");
    
    println!("✅ NeutronNova SNARK mode integration test passed");
    println!("   Proof size: {} bytes", proof.transcript.len());
    println!("   Contains SNARK marker: ✓");
}
