use neo_ccs::{mv_poly, CcsInstance, CcsStructure, CcsWitness};
use neo_commit::{AjtaiCommitter, TOY_PARAMS};
use neo_fields::{ExtF, F};
use neo_fold::FoldState;
use p3_matrix::dense::RowMajorMatrix;
use p3_field::PrimeCharacteristicRing;



/// Test transcript layout roundtrip - ensures the proof generation writes
/// data in the order that verification expects to read it
#[test]
fn test_transcript_layout_roundtrip() {
    let structure = CcsStructure::new(
        vec![RowMajorMatrix::<F>::new(vec![F::ONE], 1)],
        mv_poly(|vars: &[ExtF]| vars[0], 1)
    );
    
    let mut state = FoldState::new(structure);
    let committer = AjtaiCommitter::setup_unchecked(TOY_PARAMS);
    
    // Create valid instances
    let instance = CcsInstance {
        commitment: vec![],
        public_input: vec![],
        u: F::ONE,
        e: F::ONE,
    };
    let witness = CcsWitness {
        z: vec![ExtF::ONE],
    };
    
    // Generate proof
    let proof = state.generate_proof(
        (instance.clone(), witness.clone()),
        (instance, witness),
        &committer
    );
    
    // The key test: verification should work correctly
    assert!(state.verify(&proof.transcript, &committer),
            "Generated proof should verify correctly");
    
    // Basic sanity checks on transcript structure
    assert!(proof.transcript.len() > 100, "Transcript should be substantial");
    assert!(proof.transcript.len() < 10000, "Transcript should not be excessive");
    
    // Check that it starts with expected tag (now starts with preview for joint FS)
    let (prefix, _hash) = proof.transcript.split_at(proof.transcript.len() - 32);
    assert!(
        prefix.starts_with(b"neo_pi_ccs2_preview"),
        "Transcript didn't start with 'neo_pi_ccs2_preview'. First 24 bytes: {:?}",
        &prefix[..prefix.len().min(24)]
    );
    
    // Stronger layout check: verify the preview matches the real CCS2 commit
    let n = committer.params().n;
    let mut i = 0;

    // 1) neo_pi_ccs2_preview
    let t0 = b"neo_pi_ccs2_preview";
    assert_eq!(&prefix[i..i + t0.len()], t0);
    i += t0.len();

    //   serialized commit2 preview (len byte + len * n * 8 bytes)
    let len_prev = prefix[i] as usize;
    let prev_block_start = i;
    i += 1 + len_prev * n * 8;
    let prev_block = &prefix[prev_block_start..i];

    // 2) neo_pi_ccs1 then its commit (skip over it)
    let t1 = b"neo_pi_ccs1";
    assert_eq!(&prefix[i..i + t1.len()], t1);
    i += t1.len();
    let len_c1 = prefix[i] as usize;
    i += 1 + len_c1 * n * 8;

    // 3) find the exact "neo_pi_ccs2" tag (not the earlier "..._preview")
    let tag2 = b"neo_pi_ccs2";
    let pos2 = prefix[i..]
        .windows(tag2.len())
        .enumerate()
        .find(|(off, w)| {
            *w == tag2 && prefix[i + off + tag2.len()] != b'_' // exclude "..._preview"
        })
        .map(|(off, _)| i + off)
        .expect("neo_pi_ccs2 tag not found");
    i = pos2 + tag2.len();

    //   serialized real commit2
    let len_c2 = prefix[i] as usize;
    let c2_block_start = i;
    i += 1 + len_c2 * n * 8;
    let c2_block = &prefix[c2_block_start..i];

    // 4) the preview must equal the real commit2 (byte-for-byte)
    assert_eq!(
        prev_block, c2_block,
        "commit2 preview did not match the real commit2"
    );
    
    println!("Transcript layout validation passed! Length: {}", proof.transcript.len());
    println!("✅ Preview commit matches real commit2 - joint FS integrity verified");
}

/// Test zero polynomial full flow - ensures zero polynomial cases
/// are handled correctly throughout the entire pipeline
#[test]
fn test_zero_poly_full_flow() {
    let structure = CcsStructure::new(
        vec![RowMajorMatrix::<F>::new(vec![F::ZERO], 1)], // Zero matrix
        mv_poly(|_: &[ExtF]| ExtF::ZERO, 1) // Zero polynomial
    );
    
    let mut state = FoldState::new(structure);
    let committer = AjtaiCommitter::setup_unchecked(TOY_PARAMS);
    
    // Zero instances
    let instance = CcsInstance {
        commitment: vec![],
        public_input: vec![],
        u: F::ZERO,
        e: F::ZERO,
    };
    let witness = CcsWitness {
        z: vec![ExtF::ZERO],
    };
    
    // Generate and verify proof
    let proof = state.generate_proof(
        (instance.clone(), witness.clone()),
        (instance, witness),
        &committer
    );
    
    assert!(state.verify(&proof.transcript, &committer),
            "Zero polynomial full flow should verify correctly");
}

/// Test transcript consistency - ensures that multiple proof generations
/// with the same inputs produce transcripts with consistent structure
#[test]
fn test_transcript_consistency() {
    let structure = CcsStructure::new(
        vec![RowMajorMatrix::<F>::new(vec![F::ONE], 1)],  // Fixed: 1x1 matrix
        mv_poly(|vars: &[ExtF]| vars[0], 1)  // Fixed: 1 variable
    );
    
    let instance = CcsInstance {
        commitment: vec![],
        public_input: vec![],
        u: F::ONE,
        e: F::ONE,
    };
    let witness = CcsWitness {
        z: vec![ExtF::ONE],  // Fixed: 1 element
    };
    
    // Generate multiple proofs
    let mut proofs = Vec::new();
    for _ in 0..3 {
        let mut state = FoldState::new(structure.clone());
        let committer = AjtaiCommitter::setup_unchecked(TOY_PARAMS);
        let proof = state.generate_proof(
            (instance.clone(), witness.clone()),
            (instance.clone(), witness.clone()),
            &committer
        );
        proofs.push((proof, state));
    }
    
    // All proofs should verify
    for (proof, state) in &proofs {
        let committer = AjtaiCommitter::setup_unchecked(TOY_PARAMS);
        assert!(state.verify(&proof.transcript, &committer),
                "Each proof should verify correctly");
    }
    
    // Transcript lengths should be consistent (may vary due to randomness but structure should be same)
    let lengths: Vec<usize> = proofs.iter().map(|(p, _)| p.transcript.len()).collect();
    println!("Transcript lengths: {:?}", lengths);
    
    // All lengths should be the same for deterministic parts
    // (Note: Some randomness may cause slight variations, but major structure should be consistent)
    let min_len = *lengths.iter().min().unwrap();
    let max_len = *lengths.iter().max().unwrap();
    assert!(max_len - min_len < 200, "Transcript lengths should be reasonably consistent");
}

/// Test edge cases for transcript parsing
#[test]
fn test_transcript_edge_cases() {
    let structure = CcsStructure::new(
        vec![RowMajorMatrix::<F>::new(vec![F::ONE], 1)],
        mv_poly(|vars: &[ExtF]| vars[0], 1)
    );
    
    let mut state = FoldState::new(structure);
    let committer = AjtaiCommitter::setup_unchecked(TOY_PARAMS);
    
    let instance = CcsInstance {
        commitment: vec![],
        public_input: vec![],
        u: F::ONE,
        e: F::ONE,
    };
    let witness = CcsWitness {
        z: vec![ExtF::ONE],
    };
    
    let proof = state.generate_proof(
        (instance.clone(), witness.clone()),
        (instance, witness),
        &committer
    );
    
    // Test that verification fails gracefully with corrupted transcripts
    
    // 1. Too short transcript
    assert!(!state.verify(&[1, 2, 3], &committer), "Should reject too short transcript");
    
    // 2. Wrong hash
    let mut corrupted = proof.transcript.clone();
    let len = corrupted.len();
    corrupted[len - 1] ^= 1; // Flip one bit in hash
    assert!(!state.verify(&corrupted, &committer), "Should reject wrong hash");
    
    // 3. Truncated transcript
    let truncated = if proof.transcript.len() > 100 {
        &proof.transcript[..proof.transcript.len() - 100]
    } else {
        &proof.transcript[..proof.transcript.len().saturating_sub(32).max(1)]
    };
    assert!(!state.verify(truncated, &committer), "Should reject truncated transcript");
    
    println!("Edge case handling works correctly");
}
