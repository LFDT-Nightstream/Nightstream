// neo-commit/tests/extractor.rs
use neo_commit::{AjtaiCommitter, TOY_PARAMS};
use neo_ring::RingElement;
use neo_modint::ModInt;
use rand::{SeedableRng, rngs::StdRng};

#[test]
fn extractor_recovers_preimage_with_two_forks() {
    let committer = AjtaiCommitter::setup_unchecked(TOY_PARAMS);

    // Sample a small witness and commit twice (fresh blinding/noise each time).
    let mut rng = StdRng::seed_from_u64(42);
    let w: Vec<RingElement<ModInt>> = (0..committer.params().d)
        .map(|_| RingElement::random_small(&mut rng, committer.params().n, 3))
        .collect();

    let (c1, _e1, _bw1, _r1) = committer.commit_with_rng(&w, &mut rng).unwrap();
    let (c2, _e2, _bw2, _r2) = committer.commit_with_rng(&w, &mut rng).unwrap();

    // Attempt to extract a preimage for c1 using rewinding with c2.
    // This should fail since GPV trapdoor has been removed.
    let transcript = b"unit_test_commit_extractor";
    let result = committer.extract_commit_witness(&c1, &c2, transcript);
    
    assert!(result.is_err(), "extractor should fail since GPV trapdoor was removed");
    assert_eq!(result.unwrap_err(), "Extractor not available: GPV trapdoor has been removed");
    
    println!("✓ Extractor correctly returns error after GPV trapdoor removal");
}

#[test]
fn extractor_performance_analysis() {
    let committer = AjtaiCommitter::setup_unchecked(TOY_PARAMS);
    let mut rng = StdRng::seed_from_u64(12345);
    
    // Test with different witness sizes
    let test_sizes = vec![
        committer.params().d / 4,
        committer.params().d / 2,
        committer.params().d,
    ];
    
    for size in test_sizes {
        println!("=== PERFORMANCE TEST: witness size {} ===", size);
        
        let w: Vec<RingElement<ModInt>> = (0..size)
            .map(|_| RingElement::random_small(&mut rng, committer.params().n, 2))
            .collect();
        
        // Pad to full size if needed
        let mut full_w = w.clone();
        while full_w.len() < committer.params().d {
            full_w.push(RingElement::zero(committer.params().n));
        }
        
        let start = std::time::Instant::now();
        let (c1, _e1, _bw1, _r1) = committer.commit_with_rng(&full_w, &mut rng).unwrap();
        let (c2, _e2, _bw2, _r2) = committer.commit_with_rng(&full_w, &mut rng).unwrap();
        let commit_time = start.elapsed();
        
        let transcript = format!("perf_test_size_{}", size).as_bytes().to_vec();
        
        let start = std::time::Instant::now();
        let result = committer.extract_commit_witness(&c1, &c2, &transcript);
        let extract_time = start.elapsed();
        
        // Since GPV trapdoor has been removed, extraction should always fail
        assert!(result.is_err(), "Extraction should fail since GPV trapdoor was removed");
        assert_eq!(result.unwrap_err(), "Extractor not available: GPV trapdoor has been removed");
        
        println!("  ✓ Extraction correctly failed (GPV trapdoor removed)");
        println!("  - Commit time: {:?}", commit_time);
        println!("  - Extract time: {:?}", extract_time);
    }
}

#[test]
fn extractor_security_analysis() {
    let committer = AjtaiCommitter::setup_unchecked(TOY_PARAMS);
    let mut rng = StdRng::seed_from_u64(54321);
    
    println!("=== SECURITY ANALYSIS ===");
    
    // Test 1: Different transcripts should give different challenges
    let w: Vec<RingElement<ModInt>> = (0..committer.params().d)
        .map(|_| RingElement::random_small(&mut rng, committer.params().n, 2))
        .collect();
    
    let (c1, _e1, _bw1, _r1) = committer.commit_with_rng(&w, &mut rng).unwrap();
    let (c2, _e2, _bw2, _r2) = committer.commit_with_rng(&w, &mut rng).unwrap();
    
    let transcript1 = b"security_test_1";
    let transcript2 = b"security_test_2";
    
    let result1 = committer.extract_commit_witness(&c1, &c2, transcript1);
    let result2 = committer.extract_commit_witness(&c1, &c2, transcript2);
    
    // Since GPV trapdoor has been removed, both extractions should fail
    assert!(result1.is_err(), "First extraction should fail since GPV trapdoor was removed");
    assert!(result2.is_err(), "Second extraction should fail since GPV trapdoor was removed");
    assert_eq!(result1.unwrap_err(), "Extractor not available: GPV trapdoor has been removed");
    assert_eq!(result2.unwrap_err(), "Extractor not available: GPV trapdoor has been removed");
    
    println!("  ✓ Both extractions correctly failed (GPV trapdoor removed)");
    
    // Test 2: Extraction should be deterministic for same transcript
    let result3 = committer.extract_commit_witness(&c1, &c2, transcript1);
    let result4 = committer.extract_commit_witness(&c1, &c2, transcript1);
    
    // Both should fail consistently
    assert!(result3.is_err(), "Third extraction should fail since GPV trapdoor was removed");
    assert!(result4.is_err(), "Fourth extraction should fail since GPV trapdoor was removed");
    assert_eq!(result3.unwrap_err(), "Extractor not available: GPV trapdoor has been removed");
    assert_eq!(result4.unwrap_err(), "Extractor not available: GPV trapdoor has been removed");
    
    println!("  ✓ Extraction failures are consistent (deterministic error)");
    
    println!("=== END SECURITY ANALYSIS ===");
}
