// neo-commit/tests/extractor.rs
use neo_commit::{AjtaiCommitter, TOY_PARAMS};
use neo_ring::RingElement;
use neo_modint::{ModInt, Coeff};
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

    // Extract a preimage for c1 using rewinding with c2.
    let transcript = b"unit_test_commit_extractor";
    let w_ex = committer.extract_commit_witness(&c1, &c2, transcript)
        .expect("extractor must succeed");

    // Must satisfy A*w_ex + 0 = c1 mod q, with norms within bounds (0 noise is allowed).
    let zeros = vec![RingElement::from_scalar(ModInt::zero(), committer.params().n); committer.params().k];
    assert!(committer.verify(&c1, &w_ex, &zeros), "extracted preimage must verify");
    
    println!("✓ Extractor test passed - recovered valid preimage for c1");
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
        
        match result {
            Ok(extracted) => {
                let norms: Vec<_> = extracted.iter().map(|w| w.norm_inf()).collect();
                println!("  ✓ Extraction succeeded");
                println!("  - Commit time: {:?}", commit_time);
                println!("  - Extract time: {:?}", extract_time);
                println!("  - Max extracted norm: {}", norms.iter().max().unwrap_or(&0));
                let avg_norm = norms.iter().map(|&n| n as f64).sum::<f64>() / norms.len() as f64;
                println!("  - Avg extracted norm: {:.2}", avg_norm);
            },
            Err(e) => {
                println!("  ✗ Extraction failed: {}", e);
                panic!("Extraction should succeed for honest commitments");
            }
        }
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
    
    match (result1, result2) {
        (Ok(w1), Ok(w2)) => {
            // Different transcripts should generally produce different extractions
            // (though they should both be valid preimages)
            let w1_norms: Vec<_> = w1.iter().map(|w| w.norm_inf()).collect();
            let w2_norms: Vec<_> = w2.iter().map(|w| w.norm_inf()).collect();
            
            println!("  ✓ Both extractions succeeded");
            println!("  - Extraction 1 max norm: {}", w1_norms.iter().max().unwrap_or(&0));
            println!("  - Extraction 2 max norm: {}", w2_norms.iter().max().unwrap_or(&0));
            
            // Both should be valid preimages
            let zero_noise = vec![RingElement::from_scalar(ModInt::zero(), committer.params().n); committer.params().k];
            assert!(committer.verify(&c1, &w1, &zero_noise), "First extraction should be valid");
            assert!(committer.verify(&c1, &w2, &zero_noise), "Second extraction should be valid");
            
            println!("  ✓ Both extractions are valid preimages");
        },
        _ => {
            panic!("Both extractions should succeed for honest commitments");
        }
    }
    
    // Test 2: Extraction should be deterministic for same transcript
    let result3 = committer.extract_commit_witness(&c1, &c2, transcript1);
    let result4 = committer.extract_commit_witness(&c1, &c2, transcript1);
    
    match (result3, result4) {
        (Ok(w3), Ok(w4)) => {
            // Same transcript should produce identical results (deterministic)
            assert_eq!(w3.len(), w4.len(), "Extracted witnesses should have same length");
            for (i, (a, b)) in w3.iter().zip(w4.iter()).enumerate() {
                assert_eq!(a, b, "Extracted witness element {} should be identical", i);
            }
            println!("  ✓ Extraction is deterministic for same transcript");
        },
        _ => {
            panic!("Deterministic extractions should both succeed");
        }
    }
    
    println!("=== END SECURITY ANALYSIS ===");
}
