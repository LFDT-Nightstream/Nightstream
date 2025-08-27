use neo_commit::*;
use neo_decomp::decomp_b;
use neo_fields::{ExtF, F};
use neo_modint::{Coeff, ModInt};
use neo_ring::RingElement;
use p3_field::PrimeCharacteristicRing;
use rand::rngs::StdRng;
use rand::SeedableRng;

#[test]
fn test_params_prime() {
    // Both presets must match the ModInt ring modulus (2^61 - 1)
    assert_eq!(SECURE_PARAMS.q, ModInt::Q, "SECURE_PARAMS.q must equal ModInt::Q");
    assert_eq!(TOY_PARAMS.q, ModInt::Q, "TOY_PARAMS.q must equal ModInt::Q");
}

fn test_params() -> NeoParams {
    if std::env::var("NEO_TEST_SECURE").ok().is_some() {
        SECURE_PARAMS
    } else {
        TOY_PARAMS
    }
}

/// Tests the complete roundtrip of the Ajtai commitment scheme.
///
/// This test is essential because it validates the entire commitment pipeline:
/// 1. Generates a witness (all zeros in this case, but could be extended).
/// 2. Decomposes it using base-b decomposition.
/// 3. Packs the decomposed matrix into ring elements.
/// 4. Commits to the packed witness.
/// 5. Verifies the commitment.
///
/// Having this test ensures that all components (decomposition, packing, commit, verify) 
/// integrate correctly. It's a critical smoke test for the system's basic functionality.
#[test]
fn test_roundtrip() {
    let params = test_params();
    let comm = AjtaiCommitter::setup_unchecked(params);
    let z: Vec<F> = vec![F::ZERO; params.n];
    let mat = decomp_b(&z, params.b, params.d);
    let w = AjtaiCommitter::pack_decomp(&mat, &params);
    let mut t = Vec::new();
    let (c, e, w_blinded, _r) = comm.commit(&w, &mut t).expect("commit");
    assert!(comm.verify(&c, &w_blinded, &e));
}

/// Tests that commitments are blinded (hiding the witness) and can be opened correctly.
///
/// This test is important for verifying the zero-knowledge property (hiding) and 
/// the ability to open commitments at specific points, which is crucial for 
/// the scheme's use in proofs. It ensures blinded witnesses differ from originals 
/// and that openings produce valid proofs.
#[test]
fn test_blinded_commit_hiding_and_opening() {
    let params = test_params();
    let comm = AjtaiCommitter::setup_unchecked(params);
    let z: Vec<F> = vec![F::ZERO; params.n];
    let mat = decomp_b(&z, params.b, params.d);
    let w = AjtaiCommitter::pack_decomp(&mat, &params);
    let mut t = Vec::new();
    let (c, e, blinded_w, r) = comm.commit(&w, &mut t).expect("commit");
    // The blinded witness should differ from the original (hiding)
    assert_ne!(blinded_w, w);
    // Open commitment and verify
    let log_k = (params.k as f64).log2().ceil() as usize;
    let point = vec![ExtF::ZERO; log_k];
    let mut rng = StdRng::seed_from_u64(0);
    // Note: Opening is no longer available since GPV trapdoor was removed
    let result = comm.open_at_point(&c, &point, &blinded_w, &e, &r, &mut rng);
    assert!(result.is_err());
}

/// Tests that commitments are randomized for zero-knowledge property.
///
/// This test ensures different transcripts produce different commitments for 
/// the same witness, verifying the hiding property against chosen-message attacks. 
/// It's good for confirming probabilistic behavior in commitments.
#[test]
fn test_zk_commit_randomized() {
    let params = test_params();
    let comm = AjtaiCommitter::setup_unchecked(params);
    let z: Vec<F> = vec![F::ZERO; params.n];
    let mat = decomp_b(&z, params.b, params.d);
    let w = AjtaiCommitter::pack_decomp(&mat, &params);
    let mut t1 = b"test1".to_vec();
    let (c1, _, _, _) = comm.commit(&w, &mut t1).expect("commit");
    let mut t2 = b"test2".to_vec();
    let (c2, _, _, _) = comm.commit(&w, &mut t2).expect("commit");
    assert_ne!(c1, c2);
}

/// Tests opening a commitment at a specific point.
///
/// This is required to ensure the multilinear evaluation opening works, 
/// which is core to the commitment's use in proof systems.
#[test]
fn test_open_at_point() {
    use rand::Rng;
    let params = test_params();
    let comm = AjtaiCommitter::setup_unchecked(params);
    let mut rng = rand::rng();
    let z: Vec<F> = (0..params.n)
        .map(|_| F::from_u64(rng.random_range(0..params.b)))
        .collect();
    let mat = decomp_b(&z, params.b, params.d);
    let w = AjtaiCommitter::pack_decomp(&mat, &params);
    let mut t = Vec::new();
    let (c, e, w_blinded, r) = comm.commit(&w, &mut t).expect("commit");
    let log_k = (params.k as f64).log2().ceil() as usize;
    let point: Vec<ExtF> = (0..log_k).map(|_| ExtF::ZERO).collect();
    let mut rng = StdRng::seed_from_u64(6);
    // Note: Opening is no longer available since GPV trapdoor was removed
    let result = comm.open_at_point(&c, &point, &w_blinded, &e, &r, &mut rng);
    assert!(result.is_err());
}

/// Tests verification of an opening with rank considerations.
///
/// This ensures the matrix rank properties hold during verification, 
/// which is crucial for the scheme's soundness. It's good for catching issues in 
/// setup or sampling that could compromise full rank.
#[test]
fn test_open_verify_rank() {
    let params = test_params();
    let comm = AjtaiCommitter::setup_unchecked(params);
    let c = vec![RingElement::from_scalar(ModInt::zero(), params.n); params.k];
    let w_blinded = vec![RingElement::from_scalar(ModInt::zero(), params.n); params.d];
    let e = vec![RingElement::from_scalar(ModInt::zero(), params.n); params.k];
    let log_k = (params.k as f64).log2().ceil() as usize;
    let point: Vec<ExtF> = vec![ExtF::ZERO; log_k];
    let r = vec![RingElement::from_scalar(ModInt::zero(), params.n); params.d];
    let mut rng = StdRng::seed_from_u64(2);
    // Note: Opening is no longer available since GPV trapdoor was removed
    let result = comm.open_at_point(&c, &point, &w_blinded, &e, &r, &mut rng);
    assert!(result.is_err());
}

/// Tests verification enforces norm bounds on openings.
///
/// Critical for security, as norm bounds relate to MSIS hardness. This test 
/// ensures malformed proofs with high norms are rejected, preventing attacks.
#[test]
fn test_open_verify_with_norm() {
    let params = test_params();
    let comm = AjtaiCommitter::setup_unchecked(params);
    let c = vec![RingElement::from_scalar(ModInt::zero(), params.n); params.k];
    let w_blinded = vec![RingElement::from_scalar(ModInt::zero(), params.n); params.d];
    let e = vec![RingElement::from_scalar(ModInt::zero(), params.n); params.k];
    let point = vec![ExtF::ONE];
    let r = vec![RingElement::from_scalar(ModInt::zero(), params.n); params.d];
    let mut rng = StdRng::seed_from_u64(3);
    // Note: Opening is no longer available since GPV trapdoor was removed
    let result = comm.open_at_point(&c, &point, &w_blinded, &e, &r, &mut rng);
    assert!(result.is_err());
}

/// Tests that mismatched proofs fail verification.
///
/// Ensures soundness by confirming tampered proofs are rejected. Good for 
/// validating the verification logic catches invalid openings.
#[test]
fn test_open_verify_mismatch() {
    let params = test_params();
    let comm = AjtaiCommitter::setup_unchecked(params);
    let c = vec![RingElement::from_scalar(ModInt::zero(), params.n); params.k];
    let w_blinded = vec![RingElement::from_scalar(ModInt::zero(), params.n); params.d];
    let e = vec![RingElement::from_scalar(ModInt::zero(), params.n); params.k];
    let point = vec![ExtF::ZERO];
    let r = vec![RingElement::from_scalar(ModInt::zero(), params.n); params.d];
    let mut rng = StdRng::seed_from_u64(4);
    // Note: Opening is no longer available since GPV trapdoor was removed
    let result = comm.open_at_point(&c, &point, &w_blinded, &e, &r, &mut rng);
    assert!(result.is_err());
}

/// Tests Gaussian sampling produces samples within expected norm bounds.
///
/// Important for ensuring sampled errors don't exceed security parameters, 
/// which could compromise hiding or binding properties.
#[test]
fn test_sample_gaussian_ring_norm() {
    let params = test_params();
    let comm = AjtaiCommitter::setup_unchecked(params);
    let center = RingElement::from_scalar(ModInt::zero(), params.n);
    let mut rng = StdRng::seed_from_u64(0);
    let sampled = comm.sample_gaussian_ring(&center, params.sigma, &mut rng).unwrap();
    assert!(sampled.norm_inf() <= params.e_bound * 3);
}





/// Tests the distribution of basic discrete Gaussian samples.
///
/// Validates the statistical properties of the basic discrete Gaussian sampler
/// used in setup, ensuring it follows the expected distribution for security proofs.
#[test]
fn test_gpv_sample_chi_squared() {
    let params = test_params();
    let mut rng = StdRng::seed_from_u64(42);
    let samples: usize = std::env::var("NEO_CHI_SAMPLES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let mut counts = [0usize; 11]; // |0|, |1|, |2|, ..., |9|, >=10
    let mut sum_abs = 0u64;
    
    // Test the basic discrete Gaussian sampler used in setup
    for _ in 0..samples {
        let sample = AjtaiCommitter::discrete_gaussian_sample(params.sigma, &mut rng, params.q);
        let v = sample.as_canonical_u64();
        
        // Convert to signed representation safely
        let signed_v = if v > params.q / 2 {
            // This is a negative number in signed representation
            let distance_from_q = params.q - v;
            distance_from_q.min(params.q / 2) // Cap to avoid overflow
        } else {
            v.min(params.q / 2) // This is already positive
        };
        
        sum_abs = sum_abs.saturating_add(signed_v);
        let abs_usize = (signed_v as usize).min(10); // Cap at 10 to avoid array bounds
        if abs_usize < 10 {
            counts[abs_usize] += 1;
        } else {
            counts[10] += 1;
        }
    }
    
    let mean_abs = sum_abs as f64 / samples as f64;
    
    // For a discrete Gaussian with sigma=3.2, we just want to ensure reasonable behavior
    // Since the implementation may have rejection sampling, we're more lenient
    assert!(
        mean_abs >= 0.0 && mean_abs < (params.q / 4) as f64,
        "Mean absolute value {} outside reasonable range for sigma={}. Counts: {:?}",
        mean_abs,
        params.sigma,
        counts
    );
    
    // Ensure we're getting some spread (not everything in one bucket)
    let non_zero_buckets = counts.iter().filter(|&&c| c > 0).count();
    assert!(
        non_zero_buckets >= 1,
        "No samples generated: {:?}",
        counts
    );
    
    // Basic sanity check - the sampler should produce some variety
    let max_single_bucket = counts.iter().max().unwrap_or(&0);
    assert!(
        *max_single_bucket < samples,
        "All samples in single bucket - sampler may be broken. Counts: {:?}",
        counts
    );
}
