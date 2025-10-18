//! Pi-RLC Transcript Binding Security Tests
//!
//! These tests validate that Pi-RLC properly binds ME instance contents
//! to the transcript before sampling ρ challenges, preventing length-based
//! malleability attacks.
//!
//! Test Strategy:
//! 1. Transcript Independence: Different ME contents → different ρ values
//! 2. Permutation Sensitivity: Reordering instances → different ρ values  
//! 3. Content Binding: Modifying any field (c, X, y, r) → different ρ values
//! 4. Dimension Guards: Invalid dimensions are rejected

#![allow(non_snake_case)] // Allow X, Z naming from paper

use neo_fold::pi_rlc::{pi_rlc_prove, pi_rlc_verify};
use neo_transcript::{Transcript, Poseidon2Transcript};
use neo_ccs::{MeInstance, Mat};
use neo_ajtai::Commitment as Cmt;
use neo_math::{F, K, D};
use neo_params::NeoParams;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

/// Deterministic PRNG mixer (splitmix64-style) to generate unique test data per seed
#[inline]
fn mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Generate a stream of F elements from a seed
fn f_stream(seed: u64, n: usize) -> Vec<F> {
    let mut s = seed ^ 0xDEADBEEFCAFEBABE;
    (0..n).map(|_| {
        s = mix64(s);
        F::from_u64(s)
    }).collect()
}

/// Generate a stream of K elements from a seed
fn k_stream(seed: u64, n: usize) -> Vec<K> {
    let mut s = seed ^ 0xCAFED00D42424242;
    (0..n).map(|_| {
        s = mix64(s);
        K::from(F::from_u64(s))
    }).collect()
}

/// Helper: Create a simple test ME instance with given seed
/// NOTE: r must be the same across instances in a list (Pi-RLC requirement),
/// so we use a fixed r that's independent of seed.
fn make_test_me_instance(seed: u64, m_in: usize, t: usize) -> MeInstance<Cmt, F, K> {
    let d = D;
    let kappa = 128; // Ajtai commitment has kappa columns
    
    // Create deterministic but DIFFERENT content based on seed
    // Commitment data size must be d * kappa (not d * m_in)
    let c_data: Vec<F> = f_stream(seed.wrapping_mul(101), d * kappa);
    let c = Cmt { 
        data: c_data, 
        d: d, 
        kappa: kappa 
    };
    
    // X matrix - varies with seed
    let x_data: Vec<F> = f_stream(seed.wrapping_mul(131), d * m_in);
    let X = Mat::from_row_major(d, m_in, x_data);
    
    // y vectors - vary with seed
    let mut y: Vec<Vec<K>> = Vec::with_capacity(t);
    let mut s = seed.wrapping_mul(151);
    for _ in 0..t {
        let row: Vec<K> = (0..d).map(|_| {
            s = mix64(s);
            K::from(F::from_u64(s))
        }).collect();
        y.push(row);
    }
    
    // r must be CONSTANT across all instances (Pi-RLC requirement)
    // Use a fixed value independent of seed
    let r: Vec<K> = (0..5)
        .map(|i| K::from(F::from_u64(0x1234_5678_9ABC_DEF0u64 ^ (i as u64))))
        .collect();
    
    // y_scalars can vary with seed
    let y_scalars: Vec<K> = k_stream(seed.wrapping_mul(171), t);
    
    // fold_digest varies with seed
    let fold_digest: [u8; 32] = {
        let mut digest = [0u8; 32];
        let mut s = seed;
        for i in 0..32 {
            s = mix64(s);
            digest[i] = (s % 256) as u8;
        }
        digest
    };
    
    MeInstance {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
        c,
        X,
        y,
        y_scalars,
        r,
        m_in,
        fold_digest,
    }
}

/// Extract ρ coefficients from proof for comparison
fn extract_rho_coeffs(proof: &neo_fold::PiRlcProof) -> Vec<Vec<u64>> {
    proof.rho_elems
        .iter()
        .map(|rho_elem| {
            rho_elem.iter().map(|&f| f.as_canonical_u64()).collect()
        })
        .collect()
}

#[test]
fn test_pi_rlc_transcript_binds_commitment_content() {
    // CRITICAL: This test validates that changing commitment content
    // produces different ρ challenges (prevents substitution attacks)
    
    let params = NeoParams::goldilocks_small_circuits();
    let m_in = 4;
    let t = 2;
    
    // Create two ME lists with same length but different commitment in first instance
    let me_list_a = vec![
        make_test_me_instance(1, m_in, t),
        make_test_me_instance(2, m_in, t),
    ];
    
    let mut me_list_b = me_list_a.clone();
    // Modify only the commitment of first instance
    me_list_b[0].c.data[0] = F::from_u64(999);
    
    // Prove with fresh transcripts (no Pi-CCS binding)
    let mut tr_a = Poseidon2Transcript::new(b"test/pi_rlc_binding");
    let mut tr_b = Poseidon2Transcript::new(b"test/pi_rlc_binding");
    
    let (_, proof_a) = pi_rlc_prove(&mut tr_a, &params, &me_list_a)
        .expect("proof A should succeed");
    let (_, proof_b) = pi_rlc_prove(&mut tr_b, &params, &me_list_b)
        .expect("proof B should succeed");
    
    let rho_a = extract_rho_coeffs(&proof_a);
    let rho_b = extract_rho_coeffs(&proof_b);
    
    // SECURITY REQUIREMENT: Different commitments must produce different ρ values
    assert_ne!(
        rho_a, rho_b,
        "SECURITY VIOLATION: Different commitment contents produced identical ρ challenges!\n\
         This allows commitment substitution attacks."
    );
}

#[test]
fn test_pi_rlc_transcript_binds_x_matrix() {
    // Validates that changing X matrix produces different ρ challenges
    
    let params = NeoParams::goldilocks_small_circuits();
    let m_in = 4;
    let t = 2;
    
    let me_list_a = vec![
        make_test_me_instance(10, m_in, t),
        make_test_me_instance(20, m_in, t),
    ];
    
    let mut me_list_b = me_list_a.clone();
    // Modify X matrix of first instance
    me_list_b[0].X[(0, 0)] = F::from_u64(777);
    
    let mut tr_a = Poseidon2Transcript::new(b"test/pi_rlc_x_bind");
    let mut tr_b = Poseidon2Transcript::new(b"test/pi_rlc_x_bind");
    
    let (_, proof_a) = pi_rlc_prove(&mut tr_a, &params, &me_list_a)
        .expect("proof A should succeed");
    let (_, proof_b) = pi_rlc_prove(&mut tr_b, &params, &me_list_b)
        .expect("proof B should succeed");
    
    let rho_a = extract_rho_coeffs(&proof_a);
    let rho_b = extract_rho_coeffs(&proof_b);
    
    assert_ne!(
        rho_a, rho_b,
        "SECURITY VIOLATION: Different X matrices produced identical ρ challenges!"
    );
}

#[test]
fn test_pi_rlc_transcript_binds_y_vectors() {
    // Validates that changing y vectors produces different ρ challenges
    
    let params = NeoParams::goldilocks_small_circuits();
    let m_in = 4;
    let t = 2;
    
    let me_list_a = vec![
        make_test_me_instance(30, m_in, t),
        make_test_me_instance(40, m_in, t),
    ];
    
    let mut me_list_b = me_list_a.clone();
    // Modify y vector of first instance
    me_list_b[0].y[0][0] = K::from(F::from_u64(888));
    
    let mut tr_a = Poseidon2Transcript::new(b"test/pi_rlc_y_bind");
    let mut tr_b = Poseidon2Transcript::new(b"test/pi_rlc_y_bind");
    
    let (_, proof_a) = pi_rlc_prove(&mut tr_a, &params, &me_list_a)
        .expect("proof A should succeed");
    let (_, proof_b) = pi_rlc_prove(&mut tr_b, &params, &me_list_b)
        .expect("proof B should succeed");
    
    let rho_a = extract_rho_coeffs(&proof_a);
    let rho_b = extract_rho_coeffs(&proof_b);
    
    assert_ne!(
        rho_a, rho_b,
        "SECURITY VIOLATION: Different y vectors produced identical ρ challenges!"
    );
}

#[test]
fn test_pi_rlc_transcript_binds_r_vector() {
    // Validates that changing r (challenge point) produces different ρ challenges
    
    let params = NeoParams::goldilocks_small_circuits();
    let m_in = 4;
    let t = 2;
    
    let me_list_a = vec![
        make_test_me_instance(50, m_in, t),
        make_test_me_instance(60, m_in, t),
    ];
    
    let mut me_list_b = me_list_a.clone();
    // Modify r vector of BOTH instances (Pi-RLC requires all instances to have same r)
    for me in &mut me_list_b {
        me.r[0] = K::from(F::from_u64(555));
    }
    
    let mut tr_a = Poseidon2Transcript::new(b"test/pi_rlc_r_bind");
    let mut tr_b = Poseidon2Transcript::new(b"test/pi_rlc_r_bind");
    
    let (_, proof_a) = pi_rlc_prove(&mut tr_a, &params, &me_list_a)
        .expect("proof A should succeed");
    let (_, proof_b) = pi_rlc_prove(&mut tr_b, &params, &me_list_b)
        .expect("proof B should succeed");
    
    let rho_a = extract_rho_coeffs(&proof_a);
    let rho_b = extract_rho_coeffs(&proof_b);
    
    assert_ne!(
        rho_a, rho_b,
        "SECURITY VIOLATION: Different r vectors produced identical ρ challenges!"
    );
}

#[test]
fn test_pi_rlc_transcript_binds_fold_digest() {
    // Validates that changing fold_digest produces different ρ challenges
    // 
    // NOTE: fold_digest should be absorbed by Pi-RLC because it carries the
    // transcript state from Pi-CCS. This binds the RLC operation to the specific
    // folding pipeline and prevents cross-pipeline attacks.
    // 
    // If your implementation decides fold_digest is a "running pipeline digest"
    // not semantically part of Pi-RLC inputs, you may relax this test or ensure
    // the digest is absorbed via append_message() as bytes.
    
    let params = NeoParams::goldilocks_small_circuits();
    let m_in = 4;
    let t = 2;
    
    let me_list_a = vec![
        make_test_me_instance(70, m_in, t),
        make_test_me_instance(80, m_in, t),
    ];
    
    let mut me_list_b = me_list_a.clone();
    // Modify fold_digest of first instance
    me_list_b[0].fold_digest[0] = 0xFF;
    
    let mut tr_a = Poseidon2Transcript::new(b"test/pi_rlc_digest_bind");
    let mut tr_b = Poseidon2Transcript::new(b"test/pi_rlc_digest_bind");
    
    let (_, proof_a) = pi_rlc_prove(&mut tr_a, &params, &me_list_a)
        .expect("proof A should succeed");
    let (_, proof_b) = pi_rlc_prove(&mut tr_b, &params, &me_list_b)
        .expect("proof B should succeed");
    
    let rho_a = extract_rho_coeffs(&proof_a);
    let rho_b = extract_rho_coeffs(&proof_b);
    
    assert_ne!(
        rho_a, rho_b,
        "SECURITY VIOLATION: Different fold_digests produced identical ρ challenges!\n\
         fold_digest binds Pi-RLC to the upstream Pi-CCS transcript state."
    );
}

#[test]
fn test_pi_rlc_permutation_sensitivity() {
    // CRITICAL: Reordering instances must change ρ values
    // (prevents permutation-based attacks)
    
    let params = NeoParams::goldilocks_small_circuits();
    let m_in = 4;
    let t = 2;
    
    let me_list_forward = vec![
        make_test_me_instance(100, m_in, t),
        make_test_me_instance(200, m_in, t),
        make_test_me_instance(300, m_in, t),
    ];
    
    // Reverse order
    let me_list_reversed = vec![
        me_list_forward[2].clone(),
        me_list_forward[1].clone(),
        me_list_forward[0].clone(),
    ];
    
    let mut tr_fwd = Poseidon2Transcript::new(b"test/pi_rlc_perm");
    let mut tr_rev = Poseidon2Transcript::new(b"test/pi_rlc_perm");
    
    let (_, proof_fwd) = pi_rlc_prove(&mut tr_fwd, &params, &me_list_forward)
        .expect("forward proof should succeed");
    let (_, proof_rev) = pi_rlc_prove(&mut tr_rev, &params, &me_list_reversed)
        .expect("reversed proof should succeed");
    
    let rho_fwd = extract_rho_coeffs(&proof_fwd);
    let rho_rev = extract_rho_coeffs(&proof_rev);
    
    assert_ne!(
        rho_fwd, rho_rev,
        "SECURITY VIOLATION: Permuted instance order produced identical ρ challenges!\n\
         This allows reordering attacks."
    );
}

#[test]
fn test_pi_rlc_length_alone_insufficient() {
    // The CORE vulnerability test: Same length but different content
    // should produce different ρ values
    
    let params = NeoParams::goldilocks_small_circuits();
    let m_in = 4;
    let t = 2;
    
    // Two completely different ME lists, same length
    let me_list_alice = vec![
        make_test_me_instance(111, m_in, t),
        make_test_me_instance(222, m_in, t),
    ];
    
    let me_list_bob = vec![
        make_test_me_instance(333, m_in, t),
        make_test_me_instance(444, m_in, t),
    ];
    
    let mut tr_alice = Poseidon2Transcript::new(b"test/pi_rlc_length");
    let mut tr_bob = Poseidon2Transcript::new(b"test/pi_rlc_length");
    
    let (_, proof_alice) = pi_rlc_prove(&mut tr_alice, &params, &me_list_alice)
        .expect("Alice proof should succeed");
    let (_, proof_bob) = pi_rlc_prove(&mut tr_bob, &params, &me_list_bob)
        .expect("Bob proof should succeed");
    
    let rho_alice = extract_rho_coeffs(&proof_alice);
    let rho_bob = extract_rho_coeffs(&proof_bob);
    
    assert_ne!(
        rho_alice, rho_bob,
        "CRITICAL SECURITY VIOLATION: Completely different ME instances with same length\n\
         produced identical ρ challenges! This is the core length-malleability vulnerability.\n\
         The transcript must bind instance CONTENT, not just length."
    );
}

#[test]
fn test_pi_rlc_verifier_rejects_wrong_rho() {
    // Validates that verifier rejects proofs with tampered ρ values
    
    let params = NeoParams::goldilocks_small_circuits();
    let m_in = 4;
    let t = 2;
    
    let me_list = vec![
        make_test_me_instance(500, m_in, t),
        make_test_me_instance(600, m_in, t),
    ];
    
    let mut tr_prove = Poseidon2Transcript::new(b"test/pi_rlc_tamper");
    let (output_me, mut proof) = pi_rlc_prove(&mut tr_prove, &params, &me_list)
        .expect("prove should succeed");
    
    // Tamper with ρ in the proof
    proof.rho_elems[0][0] = F::from_u64(9999);
    
    let mut tr_verify = Poseidon2Transcript::new(b"test/pi_rlc_tamper");
    let result = pi_rlc_verify(&mut tr_verify, &params, &me_list, &output_me, &proof)
        .expect("verify should complete");
    
    assert!(
        !result,
        "Verifier should reject proof with tampered ρ values"
    );
}

#[test]
fn test_pi_rlc_chained_transcript_preserves_binding() {
    // Validates that when Pi-RLC is called with a transcript that already
    // absorbed data (like from Pi-CCS), the prior bindings still affect ρ
    
    let params = NeoParams::goldilocks_small_circuits();
    let m_in = 4;
    let t = 2;
    
    let me_list = vec![
        make_test_me_instance(700, m_in, t),
        make_test_me_instance(800, m_in, t),
    ];
    
    // Case A: Fresh transcript
    let mut tr_fresh = Poseidon2Transcript::new(b"test/chained");
    let (_, proof_fresh) = pi_rlc_prove(&mut tr_fresh, &params, &me_list)
        .expect("fresh proof should succeed");
    
    // Case B: Transcript with prior absorptions (simulating Pi-CCS output)
    let mut tr_chained = Poseidon2Transcript::new(b"test/chained");
    tr_chained.append_message(b"prior_protocol", b"some_data");
    tr_chained.append_fields(b"prior_fields", &[F::from_u64(12345)]);
    let (_, proof_chained) = pi_rlc_prove(&mut tr_chained, &params, &me_list)
        .expect("chained proof should succeed");
    
    let rho_fresh = extract_rho_coeffs(&proof_fresh);
    let rho_chained = extract_rho_coeffs(&proof_chained);
    
    assert_ne!(
        rho_fresh, rho_chained,
        "Prior transcript bindings should affect ρ sampling in chained calls"
    );
}

#[test]
fn test_pi_rlc_rejects_invalid_x_dimensions() {
    // Validates dimension guards (now implemented in the hardening section)
    // This ensures that X.rows() must equal D to prevent silent truncation
    
    let params = NeoParams::goldilocks_small_circuits();
    let m_in = 4;
    let t = 2;
    
    let mut me_invalid = make_test_me_instance(900, m_in, t);
    
    // Create X with wrong number of rows (this should be caught)
    let wrong_d = D - 1;
    let x_data: Vec<F> = (0..wrong_d * m_in)
        .map(|i| F::from_u64(i as u64))
        .collect();
    me_invalid.X = Mat::from_row_major(wrong_d, m_in, x_data);
    
    let me_list = vec![me_invalid, make_test_me_instance(1000, m_in, t)];
    
    let mut tr = Poseidon2Transcript::new(b"test/dim_guard");
    let result = pi_rlc_prove(&mut tr, &params, &me_list);
    
    // Should return an error about invalid dimensions
    assert!(result.is_err(), "Should reject invalid X dimensions");
}

#[test]
fn test_pi_rlc_determinism() {
    // Validates that identical inputs with identical transcripts produce identical ρ
    
    let params = NeoParams::goldilocks_small_circuits();
    let m_in = 4;
    let t = 2;
    
    let me_list = vec![
        make_test_me_instance(1100, m_in, t),
        make_test_me_instance(1200, m_in, t),
    ];
    
    // Run twice with identical setup
    let mut tr_1 = Poseidon2Transcript::new(b"test/determinism");
    let (_, proof_1) = pi_rlc_prove(&mut tr_1, &params, &me_list)
        .expect("proof 1 should succeed");
    
    let mut tr_2 = Poseidon2Transcript::new(b"test/determinism");
    let (_, proof_2) = pi_rlc_prove(&mut tr_2, &params, &me_list)
        .expect("proof 2 should succeed");
    
    let rho_1 = extract_rho_coeffs(&proof_1);
    let rho_2 = extract_rho_coeffs(&proof_2);
    
    assert_eq!(
        rho_1, rho_2,
        "Identical inputs and transcripts must produce identical ρ values"
    );
}

#[test]
fn test_verify_rejects_swapped_inputs_of_same_length() {
    // CRITICAL: Verifier must reject proofs when inputs are swapped
    // This validates that the verifier also binds instance content, not just prover
    
    let params = NeoParams::goldilocks_small_circuits();
    let m_in = 4;
    let t = 2;
    
    let me_list_a = vec![
        make_test_me_instance(1, m_in, t),
        make_test_me_instance(2, m_in, t),
    ];
    
    let me_list_b = vec![
        make_test_me_instance(3, m_in, t),
        make_test_me_instance(4, m_in, t),
    ];
    
    // Prove with me_list_a
    let mut tr_prove = Poseidon2Transcript::new(b"test/v-reject");
    let (output_me, proof) = pi_rlc_prove(&mut tr_prove, &params, &me_list_a)
        .expect("prove should succeed");
    
    // Try to verify with me_list_b (different content, same length)
    let mut tr_verify = Poseidon2Transcript::new(b"test/v-reject");
    let result = pi_rlc_verify(&mut tr_verify, &params, &me_list_b, &output_me, &proof)
        .expect("verify should complete");
    
    assert!(
        !result,
        "SECURITY VIOLATION: Verifier accepted proof bound to different content!\n\
         Proof was created for me_list_a but verified against me_list_b."
    );
}

#[test]
fn test_pi_rlc_shape_binding_m_in() {
    // Validates that changing m_in (shape parameter) produces different ρ
    // Even if the commitment data is similar, different shapes should yield different ρ
    
    let params = NeoParams::goldilocks_small_circuits();
    let t = 2;
    
    let me_list_shape_4 = vec![
        make_test_me_instance(1300, 4, t),
        make_test_me_instance(1400, 4, t),
    ];
    
    let me_list_shape_5 = vec![
        make_test_me_instance(1300, 5, t),
        make_test_me_instance(1400, 5, t),
    ];
    
    let mut tr_4 = Poseidon2Transcript::new(b"test/shape");
    let (_, proof_4) = pi_rlc_prove(&mut tr_4, &params, &me_list_shape_4)
        .expect("proof with m_in=4 should succeed");
    
    let mut tr_5 = Poseidon2Transcript::new(b"test/shape");
    let (_, proof_5) = pi_rlc_prove(&mut tr_5, &params, &me_list_shape_5)
        .expect("proof with m_in=5 should succeed");
    
    let rho_4 = extract_rho_coeffs(&proof_4);
    let rho_5 = extract_rho_coeffs(&proof_5);
    
    assert_ne!(
        rho_4, rho_5,
        "SECURITY: Different m_in shapes must produce different ρ challenges!\n\
         This prevents shape-substitution attacks."
    );
}

#[test]
fn test_pi_rlc_shape_binding_t() {
    // Validates that changing t (number of CCS matrices) produces different ρ
    
    let params = NeoParams::goldilocks_small_circuits();
    let m_in = 4;
    
    let me_list_t2 = vec![
        make_test_me_instance(1500, m_in, 2),
        make_test_me_instance(1600, m_in, 2),
    ];
    
    let me_list_t3 = vec![
        make_test_me_instance(1500, m_in, 3),
        make_test_me_instance(1600, m_in, 3),
    ];
    
    let mut tr_2 = Poseidon2Transcript::new(b"test/t-shape");
    let (_, proof_2) = pi_rlc_prove(&mut tr_2, &params, &me_list_t2)
        .expect("proof with t=2 should succeed");
    
    let mut tr_3 = Poseidon2Transcript::new(b"test/t-shape");
    let (_, proof_3) = pi_rlc_prove(&mut tr_3, &params, &me_list_t3)
        .expect("proof with t=3 should succeed");
    
    let rho_2 = extract_rho_coeffs(&proof_2);
    let rho_3 = extract_rho_coeffs(&proof_3);
    
    assert_ne!(
        rho_2, rho_3,
        "SECURITY: Different t values must produce different ρ challenges!\n\
         This prevents substitution attacks across different CCS arities."
    );
}

#[test]
fn test_pi_rlc_happy_path_prove_and_verify() {
    // Positive test: Ensure that valid proofs verify correctly
    // This establishes a baseline for correct operation
    
    let params = NeoParams::goldilocks_small_circuits();
    let m_in = 4;
    let t = 2;
    
    let me_list = vec![
        make_test_me_instance(1700, m_in, t),
        make_test_me_instance(1800, m_in, t),
    ];
    
    // Prove
    let mut tr_prove = Poseidon2Transcript::new(b"test/happy");
    let (output_me, proof) = pi_rlc_prove(&mut tr_prove, &params, &me_list)
        .expect("prove should succeed");
    
    // Verify with same inputs
    let mut tr_verify = Poseidon2Transcript::new(b"test/happy");
    let result = pi_rlc_verify(&mut tr_verify, &params, &me_list, &output_me, &proof)
        .expect("verify should complete");
    
    assert!(
        result,
        "Valid proof should verify successfully (happy path)"
    );
}

