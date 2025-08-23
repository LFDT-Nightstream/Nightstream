use neo_sumcheck::*;
use p3_field::PrimeField64;

#[test]
fn fs_structured_prevents_concat_ambiguity() {
    // "ab"||"c" vs "a"||"bc" collide in raw concatenation; structured framing must differ.
    let mut t1 = Vec::new();
    fs_absorb_bytes(&mut t1, b"msg", b"ab");
    fs_absorb_bytes(&mut t1, b"msg", b"c");
    let c1 = fs_challenge_ext(&mut t1, b"final");

    let mut t2 = Vec::new();
    fs_absorb_bytes(&mut t2, b"msg", b"a");
    fs_absorb_bytes(&mut t2, b"msg", b"bc");
    let c2 = fs_challenge_ext(&mut t2, b"final");

    assert_ne!(c1, c2, "structured FS must not collide on chunking differences");
}

#[test]
fn canonical_helpers_work() {
    let transcript = b"test_transcript";
    let label = "test_label";
    
    let base_challenge = fs_challenge_base_labeled(transcript, label);
    let _ext_challenge = fs_challenge_ext_labeled(transcript, label);
    let u64_challenge = fs_challenge_u64_labeled(transcript, label);
    
    // Verify u64 challenge matches base challenge
    assert_eq!(u64_challenge, base_challenge.as_canonical_u64());
    
    // Verify different labels produce different challenges
    let different_challenge = fs_challenge_base_labeled(transcript, "different_label");
    assert_ne!(base_challenge, different_challenge);
}

#[test]
fn transcript_facade_basic_functionality() {
    let mut transcript = Transcript::new("test_protocol");
    
    // Test absorbing different types
    transcript.absorb_tag("phase1");
    transcript.absorb_u64("round", 42);
    transcript.absorb_bytes("data", b"test_data");
    
    // Test challenges
    let base_challenge = transcript.challenge_base("test_base");
    let ext_challenge = transcript.challenge_ext("test_ext");
    
    // Challenges should be deterministic
    let mut transcript2 = Transcript::new("test_protocol");
    transcript2.absorb_tag("phase1");
    transcript2.absorb_u64("round", 42);
    transcript2.absorb_bytes("data", b"test_data");
    
    let base_challenge2 = transcript2.challenge_base("test_base");
    let ext_challenge2 = transcript2.challenge_ext("test_ext");
    
    assert_eq!(base_challenge, base_challenge2);
    assert_eq!(ext_challenge, ext_challenge2);
}

#[test]
fn transcript_fork_independence() {
    let mut transcript = Transcript::new("test_protocol");
    transcript.absorb_tag("common");
    
    let mut fork1 = transcript.fork("branch1");
    let mut fork2 = transcript.fork("branch2");
    
    fork1.absorb_bytes("data", b"fork1_data");
    fork2.absorb_bytes("data", b"fork2_data");
    
    let challenge1 = fork1.challenge_base("test");
    let challenge2 = fork2.challenge_base("test");
    
    // Forks should produce different challenges
    assert_ne!(challenge1, challenge2);
}

#[test]
fn transcript_modint_challenge_no_bias() {
    let mut transcript = Transcript::new("bias_test");
    
    // Generate multiple challenges and verify they're in range
    for i in 0..100 {
        transcript.absorb_u64("iteration", i);
        let challenge = transcript.challenge_modint("test_modint");
        let q = <ModInt as Coeff>::modulus();
        assert!(challenge.as_canonical_u64() < q, "Challenge {} out of range", challenge.as_canonical_u64());
    }
}

#[test]
fn transcript_rng_quality() {
    // Test that same transcript state + same label = same RNG
    let mut transcript1 = Transcript::new("rng_test");
    transcript1.absorb_tag("seed_test");
    let transcript1_clone = transcript1.clone();
    
    let mut rng1 = transcript1.rng("test_rng");
    let mut rng2 = transcript1_clone.clone().rng("test_rng"); // Same state, same label
    
    // Same seed should produce same sequence
    use rand::Rng;
    let val1: u64 = rng1.random();
    let val2: u64 = rng2.random();
    assert_eq!(val1, val2);
    
    // Different labels should give different RNGs
    let mut rng3 = transcript1_clone.clone().rng("different_label");
    let val3: u64 = rng3.random();
    assert_ne!(val1, val3);
}
