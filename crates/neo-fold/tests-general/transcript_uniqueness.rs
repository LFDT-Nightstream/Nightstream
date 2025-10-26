use p3_challenger::{DuplexChallenger, CanObserve, CanSample};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};

type TestTranscript = DuplexChallenger<Goldilocks, Poseidon2Goldilocks<16>, 16, 8>;

#[test]
fn transcript_domain_separation() {
    // Test that absorbing the same payload under different labels produces different challenges
    let perm = Poseidon2Goldilocks::<16>::new_from_rng_128(&mut rand::rng());
    
    let payload = b"test_payload_data";
    let label1 = b"domain:label1";
    let label2 = b"domain:label2";
    
    // Create two transcripts with different domain separation
    let mut transcript1 = TestTranscript::new(perm.clone());
    let mut transcript2 = TestTranscript::new(perm);
    
    // Absorb different labels followed by the same payload
    // Convert bytes to field elements and observe individually
    for &byte in label1 { transcript1.observe(Goldilocks::from_u32(byte as u32)); }
    for &byte in payload { transcript1.observe(Goldilocks::from_u32(byte as u32)); }
    
    for &byte in label2 { transcript2.observe(Goldilocks::from_u32(byte as u32)); }
    for &byte in payload { transcript2.observe(Goldilocks::from_u32(byte as u32)); }
    
    // Sample challenges - they should be different due to domain separation
    let challenge1: Goldilocks = transcript1.sample();
    let challenge2: Goldilocks = transcript2.sample();
    
    assert_ne!(challenge1, challenge2, "Domain separation failed - same challenges produced");
}

#[test]
fn transcript_ordering_sensitivity() {
    // Test that the transcript is sensitive to the order of absorption
    let perm = Poseidon2Goldilocks::<16>::new_from_rng_128(&mut rand::rng());
    
    let data1 = b"first_piece";
    let data2 = b"second_piece";
    
    // Create two transcripts
    let mut transcript_a = TestTranscript::new(perm.clone());
    let mut transcript_b = TestTranscript::new(perm);
    
    // Absorb in different orders
    // Convert bytes to field elements and observe individually
    for &byte in data1 { transcript_a.observe(Goldilocks::from_u32(byte as u32)); }
    for &byte in data2 { transcript_a.observe(Goldilocks::from_u32(byte as u32)); }
    
    for &byte in data2 { transcript_b.observe(Goldilocks::from_u32(byte as u32)); }
    for &byte in data1 { transcript_b.observe(Goldilocks::from_u32(byte as u32)); }
    
    // Sample challenges - they should be different due to ordering
    let challenge_a: Goldilocks = transcript_a.sample();
    let challenge_b: Goldilocks = transcript_b.sample();
    
    assert_ne!(challenge_a, challenge_b, "Transcript should be order-sensitive");
}

#[test]
fn transcript_length_sensitivity() {
    // Test that transcripts are sensitive to the length of absorbed data
    let perm = Poseidon2Goldilocks::<16>::new_from_rng_128(&mut rand::rng());
    
    let short_data = b"short";
    let long_data = b"short_extended";
    
    let mut transcript1 = TestTranscript::new(perm.clone());
    let mut transcript2 = TestTranscript::new(perm);
    
    // Convert bytes to field elements and observe individually
    for &byte in short_data { transcript1.observe(Goldilocks::from_u32(byte as u32)); }
    for &byte in long_data { transcript2.observe(Goldilocks::from_u32(byte as u32)); }
    
    let challenge1: Goldilocks = transcript1.sample();
    let challenge2: Goldilocks = transcript2.sample();
    
    assert_ne!(challenge1, challenge2, "Transcript should distinguish different lengths");
}

#[test]
fn transcript_deterministic() {
    // Test that the same sequence of operations produces the same challenge
    let perm = Poseidon2Goldilocks::<16>::new_from_rng_128(&mut rand::rng());
    
    let test_data = b"deterministic_test_data";
    
    let mut transcript1 = TestTranscript::new(perm.clone());
    let mut transcript2 = TestTranscript::new(perm);
    
    // Same operations on both transcripts
    // Convert bytes to field elements and observe individually
    for &byte in test_data { transcript1.observe(Goldilocks::from_u32(byte as u32)); }
    for &byte in test_data { transcript2.observe(Goldilocks::from_u32(byte as u32)); }
    
    let challenge1: Goldilocks = transcript1.sample();
    let challenge2: Goldilocks = transcript2.sample();
    
    assert_eq!(challenge1, challenge2, "Identical operations should produce identical challenges");
    
    // Sample again to make sure they remain in sync
    let challenge1_2: Goldilocks = transcript1.sample();
    let challenge2_2: Goldilocks = transcript2.sample();
    
    assert_eq!(challenge1_2, challenge2_2, "Transcripts should remain synchronized");
}
