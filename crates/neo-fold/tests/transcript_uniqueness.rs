use neo_fold::transcript::DuplexChallenger;
use p3_challenger::CanObserve;
use p3_goldilocks::Goldilocks;
use p3_poseidon2::Poseidon2;

type TestTranscript = DuplexChallenger<Goldilocks, Poseidon2<16>, 16, 8>;

#[test]
fn transcript_domain_separation() {
    // Test that absorbing the same payload under different labels produces different challenges
    let perm = Poseidon2::new_from_rng_128(&mut rand::thread_rng());
    
    let payload = b"test_payload_data";
    let label1 = b"domain:label1";
    let label2 = b"domain:label2";
    
    // Create two transcripts with different domain separation
    let mut transcript1 = TestTranscript::new(perm.clone());
    let mut transcript2 = TestTranscript::new(perm);
    
    // Absorb different labels followed by the same payload
    transcript1.observe_slice(label1);
    transcript1.observe_slice(payload);
    
    transcript2.observe_slice(label2);
    transcript2.observe_slice(payload);
    
    // Sample challenges - they should be different due to domain separation
    let challenge1 = transcript1.sample();
    let challenge2 = transcript2.sample();
    
    assert_ne!(challenge1, challenge2, "Domain separation failed - same challenges produced");
}

#[test]
fn transcript_ordering_sensitivity() {
    // Test that the transcript is sensitive to the order of absorption
    let perm = Poseidon2::new_from_rng_128(&mut rand::thread_rng());
    
    let data1 = b"first_piece";
    let data2 = b"second_piece";
    
    // Create two transcripts
    let mut transcript_a = TestTranscript::new(perm.clone());
    let mut transcript_b = TestTranscript::new(perm);
    
    // Absorb in different orders
    transcript_a.observe_slice(data1);
    transcript_a.observe_slice(data2);
    
    transcript_b.observe_slice(data2);  
    transcript_b.observe_slice(data1);
    
    // Sample challenges - they should be different due to ordering
    let challenge_a = transcript_a.sample();
    let challenge_b = transcript_b.sample();
    
    assert_ne!(challenge_a, challenge_b, "Transcript should be order-sensitive");
}

#[test]
fn transcript_length_sensitivity() {
    // Test that transcripts are sensitive to the length of absorbed data
    let perm = Poseidon2::new_from_rng_128(&mut rand::thread_rng());
    
    let short_data = b"short";
    let long_data = b"short_extended";
    
    let mut transcript1 = TestTranscript::new(perm.clone());
    let mut transcript2 = TestTranscript::new(perm);
    
    transcript1.observe_slice(short_data);
    transcript2.observe_slice(long_data);
    
    let challenge1 = transcript1.sample();
    let challenge2 = transcript2.sample();
    
    assert_ne!(challenge1, challenge2, "Transcript should distinguish different lengths");
}

#[test]
fn transcript_deterministic() {
    // Test that the same sequence of operations produces the same challenge
    let perm = Poseidon2::new_from_rng_128(&mut rand::thread_rng());
    
    let test_data = b"deterministic_test_data";
    
    let mut transcript1 = TestTranscript::new(perm.clone());
    let mut transcript2 = TestTranscript::new(perm);
    
    // Same operations on both transcripts
    transcript1.observe_slice(test_data);
    transcript2.observe_slice(test_data);
    
    let challenge1 = transcript1.sample();
    let challenge2 = transcript2.sample();
    
    assert_eq!(challenge1, challenge2, "Identical operations should produce identical challenges");
    
    // Sample again to make sure they remain in sync
    let challenge1_2 = transcript1.sample();
    let challenge2_2 = transcript2.sample();
    
    assert_eq!(challenge1_2, challenge2_2, "Transcripts should remain synchronized");
}
