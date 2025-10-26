use neo_transcript::{Poseidon2Transcript, labels as tr_labels, Transcript};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

#[test]
fn test_transcript_basic() {
    let mut tr = Poseidon2Transcript::new(b"test");
    
    // Test domain separation
    tr.append_message(tr_labels::PI_CCS, b"");
    
    // Test absorbing different types
    tr.append_u64s(b"u64s", &[1,2,3]);
    tr.append_fields(b"F", &[F::from_u32(42)]);
    tr.append_message(b"bytes", b"test_data");
    
    // Test challenge generation
    let f_challenges = (0..3).map(|_| tr.challenge_field(b"chal/f")).collect::<Vec<_>>();
    assert_eq!(f_challenges.len(), 3);
    
    let ks = tr.challenge_fields(b"chal/k", 2);
    let k_challenge = neo_math::from_complex(ks[0], ks[1]);
    println!("Extension challenge: {:?}", k_challenge);
    
    println!("Transcript test completed successfully");
}

#[test]
fn test_domain_separation() {
    let mut tr1 = Poseidon2Transcript::new(b"test");
    let mut tr2 = Poseidon2Transcript::new(b"test");
    
    // Same input, different domains
    tr1.append_message(tr_labels::PI_CCS, b"");
    tr2.append_message(tr_labels::PI_RLC, b"");
    
    tr1.append_u64s(b"u", &[1,2,3]);
    tr2.append_u64s(b"u", &[1,2,3]);
    
    let c1 = tr1.challenge_field(b"chal/f");
    let c2 = tr2.challenge_field(b"chal/f");
    
    // Should be different due to domain separation
    assert_ne!(c1, c2);
    println!("Domain separation working: {} != {}", c1.as_canonical_u64(), c2.as_canonical_u64());
}
