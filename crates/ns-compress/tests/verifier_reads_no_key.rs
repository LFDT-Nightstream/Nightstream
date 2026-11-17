mod support;

use neo_ajtai::nightstream_fprime_setup::{expanded_key_bytes, ELEMENT_OUTPUT_BYTES};
use neo_math::K;
use neo_transcript::Poseidon2Transcript;
use ns_compress::{commit, verify, Layout, ProverKey, VerificationKey};
use p3_field::PrimeCharacteristicRing;
use support::{instances, statements};

#[test]
fn the_verifier_expands_no_key_element() {
    let before = expanded_key_bytes();
    let key = ProverKey::new(100).expect("key");
    assert_eq!(expanded_key_bytes() - before, 18 * 512 * ELEMENT_OUTPUT_BYTES as u64);

    let instances = instances();
    let claims: Vec<_> = instances
        .iter()
        .map(|instance| (instance.witness.cols(), instance.norm))
        .collect();
    let layout = Layout::new(&claims, key.verification_key().blocks()).expect("layout");
    let witnesses: Vec<_> = instances.iter().map(|instance| &instance.witness).collect();
    let (proof, _) = commit(&key, layout, &witnesses)
        .expect("commitment")
        .prove(&mut Poseidon2Transcript::new_v1_2(), &statements(&instances))
        .expect("proof");
    let verification = VerificationKey::from_bytes(&key.verification_key().to_bytes()).expect("key bytes");
    drop(key);

    let before = expanded_key_bytes();
    let accepts = |proof| {
        verify(
            &mut Poseidon2Transcript::new_v1_2(),
            &verification,
            &statements(&instances),
            proof,
        )
    };
    accepts(&proof).expect("the proof verifies");
    let mut changed = proof.clone();
    changed.key_part += K::ONE;
    assert!(accepts(&changed).is_err());
    let mut changed = proof.clone();
    *changed.opening.last_mut().expect("opening") ^= 1;
    assert!(accepts(&changed).is_err());
    assert_eq!(expanded_key_bytes(), before, "the verifier expanded key elements");
}
