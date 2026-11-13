//! The verifier of the argument expands no element of the Ajtai key. This
//! file has one test, so no other thread of its process expands a key element
//! while the verifier runs. `expanded_key_bytes` counts the CPU expansion of
//! `neo-ajtai` only: an expansion in a Metal shader does not count, and this
//! test runs no Metal path.

mod support;

use neo_ajtai::nightstream_fprime_setup::expanded_key_bytes;
use neo_compress::{commit, verify, Layout, ProverKey, VerificationKey};
use neo_math::K;
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;
use support::{instances, statements};

/// The setup expands each of the 18 rows of 512 blocks once, 2,048 bytes of
/// SHAKE128 output each. The verifier accepts the proof and rejects changed
/// proofs with the count where it was.
#[test]
fn the_verifier_expands_no_key_element() {
    let before = expanded_key_bytes();
    let key = ProverKey::new(100).expect("key");
    assert_eq!(expanded_key_bytes() - before, 18 * 512 * 2048);

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
    changed.key_weight += K::ONE;
    assert!(accepts(&changed).is_err());
    let mut changed = proof.clone();
    *changed.opening.last_mut().expect("opening") ^= 1;
    assert!(accepts(&changed).is_err());
    assert_eq!(expanded_key_bytes(), before, "the verifier expanded key elements");
}
