//! The compressed proof through the public API on the Rust Poseidon2 hash
//! chain (SPEC.md section 6). This file has one test, so no other thread of
//! its process expands a key element while the verifier runs. The test holds
//! the compression setup of 10 GB and needs 25 GB, so it is ignored.
//! `neo-compress` runs the same check on small claims.

use neo_ajtai::nightstream_fprime_setup::expanded_key_bytes;
use neo_math::F;
use nightstream::application::{poseidon2_hash_chain_step, poseidon2_hash_chain_v1};
use nightstream::{Circuit, FinalProof};
use nightstream::{Engine, PrimeCharacteristicRing, State};

/// The 18 rows of 2^19 blocks of the Ajtai key, 2,048 bytes of SHAKE128
/// output for each key element.
const KEY_BYTES: u64 = 18 * (1 << 19) * 2048;

fn circuit() -> Circuit {
    Circuit::compile(poseidon2_hash_chain_v1().expect("hash chain")).expect("bounded-depth circuit")
}

fn words(values: [u64; 4]) -> [F; 4] {
    values.map(F::from_u64)
}

/// A verifier and a prover each compile the circuit and make their own key
/// from it: the verifier its compression key, the prover its compression
/// setup. Each expands the rows of the Ajtai key once. The proofs of the
/// prover verify from their bytes under the key of the verifier, and not for
/// another state, after a changed byte or under a changed key. The verifier
/// expands no key element when it verifies.
#[test]
#[ignore = "Production-size compression, 30 GB; run separately under the 300-second cap."]
fn hash_chain_compresses_and_verifies_without_the_ajtai_key() {
    let verifier = circuit()
        .verifier(Engine::Optimized, 109)
        .expect("verifier");
    let before = expanded_key_bytes();
    let key = verifier.compression_key().expect("compression key");
    assert_eq!(expanded_key_bytes() - before, KEY_BYTES);
    let key_bytes = key.to_bytes();
    assert!(key_bytes.len() < 2048, "the key has {} bytes", key_bytes.len());
    assert_eq!(
        verifier
            .decode_compression_key(&key_bytes)
            .expect("decoded key"),
        key
    );

    let prover = circuit().prover(Engine::Optimized, 109).expect("prover");
    let before = expanded_key_bytes();
    let setup = prover.compression_setup().expect("compression setup");
    assert_eq!(expanded_key_bytes() - before, KEY_BYTES);
    let initial = words([1, 2, 3, 4]);
    let message = words([5, 6, 7, 8]);
    let mut proof = prover.prove(initial, &message).expect("first step");
    let mut current = poseidon2_hash_chain_step(initial, message);
    // After the first step both slots hold the zero accumulator.
    let first = prover
        .compress(&proof, &setup)
        .expect("compressed first step");
    for _ in 2..=3 {
        proof = prover.extend(&proof, &message).expect("next step");
        current = poseidon2_hash_chain_step(current, message);
    }
    let state = State::new(3, initial, current);
    let compressed = prover.compress(&proof, &setup).expect("compressed proof");
    assert_eq!(compressed.state(), state);
    let bytes = compressed.to_bytes().expect("proof bytes");
    assert!(bytes.len() < 150_000, "the proof has {} bytes", bytes.len());
    assert!(60 * bytes.len() < prover.encode_proof(&proof).expect("bytes").len());
    drop((setup, prover));

    let before = expanded_key_bytes();
    verifier
        .verify_final(
            &State::new(1, initial, poseidon2_hash_chain_step(initial, message)),
            &key,
            &first,
        )
        .expect("the compressed first step verifies");
    let decoded = FinalProof::from_bytes(&bytes).expect("decoded proof");
    verifier
        .verify_final(&state, &key, &decoded)
        .expect("the decoded proof verifies");
    assert!(verifier.verify_final(&state, &key, &first).is_err(), "another proof");

    let mut other = current;
    other[0] += F::ONE;
    assert!(verifier
        .verify_final(&State::new(3, initial, other), &key, &decoded)
        .is_err());
    assert!(verifier
        .verify_final(&State::new(4, initial, current), &key, &decoded)
        .is_err());
    assert!(FinalProof::from_bytes(&bytes[..bytes.len() - 1]).is_err(), "truncated");
    for at in [200, bytes.len() / 2, bytes.len() - 9] {
        let mut changed = bytes.clone();
        changed[at] ^= 1;
        if let Ok(proof) = FinalProof::from_bytes(&changed) {
            assert!(
                verifier.verify_final(&state, &key, &proof).is_err(),
                "changed byte {at}"
            );
        }
    }

    assert!(
        verifier
            .decode_compression_key(&key_bytes[..key_bytes.len() - 1])
            .is_err(),
        "truncated key"
    );
    let mut longer = key_bytes.clone();
    longer.push(0);
    assert!(
        verifier.decode_compression_key(&longer).is_err(),
        "a byte after the key"
    );
    for at in [30, 200, key_bytes.len() - 100] {
        let mut changed = key_bytes.clone();
        changed[at] ^= 1;
        if let Ok(other) = verifier.decode_compression_key(&changed) {
            assert_ne!(other, key, "changed key byte {at}");
            assert!(
                verifier.verify_final(&state, &other, &decoded).is_err(),
                "changed key byte {at}"
            );
        }
    }
    assert_eq!(expanded_key_bytes(), before, "the verifier expanded key elements");
}
