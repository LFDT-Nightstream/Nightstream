//! SPEC.md section 6.

mod bounded_depth_support;

use bounded_depth_support::fold_gate_bits;
use neo_ajtai::nightstream_fprime_setup::{expanded_key_bytes, ELEMENT_OUTPUT_BYTES};
use neo_math::F;
use nightstream::application::{poseidon2_hash_chain_step, poseidon2_hash_chain_v1};
use nightstream::{Circuit, FinalProof};
use nightstream::{Engine, PrimeCharacteristicRing, State};

const KEY_BYTES: u64 = 18 * (1 << 19) * ELEMENT_OUTPUT_BYTES as u64;

fn circuit() -> Circuit {
    Circuit::compile(poseidon2_hash_chain_v1().expect("hash chain")).expect("bounded-depth circuit")
}

fn words(values: [u64; 4]) -> [F; 4] {
    values.map(F::from_u64)
}

#[test]
#[ignore = "Production-size compression, 25 GB; scripts/ci_rust_tests.sh runs it alone."]
fn hash_chain_compresses_and_verifies_without_the_ajtai_key() {
    let verifier = circuit()
        .verifier(Engine::Optimized, fold_gate_bits())
        .expect("verifier");
    let before = expanded_key_bytes();
    let key = verifier.compression_key().expect("compression key");
    let key_bytes = key.to_bytes();
    assert!(key_bytes.len() < 2048, "the key has {} bytes", key_bytes.len());
    assert_eq!(
        verifier
            .decode_compression_key(&key_bytes)
            .expect("decoded key"),
        key
    );
    assert_eq!(
        expanded_key_bytes(),
        before,
        "the key of the verifier expanded key elements"
    );

    let prover = circuit()
        .prover(Engine::Optimized, fold_gate_bits())
        .expect("prover");
    let before = expanded_key_bytes();
    let setup = prover.compression_setup().expect("compression setup");
    assert_eq!(expanded_key_bytes() - before, KEY_BYTES);
    let initial = words([1, 2, 3, 4]);
    let message = words([5, 6, 7, 8]);
    let mut proof = prover.prove(initial, &message).expect("first step");
    let mut current = poseidon2_hash_chain_step(initial, message);
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
    for at in 0..key_bytes.len() {
        let mut changed = key_bytes.clone();
        changed[at] ^= 1;
        assert!(
            verifier.decode_compression_key(&changed).is_err(),
            "changed key byte {at}"
        );
    }
    assert_eq!(expanded_key_bytes(), before, "the verifier expanded key elements");
}
