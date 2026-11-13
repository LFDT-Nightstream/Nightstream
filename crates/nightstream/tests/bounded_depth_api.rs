//! The bounded-depth public API end to end on the Rust Poseidon2 hash chain:
//! compile against the two embedded Lean packages, prove and extend, and
//! verify against the expected state.

use std::fs;

use neo_math::F;
use nightstream::application::{poseidon2_hash_chain_step, poseidon2_hash_chain_v1};
use nightstream::{Circuit, FinalProof, Prover};
use nightstream::{Engine, PrimeCharacteristicRing, PrimeField64, State};

fn circuit() -> Circuit {
    Circuit::compile(poseidon2_hash_chain_v1().expect("hash chain")).expect("bounded-depth circuit")
}

fn words(values: [u64; 4]) -> [F; 4] {
    values.map(F::from_u64)
}

/// Three steps, each verified; a proof does not verify for another state.
#[test]
fn hash_chain_proves_and_verifies() {
    let compiled = circuit();
    let path = std::env::temp_dir().join(format!("bounded-depth-circuit-{}", std::process::id()));
    compiled.write(&path).expect("saved application");
    let prover = Prover::load(&path, Engine::Optimized, 109).expect("loaded prover");
    fs::remove_file(&path).expect("remove the saved application");
    let verifier = compiled.verifier(Engine::Optimized, 109).expect("verifier");
    let initial = words([1, 2, 3, 4]);
    let message = words([5, 6, 7, 8]);
    let mut proof = prover.prove(initial, &message).expect("first step");
    let mut current = poseidon2_hash_chain_step(initial, message);
    verifier
        .verify(&State::new(1, initial, current), &proof)
        .expect("one step verifies");
    for step in 2..=3 {
        proof = prover.extend(&proof, &message).expect("next step");
        current = poseidon2_hash_chain_step(current, message);
        assert_eq!(proof.state(), State::new(step, initial, current));
        verifier
            .verify(&State::new(step, initial, current), &proof)
            .expect("the step verifies");
    }
    let mut other = current;
    other[0] += F::ONE;
    assert!(verifier
        .verify(&State::new(3, initial, other), &proof)
        .is_err());

    let bytes = prover.encode_proof(&proof).expect("proof bytes");
    let decoded = verifier.decode_proof(&bytes).expect("decoded proof");
    verifier
        .verify(&State::new(3, initial, current), &decoded)
        .expect("the decoded proof verifies");
    let extended = prover
        .extend(&decoded, &message)
        .expect("a decoded proof extends");
    verifier
        .verify(
            &State::new(4, initial, poseidon2_hash_chain_step(current, message)),
            &extended,
        )
        .expect("the extended proof verifies");
    assert!(verifier.decode_proof(&bytes[..bytes.len() - 1]).is_err(), "truncated");
    let mut changed = bytes.clone();
    let last = changed.len() - 9;
    changed[last] ^= 1;
    if let Ok(proof) = verifier.decode_proof(&changed) {
        assert!(
            verifier
                .verify(&State::new(3, initial, current), &proof)
                .is_err(),
            "changed byte"
        );
    }
    let initial_proof = prover
        .encode_proof(&prover.prove(initial, &message).expect("first step"))
        .expect("bytes");
    assert!(verifier.decode_proof(&initial_proof).is_ok());
}

/// A proof compresses to a proof without witnesses (SPEC.md section 6). It
/// verifies from its bytes for the expected state, and not for another state
/// or after a changed byte. After the first step both slots hold the zero
/// accumulator.
#[test]
fn hash_chain_compresses_and_verifies() {
    let compiled = circuit();
    let prover = compiled.prover(Engine::Optimized, 109).expect("prover");
    let verifier = compiled.verifier(Engine::Optimized, 109).expect("verifier");
    let initial = words([1, 2, 3, 4]);
    let message = words([5, 6, 7, 8]);
    let mut proof = prover.prove(initial, &message).expect("first step");
    let mut current = poseidon2_hash_chain_step(initial, message);
    let first = prover.compress(&proof).expect("compressed first step");
    verifier
        .verify_final(&State::new(1, initial, current), &first)
        .expect("the compressed first step verifies");
    for _ in 2..=3 {
        proof = prover.extend(&proof, &message).expect("next step");
        current = poseidon2_hash_chain_step(current, message);
    }
    let state = State::new(3, initial, current);
    let compressed = prover.compress(&proof).expect("compressed proof");
    assert_eq!(compressed.state(), state);
    let bytes = compressed.to_bytes().expect("proof bytes");
    assert!(40 * bytes.len() < prover.encode_proof(&proof).expect("bytes").len());
    let decoded = FinalProof::from_bytes(&bytes).expect("decoded proof");
    verifier
        .verify_final(&state, &decoded)
        .expect("the decoded proof verifies");
    assert!(verifier.verify_final(&state, &first).is_err(), "another proof");

    let mut other = current;
    other[0] += F::ONE;
    assert!(verifier
        .verify_final(&State::new(3, initial, other), &decoded)
        .is_err());
    assert!(verifier
        .verify_final(&State::new(4, initial, current), &decoded)
        .is_err());
    assert!(FinalProof::from_bytes(&bytes[..bytes.len() - 1]).is_err(), "truncated");
    for at in [200, bytes.len() / 2, bytes.len() - 9] {
        let mut changed = bytes.clone();
        changed[at] ^= 1;
        if let Ok(proof) = FinalProof::from_bytes(&changed) {
            assert!(verifier.verify_final(&state, &proof).is_err(), "changed byte {at}");
        }
    }
}

/// The iteration word and a slot depth have one encoding: `d + q` hashes like
/// `d`, so the decoder refuses a word not below `q` (`SPEC.md` section 4.5).
#[test]
fn a_word_not_below_q_does_not_decode() {
    let compiled = circuit();
    let prover = compiled.prover(Engine::Optimized, 109).expect("prover");
    let verifier = compiled.verifier(Engine::Optimized, 109).expect("verifier");
    let proof = prover
        .prove(words([1, 2, 3, 4]), &words([5, 6, 7, 8]))
        .expect("first step");
    let bytes = prover.encode_proof(&proof).expect("proof bytes");
    for (name, at) in [("iteration", 16 + 8), ("slot 0 depth", 16 + 80)] {
        let word = u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
        let mut aliased = bytes.clone();
        aliased[at..at + 8].copy_from_slice(&(word + F::ORDER_U64).to_le_bytes());
        assert!(verifier.decode_proof(&aliased).is_err(), "{name} word {word} + q");
    }
}

/// The gate charges the refresh fold's zero-query error of the random-oracle
/// result (`RefreshKnowledge.statisticalError 0`, about `2^-109.15` with the
/// census numerator 6377 of `neo-params`; SPEC.md section 3), so a
/// prover or verifier that asks for 110 bits is refused.
#[test]
fn the_gate_refuses_more_bits_than_the_fold_error_gives() {
    let compiled = circuit();
    assert!(compiled.prover(Engine::Optimized, 109).is_ok());
    assert!(compiled.prover(Engine::Optimized, 110).is_err());
    assert!(compiled.verifier(Engine::Optimized, 110).is_err());
}

/// The reference hash chain compiles to Lean's own packages, so the circuit
/// that ships is the one that Lean exports and that the Lean checks of Rust's
/// files read.
#[test]
fn the_reference_hash_chain_keeps_lean_packages() {
    assert_eq!(
        circuit().identity(),
        nightstream_fprime::BOUNDED_DEPTH_PACKAGES_IDENTITY
    );
}
