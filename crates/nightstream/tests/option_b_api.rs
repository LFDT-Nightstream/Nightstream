//! The option-B public API end to end on the Rust Poseidon2 hash chain:
//! compile against the three embedded Lean packages, prove and extend, verify
//! against the expected state, and compress and verify the final proof.

use std::fs;

use nightstream::application::{poseidon2_hash_chain_step, poseidon2_hash_chain_v1};
use nightstream::{Circuit, Prover};
use nightstream::{Engine, PrimeCharacteristicRing, PrimeField64, State};
use p3_goldilocks::Goldilocks as F;

fn circuit() -> Circuit {
    Circuit::compile(poseidon2_hash_chain_v1().expect("hash chain")).expect("option-B circuit")
}

fn words(values: [u64; 4]) -> [F; 4] {
    values.map(F::from_u64)
}

/// Three steps, each verified; a proof does not verify for another state.
#[test]
fn hash_chain_proves_and_verifies() {
    let compiled = circuit();
    let path = std::env::temp_dir().join(format!("option-b-circuit-{}", std::process::id()));
    compiled.write(&path).expect("saved application");
    let prover = Prover::load(&path, Engine::Optimized, 109).expect("loaded prover");
    fs::remove_file(&path).expect("remove the saved application");
    let verifier = compiled.verifier(Engine::Optimized, 109).expect("verifier");
    assert!(
        compiled.prover(Engine::Metal, 109).is_err(),
        "option B proves on the CPU path"
    );
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
/// result (`RefreshKnowledge.statisticalError 0`, about `2^-109.6`), so a
/// prover or verifier that asks for 110 bits is refused.
#[test]
fn the_gate_refuses_more_bits_than_the_fold_error_gives() {
    let compiled = circuit();
    assert!(compiled.prover(Engine::Optimized, 109).is_ok());
    assert!(compiled.prover(Engine::Optimized, 110).is_err());
    assert!(compiled.verifier(Engine::Optimized, 110).is_err());
}

/// Removes a setup folder when it drops, so a failed run frees its disk too.
struct SetupDir(std::path::PathBuf);

impl Drop for SetupDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The final proof of two steps verifies with the verifier's own key, and
/// not for another state.
#[test]
fn hash_chain_final_proof_verifies() {
    let circuit = circuit();
    let prover = circuit.prover(Engine::Optimized, 109).expect("prover");
    let initial = words([1, 2, 3, 4]);
    let message = words([5, 6, 7, 8]);
    let proof = prover
        .extend(&prover.prove(initial, &message).expect("first step"), &message)
        .expect("second step");
    let dir = SetupDir(std::env::temp_dir().join(format!("option-b-api-{}", std::process::id())));
    fs::create_dir_all(&dir.0).expect("setup directory");
    let final_proof = {
        let setup = prover.compression_setup(&dir.0).expect("setup");
        prover.compress(&proof, &setup).expect("final proof")
    };
    drop(dir);
    let verifier = circuit.verifier(Engine::Optimized, 109).expect("verifier");
    let key = verifier.compression_key().expect("key");
    let state = proof.state();
    verifier
        .verify_final(&state, &key, &final_proof)
        .expect("the final proof verifies");
    let mut other = state.current();
    other[0] += F::ONE;
    assert!(verifier
        .verify_final(&State::new(state.iteration(), state.z0(), other), &key, &final_proof)
        .is_err());
}
