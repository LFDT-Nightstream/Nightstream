//! The option-B public API end to end on the Rust Poseidon2 hash chain:
//! compile against the three Lean packages, prove and extend, verify against
//! the expected state, and finish and verify the final proof.

use std::fs;
use std::path::PathBuf;

use nightstream::application::{poseidon2_hash_chain_step, poseidon2_hash_chain_v1};
use nightstream::option_b::{Circuit, References};
use nightstream::{PrimeCharacteristicRing, State};
use p3_goldilocks::Goldilocks as F;

fn artifact(name: &str) -> Vec<u8> {
    fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../formal/nightstream-fprime/artifacts")
            .join(name),
    )
    .unwrap_or_else(|error| panic!("read {name}: {error}"))
}

fn circuit() -> Circuit {
    let (light, refresh, finish) = (
        artifact("nightstream-fprime-option-b-light-v1.json"),
        artifact("nightstream-fprime-option-b-refresh-v1.json"),
        artifact("nightstream-fprime-option-b-finish-v1.json"),
    );
    Circuit::compile(
        References {
            light: &light,
            refresh: &refresh,
            finish: &finish,
        },
        poseidon2_hash_chain_v1().expect("hash chain"),
    )
    .expect("option-B circuit")
}

fn words(values: [u64; 4]) -> [F; 4] {
    values.map(F::from_u64)
}

/// Three steps, each verified; a proof does not verify for another state.
#[test]
fn hash_chain_proves_and_verifies() {
    let circuit = circuit();
    let prover = circuit.prover(114).expect("prover");
    let verifier = circuit.verifier(114).expect("verifier");
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
}

/// The final proof of two steps verifies with the verifier's own key, and
/// not for another state.
#[test]
fn hash_chain_final_proof_verifies() {
    let circuit = circuit();
    let prover = circuit.prover(114).expect("prover");
    let initial = words([1, 2, 3, 4]);
    let message = words([5, 6, 7, 8]);
    let proof = prover
        .extend(&prover.prove(initial, &message).expect("first step"), &message)
        .expect("second step");
    let dir = std::env::temp_dir().join(format!("option-b-api-{}", std::process::id()));
    fs::create_dir_all(&dir).expect("setup directory");
    let final_proof = {
        let setup = prover.compression_setup(&dir).expect("setup");
        prover.finish(&proof, &setup).expect("final proof")
    };
    fs::remove_dir_all(&dir).expect("remove the setup");
    let verifier = circuit.verifier(114).expect("verifier");
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
