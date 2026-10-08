//! Public application assembly, base proving and terminal verification without Lean.

use std::{fs, path::PathBuf, time::Instant};

use nightstream::{
    application::{poseidon2_hash_chain_v1, Affine, ApplicationBuilder},
    Circuit, CompressionKey, Engine, FinalProof, State, Verifier,
};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;
use serde_json::Value;

fn artifact(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/lean")
        .join(name)
}

fn selected_reference() -> Vec<u8> {
    fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json"),
    )
    .expect("saved selected circuit reference")
}

#[test]
#[ignore = "Full production-profile check; run this test separately under the 300-second cap."]
fn poseidon_base_step_matches_lean_and_verifies() {
    poseidon_lifecycle(Engine::Optimized, false);
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "Full production base lifecycle with Metal terminal rows; run separately under the 300-second cap."]
fn poseidon_metal_base_step_matches_lean_and_verifies() {
    poseidon_lifecycle(Engine::Metal, false);
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "Full production Metal lifecycle; apply the 300-second cap unless the owner approves a longer invocation."]
fn poseidon_metal_recursive_lifecycle() {
    poseidon_lifecycle(Engine::Metal, true);
}

/// Device proving changes no proof byte: Metal proofs equal the CPU engine's
/// after the base step and after each of two folds.
#[cfg(feature = "metal")]
#[test]
#[ignore = "Full production Metal and CPU lifecycles; run this test separately under the 300-second cap."]
fn poseidon_metal_proofs_equal_cpu_proofs() {
    let circuit = Circuit::compile(&selected_reference(), poseidon2_hash_chain_v1().unwrap()).unwrap();
    let initial = [202, 203, 204, 205].map(F::from_u64);
    let message = [7, 11, 13, 17].map(F::from_u64);
    let proofs = |engine| {
        let prover = circuit.prover(engine, 114).unwrap();
        let mut proof = prover.prove(initial, &message).unwrap();
        let mut bytes = vec![prover.encode_proof(&proof).unwrap()];
        for _ in 0..2 {
            proof = prover.extend(&proof, &message).unwrap();
            bytes.push(prover.encode_proof(&proof).unwrap());
        }
        bytes
    };
    let metal = proofs(Engine::Metal);
    assert!(
        metal == proofs(Engine::Optimized),
        "Metal proof bytes differ from the CPU engine"
    );
}

/// The production compression setup, written once by
/// `poseidon_compression_setup_matches_the_derived_key` and reused by
/// `poseidon_finish_with_spartan_verifies`.
fn compression_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/compression-setup")
}

/// Writes the production setup files (about 29 GB) and its key, and checks
/// that the key equals the one a verifier derives without files.
#[test]
#[ignore = "Writes about 29 GB under target/compression-setup; run this test separately under the 300-second cap."]
fn poseidon_compression_setup_matches_the_derived_key() {
    let started = Instant::now();
    let circuit = Circuit::compile(&selected_reference(), poseidon2_hash_chain_v1().unwrap()).unwrap();
    eprintln!("compile elapsed={:?}", started.elapsed());
    let dir = compression_dir();
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let prover = circuit.prover(Engine::Optimized, 114).unwrap();
    let building = Instant::now();
    let setup = prover.compression_setup(&dir).unwrap();
    eprintln!("compression setup elapsed={:?}", building.elapsed());
    let key = setup.key().to_bytes();
    fs::write(dir.join("key.bin"), &key).unwrap();
    let deriving = Instant::now();
    let verifier = Verifier::from_package(&circuit, Engine::Optimized, 114).unwrap();
    assert!(verifier.compression_key().unwrap() == *setup.key());
    eprintln!(
        "key derivation elapsed={:?}, key bytes={}",
        deriving.elapsed(),
        key.len()
    );
}

/// Compression after one fold with the stored setup: the final proof carries
/// no witness, verifies with only the key after a byte round trip, and
/// rejects a wrong state, a changed byte or the key of another setup root.
#[test]
#[ignore = "Full production compression; needs the stored setup; run this test separately under the 300-second cap."]
fn poseidon_finish_with_spartan_verifies() {
    let started = Instant::now();
    let circuit = Circuit::compile(&selected_reference(), poseidon2_hash_chain_v1().unwrap()).unwrap();
    let prover = circuit.prover(Engine::Optimized, 114).unwrap();
    let verifier = Verifier::from_package(&circuit, Engine::Optimized, 114).unwrap();
    let key = CompressionKey::from_bytes(&fs::read(compression_dir().join("key.bin")).unwrap()).unwrap();
    let setup = prover
        .open_compression_setup(compression_dir(), &key)
        .unwrap();
    let initial = [202, 203, 204, 205].map(F::from_u64);
    let message = [7, 11, 13, 17].map(F::from_u64);
    let proof = prover.prove(initial, &message).unwrap();
    let proof = prover.extend(&proof, &message).unwrap();
    eprintln!("compile, base step and one fold elapsed={:?}", started.elapsed());

    let finishing = Instant::now();
    let finished = prover.finish_with_spartan(&proof, &setup).unwrap();
    eprintln!("finish_with_spartan elapsed={:?}", finishing.elapsed());
    let bytes = finished.to_bytes();
    eprintln!(
        "final proof bytes={} (accumulator proof bytes={})",
        bytes.len(),
        prover.encode_proof(&proof).unwrap().len()
    );
    let decoded = FinalProof::from_bytes(&bytes).unwrap();
    let verifying = Instant::now();
    verifier
        .verify_final(proof.state(), &key, &decoded)
        .unwrap();
    eprintln!("final verification elapsed={:?}", verifying.elapsed());

    let mut changed = proof.state().current();
    changed[0] += F::ONE;
    let wrong = State::new(proof.state().iteration(), initial, changed);
    assert!(verifier.verify_final(&wrong, &key, &decoded).is_err());
    let mut flipped = bytes.clone();
    let index = bytes.len() - 100;
    flipped[index] ^= 1;
    assert!(FinalProof::from_bytes(&flipped).map_or(true, |changed| verifier
        .verify_final(proof.state(), &key, &changed)
        .is_err()));
    let mut other = key.to_bytes();
    other[0] ^= 1;
    let other = CompressionKey::from_bytes(&other).unwrap();
    assert!(verifier
        .verify_final(proof.state(), &other, &decoded)
        .is_err());
}

fn poseidon_lifecycle(engine: Engine, recursive: bool) {
    let started = Instant::now();
    let reference: Value = serde_json::from_slice(
        &fs::read(artifact("nightstream-fprime-stage1-base-step-fixture-v1.json"))
            .expect("saved Lean base-step result"),
    )
    .unwrap();
    assert_eq!(reference[0], 1);
    let private: Vec<u64> = serde_json::from_value(reference[2].clone()).unwrap();
    // These are the state and application-advice fields of the saved caller packet.
    // `z0` sits in the prior preimage tail `vk, i, z0, zi`.
    let initial: [u64; 4] = private[27_811..27_815].try_into().unwrap();
    let message: [u64; 4] = private[private.len() - 4..].try_into().unwrap();
    let recorded_output: [u64; 4] = serde_json::from_value(reference[4][0].clone()).unwrap();
    let initial = initial.map(F::from_u64);
    let message = message.map(F::from_u64);
    let output = recorded_output.map(F::from_u64);

    let circuit = Circuit::compile(&selected_reference(), poseidon2_hash_chain_v1().unwrap()).unwrap();
    let prover = circuit.prover(engine, 114).unwrap();
    let verifier = Verifier::from_package(&circuit, engine, 114).unwrap();
    assert_eq!(prover.engine(), engine);
    eprintln!("poseidon preparation engine={engine:?} elapsed={:?}", started.elapsed());
    let proving = Instant::now();
    let mut proof = prover.prove(initial, &message).unwrap();
    eprintln!("poseidon base proving elapsed={:?}", proving.elapsed());
    let mut expected = State::new(1, initial, output);
    assert_eq!(proof.state(), &expected);

    assert!(prover.extend(&proof, &message[..3]).is_err());
    assert_eq!(
        proof.state(),
        &expected,
        "failed extension retains the valid prior proof"
    );

    if recursive {
        let fixture: Value = serde_json::from_slice(
            &fs::read(
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("tests/fixtures/stage1_recursive_states/nonzero-running.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let message: [u64; 4] = serde_json::from_value(fixture[3].clone()).unwrap();
        let message = message.map(F::from_u64);
        let recorded_second: [u64; 4] = serde_json::from_value(fixture[2].clone()).unwrap();
        let application = poseidon2_hash_chain_v1().unwrap();
        for step in 2..=3 {
            let next = application
                .execute(expected.current(), &message)
                .unwrap()
                .output_state();
            if step == 2 {
                assert_eq!(next, recorded_second.map(F::from_u64));
            }
            let extending = Instant::now();
            eprintln!("poseidon public extend step={step} started");
            proof = prover.extend(&proof, &message).unwrap();
            eprintln!("poseidon public extend step={step} elapsed={:?}", extending.elapsed());
            expected = State::new(step, initial, next);
            assert_eq!(proof.state(), &expected);
        }
    }

    let verification = Instant::now();
    verifier.verify(&expected, &proof).unwrap();
    eprintln!("poseidon terminal verification elapsed={:?}", verification.elapsed());
    let mut changed_output = expected.current();
    changed_output[0] += F::ONE;
    assert!(verifier
        .verify(&State::new(expected.iteration(), initial, changed_output), &proof)
        .is_err());
    eprintln!("poseidon public lifecycle elapsed={:?}", started.elapsed());
}

#[test]
#[ignore = "Full production-profile check; run this test separately under the 300-second cap."]
fn rust_addition_base_step_verifies() {
    let started = Instant::now();
    let mut builder = ApplicationBuilder::new(4).unwrap();
    let input = builder.input_state();
    let private = builder.private_inputs().to_vec();
    let mut outputs = std::array::from_fn(|_| Affine::constant(F::ZERO));
    for lane in 0..4 {
        outputs[lane] = builder
            .affine(Affine::from(input[lane]) + Affine::from(private[lane]))
            .unwrap()
            .into();
    }
    let circuit = Circuit::compile(&selected_reference(), builder.finish(outputs).unwrap()).unwrap();
    let prover = circuit.prover(Engine::Optimized, 114).unwrap();
    let verifier = Verifier::from_package(&circuit, Engine::Optimized, 114).unwrap();
    eprintln!("addition preparation elapsed={:?}", started.elapsed());
    let initial = [1, 2, 3, 4].map(F::from_u64);
    let private = [5, 6, 7, 8].map(F::from_u64);
    let expected = State::new(1, initial, [6, 8, 10, 12].map(F::from_u64));

    let proving = Instant::now();
    let proof = prover.prove(initial, &private).unwrap();
    eprintln!("addition base proving elapsed={:?}", proving.elapsed());
    assert_eq!(proof.state(), &expected);
    let verification = Instant::now();
    verifier.verify(&expected, &proof).unwrap();
    eprintln!("addition terminal verification elapsed={:?}", verification.elapsed());
    eprintln!("addition public lifecycle elapsed={:?}", started.elapsed());
}
