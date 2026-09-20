//! Public application assembly, base proving and terminal verification without Lean.

use std::{fs, path::PathBuf, time::Instant};

use nightstream::{
    application::{poseidon2_hash_chain_v1, Affine, ApplicationBuilder},
    Circuit, Engine, State,
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
#[ignore = "Full production Metal lifecycle; apply the 300-second cap unless the owner approves a longer invocation."]
fn poseidon_metal_recursive_lifecycle() {
    poseidon_lifecycle(Engine::Metal, true);
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
    let initial: [u64; 4] = private[30..34].try_into().unwrap();
    let message: [u64; 4] = private[private.len() - 4..].try_into().unwrap();
    let recorded_output: [u64; 4] = serde_json::from_value(reference[4][0].clone()).unwrap();
    let initial = initial.map(F::from_u64);
    let message = message.map(F::from_u64);
    let output = recorded_output.map(F::from_u64);

    let circuit =
        Circuit::prepare_with_engine(&selected_reference(), poseidon2_hash_chain_v1().unwrap(), engine).unwrap();
    assert_eq!(circuit.engine(), engine);
    eprintln!("poseidon preparation engine={engine:?} elapsed={:?}", started.elapsed());
    let proving = Instant::now();
    let mut proof = circuit.prove(initial, &message).unwrap();
    eprintln!("poseidon base proving elapsed={:?}", proving.elapsed());
    let mut expected = State::new(1, initial, output);
    assert_eq!(proof.state(), &expected);

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
            proof = circuit.extend(proof, &message).unwrap();
            eprintln!("poseidon public extend step={step} elapsed={:?}", extending.elapsed());
            expected = State::new(step, initial, next);
            assert_eq!(proof.state(), &expected);
        }
    }

    let verification = Instant::now();
    circuit.verify(&expected, &proof).unwrap();
    eprintln!("poseidon terminal verification elapsed={:?}", verification.elapsed());
    let mut changed_output = expected.current();
    changed_output[0] += F::ONE;
    assert!(circuit
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
    let circuit = Circuit::prepare(&selected_reference(), builder.finish(outputs).unwrap()).unwrap();
    eprintln!("addition preparation elapsed={:?}", started.elapsed());
    let initial = [1, 2, 3, 4].map(F::from_u64);
    let private = [5, 6, 7, 8].map(F::from_u64);
    let expected = State::new(1, initial, [6, 8, 10, 12].map(F::from_u64));

    let proving = Instant::now();
    let proof = circuit.prove(initial, &private).unwrap();
    eprintln!("addition base proving elapsed={:?}", proving.elapsed());
    assert_eq!(proof.state(), &expected);
    let verification = Instant::now();
    circuit.verify(&expected, &proof).unwrap();
    eprintln!("addition terminal verification elapsed={:?}", verification.elapsed());
    eprintln!("addition public lifecycle elapsed={:?}", started.elapsed());
}
