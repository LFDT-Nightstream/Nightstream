//! Public application assembly, base proving and terminal verification without Lean.

use std::{fs, path::PathBuf, time::Instant};

use nightstream::{
    application::{poseidon2_hash_chain_v1, Affine, ApplicationBuilder},
    Circuit, State,
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

    let circuit = Circuit::prepare(&selected_reference(), poseidon2_hash_chain_v1().unwrap()).unwrap();
    eprintln!("poseidon preparation elapsed={:?}", started.elapsed());
    let proving = Instant::now();
    let proof = circuit.prove(initial, &message).unwrap();
    eprintln!("poseidon base proving elapsed={:?}", proving.elapsed());
    let expected = State::new(1, initial, output);
    assert_eq!(proof.state(), &expected);

    let verification = Instant::now();
    circuit.verify(&expected, &proof).unwrap();
    eprintln!("poseidon terminal verification elapsed={:?}", verification.elapsed());
    let mut changed_output = output;
    changed_output[0] += F::ONE;
    assert!(circuit
        .verify(&State::new(1, initial, changed_output), &proof)
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
