use crate::application::{poseidon2_hash_chain_step, poseidon2_hash_chain_v1};
use crate::nebula::test_packages::assert_rows;
use neo_ccs::crypto::poseidon2::poseidon2_hash;
use neo_math::Fq;
use p3_field::PrimeCharacteristicRing;

fn step() -> ([Fq; 4], [Fq; 4]) {
    ([10, 11, 12, 13].map(Fq::from_u64), [20, 21, 22, 23].map(Fq::from_u64))
}

/// The Rust-built hash chain has the rows of Lean's `Poseidon2HashChainV1`,
/// row for row in the same variable order (`EmitMain.applicationDigest`,
/// `tests/fixtures/README.md`).
#[test]
fn poseidon2_application_rows_match_lean() {
    assert_rows(
        &poseidon2_hash_chain_v1().unwrap(),
        include_bytes!("../fixtures/poseidon2-hash-chain-v1-digest.json"),
    );
}

#[test]
fn poseidon2_witness_satisfies_the_rows_and_gives_the_native_hash() {
    let circuit = poseidon2_hash_chain_v1().unwrap();
    let (input, message) = step();
    let witness = circuit.execute(input, &message).unwrap();
    circuit.check(witness.values()).unwrap();

    let mut preimage: Vec<_> = b"Nightstream/Stage1/Poseidon2HashChain/v1"
        .iter()
        .map(|byte| Fq::from_u64(u64::from(*byte)))
        .collect();
    preimage.extend(input);
    preimage.extend(message);
    assert_eq!(witness.output_state(), poseidon2_hash(&preimage));
    assert_eq!(witness.output_state(), poseidon2_hash_chain_step(input, message));
}

#[test]
fn poseidon2_constraints_reject_changed_inputs_outputs_and_intermediates() {
    let circuit = poseidon2_hash_chain_v1().unwrap();
    let (input, message) = step();
    let witness = circuit.execute(input, &message).unwrap();
    for variable in [
        circuit.input_state()[0].index(),
        circuit.private_inputs()[0].index(),
        circuit.output_state()[0].index(),
        circuit.generated_range().start,
    ] {
        let mut changed = witness.values().to_vec();
        changed[variable] += Fq::ONE;
        assert!(
            circuit.check(&changed).is_err(),
            "changed application variable {variable}"
        );
    }
}
