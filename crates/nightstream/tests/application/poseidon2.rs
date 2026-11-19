use crate::application::{
    check_blocks_match_permutation_rows, poseidon2_hash_chain_step, poseidon2_hash_chain_v1, Affine, ApplicationBuilder,
};
use crate::nebula::test_packages::assert_rows;
use neo_ccs::crypto::poseidon2::poseidon2_hash;
use neo_math::Fq;
use p3_field::PrimeCharacteristicRing;

fn step() -> ([Fq; 4], [Fq; 4]) {
    ([10, 11, 12, 13].map(Fq::from_u64), [20, 21, 22, 23].map(Fq::from_u64))
}

/// Lean `Poseidon2HashChainV1` and `EmitMain.applicationDigest`.
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
    check_blocks_match_permutation_rows(&circuit).unwrap();
    let compiled = circuit.execute_compiled(input, &message).unwrap();
    assert_eq!(compiled.values(), witness.values());
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
fn compiled_poseidon2_writes_every_local_with_ordinary_recipes_between_blocks() {
    let mut builder = ApplicationBuilder::new(1).unwrap();
    let input = builder.input_state().map(Affine::from);
    let private = Affine::from(builder.private_inputs()[0]);
    let product = builder.multiply(input[0].clone(), private.clone()).unwrap();
    let first = builder
        .poseidon2_hash(&[product.into(), input[1].clone()])
        .unwrap();
    let next = builder
        .multiply(first[2].clone(), input[2].clone())
        .unwrap();
    let second = builder
        .poseidon2_hash(&[next.into(), first[3].clone(), private])
        .unwrap();
    let tail = builder
        .affine(second[0].clone() + input[3].clone())
        .unwrap();
    let circuit = builder
        .finish([tail.into(), second[1].clone(), second[2].clone(), second[3].clone()])
        .unwrap();
    check_blocks_match_permutation_rows(&circuit).unwrap();
    for (input, private) in [([Fq::ZERO; 4], Fq::ZERO), ([Fq::NEG_ONE; 4], Fq::TWO)] {
        let generic = circuit.execute(input, &[private]).unwrap();
        let compiled = circuit.execute_compiled(input, &[private]).unwrap();
        assert_eq!(compiled.values(), generic.values());
        assert_eq!(compiled.output_state(), generic.output_state());
    }
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
