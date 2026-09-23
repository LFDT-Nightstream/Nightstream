use std::{fs, path::PathBuf};

use nightstream::{
    application::{poseidon2_hash_chain_v1, Affine, ApplicationBuilder},
    assembly,
};
use nightstream_fprime::{
    load_poseidon2_hash_chain_v1_package, POSEIDON2_HASH_CHAIN_V1_PACKAGE_IDENTITY,
    POSEIDON2_HASH_CHAIN_V1_STRUCTURAL_IDENTIFIER, POSEIDON2_HASH_CHAIN_V1_VERIFICATION_KEY_DIGEST,
};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

fn reference() -> Vec<u8> {
    fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json"),
    )
    .unwrap()
}

#[test]
fn independent_poseidon_assembly_preserves_selected_package_key_and_matrix_rows() {
    let bytes = reference();
    let application = poseidon2_hash_chain_v1().unwrap();
    let (actual, binding) = assembly::prepare(&bytes, &application).unwrap();
    assert_eq!(
        binding.structural_identifier(),
        POSEIDON2_HASH_CHAIN_V1_STRUCTURAL_IDENTIFIER
    );
    assert_eq!(binding.package_identity(), POSEIDON2_HASH_CHAIN_V1_PACKAGE_IDENTITY);
    assert_eq!(
        binding.verification_key_digest(),
        POSEIDON2_HASH_CHAIN_V1_VERIFICATION_KEY_DIGEST
    );
    let expected = load_poseidon2_hash_chain_v1_package(&bytes).unwrap();
    assert_eq!(actual.application(), expected.application());
    assert_eq!(actual.assignment_plan(), expected.assignment_plan());
    assert_eq!(actual.ccs_relation(), expected.ccs_relation());
    // Compare the application and final boundary as ordered sparse rows.
    let mut expected_rows = Vec::new();
    let application_rows = expected.application().row_range();
    // Physical application rows are also checked byte-for-byte by the internal encoding test.
    assert_eq!(application_rows.len(), 7700);
    let boundary_rows = actual.next_preimage_row_range().len() + application.output_state().len();
    let logical_application_rows = 262;
    let rows = actual.row_count() - logical_application_rows - boundary_rows..actual.row_count();
    expected
        .visit_matrix_rows(rows.clone(), |index, row| {
            expected_rows.push((index, row));
            Ok(())
        })
        .unwrap();
    let mut position = 0;
    actual
        .visit_matrix_rows(rows, |index, row| {
            assert_eq!((index, row), expected_rows[position]);
            position += 1;
            Ok(())
        })
        .unwrap();
    assert_eq!(position, expected_rows.len());
}

#[test]
fn another_rust_application_assembles_with_its_own_binding() {
    let mut builder = ApplicationBuilder::new(4).unwrap();
    let input = builder.input_state();
    let private = builder.private_inputs().to_vec();
    let mut outputs = std::array::from_fn(|_| Affine::constant(Goldilocks::ZERO));
    for lane in 0..4 {
        outputs[lane] = builder
            .affine(Affine::from(input[lane]) + Affine::from(private[lane]))
            .unwrap()
            .into();
    }
    let application = builder.finish(outputs).unwrap();
    let (package, binding) = assembly::prepare(&reference(), &application).unwrap();
    assert_eq!(package.application().witness_word_count(), 4);
    assert_eq!(package.application().private_range().len(), 4);
    assert_eq!(package.application().row_range().len(), 8);
    assert_ne!(binding.package_identity(), POSEIDON2_HASH_CHAIN_V1_PACKAGE_IDENTITY);
    assert_ne!(
        binding.verification_key_digest(),
        POSEIDON2_HASH_CHAIN_V1_VERIFICATION_KEY_DIGEST
    );
    let witness = application
        .execute([Goldilocks::ONE; 4], &[Goldilocks::ONE; 4])
        .unwrap();
    assert_eq!(witness.output_state(), [Goldilocks::from_u64(2); 4]);
}
