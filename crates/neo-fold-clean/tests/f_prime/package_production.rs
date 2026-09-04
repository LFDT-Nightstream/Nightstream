use std::fs;
use std::path::PathBuf;

use neo_fold_clean::Poseidon2HashChainV1Package;
use nightstream_fprime::{
    POSEIDON2_HASH_CHAIN_V1_PACKAGE_IDENTITY, POSEIDON2_HASH_CHAIN_V1_STRUCTURAL_IDENTIFIER,
    POSEIDON2_HASH_CHAIN_V1_VERIFICATION_KEY_DIGEST,
};

fn package_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json")
}

fn canonical_bytes(value: &serde_json::Value) -> Vec<u8> {
    let mut bytes = serde_json::to_vec(value).expect("canonical package mutation");
    bytes.push(b'\n');
    bytes
}

fn canonical_pin_entry(package: &mut serde_json::Value) -> &mut Vec<serde_json::Value> {
    package
        .as_array_mut()
        .and_then(|sealed| sealed.get_mut(2))
        .and_then(serde_json::Value::as_array_mut)
        .and_then(|blocks| blocks.get_mut(3))
        .and_then(serde_json::Value::as_array_mut)
        .and_then(|block| block.get_mut(1))
        .and_then(serde_json::Value::as_array_mut)
        .and_then(|pin| pin.get_mut(1))
        .and_then(serde_json::Value::as_array_mut)
        .and_then(|rows| rows.get_mut(760))
        .and_then(serde_json::Value::as_array_mut)
        .and_then(|row| row.first_mut())
        .and_then(serde_json::Value::as_array_mut)
        .expect("canonical final matrix pin entry")
}

#[test]
fn production_package_pins_relation_and_key() {
    let bytes = fs::read(package_path()).expect("canonical Lean package");
    let package = Poseidon2HashChainV1Package::load(&bytes).expect("verifier-owned production package");

    assert_eq!(
        package.structural_identifier(),
        POSEIDON2_HASH_CHAIN_V1_STRUCTURAL_IDENTIFIER
    );
    assert_eq!(package.package_identity(), POSEIDON2_HASH_CHAIN_V1_PACKAGE_IDENTITY);
    assert_eq!(
        package.verification_key_digest(),
        POSEIDON2_HASH_CHAIN_V1_VERIFICATION_KEY_DIGEST
    );
    assert!(package.structure().is_verifier_artifact_header());
    assert_eq!(package.structure().t(), 14);
    let mut visited = 0;
    package
        .visit_matrix_rows(0..1, |ordinal, row| {
            assert_eq!(ordinal, 0);
            assert!(row.matrix(0).is_some());
            visited += 1;
            Ok(())
        })
        .expect("first Lean matrix row");
    assert_eq!(visited, 1);
}

#[test]
fn production_package_rejects_canonical_mutation() {
    let bytes = fs::read(package_path()).expect("canonical Lean package");
    let mut value: serde_json::Value = serde_json::from_slice(&bytes).expect("package JSON");
    let schema = value
        .as_array_mut()
        .and_then(|sealed| sealed.first_mut())
        .and_then(|value| value.as_u64())
        .expect("sealed schema");
    value.as_array_mut().expect("sealed package")[0] = serde_json::Value::from(schema + 1);
    assert!(Poseidon2HashChainV1Package::load(&canonical_bytes(&value)).is_err());
}

#[test]
fn production_package_rejects_matrix_row_column_and_coefficient_mutations() {
    let bytes = fs::read(package_path()).expect("canonical Lean package");
    let value: serde_json::Value = serde_json::from_slice(&bytes).expect("package JSON");

    let mut changed_row_order = value.clone();
    let blocks = changed_row_order
        .as_array_mut()
        .and_then(|sealed| sealed.get_mut(2))
        .and_then(serde_json::Value::as_array_mut)
        .expect("final matrix blocks");
    assert_ne!(blocks[0], blocks[1], "distinct final matrix blocks");
    blocks.swap(0, 1);
    assert!(Poseidon2HashChainV1Package::load(&canonical_bytes(&changed_row_order)).is_err());

    let mut changed_column = value.clone();
    let entry = canonical_pin_entry(&mut changed_column);
    assert_eq!(entry[0].as_u64(), Some(196_202_984));
    entry[0] = serde_json::Value::from(196_202_985_u64);
    assert!(Poseidon2HashChainV1Package::load(&canonical_bytes(&changed_column)).is_err());

    let mut changed_coefficient = value;
    let entry = canonical_pin_entry(&mut changed_coefficient);
    assert_eq!(entry[1].as_u64(), Some(1));
    entry[1] = serde_json::Value::from(2_u64);
    assert!(Poseidon2HashChainV1Package::load(&canonical_bytes(&changed_coefficient)).is_err());
}
