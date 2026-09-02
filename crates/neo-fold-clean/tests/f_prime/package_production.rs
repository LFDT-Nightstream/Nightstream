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
    package.matrix_row(0).expect("first Lean matrix row");
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
    let mut mutated = serde_json::to_vec(&value).expect("canonical mutation");
    mutated.push(b'\n');

    assert!(Poseidon2HashChainV1Package::load(&mutated).is_err());
}
