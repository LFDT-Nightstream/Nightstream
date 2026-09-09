//! Independent reconstruction of the verifier-owned Stage 1 binding.

use std::{fs, path::PathBuf};

use neo_ajtai::nightstream_fprime_setup::{
    PRODUCTION_MESSAGE_COLUMNS, PRODUCTION_SEED, PRODUCTION_VERIFIER_ROWS, SETUP_ID,
};
use neo_ccs::crypto::poseidon2_goldilocks as poseidon2;
use nightstream_fprime::{
    load_poseidon2_hash_chain_v1_package, POSEIDON2_HASH_CHAIN_V1_PACKAGE_IDENTITY,
    POSEIDON2_HASH_CHAIN_V1_STRUCTURAL_IDENTIFIER, POSEIDON2_HASH_CHAIN_V1_VERIFICATION_KEY_DIGEST,
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use serde::Deserialize;
use serde_json::Value;

const PROFILE: [u64; 14] = [4_294_967_295, 1, 2, 16, 65_536, 1, 16, 17, 16, 14, 28, 9, 54, 22];
const SCHEDULE: [u64; 10] = [1, 1, 1, 28, 10, 17, 14, 54, 16, 64];

fn artifact_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "../../formal/nightstream-fprime/artifacts/\
         nightstream-fprime-stage1-poseidon2-hash-chain-v1.json",
    )
}

fn binding_artifact_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "../../formal/nightstream-fprime/artifacts/\
         nightstream-fprime-stage1-poseidon2-hash-chain-v1-binding-v1.json",
    )
}

#[derive(Deserialize)]
struct RawLeanBinding(u64, [u64; 4], [u64; 4], Vec<u64>, Vec<u64>, [u64; 4]);

struct LeanBindingFixture {
    structural_identifier: [u64; 4],
    package_identity: [u64; 4],
    descriptor_words: Vec<u64>,
    binding_words: Vec<u64>,
    verification_key_digest: [u64; 4],
}

fn read_lean_binding() -> LeanBindingFixture {
    let bytes = fs::read(binding_artifact_path()).expect("Lean binding fixture");
    let RawLeanBinding(
        schema,
        structural_identifier,
        package_identity,
        descriptor_words,
        binding_words,
        verification_key_digest,
    ) = serde_json::from_slice(&bytes).expect("strict Lean binding fixture");
    assert_eq!(schema, 1, "Lean binding fixture schema");
    assert_eq!(descriptor_words.len(), 86, "Lean descriptor length");
    assert_eq!(binding_words.len(), 126, "Lean binding length");
    LeanBindingFixture {
        structural_identifier,
        package_identity,
        descriptor_words,
        binding_words,
        verification_key_digest,
    }
}

fn words(bytes: &[u8]) -> Vec<u64> {
    bytes.iter().copied().map(u64::from).collect()
}

fn framed(values: &[u64]) -> Vec<u64> {
    let mut result = Vec::with_capacity(values.len() + 1);
    result.push(values.len() as u64);
    result.extend_from_slice(values);
    result
}

fn hash(values: &[u64]) -> [u64; 4] {
    let fields = values
        .iter()
        .copied()
        .map(Goldilocks::from_u64)
        .collect::<Vec<_>>();
    poseidon2::poseidon2_hash(&fields).map(|value| value.as_canonical_u64())
}

fn component(component: u64, authority: &[u64]) -> [u64; 4] {
    let mut preimage = words(b"Nightstream/FPrime/context/v1_1");
    preimage.push(component);
    preimage.extend(framed(authority));
    hash(&preimage)
}

fn append_value_preimage(value: &Value, output: &mut Vec<u64>) {
    match value {
        Value::Number(number) => {
            let value = number.as_u64().expect("canonical natural-number atom");
            output.extend([0, value & 0xffff_ffff, value >> 32, 0]);
        }
        Value::Array(values) => {
            let length = u64::try_from(values.len()).expect("array length");
            output.extend([1, length & 0xffff_ffff, length >> 32, 0]);
            for child in values {
                append_value_preimage(child, output);
            }
        }
        _ => panic!("canonical package must contain only numbers and arrays"),
    }
}

fn setup_authority(seed: &[u8; 32]) -> Vec<u64> {
    let mut result = Vec::with_capacity(73);
    result.push(SETUP_ID.len() as u64);
    result.extend(SETUP_ID.iter().copied().map(u64::from));
    result.extend([PRODUCTION_VERIFIER_ROWS, PRODUCTION_MESSAGE_COLUMNS, seed.len() as u64]);
    result.extend(seed.iter().copied().map(u64::from));
    result
}

fn independent_structural_identifier(sealed: &Value) -> [u64; 4] {
    let mut preimage = words(b"Nightstream/FPrime/package/v2");
    append_value_preimage(sealed, &mut preimage);
    hash(&preimage)
}

struct IndependentBinding {
    structural_identifier: [u64; 4],
    component_digests: [[u64; 4]; 4],
    package_identity: [u64; 4],
    context_digest: [u64; 4],
    descriptor_words: Vec<u64>,
    binding_words: Vec<u64>,
    verification_key_digest: [u64; 4],
    components: [Vec<u64>; 4],
}

struct RebuiltBinding {
    package_identity: [u64; 4],
    context_digest: [u64; 4],
    descriptor_words: Vec<u64>,
    binding_words: Vec<u64>,
    verification_key_digest: [u64; 4],
}

fn rebuild_binding(structural_identifier: [u64; 4], component_digests: [[u64; 4]; 4]) -> RebuiltBinding {
    let mut descriptor = words(b"Nightstream/FPrime/verifier-context/v1_1");
    descriptor.extend(framed(&PROFILE));
    descriptor.extend(framed(&SCHEDULE));
    for digest in component_digests {
        descriptor.extend(framed(&digest));
    }
    let context_digest = hash(&descriptor);

    let mut package_preimage = words(b"Nightstream/FPrime/sealed-package/v2");
    package_preimage.extend(framed(&structural_identifier));
    package_preimage.extend(framed(&descriptor));
    let package_identity = hash(&package_preimage);

    let mut verification_key_words = words(b"Nightstream/FPrime/verifier-key/v1");
    verification_key_words.extend(framed(&package_identity));
    verification_key_words.extend(framed(&descriptor));
    let verification_key_digest = hash(&verification_key_words);

    RebuiltBinding {
        package_identity,
        context_digest,
        descriptor_words: descriptor,
        binding_words: verification_key_words,
        verification_key_digest,
    }
}

fn independent_binding(sealed: &Value, seed: &[u8; 32]) -> IndependentBinding {
    let structural_identifier = independent_structural_identifier(sealed);
    let sealed = sealed.as_array().expect("sealed package tuple");
    let relation = sealed[1]
        .as_array()
        .and_then(|package| package.get(4))
        .expect("relation metadata");
    let application = &sealed[3];

    let mut relation_words = Vec::new();
    append_value_preimage(relation, &mut relation_words);
    relation_words.extend(structural_identifier);
    let mut application_words = Vec::new();
    append_value_preimage(application, &mut application_words);
    let commitment_words = setup_authority(seed);
    let commitment_digest = component(4, &commitment_words);

    let mut nifs_key_words = words(b"Nightstream/FPrime/nifs-key/v1_1");
    nifs_key_words.extend(framed(&relation_words));
    nifs_key_words.extend(framed(&PROFILE));
    nifs_key_words.extend(framed(&SCHEDULE));
    nifs_key_words.extend(framed(&commitment_digest));

    let relation_digest = component(1, &relation_words);
    let application_digest = component(2, &application_words);
    let nifs_key_digest = component(3, &nifs_key_words);
    let component_digests = [relation_digest, application_digest, nifs_key_digest, commitment_digest];
    let rebuilt = rebuild_binding(structural_identifier, component_digests);

    IndependentBinding {
        structural_identifier,
        component_digests,
        package_identity: rebuilt.package_identity,
        context_digest: rebuilt.context_digest,
        descriptor_words: rebuilt.descriptor_words,
        binding_words: rebuilt.binding_words,
        verification_key_digest: rebuilt.verification_key_digest,
        components: [relation_words, application_words, nifs_key_words, commitment_words],
    }
}

#[test]
fn production_binding_matches_independent_lean_framing() {
    let bytes = fs::read(artifact_path()).expect("canonical Stage 1 package");
    let value: Value = serde_json::from_slice(&bytes).expect("canonical package JSON");
    let independent = independent_binding(&value, &PRODUCTION_SEED);
    let lean = read_lean_binding();

    assert_eq!(independent.structural_identifier, lean.structural_identifier);
    assert_eq!(independent.package_identity, lean.package_identity);
    assert_eq!(independent.descriptor_words, lean.descriptor_words);
    let component_start =
        words(b"Nightstream/FPrime/verifier-context/v1_1").len() + framed(&PROFILE).len() + framed(&SCHEDULE).len() + 1;
    for (index, digest) in independent.component_digests.iter().enumerate() {
        let start = component_start + index * framed(&[0; 4]).len();
        assert_eq!(&lean.descriptor_words[start..start + 4], digest);
    }
    assert_eq!(independent.binding_words, lean.binding_words);
    assert_eq!(independent.verification_key_digest, lean.verification_key_digest);
    assert_eq!(hash(&lean.binding_words), lean.verification_key_digest);

    let changed = independent_binding(&value, &{
        let mut seed = PRODUCTION_SEED;
        seed[0] ^= 1;
        seed
    });
    assert_ne!(changed.package_identity, independent.package_identity);
    assert_ne!(changed.context_digest, independent.context_digest);
    assert_ne!(changed.verification_key_digest, independent.verification_key_digest);
    drop(value);

    let package = load_poseidon2_hash_chain_v1_package(&bytes).expect("verifier-owned production package");
    let binding = package
        .production_verifier_binding()
        .expect("fixed production binding");
    assert_eq!(
        independent.structural_identifier,
        POSEIDON2_HASH_CHAIN_V1_STRUCTURAL_IDENTIFIER
    );
    assert_eq!(independent.package_identity, POSEIDON2_HASH_CHAIN_V1_PACKAGE_IDENTITY);
    assert_eq!(independent.package_identity, binding.package_identity());
    assert_eq!(independent.context_digest, binding.verifier_context().digest());
    assert_eq!(
        independent.verification_key_digest,
        POSEIDON2_HASH_CHAIN_V1_VERIFICATION_KEY_DIGEST
    );
    assert_eq!(independent.verification_key_digest, binding.verification_key_digest());
    let context = binding.verifier_context();
    assert_eq!(independent.components[0].as_slice(), context.relation_words());
    assert_eq!(independent.components[1].as_slice(), context.application_words());
    assert_eq!(independent.components[2].as_slice(), context.nifs_key_words());
    assert_eq!(independent.components[3].as_slice(), context.commitment_key_words());
    assert_eq!(
        independent.components.each_ref().map(|words| words.len()),
        [5_120, 1_922_828, 5_184, 73]
    );
    assert_eq!(binding.verifier_context().descriptor_words().len(), 86);
    assert_eq!(binding.verification_key_words().len(), 126);
}

#[test]
fn self_consistent_application_component_mutation_does_not_match_production() {
    let bytes = fs::read(artifact_path()).expect("canonical Stage 1 package");
    let value: Value = serde_json::from_slice(&bytes).expect("canonical package JSON");
    let independent = independent_binding(&value, &PRODUCTION_SEED);
    let lean = read_lean_binding();

    let mut changed_components = independent.component_digests;
    changed_components[1][0] ^= 1;
    let changed = rebuild_binding(independent.structural_identifier, changed_components);

    assert_ne!(changed.descriptor_words, independent.descriptor_words);
    assert_ne!(changed.context_digest, independent.context_digest);
    assert_ne!(changed.package_identity, independent.package_identity);
    assert_ne!(changed.binding_words, independent.binding_words);
    assert_ne!(changed.verification_key_digest, independent.verification_key_digest);
    assert_ne!(changed.descriptor_words, lean.descriptor_words);
    assert_ne!(changed.package_identity, POSEIDON2_HASH_CHAIN_V1_PACKAGE_IDENTITY);
    assert_ne!(
        changed.verification_key_digest,
        POSEIDON2_HASH_CHAIN_V1_VERIFICATION_KEY_DIGEST
    );
}

#[test]
fn package_identity_word_mutation_changes_recomputed_verification_key_digest() {
    let lean = read_lean_binding();
    let package_start = words(b"Nightstream/FPrime/verifier-key/v1").len() + 1;
    assert_eq!(
        &lean.binding_words[package_start..package_start + 4],
        &lean.package_identity
    );
    let mut changed = lean.binding_words;
    changed[package_start] ^= 1;
    assert_ne!(hash(&changed), lean.verification_key_digest);
}
