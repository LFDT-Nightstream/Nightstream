use super::*;
use crate::application::{poseidon2_hash_chain_v1, Affine, ApplicationBuilder};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;
use serde_json::json;

fn manifest_bytes() -> &'static [u8] {
    include_bytes!("../../artifacts/shared-verifier-v1.json")
}

#[test]
fn rust_poseidon_plan_preserves_every_raw_row_recipe_and_identity_byte() {
    let manifest = Manifest::parse(manifest_bytes()).unwrap();
    let application = poseidon2_hash_chain_v1().unwrap();
    let plan = application::plan(&application, &manifest).unwrap();
    let expected = include_bytes!("../fixtures/poseidon2-application-reference.json");
    let mut actual = serde_json::to_vec(&plan).unwrap();
    actual.push(b'\n');
    // Syntax, duplicate sparse terms and recipe operation order are identity inputs.
    assert_eq!(actual.len(), expected.len(), "raw application plan byte length");
    let first_difference = actual
        .iter()
        .zip(expected.iter())
        .position(|(actual, expected)| actual != expected);
    assert_eq!(first_difference, None, "first different raw application plan byte");
}

#[test]
fn independent_poseidon_assembly_equals_the_complete_reference_value() {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json");
    let bytes = std::fs::read(path).unwrap();
    let expected: Value = serde_json::from_slice(&bytes).unwrap();
    let mut reference: wire::Envelope = serde_json::from_slice(&bytes).unwrap();
    let manifest = Manifest::parse(manifest_bytes()).unwrap();
    reference.assignment.schema = 2;
    assert!(manifest.check_reference(&reference).is_err());
    reference.assignment.schema = 3;
    let application = poseidon2_hash_chain_v1().unwrap();
    let actual = assemble(reference, &manifest, &application).unwrap();

    // Compare the entire numeric-array value: physical constraints and witness
    // recipes, every matrix block, assignment transport, layout and terminal data.
    // No identity digest or selected row interval substitutes for this equality.
    assert!(
        actual == expected,
        "complete assembled value differs from the selected reference"
    );
}

#[test]
fn preparation_rejects_a_changed_reference_even_when_assembly_repairs_it() {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json");
    let bytes = std::fs::read(path).unwrap();
    let expected: Value = serde_json::from_slice(&bytes).unwrap();
    let mut reference: wire::Envelope = serde_json::from_slice(&bytes).unwrap();
    drop(bytes);
    // This part of the reference is replaced by the caller's application.
    // A correct candidate therefore cannot authorize this altered blueprint.
    let constant = &mut reference.application.rows[0].a.constant;
    *constant = u64::from(*constant == 0);
    let mut changed = serde_json::to_vec(&reference).unwrap();
    changed.push(b'\n');
    let manifest = Manifest::parse(manifest_bytes()).unwrap();
    let application = poseidon2_hash_chain_v1().unwrap();
    let candidate = assemble(reference, &manifest, &application).unwrap();
    assert!(
        candidate[3] == expected[3],
        "the rebuilt physical application repairs the changed row"
    );
    drop(candidate);
    drop(expected);
    assert!(matches!(
        prepare(&changed, &application),
        Err(AssemblyError::Package(PackageError::ExpectedIdentityMismatch { .. }))
    ));
}

#[test]
fn rust_addition_plan_uses_declared_ports_and_causal_recipes() {
    let manifest = Manifest::parse(manifest_bytes()).unwrap();
    let mut builder = ApplicationBuilder::new(4).unwrap();
    let input = builder.input_state();
    let private = builder.private_inputs().to_vec();
    let mut output = std::array::from_fn(|_| Affine::constant(Goldilocks::ZERO));
    for lane in 0..4 {
        output[lane] = builder
            .affine(Affine::from(input[lane]) + Affine::from(private[lane]))
            .unwrap()
            .into();
    }
    let circuit = builder.finish(output).unwrap();
    let plan = application::plan(&circuit, &manifest).unwrap();
    assert_eq!(plan.private_count, 4);
    assert_eq!(plan.row_count, 8);
    for lane in 0..4 {
        assert_eq!(
            plan.batches[0].recipes[lane],
            json!([2, [0, plan.input_columns[lane]], [0, plan.witness_columns[lane]]])
        );
        assert_eq!(plan.rows[lane].c.terms, vec![(plan.private_start + lane, 1)]);
    }
    manifest.check_dimensions(Counts::of(&circuit)).unwrap();
}

#[test]
fn manifest_rejects_missing_children_changed_roles_profile_and_dimensions() {
    let original: Value = serde_json::from_slice(manifest_bytes()).unwrap();
    let mut missing = original.clone();
    missing["children"].as_array_mut().unwrap().remove(0);
    let mut reordered = original.clone();
    reordered["children"].as_array_mut().unwrap().swap(0, 1);
    let mut role = original.clone();
    role["ports"][0]["role"] = json!("public_input");
    let mut profile = original.clone();
    profile["profile"][2] = json!(18);
    let mut width = original.clone();
    width["geometry"]["logical_width"] = json!([usize::MAX, 1, 1, 0]);
    let mut public = original;
    public["recursive_public"]["digest_port"] = json!("prior_public_input");
    for value in [missing, reordered, role, profile, width, public] {
        assert!(Manifest::parse(&serde_json::to_vec(&value).unwrap()).is_err());
    }
    let manifest = Manifest::parse(manifest_bytes()).unwrap();
    assert!(manifest
        .check_dimensions(Counts {
            witness: usize::MAX,
            local: 0,
            rows: 0
        })
        .is_err());
}
