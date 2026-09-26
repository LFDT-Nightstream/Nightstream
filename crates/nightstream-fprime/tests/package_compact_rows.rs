use std::{fs, path::PathBuf};

use nightstream_fprime::{load_per_application_package, load_prepared_application_value, PackageError};
use serde_json::{json, Value};

fn plan_value() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json");
    serde_json::from_slice(&fs::read(path).expect("Lean-emitted package")).unwrap()
}

fn first_compact_template(value: &mut Value) -> &mut Vec<Value> {
    value[1][8][0]
        .as_array_mut()
        .expect("first compact template")
}

#[test]
fn sealed_package_rejects_the_retired_prefix_plan() {
    assert!(load_per_application_package(b"[8,[],[],[],[]]\n", [0; 4]).is_err());
}

#[test]
fn sealed_package_rejects_a_malformed_compact_optional_output() {
    let mut value = plan_value();
    let template = first_compact_template(&mut value);
    template[4].as_array_mut().expect("compact template rows")[0]
        .as_array_mut()
        .expect("compact template row")[0] = json!([2]);

    assert!(matches!(
        load_prepared_application_value(value),
        Err(PackageError::Invalid("compact optional output"))
    ));
}

#[test]
fn sealed_package_rejects_an_output_self_dependent_recipe() {
    let mut value = plan_value();
    let template = first_compact_template(&mut value);
    let output_input = template[2].clone();
    template[3] = json!([0, output_input]);

    assert!(matches!(
        load_prepared_application_value(value),
        Err(PackageError::Invalid("compact output recipe input"))
    ));
}

#[test]
fn sealed_package_rejects_an_incomplete_compact_input_partition() {
    let mut value = plan_value();
    let template = first_compact_template(&mut value);
    let input_count = template[0].as_u64().expect("compact input count");
    template[0] = json!(input_count + 1);

    assert!(matches!(
        load_prepared_application_value(value),
        Err(PackageError::Invalid("compact input coverage"))
    ));
}
