use std::{fs, path::PathBuf};

use nightstream_fprime::{load, PackageError};
use serde_json::{json, Value};

const EXPECTED_IDENTITY: [u64; 4] = [
    5_598_780_946_789_064_029,
    15_355_422_093_920_338_696,
    10_729_673_706_357_134_548,
    3_502_763_498_223_293_662,
];

const PI_DEC_COMMITMENTS_ROLE: u64 = 11;

fn artifact_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-v1.json")
}

fn artifact_bytes() -> Vec<u8> {
    fs::read(artifact_path()).expect("Lean-emitted schema-8 package plan")
}

fn plan_value() -> Value {
    serde_json::from_slice(&artifact_bytes()).expect("package-plan JSON")
}

fn canonical_bytes(value: &Value) -> Vec<u8> {
    let mut bytes = serde_json::to_vec(value).expect("canonical package-plan JSON");
    bytes.push(b'\n');
    bytes
}

fn first_compact_template(value: &mut Value) -> &mut Vec<Value> {
    let plan = value.as_array_mut().expect("package-plan tuple");
    assert_eq!(plan[0], json!(8), "package-plan schema");
    let package = plan[1].as_array_mut().expect("embedded package tuple");
    assert_eq!(package[0], json!(7), "embedded package schema");
    package[8].as_array_mut().expect("compact templates")[0]
        .as_array_mut()
        .expect("first compact template")
}

fn witness_blocks(value: &mut Value) -> &mut Vec<Value> {
    let plan = value.as_array_mut().expect("package-plan tuple");
    assert_eq!(plan[0], json!(8), "package-plan schema");
    plan[4].as_array_mut().expect("witness-plan blocks")
}

fn pi_dec_input_start(value: &Value) -> u64 {
    let plan = value.as_array().expect("package-plan tuple");
    let package = plan[1].as_array().expect("embedded package tuple");
    let layout = package[3].as_array().expect("package layout");
    layout[5]
        .as_array()
        .expect("private segments")
        .iter()
        .find_map(|segment| {
            let fields = segment.as_array().expect("private segment");
            (fields[0] == json!(PI_DEC_COMMITMENTS_ROLE))
                .then(|| fields[1].as_u64().expect("PiDEC commitment segment start"))
        })
        .expect("PiDEC commitment segment")
}

#[test]
fn schema8_plan_places_schema7_compact_fields_at_the_lean_owned_positions() {
    let value = plan_value();
    let plan = value.as_array().expect("package-plan tuple");
    assert_eq!(plan.len(), 5, "package-plan tuple length");
    assert_eq!(plan[0], json!(8), "package-plan schema");
    let package = plan[1].as_array().expect("embedded package tuple");
    assert_eq!(package.len(), 14, "embedded package tuple length");
    assert_eq!(package[0], json!(7), "embedded package schema");
    assert_eq!(package[8].as_array().expect("compact templates").len(), 326);
    assert!(package[9]
        .as_array()
        .expect("static compact invocations")
        .is_empty());
    assert_eq!(plan[3].as_array().expect("compact plan blocks").len(), 2);
}

#[test]
fn schema8_plan_expands_every_compact_invocation_with_exact_coverage() {
    let package = load(&artifact_bytes(), EXPECTED_IDENTITY).expect("strict package-plan load");
    assert_eq!(package.compact_template_count(), 326);
    assert_eq!(package.compact_invocation_count(), 170_918);
}

#[test]
fn schema8_plan_rejects_a_malformed_compact_optional_output() {
    let mut value = plan_value();
    let template = first_compact_template(&mut value);
    template[4].as_array_mut().expect("compact template rows")[0]
        .as_array_mut()
        .expect("compact template row")[0] = json!([2]);

    assert!(matches!(
        load(&canonical_bytes(&value), [0; 4]),
        Err(PackageError::Invalid("compact optional output"))
    ));
}

#[test]
fn schema8_plan_rejects_an_output_self_dependent_recipe() {
    let mut value = plan_value();
    let template = first_compact_template(&mut value);
    let output_input = template[2].clone();
    template[3] = json!([0, output_input]);

    assert!(matches!(
        load(&canonical_bytes(&value), [0; 4]),
        Err(PackageError::Invalid("compact output recipe input"))
    ));
}

#[test]
fn schema8_plan_rejects_an_incomplete_compact_input_partition() {
    let mut value = plan_value();
    let template = first_compact_template(&mut value);
    let input_count = template[0].as_u64().expect("compact input count");
    template[0] = json!(input_count + 1);

    assert!(matches!(
        load(&canonical_bytes(&value), [0; 4]),
        Err(PackageError::Invalid("compact input coverage"))
    ));
}

#[test]
fn schema8_plan_rejects_a_wrong_digest_block_tag() {
    let mut value = plan_value();
    witness_blocks(&mut value)[0]
        .as_array_mut()
        .expect("first digest block")[0] = json!(1);

    assert!(matches!(
        load(&canonical_bytes(&value), [0; 4]),
        Err(PackageError::Invalid("witness digest block tag"))
    ));
}

#[test]
fn schema8_plan_rejects_a_missing_explicit_pidec_block() {
    let mut value = plan_value();
    witness_blocks(&mut value)
        .pop()
        .expect("running-transition witness block");
    witness_blocks(&mut value)
        .pop()
        .expect("PiDEC witness block");

    assert!(matches!(
        load(&canonical_bytes(&value), [0; 4]),
        Err(PackageError::Invalid("witness plan block count"))
    ));
}

#[test]
fn schema8_plan_rejects_a_wrong_pidec_batch_count() {
    let mut value = plan_value();
    let blocks = witness_blocks(&mut value);
    let pi_dec_block = blocks.len() - 2;
    blocks
        .get_mut(pi_dec_block)
        .expect("PiDEC witness block")
        .as_array_mut()
        .expect("tagged PiDEC witness block")[1]
        .as_array_mut()
        .expect("PiDEC witness batches")
        .pop()
        .expect("PiDEC witness batch");

    assert!(matches!(
        load(&canonical_bytes(&value), [0; 4]),
        Err(PackageError::Invalid("PiDEC witness batch count"))
    ));
}

#[test]
fn schema8_plan_rejects_a_wrong_running_transition_batch_count() {
    let mut value = plan_value();
    let blocks = witness_blocks(&mut value);
    blocks
        .last_mut()
        .expect("running-transition witness block")
        .as_array_mut()
        .expect("tagged running-transition witness block")[1]
        .as_array_mut()
        .expect("running-transition witness batches")
        .pop()
        .expect("running-transition witness batch");

    assert!(matches!(
        load(&canonical_bytes(&value), [0; 4]),
        Err(PackageError::Invalid("running-transition witness batch count"))
    ));
}

#[test]
fn schema8_plan_rejects_a_generated_write_into_pidec_inputs() {
    let mut value = plan_value();
    let pi_dec_input_start = pi_dec_input_start(&value);
    let blocks = witness_blocks(&mut value);
    let pi_dec_block = blocks.len() - 2;
    blocks
        .get_mut(pi_dec_block)
        .expect("PiDEC witness block")
        .as_array_mut()
        .expect("tagged PiDEC witness block")[1]
        .as_array_mut()
        .expect("PiDEC witness batches")[0]
        .as_array_mut()
        .expect("PiDEC witness batch")[0] = json!(pi_dec_input_start);

    assert!(matches!(
        load(&canonical_bytes(&value), [0; 4]),
        Err(PackageError::Invalid("witness interval ownership"))
    ));
}
