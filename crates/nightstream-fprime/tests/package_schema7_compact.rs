use std::{collections::BTreeMap, fs, path::PathBuf};

use nightstream_fprime::{load, PackageError};
use serde_json::{json, Value};

fn artifact_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-v1.json")
}

fn schema7_value() -> Value {
    let bytes = fs::read(artifact_path()).expect("Lean-emitted schema-6 transition artifact");
    let mut value: Value = serde_json::from_slice(&bytes).expect("package JSON");
    let package = value.as_array_mut().expect("package tuple");
    assert_eq!(package[0], json!(6), "transition source schema");
    package[0] = json!(7);
    package.insert(8, json!([]));
    package.insert(9, json!([]));
    value
}

fn canonical_bytes(value: &Value) -> Vec<u8> {
    let mut bytes = serde_json::to_vec(value).expect("canonical package JSON");
    bytes.push(b'\n');
    bytes
}

fn candidate_identity(bytes: &[u8]) -> [u64; 4] {
    match load(bytes, [0; 4]) {
        Err(PackageError::ExpectedIdentityMismatch { computed, .. }) => computed,
        other => panic!("candidate identity result: {other:?}"),
    }
}

fn zero_assertion(output: Value) -> Value {
    json!([output, [0, []], [1, []], [0, []]])
}

fn constant_zero_template() -> Value {
    json!([1, 0, 0, [1, 0], [zero_assertion(json!([0]))]])
}

fn word(value: &Value, location: &str) -> u64 {
    value.as_u64().unwrap_or_else(|| panic!("{location} word"))
}

fn combination_terms(combination: &Value) -> &[Value] {
    combination.as_array().expect("sparse combination")[1]
        .as_array()
        .expect("sparse terms")
}

fn mapped_combination(combination: &Value, slots: &BTreeMap<u64, usize>) -> Value {
    let combination = combination.as_array().expect("sparse combination");
    let terms = combination[1]
        .as_array()
        .expect("sparse terms")
        .iter()
        .map(|term| {
            let term = term.as_array().expect("sparse term");
            let column = word(&term[0], "sparse column");
            let coefficient = word(&term[1], "sparse coefficient");
            json!([[0, slots[&column]], coefficient])
        })
        .collect::<Vec<_>>();
    json!([word(&combination[0], "sparse constant"), terms])
}

fn combination_expr(combination: &Value, slots: &BTreeMap<u64, usize>) -> Value {
    let combination = combination.as_array().expect("sparse combination");
    let mut expression = json!([1, word(&combination[0], "sparse constant")]);
    for term in combination[1].as_array().expect("sparse terms") {
        let term = term.as_array().expect("sparse term");
        let column = word(&term[0], "sparse column");
        let coefficient = word(&term[1], "sparse coefficient");
        let scaled = json!([3, [1, coefficient], [0, slots[&column]]]);
        expression = json!([2, expression, scaled]);
    }
    expression
}

fn replace_two_instructions_with_compact(value: &mut Value) {
    let package = value.as_array_mut().expect("package tuple");
    let instructions = package[11].as_array_mut().expect("witness instructions");
    let pair_index = instructions
        .windows(2)
        .position(|pair| {
            let first = pair[0].as_array().expect("first instruction");
            let second = pair[1].as_array().expect("second instruction");
            word(&second[0], "second row") == word(&first[0], "first row") + 1
                && word(&second[1], "second target") == word(&first[1], "first target") + 1
        })
        .expect("adjacent generic instruction pair");
    let first = instructions[pair_index].clone();
    let second = instructions[pair_index + 1].clone();
    let first = first.as_array().expect("first instruction");
    let second = second.as_array().expect("second instruction");
    let row_start = word(&first[0], "first row");
    let output_column = word(&first[1], "first target");
    let local_start = word(&second[1], "second target");

    let mut columns = BTreeMap::<u64, usize>::new();
    for combination in [&first[2], &first[3], &second[2], &second[3]] {
        for term in combination_terms(combination) {
            let column = word(&term.as_array().expect("sparse term")[0], "sparse column");
            if column != output_column {
                let next = columns.len();
                columns.entry(column).or_insert(next);
            }
        }
    }
    let output_input = columns.len();
    columns.insert(output_column, output_input);

    let mut input_ranges = columns
        .iter()
        .map(|(column, input)| json!([input, 1, column, 1]))
        .collect::<Vec<_>>();
    input_ranges.sort_unstable_by_key(|range| {
        word(
            &range.as_array().expect("compact input range")[0],
            "compact input start",
        )
    });
    let output_recipe = json!([
        3,
        combination_expr(&first[2], &columns),
        combination_expr(&first[3], &columns)
    ]);
    let witness_row = json!([
        [1, 0],
        mapped_combination(&second[2], &columns),
        mapped_combination(&second[3], &columns),
        [0, [[[1, 0], 1]]]
    ]);
    let template = json!([
        columns.len(),
        1,
        output_input,
        output_recipe,
        [witness_row, zero_assertion(json!([0]))]
    ]);
    let invocation = json!([7, 0, row_start, local_start, input_ranges]);

    instructions.drain(pair_index..pair_index + 2);
    package[8] = json!([template]);
    package[9] = json!([invocation]);
}

#[test]
fn schema7_decodes_the_compact_fields_at_the_lean_owned_tuple_positions() {
    let mut value = schema7_value();
    value.as_array_mut().expect("package tuple")[8] = json!([constant_zero_template()]);
    let bytes = canonical_bytes(&value);
    let identity = candidate_identity(&bytes);
    let package = load(&bytes, identity).expect("strict schema-7 transition load");

    assert_eq!(package.compact_template_count(), 1);
    assert_eq!(package.compact_invocation_count(), 0);
}

#[test]
fn schema7_validates_one_compact_invocation_as_exact_row_and_column_coverage() {
    let mut value = schema7_value();
    replace_two_instructions_with_compact(&mut value);
    let bytes = canonical_bytes(&value);
    let identity = candidate_identity(&bytes);
    let package = load(&bytes, identity).expect("covered compact invocation");

    assert_eq!(package.compact_template_count(), 1);
    assert_eq!(package.compact_invocation_count(), 1);
}

#[test]
fn schema7_rejects_a_malformed_compact_optional_output() {
    let mut value = schema7_value();
    let template = json!([1, 0, 0, [1, 0], [zero_assertion(json!([2]))]]);
    value.as_array_mut().expect("package tuple")[8] = json!([template]);

    assert!(matches!(
        load(&canonical_bytes(&value), [0; 4]),
        Err(PackageError::Invalid("compact optional output"))
    ));
}

#[test]
fn schema7_rejects_an_output_self_dependent_recipe() {
    let mut value = schema7_value();
    let template = json!([1, 0, 0, [0, 0], [zero_assertion(json!([0]))]]);
    value.as_array_mut().expect("package tuple")[8] = json!([template]);

    assert!(matches!(
        load(&canonical_bytes(&value), [0; 4]),
        Err(PackageError::Invalid("compact output recipe input"))
    ));
}

#[test]
fn schema7_rejects_an_incomplete_compact_input_partition() {
    let mut value = schema7_value();
    let package = value.as_array_mut().expect("package tuple");
    package[8] = json!([constant_zero_template()]);
    package[9] = json!([[7, 0, 0, 113963, []]]);

    assert!(matches!(
        load(&canonical_bytes(&value), [0; 4]),
        Err(PackageError::Invalid("compact input coverage"))
    ));
}
