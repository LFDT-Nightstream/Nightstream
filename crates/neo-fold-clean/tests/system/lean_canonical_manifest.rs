//! Fail-closed tests for the Lean canonical-manifest wire boundary.
//!
//! The fixture is a small schema fixture. It is not a production F′ recipe
//! and its counts are not protocol measurements.

use neo_fold_clean::engine::r1cs_circuit::R1csBuilder;
use neo_fold_clean::frontends::r1cs_f_prime::lean_manifest::{
    LeanCanonicalManifest, LeanManifestEmissionError, ManifestEmission, ManifestTerm, GOLDILOCKS_MODULUS,
};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;
use serde_json::{json, Value};

fn prelude_owner() -> Value {
    json!({ "kind": "prelude" })
}

fn input_owner(slot: usize) -> Value {
    json!({
        "kind": "typed",
        "owner": {
            "kind": "input",
            "slot": slot,
        },
    })
}

fn instruction_owner(path: &[&str]) -> Value {
    json!({
        "kind": "typed",
        "owner": {
            "kind": "instruction",
            "path": path,
        },
    })
}

fn branch_owner(path: &[&str]) -> Value {
    json!({
        "kind": "typed",
        "owner": {
            "kind": "branch",
            "path": path,
        },
    })
}

fn activation_owner(path: &[&str], selected: bool) -> Value {
    json!({
        "kind": "branch_activation",
        "path": path,
        "selected": selected,
    })
}

fn column(owner: &Value, bundle_index: usize, coordinate_index: usize) -> Value {
    json!({
        "owner": owner,
        "bundle_index": bundle_index,
        "coordinate_index": coordinate_index,
    })
}

fn owned(id: Value, ownership: &str) -> Value {
    json!({
        "id": id,
        "ownership": ownership,
    })
}

fn receipt(owner: Value, kind: &str, allocations: Vec<Value>, rows: Vec<Value>) -> Value {
    json!({
        "owner": owner,
        "kind": kind,
        "allocations": allocations,
        "rows": rows,
    })
}

fn segments(entries: &[(&str, usize, &str)]) -> Vec<Value> {
    let mut offset = 0;
    entries
        .iter()
        .map(|(role, width, ownership)| {
            let segment = json!({
                "role": role,
                "width": width,
                "ownership": ownership,
                "offset": offset,
            });
            offset += width;
            segment
        })
        .collect()
}

#[derive(Clone)]
struct ProgramFixture {
    program: Value,
    result_columns: Vec<Value>,
    selector: Value,
    activations: Vec<Value>,
    cost: Value,
    fixed_cost: Value,
    application_cost: Value,
    statistics: Value,
}

#[derive(Clone, Copy)]
enum FixtureProgram {
    Step,
    Terminal,
}

fn program_fixture(input_segments: &[Value], program_kind: FixtureProgram) -> ProgramFixture {
    let one = column(&prelude_owner(), 0, 0);
    let mut receipts = vec![receipt(
        prelude_owner(),
        "prelude",
        vec![owned(one.clone(), "public")],
        vec![],
    )];
    let mut committed_columns = 0usize;
    let mut public_columns = 1usize;
    for (slot, segment) in input_segments.iter().enumerate() {
        let owner = input_owner(slot);
        let width = segment["width"].as_u64().unwrap() as usize;
        let ownership = segment["ownership"].as_str().unwrap();
        let allocations = (0..width)
            .map(|coordinate| owned(column(&owner, slot, coordinate), ownership))
            .collect();
        receipts.push(receipt(owner, "input", allocations, vec![]));
        match ownership {
            "committed" => committed_columns += width,
            "public" => public_columns += width,
            _ => panic!("input fixture uses only committed or public columns"),
        }
    }

    let root = instruction_owner(&[]);
    let (result_columns, selector, activation_path, application_cost) = match program_kind {
        FixtureProgram::Step => {
            let state_columns: Vec<_> = (0..2)
                .map(|coordinate| owned(column(&root, 0, coordinate), "committed"))
                .collect();
            let row = json!({
                "id": {
                    "owner": root,
                    "ordinal": 0,
                },
                "a": [{ "column": one.clone(), "coefficient": 1 }],
                "b": [{ "column": one.clone(), "coefficient": 1 }],
                "c": [{
                    "column": state_columns[0]["id"].clone(),
                    "coefficient": 1
                }],
            });
            receipts.push(receipt(
                instruction_owner(&[]),
                "call",
                state_columns.clone(),
                vec![row],
            ));

            let selector_owner = instruction_owner(&["rest"]);
            let selector = column(&selector_owner, 0, 0);
            receipts.push(receipt(
                selector_owner,
                "call",
                vec![owned(selector.clone(), "auxiliary")],
                vec![],
            ));

            let running_owner = branch_owner(&["rest", "rest"]);
            let running_columns: Vec<_> = (0..3)
                .map(|coordinate| owned(column(&running_owner, 0, coordinate), "committed"))
                .collect();
            receipts.push(receipt(running_owner, "branch_join", running_columns.clone(), vec![]));

            let digest_owner = instruction_owner(&["rest", "rest", "continuation"]);
            let digest_columns: Vec<_> = (0..5)
                .map(|coordinate| owned(column(&digest_owner, 0, coordinate), "public"))
                .collect();
            receipts.push(receipt(digest_owner, "call", digest_columns.clone(), vec![]));

            committed_columns += state_columns.len() + running_columns.len();
            public_columns += digest_columns.len();
            let mut result_columns = state_columns;
            result_columns.extend(running_columns);
            result_columns.extend(digest_columns);
            (
                result_columns,
                selector,
                vec!["rest", "rest"],
                json!({
                    "recurring_rows": 1,
                    "committed_columns": 2,
                    "public_columns": 0,
                    "auxiliary_columns": 0,
                }),
            )
        }
        FixtureProgram::Terminal => {
            let selector = column(&root, 0, 0);
            let row = json!({
                "id": {
                    "owner": root,
                    "ordinal": 0,
                },
                "a": [{ "column": one.clone(), "coefficient": 1 }],
                "b": [{ "column": one.clone(), "coefficient": 1 }],
                "c": [{ "column": selector.clone(), "coefficient": 1 }],
            });
            receipts.push(receipt(
                instruction_owner(&[]),
                "call",
                vec![owned(selector.clone(), "auxiliary")],
                vec![row],
            ));
            (
                vec![],
                selector,
                vec!["rest"],
                json!({
                    "recurring_rows": 0,
                    "committed_columns": 0,
                    "public_columns": 0,
                    "auxiliary_columns": 0,
                }),
            )
        }
    };

    let true_activation = column(&activation_owner(&activation_path, true), 0, 0);
    let false_activation = column(&activation_owner(&activation_path, false), 0, 0);
    receipts.push(receipt(
        activation_owner(&activation_path, true),
        "branch_control",
        vec![owned(true_activation.clone(), "auxiliary")],
        vec![],
    ));
    receipts.push(receipt(
        activation_owner(&activation_path, false),
        "branch_control",
        vec![owned(false_activation.clone(), "auxiliary")],
        vec![],
    ));

    let cost = json!({
        "recurring_rows": 1,
        "committed_columns": committed_columns,
        "public_columns": public_columns,
        "auxiliary_columns": 3,
    });
    let fixed_cost = match program_kind {
        FixtureProgram::Step => json!({
            "recurring_rows": 0,
            "committed_columns": committed_columns - 2,
            "public_columns": public_columns,
            "auxiliary_columns": 3,
        }),
        FixtureProgram::Terminal => cost.clone(),
    };
    ProgramFixture {
        program: json!({
            "one": column(&prelude_owner(), 0, 0),
            "receipts": receipts,
        }),
        result_columns,
        selector,
        activations: vec![true_activation, false_activation],
        cost,
        fixed_cost,
        application_cost,
        statistics: json!({
            "a_nonzeros": 1,
            "b_nonzeros": 1,
            "c_nonzeros": 1,
            "max_row_support": 3,
        }),
    }
}

fn valid_manifest() -> Value {
    let step_input = segments(&[
        ("iteration", 1, "committed"),
        ("initial_state", 2, "committed"),
        ("current_state", 2, "committed"),
        ("running", 3, "committed"),
        ("fresh", 4, "committed"),
        ("witness", 2, "committed"),
        ("nifs_proof", 3, "committed"),
    ]);
    let step_result = segments(&[
        ("next_state", 2, "committed"),
        ("next_running", 3, "committed"),
        ("digest", 5, "public"),
    ]);
    let terminal_input = segments(&[
        ("iteration", 1, "public"),
        ("initial_state", 2, "public"),
        ("current_state", 2, "public"),
        ("running", 3, "committed"),
        ("running_witness", 2, "committed"),
        ("fresh", 4, "committed"),
        ("fresh_witness", 3, "committed"),
    ]);
    let step = program_fixture(&step_input, FixtureProgram::Step);
    let terminal = program_fixture(&terminal_input, FixtureProgram::Terminal);
    json!({
        "schema": 1,
        "format": "nightstream/fprime-canonical-manifest",
        "goldilocks_modulus": GOLDILOCKS_MODULUS,
        "profile": {
            "name": "fixed_one_plain_270",
            "matrix_count": 7,
            "fresh_source_count": 1,
            "running_source_count": 16,
            "public_carrier_width": 270,
            "fresh_legacy_width": 257,
            "fresh_completion_width": 13,
            "running_carrier_width": 270,
            "poseidon_width": 8,
            "poseidon_rate": 4,
            "poseidon_capacity": 4,
            "poseidon_digest_width": 4,
            "binding_preimage_width": 23,
            "decomposition_base": 2,
            "decomposition_children": 16,
        },
        "widths": {
            "iteration": 1,
            "state": 2,
            "witness": 2,
            "running": 3,
            "fresh": 4,
            "nifs_proof": 3,
            "digest": 5,
            "encoded": 6,
            "running_witness": 2,
            "fresh_witness": 3,
            "bit": 1,
        },
        "step_input": step_input,
        "step_result": step_result,
        "terminal_input": terminal_input,
        "step_program": step.program,
        "terminal_program": terminal.program,
        "step_result_columns": step.result_columns,
        "step_selector": step.selector,
        "terminal_selector": terminal.selector,
        "step_activations": step.activations,
        "terminal_activations": terminal.activations,
        "step_cost": step.cost.clone(),
        "terminal_cost": terminal.cost,
        "fixed_protocol_cost": step.fixed_cost,
        "application_step_cost": step.application_cost,
        "step_statistics": step.statistics,
        "terminal_statistics": terminal.statistics,
    })
}

fn parse(manifest: &Value) -> Result<LeanCanonicalManifest, String> {
    LeanCanonicalManifest::from_json_slice(&serde_json::to_vec(manifest).unwrap()).map_err(|error| error.to_string())
}

fn expected_row(terms: &[ManifestTerm], emission: &ManifestEmission) -> Vec<(usize, F)> {
    let mut row: Vec<_> = terms
        .iter()
        .map(|term| {
            (
                emission
                    .variable(&term.column)
                    .expect("validated term has an emitted variable")
                    .col(),
                F::from_u64(term.coefficient),
            )
        })
        .collect();
    row.sort_by_key(|(column, _)| *column);
    row
}

#[test]
fn accepts_a_complete_schema_fixture_without_fixing_matrix_count_to_rust() {
    let manifest = parse(&valid_manifest()).expect("valid schema fixture");
    assert_eq!(manifest.matrix_count(), 7);
    assert_eq!(manifest.step_cost().recurring_rows(), 1);
    assert_eq!(manifest.terminal_cost().recurring_rows(), 1);
}

#[test]
fn rejects_unknown_fields() {
    let mut manifest = valid_manifest();
    manifest["rust_authority"] = json!(true);
    assert!(parse(&manifest).unwrap_err().contains("unknown field"));
}

#[test]
fn rejects_noncanonical_sparse_coefficients() {
    let mut manifest = valid_manifest();
    manifest["step_program"]["receipts"][8]["rows"][0]["a"][0]["coefficient"] = json!(GOLDILOCKS_MODULUS);
    assert!(parse(&manifest)
        .unwrap_err()
        .contains("canonical Goldilocks residue"));
}

#[test]
fn rejects_codec_segments_not_bound_to_input_allocations() {
    let mut manifest = valid_manifest();
    manifest["step_program"]["receipts"][1]["allocations"]
        .as_array_mut()
        .unwrap()
        .clear();
    assert!(parse(&manifest)
        .unwrap_err()
        .contains("canonical input receipt"));
}

#[test]
fn rejects_result_ownership_drift() {
    let mut manifest = valid_manifest();
    manifest["step_result_columns"][0]["ownership"] = json!("public");
    assert!(parse(&manifest)
        .unwrap_err()
        .contains("step_result_columns"));
}

#[test]
fn rejects_same_ownership_result_substitution() {
    let mut manifest = valid_manifest();
    manifest["step_result_columns"][0] = manifest["step_program"]["receipts"][1]["allocations"][0].clone();
    assert!(parse(&manifest)
        .unwrap_err()
        .contains("canonical Step result ABI"));
}

#[test]
fn rejects_receipt_cost_drift() {
    let mut manifest = valid_manifest();
    manifest["step_cost"]["recurring_rows"] = json!(2);
    assert!(parse(&manifest).unwrap_err().contains("step_cost"));
}

#[test]
fn rejects_balanced_application_cost_split_drift() {
    let mut manifest = valid_manifest();
    manifest["fixed_protocol_cost"]["recurring_rows"] = json!(1);
    manifest["application_step_cost"]["recurring_rows"] = json!(0);
    assert!(parse(&manifest)
        .unwrap_err()
        .contains("application_step_cost"));
}

#[test]
fn rejects_selector_substitution() {
    let mut manifest = valid_manifest();
    manifest["step_selector"] = manifest["step_activations"][0].clone();
    assert!(parse(&manifest)
        .unwrap_err()
        .contains("canonical Step selector"));
}

#[test]
fn rejects_noncanonical_activation_order() {
    let mut manifest = valid_manifest();
    let activations = manifest["step_activations"].as_array_mut().unwrap();
    activations.swap(0, 1);
    assert!(parse(&manifest).unwrap_err().contains("step_activations"));
}

#[test]
fn emits_the_validated_rows_with_public_columns_first() {
    let manifest = parse(&valid_manifest()).expect("valid schema fixture");
    let result = manifest.step_result_columns()[0].id.clone();
    let mut builder = R1csBuilder::new();
    let emission = manifest
        .emit_step(&mut builder, |_| Some(F::ONE))
        .expect("emit validated Step");
    assert_eq!(builder.rows(), manifest.step_cost().recurring_rows());
    assert_eq!(
        builder.cols(),
        manifest.step_cost().committed_columns()
            + manifest.step_cost().public_columns()
            + manifest.step_cost().auxiliary_columns()
    );
    assert_eq!(emission.public_input_len(), manifest.step_cost().public_columns());
    assert_eq!(
        emission.committed_columns().len(),
        manifest.step_cost().committed_columns()
    );
    assert_eq!(
        emission.auxiliary_columns().len(),
        manifest.step_cost().auxiliary_columns()
    );
    assert!(builder.is_satisfied());

    let snapshot = builder.snapshot();
    let rows: Vec<_> = manifest
        .step_program()
        .receipts
        .iter()
        .flat_map(|receipt| receipt.rows.iter())
        .collect();
    assert_eq!(snapshot.rows(), rows.len());
    for (index, row) in rows.iter().enumerate() {
        assert_eq!(snapshot.a_row(index), expected_row(&row.a, &emission));
        assert_eq!(snapshot.b_row(index), expected_row(&row.b, &emission));
        assert_eq!(snapshot.c_row(index), expected_row(&row.c, &emission));
    }

    let mut bad_builder = R1csBuilder::new();
    manifest
        .emit_step(&mut bad_builder, |column| {
            Some(if column == &result { F::from_u64(2) } else { F::ONE })
        })
        .expect("emit mutated witness");
    assert_eq!(bad_builder.first_unsatisfied_row(), Some(0));
}

#[test]
fn emission_requires_a_complete_witness_and_a_fresh_builder() {
    let manifest = parse(&valid_manifest()).expect("valid schema fixture");
    let mut missing = R1csBuilder::new();
    assert!(matches!(
        manifest.emit_step(&mut missing, |_| None),
        Err(LeanManifestEmissionError::MissingValue { .. })
    ));

    let mut nonfresh = R1csBuilder::new();
    nonfresh.alloc(F::ZERO);
    assert!(matches!(
        manifest.emit_step(&mut nonfresh, |_| Some(F::ONE)),
        Err(LeanManifestEmissionError::BuilderNotFresh { .. })
    ));
}
