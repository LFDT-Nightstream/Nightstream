//! Shared exact manifest fixtures for Rust integration tests.

#![allow(dead_code)]

use neo_fold_clean::frontends::r1cs_f_prime::lean_manifest::GOLDILOCKS_MODULUS;
use neo_fold_clean::frontends::r1cs_f_prime::lean_native_ccs_manifest::{
    LEAN_NATIVE_CCS_MANIFEST_FORMAT, LEAN_NATIVE_CCS_MANIFEST_SCHEMA_VERSION,
};
use neo_fold_clean::{LeanNativeCcsManifest, LeanNebulaCombinedManifest};
use serde_json::{json, Value};

pub(super) const TEST_AJTAI_SEED: u64 = 0x5445_524d_494e_414c;

pub(super) fn test_ajtai_seed() -> [u8; 32] {
    let mut seed = [0u8; 32];
    seed[..8].copy_from_slice(&TEST_AJTAI_SEED.to_le_bytes());
    seed
}

pub(super) fn prelude_owner() -> Value {
    json!({ "kind": "prelude" })
}

pub(super) fn input_owner(slot: usize) -> Value {
    json!({
        "kind": "typed",
        "owner": { "kind": "input", "slot": slot },
    })
}

pub(super) fn instruction_owner(path: &[&str]) -> Value {
    json!({
        "kind": "typed",
        "owner": { "kind": "instruction", "path": path },
    })
}

pub(super) fn branch_owner(path: &[&str]) -> Value {
    json!({
        "kind": "typed",
        "owner": { "kind": "branch", "path": path },
    })
}

pub(super) fn activation_owner(path: &[&str], selected: bool) -> Value {
    json!({
        "kind": "branch_activation",
        "path": path,
        "selected": selected,
    })
}

pub(super) fn column(owner: &Value, bundle_index: usize, coordinate_index: usize) -> Value {
    json!({
        "owner": owner,
        "bundle_index": bundle_index,
        "coordinate_index": coordinate_index,
    })
}

pub(super) fn owned(id: Value, ownership: &str) -> Value {
    json!({ "id": id, "ownership": ownership })
}

pub(super) fn term(column: Value) -> Value {
    json!({ "column": column, "coefficient": 1 })
}

pub(super) fn row(owner: Value, ordinal: usize, a: Value, b: Value, c: Value) -> Value {
    json!({
        "id": { "owner": owner, "ordinal": ordinal },
        "a": [term(a)],
        "b": [term(b)],
        "c": [term(c)],
    })
}

pub(super) fn native_receipt(
    owner: Value,
    kind: &str,
    allocations: Vec<Value>,
    selector: Value,
    rows: Vec<Value>,
) -> Value {
    json!({
        "owner": owner,
        "kind": kind,
        "allocations": allocations,
        "selector": selector,
        "rows": rows,
    })
}

pub(super) fn canonical_receipt(owner: Value, kind: &str, allocations: Vec<Value>, rows: Vec<Value>) -> Value {
    json!({
        "owner": owner,
        "kind": kind,
        "allocations": allocations,
        "rows": rows,
    })
}

pub(super) fn segments(entries: &[(&str, usize, &str)]) -> Vec<Value> {
    let mut offset = 0usize;
    entries
        .iter()
        .map(|(role, width, ownership)| {
            let value = json!({
                "role": role,
                "width": width,
                "ownership": ownership,
                "offset": offset,
            });
            offset += width;
            value
        })
        .collect()
}

pub(super) fn input_receipts(input_segments: &[Value], one: &Value, native: bool) -> Vec<Value> {
    input_segments
        .iter()
        .enumerate()
        .map(|(slot, segment)| {
            let owner = input_owner(slot);
            let width = segment["width"].as_u64().unwrap() as usize;
            let ownership = segment["ownership"].as_str().unwrap();
            let allocations = (0..width)
                .map(|coordinate| owned(column(&owner, slot, coordinate), ownership))
                .collect();
            if native {
                native_receipt(owner, "input", allocations, one.clone(), vec![])
            } else {
                canonical_receipt(owner, "input", allocations, vec![])
            }
        })
        .collect()
}

pub(super) fn valid_manifest() -> Value {
    let one = column(&prelude_owner(), 0, 0);
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

    let root = instruction_owner(&[]);
    let step_selector_owner = instruction_owner(&["rest"]);
    let step_selector = column(&step_selector_owner, 0, 0);
    let branch_path = ["rest", "rest"];
    let true_activation_owner = activation_owner(&branch_path, true);
    let false_activation_owner = activation_owner(&branch_path, false);
    let true_activation = column(&true_activation_owner, 0, 0);
    let false_activation = column(&false_activation_owner, 0, 0);

    let mut step_receipts = vec![native_receipt(
        prelude_owner(),
        "prelude",
        vec![owned(one.clone(), "public")],
        one.clone(),
        vec![],
    )];
    step_receipts.extend(input_receipts(&step_input, &one, true));

    let next_state: Vec<_> = (0..2)
        .map(|coordinate| owned(column(&root, 0, coordinate), "committed"))
        .collect();
    step_receipts.push(native_receipt(
        root.clone(),
        "call",
        next_state.clone(),
        one.clone(),
        vec![],
    ));
    step_receipts.push(native_receipt(
        step_selector_owner,
        "call",
        vec![owned(step_selector.clone(), "auxiliary")],
        one.clone(),
        vec![],
    ));
    step_receipts.push(native_receipt(
        true_activation_owner,
        "branch_control",
        vec![owned(true_activation.clone(), "auxiliary")],
        one.clone(),
        vec![],
    ));
    step_receipts.push(native_receipt(
        false_activation_owner,
        "branch_control",
        vec![owned(false_activation.clone(), "auxiliary")],
        one.clone(),
        vec![],
    ));

    let running_owner = branch_owner(&branch_path);
    let next_running: Vec<_> = (0..3)
        .map(|coordinate| owned(column(&running_owner, 0, coordinate), "committed"))
        .collect();
    step_receipts.push(native_receipt(
        running_owner,
        "branch_join",
        next_running.clone(),
        one.clone(),
        vec![],
    ));

    let digest_owner = instruction_owner(&["rest", "rest", "continuation"]);
    let digest: Vec<_> = (0..5)
        .map(|coordinate| owned(column(&digest_owner, 0, coordinate), "public"))
        .collect();
    step_receipts.push(native_receipt(
        digest_owner,
        "call",
        digest.clone(),
        one.clone(),
        vec![],
    ));

    let target_path = ["rest", "rest", "false_arm", "rest", "rest", "rest", "rest", "rest"];
    let target_owner = instruction_owner(&target_path);
    let x = column(&input_owner(1), 1, 0);
    let y = column(&input_owner(1), 1, 1);
    let product = column(&input_owner(2), 2, 0);
    let carrier_completion: Vec<_> = (0..239)
        .map(|coordinate| owned(column(&target_owner, 1, coordinate), "auxiliary"))
        .collect();
    step_receipts.push(native_receipt(
        target_owner.clone(),
        "call",
        carrier_completion,
        false_activation.clone(),
        vec![row(target_owner, 0, x, y, product)],
    ));

    let terminal_selector = column(&root, 0, 0);
    let terminal_branch_path = ["rest"];
    let terminal_true_owner = activation_owner(&terminal_branch_path, true);
    let terminal_false_owner = activation_owner(&terminal_branch_path, false);
    let terminal_true = column(&terminal_true_owner, 0, 0);
    let terminal_false = column(&terminal_false_owner, 0, 0);
    let mut terminal_receipts = vec![canonical_receipt(
        prelude_owner(),
        "prelude",
        vec![owned(one.clone(), "public")],
        vec![],
    )];
    terminal_receipts.extend(input_receipts(&terminal_input, &one, false));
    let terminal_iteration = column(&input_owner(0), 0, 0);
    let terminal_running = column(&input_owner(3), 3, 0);
    terminal_receipts.push(canonical_receipt(
        root.clone(),
        "call",
        vec![owned(terminal_selector.clone(), "auxiliary")],
        vec![row(root, 0, terminal_iteration, one.clone(), terminal_running)],
    ));
    terminal_receipts.push(canonical_receipt(
        terminal_true_owner,
        "branch_control",
        vec![owned(terminal_true.clone(), "auxiliary")],
        vec![],
    ));
    terminal_receipts.push(canonical_receipt(
        terminal_false_owner,
        "branch_control",
        vec![owned(terminal_false.clone(), "auxiliary")],
        vec![],
    ));

    let mut step_result_columns = next_state;
    step_result_columns.extend(next_running);
    step_result_columns.extend(digest);

    json!({
        "schema": LEAN_NATIVE_CCS_MANIFEST_SCHEMA_VERSION,
        "format": LEAN_NATIVE_CCS_MANIFEST_FORMAT,
        "goldilocks_modulus": GOLDILOCKS_MODULUS,
        "ajtai_setup": {
            "algorithm": "chacha8_phi81_rejection_v1",
            "seed": test_ajtai_seed(),
            "rejection_fuel": 8,
        },
        "profile": {
            "name": "fixed_one_plain_270",
            "matrix_count": 4,
            "fresh_source_count": 1,
            "running_source_count": 14,
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
            "decomposition_children": 14,
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
        "step_program": {
            "one": one.clone(),
            "matrix_count": 4,
            "polynomial_degree": 3,
            "polynomial": [
                { "sign": "positive", "exponents": [1, 1, 0, 1] },
                { "sign": "negative", "exponents": [0, 0, 1, 1] },
            ],
            "receipts": step_receipts,
        },
        "terminal_program": {
            "one": one,
            "receipts": terminal_receipts,
        },
        "terminal_r1cs": {
            "row_variables": 0,
            "logical_width": 270,
            "recursive_rows": 1,
            "fresh_relation_rows": 2,
            "fresh_relation_auxiliary_columns": 1,
            "matrix_count": 4,
            "public_ring_columns": 5,
            "verifier_rows": 18,
            "cost": {
                "recurring_rows": 34_292,
                "committed_columns": 4_050,
                "public_columns": 26_191,
                "auxiliary_columns": 4_051,
            },
        },
        "step_result_columns": step_result_columns,
        "step_selector": step_selector,
        "terminal_selector": terminal_selector,
        "step_activations": [true_activation, false_activation],
        "terminal_activations": [terminal_true, terminal_false],
        "step_cost": {
            "recurring_rows": 1,
            "committed_columns": 22,
            "public_columns": 6,
            "auxiliary_columns": 242,
        },
        "terminal_cost": {
            "recurring_rows": 1,
            "committed_columns": 12,
            "public_columns": 6,
            "auxiliary_columns": 3,
        },
    })
}

pub(super) fn lifecycle_manifest() -> Value {
    let mut manifest = valid_manifest();
    let one = manifest["step_program"]["one"].clone();
    let receipts = manifest["step_program"]["receipts"]
        .as_array_mut()
        .expect("receipt array");
    let selected = receipts.last_mut().expect("selected receipt");
    let carried = selected["rows"][0]["b"].clone();
    selected["rows"][0]["a"] = json!([term(one)]);
    selected["rows"][0]["c"] = carried;
    let mut second_row = selected["rows"][0].clone();
    second_row["id"]["ordinal"] = json!(1);
    selected["rows"]
        .as_array_mut()
        .expect("selected rows")
        .push(second_row);
    manifest["step_cost"]["recurring_rows"] = json!(2);
    manifest["terminal_r1cs"]["row_variables"] = json!(1);
    manifest["terminal_r1cs"]["recursive_rows"] = json!(2);
    manifest["terminal_r1cs"]["fresh_relation_rows"] = json!(4);
    manifest["terminal_r1cs"]["fresh_relation_auxiliary_columns"] = json!(2);
    manifest["terminal_r1cs"]["cost"]["recurring_rows"] = json!(34_294);
    manifest["terminal_r1cs"]["cost"]["auxiliary_columns"] = json!(4_052);
    manifest
}

pub(super) fn parse(value: &Value) -> Result<LeanNativeCcsManifest, String> {
    LeanNativeCcsManifest::from_json_slice(&serde_json::to_vec(value).unwrap()).map_err(|error| error.to_string())
}

pub(super) fn combined_polynomial() -> Vec<Value> {
    let minus_one = GOLDILOCKS_MODULUS - 1;
    let term = |coefficient: u64, powers: &[(usize, u32)]| {
        let mut exponents = vec![0u32; 19];
        for &(matrix, exponent) in powers {
            exponents[matrix] = exponent;
        }
        json!({ "coefficient": coefficient, "exponents": exponents })
    };
    vec![
        term(1, &[(0, 1), (1, 1), (3, 1)]),
        term(minus_one, &[(2, 1), (3, 1)]),
        term(1, &[(4, 2)]),
        term(minus_one, &[(4, 1)]),
        term(1, &[(5, 1), (6, 1)]),
        term(1, &[(7, 1)]),
        term(minus_one, &[(8, 1)]),
        term(minus_one, &[(9, 1)]),
        term(1, &[(10, 1), (12, 1)]),
        term(1, &[(10, 1), (13, 1), (14, 1)]),
        term(minus_one, &[(10, 1), (13, 1), (16, 1), (18, 1)]),
        term(1, &[(11, 1), (13, 1), (15, 1)]),
        term(minus_one, &[(11, 1), (13, 1), (17, 1), (18, 1)]),
    ]
}

fn combined_core(native: &Value) -> Value {
    json!({
        "widths": native["widths"].clone(),
        "step_input": native["step_input"].clone(),
        "step_result": native["step_result"].clone(),
        "terminal_input": native["terminal_input"].clone(),
        "step_program": native["step_program"].clone(),
        "terminal_program": native["terminal_program"].clone(),
        "step_result_columns": native["step_result_columns"].clone(),
        "step_selector": native["step_selector"].clone(),
        "terminal_selector": native["terminal_selector"].clone(),
        "step_activations": native["step_activations"].clone(),
        "terminal_activations": native["terminal_activations"].clone(),
        "step_cost": native["step_cost"].clone(),
        "terminal_cost": native["terminal_cost"].clone(),
    })
}

pub(super) fn combined_manifest() -> Value {
    let native = valid_manifest();
    let core = combined_core(&native);
    let empty = json!([]);
    let images = json!({
        "bit": [{ "column": 1, "coefficient": 1 }],
        "product_left": empty,
        "product_right": [],
        "linear_left": [],
        "linear_right": [],
        "output": [],
        "extension_a": [],
        "extension_b": [],
        "pad": [],
        "active": [],
        "fingerprint_a": [],
        "fingerprint_b": [],
        "value_a": [],
        "value_b": [],
        "value": [],
    });
    json!({
        "schema": 4,
        "format": "nightstream/fprime-nebula-combined-manifest",
        "goldilocks_modulus": GOLDILOCKS_MODULUS,
        "ajtai_setup": native["ajtai_setup"].clone(),
        "core": core,
        "relation": {
            "matrix_count": 19,
            "strict_degree_bound": 5,
            "fresh_source_count": 1,
            "running_source_count": 14,
            "polynomial": combined_polynomial(),
            "layout": {
                "row_variables": 1,
                "native_logical_width": 270,
                "native_rows": 1,
                "native_public_width": 257,
                "combined_logical_width": 284,
                "combined_public_width": 270,
                "nebula_column_count": 2,
                "nebula_public_end": 1,
                "nebula_private_width": 1,
            },
            "application": {
                "matrix_count": 15,
                "strict_degree_bound": 5,
                "column_count": 2,
                "public_end": 1,
                "rows": [{
                    "id": {
                        "family": "operation_bit",
                        "slot": 0,
                        "component": 0,
                        "ordinal": 0,
                        "position": 0,
                    },
                    "images": images,
                }],
            },
        },
        "terminal_r1cs": {
            "row_variables": 1,
            "logical_width": 284,
            "recursive_rows": 2,
            "fresh_relation_rows": 3,
            "fresh_relation_auxiliary_columns": 1,
            "matrix_count": 19,
            "public_ring_columns": 5,
            "verifier_rows": 18,
            "cost": {
                "recurring_rows": 58_593,
                "committed_columns": 4_860,
                "public_columns": 48_871,
                "auxiliary_columns": 4_861,
            },
        },
    })
}

pub(super) fn combined_lifecycle_manifest() -> Value {
    let native = lifecycle_manifest();
    let mut manifest = combined_manifest();
    manifest["core"] = combined_core(&native);
    manifest["relation"]["layout"]["row_variables"] = json!(2);
    manifest["relation"]["layout"]["native_rows"] = json!(2);
    manifest["terminal_r1cs"]["row_variables"] = json!(2);
    manifest["terminal_r1cs"]["recursive_rows"] = json!(3);
    manifest["terminal_r1cs"]["fresh_relation_rows"] = json!(5);
    manifest["terminal_r1cs"]["fresh_relation_auxiliary_columns"] = json!(2);
    manifest["terminal_r1cs"]["cost"]["recurring_rows"] = json!(58_595);
    manifest["terminal_r1cs"]["cost"]["auxiliary_columns"] = json!(4_862);
    manifest
}

pub(super) fn combined_public_suffix_manifest() -> Value {
    let mut manifest = combined_manifest();
    manifest["relation"]["layout"]["combined_logical_width"] = json!(283);
    manifest["relation"]["layout"]["nebula_public_end"] = json!(2);
    manifest["relation"]["layout"]["nebula_private_width"] = json!(0);
    manifest["relation"]["application"]["public_end"] = json!(2);
    manifest["terminal_r1cs"]["logical_width"] = json!(283);
    manifest
}

pub(super) fn extension_combined_manifest() -> Value {
    let mut manifest = combined_manifest();
    let row = &mut manifest["relation"]["application"]["rows"][0];
    row["id"]["family"] = json!("read_product");
    row["images"]["bit"] = json!([]);
    row["images"]["output"] = json!([{ "column": 1, "coefficient": 1 }]);
    row["images"]["extension_a"] = json!([{ "column": 0, "coefficient": 1 }]);
    row["images"]["pad"] = json!([{ "column": 0, "coefficient": 1 }]);
    manifest["terminal_r1cs"]["fresh_relation_rows"] = json!(8);
    manifest["terminal_r1cs"]["fresh_relation_auxiliary_columns"] = json!(6);
    manifest["terminal_r1cs"]["cost"]["recurring_rows"] = json!(58_598);
    manifest["terminal_r1cs"]["cost"]["auxiliary_columns"] = json!(4_866);
    manifest
}

pub(super) fn parse_combined(value: &Value) -> Result<LeanNebulaCombinedManifest, String> {
    LeanNebulaCombinedManifest::from_json_slice(&serde_json::to_vec(value).unwrap()).map_err(|error| error.to_string())
}
