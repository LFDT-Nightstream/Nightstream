//! Fail-closed tests for the Lean native-CCS manifest boundary.
//!
//! The fixture is small. It tests the exact wire contract and selector
//! semantics. It is not a production F′ cost measurement.

use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_fold_clean::frontends::direct_ccs::ajtai;
use neo_fold_clean::frontends::r1cs_f_prime::lean_manifest::{ColumnId, PhysicalOwner, TypedOwner, GOLDILOCKS_MODULUS};
use neo_fold_clean::frontends::r1cs_f_prime::lean_native_ccs_manifest::{
    LEAN_NATIVE_CCS_MANIFEST_FORMAT, LEAN_NATIVE_CCS_MANIFEST_SCHEMA_VERSION,
};
use neo_fold_clean::frontends::r1cs_f_prime::terminal_r1cs::{
    compile_combined_terminal_r1cs, compile_combined_terminal_r1cs_statement, compile_terminal_r1cs,
    compile_terminal_r1cs_statement, TerminalR1csInput, TerminalR1csStatement, TerminalSpartanEngine,
};
use neo_fold_clean::paper::construction2::{self, LatestInstance, ProofState, RunningInstance, State};
use neo_fold_clean::paper::digest::{
    digest_fields_as_digest32, f_prime_chunk_public_digest_for_uniform_shape, initial_boundary_digest,
    state_x_out_digest_with_mode, AccumulatorHandle, StateXOutDigestMode,
};
use neo_fold_clean::paper::f_prime::r1cs::encode_f_prime_superneo_public_input;
use neo_fold_clean::paper::params::Params;
use neo_fold_clean::paper::relations::{CcsClaim, CcsInstance, CeClaim, WitnessMat};
use neo_fold_clean::{
    finish_uncompressed, finish_with_spartan, prove, verify_spartan, verify_uncompressed, LeanNativeCcsManifest,
    LeanNativeCcsPreprocessing, LeanNebulaCombinedManifest, LeanNebulaCombinedPreprocessing, TerminalR1csError,
    Uncompressed,
};
use neo_math::{D, F, K};
use p3_field::PrimeCharacteristicRing;
use serde_json::{json, Value};
use toy_spartan::spartan::{RepeatedR1CSSNARK, R1CSSNARK};

const TEST_AJTAI_SEED: u64 = 0x5445_524d_494e_414c;

fn test_ajtai_seed() -> [u8; 32] {
    let mut seed = [0u8; 32];
    seed[..8].copy_from_slice(&TEST_AJTAI_SEED.to_le_bytes());
    seed
}

fn prelude_owner() -> Value {
    json!({ "kind": "prelude" })
}

fn input_owner(slot: usize) -> Value {
    json!({
        "kind": "typed",
        "owner": { "kind": "input", "slot": slot },
    })
}

fn instruction_owner(path: &[&str]) -> Value {
    json!({
        "kind": "typed",
        "owner": { "kind": "instruction", "path": path },
    })
}

fn branch_owner(path: &[&str]) -> Value {
    json!({
        "kind": "typed",
        "owner": { "kind": "branch", "path": path },
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
    json!({ "id": id, "ownership": ownership })
}

fn term(column: Value) -> Value {
    json!({ "column": column, "coefficient": 1 })
}

fn row(owner: Value, ordinal: usize, a: Value, b: Value, c: Value) -> Value {
    json!({
        "id": { "owner": owner, "ordinal": ordinal },
        "a": [term(a)],
        "b": [term(b)],
        "c": [term(c)],
    })
}

fn native_receipt(owner: Value, kind: &str, allocations: Vec<Value>, selector: Value, rows: Vec<Value>) -> Value {
    json!({
        "owner": owner,
        "kind": kind,
        "allocations": allocations,
        "selector": selector,
        "rows": rows,
    })
}

fn canonical_receipt(owner: Value, kind: &str, allocations: Vec<Value>, rows: Vec<Value>) -> Value {
    json!({
        "owner": owner,
        "kind": kind,
        "allocations": allocations,
        "rows": rows,
    })
}

fn segments(entries: &[(&str, usize, &str)]) -> Vec<Value> {
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

fn input_receipts(input_segments: &[Value], one: &Value, native: bool) -> Vec<Value> {
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

fn valid_manifest() -> Value {
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
                "recurring_rows": 32_780,
                "committed_columns": 4_050,
                "public_columns": 24_679,
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

fn lifecycle_manifest() -> Value {
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
    manifest["terminal_r1cs"]["cost"]["recurring_rows"] = json!(32_782);
    manifest["terminal_r1cs"]["cost"]["auxiliary_columns"] = json!(4_052);
    manifest
}

fn parse(value: &Value) -> Result<LeanNativeCcsManifest, String> {
    LeanNativeCcsManifest::from_json_slice(&serde_json::to_vec(value).unwrap()).map_err(|error| error.to_string())
}

fn combined_polynomial() -> Vec<Value> {
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

fn combined_manifest() -> Value {
    let native = valid_manifest();
    let core = json!({
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
    });
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
                "recurring_rows": 57_081,
                "committed_columns": 4_860,
                "public_columns": 47_359,
                "auxiliary_columns": 4_861,
            },
        },
    })
}

fn extension_combined_manifest() -> Value {
    let mut manifest = combined_manifest();
    let row = &mut manifest["relation"]["application"]["rows"][0];
    row["id"]["family"] = json!("read_product");
    row["images"]["bit"] = json!([]);
    row["images"]["output"] = json!([{ "column": 1, "coefficient": 1 }]);
    row["images"]["extension_a"] = json!([{ "column": 0, "coefficient": 1 }]);
    row["images"]["pad"] = json!([{ "column": 0, "coefficient": 1 }]);
    manifest["terminal_r1cs"]["fresh_relation_rows"] = json!(8);
    manifest["terminal_r1cs"]["fresh_relation_auxiliary_columns"] = json!(6);
    manifest["terminal_r1cs"]["cost"]["recurring_rows"] = json!(57_086);
    manifest["terminal_r1cs"]["cost"]["auxiliary_columns"] = json!(4_866);
    manifest
}

fn parse_combined(value: &Value) -> Result<LeanNebulaCombinedManifest, String> {
    LeanNebulaCombinedManifest::from_json_slice(&serde_json::to_vec(value).unwrap()).map_err(|error| error.to_string())
}

fn field_value(column: &ColumnId, active: bool, valid: bool) -> F {
    match &column.owner {
        PhysicalOwner::BranchActivation { selected, .. } => {
            if *selected == !active {
                F::ONE
            } else {
                F::ZERO
            }
        }
        PhysicalOwner::Typed {
            owner: TypedOwner::Input { slot: 1 },
        } if column.coordinate_index == 0 => F::ONE,
        PhysicalOwner::Typed {
            owner: TypedOwner::Input { slot: 1 },
        } if column.coordinate_index == 1 => F::ONE,
        PhysicalOwner::Typed {
            owner: TypedOwner::Input { slot: 2 },
        } if column.coordinate_index == 0 => {
            if valid {
                F::ONE
            } else {
                F::ZERO
            }
        }
        _ => F::ONE,
    }
}

fn direct_terminal_fixture(
    manifest: &LeanNativeCcsManifest,
) -> (neo_ajtai::AjtaiSModule, Vec<CeClaim>, Vec<WitnessMat>, CcsInstance) {
    let step = manifest
        .emit_phi81_step(|column| Some(field_value(column, true, true)))
        .expect("honest Phi81 Step");
    assert!(step.is_satisfied());
    let params = Params::goldilocks_paper_b2();
    let log = ajtai::setup_seeded(&params, step.structure(), TEST_AJTAI_SEED);
    let fresh = CcsInstance::from_low_norm_assignment(
        &params,
        &log,
        step.structure(),
        step.assignment(),
        manifest.public_carrier_width(),
    )
    .expect("honest fresh instance");
    let zero_witness = Mat::zero(D, step.structure().m / D, F::ZERO);
    let zero_claim = CeClaim {
        c: Commitment::zeros(D, manifest.terminal_r1cs().verifier_rows()),
        X: Mat::zero(D, manifest.public_carrier_width(), F::ZERO),
        r: Vec::new(),
        y_ring: vec![vec![K::ZERO; D.next_power_of_two()]; step.structure().t()],
        ct: vec![K::ZERO; step.structure().t()],
        m_in: manifest.public_carrier_width(),
        fold_digest: [0; 32],
        adv: None,
    };
    (
        log,
        vec![zero_claim; manifest.running_claim_count()],
        vec![zero_witness; manifest.running_claim_count()],
        fresh,
    )
}

fn direct_combined_terminal_fixture(
    manifest: &LeanNebulaCombinedManifest,
    nebula_private: &[F],
) -> (neo_ajtai::AjtaiSModule, Vec<CeClaim>, Vec<WitnessMat>, CcsInstance) {
    let mut public = vec![F::ZERO; manifest.public_carrier_width()];
    public[0] = F::ONE;
    let emission = manifest
        .emit(&public, |_| Some(F::ZERO), nebula_private)
        .expect("honest combined emission");
    assert!(emission.is_satisfied());
    let params = Params::goldilocks_paper_b2();
    let log = ajtai::setup_seeded(&params, emission.structure(), TEST_AJTAI_SEED);
    let fresh = CcsInstance::from_low_norm_assignment(
        &params,
        &log,
        emission.structure(),
        emission.assignment(),
        manifest.public_carrier_width(),
    )
    .expect("honest combined fresh instance");
    let zero_witness = Mat::zero(D, emission.structure().m / D, F::ZERO);
    let joint_row_variables = emission
        .structure()
        .n
        .max(emission.structure().m)
        .next_power_of_two()
        .max(2)
        .trailing_zeros() as usize;
    let zero_claim = CeClaim {
        c: Commitment::zeros(D, manifest.terminal_r1cs().verifier_rows()),
        X: Mat::zero(D, manifest.public_carrier_width(), F::ZERO),
        r: vec![K::ZERO; joint_row_variables],
        y_ring: vec![vec![K::ZERO; D.next_power_of_two()]; emission.structure().t()],
        ct: vec![K::ZERO; emission.structure().t()],
        m_in: manifest.public_carrier_width(),
        fold_digest: [0; 32],
        adv: None,
    };
    (log, vec![zero_claim; 14], vec![zero_witness; 14], fresh)
}

fn terminal_lifecycle_fixture(manifest: &LeanNativeCcsManifest) -> (neo_fold_clean::Preprocessing, Uncompressed) {
    let probe = manifest
        .emit_phi81_step(|_| Some(F::ZERO))
        .expect("terminal relation probe");
    let params = neo_fold_clean::config::ccs_params(
        probe.structure().n,
        probe.structure().m,
        probe.structure().t(),
        probe.structure().max_degree(),
    )
    .expect("shape-derived native CCS parameters");
    let log = ajtai::setup_seeded(&params, probe.structure(), TEST_AJTAI_SEED);
    let prep = neo_fold_clean::lifecycle::preprocess_with_test_log(
        params.clone(),
        probe.structure().clone(),
        log.clone(),
        Some(manifest.public_carrier_width()),
    )
    .expect("terminal preprocessing");

    let zero_witness = Mat::zero(D, probe.structure().m / D, F::ZERO);
    let zero_claim = CeClaim {
        c: Commitment::zeros(D, manifest.terminal_r1cs().verifier_rows()),
        X: Mat::zero(D, manifest.public_carrier_width(), F::ZERO),
        r: Vec::new(),
        y_ring: vec![vec![K::ZERO; D.next_power_of_two()]; probe.structure().t()],
        ct: vec![K::ZERO; probe.structure().t()],
        m_in: manifest.public_carrier_width(),
        fold_digest: [0; 32],
        adv: None,
    };
    let running_claims = vec![zero_claim.clone(); manifest.running_claim_count()];
    let running_witnesses = vec![zero_witness; manifest.running_claim_count()];
    let acc_digest = AccumulatorHandle::from_claims(&running_claims).digest();
    let z_0 = initial_boundary_digest(prep.structure_digest(), prep.public_input_len);

    let placeholder = CcsClaim {
        c: Commitment::zeros(D, manifest.terminal_r1cs().verifier_rows()),
        x: vec![F::ZERO; manifest.public_carrier_width()],
        m_in: manifest.public_carrier_width(),
        adv: None,
    };
    let false_activation = manifest
        .step_program()
        .receipts
        .iter()
        .flat_map(|receipt| &receipt.allocations)
        .find(|allocation| {
            matches!(
                allocation.id.owner,
                PhysicalOwner::BranchActivation { selected: false, .. }
            )
        })
        .expect("false activation")
        .id
        .clone();
    let row = manifest
        .step_program()
        .receipts
        .iter()
        .flat_map(|receipt| &receipt.rows)
        .next()
        .expect("one selected row");
    let row_indices = [
        probe.column_index(&row.a[0].column).expect("A column"),
        probe.column_index(&row.b[0].column).expect("B column"),
        probe.column_index(&row.c[0].column).expect("C column"),
    ];
    let activation_index = probe
        .column_index(&false_activation)
        .expect("activation column");

    let mut selected = None;
    for step_count in 1..=256u64 {
        let boundary = digest_fields_as_digest32(f_prime_chunk_public_digest_for_uniform_shape(
            step_count - 1,
            1,
            placeholder.c.d,
            placeholder.c.kappa,
            placeholder.m_in,
        ));
        let digest = state_x_out_digest_with_mode(
            StateXOutDigestMode::Stateless,
            prep.vk.digest(),
            prep.pi_ccs_header_bundle(),
            prep.structure_digest(),
            step_count,
            step_count,
            z_0,
            boundary,
            neo_fold_clean::paper::construction2::TRIVIAL_PC,
            acc_digest,
            acc_digest,
            boundary,
            None,
        );
        let carrier = encode_f_prime_superneo_public_input(neo_fold_clean::paper::digest::digest32_as_fields(digest));
        let row_ok = carrier[activation_index] == F::ZERO
            || carrier[row_indices[0]] * carrier[row_indices[1]] == carrier[row_indices[2]];
        if row_ok {
            selected = Some((step_count, boundary, carrier));
            break;
        }
    }
    let (step_count, boundary, assignment) = selected.expect("bounded terminal-link fixture");
    let emission = manifest
        .emit_phi81_step(|column| probe.column_index(column).map(|index| assignment[index]))
        .expect("linked terminal relation");
    assert!(emission.is_satisfied());
    let fresh = CcsInstance::from_low_norm_assignment(
        &params,
        &log,
        emission.structure(),
        emission.assignment(),
        manifest.public_carrier_width(),
    )
    .expect("linked fresh instance");
    assert_eq!(
        boundary,
        digest_fields_as_digest32(f_prime_chunk_public_digest_for_uniform_shape(
            step_count - 1,
            1,
            fresh.claim.c.d,
            fresh.claim.c.kappa,
            fresh.claim.m_in,
        ))
    );

    let running = RunningInstance::new(running_claims, running_witnesses, Some(zero_claim));
    let state = State {
        chunk_count: step_count,
        step_count,
        z_0,
        z_i: boundary,
        pc: neo_fold_clean::paper::construction2::TRIVIAL_PC,
        initial_semantic_state_digest: prep.initial_semantic_state_digest(),
        semantic_state_digest: acc_digest,
        acc_digest,
        public_trace: boundary,
        proof: ProofState::active(running, LatestInstance::from_instances(vec![fresh])),
        nebula: None,
    };
    (
        prep,
        Uncompressed {
            state,
            final_fold: None,
        },
    )
}

#[test]
fn emits_the_exact_four_matrix_native_selector() {
    let manifest = parse(&valid_manifest()).expect("valid native manifest");
    let emission = manifest
        .emit_step(|column| Some(field_value(column, true, true)))
        .expect("emit native Step");

    assert_eq!(manifest.matrix_count(), 4);
    assert_eq!(manifest.polynomial_degree(), 3);
    assert_eq!(emission.structure().t(), 4);
    assert_eq!(emission.structure().max_degree(), 3);
    assert_eq!(emission.structure().m, 270);
    assert_eq!(emission.structure().n, 1);
    assert!(emission.is_satisfied());
}

#[test]
fn preprocessing_and_instance_are_derived_from_the_manifest() {
    let manifest = parse(&valid_manifest()).expect("valid native manifest");
    let setup = LeanNativeCcsPreprocessing::new(manifest).expect("manifest-owned native preprocessing");

    assert!(setup.preprocessing().enforces_terminal_induction());
    assert_eq!(
        setup.preprocessing().public_input_len,
        Some(setup.manifest().public_carrier_width())
    );
    assert_eq!(
        setup.preprocessing().log.seeded_params(),
        Some((
            setup.manifest().terminal_r1cs().verifier_rows(),
            setup.manifest().ajtai_setup_seed(),
        ))
    );

    let instance = setup
        .build_instance(|column| Some(field_value(column, true, true)))
        .expect("satisfying manifest assignment");
    assert_eq!(instance.claim.m_in, setup.manifest().public_carrier_width());
}

#[test]
#[ignore = "runs one complete native F-prime fold and the 32,780-row Spartan/WHIR terminal proof"]
fn manifest_owned_lifecycle_proves_and_verifies() {
    let manifest = parse(&lifecycle_manifest()).expect("valid native lifecycle manifest");
    let setup = LeanNativeCcsPreprocessing::new(manifest).expect("manifest-owned native preprocessing");
    let prep = setup.preprocessing();

    let base = prove(prep, Vec::<Vec<CcsInstance>>::new()).expect("base lifecycle state");
    let mut post_state = base.proof.state.clone();
    let boundary = digest_fields_as_digest32(f_prime_chunk_public_digest_for_uniform_shape(
        0,
        1,
        D,
        prep.params.kappa() as usize,
        setup.manifest().public_carrier_width(),
    ));
    post_state.chunk_count = 1;
    post_state.step_count = 1;
    post_state.z_i = boundary;
    post_state.public_trace = boundary;
    let default_running = RunningInstance::canonical_zero(
        &prep.params,
        prep.structure(),
        setup.manifest().public_carrier_width(),
        neo_fold_clean::paper::construction2::LaneCommitmentMode::Plain,
    )
    .expect("paper default running accumulator");
    post_state.acc_digest = AccumulatorHandle::from_claims(&default_running.claims).digest();
    post_state.semantic_state_digest = post_state.acc_digest;
    let state = &post_state;
    let mode = match prep.semantic_state_mode() {
        construction2::SemanticStateMode::Stateless => StateXOutDigestMode::Stateless,
        construction2::SemanticStateMode::Stateful => StateXOutDigestMode::Stateful,
    };
    let expected = state_x_out_digest_with_mode(
        mode,
        prep.vk.digest(),
        prep.pi_ccs_header_bundle(),
        prep.structure_digest(),
        state.chunk_count,
        state.step_count,
        state.z_0,
        state.z_i,
        state.pc,
        state.semantic_state_digest,
        state.acc_digest,
        state.public_trace,
        None,
    );
    let carrier = encode_f_prime_superneo_public_input(neo_fold_clean::paper::digest::digest32_as_fields(expected));

    let baseline = setup
        .manifest()
        .emit_phi81_step(|column| Some(field_value(column, true, true)))
        .expect("baseline manifest assignment");
    let mut assignment = baseline.assignment().to_vec();
    assignment[..carrier.len()].copy_from_slice(&carrier);
    let instance = setup
        .build_instance(|column| baseline.column_index(column).map(|index| assignment[index]))
        .expect("base-linked manifest instance");
    assert_eq!(instance.claim.x, carrier);
    let audit = prove(prep, [vec![instance]]).expect("one native F-prime fold");
    assert_eq!(audit.proof.state.acc_digest, post_state.acc_digest);
    assert_eq!(
        audit.proof.state.semantic_state_digest,
        post_state.semantic_state_digest
    );
    let terminal = finish_uncompressed(prep, audit).expect("plain HyperNova running/latest terminal state");
    verify_uncompressed(prep, &terminal).expect("uncompressed terminal verification");

    let (statement, proof) =
        finish_with_spartan(prep, setup.manifest(), terminal).expect("terminal Spartan/WHIR proof");
    let public_image = statement.public_image().clone();
    verify_spartan(prep, setup.manifest(), &public_image, &statement, &proof)
        .expect("terminal Spartan/WHIR verification");
}

#[test]
fn active_selector_enforces_and_inactive_selector_disables() {
    let manifest = parse(&valid_manifest()).expect("valid native manifest");
    let invalid_active = manifest
        .emit_step(|column| Some(field_value(column, true, false)))
        .expect("emit active native Step");
    assert!(!invalid_active.is_satisfied());

    let invalid_inactive = manifest
        .emit_step(|column| Some(field_value(column, false, false)))
        .expect("emit inactive native Step");
    assert!(invalid_inactive.is_satisfied());
}

#[test]
fn native_step_has_no_residual_row_or_column() {
    let manifest = parse(&valid_manifest()).expect("valid native manifest");
    let emission = manifest
        .emit_step(|column| Some(field_value(column, true, true)))
        .expect("emit native Step");

    assert_eq!(manifest.step_cost().recurring_rows(), 1);
    assert_eq!(manifest.step_cost().auxiliary_columns(), 242);
    assert_eq!(emission.auxiliary_columns().len(), 242);
    assert_eq!(emission.structure().n, 1);
    assert_eq!(manifest.terminal_r1cs().logical_width(), 270);
    assert_eq!(manifest.terminal_r1cs().recursive_rows(), 1);
    assert_eq!(manifest.terminal_r1cs().cost().recurring_rows(), 32_780);
}

#[test]
fn rejects_polynomial_or_matrix_shape_drift() {
    let mut matrix_count = valid_manifest();
    matrix_count["step_program"]["matrix_count"] = json!(5);
    assert!(parse(&matrix_count).unwrap_err().contains("matrix_count"));

    let mut degree = valid_manifest();
    degree["step_program"]["polynomial_degree"] = json!(4);
    assert!(parse(&degree).unwrap_err().contains("polynomial_degree"));

    let mut exponent = valid_manifest();
    exponent["step_program"]["polynomial"][0]["exponents"][3] = json!(0);
    assert!(parse(&exponent).unwrap_err().contains("polynomial"));
}

#[test]
fn rejects_selector_substitution_and_wrong_selected_owner() {
    let mut selector = valid_manifest();
    let one = selector["step_program"]["one"].clone();
    let receipts = selector["step_program"]["receipts"].as_array_mut().unwrap();
    let target = receipts.len() - 1;
    receipts[target]["selector"] = one;
    assert!(parse(&selector).unwrap_err().contains("selected receipts"));

    let mut owner = valid_manifest();
    let receipts = owner["step_program"]["receipts"].as_array_mut().unwrap();
    let target = receipts.len() - 1;
    let wrong_owner = instruction_owner(&["rest"]);
    receipts[target]["owner"] = wrong_owner.clone();
    receipts[target]["rows"][0]["id"]["owner"] = wrong_owner.clone();
    for allocation in receipts[target]["allocations"].as_array_mut().unwrap() {
        allocation["id"]["owner"] = wrong_owner.clone();
    }
    assert!(parse(&owner)
        .unwrap_err()
        .contains("only native-selected receipt"));
}

#[test]
fn rejects_cost_drift_unknown_fields_and_noncanonical_coefficients() {
    let mut cost = valid_manifest();
    cost["step_cost"]["recurring_rows"] = json!(2);
    assert!(parse(&cost).unwrap_err().contains("step_cost"));

    let mut unknown = valid_manifest();
    unknown["rust_authority"] = json!(true);
    assert!(parse(&unknown).unwrap_err().contains("unknown field"));

    let mut coefficient = valid_manifest();
    let receipts = coefficient["step_program"]["receipts"]
        .as_array_mut()
        .unwrap();
    let target = receipts.len() - 1;
    receipts[target]["rows"][0]["a"][0]["coefficient"] = json!(GOLDILOCKS_MODULUS);
    assert!(parse(&coefficient)
        .unwrap_err()
        .contains("canonical Goldilocks residue"));
}

#[test]
fn rejects_invalid_or_mismatched_ajtai_setup() {
    let mut algorithm = valid_manifest();
    algorithm["ajtai_setup"]["algorithm"] = json!("different_sampler");
    assert!(parse(&algorithm).is_err());

    let mut short_seed = valid_manifest();
    short_seed["ajtai_setup"]["seed"] = json!(vec![0u8; 31]);
    assert!(parse(&short_seed).is_err());

    let mut no_fuel = valid_manifest();
    no_fuel["ajtai_setup"]["rejection_fuel"] = json!(0);
    assert!(parse(&no_fuel).unwrap_err().contains("rejection_fuel"));

    let mut different_seed = valid_manifest();
    different_seed["ajtai_setup"]["seed"][0] = json!(test_ajtai_seed()[0] ^ 1);
    let manifest = parse(&different_seed).expect("structurally valid alternate setup");
    let (log, running_claims, running_witnesses, fresh) = direct_terminal_fixture(&manifest);
    assert!(matches!(
        compile_terminal_r1cs(
            &manifest,
            &log,
            TerminalR1csInput {
                running_claims: &running_claims,
                running_witnesses: &running_witnesses,
                fresh: &fresh,
            },
        ),
        Err(TerminalR1csError::SetupMismatch)
    ));
}

#[test]
fn rejects_terminal_r1cs_shape_or_cost_drift() {
    let mut width = valid_manifest();
    width["terminal_r1cs"]["logical_width"] = json!(269);
    assert!(parse(&width).unwrap_err().contains("logical_width"));

    let mut rows = valid_manifest();
    rows["terminal_r1cs"]["recursive_rows"] = json!(2);
    assert!(parse(&rows).unwrap_err().contains("recursive_rows"));

    let mut domain = valid_manifest();
    domain["terminal_r1cs"]["row_variables"] = json!(1);
    assert!(parse(&domain).unwrap_err().contains("least power-of-two"));

    let mut matrix = valid_manifest();
    matrix["terminal_r1cs"]["matrix_count"] = json!(3);
    assert!(parse(&matrix).unwrap_err().contains("matrix_count"));

    let mut public = valid_manifest();
    public["terminal_r1cs"]["public_ring_columns"] = json!(4);
    assert!(parse(&public).unwrap_err().contains("public_ring_columns"));

    let mut verifier = valid_manifest();
    verifier["terminal_r1cs"]["verifier_rows"] = json!(17);
    assert!(parse(&verifier).unwrap_err().contains("verifier_rows"));

    let mut cost = valid_manifest();
    cost["terminal_r1cs"]["cost"]["auxiliary_columns"] = json!(4_052);
    assert!(parse(&cost).unwrap_err().contains("terminal_r1cs.cost"));
}

#[test]
#[ignore = "materializes the complete 14-running terminal reference relation"]
fn terminal_r1cs_compiles_with_the_exact_lean_cost() {
    let manifest = parse(&valid_manifest()).expect("valid native manifest");
    let (log, running_claims, running_witnesses, fresh) = direct_terminal_fixture(&manifest);
    let relation = compile_terminal_r1cs(
        &manifest,
        &log,
        TerminalR1csInput {
            running_claims: &running_claims,
            running_witnesses: &running_witnesses,
            fresh: &fresh,
        },
    )
    .expect("honest terminal R1CS");

    assert_eq!(relation.shape().num_constraints_unpadded(), 32_780);
    assert_eq!(relation.shape().num_rest_unpadded(), 8_101);
    assert_eq!(relation.shape().num_public(), 24_678);
    assert_eq!(relation.lean_public_columns(), 24_679);
}

#[test]
#[ignore = "proves the complete bounded terminal reference relation with WHIR"]
fn terminal_r1cs_proves_and_verifies_with_spartan_and_whir() {
    let manifest = parse(&valid_manifest()).expect("valid native manifest");
    let (log, running_claims, running_witnesses, fresh) = direct_terminal_fixture(&manifest);
    let relation = compile_terminal_r1cs(
        &manifest,
        &log,
        TerminalR1csInput {
            running_claims: &running_claims,
            running_witnesses: &running_witnesses,
            fresh: &fresh,
        },
    )
    .expect("honest terminal R1CS");
    let (_, witness, public) = relation.into_parts();
    let statement = compile_terminal_r1cs_statement(
        &manifest,
        &log,
        TerminalR1csStatement {
            running_claims: &running_claims,
            fresh_claim: &fresh.claim,
        },
    )
    .expect("verifier terminal R1CS");
    assert_eq!(statement.public_values(), public);
    let (shape, verifier_public) = statement.into_parts();
    let (prover_key, verifier_key) =
        R1CSSNARK::<TerminalSpartanEngine>::setup_direct(shape).expect("direct Spartan setup");
    let proof = RepeatedR1CSSNARK::<TerminalSpartanEngine>::prove_direct(&prover_key, &witness, &public, true)
        .expect("direct Spartan proof");

    assert_eq!(proof.verify(&verifier_key).expect("WHIR verification"), verifier_public);
}

#[test]
fn terminal_lifecycle_rejects_preprocessing_without_recursive_induction() {
    let manifest = parse(&valid_manifest()).expect("valid native manifest");
    let (prep, uncompressed) = terminal_lifecycle_fixture(&manifest);
    assert!(matches!(
        finish_with_spartan(&prep, &manifest, uncompressed),
        Err(TerminalR1csError::UncertifiedInduction)
    ));
}

#[test]
fn combined_manifest_emits_the_exact_nineteen_matrix_relation() {
    let manifest = parse_combined(&combined_manifest()).expect("valid combined manifest");
    let mut public = vec![F::ZERO; manifest.public_carrier_width()];
    public[0] = F::ONE;
    let private = vec![F::ZERO; manifest.nebula_private_width()];
    let emission = manifest
        .emit(&public, |_| Some(F::ZERO), &private)
        .expect("exact combined emission");

    assert_eq!(manifest.core().matrix_count(), 4);
    assert_eq!(manifest.matrix_count(), 19);
    assert_eq!(manifest.strict_degree_bound(), 5);
    assert_eq!(emission.structure().t(), 19);
    assert_eq!(emission.structure().max_degree(), 4);
    assert_eq!(emission.structure().n, 2);
    assert_eq!(emission.structure().m, 324);
    assert_eq!(emission.logical_width(), 284);
    assert_eq!(emission.public_width(), 270);
    assert!(emission.is_satisfied());
}

#[test]
fn combined_preprocessing_and_instance_are_manifest_owned() {
    let manifest = parse_combined(&combined_manifest()).expect("valid combined manifest");
    let setup = LeanNebulaCombinedPreprocessing::new(manifest).expect("combined preprocessing");
    let mut public = vec![F::ZERO; setup.manifest().public_carrier_width()];
    public[0] = F::ONE;
    let private = vec![F::ZERO; setup.manifest().nebula_private_width()];
    let instance = setup
        .build_instance(&public, |_| Some(F::ZERO), &private)
        .expect("combined instance");

    assert!(setup.preprocessing().enforces_terminal_induction());
    assert_eq!(setup.preprocessing().structure().t(), 19);
    assert_eq!(instance.claim.m_in, 270);
    assert_eq!(instance.claim.x, public);
}

#[test]
fn combined_terminal_r1cs_compiles_the_exact_lean_bit_lowering() {
    let manifest = parse_combined(&combined_manifest()).expect("valid combined manifest");
    let (log, running_claims, running_witnesses, fresh) = direct_combined_terminal_fixture(&manifest, &[F::ZERO]);
    let relation = compile_combined_terminal_r1cs(
        &manifest,
        &log,
        TerminalR1csInput {
            running_claims: &running_claims,
            running_witnesses: &running_witnesses,
            fresh: &fresh,
        },
    )
    .expect("honest combined terminal R1CS");
    let statement = compile_combined_terminal_r1cs_statement(
        &manifest,
        &log,
        TerminalR1csStatement {
            running_claims: &running_claims,
            fresh_claim: &fresh.claim,
        },
    )
    .expect("combined terminal statement");

    assert_eq!(relation.shape().num_constraints_unpadded(), 57_081);
    assert_eq!(relation.shape().num_rest_unpadded(), 9_721);
    assert_eq!(relation.shape().num_public(), 47_358);
    assert_eq!(relation.lean_public_columns(), 47_359);
    assert_eq!(statement.shape(), relation.shape());
    assert_eq!(statement.public_values(), relation.public_values());
}

#[test]
fn combined_terminal_r1cs_compiles_the_exact_lean_extension_lowering() {
    let manifest = parse_combined(&extension_combined_manifest()).expect("valid extension manifest");
    let (log, running_claims, running_witnesses, fresh) = direct_combined_terminal_fixture(&manifest, &[F::ONE]);
    let relation = compile_combined_terminal_r1cs(
        &manifest,
        &log,
        TerminalR1csInput {
            running_claims: &running_claims,
            running_witnesses: &running_witnesses,
            fresh: &fresh,
        },
    )
    .expect("honest extension terminal R1CS");

    assert_eq!(relation.shape().num_constraints_unpadded(), 57_086);
    assert_eq!(relation.shape().num_rest_unpadded(), 9_726);
    assert_eq!(relation.shape().num_public(), 47_358);
    assert_eq!(relation.lean_public_columns(), 47_359);
}

#[test]
fn combined_manifest_rejects_relation_and_layout_drift() {
    let mut arity = combined_manifest();
    arity["relation"]["matrix_count"] = json!(4);
    assert!(parse_combined(&arity).unwrap_err().contains("matrix_count"));

    let mut polynomial = combined_manifest();
    polynomial["relation"]["polynomial"][0]["coefficient"] = json!(2);
    assert!(parse_combined(&polynomial)
        .unwrap_err()
        .contains("polynomial"));

    let mut position = combined_manifest();
    position["relation"]["application"]["rows"][0]["id"]["position"] = json!(1);
    assert!(parse_combined(&position).unwrap_err().contains("position"));

    let mut coefficient = combined_manifest();
    coefficient["relation"]["application"]["rows"][0]["images"]["bit"][0]["coefficient"] = json!(0);
    assert!(parse_combined(&coefficient)
        .unwrap_err()
        .contains("coefficient"));

    let mut layout = combined_manifest();
    layout["relation"]["layout"]["combined_logical_width"] = json!(285);
    assert!(parse_combined(&layout)
        .unwrap_err()
        .contains("combined_logical_width"));

    let mut terminal = combined_manifest();
    terminal["terminal_r1cs"]["cost"]["auxiliary_columns"] = json!(4_863);
    assert!(parse_combined(&terminal)
        .unwrap_err()
        .contains("terminal_r1cs.cost"));
}

#[test]
fn combined_emission_rejects_shared_and_witness_drift() {
    let manifest = parse_combined(&combined_manifest()).expect("valid combined manifest");
    let private = vec![F::ZERO; manifest.nebula_private_width()];

    let public = vec![F::ZERO; manifest.public_carrier_width()];
    assert!(manifest.emit(&public, |_| Some(F::ZERO), &private).is_err());

    let mut public = vec![F::ZERO; manifest.public_carrier_width()];
    public[0] = F::ONE;
    assert!(manifest.emit(&public, |_| Some(F::ZERO), &[]).is_err());
    assert!(manifest.emit(&public, |_| Some(F::ONE), &private).is_err());
}
