//! The Rust PiCCS field numerator must equal the Lean-proved selected test
//! error, `VerifierErrorBudget.test_error_eq_selected`, for the same counts.

use std::{fs, path::PathBuf};

use neo_params::pi_ccs_padded_row_field_numerator;

fn lean_artifact() -> [u64; 8] {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-piccs-test-error-v1.json");
    let bytes = fs::read(path).expect("Lean PiCCS test-error artifact");
    serde_json::from_slice(&bytes).expect("schema-1 Lean PiCCS test-error artifact")
}

#[test]
fn field_numerator_matches_lean_selected_test_error() {
    let [schema, cube_variables, width, fresh, running, ring_degree, matrices, numerator] = lean_artifact();
    assert_eq!(schema, 1, "Lean PiCCS test-error schema");
    let (sumcheck, mixing) = pi_ccs_padded_row_field_numerator(
        u32::try_from(cube_variables).unwrap(),
        u32::try_from(width).unwrap(),
        fresh.into(),
        running.into(),
        ring_degree.into(),
        matrices.into(),
    )
    .unwrap();
    assert_eq!(
        sumcheck + mixing,
        u128::from(numerator),
        "Rust PiCCS field numerator differs from the Lean test error"
    );
}
