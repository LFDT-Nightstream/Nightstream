//! Exhaustive execution of the Lean-authored per-application matrix program.
//!
//! This gate validates the sealed interpreter over every live logical row.
//! It does not replace the independent Lean-row comparison or raw-assignment
//! evaluator required for final conformance.

use std::{fs, path::PathBuf};

use nightstream_fprime::load_poseidon2_hash_chain_v1_package;

const LOGICAL_ROWS: usize = 6_377_559;
const LOGICAL_COLUMNS: usize = 264_627_433;
const ZERO_MATRIX: usize = 13;

fn artifact_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "../../formal/nightstream-fprime/artifacts/\
         nightstream-fprime-stage1-poseidon2-hash-chain-v1.json",
    )
}

#[test]
#[ignore = "full logical-matrix traversal; run this target explicitly under the 300-second cap"]
fn every_sealed_logical_matrix_row_is_canonical_and_in_range() {
    let bytes = fs::read(artifact_path()).expect("Lean-emitted per-application package");
    let package = load_poseidon2_hash_chain_v1_package(&bytes).expect("verifier-owned production package");
    assert_eq!(package.row_count(), LOGICAL_ROWS);
    assert_eq!(package.logical_column_count(), LOGICAL_COLUMNS);

    let nonzeros = package
        .validate_all_matrix_rows()
        .expect("every Lean-authored logical matrix row");

    assert_eq!(nonzeros[ZERO_MATRIX], 0);
    eprintln!("sealed_logical_matrix_nonzeros={nonzeros:?}");
}
