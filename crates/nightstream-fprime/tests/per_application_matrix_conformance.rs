//! Exact final A/B/C comparison against the separate Lean final-package
//! expansion and owner-family mutation checks on the sealed identity.

use std::{fs, path::PathBuf};

use nightstream_fprime::load_poseidon2_hash_chain_v1_package;
use serde_json::Value;

#[allow(dead_code, unused_imports)]
#[path = "../src/bin/check_package_conformance/support.rs"]
mod conformance_support;

fn artifact_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/nightstream-fprime/artifacts")
        .join(name)
}

#[test]
#[ignore = "exact final Lean-row comparison; run this target explicitly under the 300-second cap"]
fn final_matrices_equal_the_separate_lean_expansion() {
    let sealed_bytes = fs::read(artifact_path("nightstream-fprime-stage1-poseidon2-hash-chain-v1.json"))
        .expect("Lean-emitted sealed package");
    let expanded_bytes = fs::read(artifact_path(
        "nightstream-fprime-stage1-poseidon2-hash-chain-v1-expanded.json",
    ))
    .expect("separate Lean final-package expansion");

    let sealed: Value = serde_json::from_slice(&sealed_bytes).expect("sealed package JSON");
    let mut sealed_inner = serde_json::to_vec(&sealed[1]).expect("canonical sealed inner package");
    sealed_inner.push(b'\n');
    assert_eq!(sealed_inner, expanded_bytes, "sealed package and Lean final expansion");

    let package = load_poseidon2_hash_chain_v1_package(&sealed_bytes).expect("verifier-owned production package");
    let matrices = package
        .r1cs_matrices()
        .expect("final production A/B/C matrices");
    let (nonzeros, row_mutations, column_mutations) =
        conformance_support::compare_lean_expanded_matrices(&expanded_bytes, &matrices);
    assert_eq!(nonzeros, [93_701_820, 39_358_148, 28_868_018]);
    assert_eq!(row_mutations, 156);
    assert_eq!(column_mutations, 81);
    eprintln!(
        "lean_final_matrix_nonzeros={nonzeros:?} row_owner_mutations={row_mutations} column_owner_mutations={column_mutations}"
    );
}
