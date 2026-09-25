//! Integration checks for the actual producer and retained Lean comparison.

#[path = "stage1_nifs.rs"]
pub mod runner;

use runner::{compare, prove};
use serde_json::Value;
use std::{fs, path::Path};

fn artifact(name: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/nightstream-fprime/artifacts")
        .join(name)
}

#[test]
#[ignore = "The complete producer exceeds the 300-second cap; use the separately capped stages and saved-result comparison."]
fn selected_nifs_prover_from_actual_base_sources() {
    // A missing result is an error before the expensive prover starts.
    let expected: Value = serde_json::from_slice(
        &fs::read(artifact("nightstream-fprime-stage1-base-nifs-result-v1.json"))
            .expect("independent checked Lean base C/R/D result"),
    )
    .unwrap();
    let actual = prove(
        &artifact("nightstream-fprime-stage1-poseidon2-hash-chain-v1.json"),
        &artifact("nightstream-fprime-stage1-base-step-fixture-v1.json"),
    );
    compare(&actual, &expected);
}

/// Recheck the retained actual execution against the current package and
/// independent Lean result without repeating the expensive witness producer.
#[cfg(test)]
#[test]
fn selected_nifs_saved_actual_result_matches_lean() {
    let saved = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/nifs/fixtures/stage1_actual_nifs");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_generate_pi_ccs_fixture"))
        .arg("check-owned-nifs")
        .arg(artifact("nightstream-fprime-stage1-poseidon2-hash-chain-v1.json"))
        .arg(saved)
        .arg(artifact("nightstream-fprime-stage1-base-nifs-result-v1.json"))
        .output()
        .expect("saved complete NIFS comparison executable");
    assert!(
        output.status.success(),
        "saved comparison failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("complete_nifs_wire=passed"));
    assert!(stdout.contains("saved_actual_pi_dec=passed complete_fields=17 normal_wrapper=true"));
    assert!(stdout.contains("actual_selected_nifs_Lean_comparison=passed"));
}
