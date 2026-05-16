use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use neo_fold_prototype::rv64im::{
    build_mixed_opcode_perf_source_case, build_parity_case_from_source, RV64IM_MIXED_OPCODE_PERF_DEFAULT_N,
};
use nightstream_rust_vectors::render_rv64im_single_case_compat_module;

fn formal_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn lake_binary() -> PathBuf {
    if let Some(home) = std::env::var_os("HOME") {
        let elan_lake = PathBuf::from(home).join(".elan").join("bin").join("lake");
        if elan_lake.exists() {
            return elan_lake;
        }
    }
    PathBuf::from("lake")
}

fn perf_opcode_count_from_env() -> usize {
    match std::env::var("NS_DEBUG_N") {
        Ok(raw) => raw.parse().expect("NS_DEBUG_N must parse as usize"),
        Err(_) => RV64IM_MIXED_OPCODE_PERF_DEFAULT_N,
    }
}

fn build_lean_proof_boundary_targets() {
    let output = Command::new(lake_binary())
        .current_dir(formal_root())
        .arg("build")
        .arg("Nightstream.Rv64IM.ProofBoundaryChecks")
        .output()
        .expect("run lake build for RV64IM proof boundary checks");

    assert!(
        output.status.success(),
        "lake build failed for RV64IM proof boundary checks\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

#[test]
fn rv64im_mixed_opcode_perf_snapshot_matches_lean_checks() {
    let opcode_count = perf_opcode_count_from_env();
    let source = build_mixed_opcode_perf_source_case(opcode_count);
    let max_steps = source.program_words.len();
    let (_, derived) =
        build_parity_case_from_source(source.clone(), max_steps).expect("build mixed opcode perf parity case");
    let module_name = format!("Rv64imMixedOpcodePerfSnapshotN{opcode_count}");
    let lean_module = render_rv64im_single_case_compat_module(&module_name, &source, &derived);

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("current time after unix epoch")
        .as_nanos();
    let temp_path = std::env::temp_dir().join(format!(
        "rv64im_mixed_opcode_perf_snapshot_lean_compat_{}_{}.lean",
        std::process::id(),
        unique
    ));
    fs::write(&temp_path, lean_module).expect("write temporary Lean compatibility module");

    build_lean_proof_boundary_targets();

    let output = Command::new(lake_binary())
        .current_dir(formal_root())
        .arg("env")
        .arg("lean")
        .arg(&temp_path)
        .output()
        .expect("run lake env lean for mixed opcode perf snapshot compatibility");

    let _ = fs::remove_file(&temp_path);

    assert!(
        output.status.success(),
        "Lean compatibility check failed for mixed opcode perf snapshot\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}
