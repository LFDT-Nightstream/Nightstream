#![cfg(unix)]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use recursive_constraint_minimizer::{
    render_query, run_cvc5, Conclusion, Problem, Selection, SolverConfig, SolverMode, SolverStatus,
};

fn fixture() -> Problem {
    serde_json::from_str(include_str!("../examples/known-local.json")).expect("valid fixture")
}

fn fake_solver(output: &str) -> PathBuf {
    static NEXT_ID: AtomicU64 = AtomicU64::new(0);
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after epoch")
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "recursive-constraint-minimizer-{}-{nonce}-{}.sh",
        std::process::id(),
        NEXT_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let script = format!("#!/bin/sh\ncat >/dev/null\nprintf '%s\\n' '{output}'\n");
    fs::write(&path, script).expect("write fake solver");
    let mut permissions = fs::metadata(&path)
        .expect("fake solver metadata")
        .permissions();
    permissions.set_mode(0o700);
    fs::set_permissions(&path, permissions).expect("make fake solver executable");
    path
}

fn config(executable: PathBuf) -> SolverConfig {
    SolverConfig {
        executable,
        mode: SolverMode::Gb,
        timeout_ms: 1_000,
    }
}

#[test]
fn unsat_is_only_a_redundancy_candidate() {
    let executable = fake_solver("unsat\n(constant_one keep_0 keep_1 candidate_violation)");
    let query = render_query(&fixture(), &Selection::Row("zero_copy".to_owned())).expect("query");
    let run = run_cvc5(&query, &config(executable.clone())).expect("solver run");
    fs::remove_file(executable).expect("remove fake solver");
    assert_eq!(run.status, SolverStatus::Unsat);
    assert_eq!(run.conclusion, Conclusion::RedundancyCandidate);
    assert!(run.stdout.contains("candidate_violation"));
}

#[test]
fn sat_is_a_counterexample_candidate() {
    let executable = fake_solver("sat\n(model (define-fun x_1 () F (as ff1 F)))");
    let query = render_query(&fixture(), &Selection::Family("zero".to_owned())).expect("query");
    let run = run_cvc5(&query, &config(executable.clone())).expect("solver run");
    fs::remove_file(executable).expect("remove fake solver");
    assert_eq!(run.status, SolverStatus::Sat);
    assert_eq!(run.conclusion, Conclusion::CounterexampleCandidate);
    assert!(run.stdout.contains("define-fun x_1"));
}

#[test]
fn unrecognized_solver_output_fails_closed() {
    let executable = fake_solver("solver made no decision");
    let query = render_query(&fixture(), &Selection::Family("zero".to_owned())).expect("query");
    let error = run_cvc5(&query, &config(executable.clone())).expect_err("must reject missing status");
    fs::remove_file(executable).expect("remove fake solver");
    assert!(error.to_string().contains("did not contain"));
}
