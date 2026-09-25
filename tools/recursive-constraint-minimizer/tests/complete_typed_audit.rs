#![cfg(unix)]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use recursive_constraint_minimizer::{
    audit_complete_typed_candidate, audit_complete_typed_family, Conclusion, FieldModel, Problem, Selection,
    SolverConfig, SolverMode, SolverStatus, TypedTarget, TypedTargetRow,
};

fn fixture() -> Problem {
    serde_json::from_str(include_str!("../examples/known-local.json")).expect("valid fixture")
}

fn zero_target(problem: &Problem) -> TypedTarget {
    TypedTarget {
        id: "typed.zero".to_owned(),
        column_count: problem.column_count,
        rows: vec![TypedTargetRow {
            id: "typed.zero.row".to_owned(),
            a: problem.rows[1].a.clone(),
            b: problem.rows[1].b.clone(),
            c: problem.rows[1].c.clone(),
        }],
    }
}

fn fake_solver(output: &str) -> PathBuf {
    static NEXT_ID: AtomicU64 = AtomicU64::new(0);
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after epoch")
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "recursive-constraint-minimizer-strict-{}-{nonce}-{}.sh",
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

fn config(executable: impl Into<PathBuf>) -> SolverConfig {
    SolverConfig {
        executable: executable.into(),
        mode: SolverMode::Gb,
        timeout_ms: 5_000,
    }
}

#[test]
fn sat_requires_full_retained_and_typed_target_replay() {
    let problem = fixture();
    let executable = fake_solver(
        "sat\n(model (define-fun x_0 () F (as ff1 F)) \
         (define-fun x_1 () F (as ff1 F)))",
    );
    let report = audit_complete_typed_family(
        &problem,
        &Selection::Family("zero".to_owned()),
        &zero_target(&problem),
        &config(executable.clone()),
    )
    .expect("strict audit");
    fs::remove_file(executable).expect("remove fake solver");

    assert_eq!(report.conclusion, Conclusion::CounterexampleCandidate);
    assert_eq!(report.retained_rows_replayed, [0]);
    assert_eq!(report.violated_target_rows, [0]);
    assert_eq!(report.model.expect("SAT model").values(), [1, 1]);
}

#[test]
fn sat_parse_failure_is_inconclusive_for_removal() {
    let problem = fixture();
    let executable = fake_solver("sat\n(model (define-fun x_0 () F (as ff1 F)))");
    let error = audit_complete_typed_family(
        &problem,
        &Selection::Family("zero".to_owned()),
        &zero_target(&problem),
        &config(executable.clone()),
    )
    .expect_err("missing full assignment must fail closed");
    fs::remove_file(executable).expect("remove fake solver");
    assert!(error.to_string().contains("does not define x_1"));
}

#[test]
fn unsat_remains_only_a_redundancy_candidate() {
    let problem = fixture();
    let executable = fake_solver("unsat\n(constant_one keep_0 typed_target_violation)");
    let report = audit_complete_typed_family(
        &problem,
        &Selection::Family("zero".to_owned()),
        &zero_target(&problem),
        &config(executable.clone()),
    )
    .expect("strict audit");
    fs::remove_file(executable).expect("remove fake solver");
    assert_eq!(report.solver_run.status, SolverStatus::Unsat);
    assert_eq!(report.conclusion, Conclusion::RedundancyCandidate);
    assert!(report.model.is_none());
}

#[test]
fn pinned_candidate_requires_the_same_complete_solver_model() {
    let problem = fixture();
    let candidate = FieldModel::from_canonical_values(vec![1, 1]).expect("candidate");
    let executable = fake_solver(
        "sat\n(model (define-fun x_0 () F (as ff1 F)) \
         (define-fun x_1 () F (as ff0 F)))",
    );
    let error = audit_complete_typed_candidate(
        &problem,
        &Selection::Family("zero".to_owned()),
        &zero_target(&problem),
        &candidate,
        &config(executable.clone()),
    )
    .expect_err("solver model must equal the pinned candidate");
    fs::remove_file(executable).expect("remove fake solver");
    assert!(error
        .to_string()
        .contains("differs from the complete pinned candidate"));
}

#[test]
#[ignore = "requires the official cvc5 build with CoCoA finite-field support"]
fn installed_cvc5_finds_and_replays_the_strict_goldilocks_counterexample() {
    let executable = Path::new("/Users/nijaar/.local/bin/cvc5");
    assert!(executable.is_file(), "installed cvc5 is missing");
    let problem = fixture();
    let report = audit_complete_typed_family(
        &problem,
        &Selection::Family("zero".to_owned()),
        &zero_target(&problem),
        &config(executable),
    )
    .expect("strict cvc5 audit");

    assert_eq!(report.solver_run.status, SolverStatus::Sat);
    assert_eq!(report.conclusion, Conclusion::CounterexampleCandidate);
    assert_eq!(report.retained_rows_replayed, [0]);
    assert_eq!(report.violated_target_rows, [0]);
    assert_eq!(report.model.expect("SAT model").values(), [1, 1]);
}
