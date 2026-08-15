#![cfg(unix)]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use neo_fold_clean::engine::r1cs_circuit::builder::RowFamilyRange;
use neo_fold_clean::engine::r1cs_circuit::{Lc, R1csBuilder, R1csSnapshot};
use neo_fold_clean::frontends::r1cs_f_prime::{lower_field_r1cs, SparseR1cs};
use neo_math::F;
use nightstream_constraint_exporter::{refine_sparse_with_cvc5, refine_with_cvc5, ExportRequest};
use p3_field::PrimeCharacteristicRing;
use recursive_constraint_minimizer::{Conclusion, Scope, Selection, SolverConfig, SolverMode, GOLDILOCKS_MODULUS};

fn duplicate_source() -> (R1csSnapshot, Vec<RowFamilyRange>) {
    let mut builder = R1csBuilder::new();
    let x = builder.alloc(F::ZERO);
    let one = Lc::from_const(F::ONE);
    let zero = Lc::zero();
    builder.enforce(&Lc::from_var(x), &one, &zero);
    builder.record_row_family("candidate", 0);
    builder.enforce(&Lc::from_var(x), &one, &zero);
    builder.record_row_family("copy", 1);
    (builder.snapshot(), builder.row_family_ranges().to_vec())
}

fn necessary_source() -> (R1csSnapshot, Vec<RowFamilyRange>) {
    let mut builder = R1csBuilder::new();
    let x = builder.alloc(F::ZERO);
    let one = Lc::from_const(F::ONE);
    let zero = Lc::zero();
    builder.enforce(&Lc::from_var(x), &one, &zero);
    builder.record_row_family("candidate", 0);
    let mut x_minus_one = Lc::from_var(x);
    x_minus_one.add_constant(-F::ONE);
    builder.enforce(&Lc::from_var(x), &x_minus_one, &zero);
    builder.record_row_family("bitness", 1);
    (builder.snapshot(), builder.row_family_ranges().to_vec())
}

fn duplicate_sparse_source() -> (SparseR1cs, Vec<F>) {
    let mut builder = R1csBuilder::new();
    builder.begin_encoding_stage("candidate");
    let x = builder.alloc(F::ZERO);
    let one = Lc::from_const(F::ONE);
    let zero = Lc::zero();
    builder.enforce(&Lc::from_var(x), &one, &zero);
    builder.begin_encoding_stage("copy");
    builder.enforce(&Lc::from_var(x), &one, &zero);
    lower_field_r1cs(builder, &[])
        .expect("lower staged sparse fixture")
        .into_parts()
}

fn necessary_sparse_source() -> (SparseR1cs, Vec<F>) {
    let mut builder = R1csBuilder::new();
    builder.begin_encoding_stage("candidate");
    let x = builder.alloc(F::ZERO);
    let one = Lc::from_const(F::ONE);
    let zero = Lc::zero();
    builder.enforce(&Lc::from_var(x), &one, &zero);
    builder.begin_encoding_stage("bitness");
    let mut x_minus_one = Lc::from_var(x);
    x_minus_one.add_constant(-F::ONE);
    builder.enforce(&Lc::from_var(x), &x_minus_one, &zero);
    lower_field_r1cs(builder, &[])
        .expect("lower staged sparse fixture")
        .into_parts()
}

fn request() -> ExportRequest {
    ExportRequest {
        profile: "refinement-test".to_owned(),
        scope: Scope::Local,
        public_input_count: 1,
        source_rows: vec![0],
        complete_families: Vec::new(),
    }
}

fn unique_path(label: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("nightstream-refinement-{label}-{}-{nonce}", std::process::id()))
}

fn make_solver(script: &str) -> PathBuf {
    let path = unique_path("solver.sh");
    fs::write(&path, script).expect("write fake solver");
    let mut permissions = fs::metadata(&path).expect("solver metadata").permissions();
    permissions.set_mode(0o700);
    fs::set_permissions(&path, permissions).expect("make solver executable");
    path
}

fn config(executable: PathBuf) -> SolverConfig {
    SolverConfig {
        executable,
        mode: SolverMode::Gb,
        timeout_ms: 1_000,
    }
}

fn sat_model() -> String {
    format!(
        "sat\n(\n(define-fun x_0 () (_ FiniteField {0}) #f1m{0})\n\
         (define-fun x_1 () (_ FiniteField {0}) #f1m{0})\n)\n",
        GOLDILOCKS_MODULUS
    )
}

fn remove_file(path: &Path) {
    fs::remove_file(path).expect("remove temporary file");
}

#[test]
fn adds_a_violated_source_row_before_accepting_unsat() {
    let state = unique_path("state");
    let script = format!(
        "#!/bin/sh\ncat >/dev/null\nif [ -e '{state}' ]; then\n  printf '%s\\n' 'unsat'\nelse\n  : > '{state}'\n  printf '%s' '{sat}'\nfi\n",
        state = state.display(),
        sat = sat_model()
    );
    let executable = make_solver(&script);
    let solver = config(executable.clone());
    let (snapshot, ranges) = duplicate_source();
    let report = refine_with_cvc5(
        &snapshot,
        &ranges,
        request(),
        &Selection::Row("r1cs.row.0".to_owned()),
        &solver,
        2,
    )
    .expect("bounded refinement");

    remove_file(&executable);
    remove_file(&state);
    assert_eq!(report.conclusion, Conclusion::RedundancyCandidate);
    assert_eq!(report.iterations, 2);
    assert_eq!(
        report
            .problem
            .rows
            .iter()
            .map(|row| row.source_index)
            .collect::<Vec<_>>(),
        [0, 1]
    );
}

#[test]
fn accepts_sat_only_after_full_retained_relation_replay() {
    let executable = make_solver(&format!("#!/bin/sh\ncat >/dev/null\nprintf '%s' '{}'\n", sat_model()));
    let solver = config(executable.clone());
    let (snapshot, ranges) = necessary_source();
    let report = refine_with_cvc5(
        &snapshot,
        &ranges,
        request(),
        &Selection::Row("r1cs.row.0".to_owned()),
        &solver,
        2,
    )
    .expect("full replay");

    remove_file(&executable);
    assert_eq!(report.conclusion, Conclusion::CounterexampleCandidate);
    assert_eq!(report.iterations, 1);
    assert_eq!(report.violated_candidate_rows, [0]);
    assert!(report.pending_retained_row.is_none());
}

#[test]
fn iteration_cap_is_inconclusive_and_keeps_the_candidate() {
    let executable = make_solver(&format!("#!/bin/sh\ncat >/dev/null\nprintf '%s' '{}'\n", sat_model()));
    let solver = config(executable.clone());
    let (snapshot, ranges) = duplicate_source();
    let report = refine_with_cvc5(
        &snapshot,
        &ranges,
        request(),
        &Selection::Row("r1cs.row.0".to_owned()),
        &solver,
        1,
    )
    .expect("bounded result");

    remove_file(&executable);
    assert_eq!(report.conclusion, Conclusion::Inconclusive);
    assert_eq!(report.pending_retained_row, Some(1));
}

#[test]
fn sparse_replay_adds_a_violated_retained_row_before_unsat() {
    let state = unique_path("sparse-state");
    let script = format!(
        "#!/bin/sh\ncat >/dev/null\nif [ -e '{state}' ]; then\n  printf '%s\\n' 'unsat'\nelse\n  : > '{state}'\n  printf '%s' '{sat}'\nfi\n",
        state = state.display(),
        sat = sat_model()
    );
    let executable = make_solver(&script);
    let solver = config(executable.clone());
    let (arm, assignment) = duplicate_sparse_source();
    let report = refine_sparse_with_cvc5(
        &arm,
        &assignment,
        request(),
        &Selection::Row("r1cs.row.0".to_owned()),
        &solver,
        2,
    )
    .expect("bounded sparse refinement");

    remove_file(&executable);
    remove_file(&state);
    assert_eq!(report.conclusion, Conclusion::RedundancyCandidate);
    assert_eq!(report.iterations, 2);
    assert_eq!(
        report
            .problem
            .rows
            .iter()
            .map(|row| row.source_index)
            .collect::<Vec<_>>(),
        [0, 1]
    );
}

#[test]
fn sparse_sat_requires_every_retained_rust_row_to_hold() {
    let executable = make_solver(&format!("#!/bin/sh\ncat >/dev/null\nprintf '%s' '{}'\n", sat_model()));
    let solver = config(executable.clone());
    let (arm, assignment) = necessary_sparse_source();
    let report = refine_sparse_with_cvc5(
        &arm,
        &assignment,
        request(),
        &Selection::Row("r1cs.row.0".to_owned()),
        &solver,
        2,
    )
    .expect("complete sparse Rust replay");

    remove_file(&executable);
    assert_eq!(report.conclusion, Conclusion::CounterexampleCandidate);
    assert_eq!(report.iterations, 1);
    assert_eq!(report.violated_candidate_rows, [0]);
    assert!(report.pending_retained_row.is_none());
}
