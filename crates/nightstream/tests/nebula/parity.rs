//! The Rust-built memory circuit of each package has the Lean package's
//! relation: the same dimensions and the same matrix rows, row for row. The
//! witness programs differ in encoding (Lean: recipe batches and witness
//! instructions; Rust: one recipe per variable), so the identifiers differ.

use std::{fs, ops::ControlFlow, path::PathBuf};

use nightstream_fprime::{
    load_per_application_package, ApplicationRecords, ApplicationRowHeader, ApplicationTerm,
    LoadedPerApplicationPackage, LogicalMatrixRow, StructuralIdentifier,
};

use super::memory_circuit;
use crate::{
    assembly,
    nebula::{Context, MachineState, Package, Program, Run, Step},
};

fn artifact(name: &str) -> Vec<u8> {
    fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("artifacts")
            .join(name),
    )
    .unwrap()
}

fn assert_same_relation(package: Package, lean: &str) {
    let reference = artifact("nightstream-fprime-stage1-poseidon2-hash-chain-v1.json");
    let application = package.application().unwrap();
    let (actual, _) = assembly::prepare(&reference, &application).unwrap();
    let expected = load_per_application_package(
        &artifact(lean),
        StructuralIdentifier::from_words(package.structural_identifier()),
    )
    .unwrap();

    let rows = expected.application().row_range();
    assert_eq!(actual.application().row_range(), rows, "application rows");
    assert_eq!(
        actual.application().private_range(),
        expected.application().private_range(),
        "application variables"
    );
    assert_eq!(actual.row_count(), expected.row_count(), "package rows");
    assert_eq!(
        actual.logical_column_count(),
        expected.logical_column_count(),
        "columns"
    );

    // Every logical row: the shared verifier rows relocated around the
    // application, the application rows, and the boundary rows. The test
    // compares one row at a time, because holding every expanded row at once
    // took about 34 GB.
    for index in 0..expected.row_count() {
        assert_eq!(
            logical_row(&actual, index),
            logical_row(&expected, index),
            "logical row {index}"
        );
    }
}

fn logical_row(package: &LoadedPerApplicationPackage, index: usize) -> LogicalMatrixRow {
    let mut found = None;
    package
        .visit_matrix_rows(index..index + 1, |_, row| {
            found = Some(row);
            Ok(())
        })
        .unwrap();
    found.unwrap()
}

#[test]
fn first_package_relation_matches_lean() {
    assert_same_relation(Package::V1, "nightstream-fprime-stage2-nebula-memory-v1.json");
}

#[test]
fn second_package_relation_matches_lean() {
    assert_same_relation(Package::N1, "nightstream-fprime-stage2-nebula-memory-n1.json");
}

/// One setup per geometry: a program with other images has the same
/// application rows and the same package identity as the package's default
/// program.
#[test]
fn programs_of_one_geometry_share_the_circuit() {
    let other = Program::new(Package::V1, vec![31, 6, 5, 0], vec![0, 0, 0, 9]).unwrap();
    assert_ne!(other, Package::V1.program());
    let [default, other] = [Package::V1.program(), other].map(|program| memory_circuit(&program.plan).unwrap());
    let (rows, other_rows) = (default.records(), other.records());
    assert_eq!(rows.row_count(), other_rows.row_count());
    for row in 0..rows.row_count() {
        assert_eq!(row_terms(rows, row), row_terms(other_rows, row), "row {row}");
    }
    let reference = artifact("nightstream-fprime-stage1-poseidon2-hash-chain-v1.json");
    let identifier = |application| {
        assembly::prepare(&reference, application)
            .unwrap()
            .0
            .structural_identifier()
    };
    assert_eq!(identifier(&default), identifier(&other));
}

fn row_terms(records: &ApplicationRecords, row: usize) -> (ApplicationRowHeader, Vec<ApplicationTerm>) {
    let mut terms = Vec::new();
    assert!(records
        .visit_terms(row, |term| {
            terms.push(term);
            Ok(ControlFlow::Continue(()))
        })
        .unwrap()
        .is_continue());
    (records.row_header(row).unwrap(), terms)
}

/// The Rust circuit's witness program completes every honest invocation of a
/// native run, and its output is the next Stage 1 state.
fn assert_honest_invocations_execute(package: Package, segments: &[&[Step]]) {
    let application = package.application().unwrap();
    let mut run = Run::new(Context::new(&package.program()), MachineState::default());
    let mut state = run.initial_state();
    for steps in segments {
        for invocation in run.segment(steps).unwrap() {
            let witness = application.execute(state, &invocation.words).unwrap();
            assert_eq!(witness.output_state(), invocation.output);
            state = invocation.output;
        }
    }
}

#[test]
fn first_package_executes_honest_invocations() {
    let (execute, idle) = ([Step::Execute; 2], [Step::Idle; 2]);
    assert_honest_invocations_execute(Package::V1, &[&execute, &execute, &idle, &idle]);
}

/// `N = 1`: every invocation opens and closes a segment, and the O6 rows exist.
#[test]
fn second_package_executes_honest_invocations() {
    let segments = vec![&[Step::Execute][..]; Package::N1.plan().s_max];
    assert_honest_invocations_execute(Package::N1, &segments);
}
