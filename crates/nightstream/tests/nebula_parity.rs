//! The Rust-built memory circuit of each package has the Lean package's
//! relation: the same dimensions and the same matrix rows, row for row. The
//! witness programs differ in encoding (Lean: recipe batches and witness
//! instructions; Rust: one recipe per variable), so the identifiers differ.

use std::{fs, path::PathBuf};

use nightstream::{assembly, nebula::Package};

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
    let value = serde_json::from_slice(&artifact(lean)).unwrap();
    let expected = nightstream_fprime::load_prepared_application_value(value).unwrap();

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
    // application, the application rows, and the boundary rows.
    let compared = 0..expected.row_count();
    let mut lean_rows = Vec::with_capacity(compared.len());
    expected
        .visit_matrix_rows(compared.clone(), |index, row| {
            lean_rows.push((index, row));
            Ok(())
        })
        .unwrap();
    let mut position = 0;
    actual
        .visit_matrix_rows(compared, |index, row| {
            let (lean_index, lean_row) = &lean_rows[position];
            assert_eq!((index, &row), (*lean_index, lean_row), "logical row {index}");
            position += 1;
            Ok(())
        })
        .unwrap();
    assert_eq!(position, lean_rows.len());
}

#[test]
fn first_package_relation_matches_lean() {
    assert_same_relation(Package::V1, "nightstream-fprime-stage2-nebula-memory-v1.json");
}

#[test]
fn second_package_relation_matches_lean() {
    assert_same_relation(Package::N1, "nightstream-fprime-stage2-nebula-memory-n1.json");
}

/// The Rust circuit's witness program completes every honest invocation of a
/// native run, and its output is the next Stage 1 state.
#[test]
fn rust_circuit_executes_honest_invocations() {
    use nightstream::nebula::{Context, MachineState, Run, Step};
    let application = Package::V1.application().unwrap();
    let mut run = Run::new(Context::new(Package::V1), MachineState::default());
    let mut state = run.initial_state();
    for steps in [[Step::Execute; 2], [Step::Execute; 2], [Step::Idle; 2], [Step::Idle; 2]] {
        for invocation in run.segment(&steps).unwrap() {
            let witness = application.execute(state, &invocation.words).unwrap();
            assert_eq!(witness.output_state(), invocation.output);
            state = invocation.output;
        }
    }
}
