//! The Rust-built memory circuit of each package has the Lean package's
//! relation: the same dimensions and the same matrix rows, row for row. The
//! witness programs differ in encoding (Lean: recipe batches and witness
//! instructions; Rust: one recipe per variable), so the identifiers differ.
//! The Rust-built circuit also saves and loads.

use std::{cell::RefCell, collections::BTreeSet, fs, path::PathBuf};

use nightstream_fprime::{
    load_per_application_package, LoadedPerApplicationPackage, LogicalMatrixRow, StructuralIdentifier,
};

use crate::{
    application::ApplicationCircuit,
    assembly,
    nebula::{
        Application, Circuit, Context, Expr, Machine, MachineState, Memory, Package, Program, Run, Step, Transition,
        View,
    },
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

/// A Stage 1 verifier checks no SPEC.md Algorithm 10 terminal condition, so it would
/// accept a state inside an open segment, whose reads no product test has
/// checked. A memory package makes only the memory verifier.
#[test]
fn memory_packages_make_no_stage1_verifier() {
    let loaded = Package::V1
        .load(&artifact("nightstream-fprime-stage2-nebula-memory-v1.json"))
        .unwrap();
    assert!(matches!(
        crate::Verifier::from_package(&loaded.circuit, crate::Engine::Optimized, 114),
        Err(crate::Error::Circuit("a memory relation needs nebula::Verifier"))
    ));
    assert!(crate::nebula::Verifier::new(&loaded, crate::Engine::Optimized, 114).is_ok());
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
/// native run of `program`, and its output is the next Stage 1 state.
fn assert_honest_invocations_execute(program: &Program, segments: &[&[Step]]) {
    let application = super::memory_circuit(&program.geometry, &Machine).unwrap();
    assert_application_executes(&application, program, segments);
}

fn assert_application_executes(application: &ApplicationCircuit, program: &Program, segments: &[&[Step]]) {
    let mut run = Run::new(Context::new(Machine, program), MachineState::default());
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
    assert_honest_invocations_execute(&Package::V1.program(), &[&execute, &execute, &idle, &idle]);
}

/// One setup per geometry: the package circuit completes the honest
/// invocations of a second program, `loadi 7; store 1; load 1; halt` with RAM
/// `[0, 0, 0, 9]`.
#[test]
fn first_package_executes_a_second_program() {
    let program = Program::new(Package::V1.geometry(), vec![31, 6, 5, 0], vec![0, 0, 0, 9]).unwrap();
    assert_ne!(program, Package::V1.program());
    let (execute, idle) = ([Step::Execute; 2], [Step::Idle; 2]);
    assert_honest_invocations_execute(&program, &[&execute, &execute, &idle, &idle]);
}

/// `N = 1`: every invocation opens and closes a segment, and the O6 rows exist.
#[test]
fn second_package_executes_honest_invocations() {
    let segments = vec![&[Step::Execute][..]; Package::N1.geometry().s_max];
    assert_honest_invocations_execute(&Package::N1.program(), &segments);
}

/// `Circuit::write` keeps each output row's position, so the machine's
/// circuit, whose output rows are the `state_out` hash rows and not the last
/// rows, saves and loads. The loaded circuit has the same identity and
/// completes the honest invocations with the same output states.
#[test]
fn rust_built_circuit_writes_and_loads() {
    let reference = artifact("nightstream-fprime-stage1-poseidon2-hash-chain-v1.json");
    let circuit = Circuit::compile(&reference, Package::V1.geometry(), Machine).unwrap();
    let path = std::env::temp_dir().join(format!("nightstream-nebula-circuit-{}.package", std::process::id()));
    let _ = fs::remove_file(&path);
    circuit.write(&path).unwrap();
    let loaded = crate::Circuit::load(&path);
    fs::remove_file(&path).unwrap();
    let loaded = loaded.unwrap();
    assert_eq!(loaded.identity(), circuit.circuit.identity());
    assert!(matches!(
        crate::Verifier::from_package(&loaded, crate::Engine::Optimized, 114),
        Err(crate::Error::Circuit("a memory relation needs nebula::Verifier"))
    ));
    let application = loaded.application().unwrap();
    let size = crate::ApplicationSize {
        witness_words: application.private_input_count(),
        local_words: application.generated_range().len(),
        rows: application.row_count(),
    };
    assert_eq!(circuit.application_size(), Some(size));
    assert_eq!(loaded.application_size(), Some(size));
    let (execute, idle) = ([Step::Execute; 2], [Step::Idle; 2]);
    assert_application_executes(
        loaded.application().unwrap(),
        &Package::V1.program(),
        &[&execute, &execute, &idle, &idle],
    );
}

/// Lean indexes the view with natural numbers, so an index past the witness
/// reads zero. An application that reads its words at `usize::MAX` must get
/// zero too, not a witness word after the index sum wraps.
#[test]
fn view_reads_past_the_witness_are_zero() {
    struct Probe(RefCell<Vec<&'static str>>);

    impl Application for Probe {
        type Input = ();
        type State = ();
        type Error = std::convert::Infallible;

        fn width(&self) -> usize {
            1
        }
        fn aux(&self) -> usize {
            1
        }
        fn port_count(&self) -> usize {
            Package::V1.geometry().b_ops
        }
        fn words(&self, _: &()) -> Vec<u64> {
            vec![0]
        }
        fn rows(&self, view: &View) -> Result<Vec<Expr>, crate::Error> {
            let far = usize::MAX;
            let reads = [
                ("input", view.input(far)),
                ("output", view.output(far)),
                ("aux", view.aux(far)),
                ("pad", view.pad(far)),
                ("is_write", view.is_write(far)),
                ("is_ram", view.is_ram(far)),
                ("addr", view.addr(far)),
                ("vr", view.vr(far)),
                ("vw", view.vw(far)),
            ];
            for (name, read) in reads {
                if read.variable_outside(&BTreeSet::new()).is_some() {
                    self.0.borrow_mut().push(name);
                }
            }
            Ok(Vec::new())
        }
        fn step(&self, _: &(), _: &(), _: &Memory) -> Result<Transition<()>, Self::Error> {
            unreachable!("the probe runs no step")
        }
    }

    let probe = Probe(RefCell::new(Vec::new()));
    super::memory_circuit(&Package::V1.geometry(), &probe).unwrap();
    assert_eq!(*probe.0.borrow(), Vec::<&str>::new(), "reads that give a witness word");
}
