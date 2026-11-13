//! The Rust-built memory circuit of each package has the rows of Lean's memory
//! application: the same rows, row for row, in the same variable order
//! (`EmitMain.applicationDigest`, `tests/fixtures/nebula-memory-*-digest.json`).
//! The witness programs differ in encoding (Lean: recipe batches and witness
//! instructions; Rust: one recipe per variable). The Rust-built circuit also
//! saves and loads.

use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    fs,
    ops::ControlFlow,
};

use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks as F;

use crate::nebula::test_packages::{bounded_depth_packages, references};
use crate::{
    application::ApplicationCircuit,
    nebula::{
        Application, Circuit, Expr, Machine, MachineState, Memory, Package, Program, Run, Step, Transition, View,
    },
};

/// FNV-1a over the little-endian bytes of `word`.
fn fnv(hash: u64, word: u64) -> u64 {
    word.to_le_bytes()
        .iter()
        .fold(hash, |hash, &byte| (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3))
}

/// The digest that Lean's emitter writes for the package's application: each
/// side's constant, then its terms by variable, with equal variables merged
/// and zero coefficients dropped.
fn assert_same_rows(package: Package, fixture: &[u8]) {
    assert_rows(&package.application().unwrap(), fixture);
}

fn assert_rows(application: &ApplicationCircuit, fixture: &[u8]) {
    let records = application.records();
    let mut hash = fnv(0xcbf29ce484222325, records.row_count() as u64);
    for row in 0..records.row_count() {
        let header = records.row_header(row).unwrap();
        let mut sides: [BTreeMap<usize, F>; 3] = Default::default();
        let _: ControlFlow<()> = records
            .visit_terms(row, |term| {
                *sides[term.form.index()]
                    .entry(term.variable)
                    .or_insert(F::ZERO) += F::from_u64(term.coefficient);
                Ok(ControlFlow::Continue(()))
            })
            .unwrap();
        for (side, constant) in sides.iter().zip(header.constants) {
            let terms: Vec<_> = side
                .iter()
                .filter(|(_, value)| **value != F::ZERO)
                .collect();
            hash = fnv(fnv(hash, constant), terms.len() as u64);
            for (variable, value) in terms {
                hash = fnv(fnv(hash, *variable as u64), value.as_canonical_u64());
            }
        }
    }
    let expected: serde_json::Value = serde_json::from_slice(fixture).unwrap();
    assert_eq!(Some(records.row_count() as u64), expected["rows"].as_u64(), "rows");
    assert_eq!(Some(hash), expected["digest"].as_u64(), "row digest");
}

/// A Stage 1 verifier checks no SPEC.md Algorithm 8 terminal condition, so it would
/// accept a state inside an open segment, whose reads no product test has
/// checked. A memory package makes only the memory verifier.
#[test]
fn memory_packages_make_no_stage1_verifier() {
    let packages = bounded_depth_packages("nebula-v1-");
    let loaded = Package::V1.load(references(&packages)).unwrap();
    assert!(matches!(
        loaded.circuit.verifier(crate::Engine::Optimized, 109),
        Err(crate::Error::Circuit(
            "a memory relation needs nebula::Circuit::verifier"
        ))
    ));
    assert!(loaded.verifier(crate::Engine::Optimized, 109).is_ok());
}

#[test]
fn first_package_rows_match_lean() {
    assert_same_rows(Package::V1, include_bytes!("../fixtures/nebula-memory-v1-digest.json"));
}

#[test]
fn second_package_rows_match_lean() {
    assert_same_rows(Package::N1, include_bytes!("../fixtures/nebula-memory-n1-digest.json"));
}

/// A RAM smaller than the ROM: the address field has `r` bits and row O6 has
/// its RAM range rows (`ThirdPlan.lean`).
#[test]
fn rows_of_a_ram_smaller_than_the_rom_match_lean() {
    let application = super::memory_circuit(&crate::nebula::Geometry::third(), &Machine).unwrap();
    assert_rows(&application, include_bytes!("../fixtures/nebula-memory-r1-digest.json"));
}

/// The slot coordinates of each kind (`Kind::width`) of `package`'s columns.
fn coordinates(package: &nightstream_fprime::StepPackage) -> BTreeMap<String, usize> {
    let layout = package.layout().unwrap();
    let mut counts = BTreeMap::new();
    for column in 270..package.sbox_start() + 150 * package.block_count() {
        if let Some((_, kind)) = layout.slot(column) {
            *counts.entry(format!("{kind:?}")).or_insert(0) += kind.width();
        }
    }
    counts
}

/// The witness words and locals of `application` that one of its rows forces
/// to 0 or 1: the row reads only that variable and holds at 0 and at 1
/// (Lean `SlotLayout.forcedColumn?`).
fn forced_words(application: &ApplicationCircuit) -> usize {
    let records = application.records();
    let witness = 4..4 + application.private_input_count();
    let mut forced = BTreeSet::new();
    for row in 0..records.row_count() {
        let header = records.row_header(row).unwrap();
        let mut variables = BTreeSet::new();
        let mut high = [F::ZERO; 3];
        let _: ControlFlow<()> = records
            .visit_terms(row, |term| {
                variables.insert(term.variable);
                high[term.form.index()] += F::from_u64(term.coefficient);
                Ok(ControlFlow::Continue(()))
            })
            .unwrap();
        let [variable] = variables.into_iter().collect::<Vec<_>>()[..] else {
            continue;
        };
        let low = header.constants.map(F::from_u64);
        if high[0] * high[1] != F::ZERO
            && low[0] * low[1] == low[2]
            && (low[0] + high[0]) * (low[1] + high[1]) == low[2] + high[2]
            && (witness.contains(&variable) || variable >= witness.end + 4)
        {
            forced.insert(variable);
        }
    }
    forced.len()
}

/// The machine's Rust-built memory circuit in place of the fixture's
/// application gives each Lean memory package's step relation: the same
/// blocks and plan rows, and the same slots except that each witness word or
/// local that a row forces to 0 or 1 is a bit slot (1 coordinate) where
/// Lean's is a field slot (41).
fn assert_native_packages_match_lean(package: Package, stem: &str) {
    let application = package.application().unwrap();
    let bits = forced_words(&application);
    assert!(bits > 0);
    let load = |stem: &str, identity| {
        let bytes = bounded_depth_packages(stem);
        nightstream_fprime::load_bounded_depth_packages(&bytes[0], &bytes[1], identity).unwrap()
    };
    let native = load("", nightstream_fprime::BOUNDED_DEPTH_PACKAGES_IDENTITY)
        .with_application(
            application.records().clone(),
            application.private_input_count(),
            application.shape(),
        )
        .unwrap();
    let lean = load(stem, package.bounded_depth_identity());
    for (native, lean) in [(native.light(), lean.light()), (native.refresh(), lean.refresh())] {
        assert_eq!(native.block_count(), lean.block_count());
        assert_eq!(
            native.plan_rows().unwrap().row_count(),
            lean.plan_rows().unwrap().row_count()
        );
        let mut expected = coordinates(lean);
        *expected.get_mut("Field").unwrap() -= 41 * bits;
        *expected.get_mut("Bit").unwrap() += bits;
        assert_eq!(coordinates(native), expected);
    }
}

#[test]
fn first_package_native_packages_match_lean() {
    assert_native_packages_match_lean(Package::V1, "nebula-v1-");
}

#[test]
fn second_package_native_packages_match_lean() {
    assert_native_packages_match_lean(Package::N1, "nebula-n1-");
}

/// The Rust circuit's witness program completes every honest invocation of a
/// native run of `program`, and its output is the next Stage 1 state.
fn assert_honest_invocations_execute(program: &Program, segments: &[&[Step]]) {
    let application = super::memory_circuit(&program.geometry, &Machine).unwrap();
    assert_application_executes(&application, program, segments);
}

fn assert_application_executes(application: &ApplicationCircuit, program: &Program, segments: &[&[Step]]) {
    let mut run = Run::new(Machine, program, MachineState::default());
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
    let circuit = Circuit::compile(Package::V1.geometry(), Machine).unwrap();
    let path = std::env::temp_dir().join(format!("nightstream-nebula-circuit-{}.package", std::process::id()));
    let _ = fs::remove_file(&path);
    circuit.write(&path).unwrap();
    let loaded = crate::Circuit::load(&path);
    fs::remove_file(&path).unwrap();
    let loaded = loaded.unwrap();
    assert_eq!(loaded.identity(), circuit.circuit.identity());
    assert!(matches!(
        loaded.verifier(crate::Engine::Optimized, 109),
        Err(crate::Error::Circuit(
            "a memory relation needs nebula::Circuit::verifier"
        ))
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
