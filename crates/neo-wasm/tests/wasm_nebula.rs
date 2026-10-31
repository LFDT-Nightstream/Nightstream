//! Enzo's WASM relation proved with Nebula memory checking (SPEC.md section
//! 7.6). One Nebula step proves `per_step` consecutive trace rows: its
//! private words are the witnesses `z` of its rows, its state is the
//! continuity words (one per link, `_AFTER` of a row equal to `_BEFORE` of
//! the next), and `nightstream::nebula::CatalogBinding` checks every memory
//! of the batched catalog. The regions come from what the test programs
//! need, so one circuit serves each program of a setup. The slot plan is
//! `plan_ordered_slots`, which keeps each pair of one memory with a write in
//! trace order on every row that the relation accepts.

mod common;

use std::cell::RefCell;
use std::num::NonZeroUsize;
use std::rc::Rc;

use neo_math::F;
use neo_wasm::batch::build_batched_wasm_ccs;
use neo_wasm::host_event_bindings::HostEventBindings;
use neo_wasm::memory_semantics::preload_host_event_tables;
use neo_wasm::nebula::{
    is_active, plan_ordered_slots, smallest_geometry, smallest_regions, state_words, step_inputs, WasmApplication,
    WasmApplicationError, WasmStepError,
};
use neo_wasm::{
    build_wasm_relation_layout, plan_wasm_memory_slots, preload_from_program_artifacts, top_level_initial_state,
    WasmMemoryId, WasmMemoryPreload,
};
use nightstream::nebula::{
    size_parts, Application, CatalogRowError, Context, Expr, Geometry, Memory, MemoryPortActivation, MemoryPortKind,
    Program, Region, Run, SegmentError, Transition, View,
};
use p3_field::PrimeCharacteristicRing;

/// Stack and linear-memory traffic: store 42 at address 0, then load it.
const LINEAR_MEMORY: &str = r#"(module
     (memory 1)
     (func (export "main") (result i32)
       i32.const 0
       i32.const 42
       i32.store
       i32.const 0
       i32.load))"#;

/// The store of `LINEAR_MEMORY` with the load right after it: the address is
/// pushed twice, and `nop` moves the store to an even row.
const STORE_LOAD: &str = r#"(module
     (memory 1)
     (func (export "main") (result i32)
       nop
       i32.const 0
       i32.const 0
       i32.const 42
       i32.store
       i32.load))"#;

/// Rows with no memory traffic but the stack.
const NOPS: &str = r#"(module
     (func (export "main") (result i32)
       nop nop nop nop nop
       i32.const 7))"#;

/// A guest call with one argument.
const CALL: &str = r#"(module
    (func $add_one (param i32) (result i32)
        local.get 0
        i32.const 1
        i32.add)
    (func (export "main") (result i32)
        i32.const 5
        call $add_one))"#;

/// Enzo's iterative Fibonacci program (`tests/wasm_fibonacci.rs`): `n` loop
/// rounds, then `fib(n)`.
fn fib_wat(n: u32) -> String {
    format!(
        r#"
(module
  (func (export "main") (result i32)
    (local i32 i32 i32)
    i32.const {n}
    local.set 0
    i32.const 1
    local.set 1
    i32.const 0
    local.set 2
    (loop $L
      local.get 1
      local.get 2
      i32.add
      local.get 2
      local.set 1
      local.set 2
      local.get 0
      i32.const 1
      i32.sub
      local.tee 0
      br_if $L
    )
    local.get 2
  )
)
"#
    )
}

/// Trace rows per segment: the call trace has 8 rows, the linear-memory
/// trace 6 rows and 2 padding rows.
const N: usize = 8;

fn per_step(rows: usize) -> NonZeroUsize {
    NonZeroUsize::new(rows).expect("at least one row per step")
}

/// A test program as its verifier and prover see it.
struct Traced {
    /// The one-row witness of each trace row, then of each padding row, one
    /// segment in all.
    rows: Vec<Vec<F>>,
    /// The step inputs of the segment: `per_step` rows each.
    inputs: Vec<Vec<F>>,
    real: usize,
    preload: WasmMemoryPreload,
    /// The continuity words of the entry state (`top_level_initial_state`).
    initial: Vec<u64>,
    /// The continuity words after the last trace row.
    last: Vec<u64>,
}

fn traced(wat: &str, steps: usize, per_step: NonZeroUsize) -> Traced {
    let checked = common::checked_main(wat);
    let tables = &checked.artifacts.tables;
    // `main` is the last function of each test module.
    let (export_fref, entry_pc) = tables
        .function_entries
        .iter()
        .copied()
        .max_by_key(|(_, pc)| *pc)
        .expect("a function");
    let mut preload = preload_from_program_artifacts(&checked.artifacts);
    preload_host_event_tables(
        &mut preload,
        &HostEventBindings::import_free(u32::try_from(export_fref).unwrap()),
    );
    Traced {
        rows: step_inputs(&checked.trace, steps * per_step.get(), NonZeroUsize::MIN).expect("the trace fits"),
        inputs: step_inputs(&checked.trace, steps, per_step).expect("the trace fits one segment"),
        real: checked.trace.len(),
        preload,
        initial: state_words(top_level_initial_state(tables, entry_pc)),
        last: state_words(checked.trace.last().expect("a trace row").state_after),
    }
}

struct Setup {
    programs: Vec<Traced>,
    activations: Vec<MemoryPortActivation>,
    regions: Vec<(WasmMemoryId, Region)>,
    log2: [usize; 2],
    /// Steps per segment.
    n: usize,
    per_step: NonZeroUsize,
}

/// The linear-memory and call programs, `N` trace rows each, in steps of
/// `per_step` rows.
fn setup(per_step: NonZeroUsize) -> Setup {
    let n = N / per_step.get();
    Setup::of(
        vec![traced(LINEAR_MEMORY, n, per_step), traced(CALL, n, per_step)],
        n,
        per_step,
    )
}

/// One program in the fewest steps of `per_step` rows: a power of two, with
/// at least one padding row when `pad` is set.
fn setup_of(wat: &str, per_step: NonZeroUsize, pad: bool) -> Setup {
    let rows = common::checked_main(wat).trace.len() + usize::from(pad);
    let n = rows.div_ceil(per_step.get()).next_power_of_two();
    Setup::of(vec![traced(wat, n, per_step)], n, per_step)
}

impl Setup {
    /// One circuit for `programs`, each traced to one segment of `n` steps of
    /// `per_step` rows.
    fn of(programs: Vec<Traced>, n: usize, per_step: NonZeroUsize) -> Setup {
        let catalog = &build_wasm_relation_layout().auxiliary.memory;
        let (regions, log2) = smallest_regions(
            &programs
                .iter()
                .map(|traced| &traced.preload)
                .collect::<Vec<_>>(),
            &programs
                .iter()
                .flat_map(|traced| traced.rows.clone())
                .collect::<Vec<_>>(),
        );
        Setup {
            activations: catalog
                .entries()
                .iter()
                .flat_map(|memory| memory.ports.iter().map(|port| port.activation))
                .collect(),
            regions,
            log2,
            programs,
            n,
            per_step,
        }
    }

    /// The application and geometry of a slot plan.
    fn app(&self, slots: &[Vec<usize>]) -> (WasmApplication, Geometry) {
        let geometry = smallest_geometry(self.log2, slots.len(), self.n, 1);
        let app =
            WasmApplication::with_slots(&geometry, self.regions.clone(), self.per_step, slots).expect("an application");
        (app, geometry)
    }

    /// The application of the static plan that keeps each dependent pair in
    /// trace order.
    fn complete(&self) -> (WasmApplication, Geometry) {
        self.app(&plan_ordered_slots(self.per_step))
    }

    fn run(&self, app: &WasmApplication, traced: &Traced) -> (Program, Run<WasmApplication>) {
        let program = app
            .program(&traced.preload)
            .expect("the regions hold the preload");
        let run = Run::new(Context::new(app.clone(), &program), traced.initial.clone());
        (program, run)
    }

    /// The step input and the column in it of column `column` of trace row
    /// `row`: each step input is `per_step` one-row witnesses in trace order.
    fn position(&self, traced: &Traced, row: usize, column: usize) -> (usize, usize) {
        let per_step = self.per_step.get();
        (row / per_step, row % per_step * traced.rows[0].len() + column)
    }
}

/// `WasmApplication` with the rows of its last `rows` call kept, so that a
/// test can evaluate them on an invocation's words.
#[derive(Clone)]
struct Kept {
    app: WasmApplication,
    rows: Rc<RefCell<Vec<Expr>>>,
}

impl Application for Kept {
    type Input = Vec<F>;
    type State = Vec<u64>;
    type Error = WasmStepError;

    fn width(&self) -> usize {
        self.app.width()
    }

    fn aux(&self) -> usize {
        self.app.aux()
    }

    fn port_count(&self) -> usize {
        self.app.port_count()
    }

    fn words(&self, state: &Vec<u64>) -> Vec<u64> {
        Application::words(&self.app, state)
    }

    fn rows(&self, view: &View) -> Result<Vec<Expr>, nightstream::Error> {
        let rows = self.app.rows(view)?;
        self.rows.replace(rows.clone());
        Ok(rows)
    }

    fn step(&self, state: &Vec<u64>, z: &Vec<F>, memory: &Memory) -> Result<Transition<Vec<u64>>, WasmStepError> {
        self.app.step(state, z, memory)
    }
}

/// The application rows of `app` over the view of `geometry`.
fn application_rows(app: &WasmApplication, geometry: &Geometry) -> Vec<Expr> {
    let kept = Kept {
        app: app.clone(),
        rows: Rc::default(),
    };
    size_parts(geometry, &kept).expect("the memory rows of the geometry");
    kept.rows.take()
}

/// Lean `Circuit.Expr.eval` on the private words of an invocation, which
/// follow the four words of the input state digest.
fn eval(expr: &Expr, words: &[F]) -> F {
    match expr {
        Expr::Var(variable) => words[variable.index() - 4],
        Expr::Const(value) => *value,
        Expr::Add(left, right) => eval(left, words) + eval(right, words),
        Expr::Mul(left, right) => eval(left, words) * eval(right, words),
    }
}

/// The positions of the application rows that do not hold on `words`.
fn failing(rows: &[Expr], words: &[F]) -> Vec<usize> {
    rows.iter()
        .enumerate()
        .filter(|(_, row)| eval(row, words) != F::ZERO)
        .map(|(position, _)| position)
        .collect()
}

/// Both honest padded traces, in steps of 1, 2, and 4 rows, run through the
/// native Nebula runner, which applies each step's ports and rejects a port
/// that reads another value than its cell holds, and every application row
/// holds on each invocation. Every port of every row is active exactly when
/// its catalog port is: the final timestamp counts one access per active
/// catalog port. Padding rows activate no port and keep the state, also when
/// they share the last step with trace rows.
#[test]
fn honest_padded_traces_run_through_nebula() {
    for rows in [1, 2, 4] {
        let setup = setup(per_step(rows));
        let (app, geometry) = setup.complete();
        assert!(app.binding().host_checked().is_empty(), "every memory is checked");
        let relation = application_rows(&app, &geometry);
        eprintln!("{rows} rows per step: geometry {geometry:?}");
        for traced in &setup.programs {
            let (_, mut run) = setup.run(&app, traced);
            let invocations = run.segment(&traced.inputs).expect("the honest trace runs");
            for (step, invocation) in invocations.iter().enumerate() {
                assert_eq!(
                    failing(&relation, &invocation.words),
                    Vec::<usize>::new(),
                    "step {step} holds"
                );
            }
            let statement = run.statement();
            assert_eq!(
                statement.final_, traced.last,
                "padding keeps the state of the last trace row"
            );
            for z in &traced.rows[traced.real..] {
                assert!(
                    setup
                        .activations
                        .iter()
                        .all(|activation| !is_active(z, *activation)),
                    "a padding row activates no port"
                );
            }
            let events = traced
                .rows
                .iter()
                .map(|z| {
                    setup
                        .activations
                        .iter()
                        .filter(|activation| is_active(z, **activation))
                        .count() as u64
                })
                .sum::<u64>();
            assert_eq!(statement.final_ts, events, "one access per active catalog port");
            eprintln!(
                "rows={} events={events} witness words per step={}",
                traced.real,
                invocations[0].words.len()
            );
        }
        if rows == 4 {
            assert_ne!(
                setup.programs[0].real % 4,
                0,
                "the last linear-memory step has padding rows"
            );
        }
    }
}

/// The `i32.load` of `STORE_LOAD` reads the `i32.store` of the row before it,
/// in the same step of 2 or 4 rows. The honest run holds. The load with a
/// value that its cell does not hold fails a row on the honest memory words,
/// and the runner rejects its port.
#[test]
fn a_load_reads_a_store_of_an_earlier_row_of_its_step() {
    for rows in [2, 4] {
        let setup = setup_of(STORE_LOAD, per_step(rows), false);
        let (app, geometry) = setup.complete();
        let relation = application_rows(&app, &geometry);
        let traced = &setup.programs[0];
        let (store, _) = access(traced, true);
        let (load, column) = access(traced, false);
        assert!(store < load, "the store comes first");
        let (step, position) = setup.position(traced, load, column);
        assert_eq!(
            setup.position(traced, store, 0).0,
            step,
            "the store and the load share a step"
        );

        let (_, mut run) = setup.run(&app, traced);
        let invocations = run.segment(&traced.inputs).expect("the honest trace runs");
        for invocation in &invocations {
            assert_eq!(failing(&relation, &invocation.words), Vec::<usize>::new());
        }

        let mut words = invocations[step].words.clone();
        let start = words.len() - app.aux();
        words[start + position] += F::ONE;
        assert!(!failing(&relation, &words).is_empty(), "a row fails");

        let mut inputs = traced.inputs.clone();
        inputs[step][position] += F::ONE;
        let (_, mut run) = setup.run(&app, traced);
        assert!(matches!(run.segment(&inputs), Err(SegmentError::Access)));
    }
}

/// A step of trace rows 0 and 2, which skips row 1: the runner, which reads
/// only the first row's `_BEFORE` and the last row's `_AFTER` columns,
/// accepts it, and only the continuity rows between its two rows fail.
#[test]
fn a_step_of_rows_that_are_not_neighbors_fails_only_its_link_rows() {
    let setup = setup_of(NOPS, per_step(2), true);
    let (app, geometry) = setup.complete();
    let relation = application_rows(&app, &geometry);
    let traced = &setup.programs[0];
    assert!(traced.real < traced.rows.len(), "the last row is a padding row");
    let mut rows: Vec<&[F]> = traced.rows.iter().map(Vec::as_slice).collect();
    rows.remove(1);
    rows.push(rows[rows.len() - 1]);
    let inputs: Vec<Vec<F>> = rows.chunks(2).map(|pair| pair.concat()).collect();

    let (_, mut run) = setup.run(&app, traced);
    let invocations = run.segment(&inputs).expect("the runner accepts the steps");
    let links = 2 * build_batched_wasm_ccs(1).unwrap().sparse_r1cs.n..app.relation_rows();
    let failed = failing(&relation, &invocations[0].words);
    assert!(!failed.is_empty(), "the skipped row breaks a link");
    assert!(
        failed.iter().all(|row| links.contains(row)),
        "only link rows fail: {failed:?}"
    );
    for invocation in &invocations[1..] {
        assert_eq!(failing(&relation, &invocation.words), Vec::<usize>::new());
    }
}

/// `plan_wasm_memory_slots` puts the stack reads (ports 10 and 14) in slots 0
/// and 1 and the stack writes (ports 8 and 9) in slots 6 and 7. Row 2 of the
/// call trace (`local.get 0`) activates all four, so a stack read shows before
/// a stack write that comes first in catalog order (SPEC.md section 7.6 I2) and the
/// honest trace fails. The linear-memory trace runs: the pairs that its rows
/// show out of catalog order are on distinct memories.
#[test]
fn enzo_slot_plan_breaks_a_dependent_pair_on_an_honest_row() {
    let setup = setup(NonZeroUsize::MIN);
    let plan = plan_wasm_memory_slots(build_wasm_relation_layout().auxiliary.memory.entries());
    let (app, _) = setup.app(&plan.slots);
    let (linear, call) = (&setup.programs[0], &setup.programs[1]);
    let (_, mut run) = setup.run(&app, linear);
    run.segment(&linear.inputs)
        .expect("the linear-memory trace runs");
    let (_, mut run) = setup.run(&app, call);
    let Err(SegmentError::Application(error)) = run.segment(&call.inputs) else {
        panic!("the honest call trace fails");
    };
    assert!(matches!(
        error.downcast_ref::<WasmStepError>(),
        Some(WasmStepError::Row(CatalogRowError::Plan))
    ));
}

/// One segment of step inputs of another number of rows than the
/// application's: the step rejects them, in both directions.
#[test]
fn a_step_input_of_another_row_count_is_rejected() {
    for (app_rows, input_rows) in [(2, 1), (1, 2)] {
        let setup = setup(per_step(app_rows));
        let (app, _) = setup.complete();
        let linear = &setup.programs[0];
        let checked = common::checked_main(LINEAR_MEMORY);
        let mut inputs = step_inputs(&checked.trace, N, per_step(input_rows)).unwrap();
        inputs.truncate(setup.n);
        let (_, mut run) = setup.run(&app, linear);
        let Err(SegmentError::Application(error)) = run.segment(&inputs) else {
            panic!("a step input of {input_rows} rows fails an application of {app_rows} rows");
        };
        assert!(matches!(
            error.downcast_ref::<WasmStepError>(),
            Some(WasmStepError::Width { .. })
        ));
    }
}

/// The `i32.load` row with a load value that its cell does not hold. Every
/// row still binds, and the runner rejects the port.
#[test]
fn runner_rejects_a_load_of_another_value() {
    let setup = setup(NonZeroUsize::MIN);
    let (app, _) = setup.complete();
    let traced = &setup.programs[0];
    let (row, column) = access(traced, false);
    let mut inputs = traced.inputs.clone();
    inputs[row][column] += F::ONE;
    let (_, mut run) = setup.run(&app, traced);
    assert!(matches!(run.segment(&inputs), Err(SegmentError::Access)));
}

/// A run has room for the trace only when its steps have at least one row
/// per trace row.
#[test]
fn step_inputs_reject_a_trace_longer_than_the_run() {
    let checked = common::checked_main(LINEAR_MEMORY);
    let rows = checked.trace.len();
    for (steps, per_step) in [
        (rows - 1, per_step(1)),
        (rows.div_ceil(2) - 1, per_step(2)),
        (usize::MAX, per_step(2)),
    ] {
        assert!(matches!(
            step_inputs(&checked.trace, steps, per_step),
            Err(WasmApplicationError::Steps { .. })
        ));
    }
    assert_eq!(
        step_inputs(&checked.trace, rows, NonZeroUsize::MIN)
            .unwrap()
            .len(),
        rows
    );
    let inputs = step_inputs(&checked.trace, rows.div_ceil(2), per_step(2)).unwrap();
    let one_row = step_inputs(&checked.trace, 2 * rows.div_ceil(2), NonZeroUsize::MIN).unwrap();
    assert_eq!(
        inputs,
        one_row
            .chunks(2)
            .map(|pair| pair.concat())
            .collect::<Vec<_>>()
    );
}

/// Small memories and a long segment: the geometry still covers both
/// namespaces in `n` steps, so a program of it exists.
#[test]
fn smallest_geometry_covers_small_memories_in_a_long_segment() {
    let geometry = smallest_geometry([1, 2], 3, 64, 2);
    assert_eq!((geometry.r, geometry.mu), (6, 6));
    assert!(Program::new(geometry, vec![0; 1 << geometry.r], vec![0; 1 << geometry.mu]).is_ok());
}

/// The trace row and one-row value column of the first active
/// `linear_memory` write when `write` is set, else read.
fn access(traced: &Traced, write: bool) -> (usize, usize) {
    let catalog = &build_wasm_relation_layout().auxiliary.memory;
    let memory = catalog
        .entries()
        .iter()
        .find(|memory| memory.id == WasmMemoryId::LinearMemory)
        .unwrap();
    traced
        .rows
        .iter()
        .enumerate()
        .find_map(|(row, z)| {
            memory
                .ports
                .iter()
                .find(|port| {
                    matches!(port.kind, MemoryPortKind::Write { .. }) == write && is_active(z, port.activation)
                })
                .map(|port| (row, port.value_column))
        })
        .expect("the trace has the access")
}

/// The Stage 1 package that every Nebula circuit extends.
fn reference() -> Vec<u8> {
    std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../nightstream/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json"),
    )
    .unwrap()
}

/// The linear-memory program in steps of `per_step` rows through
/// `Circuit::compile`, the prover, and the memory verifier. The honest proof
/// verifies. The step of the `i32.load` row with another load value and
/// honest memory words fails the relation, so the prover cannot extend with
/// it. A statement with another final `pc` does not verify.
fn proof_verifies_and_rejects_a_changed_load_on_cpu(per_step: NonZeroUsize) {
    use nightstream::nebula::{Circuit, Verifier, VerifyError};
    let started = std::time::Instant::now();
    let setup = setup(per_step);
    let (app, geometry) = setup.complete();
    let traced = &setup.programs[0];
    let engine = nightstream::Engine::Optimized;
    let circuit = Circuit::compile(&reference(), geometry, app.clone()).unwrap();
    let prover = circuit.prover(engine, 114).unwrap();
    let verifier = Verifier::new(&circuit, engine, 114).unwrap();
    eprintln!("wasm setup elapsed={:?}", started.elapsed());

    let (program, mut run) = setup.run(&app, traced);
    let invocations = run.segment(&traced.inputs).unwrap();
    let (row, column) = access(traced, false);
    let (step, position) = setup.position(traced, row, column);
    let proving = std::time::Instant::now();
    // A prover keeps only its last proof. The test also keeps the proof that
    // the step of the changed load extends.
    let mut last = prover
        .prove(run.initial_state(), &invocations[0].words)
        .unwrap();
    let mut before_load = None;
    for (k, invocation) in invocations.iter().enumerate().skip(1) {
        let next = prover.extend(&last, &invocation.words).unwrap();
        let previous = std::mem::replace(&mut last, next);
        if k == step {
            before_load = Some(previous);
        }
    }
    eprintln!(
        "wasm {} steps of {per_step} rows elapsed={:?}",
        setup.n,
        proving.elapsed()
    );
    let statement = run.statement();
    let verifying = std::time::Instant::now();
    verifier
        .verify(&program, &statement, run.carry(), &last)
        .unwrap();
    eprintln!("wasm verify elapsed={:?}", verifying.elapsed());

    let mut words = invocations[step].words.clone();
    let start = words.len() - app.aux();
    words[start + position] += F::ONE;
    let before_load = before_load.expect("the load is after the first step");
    assert!(prover.extend(&before_load, &words).is_err());

    let pc = app.state_position(neo_wasm::layout::COL_PC_AFTER).unwrap();
    let mut changed = statement.clone();
    changed.final_[pc] += 1;
    assert!(matches!(
        verifier.verify(&program, &changed, run.carry(), &last),
        Err(VerifyError::Proof(_))
    ));
    eprintln!("wasm elapsed={:?}", started.elapsed());
}

#[test]
#[ignore = "Full production-profile memory proof; run separately under the 300-second cap."]
fn wasm_proof_verifies_and_rejects_a_changed_load_on_cpu() {
    proof_verifies_and_rejects_a_changed_load_on_cpu(NonZeroUsize::MIN);
}

#[test]
#[ignore = "Full production-profile memory proof; run separately under the 300-second cap."]
fn wasm_proof_of_two_rows_per_step_verifies_and_rejects_a_changed_load_on_cpu() {
    proof_verifies_and_rejects_a_changed_load_on_cpu(per_step(2));
}

/// `fib(2)` is 32 rows, so it fills a 32-step segment with no padding step.
const FIB_ROUNDS: u32 = 2;
const FIB_STEPS: usize = 32;

/// Prove and verify one segment of `inputs` from `initial` on Metal, and print
/// the time of each phase and of each step.
fn time_on_metal<A: Application + Clone>(
    app: A,
    geometry: Geometry,
    program: &Program,
    initial: A::State,
    inputs: &[A::Input],
) {
    use nightstream::nebula::{Circuit, Verifier};
    use std::time::Instant;
    let engine = nightstream::Engine::Metal;
    let started = Instant::now();
    eprintln!("{geometry:?}");
    let circuit = Circuit::compile(&reference(), geometry, app.clone()).unwrap();
    eprintln!("compile elapsed={:?}", started.elapsed());
    eprintln!("application size={:?}", circuit.application_size().unwrap());
    let prover = circuit.prover(engine, 114).unwrap();
    let verifier = Verifier::new(&circuit, engine, 114).unwrap();
    eprintln!("setup elapsed={:?}", started.elapsed());

    let mut run = Run::new(Context::new(app, program), initial);
    let invocations = run.segment(inputs).unwrap();
    eprintln!("witness words per step={}", invocations[0].words.len());
    let proving = Instant::now();
    // A prover keeps only its last proof.
    let mut last: Option<nightstream::Proof> = None;
    for (k, invocation) in invocations.iter().enumerate() {
        let step = Instant::now();
        let proof = match &last {
            None => prover.prove(run.initial_state(), &invocation.words),
            Some(last) => prover.extend(last, &invocation.words),
        }
        .unwrap();
        eprintln!("step {k} elapsed={:?}", step.elapsed());
        last = Some(proof);
    }
    eprintln!("prove {} steps elapsed={:?}", invocations.len(), proving.elapsed());
    let verifying = Instant::now();
    verifier
        .verify(program, &run.statement(), run.carry(), last.as_ref().unwrap())
        .unwrap();
    eprintln!("verify elapsed={:?}", verifying.elapsed());
    eprintln!("total elapsed={:?}", started.elapsed());
}

/// Enzo's Fibonacci program proved and verified on Metal in one segment of
/// `steps` steps of `per_step` rows: the largest `fib(n)` whose trace fits.
fn fibonacci_proof_times_on_metal(steps: usize, per_step: NonZeroUsize) {
    let capacity = steps * per_step.get();
    let rounds = (1..)
        .take_while(|n| common::checked_main(&fib_wat(*n)).trace.len() <= capacity)
        .last()
        .expect("fib(1) fits the segment");
    let setup = Setup::of(vec![traced(&fib_wat(rounds), steps, per_step)], steps, per_step);
    let (app, geometry) = setup.complete();
    let traced = &setup.programs[0];
    eprintln!(
        "fib({rounds}) rows={} of {capacity}, {per_step} rows per step",
        traced.real
    );
    let program = app
        .program(&traced.preload)
        .expect("the regions hold the preload");
    time_on_metal(app, geometry, &program, traced.initial.clone(), &traced.inputs);
}

/// `fib(2)` at one row per step: 32 steps.
#[test]
#[ignore = "Metal timing run; build with --features nightstream/metal and run separately under the 300-second cap."]
fn fibonacci_proof_times_on_metal_at_one_row_per_step() {
    fibonacci_proof_times_on_metal(FIB_STEPS, NonZeroUsize::MIN);
}

/// 16 steps of 2, 4 and 8 rows: one segment geometry, so only the rows per
/// step change between the three runs.
#[test]
#[ignore = "Metal timing run; build with --features nightstream/metal and run separately under the 300-second cap."]
fn fibonacci_proof_times_on_metal_at_two_rows_per_step() {
    fibonacci_proof_times_on_metal(16, per_step(2));
}

#[test]
#[ignore = "Metal timing run; build with --features nightstream/metal and run separately under the 300-second cap."]
fn fibonacci_proof_times_on_metal_at_four_rows_per_step() {
    fibonacci_proof_times_on_metal(16, per_step(4));
}

#[test]
#[ignore = "Metal timing run; build with --features nightstream/metal and run separately under the 300-second cap."]
fn fibonacci_proof_times_on_metal_at_eight_rows_per_step() {
    fibonacci_proof_times_on_metal(16, per_step(8));
}

#[test]
#[ignore = "Metal timing run; build with --features nightstream/metal and run separately under the 300-second cap."]
fn fibonacci_proof_times_on_metal_at_sixteen_rows_per_step() {
    fibonacci_proof_times_on_metal(16, per_step(16));
}
/// The memory rows of a geometry with `width` state words and no private word
/// or row. Every step is idle.
#[derive(Clone)]
struct Idle {
    width: usize,
    b_ops: usize,
}

impl Application for Idle {
    type Input = ();
    type State = ();
    type Error = std::convert::Infallible;

    fn width(&self) -> usize {
        self.width
    }

    fn aux(&self) -> usize {
        0
    }

    fn port_count(&self) -> usize {
        self.b_ops
    }

    fn words(&self, _: &()) -> Vec<u64> {
        vec![0; self.width]
    }

    fn rows(&self, _: &View) -> Result<Vec<Expr>, nightstream::Error> {
        Ok(Vec::new())
    }

    fn step(&self, _: &(), _: &(), _: &Memory) -> Result<Transition<()>, Self::Error> {
        Ok(Transition {
            next: (),
            ports: vec![None; self.b_ops],
            aux: Vec::new(),
        })
    }
}

/// The control of `fibonacci_proof_times_on_metal`: its geometry and state
/// words with no WASM row and no private word, so a step costs only the
/// recursion and the memory rows.
#[test]
#[ignore = "Metal timing run; build with --features nightstream/metal and run separately under the 300-second cap."]
fn idle_proof_times_on_metal() {
    let setup = Setup::of(
        vec![traced(&fib_wat(FIB_ROUNDS), FIB_STEPS, NonZeroUsize::MIN)],
        FIB_STEPS,
        NonZeroUsize::MIN,
    );
    let (wasm, geometry) = setup.complete();
    let app = Idle {
        width: wasm.width(),
        b_ops: geometry.b_ops,
    };
    let program = Program::new(geometry, vec![0; 1 << geometry.r], vec![0; 1 << geometry.mu]).unwrap();
    time_on_metal(app, geometry, &program, (), &vec![(); geometry.n]);
}

/// The parts of the `fib(2)` step for `scripts/fprime_breakdown.py`: the
/// memory phases and witness regions of `nebula::size_parts`, with the
/// application rows split into the WASM relation, the constant and
/// continuity rows, and the catalog binding. They sum to the compiled
/// circuit's application size. It prints the chart command.
#[test]
#[ignore = "Compiles one circuit; run separately under the 300-second cap."]
fn fibonacci_step_size_for_the_breakdown_chart() {
    use nightstream::nebula::{size_parts, Circuit};
    use nightstream::ApplicationSize;
    let setup = Setup::of(
        vec![traced(&fib_wat(FIB_ROUNDS), FIB_STEPS, NonZeroUsize::MIN)],
        FIB_STEPS,
        NonZeroUsize::MIN,
    );
    let (wasm, geometry) = setup.complete();
    let rows = |rows| ApplicationSize {
        witness_words: 0,
        local_words: 0,
        rows,
    };
    let mut parts: Vec<(String, ApplicationSize)> = Vec::new();
    for part in size_parts(&geometry, &wasm).unwrap() {
        match part.name {
            "Application rows" => {
                assert_eq!(part.size.local_words, 0, "the WASM rows are rank-one rows");
                let (relation, links) = (wasm.relation_rows(), 1 + 2 * wasm.width());
                parts.push(("WASM step / Relation rows".into(), rows(relation)));
                parts.push(("WASM step / Constant and continuity rows".into(), rows(links)));
                parts.push((
                    "WASM step / Catalog binding rows".into(),
                    rows(part.size.rows - relation - links),
                ));
            }
            "Application state words" => parts.push(("WASM step / Continuity state words".into(), part.size)),
            "Application private words" => parts.push(("WASM step / Witness z (private words)".into(), part.size)),
            name => parts.push((format!("Nebula memory / {name}"), part.size)),
        }
    }
    let total = parts
        .iter()
        .fold(rows(0), |total, (_, size)| ApplicationSize {
            witness_words: total.witness_words + size.witness_words,
            local_words: total.local_words + size.local_words,
            rows: total.rows + size.rows,
        });
    let compiled = Circuit::compile(&reference(), geometry, wasm)
        .unwrap()
        .application_size()
        .unwrap();
    assert_eq!(total, compiled, "the parts sum to the compiled application size");
    let numbers = |size: &ApplicationSize| format!("{},{},{}", size.witness_words, size.local_words, size.rows);
    let flags: Vec<String> = parts
        .iter()
        .map(|(name, size)| format!("--part '{name}={}'", numbers(size)))
        .collect();
    println!(
        "python3 scripts/fprime_breakdown.py --name 'Enzo fib({FIB_ROUNDS}) WASM step' --application {} {} \
         --out wasm-fib-step.html",
        numbers(&total),
        flags.join(" ")
    );
}
