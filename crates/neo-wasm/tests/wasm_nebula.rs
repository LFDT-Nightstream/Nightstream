//! Enzo's WASM relation proved with Nebula memory checking (SPEC.md section
//! 7.6). One Nebula step proves `per_step` consecutive trace rows: its
//! private words are the witnesses `z` of its rows, its state is the
//! continuity words (one per link, `_AFTER` of a row equal to `_BEFORE` of
//! the next), and `nightstream::nebula::R1csApplication` checks every memory
//! of the batched catalog. The regions come from what the test programs
//! need (`WasmLimits::of`), so one circuit serves each program of a setup. The slot plan is
//! `plan_ordered_slots`, which keeps each pair of one memory with a write in
//! trace order on every row that the relation accepts.

mod common;

#[path = "redteam/table_bounds.rs"]
mod table_bounds_redteam;

#[path = "redteam/sign_extension.rs"]
mod sign_extension_redteam;

#[path = "redteam/inactive_output.rs"]
mod inactive_output_redteam;

#[path = "redteam/count_field_alias.rs"]
mod count_field_alias_redteam;

#[path = "redteam/module_initialization.rs"]
mod module_initialization_redteam;

use std::cell::RefCell;
use std::num::NonZeroUsize;
use std::rc::Rc;

use neo_math::F;
use neo_wasm::batch::build_batched_wasm_ccs;
use neo_wasm::host_event_bindings::HostEventBindings;
use neo_wasm::memory_semantics::preload_host_event_tables;
use neo_wasm::nebula::{
    is_active, plan_ordered_slots, state_words, step_inputs, WasmApplication, WasmApplicationError, WasmLimits,
    WasmStep, WasmStepError,
};
use neo_wasm::{
    build_wasm_relation_layout, preload_from_program_artifacts, top_level_initial_state, WasmMemoryId,
    WasmMemoryPreload, WasmStepState,
};
use nightstream::nebula::{
    size_parts, Application, CatalogRowError, Expr, Geometry, Memory, Program, R1csStepError, Region, Run,
    SegmentError, Transition, View,
};
use nightstream::relation::{MemoryPortActivation, MemoryPortKind};
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
    /// The state after each row of `rows`; a padding row keeps the state.
    states: Vec<WasmStepState>,
    /// The step inputs of the segment: `per_step` rows each.
    inputs: Vec<WasmStep>,
    real: usize,
    preload: WasmMemoryPreload,
    /// The entry state (`top_level_initial_state`).
    initial: WasmStepState,
    /// The state after the last trace row.
    last: WasmStepState,
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
        &checked.artifacts,
    );
    let rows = step_inputs(&checked.trace, steps * per_step.get(), NonZeroUsize::MIN).expect("the trace fits");
    Traced {
        states: rows.iter().map(|row| row.next).collect(),
        rows: rows.into_iter().map(|row| row.z).collect(),
        inputs: step_inputs(&checked.trace, steps, per_step).expect("the trace fits one segment"),
        real: checked.trace.len(),
        preload,
        initial: top_level_initial_state(tables, entry_pc).unwrap(),
        last: checked.trace.last().expect("a trace row").state_after,
    }
}

fn bits(value: u64) -> usize {
    (u64::BITS - value.leading_zeros()) as usize
}

/// The smallest geometry with ROM and RAM of at least `2^log2` cells, `b_ops`
/// slots, segments of `n` steps, and at most `s_max` segments, so that a test
/// proof stays small. `n` must be a power of two. Each space has at least `n`
/// cells, so that the scan covers both in `n` steps (`SPEC.md` section 7.1,
/// rule 1).
fn smallest_geometry(log2: [usize; 2], b_ops: usize, n: usize, s_max: usize) -> Geometry {
    let low = n.trailing_zeros() as usize;
    // Rule 4: the RAM is at least as large as the ROM.
    let r = log2[0].max(low);
    let mu = log2[1].max(r);
    Geometry::new(r, mu, b_ops, n, s_max).expect("a valid, secure test geometry")
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
        let (regions, used) = WasmLimits::of(
            &programs
                .iter()
                .map(|traced| &traced.preload)
                .collect::<Vec<_>>(),
            &programs
                .iter()
                .flat_map(|traced| traced.rows.clone())
                .collect::<Vec<_>>(),
        )
        .unwrap()
        .regions()
        .unwrap();
        Setup {
            activations: catalog
                .entries()
                .iter()
                .flat_map(|memory| memory.ports.iter().map(|port| port.activation))
                .collect(),
            regions,
            log2: used.map(|cells| bits(cells - 1)),
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
        self.app(&plan_ordered_slots(self.per_step).unwrap())
    }

    fn run(&self, app: &WasmApplication, traced: &Traced) -> (Program, Run<WasmApplication>) {
        let program = app
            .program(&traced.preload)
            .expect("the regions hold the preload");
        let run = Run::new(app.clone(), &program, traced.initial);
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
    type Input = WasmStep;
    type State = WasmStepState;
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

    fn words(&self, state: &WasmStepState) -> Vec<u64> {
        Application::words(&self.app, state)
    }

    fn rows(&self, view: &View) -> Result<Vec<Expr>, nightstream::Error> {
        let rows = self.app.rows(view)?;
        self.rows.replace(rows.clone());
        Ok(rows)
    }

    fn step(
        &self,
        state: &WasmStepState,
        input: &WasmStep,
        memory: &Memory,
    ) -> Result<Transition<WasmStepState>, WasmStepError> {
        self.app.step(state, input, memory)
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
            assert_eq!(
                *run.state(),
                traced.last,
                "padding keeps the state of the last trace row"
            );
            for z in &traced.rows[traced.real..] {
                assert!(
                    setup
                        .activations
                        .iter()
                        .all(|activation| !is_active(z, *activation).unwrap()),
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
                        .filter(|activation| is_active(z, **activation).unwrap())
                        .count() as u64
                })
                .sum::<u64>();
            assert_eq!(run.timestamp(), events, "one access per active catalog port");
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
        inputs[step].z[position] += F::ONE;
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
    let mut rows: Vec<(&[F], WasmStepState)> = traced
        .rows
        .iter()
        .map(Vec::as_slice)
        .zip(traced.states.iter().copied())
        .collect();
    rows.remove(1);
    rows.push(rows[rows.len() - 1]);
    let inputs: Vec<WasmStep> = rows
        .chunks(2)
        .map(|pair| WasmStep {
            z: pair.iter().flat_map(|(z, _)| z.iter().copied()).collect(),
            next: pair[1].1,
        })
        .collect();

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

/// Reversing the slots violates catalog order for dependent ports
/// (SPEC.md section 7.6 I2), so an honest call trace must be rejected.
#[test]
fn reversed_slot_plan_rejects_a_dependent_pair_on_an_honest_row() {
    let setup = setup(NonZeroUsize::MIN);
    let mut slots = plan_ordered_slots(NonZeroUsize::MIN).unwrap();
    slots.reverse();
    let (app, _) = setup.app(&slots);
    let call = &setup.programs[1];
    let (_, mut run) = setup.run(&app, call);
    let Err(SegmentError::Application(error)) = run.segment(&call.inputs) else {
        panic!("the honest call trace fails");
    };
    assert!(matches!(
        error.downcast_ref::<WasmStepError>(),
        Some(WasmStepError::Step(R1csStepError::Row(CatalogRowError::Plan)))
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
            Some(WasmStepError::Step(R1csStepError::Width { .. }))
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
    inputs[row].z[column] += F::ONE;
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
            .map(|pair| WasmStep {
                z: pair.iter().flat_map(|row| row.z.iter().copied()).collect(),
                next: pair[1].next,
            })
            .collect::<Vec<_>>()
    );
}

/// Every field of `WasmStepState` changes its words, so a verifier that reads
/// a field of a proved last state reads a proved value. The pattern names
/// every field, so a new field does not compile until it gets a change here.
#[test]
fn every_state_field_changes_the_state_words() {
    use neo_wasm::{WasmCountdownState, WasmEventAbsorbState, WasmFrameMoveState, WasmHostEventState, WasmOutputState};
    let checked = common::checked_main(LINEAR_MEMORY);
    let base = checked.trace[0].state_after;
    let WasmStepState {
        pc: _,
        sp: _,
        stack_frame_base: _,
        output: WasmOutputState {
            enabled: _,
            value_lo: _,
            value_hi: _,
        },
        call_stack_depth: _,
        memory_pages: _,
        max_memory_pages: _,
        locals_fbp: _,
        activation_finished: _,
        trapped: _,
        frame_move:
            WasmFrameMoveState {
                active: _,
                remaining: _,
                distance: _,
            },
        local_zero: WasmCountdownState {
            active: _,
            remaining: _,
        },
        host_callee_fref: _,
        comm_chain: _,
        event_absorb:
            WasmEventAbsorbState {
                outer_prefix: _,
                outer_chain: _,
                object_active: _,
                evbuf: _,
            },
        host_events:
            WasmHostEventState {
                turn_export_fref: _,
                host_frame_depth: _,
                awaiting_host: _,
                suspend_pending: _,
                events_remaining: _,
                event_index: _,
                args_base: _,
                slot_cursor: _,
            },
    } = base;
    let next = |pages: Option<u32>| Some(pages.unwrap_or(0) + 1);
    let mut changes: Vec<(String, Box<dyn Fn(&mut WasmStepState)>)> = vec![
        ("pc".into(), Box::new(|s| s.pc += 1)),
        ("sp".into(), Box::new(|s| s.sp += 1)),
        ("stack_frame_base".into(), Box::new(|s| s.stack_frame_base += 1)),
        ("output.enabled".into(), Box::new(|s| s.output.enabled ^= true)),
        ("output.value_lo".into(), Box::new(|s| s.output.value_lo += 1)),
        ("output.value_hi".into(), Box::new(|s| s.output.value_hi += 1)),
        ("call_stack_depth".into(), Box::new(|s| s.call_stack_depth += 1)),
        (
            "memory_pages".into(),
            Box::new(move |s| s.memory_pages = next(s.memory_pages)),
        ),
        (
            "max_memory_pages".into(),
            Box::new(move |s| s.max_memory_pages = next(s.max_memory_pages)),
        ),
        ("locals_fbp".into(), Box::new(|s| s.locals_fbp += 1)),
        (
            "activation_finished".into(),
            Box::new(|s| s.activation_finished ^= true),
        ),
        ("trapped".into(), Box::new(|s| s.trapped ^= true)),
        ("frame_move.active".into(), Box::new(|s| s.frame_move.active ^= true)),
        ("frame_move.remaining".into(), Box::new(|s| s.frame_move.remaining += 1)),
        ("frame_move.distance".into(), Box::new(|s| s.frame_move.distance += 1)),
        ("local_zero.active".into(), Box::new(|s| s.local_zero.active ^= true)),
        ("local_zero.remaining".into(), Box::new(|s| s.local_zero.remaining += 1)),
        ("host_callee_fref".into(), Box::new(|s| s.host_callee_fref += 1)),
        (
            "event_absorb.object_active".into(),
            Box::new(|s| s.event_absorb.object_active ^= true),
        ),
        (
            "host_events.turn_export_fref".into(),
            Box::new(|s| s.host_events.turn_export_fref += 1),
        ),
        (
            "host_events.host_frame_depth".into(),
            Box::new(|s| s.host_events.host_frame_depth += 1),
        ),
        (
            "host_events.awaiting_host".into(),
            Box::new(|s| s.host_events.awaiting_host ^= true),
        ),
        (
            "host_events.suspend_pending".into(),
            Box::new(|s| s.host_events.suspend_pending ^= true),
        ),
        (
            "host_events.events_remaining".into(),
            Box::new(|s| s.host_events.events_remaining += 1),
        ),
        (
            "host_events.event_index".into(),
            Box::new(|s| s.host_events.event_index += 1),
        ),
        (
            "host_events.args_base".into(),
            Box::new(|s| s.host_events.args_base += 1),
        ),
        (
            "host_events.slot_cursor".into(),
            Box::new(|s| s.host_events.slot_cursor += 1),
        ),
    ];
    for lane in 0..4 {
        changes.push((
            format!("comm_chain[{lane}]"),
            Box::new(move |s| s.comm_chain[lane] += 1),
        ));
        changes.push((
            format!("event_absorb.outer_prefix[{lane}]"),
            Box::new(move |s| s.event_absorb.outer_prefix[lane] += 1),
        ));
        changes.push((
            format!("event_absorb.outer_chain[{lane}]"),
            Box::new(move |s| s.event_absorb.outer_chain[lane] += 1),
        ));
    }
    for lane in 0..8 {
        changes.push((
            format!("event_absorb.evbuf[{lane}]"),
            Box::new(move |s| s.event_absorb.evbuf[lane] += 1),
        ));
    }
    let words = state_words(base);
    for (name, change) in &changes {
        let mut changed = base;
        change(&mut changed);
        assert_ne!(state_words(changed), words, "{name} has no words");
    }
    assert_eq!(changes.len(), 27 + 3 * 4 + 8, "one change for each field and lane");
}

#[test]
fn absent_memory_and_zero_page_memory_have_different_state_words() {
    let checked = common::checked_main(NOPS);
    let absent = checked.trace.last().unwrap().state_after;
    assert_eq!((absent.memory_pages, absent.max_memory_pages), (None, None));
    let present = WasmStepState {
        memory_pages: Some(0),
        max_memory_pages: Some(0),
        ..absent
    };
    assert_ne!(absent, present);
    assert_ne!(state_words(absent), state_words(present));
}

/// A field raised by `q` keeps a word of `q` or more, which the terminal check
/// of SPEC.md Algorithm 8 refuses, so a claim cannot name a field that differs
/// from the proved word by `q`. A step input whose next state is raised by `q`
/// has no step.
#[test]
fn a_state_field_of_q_or_more_has_no_canonical_word() {
    use p3_field::PrimeField64;
    let q = F::ORDER_U64;
    let setup = setup(NonZeroUsize::MIN);
    let traced = &setup.programs[0];
    let base = traced.last;
    let raises: [(&str, fn(&mut WasmStepState, u64)); 5] = [
        ("pc", |state, q| state.pc += q),
        ("sp", |state, q| state.sp += q),
        ("call_stack_depth", |state, q| state.call_stack_depth += q),
        ("host_frame_depth", |state, q| state.host_events.host_frame_depth += q),
        ("comm_chain[0]", |state, q| {
            state.comm_chain[0] = state.comm_chain[0].checked_add(q).unwrap()
        }),
    ];
    for (name, raise) in raises {
        let mut raised = base;
        raise(&mut raised, q);
        let words = state_words(raised);
        assert_ne!(words, state_words(base), "{name}");
        assert!(words.iter().any(|word| *word >= q), "{name} keeps a word of q or more");
    }

    let (app, _) = setup.complete();
    let (_, mut run) = setup.run(&app, traced);
    let mut inputs = traced.inputs.clone();
    inputs.last_mut().unwrap().next.pc += q;
    let Err(SegmentError::Application(error)) = run.segment(&inputs) else {
        panic!("a next state raised by q has a step");
    };
    assert!(matches!(
        error.downcast_ref::<WasmStepError>(),
        Some(WasmStepError::Next)
    ));
}

/// Both test programs, 8 one-row steps each, in two segments of four steps
/// (`S_max = 2`).
fn two_segments() -> (Setup, WasmApplication, Geometry) {
    let setup = setup(NonZeroUsize::MIN);
    let slots = plan_ordered_slots(NonZeroUsize::MIN).unwrap();
    let geometry = smallest_geometry(setup.log2, slots.len(), N / 2, 2);
    let app = WasmApplication::with_slots(&geometry, setup.regions.clone(), NonZeroUsize::MIN, &slots)
        .expect("an application");
    (setup, app, geometry)
}

/// Each trace runs natively in two segments and ends in its last state. A
/// third segment is past `S_max`.
#[test]
fn a_trace_runs_in_two_segments() {
    let (setup, app, geometry) = two_segments();
    for traced in &setup.programs {
        let program = app.program(&traced.preload).unwrap();
        let mut run = Run::new(app.clone(), &program, traced.initial);
        let (first, second) = traced.inputs.split_at(geometry.n);
        run.segment(first).expect("the first segment runs");
        run.segment(second).expect("the second segment runs");
        assert_eq!(*run.state(), traced.last);
        assert!(matches!(run.segment(second), Err(SegmentError::Carry)));
    }
}

/// The call trace proved in two segments verifies with 8 steps.
#[test]
#[ignore = "Full production-profile memory proof; run separately under the 300-second cap."]
fn a_proof_of_two_segments_verifies_on_cpu() {
    use nightstream::nebula::{Circuit, ProveError};
    let started = std::time::Instant::now();
    let (setup, app, geometry) = two_segments();
    let traced = &setup.programs[1];
    let engine = nightstream::Engine::Optimized;
    let circuit = Circuit::compile(geometry, app.clone()).unwrap();
    let prover = circuit.prover(engine, 109).unwrap();
    let verifier = circuit.verifier(engine, 109).unwrap();
    let program = app.program(&traced.preload).unwrap();
    let mut run = prover.start(&program, traced.initial).unwrap();
    for segment in traced.inputs.chunks(geometry.n) {
        run.segment(segment).unwrap();
    }
    let last = *run.state();
    let proof = run.finish().unwrap();
    let verified = verifier
        .verify(&program, &traced.initial, &last, &proof)
        .unwrap();
    assert_eq!(verified.steps, 8);
    let aliased_last = WasmStepState {
        memory_pages: Some(0),
        max_memory_pages: Some(0),
        ..last
    };
    assert_ne!(aliased_last, last);
    assert!(
        verifier
            .verify(&program, &traced.initial, &aliased_last, &proof)
            .is_err(),
        "a proof must not verify for a distinct typed memory state"
    );

    let mut forged_inputs = traced.inputs.clone();
    let forged_last_input = forged_inputs
        .last_mut()
        .expect("the call trace has a final End row");
    forged_last_input.z[neo_wasm::layout::COL_PC_AFTER] += F::ONE;
    let forged_pc = forged_last_input.next.pc + 1;
    for (bit, column) in neo_wasm::range_check::range_checked_bit_columns(neo_wasm::layout::COL_PC_AFTER)
        .expect("pc has range bits")
        .enumerate()
    {
        forged_last_input.z[column] = F::from_u64((forged_pc >> bit) & 1);
    }
    neo_wasm::write_turn_schedule_guard_witness(&mut forged_last_input.z).unwrap();
    forged_last_input.next.pc += 1;
    let mut forged_run = prover.start(&program, traced.initial).unwrap();
    let error = forged_inputs
        .chunks(geometry.n)
        .try_for_each(|segment| forged_run.segment(segment))
        .expect_err("a forged terminal pc must not have a proof");
    assert!(matches!(error, ProveError::Proof(_)));
    eprintln!("two segments elapsed={:?}", started.elapsed());
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
                    matches!(port.kind, MemoryPortKind::Write { .. }) == write && is_active(z, port.activation).unwrap()
                })
                .map(|port| (row, port.value_column))
        })
        .expect("the trace has the access")
}

/// The linear-memory program in steps of `per_step` rows through
/// `Circuit::compile`, the prover, and the memory verifier. The honest proof
/// verifies after its bytes round-trip. A run whose `i32.load` reads another
/// value has no proof: its segment fails before the prover proves it. A
/// final state with another `pc` does not verify.
fn proof_verifies_and_rejects_a_changed_load_on_cpu(per_step: NonZeroUsize) {
    use nightstream::nebula::{Circuit, ProveError, TerminalError, VerifyError};
    use p3_field::PrimeField64;
    let started = std::time::Instant::now();
    let setup = setup(per_step);
    let (app, geometry) = setup.complete();
    let traced = &setup.programs[0];
    let engine = nightstream::Engine::Optimized;
    let circuit = Circuit::compile(geometry, app.clone()).unwrap();
    let prover = circuit.prover(engine, 109).unwrap();
    let verifier = circuit.verifier(engine, 109).unwrap();
    eprintln!("wasm setup elapsed={:?}", started.elapsed());

    let program = app.program(&traced.preload).unwrap();
    let proving = std::time::Instant::now();
    let mut run = prover.start(&program, traced.initial).unwrap();
    run.segment(&traced.inputs).unwrap();
    let last = *run.state();
    let proof = run.finish().unwrap();
    eprintln!(
        "wasm {} steps of {per_step} rows elapsed={:?}",
        setup.n,
        proving.elapsed()
    );
    let proof = verifier
        .decode_proof(&prover.encode_proof(&proof).unwrap())
        .unwrap();
    let verifying = std::time::Instant::now();
    let verified = verifier
        .verify(&program, &traced.initial, &last, &proof)
        .unwrap();
    assert_eq!(verified.steps, setup.n as u64);
    eprintln!("wasm verify elapsed={:?}", verifying.elapsed());

    let (row, column) = access(traced, false);
    let (step, position) = setup.position(traced, row, column);
    let mut inputs = traced.inputs.clone();
    inputs[step].z[position] += F::ONE;
    let mut changed_run = prover.start(&program, traced.initial).unwrap();
    assert!(matches!(
        changed_run.segment(&inputs),
        Err(ProveError::Segment(SegmentError::Access))
    ));

    let changes: [(&str, fn(&mut WasmStepState)); 3] = [
        ("another pc", |state| state.pc += 1),
        ("another event chain", |state| state.comm_chain[0] ^= 1),
        ("a trap", |state| state.trapped = !state.trapped),
    ];
    for (name, change) in changes {
        let mut changed = last;
        change(&mut changed);
        assert!(
            matches!(
                verifier.verify(&program, &traced.initial, &changed, &proof),
                Err(VerifyError::Proof(_))
            ),
            "{name} does not verify"
        );
    }

    // A field raised by q has the word of the honest field after reduction,
    // so only the terminal check can refuse it.
    let q = F::ORDER_U64;
    for (name, initial, changed) in [
        (
            "a last pc raised by q",
            traced.initial,
            WasmStepState {
                pc: last.pc + q,
                ..last
            },
        ),
        (
            "a last host frame depth raised by q",
            traced.initial,
            WasmStepState {
                host_events: neo_wasm::WasmHostEventState {
                    host_frame_depth: last.host_events.host_frame_depth + q,
                    ..last.host_events
                },
                ..last
            },
        ),
        (
            "an initial pc raised by q",
            WasmStepState {
                pc: traced.initial.pc + q,
                ..traced.initial
            },
            last,
        ),
    ] {
        assert!(
            matches!(
                verifier.verify(&program, &initial, &changed, &proof),
                Err(VerifyError::Terminal(TerminalError::NonCanonical))
            ),
            "{name} does not verify"
        );
    }
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

/// Prove and verify one segment of `inputs` from `initial` on the CPU, and
/// print the time of each phase and of each step.
fn time_proofs<A: Application + Clone>(
    app: A,
    geometry: Geometry,
    program: &Program,
    initial: A::State,
    inputs: &[A::Input],
) {
    use nightstream::nebula::Circuit;
    use std::time::Instant;
    let engine = nightstream::Engine::Optimized;
    let started = Instant::now();
    eprintln!("{geometry:?}");
    let circuit = Circuit::compile(geometry, app).unwrap();
    eprintln!("compile elapsed={:?}", started.elapsed());
    eprintln!("application size={:?}", circuit.application_size().unwrap());
    let prover = circuit.prover(engine, 109).unwrap();
    let verifier = circuit.verifier(engine, 109).unwrap();
    eprintln!("setup elapsed={:?}", started.elapsed());

    let proving = Instant::now();
    let mut run = prover.start(program, initial.clone()).unwrap();
    run.segment(inputs).unwrap();
    let last = run.state().clone();
    let proof = run.finish().unwrap();
    eprintln!(
        "prove {} steps elapsed={:?}, {:?} per step",
        inputs.len(),
        proving.elapsed(),
        proving.elapsed() / inputs.len() as u32
    );
    let verifying = Instant::now();
    verifier.verify(program, &initial, &last, &proof).unwrap();
    eprintln!("verify elapsed={:?}", verifying.elapsed());
    eprintln!("total elapsed={:?}", started.elapsed());
}

/// Enzo's Fibonacci program proved and verified on the CPU in one segment of
/// `steps` steps of `per_step` rows: the largest `fib(n)` whose trace fits.
fn fibonacci_proof_times(steps: usize, per_step: NonZeroUsize) {
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
    time_proofs(app, geometry, &program, traced.initial, &traced.inputs);
}

/// `fib(2)` at one row per step: 32 steps.
#[test]
#[ignore = "Timing run; run separately under the 300-second cap."]
fn fibonacci_proof_times_at_one_row_per_step() {
    fibonacci_proof_times(FIB_STEPS, NonZeroUsize::MIN);
}

/// 16 steps of 2, 4 and 8 rows: one segment geometry, so only the rows per
/// step change between the three runs.
#[test]
#[ignore = "Timing run; run separately under the 300-second cap."]
fn fibonacci_proof_times_at_two_rows_per_step() {
    fibonacci_proof_times(16, per_step(2));
}

#[test]
#[ignore = "Timing run; run separately under the 300-second cap."]
fn fibonacci_proof_times_at_four_rows_per_step() {
    fibonacci_proof_times(16, per_step(4));
}

#[test]
#[ignore = "Timing run; run separately under the 300-second cap."]
fn fibonacci_proof_times_at_eight_rows_per_step() {
    fibonacci_proof_times(16, per_step(8));
}

#[test]
#[ignore = "Timing run; run separately under the 300-second cap."]
fn fibonacci_proof_times_at_sixteen_rows_per_step() {
    fibonacci_proof_times(16, per_step(16));
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

/// The control of `fibonacci_proof_times`: its geometry and state
/// words with no WASM row and no private word, so a step costs only the
/// recursion and the memory rows.
#[test]
#[ignore = "Timing run; run separately under the 300-second cap."]
fn idle_proof_times() {
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
    time_proofs(app, geometry, &program, (), &vec![(); geometry.n]);
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
    let compiled = Circuit::compile(geometry, wasm)
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
