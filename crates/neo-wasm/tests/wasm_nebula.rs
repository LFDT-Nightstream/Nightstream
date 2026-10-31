//! Enzo's WASM relation proved with Nebula memory checking (SPEC.md section 7.6). One
//! trace row is one Nebula step: its private words are the row's witness `z`,
//! its state is the continuity words (one per link, `_AFTER` of a row equal to
//! `_BEFORE` of the next), and `nightstream::nebula::CatalogBinding` checks
//! every memory of the catalog. The regions come from what the two test
//! programs need, so one circuit serves both programs. The slot plan is
//! `plan_ordered_wasm_memory_slots`, which keeps each pair of one memory with
//! a write in catalog order on every row that the relation accepts.

mod common;

use neo_math::F;
use neo_wasm::host_event_bindings::HostEventBindings;
use neo_wasm::memory_semantics::preload_host_event_tables;
use neo_wasm::nebula::{
    is_active, smallest_geometry, smallest_regions, state_words, step_inputs, WasmApplication, WasmStepError,
};
use neo_wasm::{
    build_wasm_relation_layout, plan_ordered_wasm_memory_slots, plan_wasm_memory_slots, preload_from_program_artifacts,
    top_level_initial_state, WasmMemoryId, WasmMemoryPreload,
};
use nightstream::nebula::{
    Application, CatalogRowError, Context, Expr, Geometry, Memory, MemoryPortActivation, MemoryPortKind, Program,
    Region, Run, SegmentError, Transition, View,
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

/// Steps per segment: the call trace has 8 rows, the linear-memory trace 6
/// rows and 2 padding rows.
const N: usize = 8;

/// A test program as its verifier and prover see it.
struct Traced {
    /// The witness of each trace row, then of each padding row, one segment
    /// in all.
    rows: Vec<Vec<F>>,
    real: usize,
    preload: WasmMemoryPreload,
    /// The continuity words of the entry state (`top_level_initial_state`).
    initial: Vec<u64>,
}

fn traced(wat: &str, n: usize) -> Traced {
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
        rows: step_inputs(&checked.trace, n).expect("the trace fits one segment"),
        real: checked.trace.len(),
        preload,
        initial: state_words(top_level_initial_state(tables, entry_pc)),
    }
}

struct Setup {
    programs: Vec<Traced>,
    activations: Vec<MemoryPortActivation>,
    regions: Vec<(WasmMemoryId, Region)>,
    log2: [usize; 2],
    /// Steps per segment.
    n: usize,
}

/// The linear-memory and call programs, `N` steps each.
fn setup() -> Setup {
    Setup::of(vec![traced(LINEAR_MEMORY, N), traced(CALL, N)], N)
}

impl Setup {
    /// One circuit for `programs`, each traced to one segment of `n` steps.
    fn of(programs: Vec<Traced>, n: usize) -> Setup {
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
        }
    }

    /// The application and geometry of a slot plan.
    fn app(&self, slots: &[Vec<usize>]) -> (WasmApplication, Geometry) {
        let geometry = smallest_geometry(self.log2, slots.len(), self.n, 1);
        let app = WasmApplication::with_slots(&geometry, self.regions.clone(), slots).expect("an application");
        (app, geometry)
    }

    /// The application of the static plan that keeps each dependent pair in
    /// catalog order.
    fn complete(&self) -> (WasmApplication, Geometry) {
        self.app(&plan_ordered_wasm_memory_slots(build_wasm_relation_layout().auxiliary.memory.entries()).slots)
    }

    fn run(&self, app: &WasmApplication, traced: &Traced) -> (Program, Run<WasmApplication>) {
        let program = app
            .program(&traced.preload)
            .expect("the regions hold the preload");
        let run = Run::new(Context::new(app.clone(), &program), traced.initial.clone());
        (program, run)
    }
}

/// Both honest padded traces run through the native Nebula runner, which
/// applies each step's ports and rejects a port that reads another value than
/// its cell holds. Every port of every row is active exactly when its catalog
/// port is: the final timestamp counts one access per active catalog port.
/// Padding rows activate no port and keep the state.
#[test]
fn honest_padded_traces_run_through_nebula() {
    let setup = setup();
    let (app, geometry) = setup.complete();
    assert!(app.binding().host_checked().is_empty(), "every memory is checked");
    eprintln!("geometry {geometry:?}");
    for (id, region) in &setup.regions {
        eprintln!("region {id} base={} widths={:?}", region.base, region.widths);
    }
    for traced in &setup.programs {
        let (_, mut run) = setup.run(&app, traced);
        let invocations = run.segment(&traced.rows).expect("the honest trace runs");
        let statement = run.statement();
        assert_eq!(
            statement.final_,
            app.output_words(&traced.rows[traced.real - 1]),
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
}

/// `plan_wasm_memory_slots` puts the stack reads (ports 10 and 14) in slots 0
/// and 1 and the stack writes (ports 8 and 9) in slots 6 and 7. Row 2 of the
/// call trace (`local.get 0`) activates all four, so a stack read shows before
/// a stack write that comes first in catalog order (SPEC.md section 7.6 I2) and the
/// honest trace fails. The linear-memory trace runs: the pairs that its rows
/// show out of catalog order are on distinct memories.
#[test]
fn enzo_slot_plan_breaks_a_dependent_pair_on_an_honest_row() {
    let setup = setup();
    let plan = plan_wasm_memory_slots(build_wasm_relation_layout().auxiliary.memory.entries());
    let (app, _) = setup.app(&plan.slots);
    let (linear, call) = (&setup.programs[0], &setup.programs[1]);
    let (_, mut run) = setup.run(&app, linear);
    run.segment(&linear.rows)
        .expect("the linear-memory trace runs");
    let (_, mut run) = setup.run(&app, call);
    let Err(SegmentError::Application(error)) = run.segment(&call.rows) else {
        panic!("the honest call trace fails");
    };
    assert!(matches!(
        error.downcast_ref::<WasmStepError>(),
        Some(WasmStepError::Row(CatalogRowError::Plan))
    ));
}

/// The `i32.load` row with a load value that its cell does not hold. Every
/// row still binds, and the runner rejects the port.
#[test]
fn runner_rejects_a_load_of_another_value() {
    let setup = setup();
    let (app, _) = setup.complete();
    let traced = &setup.programs[0];
    let (row, column) = load(traced);
    let mut rows = traced.rows.clone();
    rows[row][column] += F::ONE;
    let (_, mut run) = setup.run(&app, traced);
    assert!(matches!(run.segment(&rows), Err(SegmentError::Access)));
}

/// A run has room for the trace only when it has at least one step per row.
#[test]
fn step_inputs_reject_a_trace_longer_than_the_run() {
    let checked = common::checked_main(LINEAR_MEMORY);
    let rows = checked.trace.len();
    assert!(matches!(
        step_inputs(&checked.trace, rows - 1),
        Err(neo_wasm::nebula::WasmApplicationError::Steps { .. })
    ));
    assert_eq!(step_inputs(&checked.trace, rows).unwrap().len(), rows);
}

/// Small memories and a long segment: the geometry still covers both
/// namespaces in `n` steps, so a program of it exists.
#[test]
fn smallest_geometry_covers_small_memories_in_a_long_segment() {
    let geometry = smallest_geometry([1, 2], 3, 64, 2);
    assert_eq!((geometry.r, geometry.mu), (6, 6));
    assert!(Program::new(geometry, vec![0; 1 << geometry.r], vec![0; 1 << geometry.mu]).is_ok());
}

/// The row and value column of the first active `linear_memory` read.
fn load(traced: &Traced) -> (usize, usize) {
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
                .find(|port| port.kind == MemoryPortKind::Read && is_active(z, port.activation))
                .map(|port| (row, port.value_column))
        })
        .expect("the trace loads")
}

/// The Stage 1 package that every Nebula circuit extends.
fn reference() -> Vec<u8> {
    std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../nightstream/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json"),
    )
    .unwrap()
}

/// The linear-memory program through `Circuit::compile`, the prover, and the
/// memory verifier. The honest proof verifies. The `i32.load` step with
/// another load value and honest memory words fails the relation, so the
/// prover cannot extend with it. A statement with another final `pc` does
/// not verify.
#[test]
#[ignore = "Full production-profile memory proof; run separately under the 300-second cap."]
fn wasm_proof_verifies_and_rejects_a_changed_load_on_cpu() {
    use nightstream::nebula::{Circuit, Verifier, VerifyError};
    let started = std::time::Instant::now();
    let setup = setup();
    let (app, geometry) = setup.complete();
    let traced = &setup.programs[0];
    let engine = nightstream::Engine::Optimized;
    let circuit = Circuit::compile(&reference(), geometry, app.clone()).unwrap();
    let prover = circuit.prover(engine, 114).unwrap();
    let verifier = Verifier::new(&circuit, engine, 114).unwrap();
    eprintln!("wasm setup elapsed={:?}", started.elapsed());

    let (program, mut run) = setup.run(&app, traced);
    let invocations = run.segment(&traced.rows).unwrap();
    let proving = std::time::Instant::now();
    let mut proofs = vec![prover
        .prove(run.initial_state(), &invocations[0].words)
        .unwrap()];
    for invocation in &invocations[1..] {
        let next = prover
            .extend(proofs.last().unwrap(), &invocation.words)
            .unwrap();
        proofs.push(next);
    }
    eprintln!("wasm {N} steps elapsed={:?}", proving.elapsed());
    let statement = run.statement();
    let verifying = std::time::Instant::now();
    verifier
        .verify(&program, &statement, run.carry(), &proofs[N - 1])
        .unwrap();
    eprintln!("wasm verify elapsed={:?}", verifying.elapsed());

    let (row, column) = load(traced);
    let mut words = invocations[row].words.clone();
    let start = words.len() - app.aux();
    words[start + column] += F::ONE;
    assert!(prover.extend(&proofs[row - 1], &words).is_err());

    let pc = app.state_position(neo_wasm::layout::COL_PC_AFTER).unwrap();
    let mut changed = statement.clone();
    changed.final_[pc] += 1;
    assert!(matches!(
        verifier.verify(&program, &changed, run.carry(), &proofs[N - 1]),
        Err(VerifyError::Proof(_))
    ));
    eprintln!("wasm elapsed={:?}", started.elapsed());
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
    let prover = circuit.prover(engine, 114).unwrap();
    let verifier = Verifier::new(&circuit, engine, 114).unwrap();
    eprintln!("setup elapsed={:?}", started.elapsed());

    let mut run = Run::new(Context::new(app, program), initial);
    let invocations = run.segment(inputs).unwrap();
    eprintln!("witness words per step={}", invocations[0].words.len());
    let proving = Instant::now();
    let mut proofs: Vec<nightstream::Proof> = Vec::with_capacity(invocations.len());
    for (k, invocation) in invocations.iter().enumerate() {
        let step = Instant::now();
        let proof = match proofs.last() {
            None => prover.prove(run.initial_state(), &invocation.words),
            Some(last) => prover.extend(last, &invocation.words),
        }
        .unwrap();
        eprintln!("step {k} elapsed={:?}", step.elapsed());
        proofs.push(proof);
    }
    eprintln!("prove {} steps elapsed={:?}", proofs.len(), proving.elapsed());
    let verifying = Instant::now();
    verifier
        .verify(program, &run.statement(), run.carry(), proofs.last().unwrap())
        .unwrap();
    eprintln!("verify elapsed={:?}", verifying.elapsed());
    eprintln!("total elapsed={:?}", started.elapsed());
}

/// Enzo's Fibonacci program proved and verified on Metal in one segment.
#[test]
#[ignore = "Metal timing run; build with --features nightstream/metal and run separately under the 300-second cap."]
fn fibonacci_proof_times_on_metal() {
    let setup = Setup::of(vec![traced(&fib_wat(FIB_ROUNDS), FIB_STEPS)], FIB_STEPS);
    let (app, geometry) = setup.complete();
    let traced = &setup.programs[0];
    eprintln!("fib({FIB_ROUNDS}) rows={}", traced.real);
    let program = app
        .program(&traced.preload)
        .expect("the regions hold the preload");
    time_on_metal(app, geometry, &program, traced.initial.clone(), &traced.rows);
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
    let setup = Setup::of(vec![traced(&fib_wat(FIB_ROUNDS), FIB_STEPS)], FIB_STEPS);
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
    let setup = Setup::of(vec![traced(&fib_wat(FIB_ROUNDS), FIB_STEPS)], FIB_STEPS);
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
