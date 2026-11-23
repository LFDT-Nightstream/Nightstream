#[path = "fold_workloads/counter.rs"]
mod counter;

use std::collections::BTreeSet;
use std::error::Error;
use std::num::NonZeroUsize;
use std::time::Instant;

use neo_wasm::host_event_bindings::{EventBlock, ExportTemplate, HostEventBindings, Limb, SlotBinding};
use neo_wasm::memory_semantics::preload_host_event_tables;
use neo_wasm::nebula::{plan_ordered_slots, step_inputs, WasmApplication, WasmLimits, WasmSize, WasmStep};
use neo_wasm::{WasmOpcode, WasmOutputState, WasmProgramArtifacts, WasmStepState, WasmVmStep};
use nightstream::nebula::{Circuit, Geometry};
use wasmtime::Val;

const ROWS: NonZeroUsize = NonZeroUsize::new(16).unwrap();
const STEPS_PER_SEGMENT: usize = 4;
const TRIALS: usize = 5;

#[derive(Clone, Copy)]
enum Workload {
    Hash,
    Memory { words: u32 },
}

impl Workload {
    fn export(self) -> (&'static str, u32) {
        match self {
            Self::Hash => ("hash", 1),
            Self::Memory { .. } => ("memory", 2),
        }
    }

    fn check(self, rounds: u32, trace: &[WasmVmStep]) -> (u32, usize) {
        let mut accesses = trace
            .iter()
            .filter_map(|step| step.linear_memory.map(|access| (step.opcode, access)));
        let mut touched = BTreeSet::new();
        let result = match self {
            Self::Hash => (0..rounds).fold(0x811c9dc5u32, |hash, i| {
                (hash ^ u32::from(i as u8)).wrapping_mul(0x01000193)
            }),
            Self::Memory { words } => {
                let mut image = vec![0u32; words as usize];
                let mut seed = 0x12345678u32;
                let mut sum = 0u32;
                for _ in 0..rounds {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    let index = (seed >> 28) % words;
                    touched.insert(index);
                    let cell = &mut image[index as usize];
                    let next = cell.wrapping_add(seed);
                    for (opcode, after) in [(WasmOpcode::I32Load, *cell), (WasmOpcode::I32Store, next)] {
                        let (op, access) = accesses.next().expect("missing guest memory access");
                        assert_eq!(
                            (op, access.lane0.word_addr, access.width_bytes, access.byte_offset),
                            (opcode, u64::from(index), 4, 0)
                        );
                        assert_eq!((access.lane0.value_before, access.lane0.value_after), (*cell, after));
                    }
                    *cell = next;
                    sum = sum.wrapping_add(next);
                }
                sum
            }
        };
        assert!(accesses.next().is_none());
        (result, touched.len())
    }
}

struct Case {
    name: &'static str,
    region_words: u32,
    rounds: u32,
    touched: usize,
    run: Run,
}

struct Run {
    instructions: usize,
    trace_rows: usize,
    initial: WasmStepState,
    last: WasmStepState,
    inputs: Vec<WasmStep>,
}

fn prepare(wasm: &[u8], bindings: &HostEventBindings, workload: Workload, rounds: u32) -> Result<Case, Box<dyn Error>> {
    let (name, fref) = workload.export();
    let mut args = vec![Val::I32(rounds as i32)];
    if let Workload::Memory { words } = workload {
        args.push(Val::I32((words - 1) as i32));
    }
    let run = neo_wasm::collect_wasmtime_steps(wasm, bindings, name, &args)?;
    let artifacts = run.artifacts();
    let trace = neo_wasm::traces_from_wasmtime_steps_with_host_events(&run.steps, artifacts, Default::default())?;
    let (expected, touched) = workload.check(rounds, &trace);
    assert_eq!(run.results, [(expected as i32).to_string()]);
    let last = trace.last().unwrap().state_after;
    assert_eq!(
        last.output,
        WasmOutputState {
            enabled: true,
            value_lo: expected,
            value_hi: 0
        }
    );
    let region_words = match workload {
        Workload::Hash => 0,
        Workload::Memory { words } => words,
    };
    Ok(Case {
        name,
        rounds,
        region_words,
        touched,
        run: prepare_run(&trace, artifacts, bindings, fref)?,
    })
}

fn prepare_run(
    trace: &[WasmVmStep],
    artifacts: &WasmProgramArtifacts,
    bindings: &HostEventBindings,
    fref: u32,
) -> Result<Run, Box<dyn Error>> {
    let last = trace.last().unwrap().state_after;
    assert!(last.is_terminal() && !last.trapped);
    let pc = artifacts
        .tables
        .function_entries
        .iter()
        .find(|(id, _)| *id == u64::from(fref))
        .unwrap()
        .1;
    let initial =
        neo_wasm::host_event_top_level_initial_state(&artifacts.tables, pc, bindings, fref, Default::default())?;
    assert_eq!(initial, trace[0].state_before);
    let steps = trace.len().div_ceil(ROWS.get() * STEPS_PER_SEGMENT) * STEPS_PER_SEGMENT;
    Ok(Run {
        instructions: trace
            .iter()
            .filter(|step| step.row_kind.is_program())
            .count(),
        trace_rows: trace.len(),
        initial,
        last,
        inputs: step_inputs(trace, steps, ROWS)?,
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let wasm = wat::parse_str(include_str!("fold_workloads.wat"))?;
    let mut bindings = HostEventBindings::default();
    for (fref, params) in [(1, 1), (2, 2)] {
        let mut slots = [SlotBinding::Const(0); neo_wasm::comm_chain::COMM_CHAIN_EVENT_ARGS];
        for local in 0..params {
            slots[usize::from(local)] = SlotBinding::InputLocal {
                input: local,
                local,
                limb: Limb::Lo,
            };
        }
        bindings.exports.insert(
            fref,
            ExportTemplate {
                entry: vec![EventBlock::op(u64::from(fref), slots)],
                exit: vec![],
                entry_input_count: params,
            },
        );
    }
    let artifacts = neo_wasm::extract_wasm_program_artifacts(&wasm)?;
    let mut preload = neo_wasm::preload_from_program_artifacts(&artifacts);
    preload_host_event_tables(&mut preload, &bindings, &artifacts);
    let (counter_preload, counter) = counter::prepare()?;
    let needed = WasmLimits::of(&[&preload, &counter_preload], &[])?;
    let base = WasmSize::Micro.limits();
    let limits = WasmLimits {
        pc: base.pc.max(needed.pc),
        function: base.function.max(needed.function),
        global: base.global.max(needed.global),
        event: base.event.max(needed.event),
        ..base
    };
    assert!(limits.holds(&needed));
    let (regions, cells) = limits.regions()?;
    let [rom, ram] = cells.map(|n| n.next_power_of_two().ilog2() as usize);
    let slots = plan_ordered_slots(ROWS)?;
    let geometry = Geometry::new(rom, ram, slots.len(), STEPS_PER_SEGMENT, 12).unwrap();
    eprintln!("rows/step={ROWS} steps/segment={STEPS_PER_SEGMENT} trials={TRIALS} geometry={geometry:?}");
    let app = WasmApplication::with_slots(&geometry, regions, ROWS, &slots)?;
    let loops = app.program(&preload)?;
    let contract = app.program(&counter_preload)?;
    let mut cases = vec![(&contract, counter)];
    for rounds in [4, 8, 16] {
        for workload in [Workload::Hash, Workload::Memory { words: 16 }] {
            cases.push((&loops, prepare(&wasm, &bindings, workload, rounds)?));
        }
    }
    cases.push((&loops, prepare(&wasm, &bindings, Workload::Memory { words: 4 }, 16)?));

    let circuit = Circuit::compile(geometry, app)?;
    let prover = circuit.prover(nightstream::Engine::Optimized, 109)?;
    let verifier = circuit.verifier(nightstream::Engine::Optimized, 109)?;

    println!("trial\tworkload\trounds\tregion_words\tdistinct_words\tinstructions\ttrace_rows\tsteps\tfolds\tprove_s");
    for trial in 0..TRIALS {
        let count = cases.len();
        for offset in 0..count {
            let (program, case) = &cases[(trial + offset) % count];
            let start = Instant::now();
            let mut proof_run = prover.start(program, case.run.initial)?;
            for segment in case.run.inputs.chunks_exact(STEPS_PER_SEGMENT) {
                proof_run.segment(segment)?;
            }
            let proof = proof_run.finish()?;
            let elapsed = start.elapsed();
            let verified = verifier.verify(program, &case.run.initial, &case.run.last, &proof)?;
            assert_eq!(verified.steps, case.run.inputs.len() as u64);
            println!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.3}",
                trial + 1,
                case.name,
                case.rounds,
                case.region_words,
                case.touched,
                case.run.instructions,
                case.run.trace_rows,
                verified.steps,
                verified.steps - 1,
                elapsed.as_secs_f64()
            );
        }
    }
    Ok(())
}
