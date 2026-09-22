//! Multi-turn export traces with persistent module state, per-turn input
//! bootstrapping, boundary constraints, and transcript binding.

mod common;

use common::audit::{prove_batched, verify_with_transcript, AuditProveError};
use neo_wasm::comm_chain::COMM_CHAIN_EVENT_ARGS;
use neo_wasm::host_event_bindings::{
    absorbed_blocks, EventBlock, ExportTemplate, HostEventBindings, Limb, SlotBinding,
};
use neo_wasm::witness_builder::build_witness_vector;
use neo_wasm::{host_event_top_level_initial_state_digest, preprocess_seeded_batched, WasmVmStep};
use p3_field::PrimeCharacteristicRing;
use wasmtime::component::{Component, Instance, Linker, Val as ComponentVal};
use wasmtime::{Config, Engine, Store};

const ZERO: SlotBinding = SlotBinding::Const(0);

fn slots(entries: &[(usize, SlotBinding)]) -> [SlotBinding; COMM_CHAIN_EVENT_ARGS] {
    let mut out = [ZERO; COMM_CHAIN_EVENT_ARGS];
    for &(idx, source) in entries {
        out[idx] = source;
    }
    out
}

struct TracedTestComponent {
    store: Store<neo_wasm::WasmtimeTraceRegistry>,
    instance: Instance,
}

struct CollectedTestTrace {
    steps: Vec<neo_wasm::WasmtimeTraceStep>,
    artifacts: neo_wasm::WasmProgramArtifacts,
}

impl TracedTestComponent {
    fn new(component_bytes: &[u8], bindings: &HostEventBindings) -> Self {
        let mut config = Config::new();
        config.guest_debug(true);
        config.wasm_reference_types(true);
        config.wasm_function_references(true);
        config.wasm_component_model(true);
        let engine = Engine::new(&config).expect("engine");
        let component = Component::new(&engine, component_bytes).expect("component");
        let mut registry = neo_wasm::WasmtimeTraceRegistry::default();
        for payload in wasmparser::Parser::new(0).parse_all(component_bytes) {
            if let wasmparser::Payload::ModuleSection { unchecked_range, .. } = payload.unwrap() {
                registry
                    .register_module(&component_bytes[unchecked_range], bindings.clone())
                    .unwrap();
            }
        }
        let mut store = Store::new(&engine, registry);
        store.set_debug_handler(neo_wasm::WasmtimeTraceHandler::new());
        store
            .edit_breakpoints()
            .expect("guest debug enabled")
            .single_step(true)
            .expect("single-step debugging");
        let linker = Linker::new(&engine);
        let instance = futures::executor::block_on(linker.instantiate_async(&mut store, &component))
            .expect("instantiate component");
        Self { store, instance }
    }

    fn call(&mut self, export: &str, args: &[ComponentVal], results: &mut [ComponentVal]) {
        let func = self
            .instance
            .get_func(&mut self.store, export)
            .unwrap_or_else(|| panic!("component export '{export}'"));
        futures::executor::block_on(func.call_async(&mut self.store, args, results))
            .unwrap_or_else(|error| panic!("call component export '{export}': {error}"));
    }

    fn finish(self) -> CollectedTestTrace {
        let captured = self.store.data().single_instance().unwrap();
        CollectedTestTrace {
            steps: captured.steps().to_vec(),
            artifacts: captured.artifacts().clone(),
        }
    }
}

fn run_counter_turns(component_bytes: &[u8], bindings: &HostEventBindings) -> CollectedTestTrace {
    let mut runtime = TracedTestComponent::new(component_bytes, bindings);
    let mut first = [ComponentVal::S32(0)];
    runtime.call("add", &[ComponentVal::S32(7)], &mut first);
    let mut second = [ComponentVal::S32(0)];
    runtime.call("add", &[ComponentVal::S32(35)], &mut second);
    assert_eq!((first, second), ([ComponentVal::S32(7)], [ComponentVal::S32(42)]));
    runtime.finish()
}

/// Stateful export used to test cross-turn global persistence.
fn counter_component_wat() -> &'static str {
    r#"
    (component
      (type $add-type (func (param "x" s32) (result s32)))
      (core module $m
        (global $acc (mut i32) (i32.const 0))
        (func (export "add") (param i32) (result i32)
          global.get $acc
          local.get 0
          i32.add
          global.set $acc
          global.get $acc))
      (core instance $i (instantiate $m))
      (alias core export $i "add" (core func $add))
      (func (export "add") (type $add-type)
        (canon lift (core func $add))))
    "#
}

fn zero_local_component_wat() -> &'static str {
    r#"
    (component
      (type $tick-type (func (result s32)))
      (core module $m
        (global $counter (mut i32) (i32.const 0))
        (func (export "tick") (result i32)
          global.get $counter
          i32.const 1
          i32.add
          global.set $counter
          global.get $counter))
      (core instance $i (instantiate $m))
      (alias core export $i "tick" (core func $tick))
      (func (export "tick") (type $tick-type)
        (canon lift (core func $tick))))
    "#
}

/// Entry initializes local 0; exit absorbs the captured output.
fn add_template() -> ExportTemplate {
    ExportTemplate {
        entry: vec![EventBlock::op(
            8,
            slots(&[(
                1,
                SlotBinding::InputLocal {
                    input: 0,
                    local: 0,
                    limb: Limb::Lo,
                },
            )]),
        )],
        exit: vec![EventBlock::op(
            17,
            slots(&[(0, SlotBinding::OutputElem { limb: Limb::Lo })]),
        )],
        entry_input_count: 1,
    }
}

fn turn_inputs() -> [Vec<u64>; 2] {
    [vec![7], vec![35]]
}

struct MultiTurnSetup {
    trace: Vec<WasmVmStep>,
    bindings: HostEventBindings,
    add_fref: u32,
    component_bytes: Vec<u8>,
}

fn multi_turn_setup() -> MultiTurnSetup {
    let component_bytes = wat::parse_str(counter_component_wat()).expect("component wat");
    let add_fref = 1;
    let mut bindings = HostEventBindings::default();
    bindings.exports.insert(add_fref, add_template());
    let run = run_counter_turns(&component_bytes, &bindings);

    let without_bindings = neo_wasm::traces_from_wasmtime_steps(&run.steps);
    assert!(
        without_bindings.is_err(),
        "multi-turn traces require explicit export bindings"
    );

    let mut missing_bindings = run.artifacts.clone();
    missing_bindings.host_event_bindings = HostEventBindings::default();
    assert!(
        neo_wasm::traces_from_wasmtime_steps_with_host_events(&run.steps, &missing_bindings, Default::default(),)
            .is_err(),
        "missing export template must be rejected"
    );

    let trace = neo_wasm::traces_from_wasmtime_steps_with_host_events(&run.steps, &run.artifacts, Default::default())
        .expect("multi-turn bindings trace");
    common::check_native_event_hashes(&trace).expect("native event hashes");
    common::ccs_check_trace(&trace);

    MultiTurnSetup {
        trace,
        bindings,
        add_fref,
        component_bytes,
    }
}

fn expected_transcript(
    bindings: &HostEventBindings,
    add_fref: u32,
    turns: &[Vec<u64>],
    outputs: &[u32],
) -> Vec<[p3_goldilocks::Goldilocks; 8]> {
    let template = bindings.exports.get(&add_fref).expect("template");
    let mut blocks = Vec::new();
    for (turn, &output) in turns.iter().zip(outputs) {
        let entry = neo_wasm::host_event_bindings::expand_export_entry(template, turn).expect("entry");
        blocks.extend(absorbed_blocks(&template.entry, &entry).expect("absorbed entry"));
        let exit = neo_wasm::host_event_bindings::expand_export_exit(template, Some((output, 0)), &[]).expect("exit");
        blocks.extend(absorbed_blocks(&template.exit, &exit).expect("absorbed exit"));
    }
    blocks
        .into_iter()
        .map(|block| block.map(p3_goldilocks::Goldilocks::from_u64))
        .collect()
}

#[test]
fn multi_turn_rejects_an_empty_reentry_template() {
    let component_bytes = wat::parse_str(zero_local_component_wat()).expect("component wat");

    let fref = 1;
    let mut bindings = HostEventBindings::default();
    bindings.exports.insert(fref, ExportTemplate::default());
    let mut runtime = TracedTestComponent::new(&component_bytes, &bindings);
    let mut first = [ComponentVal::S32(0)];
    runtime.call("tick", &[], &mut first);
    let mut second = [ComponentVal::S32(0)];
    runtime.call("tick", &[], &mut second);
    let run = runtime.finish();
    let error = neo_wasm::traces_from_wasmtime_steps_with_host_events(&run.steps, &run.artifacts, Default::default())
        .expect_err("re-entry without any events must be rejected");
    assert!(error
        .to_string()
        .contains("requires at least one entry or exit event"));
}

#[test]
fn exit_only_template_allows_reentry_and_commits_each_return() {
    let component_bytes = wat::parse_str(zero_local_component_wat()).unwrap();

    let fref = 1;
    let mut bindings = HostEventBindings::default();
    bindings.exports.insert(
        fref,
        ExportTemplate {
            entry: vec![],
            exit: vec![EventBlock::op(
                17,
                slots(&[(0, SlotBinding::OutputElem { limb: Limb::Lo })]),
            )],
            entry_input_count: 0,
        },
    );
    let mut runtime = TracedTestComponent::new(&component_bytes, &bindings);
    for expected in [1, 2] {
        let mut result = [ComponentVal::S32(0)];
        runtime.call("tick", &[], &mut result);
        assert_eq!(result, [ComponentVal::S32(expected)]);
    }
    let run = runtime.finish();
    let trace =
        neo_wasm::traces_from_wasmtime_steps_with_host_events(&run.steps, &run.artifacts, Default::default()).unwrap();
    let artifacts = neo_wasm::extract_first_component_core_program_artifacts(&component_bytes).unwrap();
    let witnesses = common::sanity_check_trace_with_bindings(&trace, &artifacts, &bindings);
    common::ccs_check_trace(&trace);
    let events = neo_wasm::comm_chain::absorbed_event_blocks(&trace);
    assert_eq!(
        events.iter().map(|event| event.words).collect::<Vec<_>>(),
        vec![[17, 1, 0, 0, 0, 0, 0, 0], [17, 2, 0, 0, 0, 0, 0, 0]]
    );
    assert!(trace.last().unwrap().state_after.is_terminal());

    let boundary = trace
        .iter()
        .position(|row| row.row_kind.is_turn_boundary())
        .unwrap();
    assert_eq!(trace[boundary].host_event_initial_schedule_count, Some(1));
    assert_eq!(trace[boundary].host_event_exit_schedule_count, Some(1));
    let mut forged = witnesses[boundary].clone();
    forged[neo_wasm::layout::COL_HOST_EVENT_EXIT_SCHEDULE_COUNT] = neo_math::F::ZERO;
    common::assert_rejected(&forged, "empty entry and exit cannot re-enter");

    // Changing the count and its inverse together must still fail the ROM.
    let mut forged_rows = witnesses;
    forged_rows[boundary][neo_wasm::layout::COL_HOST_EVENT_EXIT_SCHEDULE_COUNT] = neo_math::F::from_u64(2);
    common::assert_satisfied(&forged_rows[boundary], "row-local alternative nonempty count");
    let mut preload = neo_wasm::memory_semantics::preload_from_program_artifacts(&artifacts);
    neo_wasm::memory_semantics::preload_host_event_tables(&mut preload, &bindings);
    assert!(
        neo_wasm::memory_semantics::sanity_check_memory_rows(
            &neo_wasm::build_wasm_relation_layout(),
            &forged_rows,
            &preload,
        )
        .is_err(),
        "exit count at re-entry is verifier-bound"
    );

    let halt = trace
        .iter()
        .find(|row| !row.state_before.halted && row.state_after.halted)
        .unwrap();
    assert!(
        !halt.state_after.is_terminal(),
        "halt must still spend the exit schedule"
    );
    let mut forged = build_witness_vector(halt);
    forged[neo_wasm::layout::COL_HOST_EVENTS_REMAINING_AFTER] = neo_math::F::ZERO;
    common::assert_rejected(&forged, "halt cannot skip its return event");
}

#[test]
fn export_advice_preserves_arguments_without_absorbing_them() {
    use neo_wasm::host_event_bindings::EventSequenceBuilder;
    let component_bytes = wat::parse_str(
        r#"
        (component
          (type $run-type (func (param "x" s32) (result s32)))
          (core module $m
            (func (export "run") (param i32) (result i32)
              local.get 0))
          (core instance $i (instantiate $m))
          (alias core export $i "run" (core func $run))
          (func (export "run") (type $run-type) (canon lift (core func $run))))
    "#,
    )
    .unwrap();

    let fref = 1;
    let template = ExportTemplate {
        entry: EventSequenceBuilder::advice()
            .input_local_i32(0, 0)
            .unwrap()
            .finish()
            .unwrap(),
        exit: EventSequenceBuilder::op(17)
            .output_i32()
            .unwrap()
            .finish()
            .unwrap(),
        entry_input_count: 1,
    };
    let mut bindings = HostEventBindings::default();
    bindings.exports.insert(fref, template);
    let mut runtime = TracedTestComponent::new(&component_bytes, &bindings);
    for x in [7, 35] {
        let mut result = [ComponentVal::S32(0)];
        runtime.call("run", &[ComponentVal::S32(x)], &mut result);
        assert_eq!(result, [ComponentVal::S32(x)]);
    }
    let run = runtime.finish();
    let trace =
        neo_wasm::traces_from_wasmtime_steps_with_host_events(&run.steps, &run.artifacts, Default::default()).unwrap();
    let artifacts = neo_wasm::extract_first_component_core_program_artifacts(&component_bytes).unwrap();
    common::sanity_check_trace_with_bindings(&trace, &artifacts, &bindings);
    common::ccs_check_trace(&trace);
    let events = neo_wasm::comm_chain::absorbed_event_blocks(&trace);
    assert_eq!(
        events.iter().map(|event| event.words).collect::<Vec<_>>(),
        vec![[17, 7, 0, 0, 0, 0, 0, 0], [17, 35, 0, 0, 0, 0, 0, 0]]
    );
    // Advice policy is part of the verifier's ROM even on non-final slots,
    // where changing it alone does not violate row-local constraints.
    let input = trace
        .iter()
        .find(|row| {
            row.host_event_rom_slot
                .is_some_and(|rom| rom.kind == neo_wasm::WasmHostEventSlotKind::InputLocal)
        })
        .unwrap();
    let mut forged = input.clone();
    forged.host_event_rom_slot.as_mut().unwrap().advice = false;
    common::assert_satisfied(&build_witness_vector(&forged), "non-final slot before ROM checking");
    let mut forged_rows: Vec<_> = trace.iter().map(build_witness_vector).collect();
    forged_rows[input.cycle as usize] = build_witness_vector(&forged);
    let mut preload = neo_wasm::memory_semantics::preload_from_program_artifacts(&artifacts);
    neo_wasm::memory_semantics::preload_host_event_tables(&mut preload, &bindings);
    assert!(
        neo_wasm::sanity_check_memory_rows(&neo_wasm::build_wasm_relation_layout(), &forged_rows, &preload,).is_err(),
        "export advice flag is verifier-bound"
    );
    let last_advice = trace
        .iter()
        .find(|row| {
            row.host_event_rom_slot.is_some_and(|rom| rom.advice) && row.state_before.host_events.slot_cursor == 7
        })
        .unwrap();
    let mut forged = build_witness_vector(last_advice);
    forged[neo_wasm::layout::COL_PERM_PENDING_AFTER] = neo_math::F::ONE;
    common::assert_rejected(&forged, "export advice cannot start absorption");
}

#[test]
fn turn_boundary_row_bridges_the_turns() {
    let setup = multi_turn_setup();
    let boundaries: Vec<&WasmVmStep> = setup
        .trace
        .iter()
        .filter(|row| row.row_kind.is_turn_boundary())
        .collect();
    assert_eq!(boundaries.len(), 1, "two turns, one boundary");
    let tb = boundaries[0];

    // Completed turn.
    assert_eq!(tb.state_before.sp, 0);
    assert!(tb.state_before.output.enabled);
    assert_eq!(tb.state_before.output.value_lo, 7);
    assert_eq!(tb.state_before.host_events.events_remaining, 0);
    assert_eq!(tb.state_before.call_stack_depth, 0);

    // Fresh turn.
    assert_eq!(tb.state_after.sp, 0);
    assert!(!tb.state_after.output.enabled);
    assert_eq!(tb.state_after.host_events.events_remaining, 1);
    assert_eq!(tb.state_after.host_events.event_index, 0);
    assert_eq!(tb.state_after.host_callee_fref, setup.add_fref);
    assert_ne!(
        tb.state_before.pc, tb.state_after.pc,
        "the boundary bridges the pc jump"
    );

    // Cross-turn state.
    assert_eq!(tb.state_before.comm_chain, tb.state_after.comm_chain);
    assert_eq!(tb.state_before.event_absorb, tb.state_after.event_absorb);

    // Entry events drain before program execution resumes.
    let tb_idx = setup
        .trace
        .iter()
        .position(|row| row.row_kind.is_turn_boundary())
        .expect("boundary");
    let next_program = setup.trace[tb_idx + 1..]
        .iter()
        .find(|row| row.row_kind.is_program())
        .expect("turn-2 program row");
    assert_eq!(next_program.state_before.host_events.events_remaining, 0);
    assert_eq!(next_program.state_before.pc, tb.state_after.pc);
}

#[test]
fn multi_turn_proof_binds_both_turns_inputs() {
    let setup = multi_turn_setup();
    let artifacts =
        neo_wasm::extract_first_component_core_program_artifacts(&setup.component_bytes).expect("artifacts");
    let entry_pc = common::entry_pc_for_function_ref(&artifacts, u64::from(setup.add_fref));
    let digest = host_event_top_level_initial_state_digest(
        &artifacts.tables,
        entry_pc,
        &setup.bindings,
        setup.add_fref,
        Default::default(),
    )
    .expect("bindings anchor");
    assert_eq!(
        digest,
        neo_wasm::semantic_state_digest(setup.trace[0].state_before),
        "verifier initial state must match the trace's first before-state"
    );

    let batch_size = 8;
    let prep = preprocess_seeded_batched(batch_size, digest).expect("prep");
    let proof = prove_batched(&prep, &setup.trace, batch_size).expect("prove");
    let final_state = common::final_state(&setup.trace);
    assert_eq!((final_state.output.value_lo, final_state.output.value_hi), (42, 0));

    let transcript = expected_transcript(&setup.bindings, setup.add_fref, &turn_inputs(), &[7, 42]);
    verify_with_transcript(&prep, &proof, final_state, Default::default(), &transcript)
        .expect("verify with the two-turn transcript");

    let mut wrong_turns = turn_inputs();
    wrong_turns[1][0] = 34;
    let wrong = expected_transcript(&setup.bindings, setup.add_fref, &wrong_turns, &[7, 42]);
    assert!(
        matches!(
            verify_with_transcript(&prep, &proof, final_state, Default::default(), &wrong),
            Err(AuditProveError::TranscriptMismatch)
        ),
        "a transcript claiming a different turn-2 input must be rejected"
    );
}

#[test]
fn memory_model_carries_state_across_turns() {
    let setup = multi_turn_setup();
    let artifacts =
        neo_wasm::extract_first_component_core_program_artifacts(&setup.component_bytes).expect("artifacts");
    let mut preload = neo_wasm::memory_semantics::preload_from_program_artifacts(&artifacts);
    neo_wasm::memory_semantics::preload_host_event_tables(&mut preload, &setup.bindings);
    let witness_rows: Vec<Vec<neo_math::F>> = setup.trace.iter().map(build_witness_vector).collect();
    let layout = neo_wasm::relation_layout::build_wasm_relation_layout();
    neo_wasm::memory_semantics::sanity_check_memory_rows(&layout, &witness_rows, &preload)
        .expect("locals overwrite + global persistence check out");
}

#[test]
fn ccs_rejects_forged_turn_boundary() {
    let setup = multi_turn_setup();
    let tb = setup
        .trace
        .iter()
        .find(|row| row.row_kind.is_turn_boundary())
        .expect("boundary row");

    let witness = build_witness_vector(tb);
    common::assert_satisfied(&witness, "untampered turn boundary");

    let mut forged = witness.clone();
    forged[neo_wasm::layout::COL_OUTPUT_ENABLED_AFTER] = neo_math::F::ONE;
    common::assert_rejected(&forged, "boundary keeping the previous turn's output armed");

    let mut forged = witness.clone();
    forged[neo_wasm::layout::COL_HOST_EVENTS_REMAINING_AFTER] = neo_math::F::ZERO;
    common::assert_rejected(&forged, "boundary skipping the next turn's entry schedule");

    // Silent re-entry: both schedules empty would re-run the export without
    // moving the transcript. The nonempty-template guard has no inverse.
    let mut forged = witness.clone();
    forged[neo_wasm::layout::COL_HOST_EVENT_INITIAL_SCHEDULE_COUNT] = neo_math::F::ONE;
    forged[neo_wasm::layout::COL_HOST_EVENT_EXIT_SCHEDULE_COUNT] = neo_math::F::ZERO;
    forged[neo_wasm::layout::COL_HOST_EVENTS_REMAINING_AFTER] = neo_math::F::ZERO;
    common::assert_rejected(&forged, "boundary re-entering through an entirely empty template");

    let mut forged = witness.clone();
    forged[neo_wasm::layout::COL_SP_BEFORE] = neo_math::F::ONE;
    forged[neo_wasm::layout::COL_SP_AFTER] = neo_math::F::ONE;
    common::assert_rejected(&forged, "boundary firing with a live operand stack");

    let mut forged = witness.clone();
    forged[neo_wasm::layout::COL_TURN_EXPORT_FREF_AFTER] += neo_math::F::ONE;
    common::assert_rejected(&forged, "boundary carrying a different export from its target");

    // A boundary can't fire while the previous turn still owes events.
    let mut forged = witness.clone();
    forged[neo_wasm::layout::COL_HOST_EVENTS_REMAINING_BEFORE] = neo_math::F::ONE;
    common::assert_rejected(&forged, "boundary before the previous schedule is spent");

    // Presence binding: a boundary pointed at a fref with no export
    // template reads the zero-filled count cell (the memory model pins the
    // claim; see memory_model_rejects_boundary_into_undeclared_fref), and
    // under the biased load no normal schedule satisfies the row.
    let undeclared_fref = neo_math::F::from_u64(u64::from(setup.add_fref) + 7);
    let mut forged = witness.clone();
    forged[neo_wasm::layout::COL_HOST_CALLEE_FREF_AFTER] = undeclared_fref;
    forged[neo_wasm::layout::COL_TURN_EXPORT_FREF_AFTER] = undeclared_fref;
    forged[neo_wasm::layout::COL_HOST_EVENT_INITIAL_SCHEDULE_COUNT] = neo_math::F::ZERO;
    forged[neo_wasm::layout::COL_HOST_EVENT_EXIT_SCHEDULE_COUNT] = neo_math::F::ZERO;
    common::assert_rejected(&forged, "boundary entering an undeclared fref with a normal schedule");

    // The only row-locally satisfiable assignment loads the poisoned
    // schedule events_remaining = -1 = p-1 ...
    let mut poisoned = forged.clone();
    poisoned[neo_wasm::layout::COL_HOST_EVENTS_REMAINING_AFTER] = -neo_math::F::ONE;
    common::assert_satisfied(&poisoned, "undeclared boundary target loads the poisoned schedule");

    // ... which the composed circuit's host-event ROM address bound prevents
    // from draining before another program row can run.
    let program_row = setup
        .trace
        .iter()
        .find(|row| row.row_kind.is_program())
        .expect("program row");
    let mut wedged = build_witness_vector(program_row);
    wedged[neo_wasm::layout::COL_HOST_EVENTS_REMAINING_BEFORE] = -neo_math::F::ONE;
    common::assert_rejected(&wedged, "program row under the poisoned schedule");
}

/// The claim side of the presence binding: an undeclared boundary target
/// cannot fake a declared export's biased count cell — the export
/// entry-count family has no cell for it, so the ROM read mismatches.
#[test]
fn memory_model_rejects_boundary_into_undeclared_fref() {
    let setup = multi_turn_setup();
    let artifacts =
        neo_wasm::extract_first_component_core_program_artifacts(&setup.component_bytes).expect("artifacts");
    let mut preload = neo_wasm::memory_semantics::preload_from_program_artifacts(&artifacts);
    neo_wasm::memory_semantics::preload_host_event_tables(&mut preload, &setup.bindings);
    let layout = neo_wasm::relation_layout::build_wasm_relation_layout();

    let tb_index = setup
        .trace
        .iter()
        .position(|row| row.row_kind.is_turn_boundary())
        .expect("boundary row");
    let mut witness_rows: Vec<Vec<neo_math::F>> = setup.trace.iter().map(build_witness_vector).collect();
    neo_wasm::memory_semantics::sanity_check_memory_rows(&layout, &witness_rows, &preload)
        .expect("the honest trace must pass");
    witness_rows[tb_index][neo_wasm::layout::COL_HOST_CALLEE_FREF_AFTER] =
        neo_math::F::from_u64(u64::from(setup.add_fref) + 7);
    assert!(
        neo_wasm::memory_semantics::sanity_check_memory_rows(&layout, &witness_rows, &preload).is_err(),
        "an undeclared boundary target must not read a declared export's count cell"
    );
}

#[test]
fn ccs_rejects_execution_after_halt() {
    let setup = multi_turn_setup();

    // A program row claiming the turn already finished is rejected.
    let program_row = setup
        .trace
        .iter()
        .find(|row| row.row_kind.is_program())
        .expect("program row");
    let mut witness = build_witness_vector(program_row);
    common::assert_satisfied(&witness, "untampered program row");
    witness[neo_wasm::layout::COL_HALTED_BEFORE] = neo_math::F::ONE;
    witness[neo_wasm::layout::COL_HALTED] = neo_math::F::ONE;
    common::assert_rejected(&witness, "program row executing after a halt");

    // The halting row cannot suppress the latch (with or without capture).
    let halting_row = setup
        .trace
        .iter()
        .find(|row| row.row_kind.is_program() && row.state_after.halted)
        .expect("halting row");
    let mut witness = build_witness_vector(halting_row);
    common::assert_satisfied(&witness, "untampered halting row");
    witness[neo_wasm::layout::COL_HALTED] = neo_math::F::ZERO;
    common::assert_rejected(&witness, "halting row pretending the turn is not done");
}

#[test]
fn resultless_turn_can_precede_another_turn() {
    let component_bytes = wat::parse_str(
        r#"
        (component
          (type $poke-type (func (param "x" s32)))
          (type $read-type (func (result s32)))
          (core module $m
            (global $acc (mut i32) (i32.const 0))
            (func (export "poke") (param i32)
              local.get 0
              global.set $acc)
            (func (export "read") (result i32)
              global.get $acc))
          (core instance $i (instantiate $m))
          (alias core export $i "poke" (core func $poke))
          (alias core export $i "read" (core func $read))
          (func (export "poke") (type $poke-type)
            (canon lift (core func $poke)))
          (func (export "read") (type $read-type)
            (canon lift (core func $read))))
        "#,
    )
    .expect("component wat");

    let poke_fref = 1;
    let read_fref = 2;

    let mut bindings = HostEventBindings::default();
    bindings.exports.insert(
        poke_fref,
        ExportTemplate {
            entry: vec![EventBlock::op(
                8,
                slots(&[(
                    0,
                    SlotBinding::InputLocal {
                        input: 0,
                        local: 0,
                        limb: Limb::Lo,
                    },
                )]),
            )],
            exit: vec![EventBlock::op(16, slots(&[]))],
            entry_input_count: 1,
        },
    );
    bindings.exports.insert(
        read_fref,
        ExportTemplate {
            entry: vec![EventBlock::op(9, slots(&[]))],
            exit: vec![EventBlock::op(
                17,
                slots(&[(0, SlotBinding::OutputElem { limb: Limb::Lo })]),
            )],
            entry_input_count: 0,
        },
    );
    let mut runtime = TracedTestComponent::new(&component_bytes, &bindings);
    runtime.call("poke", &[ComponentVal::S32(41)], &mut []);
    let mut read_result = [ComponentVal::S32(0)];
    runtime.call("read", &[], &mut read_result);
    assert_eq!(read_result, [ComponentVal::S32(41)]);
    let mut run = runtime.finish();
    let trace = neo_wasm::traces_from_wasmtime_steps_with_host_events(&run.steps, &run.artifacts, Default::default())
        .expect("resultless-then-value trace");
    common::check_native_event_hashes(&trace).expect("native event hashes");
    common::ccs_check_trace(&trace);

    let tb = trace
        .iter()
        .find(|row| row.row_kind.is_turn_boundary())
        .expect("boundary row");
    assert!(!tb.state_before.output.enabled);
    assert!(!tb.state_after.output.enabled);
    assert!(tb.state_before.halted);
    assert!(!tb.state_after.halted);
    assert_eq!(tb.state_before.host_events.turn_export_fref, poke_fref);
    assert_eq!(tb.state_after.host_events.turn_export_fref, read_fref);

    let event_metadata: Vec<_> = neo_wasm::comm_chain::absorbed_event_blocks(&trace)
        .into_iter()
        .map(|event| (event.metadata.attributed_fref, event.metadata.turn_export_fref))
        .collect();
    assert_eq!(
        event_metadata,
        [
            (poke_fref, poke_fref),
            (poke_fref, poke_fref),
            (read_fref, read_fref),
            (read_fref, read_fref),
        ]
    );

    let mut blocks =
        neo_wasm::host_event_bindings::expand_export_entry(&bindings.exports[&poke_fref], &[41]).expect("poke entry");
    blocks = absorbed_blocks(&bindings.exports[&poke_fref].entry, &blocks).expect("absorbed poke entry");
    blocks.extend(
        absorbed_blocks(
            &bindings.exports[&poke_fref].exit,
            &neo_wasm::host_event_bindings::expand_export_exit(&bindings.exports[&poke_fref], None, &[])
                .expect("resultless poke exit"),
        )
        .expect("absorbed poke exit"),
    );
    blocks.extend(
        absorbed_blocks(
            &bindings.exports[&read_fref].entry,
            &neo_wasm::host_event_bindings::expand_export_entry(&bindings.exports[&read_fref], &[])
                .expect("read entry"),
        )
        .expect("absorbed read entry"),
    );
    blocks.extend(
        absorbed_blocks(
            &bindings.exports[&read_fref].exit,
            &neo_wasm::host_event_bindings::expand_export_exit(&bindings.exports[&read_fref], Some((41, 0)), &[])
                .expect("exit"),
        )
        .expect("absorbed read exit"),
    );
    let lifted: Vec<[p3_goldilocks::Goldilocks; 8]> = blocks
        .into_iter()
        .map(|block| block.map(p3_goldilocks::Goldilocks::from_u64))
        .collect();
    let final_state = common::final_state(&trace);
    assert_eq!((final_state.output.value_lo, final_state.output.enabled), (41, true));
    assert_eq!(
        final_state.comm_chain,
        neo_wasm::comm_chain::fold_event_blocks(Default::default(), &lifted).canonical_u64()
    );

    // Resultless exits may not reference a captured output.
    let mut bad_bindings = bindings.clone();
    bad_bindings.exports.get_mut(&poke_fref).expect("poke").exit = vec![EventBlock::op(
        17,
        slots(&[(0, SlotBinding::OutputElem { limb: Limb::Lo })]),
    )];
    run.artifacts.host_event_bindings = bad_bindings.clone();
    assert!(
        neo_wasm::traces_from_wasmtime_steps_with_host_events(&run.steps, &run.artifacts, Default::default()).is_err(),
        "output-dependent exit events on a resultless turn must be rejected"
    );
}
