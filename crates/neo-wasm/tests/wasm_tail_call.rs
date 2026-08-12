mod common;

use neo_wasm::comm_chain::COMM_CHAIN_EVENT_ARGS;
use neo_wasm::event_grammar::{
    ExportTemplate, GrammarEvent, HostEventGrammar, ImportTemplate, Limb, SlotSource, TurnClaims,
};
use neo_wasm::{
    collect_wasmtime_component_run_with_linker, traces_from_wasmtime_steps_with_grammar,
    witness_builder::build_witness_vector, WasmAuxOpcode, WasmBuildError, WasmOpcode, WasmRowKind,
};

#[test]
fn return_call_replaces_the_top_level_frame() {
    let checked = common::checked_main(
        r#"(module
            (func $add_one (param i32) (result i32)
                local.get 0
                i32.const 1
                i32.add)
            (func (export "main") (result i32)
                i32.const 99
                i32.const 41
                return_call $add_one))"#,
    );

    assert_eq!(checked.run.results, ["42"]);
    let tail = checked
        .trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::ReturnCall)
        .expect("return_call row");
    assert!(tail.target_function_is_guest);
    assert!(tail.call_stack_push.is_none());
    assert_eq!(tail.state_before.call_stack_depth, 0);
    assert_eq!(tail.state_after.call_stack_depth, 0);
    assert!(tail.state_after.tail_call_pending);

    let enter = checked
        .trace
        .iter()
        .find(|row| row.row_kind == WasmRowKind::Aux(WasmAuxOpcode::TailEnter))
        .expect("tail-enter row");
    assert_eq!(
        enter.state_before.sp, 1,
        "the unrelated operand remains after the arg pop"
    );
    assert_eq!(enter.state_after.sp, 0, "tail entry discards the replaced frame");
    assert!(!enter.state_after.tail_call_pending);
}

#[test]
fn nested_return_call_inherits_the_original_continuation() {
    let checked = common::checked_main(
        r#"(module
            (func $add_one (param i32) (result i32)
                local.get 0
                i32.const 1
                i32.add)
            (func $tail (param i32) (result i32)
                i32.const 77
                local.get 0
                return_call $add_one)
            (func (export "main") (result i32)
                i32.const 41
                call $tail
                i32.const 1
                i32.add))"#,
    );

    assert_eq!(checked.run.results, ["43"]);
    let tail = checked
        .trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::ReturnCall)
        .expect("return_call row");
    assert_eq!(tail.state_before.call_stack_depth, 1);
    assert_eq!(tail.state_after.call_stack_depth, 1);
    assert!(tail.call_stack_push.is_none());
    assert!(
        checked.trace.iter().any(|row| row.call_stack_pop.is_some()),
        "the leaf return must restore main's continuation"
    );
}

#[test]
fn return_call_indirect_replaces_the_current_frame() {
    let checked = common::checked_main(
        r#"(module
            (type $unary (func (param i32) (result i32)))
            (func $add_one (type $unary)
                local.get 0
                i32.const 1
                i32.add)
            (table 1 funcref)
            (elem (i32.const 0) func $add_one)
            (func (export "main") (result i32)
                i32.const 99
                i32.const 41
                i32.const 0
                return_call_indirect (type $unary)))"#,
    );

    assert_eq!(checked.run.results, ["42"]);
    let tail = checked
        .trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::ReturnCallIndirect)
        .expect("return_call_indirect row");
    assert_eq!(tail.table_index, Some(0));
    assert!(tail.target_function_is_guest);
    assert!(tail.call_stack_push.is_none());
    assert_eq!(tail.state_before.call_stack_depth, tail.state_after.call_stack_depth);
}

#[test]
fn return_call_indirect_trap_does_not_enter_a_frame() {
    let checked = common::checked_main(
        r#"(module
            (type $unary (func (param i32) (result i32)))
            (func $add_one (type $unary)
                local.get 0
                i32.const 1
                i32.add)
            (table 2 funcref)
            (elem (i32.const 0) func $add_one)
            (func (export "main") (result i32)
                i32.const 41
                i32.const 1
                return_call_indirect (type $unary)))"#,
    );

    assert!(checked.run.results.is_empty());
    let last = checked.trace.last().expect("non-empty trace");
    assert_eq!(last.opcode, WasmOpcode::ReturnCallIndirect);
    assert!(last.state_after.trapped);
    assert!(!last.state_after.tail_call_pending);
    assert!(last.call_stack_push.is_none());
}

#[test]
fn grammar_exit_events_remain_attributed_to_the_export_after_a_guest_tail_call() {
    let component = wat::parse_str(
        r#"(component
            (type $touch-type (func (param "x" s32)))
            (type $run-type (func (result s32)))
            (import "host-touch" (func $host-touch (type $touch-type)))
            (core module $m
                (import "" "0" (func $touch (param i32)))
                (func $identity (param i32) (result i32)
                    local.get 0)
                (func (export "run") (result i32)
                    i32.const 9
                    call $touch
                    i32.const 42
                    return_call $identity))
            (core func $lowered-touch (canon lower (func $host-touch)))
            (core instance $host
                (export "0" (func $lowered-touch)))
            (core instance $i
                (instantiate $m
                    (with "" (instance $host))))
            (alias core export $i "run" (core func $run))
            (func (export "run") (type $run-type)
                (canon lift (core func $run))))"#,
    )
    .expect("valid component");
    let run = collect_wasmtime_component_run_with_linker(&component, "run", |linker| {
        linker
            .root()
            .func_wrap("host-touch", |_store, (_x,): (i32,)| Ok(()))
            .map_err(|err| WasmBuildError::Trace(format!("failed to define host-touch: {err}")))
    })
    .expect("component trace");
    let export_fref = run
        .steps
        .iter()
        .find_map(|row| row.current_function_ref)
        .expect("export function ref");
    let import_fref = run
        .steps
        .iter()
        .find(|row| matches!(row.opcode_decoded, Some(WasmOpcode::Call)) && !row.target_function_is_guest)
        .and_then(|row| row.function_ref)
        .expect("import function ref");
    assert_ne!(import_fref, export_fref);

    let mut slots = [SlotSource::Const(0); COMM_CHAIN_EVENT_ARGS];
    slots[0] = SlotSource::OutputElem { limb: Limb::Lo };
    let mut grammar = HostEventGrammar::default();
    grammar
        .imports
        .insert(import_fref, ImportTemplate::default());
    grammar.exports.insert(
        export_fref,
        ExportTemplate {
            exit: vec![GrammarEvent::op(17, slots)],
            ..Default::default()
        },
    );
    let trace = traces_from_wasmtime_steps_with_grammar(
        &run.steps,
        &run.program_tables,
        &grammar,
        &[TurnClaims::default()],
        Default::default(),
    )
    .expect("guest tail call with event binding");
    common::ccs_check_trace(&trace);
    neo_wasm::comm_chain::sanity_check_comm_chain(&trace).expect("chain checker");

    let capture = trace
        .iter()
        .find(|row| row.output_captured)
        .expect("output capture");
    assert_ne!(
        capture.current_function_ref, export_fref,
        "the tail callee halts the turn"
    );
    assert_eq!(capture.state_before.grammar.turn_export_fref, export_fref);
    let exit_rows: Vec<_> = trace
        .iter()
        .filter(|row| row.row_kind.is_host_event_gather())
        .collect();
    assert_eq!(exit_rows.len(), 8);
    assert!(exit_rows.iter().all(|row| {
        row.state_before.host_callee_fref == export_fref && row.state_before.grammar.turn_export_fref == export_fref
    }));
    assert_eq!(
        exit_rows
            .last()
            .expect("exit word 7")
            .state_after
            .event_absorb
            .evbuf,
        [17, 42, 0, 0, 0, 0, 0, 0]
    );

    let artifacts = neo_wasm::extract_first_component_core_program_artifacts(&component).expect("program artifacts");
    let mut preload = neo_wasm::preload_from_program_artifacts(&artifacts, &run.initial_locals);
    neo_wasm::memory_semantics::preload_grammar_tables(&mut preload, &grammar);
    let witnesses: Vec<_> = trace.iter().map(build_witness_vector).collect();
    neo_wasm::sanity_check_memory_rows(neo_wasm::build_wasm_relation_layout(), &witnesses, &preload)
        .expect("memory and grammar ROM bindings");
}

#[test]
fn return_call_to_an_import_remains_explicitly_unsupported() {
    let component = wat::parse_str(
        r#"(component
            (type $identity-type (func (param "x" s32) (result s32)))
            (import "host-identity" (func $host-identity (type $identity-type)))
            (core module $m
                (import "" "0" (func $identity (param i32) (result i32)))
                (func (export "run") (param i32) (result i32)
                    local.get 0
                    return_call $identity))
            (core func $lowered-identity (canon lower (func $host-identity)))
            (core instance $host
                (export "0" (func $lowered-identity)))
            (core instance $i
                (instantiate $m
                    (with "" (instance $host))))
            (alias core export $i "run" (core func $run))
            (func (export "run") (type $identity-type)
                (canon lift (core func $run))))"#,
    )
    .expect("valid component");
    let run = neo_wasm::collect_wasmtime_component_run_with_linker_and_args(
        &component,
        "run",
        &[wasmtime::component::Val::S32(7)],
        |linker| {
            linker
                .root()
                .func_wrap("host-identity", |_store, (x,): (i32,)| Ok((x,)))
                .map_err(|err| WasmBuildError::Trace(format!("failed to define host-identity: {err}")))
        },
    )
    .expect("component trace");
    let err = neo_wasm::traces_from_wasmtime_steps(&run.steps).expect_err("import tail call must fail explicitly");
    assert!(matches!(err, WasmBuildError::Unsupported(_)));
    assert!(err.to_string().contains("return_call to a host import"));
}
