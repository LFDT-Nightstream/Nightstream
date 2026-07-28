mod common;

use neo_wasm::event_grammar::HostEventGrammar;
use neo_wasm::{
    collect_wasmtime_steps, traces_from_wasmtime_steps_with_grammar, witness_builder::build_witness_vector,
    CommChainState, WasmAuxOpcode, WasmBuildError, WasmOpcode, WasmRowKind,
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

    let mut forged = tail.clone();
    forged.state_before.grammar_mode = true;
    forged.state_after.grammar_mode = true;
    common::assert_rejected(&build_witness_vector(&forged), "grammar-mode tail call");
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
fn event_grammar_rejects_tail_calls_until_attribution_is_supported() {
    let wasm = wat::parse_str(
        r#"(module
            (func $identity (param i32) (result i32)
                local.get 0)
            (func (export "main") (result i32)
                i32.const 7
                return_call $identity))"#,
    )
    .expect("valid WAT");
    let run = collect_wasmtime_steps(&wasm, "main", &[]).expect("wasmtime trace");
    let err = traces_from_wasmtime_steps_with_grammar(
        &run.steps,
        &HostEventGrammar::default(),
        &[],
        CommChainState::default(),
    )
    .expect_err("grammar tail calls must fail explicitly");
    assert!(matches!(err, WasmBuildError::Unsupported(_)));
    assert!(err
        .to_string()
        .contains("tail calls in event-grammar traces"));
}
