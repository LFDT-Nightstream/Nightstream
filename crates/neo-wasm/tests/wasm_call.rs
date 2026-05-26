mod common;

use neo_wasm::{
    build_wasm_lookup_binding_layout, collect_wasmtime_steps, preload_from_wasmtime_run, sanity_check_memory_rows,
    traces_from_wasmtime_steps, WasmAuxOpcode, WasmOpcode, WasmRowKind, WasmStepTrace,
};

fn add_one_wasm() -> Vec<u8> {
    wat::parse_str(
        r#"(module
            (func $add_one (param i32) (result i32)
                local.get 0
                i32.const 1
                i32.add)
            (func (export "run") (result i32)
                i32.const 5
                call $add_one)
        )"#,
    )
    .expect("wat parse")
}

fn nested_two_param_wasm() -> Vec<u8> {
    wat::parse_str(
        r#"(module
            (func $sum2 (param i32 i32) (result i32)
                local.get 0
                local.get 1
                i32.add)
            (func $call_sum2 (param i32 i32) (result i32)
                local.get 0
                local.get 1
                call $sum2)
            (func (export "run") (result i32)
                i32.const 4
                i32.const 7
                call $call_sum2)
        )"#,
    )
    .expect("wat parse")
}

fn call_indirect_param_wasm() -> Vec<u8> {
    wat::parse_str(
        r#"(module
            (type $t (func (param i32) (result i32)))
            (func $add_one (type $t)
                local.get 0
                i32.const 1
                i32.add)
            (table 1 funcref)
            (elem (i32.const 0) func $add_one)
            (func (export "run") (result i32)
                i32.const 5
                i32.const 0
                call_indirect (type $t))
        )"#,
    )
    .expect("wat parse")
}

fn build_witnesses(trace: &[WasmStepTrace]) -> Vec<Vec<neo_math::F>> {
    trace
        .iter()
        .map(neo_wasm::witness_builder::build_witness_vector)
        .collect()
}

#[test]
fn call_trace_has_correct_fbp_and_call_stack_fields() {
    let wasm = add_one_wasm();
    let run = collect_wasmtime_steps(&wasm, "run", &[]).expect("trace");
    assert_eq!(run.results.as_slice(), &["6".to_string()], "expected add_one(5) = 6");

    let trace = traces_from_wasmtime_steps(&run.steps).expect("normalize");
    let opcodes: Vec<_> = trace.iter().map(|r| r.opcode).collect();
    assert!(
        opcodes.contains(&WasmOpcode::Call),
        "expected Call in normalized trace: {opcodes:?}"
    );

    // Exactly one call step, with call_stack_push populated.
    let call_steps: Vec<_> = trace
        .iter()
        .filter(|r| r.opcode == WasmOpcode::Call)
        .collect();
    assert_eq!(call_steps.len(), 1, "expected one call step");
    let call_step = call_steps[0];
    assert!(
        call_step.call_stack_push.is_some(),
        "call step must have call_stack_push"
    );
    assert!(
        call_step.call_stack_pop.is_none(),
        "call step must not have call_stack_pop"
    );

    let aux_param_steps: Vec<_> = trace
        .iter()
        .filter(|r| r.row_kind == WasmRowKind::Aux(WasmAuxOpcode::CallParamInit))
        .collect();
    assert_eq!(aux_param_steps.len(), 1, "expected one call-param init row");
    let aux = aux_param_steps[0];
    assert!(call_step.param_init_after.active, "call must enter param-init mode");
    assert_eq!(call_step.param_init_after.remaining, 1);
    assert!(
        aux.param_init_before.active,
        "aux row must execute inside param-init mode"
    );
    assert_eq!(aux.param_init_before.remaining, 1);
    assert_eq!(aux.param_init_after.remaining, 0);
    assert!(!aux.param_init_after.active, "last aux row must exit param-init mode");
    assert_eq!(aux.stack_read0.expect("aux stack read").value, 5);
    assert_eq!(aux.local_index, Some(0), "callee param at local addr 0");
    assert_eq!(
        aux.locals_fbp, aux.locals_fbp_after,
        "aux row must stay in the callee frame while initializing params"
    );
    assert_eq!(aux.local_write_value, Some(5), "callee param value must be 5");

    // There should be exactly one non-final return-like step (callee return), with
    // call_stack_pop populated. The final function-end return may be represented by End.
    let return_like_steps: Vec<_> = trace
        .iter()
        .filter(|r| matches!(r.opcode, WasmOpcode::Return | WasmOpcode::End))
        .collect();
    assert!(
        return_like_steps.len() >= 2,
        "expected at least callee return-like step and caller function end"
    );

    // The non-final return-like row is the callee's — it has call_stack_pop.
    let non_final_returns: Vec<_> = return_like_steps
        .iter()
        .filter(|r| r.call_stack_pop.is_some())
        .collect();
    assert_eq!(non_final_returns.len(), 1, "expected one non-final return");
    assert!(
        non_final_returns[0].call_stack_pop.is_some(),
        "non-final return must have call_stack_pop"
    );

    // Only the very last step is halted.
    let last = trace.last().expect("non-empty trace");
    assert!(last.halted, "last step must be halted");
    for step in &trace[..trace.len() - 1] {
        assert!(!step.halted, "intermediate step {} must not be halted", step.cycle);
    }
}

#[test]
fn nested_two_param_call_trace_counts_down_param_init_rows() {
    let wasm = nested_two_param_wasm();
    let run = collect_wasmtime_steps(&wasm, "run", &[]).expect("trace");
    assert_eq!(run.results.as_slice(), &["11".to_string()], "expected sum2(4, 7) = 11");

    let trace = traces_from_wasmtime_steps(&run.steps).expect("normalize");
    let aux_rows: Vec<_> = trace
        .iter()
        .filter(|row| row.row_kind == WasmRowKind::Aux(WasmAuxOpcode::CallParamInit))
        .collect();
    assert_eq!(aux_rows.len(), 4, "two 2-param guest calls require four aux rows");

    for chunk in aux_rows.chunks_exact(2) {
        assert_eq!(chunk[0].param_init_before.remaining, 2);
        assert_eq!(chunk[0].param_init_after.remaining, 1);
        assert!(chunk[0].param_init_after.active);
        assert_eq!(chunk[1].param_init_before.remaining, 1);
        assert_eq!(chunk[1].param_init_after.remaining, 0);
        assert!(!chunk[1].param_init_after.active);
        for (param_index, row) in chunk.iter().enumerate() {
            assert_eq!(row.local_index, Some(param_index as u32));
            assert_eq!(
                row.locals_fbp, row.locals_fbp_after,
                "aux row must stay in the callee frame while initializing params"
            );
        }
    }

    assert!(
        aux_rows.iter().any(|row| row.locals_fbp > 0),
        "nested call should exercise a non-zero callee frame base"
    );
}

#[test]
fn call_indirect_guest_target_initializes_params() {
    let wasm = call_indirect_param_wasm();
    let run = collect_wasmtime_steps(&wasm, "run", &[]).expect("trace");
    assert_eq!(
        run.results.as_slice(),
        &["6".to_string()],
        "expected indirect add_one(5) = 6"
    );

    let trace = traces_from_wasmtime_steps(&run.steps).expect("normalize");
    let call = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::CallIndirect)
        .expect("call_indirect row");
    assert!(call.target_function_is_guest);
    assert!(call.call_stack_push.is_some());
    assert_eq!(call.param_init_after.remaining, 1);

    let aux_rows: Vec<_> = trace
        .iter()
        .filter(|row| row.row_kind == WasmRowKind::Aux(WasmAuxOpcode::CallParamInit))
        .collect();
    assert_eq!(aux_rows.len(), 1);
    assert_eq!(aux_rows[0].stack_read0.expect("param read").value, 5);
    assert_eq!(aux_rows[0].local_write_value, Some(5));
}

#[test]
fn call_trace_passes_witness_checks() {
    let wasm = add_one_wasm();
    let run = collect_wasmtime_steps(&wasm, "run", &[]).expect("trace");
    let trace = traces_from_wasmtime_steps(&run.steps).expect("normalize");
    common::sanity_check_trace(&trace, &run);
    common::ccs_check_trace(&trace);

    assert!(
        trace.iter().any(|row| row.call_stack_push.is_some()),
        "trace must contain a call_stack_push"
    );
    assert!(
        trace.iter().any(|row| row.call_stack_pop.is_some()),
        "trace must contain a call_stack_pop"
    );
}

#[test]
fn memory_semantics_rejects_missing_param_init_aux_row() {
    let wasm = add_one_wasm();
    let run = collect_wasmtime_steps(&wasm, "run", &[]).expect("trace");
    let mut trace = traces_from_wasmtime_steps(&run.steps).expect("normalize");

    trace.retain(|row| row.row_kind != WasmRowKind::Aux(WasmAuxOpcode::CallParamInit));

    let layout = build_wasm_lookup_binding_layout();
    let witnesses = build_witnesses(&trace);
    let preload = preload_from_wasmtime_run(&run, &run.initial_locals);
    let err = sanity_check_memory_rows(layout, &witnesses, &preload)
        .err()
        .expect("must reject missing param-init aux row");
    assert!(err.contains("param_init_continuity"), "unexpected error: {err}");
}
