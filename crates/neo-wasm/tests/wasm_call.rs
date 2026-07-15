mod common;

use neo_wasm::layout::{COL_OUTPUT_VALUE_LO_AFTER, COL_PC_ROM_CALL_RETURN_CHOICE, COL_STACK_READ0_VALUE_LO};
use neo_wasm::{
    build_wasm_relation_layout, collect_wasmtime_steps, extract_wasm_program_artifacts, preload_from_program_artifacts,
    sanity_check_memory_rows, traces_from_wasmtime_steps, WasmAuxOpcode, WasmOpcode, WasmRowKind, WasmVmStep,
};
use p3_field::PrimeCharacteristicRing;

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

fn build_witnesses(trace: &[WasmVmStep]) -> Vec<Vec<neo_math::F>> {
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
    assert!(
        call_step.state_after.param_init.active,
        "call must enter param-init mode"
    );
    assert_eq!(call_step.state_after.param_init.remaining, 1);
    assert!(
        aux.state_before.param_init.active,
        "aux row must execute inside param-init mode"
    );
    assert_eq!(aux.state_before.param_init.remaining, 1);
    assert_eq!(aux.state_after.param_init.remaining, 0);
    assert!(
        !aux.state_after.param_init.active,
        "last aux row must exit param-init mode"
    );
    assert_eq!(aux.stack_read0.expect("aux stack read").value_lo, 5);
    assert_eq!(aux.local_index, Some(0), "callee param at local addr 0");
    assert_eq!(
        aux.state_before.locals_fbp, aux.state_after.locals_fbp,
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
    assert!(last.state_after.halted, "last step must be halted");
    for step in &trace[..trace.len() - 1] {
        assert!(
            !step.state_after.halted,
            "intermediate step {} must not be halted",
            step.cycle
        );
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
        assert_eq!(chunk[0].state_before.param_init.remaining, 2);
        assert_eq!(chunk[0].state_after.param_init.remaining, 1);
        assert!(chunk[0].state_after.param_init.active);
        assert_eq!(chunk[1].state_before.param_init.remaining, 1);
        assert_eq!(chunk[1].state_after.param_init.remaining, 0);
        assert!(!chunk[1].state_after.param_init.active);
        for (pop_index, row) in chunk.iter().enumerate() {
            // Aux rows pop the operand-stack top, so locals initialize in
            // reverse param order.
            assert_eq!(row.local_index, Some(1 - pop_index as u32));
            assert_eq!(
                row.state_before.sp,
                row.state_after.sp + 1,
                "each aux row pops one arg slot"
            );
            assert_eq!(
                row.state_before.locals_fbp, row.state_after.locals_fbp,
                "aux row must stay in the callee frame while initializing params"
            );
        }
    }

    assert!(
        aux_rows.iter().any(|row| row.state_before.locals_fbp > 0),
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
    assert_eq!(call.state_after.param_init.remaining, 1);

    let aux_rows: Vec<_> = trace
        .iter()
        .filter(|row| row.row_kind == WasmRowKind::Aux(WasmAuxOpcode::CallParamInit))
        .collect();
    assert_eq!(aux_rows.len(), 1);
    assert_eq!(aux_rows[0].stack_read0.expect("param read").value_lo, 5);
    assert_eq!(aux_rows[0].local_write_value, Some(5));
}

#[test]
fn call_trace_passes_witness_checks() {
    let wasm = add_one_wasm();
    let artifacts = extract_wasm_program_artifacts(&wasm).expect("program artifacts");
    let run = collect_wasmtime_steps(&wasm, "run", &[]).expect("trace");
    let trace = traces_from_wasmtime_steps(&run.steps).expect("normalize");
    common::sanity_check_trace(&trace, &artifacts, &run.initial_locals);
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
fn guest_call_with_loop_only_pops_frame_at_function_end() {
    let checked = common::checked_wasm_run(
        r#"(module
            (func $count_to (param $limit i32) (result i32)
                (local $counter i32)
                (loop $again
                    local.get $counter
                    i32.const 1
                    i32.add
                    local.tee $counter
                    local.get $limit
                    i32.lt_u
                    br_if $again)
                local.get $counter)
            (func (export "main") (result i32)
                i32.const 5
                call $count_to))"#,
        "main",
        &[],
    );

    assert_eq!(checked.run.results, ["5"]);
    let nested_ends: Vec<_> = checked
        .trace
        .iter()
        .filter(|row| row.opcode == WasmOpcode::End && row.state_before.call_stack_depth == 1)
        .collect();
    assert!(nested_ends.len() >= 2, "expected loop and function end rows");
    assert!(nested_ends
        .iter()
        .any(|row| row.pc_edge_kind == neo_wasm::WasmPcEdgeKind::Static && row.call_stack_pop.is_none()));
    assert_eq!(
        nested_ends
            .iter()
            .filter(|row| row.pc_edge_kind == neo_wasm::WasmPcEdgeKind::ReturnLike && row.call_stack_pop.is_some())
            .count(),
        1,
        "only the function-ending End may pop the caller frame"
    );
}

#[test]
fn clean_halted_row_requires_empty_call_stack_depth() {
    let wasm = add_one_wasm();
    let run = collect_wasmtime_steps(&wasm, "run", &[]).expect("trace");
    let trace = traces_from_wasmtime_steps(&run.steps).expect("normalize");
    let mut final_row = trace
        .iter()
        .find(|row| row.state_after.halted)
        .expect("halted row")
        .clone();
    final_row.state_before.call_stack_depth = 1;
    final_row.state_after.call_stack_depth = 1;

    let witness = neo_wasm::witness_builder::build_witness_vector(&final_row);
    common::assert_rejected(&witness, "clean halted row with non-empty call stack depth");
}

#[test]
fn nested_trap_may_halt_with_nonempty_call_stack_depth() {
    let wasm = wat::parse_str(
        r#"(module
            (func $divide (param i32 i32) (result i32)
                local.get 0
                local.get 1
                i32.div_u)
            (func (export "run") (result i32)
                i32.const 42
                i32.const 0
                call $divide))"#,
    )
    .expect("wat parse");
    let run = collect_wasmtime_steps(&wasm, "run", &[]).expect("trace nested trap");
    let trace = traces_from_wasmtime_steps(&run.steps).expect("normalize nested trap");
    let trap = trace.last().expect("terminal trap row");

    assert!(trap.state_after.halted);
    assert!(trap.state_after.trapped);
    assert_eq!(trap.state_before.call_stack_depth, 1);
    let witness = neo_wasm::witness_builder::build_witness_vector(trap);
    common::assert_satisfied(&witness, "nested trap with abandoned caller frame");
}

#[test]
fn call_row_pins_return_pc_rom_choice() {
    let wasm = add_one_wasm();
    let run = collect_wasmtime_steps(&wasm, "run", &[]).expect("trace");
    let trace = traces_from_wasmtime_steps(&run.steps).expect("normalize");
    let call_row = trace
        .iter()
        .find(|row| row.call_stack_push.is_some())
        .expect("call row");
    let mut witness = neo_wasm::witness_builder::build_witness_vector(call_row);
    witness[COL_PC_ROM_CALL_RETURN_CHOICE] = neo_math::F::ZERO;
    common::assert_rejected(&witness, "call row with unpinned return pc choice");
}

#[test]
fn final_halt_captures_simple_output() {
    let wasm = add_one_wasm();
    let run = collect_wasmtime_steps(&wasm, "run", &[]).expect("trace");
    let trace = traces_from_wasmtime_steps(&run.steps).expect("normalize");
    let final_row = trace
        .iter()
        .find(|row| row.state_after.halted)
        .expect("halted row");

    assert!(final_row.output_captured, "halted row should capture the result");
    assert!(final_row.state_after.output.enabled, "result carry should be enabled");
    assert_eq!(final_row.state_after.output.value_lo, 6);
    assert_eq!(final_row.state_after.output.value_hi, 0);
}

#[test]
fn final_halt_output_low_is_row_bound() {
    let wasm = add_one_wasm();
    let run = collect_wasmtime_steps(&wasm, "run", &[]).expect("trace");
    let trace = traces_from_wasmtime_steps(&run.steps).expect("normalize");
    let final_row = trace
        .iter()
        .find(|row| row.state_after.halted)
        .expect("halted row");
    let mut witness = neo_wasm::witness_builder::build_witness_vector(final_row);

    witness[COL_OUTPUT_VALUE_LO_AFTER] = neo_math::F::from_u64(7);

    common::assert_rejected(&witness, "halted output with mismatched low limb");
}

#[test]
fn final_halt_output_low_is_stack_memory_bound() {
    let wasm = add_one_wasm();
    let artifacts = extract_wasm_program_artifacts(&wasm).expect("program artifacts");
    let run = collect_wasmtime_steps(&wasm, "run", &[]).expect("trace");
    let trace = traces_from_wasmtime_steps(&run.steps).expect("normalize");
    let layout = build_wasm_relation_layout();
    let mut witnesses = build_witnesses(&trace);
    let final_idx = trace
        .iter()
        .position(|row| row.state_after.halted)
        .expect("halted row");

    witnesses[final_idx][COL_OUTPUT_VALUE_LO_AFTER] = neo_math::F::from_u64(7);
    witnesses[final_idx][COL_STACK_READ0_VALUE_LO] = neo_math::F::from_u64(7);

    let preload = preload_from_program_artifacts(&artifacts, &run.initial_locals);
    let err = sanity_check_memory_rows(layout, &witnesses, &preload)
        .err()
        .expect("must reject output stack memory mismatch");
    assert!(err.contains("memory `stack`"), "unexpected error: {err}");
}
