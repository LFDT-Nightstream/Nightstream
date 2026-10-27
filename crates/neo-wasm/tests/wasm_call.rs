mod common;

use neo_wasm::host_event_bindings::HostEventBindings;
use neo_wasm::layout::{COL_OUTPUT_VALUE_LO_AFTER, COL_PC_ROM_CALL_RETURN_CHOICE, COL_STACK_READ_VALUE_LO};
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
fn first_export_locals_start_zero_without_aux_rows() {
    let checked = common::checked_main(
        r#"(module
            (func (export "main") (result i64) (local i32 i64)
                local.get 0 i64.extend_i32_u local.get 1 i64.or))"#,
    );
    assert_eq!(checked.run.results, ["0"]);
    assert!(checked
        .trace
        .iter()
        .all(|row| !row.row_kind.is_local_zero()));
    let initial = checked.trace[0].state_before;
    assert_eq!(initial.sp, 2);
    assert_eq!(initial.stack_frame_base, 2);
    assert_eq!(
        neo_wasm::semantic_state_digest(initial),
        common::verifier_initial_state_digest(&checked.artifacts),
        "the verifier must reserve the first frame without pending zeroing"
    );
}

#[test]
fn call_trace_has_correct_fbp_and_call_stack_fields() {
    let wasm = add_one_wasm();
    let run = collect_wasmtime_steps(&wasm, &HostEventBindings::default(), "run", &[]).expect("trace");
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

    // The argument already sits where the callee frame starts.
    assert!(!call_step.state_after.frame_move.active);
    assert_eq!(call_step.state_after.locals_fbp, call_step.state_after.sp - 1);
    assert_eq!(call_step.state_after.stack_frame_base, call_step.state_after.sp);
    assert!(!trace
        .iter()
        .any(|row| row.row_kind == WasmRowKind::Aux(WasmAuxOpcode::LocalZero)));
    let local_get = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::LocalGet)
        .expect("callee local.get");
    assert_eq!(local_get.local_read_value, Some(5), "callee param value must be 5");

    // The return moves the result down over the callee's parameter.
    let moves: Vec<_> = trace
        .iter()
        .filter(|r| r.row_kind == WasmRowKind::Aux(WasmAuxOpcode::FrameMove))
        .collect();
    assert_eq!(moves.len(), 1, "expected one result move");
    let mv = moves[0];
    assert_eq!(mv.state_before.frame_move.remaining, 1);
    assert_eq!(mv.state_before.frame_move.distance, 1);
    assert!(!mv.state_after.frame_move.active);
    assert_eq!(mv.stack_read0.map(|read| (read.addr_lo, read.value_lo)), Some((2, 6)));
    assert_eq!(
        mv.stack_write0.map(|write| (write.addr_lo, write.value_lo)),
        Some((0, 6))
    );

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

    // Only the very last step is activation_finished.
    let last = trace.last().expect("non-empty trace");
    assert!(
        last.state_after.activation_finished,
        "last step must be activation_finished"
    );
    for step in &trace[..trace.len() - 1] {
        assert!(
            !step.state_after.activation_finished,
            "intermediate step {} must not be activation_finished",
            step.cycle
        );
    }
}

#[test]
fn nested_two_param_calls_start_frames_at_their_arguments() {
    let wasm = nested_two_param_wasm();
    let run = collect_wasmtime_steps(&wasm, &HostEventBindings::default(), "run", &[]).expect("trace");
    assert_eq!(run.results.as_slice(), &["11".to_string()], "expected sum2(4, 7) = 11");

    let trace = traces_from_wasmtime_steps(&run.steps).expect("normalize");
    let calls: Vec<_> = trace
        .iter()
        .filter(|row| row.call_stack_push.is_some())
        .collect();
    assert_eq!(calls.len(), 2);
    for call in &calls {
        assert_eq!(call.state_after.locals_fbp, call.state_after.sp - 2);
        assert_eq!(call.state_after.stack_frame_base, call.state_after.sp);
    }
    assert_eq!(
        calls[1].state_after.locals_fbp, 2,
        "the inner frame starts at the args pushed above the outer frame's params"
    );

    // Each return moves its one result down over the callee's two params.
    let moves: Vec<_> = trace
        .iter()
        .filter(|row| row.row_kind == WasmRowKind::Aux(WasmAuxOpcode::FrameMove))
        .collect();
    assert_eq!(moves.len(), 2);
    for mv in moves {
        assert_eq!(mv.state_before.frame_move.distance, 2);
        assert_eq!(mv.state_before.sp, mv.state_after.sp, "moves keep sp");
    }
}

#[test]
fn call_indirect_guest_target_initializes_params() {
    let wasm = call_indirect_param_wasm();
    let run = collect_wasmtime_steps(&wasm, &HostEventBindings::default(), "run", &[]).expect("trace");
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
    // The table index is popped; the argument below it becomes the param.
    assert_eq!(call.state_after.sp, 1);
    assert_eq!(call.state_after.locals_fbp, 0);
    let local_get = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::LocalGet)
        .expect("callee local.get");
    assert_eq!(local_get.local_read_value, Some(5));
}

#[test]
fn call_trace_passes_witness_checks() {
    let wasm = add_one_wasm();
    let artifacts = extract_wasm_program_artifacts(&wasm).expect("program artifacts");
    let run = collect_wasmtime_steps(&wasm, &HostEventBindings::default(), "run", &[]).expect("trace");
    let trace = traces_from_wasmtime_steps(&run.steps).expect("normalize");
    common::sanity_check_trace(&trace, &artifacts);
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
fn reused_guest_frame_zeroes_both_lanes_before_a_local_read() {
    let checked = common::checked_wasm_run(
        r#"(module
            (func $dirty (local i64)
                i64.const -1 local.set 0)
            (func $read (result i64) (local i64)
                local.get 0)
            (func (export "main") (result i64)
                call $dirty
                call $read))"#,
        "main",
    );
    assert_eq!(checked.run.results, ["0"]);
    let zero_rows: Vec<_> = checked
        .trace
        .iter()
        .filter(|row| row.row_kind == WasmRowKind::Aux(WasmAuxOpcode::LocalZero))
        .collect();
    assert_eq!(zero_rows.len(), 2);
    // Both callees open their frame at cell 0 and push their local there.
    assert!(zero_rows.iter().all(|row| {
        row.stack_write0
            .is_some_and(|write| write.addr_lo == 0 && write.value_lo == 0 && write.value_hi == Some(0))
    }));
    common::sanity_check_trace(&checked.trace, &checked.artifacts);
    common::ccs_check_trace(&checked.trace);

    let mut forged = neo_wasm::witness_builder::build_witness_vector(zero_rows[1]);
    forged[neo_wasm::layout::COL_STACK_WRITE0_VALUE_HI] = neo_math::F::ONE;
    common::assert_rejected(&forged, "local zero row cannot retain the previous high limb");

    let mut omitted = checked.trace.clone();
    let second_zero = omitted
        .iter()
        .rposition(|row| row.row_kind == WasmRowKind::Aux(WasmAuxOpcode::LocalZero))
        .unwrap();
    omitted.remove(second_zero);
    let witnesses = build_witnesses(&omitted);
    let layout = build_wasm_relation_layout();
    assert!(neo_application::check_continuity_rows(&layout.auxiliary.continuity, &witnesses).is_err());

    let mut forged_count = checked.trace.clone();
    let call = forged_count
        .iter_mut()
        .find(|row| row.opcode == WasmOpcode::Call)
        .unwrap();
    call.state_after.local_zero.remaining = 0;
    call.state_after.local_zero.active = false;
    let witnesses = build_witnesses(&forged_count);
    let preload = preload_from_program_artifacts(&checked.artifacts);
    let error = sanity_check_memory_rows(&layout, &witnesses, &preload).unwrap_err();
    assert!(error.contains("function_local_count"), "{error}");
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
fn clean_activation_finished_row_requires_empty_call_stack_depth() {
    let wasm = add_one_wasm();
    let run = collect_wasmtime_steps(&wasm, &HostEventBindings::default(), "run", &[]).expect("trace");
    let trace = traces_from_wasmtime_steps(&run.steps).expect("normalize");
    let mut final_row = trace
        .iter()
        .find(|row| row.state_after.activation_finished)
        .expect("activation_finished row")
        .clone();
    final_row.state_before.call_stack_depth = 1;
    final_row.state_after.call_stack_depth = 1;

    let witness = neo_wasm::witness_builder::build_witness_vector(&final_row);
    common::assert_rejected(
        &witness,
        "clean activation_finished row with non-empty call stack depth",
    );
}

#[test]
fn nested_trap_may_finish_activation_with_nonempty_call_stack_depth() {
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
    let run = collect_wasmtime_steps(&wasm, &HostEventBindings::default(), "run", &[]).expect("trace nested trap");
    let trace = traces_from_wasmtime_steps(&run.steps).expect("normalize nested trap");
    let trap = trace.last().expect("terminal trap row");

    assert!(trap.state_after.activation_finished);
    assert!(trap.state_after.trapped);
    assert_eq!(trap.state_before.call_stack_depth, 1);
    let witness = neo_wasm::witness_builder::build_witness_vector(trap);
    common::assert_satisfied(&witness, "nested trap with abandoned caller frame");
}

#[test]
fn call_row_pins_return_pc_rom_choice() {
    let wasm = add_one_wasm();
    let run = collect_wasmtime_steps(&wasm, &HostEventBindings::default(), "run", &[]).expect("trace");
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
fn activation_return_captures_simple_output() {
    let wasm = add_one_wasm();
    let run = collect_wasmtime_steps(&wasm, &HostEventBindings::default(), "run", &[]).expect("trace");
    let trace = traces_from_wasmtime_steps(&run.steps).expect("normalize");
    let final_row = trace
        .iter()
        .find(|row| row.state_after.activation_finished)
        .expect("activation_finished row");

    assert!(
        final_row.output_captured,
        "activation_finished row should capture the result"
    );
    assert!(final_row.state_after.output.enabled, "result carry should be enabled");
    assert_eq!(final_row.state_after.output.value_lo, 6);
    assert_eq!(final_row.state_after.output.value_hi, 0);
}

#[test]
fn activation_return_with_a_result_requires_output_capture() {
    let wasm = add_one_wasm();
    let run = collect_wasmtime_steps(&wasm, &HostEventBindings::default(), "run", &[]).expect("trace");
    let trace = traces_from_wasmtime_steps(&run.steps).expect("normalize");
    let mut final_row = trace
        .iter()
        .find(|row| row.state_after.activation_finished)
        .expect("activation_finished row")
        .clone();

    final_row.output_captured = false;
    final_row.state_after.output.enabled = false;
    final_row.state_after.output.value_lo = 0;
    final_row.state_after.output.value_hi = 0;
    final_row.state_after.sp = final_row.state_before.sp;

    let witness = neo_wasm::witness_builder::build_witness_vector(&final_row);
    common::assert_rejected(&witness, "activation return leaving its result uncaptured");
}

#[test]
fn activation_return_output_low_is_row_bound() {
    let wasm = add_one_wasm();
    let run = collect_wasmtime_steps(&wasm, &HostEventBindings::default(), "run", &[]).expect("trace");
    let trace = traces_from_wasmtime_steps(&run.steps).expect("normalize");
    let final_row = trace
        .iter()
        .find(|row| row.state_after.activation_finished)
        .expect("activation_finished row");
    let mut witness = neo_wasm::witness_builder::build_witness_vector(final_row);

    witness[COL_OUTPUT_VALUE_LO_AFTER] = neo_math::F::from_u64(7);

    common::assert_rejected(&witness, "activation_finished output with mismatched low limb");
}

#[test]
fn activation_return_output_low_is_stack_memory_bound() {
    let wasm = add_one_wasm();
    let artifacts = extract_wasm_program_artifacts(&wasm).expect("program artifacts");
    let run = collect_wasmtime_steps(&wasm, &HostEventBindings::default(), "run", &[]).expect("trace");
    let trace = traces_from_wasmtime_steps(&run.steps).expect("normalize");
    let layout = build_wasm_relation_layout();
    let mut witnesses = build_witnesses(&trace);
    let final_idx = trace
        .iter()
        .position(|row| row.state_after.activation_finished)
        .expect("activation_finished row");

    witnesses[final_idx][COL_OUTPUT_VALUE_LO_AFTER] = neo_math::F::from_u64(7);
    witnesses[final_idx][COL_STACK_READ_VALUE_LO[0]] = neo_math::F::from_u64(7);

    let preload = preload_from_program_artifacts(&artifacts);
    let err = sanity_check_memory_rows(layout, &witnesses, &preload)
        .err()
        .expect("must reject output stack memory mismatch");
    assert!(err.contains("memory `stack`"), "unexpected error: {err}");
}

#[test]
fn padding_right_after_a_result_move_preserves_the_move_state() {
    let wasm = add_one_wasm();
    let run = collect_wasmtime_steps(&wasm, &HostEventBindings::default(), "run", &[]).expect("trace");
    let trace = traces_from_wasmtime_steps(&run.steps).expect("normalize");
    let last_move = trace
        .iter()
        .rfind(|row| row.row_kind == WasmRowKind::Aux(WasmAuxOpcode::FrameMove))
        .expect("result move row");
    assert!(!last_move.state_after.frame_move.active);
    assert_ne!(last_move.state_after.frame_move.distance, 0);

    let padding = neo_wasm::batch::padding_step_after(last_move);
    assert_eq!(padding.state_before, padding.state_after);
    let witness = neo_wasm::witness_builder::build_witness_vector(&padding);
    common::assert_satisfied(&witness, "padding after a completed frame move");
}
