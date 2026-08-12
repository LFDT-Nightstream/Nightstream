//! Row-level CCS tests for integer conversion opcodes.

mod common;

use common::{assert_rejected, assert_satisfied};
use neo_math::F;
use neo_wasm::layout::{COL_STACK_WRITE0_VALUE_HI, COL_STACK_WRITE0_VALUE_LO};
use neo_wasm::witness_builder::build_witness_vector;
use neo_wasm::{
    opcode_code, opcode_info_from_code, StackValueAccess, WasmCountdownState, WasmOpcode, WasmOutputState,
    WasmPcEdgeKind, WasmRowKind, WasmStepState, WasmVmStep,
};
use p3_field::PrimeCharacteristicRing;

fn sign_extend_u32(value: u32, width_bytes: usize) -> u32 {
    let shift = 32 - width_bytes * 8;
    ((value << shift) as i32 >> shift) as u32
}

fn sign_extend_high(value: u32, width_bytes: usize) -> u32 {
    let sign_source = value.to_le_bytes()[width_bytes - 1];
    if sign_source & 0x80 == 0 {
        0
    } else {
        u32::MAX
    }
}

fn conversion_row(opcode: WasmOpcode, value: u32, width_bytes: usize, writes_i64: bool) -> WasmVmStep {
    let code = opcode_code(opcode);
    let output_lo = sign_extend_u32(value, width_bytes);
    let output_hi = if writes_i64 {
        Some(sign_extend_high(value, width_bytes))
    } else {
        None
    };
    let state_before = WasmStepState {
        pc: 2,
        sp: 1,
        stack_frame_base: 0,
        output: WasmOutputState::ZERO,
        call_stack_depth: 0,
        memory_pages: None,
        max_memory_pages: None,
        locals_fbp: 0,
        halted: false,
        trapped: false,
        param_init: WasmCountdownState::ZERO,
        tail_call_pending: false,
        host_callee_fref: 0,
        comm_chain: [0; 4],
        event_absorb: neo_wasm::WasmEventAbsorbState::ZERO,
        host_events: neo_wasm::WasmHostEventState::ZERO,
    };
    let state_after = WasmStepState {
        pc: 3,
        sp: 1,
        stack_frame_base: 0,
        output: WasmOutputState::ZERO,
        call_stack_depth: 0,
        memory_pages: None,
        max_memory_pages: None,
        locals_fbp: 0,
        halted: false,
        trapped: false,
        param_init: WasmCountdownState::ZERO,
        tail_call_pending: false,
        host_callee_fref: 0,
        comm_chain: [0; 4],
        event_absorb: neo_wasm::WasmEventAbsorbState::ZERO,
        host_events: neo_wasm::WasmHostEventState::ZERO,
    };
    WasmVmStep {
        cycle: 0,
        row_kind: WasmRowKind::Program,
        state_before,
        state_after,
        control_choice: 0,
        pc_edge_kind: WasmPcEdgeKind::Static,
        wide_values_enabled: writes_i64,
        opcode,
        info: opcode_info_from_code(code),
        stack_reads_override: None,
        stack_writes_override: None,
        output_captured: false,
        current_function_ref: 0,
        current_function_num_locals: 0,
        stack_read0: Some(StackValueAccess::new(0, value).with_optional_hi(if writes_i64 {
            Some(0x1234_5678)
        } else {
            None
        })),
        stack_read1: None,
        stack_read2: None,
        stack_write0: Some(StackValueAccess::new(0, output_lo).with_optional_hi(output_hi)),
        linear_memory: None,
        linear_memory_offset: 0,
        local_index: None,
        local_read_value: None,
        local_read_value_hi: None,
        local_write_value: None,
        local_write_value_hi: None,
        global_index: None,
        global_read_value: None,
        global_read_value_hi: None,
        global_write_value: None,
        global_write_value_hi: None,
        table_id: None,
        table_index: None,
        table_value: None,
        function_ref: None,
        target_function_is_guest: false,
        function_type_id: None,
        call_indirect_type_index: None,
        expected_type_id: None,
        table_size: None,
        call_param_count: None,
        call_result_count: None,
        call_stack_push: None,
        call_stack_pop: None,
        host_event_rom_slot: None,
        host_event_pre_count: None,
        host_event_post_count: None,
    }
}

#[test]
fn integer_sign_extension_rows_are_accepted() {
    for (opcode, value, width_bytes, writes_i64) in [
        (WasmOpcode::I32Extend8S, 0x1234_0080, 1, false),
        (WasmOpcode::I32Extend16S, 0x1234_8001, 2, false),
        (WasmOpcode::I64Extend8S, 0x1234_007f, 1, true),
        (WasmOpcode::I64Extend16S, 0x1234_8001, 2, true),
        (WasmOpcode::I64Extend32S, 0x8000_0001, 4, true),
    ] {
        let row = conversion_row(opcode, value, width_bytes, writes_i64);
        assert_satisfied(&build_witness_vector(&row), opcode.name());
    }
}

#[test]
fn integer_sign_extension_rows_reject_tampered_low_output() {
    for (opcode, value, width_bytes, writes_i64) in [
        (WasmOpcode::I32Extend8S, 0x1234_0080, 1, false),
        (WasmOpcode::I32Extend16S, 0x1234_8001, 2, false),
        (WasmOpcode::I64Extend8S, 0x1234_0080, 1, true),
        (WasmOpcode::I64Extend16S, 0x1234_8001, 2, true),
        (WasmOpcode::I64Extend32S, 0x8000_0001, 4, true),
    ] {
        let row = conversion_row(opcode, value, width_bytes, writes_i64);
        let mut witness = build_witness_vector(&row);
        witness[COL_STACK_WRITE0_VALUE_LO] += F::ONE;
        assert_rejected(&witness, opcode.name());
    }
}

#[test]
fn i64_sign_extension_rows_reject_tampered_high_output() {
    for (opcode, value, width_bytes) in [
        (WasmOpcode::I64Extend8S, 0x1234_0080, 1),
        (WasmOpcode::I64Extend16S, 0x1234_8001, 2),
        (WasmOpcode::I64Extend32S, 0x8000_0001, 4),
    ] {
        let row = conversion_row(opcode, value, width_bytes, true);
        let mut witness = build_witness_vector(&row);
        witness[COL_STACK_WRITE0_VALUE_HI] = F::ZERO;
        assert_rejected(&witness, opcode.name());
    }
}
