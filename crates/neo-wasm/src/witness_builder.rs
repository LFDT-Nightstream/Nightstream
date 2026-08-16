use super::gadgets::{unsigned_ge_witness, zero_test_witness_field, zero_test_witness_u64};
use super::ir::{pack_function_call_metadata, WasmHostEventSlotKind, WasmRowKind, WasmVmStep};
use super::layout::{
    selector_col, COL_CALL_INDIRECT_IS_NOT_TRAP, COL_CALL_INDIRECT_IS_TRAP, COL_CALL_INDIRECT_TYPE_INDEX,
    COL_CALL_PARAM_COUNT, COL_CALL_RESULT_COUNT, COL_CALL_STACK_ADDR, COL_CALL_STACK_CALLER_FBP_VALUE,
    COL_CALL_STACK_CALLER_SP_BASE_VALUE, COL_CALL_STACK_DEPTH_AFTER, COL_CALL_STACK_DEPTH_BEFORE,
    COL_CALL_STACK_POP_PRESENT, COL_CALL_STACK_PUSH_PRESENT, COL_CALL_STACK_RETURN_PC_VALUE, COL_CALL_TARGET_METADATA,
    COL_CI_ENTRY_IS_NULL, COL_CI_ENTRY_NULL_INV, COL_CI_HOST_CALL, COL_CI_OOB, COL_CI_TYPE_EQ, COL_CI_TYPE_EQ_INV,
    COL_CMP_GE, COL_CMP_LOW, COL_COMM_CHAIN_AFTER, COL_COMM_CHAIN_BEFORE, COL_CONTROL_CHOICE,
    COL_CURRENT_FUNCTION_NUM_LOCALS, COL_CURRENT_FUNCTION_REF, COL_DIV_DIVIDEND_IS_MIN, COL_DIV_DIVIDEND_MIN_INV,
    COL_DIV_DIVISOR_INV, COL_DIV_DIVISOR_IS_NEG1, COL_DIV_DIVISOR_IS_ZERO, COL_DIV_DIVISOR_NEG1_INV, COL_DIV_OVERFLOW,
    COL_DIV_OVERFLOW_COND, COL_DIV_TRAP, COL_EXPECTED_TYPE_ID, COL_FUNCTION_CALL_TYPE_LOOKUP_GATE, COL_FUNCTION_REF,
    COL_FUNCTION_TYPE_ID, COL_GLOBAL_INDEX, COL_GLOBAL_VALUE, COL_GLOBAL_VALUE_HI, COL_GROW_SUCCESS,
    COL_GUEST_ENTRY_ACTIVE, COL_HALTED, COL_HALTED_BEFORE, COL_HOST_CALLEE_FREF_AFTER, COL_HOST_CALLEE_FREF_BEFORE,
    COL_IS_PROGRAM_ROW, COL_LINEAR_MEM_ACCESS_BYTE0, COL_LINEAR_MEM_ACCESS_BYTE1, COL_LINEAR_MEM_ACCESS_BYTE2,
    COL_LINEAR_MEM_ACCESS_BYTE3, COL_LINEAR_MEM_ACCESS_BYTE4, COL_LINEAR_MEM_ACCESS_BYTE5, COL_LINEAR_MEM_ACCESS_BYTE6,
    COL_LINEAR_MEM_ACCESS_BYTE7, COL_LINEAR_MEM_BYTE_OFFSET, COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_0,
    COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_1, COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_2,
    COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_3, COL_LINEAR_MEM_DOUBLE_WIDTH_OFFSET_IS_0,
    COL_LINEAR_MEM_DOUBLE_WIDTH_OFFSET_IS_1, COL_LINEAR_MEM_DOUBLE_WIDTH_OFFSET_IS_2,
    COL_LINEAR_MEM_DOUBLE_WIDTH_OFFSET_IS_3, COL_LINEAR_MEM_FULL_WIDTH_OFFSET_IS_0,
    COL_LINEAR_MEM_FULL_WIDTH_OFFSET_IS_1, COL_LINEAR_MEM_FULL_WIDTH_OFFSET_IS_2,
    COL_LINEAR_MEM_FULL_WIDTH_OFFSET_IS_3, COL_LINEAR_MEM_HALF_WIDTH_OFFSET_IS_0,
    COL_LINEAR_MEM_HALF_WIDTH_OFFSET_IS_1, COL_LINEAR_MEM_HALF_WIDTH_OFFSET_IS_2,
    COL_LINEAR_MEM_HALF_WIDTH_OFFSET_IS_3, COL_LINEAR_MEM_I64_LOAD_OFFSET_IS_0, COL_LINEAR_MEM_I64_LOAD_OFFSET_IS_1,
    COL_LINEAR_MEM_I64_LOAD_OFFSET_IS_2, COL_LINEAR_MEM_I64_LOAD_OFFSET_IS_3, COL_LINEAR_MEM_I64_STORE_OFFSET_IS_0,
    COL_LINEAR_MEM_I64_STORE_OFFSET_IS_1, COL_LINEAR_MEM_I64_STORE_OFFSET_IS_2, COL_LINEAR_MEM_I64_STORE_OFFSET_IS_3,
    COL_LINEAR_MEM_IMM_OFFSET, COL_LINEAR_MEM_IS_BYTE_WIDTH, COL_LINEAR_MEM_IS_DOUBLE_WIDTH,
    COL_LINEAR_MEM_IS_FULL_WIDTH, COL_LINEAR_MEM_IS_HALF_WIDTH, COL_LINEAR_MEM_LANE0_BYTE0,
    COL_LINEAR_MEM_LANE0_BYTE0_BEFORE, COL_LINEAR_MEM_LANE0_BYTE1, COL_LINEAR_MEM_LANE0_BYTE1_BEFORE,
    COL_LINEAR_MEM_LANE0_BYTE2, COL_LINEAR_MEM_LANE0_BYTE2_BEFORE, COL_LINEAR_MEM_LANE0_BYTE3,
    COL_LINEAR_MEM_LANE0_BYTE3_BEFORE, COL_LINEAR_MEM_LANE1_BYTE0, COL_LINEAR_MEM_LANE1_BYTE0_BEFORE,
    COL_LINEAR_MEM_LANE1_BYTE1, COL_LINEAR_MEM_LANE1_BYTE1_BEFORE, COL_LINEAR_MEM_LANE1_BYTE2,
    COL_LINEAR_MEM_LANE1_BYTE2_BEFORE, COL_LINEAR_MEM_LANE1_BYTE3, COL_LINEAR_MEM_LANE1_BYTE3_BEFORE,
    COL_LINEAR_MEM_LANE2_BYTE0, COL_LINEAR_MEM_LANE2_BYTE0_BEFORE, COL_LINEAR_MEM_LANE2_BYTE1,
    COL_LINEAR_MEM_LANE2_BYTE1_BEFORE, COL_LINEAR_MEM_LANE2_BYTE2, COL_LINEAR_MEM_LANE2_BYTE2_BEFORE,
    COL_LINEAR_MEM_LANE2_BYTE3, COL_LINEAR_MEM_LANE2_BYTE3_BEFORE, COL_LINEAR_MEM_LANE_ADDR,
    COL_LINEAR_MEM_LANE_LOAD_ACTIVE, COL_LINEAR_MEM_LANE_STORE_ACTIVE, COL_LINEAR_MEM_LANE_VALUE,
    COL_LINEAR_MEM_LANE_VALUE_BEFORE, COL_LINEAR_MEM_OFFSET_IS_0, COL_LINEAR_MEM_OFFSET_IS_1,
    COL_LINEAR_MEM_OFFSET_IS_2, COL_LINEAR_MEM_OFFSET_IS_3, COL_LINEAR_MEM_USE_LANE0, COL_LINEAR_MEM_USE_LANE1,
    COL_LINEAR_MEM_USE_LANE2, COL_LOCALS_FBP_AFTER, COL_LOCALS_FBP_BEFORE, COL_LOCAL_INDEX, COL_LOCAL_VALUE,
    COL_LOCAL_VALUE_HI, COL_LOCAL_WRITE_ENABLED, COL_MAX_MEMORY_PAGES_AFTER, COL_MAX_MEMORY_PAGES_BEFORE,
    COL_MEMORY_PAGES_AFTER, COL_MEMORY_PAGES_BEFORE, COL_MEM_LOAD_LIVE, COL_MEM_OOB, COL_MEM_STORE_LIVE, COL_ONE,
    COL_OPCODE_CODE, COL_OP_TABLE_ENABLED, COL_OP_TABLE_ID, COL_OP_TABLE_VALUE, COL_OUTPUT_CAPTURED,
    COL_OUTPUT_ENABLED_AFTER, COL_OUTPUT_ENABLED_BEFORE, COL_OUTPUT_VALUE_HI_AFTER, COL_OUTPUT_VALUE_HI_BEFORE,
    COL_OUTPUT_VALUE_LO_AFTER, COL_OUTPUT_VALUE_LO_BEFORE, COL_PADDING_ACTIVE, COL_PARAM_INIT_ACTIVE_AFTER,
    COL_PARAM_INIT_ACTIVE_BEFORE, COL_PARAM_INIT_REMAINING_AFTER, COL_PARAM_INIT_REMAINING_AFTER_INV,
    COL_PARAM_INIT_REMAINING_AFTER_IS_ZERO, COL_PARAM_INIT_REMAINING_BEFORE, COL_PC_AFTER, COL_PC_BEFORE,
    COL_PC_EDGE_KIND, COL_PC_EDGE_KIND_INV, COL_PC_EDGE_KIND_IS_STATIC, COL_PC_ROM_ACTIVE,
    COL_PC_ROM_CALL_RETURN_CHOICE, COL_PROGRAM_CALL_INDIRECT_IMMEDIATES_ACTIVE, COL_PROGRAM_GLOBAL_INDEX_ACTIVE,
    COL_PROGRAM_LOCAL_INDEX_ACTIVE, COL_PROGRAM_TABLE_ID_ACTIVE, COL_SELECT_OUT_DELTA_HI, COL_SELECT_OUT_DELTA_LO,
    COL_SIGN_EXT_BIT, COL_SIGN_EXT_LOW7, COL_SP_AFTER, COL_SP_BEFORE, COL_STACK_FRAME_BASE_AFTER,
    COL_STACK_FRAME_BASE_BEFORE, COL_STACK_READS, COL_STACK_READ_ACTIVE, COL_STACK_READ_ADDR_HI,
    COL_STACK_READ_ADDR_LO, COL_STACK_READ_VALUE_HI, COL_STACK_READ_VALUE_LO, COL_STACK_WRITE0_ACTIVE,
    COL_STACK_WRITE0_ADDR_HI, COL_STACK_WRITE0_ADDR_LO, COL_STACK_WRITE0_VALUE_HI, COL_STACK_WRITE0_VALUE_LO,
    COL_STACK_WRITES, COL_TABLE_ID, COL_TABLE_INDEX, COL_TABLE_READ_ENABLED, COL_TABLE_SIZE,
    COL_TABLE_SIZE_READ_ENABLED, COL_TABLE_VALUE, COL_TAIL_CALL_PENDING_AFTER, COL_TAIL_CALL_PENDING_BEFORE,
    COL_TAIL_DISCARD_COUNT, COL_TAIL_ENTER_ACTIVE, COL_TARGET_FUNCTION_IS_GUEST, COL_TRAPPED_AFTER, COL_TRAPPED_BEFORE,
    COL_TURN_EXPORT_FREF_AFTER, COL_TURN_EXPORT_FREF_BEFORE, COL_WIDE_AUX0, COL_WIDE_AUX1, COL_WIDE_VALUES_ENABLED,
    PC_ROM_CALL_RETURN_CHOICE,
};
use crate::layout::{
    COL_CMP_AND, COL_CMP_HI_DIFF, COL_CMP_HI_INV, COL_CMP_HI_IS_ZERO, COL_CMP_LO_DIFF, COL_CMP_LO_INV,
    COL_CMP_LO_IS_ZERO, COL_SELECT_COND_IS_ZERO, COL_SELECT_SCRATCH_INV,
};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

pub fn build_witness_vector(trace: &WasmVmStep) -> Vec<F> {
    let mut wit = vec![F::ZERO; crate::RANGE_CHECKED_WITNESS_WIDTH];
    wit[COL_ONE] = F::ONE;
    // High-limb stack addresses are constrained unconditionally as
    // `addr_hi = addr_lo + 1`. Inactive low addresses default to 0 and
    // inactive memory specs are gated off, so 1 is the canonical inactive
    // high address. Active lanes overwrite this below with `addr + 1`.
    wit[COL_STACK_READ_ADDR_HI[0]] = F::ONE;
    wit[COL_STACK_READ_ADDR_HI[1]] = F::ONE;
    wit[COL_STACK_READ_ADDR_HI[2]] = F::ONE;
    wit[COL_STACK_WRITE0_ADDR_HI] = F::ONE;
    let opcode_code = if trace.row_kind.is_program() {
        trace.info.code
    } else {
        0
    };
    wit[COL_OPCODE_CODE] = F::from_u64(u64::from(opcode_code));
    wit[COL_PC_BEFORE] = F::from_u64(trace.state_before.pc);
    wit[COL_PC_AFTER] = F::from_u64(trace.state_after.pc);
    wit[COL_STACK_FRAME_BASE_BEFORE] = F::from_u64(trace.state_before.stack_frame_base);
    wit[COL_STACK_FRAME_BASE_AFTER] = F::from_u64(trace.state_after.stack_frame_base);
    wit[COL_TAIL_CALL_PENDING_BEFORE] = if trace.state_before.tail_call_pending {
        F::ONE
    } else {
        F::ZERO
    };
    wit[COL_TAIL_CALL_PENDING_AFTER] = if trace.state_after.tail_call_pending {
        F::ONE
    } else {
        F::ZERO
    };
    wit[COL_TAIL_ENTER_ACTIVE] = if trace.row_kind.is_tail_enter() {
        F::ONE
    } else {
        F::ZERO
    };
    if trace.row_kind.is_tail_enter() {
        wit[COL_TAIL_DISCARD_COUNT] = F::from_u64(trace.state_before.sp - trace.state_after.sp);
    }
    wit[COL_CONTROL_CHOICE] = F::from_u64(u64::from(trace.control_choice));
    wit[COL_PC_EDGE_KIND] = F::from_u64(u64::from(trace.pc_edge_kind.as_u32()));
    let (pc_edge_kind_is_static, pc_edge_kind_inv) = zero_test_witness_u64(u64::from(trace.pc_edge_kind.as_u32()));
    wit[COL_PC_EDGE_KIND_IS_STATIC] = pc_edge_kind_is_static;
    wit[COL_PC_EDGE_KIND_INV] = pc_edge_kind_inv;
    write_param_init_state(&mut wit, true, trace.state_before.param_init);
    write_param_init_state(&mut wit, false, trace.state_after.param_init);
    let (remaining_is_zero, remaining_inv) = zero_test_witness_u64(u64::from(trace.state_after.param_init.remaining));
    wit[COL_PARAM_INIT_REMAINING_AFTER_IS_ZERO] = remaining_is_zero;
    wit[COL_PARAM_INIT_REMAINING_AFTER_INV] = remaining_inv;
    wit[COL_HOST_CALLEE_FREF_BEFORE] = F::from_u64(u64::from(trace.state_before.host_callee_fref));
    wit[COL_HOST_CALLEE_FREF_AFTER] = F::from_u64(u64::from(trace.state_after.host_callee_fref));
    wit[COL_TURN_EXPORT_FREF_BEFORE] = F::from_u64(u64::from(trace.state_before.host_events.turn_export_fref));
    wit[COL_TURN_EXPORT_FREF_AFTER] = F::from_u64(u64::from(trace.state_after.host_events.turn_export_fref));
    for (i, (before_col, after_col)) in COL_COMM_CHAIN_BEFORE
        .into_iter()
        .zip(COL_COMM_CHAIN_AFTER)
        .enumerate()
    {
        wit[before_col] = F::from_u64(trace.state_before.comm_chain[i]);
        wit[after_col] = F::from_u64(trace.state_after.comm_chain[i]);
    }
    wit[COL_WIDE_VALUES_ENABLED] = if trace.wide_values_enabled { F::ONE } else { F::ZERO };
    wit[COL_SP_BEFORE] = F::from_u64(trace.state_before.sp);
    wit[COL_SP_AFTER] = F::from_u64(trace.state_after.sp);
    wit[COL_OUTPUT_ENABLED_BEFORE] = if trace.state_before.output.enabled {
        F::ONE
    } else {
        F::ZERO
    };
    wit[COL_OUTPUT_ENABLED_AFTER] = if trace.state_after.output.enabled {
        F::ONE
    } else {
        F::ZERO
    };
    wit[COL_OUTPUT_VALUE_LO_BEFORE] = F::from_u64(u64::from(trace.state_before.output.value_lo));
    wit[COL_OUTPUT_VALUE_LO_AFTER] = F::from_u64(u64::from(trace.state_after.output.value_lo));
    wit[COL_OUTPUT_VALUE_HI_BEFORE] = F::from_u64(u64::from(trace.state_before.output.value_hi));
    wit[COL_OUTPUT_VALUE_HI_AFTER] = F::from_u64(u64::from(trace.state_after.output.value_hi));
    wit[COL_OUTPUT_CAPTURED] = if trace.output_captured { F::ONE } else { F::ZERO };
    wit[COL_CALL_STACK_DEPTH_BEFORE] = F::from_u64(trace.state_before.call_stack_depth);
    wit[COL_CALL_STACK_DEPTH_AFTER] = F::from_u64(trace.state_after.call_stack_depth);
    wit[COL_CURRENT_FUNCTION_REF] = F::from_u64(u64::from(trace.current_function_ref));
    wit[COL_CURRENT_FUNCTION_NUM_LOCALS] = F::from_u64(u64::from(trace.current_function_num_locals));
    wit[COL_LOCALS_FBP_BEFORE] = F::from_u64(trace.state_before.locals_fbp);
    wit[COL_LOCALS_FBP_AFTER] = F::from_u64(trace.state_after.locals_fbp);
    wit[COL_HALTED] = if trace.state_after.halted { F::ONE } else { F::ZERO };
    wit[COL_HALTED_BEFORE] = if trace.state_before.halted { F::ONE } else { F::ZERO };
    wit[COL_TRAPPED_BEFORE] = if trace.state_before.trapped { F::ONE } else { F::ZERO };
    wit[COL_TRAPPED_AFTER] = if trace.state_after.trapped { F::ONE } else { F::ZERO };
    wit[COL_IS_PROGRAM_ROW] = if trace.row_kind.is_program() { F::ONE } else { F::ZERO };
    wit[COL_PADDING_ACTIVE] = if trace.row_kind.is_padding() { F::ONE } else { F::ZERO };
    wit[COL_PC_ROM_ACTIVE] = if trace.row_kind.is_program() && trace.pc_edge_kind.as_u32() == 0 {
        F::ONE
    } else {
        F::ZERO
    };
    wit[COL_CALL_STACK_POP_PRESENT] = if trace.call_stack_pop.is_some() {
        F::ONE
    } else {
        F::ZERO
    };
    wit[COL_CALL_STACK_PUSH_PRESENT] = if trace.call_stack_push.is_some() {
        F::ONE
    } else {
        F::ZERO
    };
    wit[COL_GUEST_ENTRY_ACTIVE] = if trace.target_function_is_guest
        && matches!(
            trace.opcode,
            super::isa::WasmOpcode::Call
                | super::isa::WasmOpcode::CallIndirect
                | super::isa::WasmOpcode::ReturnCall
                | super::isa::WasmOpcode::ReturnCallIndirect
        ) {
        F::ONE
    } else {
        F::ZERO
    };
    if let Some((return_pc, caller_fbp, caller_stack_base)) = trace.call_stack_push.or(trace.call_stack_pop) {
        wit[COL_CALL_STACK_RETURN_PC_VALUE] = F::from_u64(return_pc);
        wit[COL_CALL_STACK_CALLER_FBP_VALUE] = F::from_u64(caller_fbp);
        wit[COL_CALL_STACK_CALLER_SP_BASE_VALUE] = F::from_u64(caller_stack_base);
    }
    if trace.call_stack_push.is_some() {
        wit[COL_CALL_STACK_ADDR] = F::from_u64(trace.state_before.call_stack_depth);
        wit[COL_PC_ROM_CALL_RETURN_CHOICE] = F::from_u64(PC_ROM_CALL_RETURN_CHOICE);
    } else if trace.call_stack_pop.is_some() {
        wit[COL_CALL_STACK_ADDR] = F::from_u64(trace.state_after.call_stack_depth);
    }
    if let Some(pages) = trace.state_before.memory_pages {
        wit[COL_MEMORY_PAGES_BEFORE] = F::from_u64(u64::from(pages));
    }
    if let Some(pages) = trace.state_after.memory_pages {
        wit[COL_MEMORY_PAGES_AFTER] = F::from_u64(u64::from(pages));
    }
    if let Some(max) = trace.state_before.max_memory_pages {
        wit[COL_MAX_MEMORY_PAGES_BEFORE] = F::from_u64(u64::from(max));
    }
    if let Some(max) = trace.state_after.max_memory_pages {
        wit[COL_MAX_MEMORY_PAGES_AFTER] = F::from_u64(u64::from(max));
    }
    if matches!(trace.opcode, super::isa::WasmOpcode::MemoryGrow) {
        let before = u64::from(trace.state_before.memory_pages.unwrap_or(0));
        let max = u64::from(trace.state_before.max_memory_pages.unwrap_or(0));
        let delta = u64::from(trace.stack_read0.map(|access| access.value_lo).unwrap_or(0));
        // Mirrors the CCS lhs `max_before - pages_before`.
        let available_pages = max
            .checked_sub(before)
            .expect("validated memory state keeps current pages <= max pages");
        let (cmp_low, cmp_ge) = unsigned_ge_witness(available_pages, delta);
        wit[COL_CMP_LOW] = cmp_low;
        wit[COL_CMP_GE] = cmp_ge;
        wit[COL_GROW_SUCCESS] = cmp_ge;
    }
    wit[COL_STACK_READS] = F::from_u64(u64::from(trace.stack_reads_override.unwrap_or(trace.info.stack_reads)));
    wit[COL_STACK_WRITES] = F::from_u64(u64::from(
        trace
            .stack_writes_override
            .unwrap_or(trace.info.stack_writes),
    ));
    let stack_reads = trace.stack_reads_override.unwrap_or(trace.info.stack_reads);
    let stack_writes = trace
        .stack_writes_override
        .unwrap_or(trace.info.stack_writes);
    wit[COL_STACK_READ_ACTIVE[0]] = if stack_reads >= 1 { F::ONE } else { F::ZERO };
    wit[COL_STACK_READ_ACTIVE[1]] = if stack_reads >= 2 { F::ONE } else { F::ZERO };
    wit[COL_STACK_READ_ACTIVE[2]] = if stack_reads >= 3 { F::ONE } else { F::ZERO };
    wit[COL_STACK_WRITE0_ACTIVE] = if stack_writes >= 1 { F::ONE } else { F::ZERO };
    wit[COL_OP_TABLE_ENABLED] = if trace.info.uses_op_table { F::ONE } else { F::ZERO };
    let is_core_linear_memory = trace.row_kind.is_program() && trace.opcode.uses_linear_memory();
    let is_host_event_byte_memory = trace
        .host_event_rom_slot
        .is_some_and(|rom| rom.variant.uses_byte_memory_width());
    let is_host_event_half_memory = trace
        .host_event_rom_slot
        .is_some_and(|rom| rom.variant.uses_half_memory_width());
    let is_host_event_subword_memory = is_host_event_byte_memory || is_host_event_half_memory;
    wit[COL_LINEAR_MEM_USE_LANE0] = if is_core_linear_memory { F::ONE } else { F::ZERO };
    wit[COL_LOCAL_WRITE_ENABLED] = if matches!(
        trace.opcode,
        super::isa::WasmOpcode::LocalSet | super::isa::WasmOpcode::LocalTee
    ) {
        F::ONE
    } else {
        F::ZERO
    };
    wit[COL_TABLE_READ_ENABLED] = if matches!(
        trace.opcode,
        super::isa::WasmOpcode::TableGet
            | super::isa::WasmOpcode::CallIndirect
            | super::isa::WasmOpcode::ReturnCallIndirect
    ) {
        F::ONE
    } else {
        F::ZERO
    };
    // table_sizes is read by table.size and by call_indirect (the OOB check).
    wit[COL_TABLE_SIZE_READ_ENABLED] = if matches!(
        trace.opcode,
        super::isa::WasmOpcode::TableSize
            | super::isa::WasmOpcode::CallIndirect
            | super::isa::WasmOpcode::ReturnCallIndirect
    ) {
        F::ONE
    } else {
        F::ZERO
    };

    let is_program_row = matches!(trace.row_kind, WasmRowKind::Program);
    if is_program_row {
        if let Some(col) = selector_col(trace.opcode) {
            wit[col] = F::ONE;
        }
    }
    let program_immediate_gate = |consumes| {
        if is_program_row && consumes {
            F::ONE
        } else {
            F::ZERO
        }
    };
    wit[COL_PROGRAM_LOCAL_INDEX_ACTIVE] = program_immediate_gate(trace.opcode.uses_local_index_immediate());
    wit[COL_PROGRAM_GLOBAL_INDEX_ACTIVE] = program_immediate_gate(trace.opcode.uses_global_index_immediate());
    wit[COL_PROGRAM_TABLE_ID_ACTIVE] = program_immediate_gate(trace.opcode.uses_table_id_immediate());
    wit[COL_PROGRAM_CALL_INDIRECT_IMMEDIATES_ACTIVE] =
        program_immediate_gate(trace.opcode.uses_call_indirect_immediates());
    if let Some(read) = trace.stack_read0 {
        wit[COL_STACK_READ_ADDR_LO[0]] = F::from_u64(read.addr_lo);
        wit[COL_STACK_READ_ADDR_HI[0]] = F::from_u64(read.addr_lo + 1);
        wit[COL_STACK_READ_VALUE_LO[0]] = F::from_u64(u64::from(read.value_lo));
    }
    if trace.output_captured {
        debug_assert_eq!(
            trace.stack_reads_override.unwrap_or(trace.info.stack_reads),
            0,
            "output capture reuses inactive stack_read0 columns"
        );
        let output_addr = trace.state_before.sp.saturating_sub(1).saturating_mul(2);
        wit[COL_STACK_READ_ADDR_LO[0]] = F::from_u64(output_addr);
        wit[COL_STACK_READ_ADDR_HI[0]] = F::from_u64(output_addr + 1);
        wit[COL_STACK_READ_VALUE_LO[0]] = F::from_u64(u64::from(trace.state_after.output.value_lo));
        wit[COL_STACK_READ_VALUE_HI[0]] = F::from_u64(u64::from(trace.state_after.output.value_hi));
    }
    if trace.wide_values_enabled {
        if let Some(read0_value_hi) = trace.stack_read0.and_then(|read| read.value_hi) {
            wit[COL_STACK_READ_VALUE_HI[0]] = F::from_u64(u64::from(read0_value_hi));
        }
    }
    if let Some(read) = trace.stack_read1 {
        wit[COL_STACK_READ_ADDR_LO[1]] = F::from_u64(read.addr_lo);
        wit[COL_STACK_READ_ADDR_HI[1]] = F::from_u64(read.addr_lo + 1);
        wit[COL_STACK_READ_VALUE_LO[1]] = F::from_u64(u64::from(read.value_lo));
    }
    if trace.wide_values_enabled {
        if let Some(read1_value_hi) = trace.stack_read1.and_then(|read| read.value_hi) {
            wit[COL_STACK_READ_VALUE_HI[1]] = F::from_u64(u64::from(read1_value_hi));
        }
    }
    if let Some(read) = trace.stack_read2 {
        wit[COL_STACK_READ_ADDR_LO[2]] = F::from_u64(read.addr_lo);
        wit[COL_STACK_READ_ADDR_HI[2]] = F::from_u64(read.addr_lo + 1);
        wit[COL_STACK_READ_VALUE_LO[2]] = F::from_u64(u64::from(read.value_lo));
    }
    if trace.wide_values_enabled {
        if let Some(read2_value_hi) = trace.stack_read2.and_then(|read| read.value_hi) {
            wit[COL_STACK_READ_VALUE_HI[2]] = F::from_u64(u64::from(read2_value_hi));
        }
    }
    if let Some(write) = trace.stack_write0 {
        wit[COL_STACK_WRITE0_ADDR_LO] = F::from_u64(write.addr_lo);
        wit[COL_STACK_WRITE0_ADDR_HI] = F::from_u64(write.addr_lo + 1);
        wit[COL_STACK_WRITE0_VALUE_LO] = F::from_u64(u64::from(write.value_lo));
    }
    if trace.wide_values_enabled {
        if let Some(write0_value_hi) = trace.stack_write0.and_then(|write| write.value_hi) {
            wit[COL_STACK_WRITE0_VALUE_HI] = F::from_u64(u64::from(write0_value_hi));
        }
    }

    // Mirror the CCS div/rem trap gates (zero divisor and MIN / -1 overflow).
    let divisor = wit[COL_STACK_READ_VALUE_LO[1]] + wit[COL_STACK_READ_VALUE_HI[1]];
    let (divisor_is_zero, divisor_inv) = zero_test_witness_field(divisor);
    wit[COL_DIV_DIVISOR_IS_ZERO] = divisor_is_zero;
    wit[COL_DIV_DIVISOR_INV] = divisor_inv;
    let sel_i32_div_s = wit[selector_col(super::isa::WasmOpcode::I32DivS).expect("i32.div_s selector")];
    let sel_i32_rem_s = wit[selector_col(super::isa::WasmOpcode::I32RemS).expect("i32.rem_s selector")];
    let sel_i64_div_s = wit[selector_col(super::isa::WasmOpcode::I64DivS).expect("i64.div_s selector")];
    let sel_i64_rem_s = wit[selector_col(super::isa::WasmOpcode::I64RemS).expect("i64.rem_s selector")];
    let sel_i32_signed = sel_i32_div_s + sel_i32_rem_s;
    let sel_i64_signed = sel_i64_div_s + sel_i64_rem_s;
    let dividend_min = wit[COL_STACK_READ_VALUE_LO[0]] + wit[COL_STACK_READ_VALUE_HI[0]] * F::from_u64(1 << 32)
        - sel_i32_signed * F::from_u64(1 << 31)
        - sel_i64_signed * F::from_u64(1 << 63);
    let (dividend_is_min, dividend_min_inv) = zero_test_witness_field(dividend_min);
    wit[COL_DIV_DIVIDEND_IS_MIN] = dividend_is_min;
    wit[COL_DIV_DIVIDEND_MIN_INV] = dividend_min_inv;
    let divisor_neg1 = wit[COL_STACK_READ_VALUE_LO[1]] + wit[COL_STACK_READ_VALUE_HI[1]]
        - sel_i32_signed * F::from_u64(0xFFFF_FFFF)
        - sel_i64_signed * F::from_u64(0x1_FFFF_FFFE);
    let (divisor_is_neg1, divisor_neg1_inv) = zero_test_witness_field(divisor_neg1);
    wit[COL_DIV_DIVISOR_IS_NEG1] = divisor_is_neg1;
    wit[COL_DIV_DIVISOR_NEG1_INV] = divisor_neg1_inv;
    let overflow_cond = dividend_is_min * divisor_is_neg1;
    wit[COL_DIV_OVERFLOW_COND] = overflow_cond;
    let div_overflow = (sel_i32_div_s + sel_i64_div_s) * overflow_cond;
    wit[COL_DIV_OVERFLOW] = div_overflow;
    let div_zero_trap =
        trace.row_kind.is_program() && trace.opcode.traps_on_zero_divisor() && divisor_is_zero == F::ONE;
    if div_zero_trap || div_overflow == F::ONE {
        wit[COL_DIV_TRAP] = F::ONE;
        wit[COL_OP_TABLE_ENABLED] = F::ZERO;
    }
    if let Some(access) = trace.linear_memory {
        if is_core_linear_memory {
            wit[COL_LINEAR_MEM_IMM_OFFSET] = F::from_u64(trace.linear_memory_offset);
        }
        if is_core_linear_memory || is_host_event_subword_memory {
            wit[COL_LINEAR_MEM_BYTE_OFFSET] = F::from_u64(u64::from(access.byte_offset));
            wit[COL_LINEAR_MEM_USE_LANE1] = if access.lane1.is_some() { F::ONE } else { F::ZERO };
            wit[COL_LINEAR_MEM_USE_LANE2] = if access.lane2.is_some() { F::ONE } else { F::ZERO };
        }
        // Witness the CCS-bound load/store lane gates used by the memory spec.
        let is_load = trace
            .opcode
            .memory_access_info()
            .is_some_and(|info| info.kind == super::isa::WasmMemoryAccessKind::Load);
        let is_store = trace
            .opcode
            .memory_access_info()
            .is_some_and(|info| info.kind == super::isa::WasmMemoryAccessKind::Store);
        // OOB rows de-gate memory tuples.
        let pages_before = u64::from(trace.state_before.memory_pages.unwrap_or(0));
        let last_lane_addr =
            access.lane0.word_addr + u64::from(access.lane1.is_some()) + u64::from(access.lane2.is_some());
        let (cmp_low, cmp_ge) = unsigned_ge_witness(last_lane_addr, pages_before * 16384);
        wit[COL_CMP_LOW] = cmp_low;
        wit[COL_CMP_GE] = cmp_ge;
        wit[COL_MEM_OOB] = cmp_ge;
        let not_oob = F::ONE - cmp_ge;
        let load_live = if is_load { not_oob } else { F::ZERO };
        let store_live = if is_store { not_oob } else { F::ZERO };
        wit[COL_MEM_LOAD_LIVE] = load_live;
        wit[COL_MEM_STORE_LIVE] = store_live;
        wit[COL_LINEAR_MEM_LANE_LOAD_ACTIVE[0]] = load_live;
        wit[COL_LINEAR_MEM_LANE_LOAD_ACTIVE[1]] = if access.lane1.is_some() { load_live } else { F::ZERO };
        wit[COL_LINEAR_MEM_LANE_LOAD_ACTIVE[2]] = if access.lane2.is_some() { load_live } else { F::ZERO };
        wit[COL_LINEAR_MEM_LANE_STORE_ACTIVE[0]] = store_live;
        wit[COL_LINEAR_MEM_LANE_STORE_ACTIVE[1]] = if access.lane1.is_some() { store_live } else { F::ZERO };
        wit[COL_LINEAR_MEM_LANE_STORE_ACTIVE[2]] = if access.lane2.is_some() { store_live } else { F::ZERO };
        if is_core_linear_memory || is_host_event_subword_memory {
            match access.byte_offset {
                0 => wit[COL_LINEAR_MEM_OFFSET_IS_0] = F::ONE,
                1 => wit[COL_LINEAR_MEM_OFFSET_IS_1] = F::ONE,
                2 => wit[COL_LINEAR_MEM_OFFSET_IS_2] = F::ONE,
                3 => wit[COL_LINEAR_MEM_OFFSET_IS_3] = F::ONE,
                _ => {}
            }
            if access.width_bytes == 4 {
                wit[COL_LINEAR_MEM_IS_FULL_WIDTH] = F::ONE;
                match access.byte_offset {
                    0 => wit[COL_LINEAR_MEM_FULL_WIDTH_OFFSET_IS_0] = F::ONE,
                    1 => wit[COL_LINEAR_MEM_FULL_WIDTH_OFFSET_IS_1] = F::ONE,
                    2 => wit[COL_LINEAR_MEM_FULL_WIDTH_OFFSET_IS_2] = F::ONE,
                    3 => wit[COL_LINEAR_MEM_FULL_WIDTH_OFFSET_IS_3] = F::ONE,
                    _ => {}
                }
            } else if access.width_bytes == 8 {
                wit[COL_LINEAR_MEM_IS_DOUBLE_WIDTH] = F::ONE;
                match access.byte_offset {
                    0 => wit[COL_LINEAR_MEM_DOUBLE_WIDTH_OFFSET_IS_0] = F::ONE,
                    1 => wit[COL_LINEAR_MEM_DOUBLE_WIDTH_OFFSET_IS_1] = F::ONE,
                    2 => wit[COL_LINEAR_MEM_DOUBLE_WIDTH_OFFSET_IS_2] = F::ONE,
                    3 => wit[COL_LINEAR_MEM_DOUBLE_WIDTH_OFFSET_IS_3] = F::ONE,
                    _ => {}
                }
                match trace.opcode {
                    super::isa::WasmOpcode::I64Load => match access.byte_offset {
                        0 => wit[COL_LINEAR_MEM_I64_LOAD_OFFSET_IS_0] = F::ONE,
                        1 => wit[COL_LINEAR_MEM_I64_LOAD_OFFSET_IS_1] = F::ONE,
                        2 => wit[COL_LINEAR_MEM_I64_LOAD_OFFSET_IS_2] = F::ONE,
                        3 => wit[COL_LINEAR_MEM_I64_LOAD_OFFSET_IS_3] = F::ONE,
                        _ => {}
                    },
                    super::isa::WasmOpcode::I64Store => match access.byte_offset {
                        0 => wit[COL_LINEAR_MEM_I64_STORE_OFFSET_IS_0] = F::ONE,
                        1 => wit[COL_LINEAR_MEM_I64_STORE_OFFSET_IS_1] = F::ONE,
                        2 => wit[COL_LINEAR_MEM_I64_STORE_OFFSET_IS_2] = F::ONE,
                        3 => wit[COL_LINEAR_MEM_I64_STORE_OFFSET_IS_3] = F::ONE,
                        _ => {}
                    },
                    _ => {}
                }
            } else if access.width_bytes == 1 {
                wit[COL_LINEAR_MEM_IS_BYTE_WIDTH] = F::ONE;
                match access.byte_offset {
                    0 => wit[COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_0] = F::ONE,
                    1 => wit[COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_1] = F::ONE,
                    2 => wit[COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_2] = F::ONE,
                    3 => wit[COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_3] = F::ONE,
                    _ => {}
                }
            } else if access.width_bytes == 2 {
                wit[COL_LINEAR_MEM_IS_HALF_WIDTH] = F::ONE;
                match access.byte_offset {
                    0 => wit[COL_LINEAR_MEM_HALF_WIDTH_OFFSET_IS_0] = F::ONE,
                    1 => wit[COL_LINEAR_MEM_HALF_WIDTH_OFFSET_IS_1] = F::ONE,
                    2 => wit[COL_LINEAR_MEM_HALF_WIDTH_OFFSET_IS_2] = F::ONE,
                    3 => wit[COL_LINEAR_MEM_HALF_WIDTH_OFFSET_IS_3] = F::ONE,
                    _ => {}
                }
            }
        }
        wit[COL_LINEAR_MEM_LANE_ADDR[0]] = F::from_u64(access.lane0.word_addr);
        let lane0_value = match trace.opcode {
            super::isa::WasmOpcode::I32Load
            | super::isa::WasmOpcode::I64Load
            | super::isa::WasmOpcode::I32Load8S
            | super::isa::WasmOpcode::I32Load8U
            | super::isa::WasmOpcode::I32Load16S
            | super::isa::WasmOpcode::I32Load16U => access.lane0.value_before,
            super::isa::WasmOpcode::I32Store
            | super::isa::WasmOpcode::I64Store
            | super::isa::WasmOpcode::I32Store8
            | super::isa::WasmOpcode::I32Store16 => access.lane0.value_after,
            _ => access.lane0.value_after,
        };
        wit[COL_LINEAR_MEM_LANE_VALUE[0]] = F::from_u64(u64::from(lane0_value));
        write_u32_le_bytes(
            &mut wit,
            [
                COL_LINEAR_MEM_LANE0_BYTE0,
                COL_LINEAR_MEM_LANE0_BYTE1,
                COL_LINEAR_MEM_LANE0_BYTE2,
                COL_LINEAR_MEM_LANE0_BYTE3,
            ],
            lane0_value,
        );
        // Prior word state for the store-side RMW read and byte preservation.
        wit[COL_LINEAR_MEM_LANE_VALUE_BEFORE[0]] = F::from_u64(u64::from(access.lane0.value_before));
        write_u32_le_bytes(
            &mut wit,
            [
                COL_LINEAR_MEM_LANE0_BYTE0_BEFORE,
                COL_LINEAR_MEM_LANE0_BYTE1_BEFORE,
                COL_LINEAR_MEM_LANE0_BYTE2_BEFORE,
                COL_LINEAR_MEM_LANE0_BYTE3_BEFORE,
            ],
            access.lane0.value_before,
        );
        if let Some(lane1) = access.lane1 {
            wit[COL_LINEAR_MEM_LANE_ADDR[1]] = F::from_u64(lane1.word_addr);
            let lane1_value = match trace.opcode {
                super::isa::WasmOpcode::I32Load
                | super::isa::WasmOpcode::I64Load
                | super::isa::WasmOpcode::I32Load8S
                | super::isa::WasmOpcode::I32Load8U
                | super::isa::WasmOpcode::I32Load16S
                | super::isa::WasmOpcode::I32Load16U => lane1.value_before,
                super::isa::WasmOpcode::I32Store
                | super::isa::WasmOpcode::I64Store
                | super::isa::WasmOpcode::I32Store8
                | super::isa::WasmOpcode::I32Store16 => lane1.value_after,
                _ => lane1.value_after,
            };
            wit[COL_LINEAR_MEM_LANE_VALUE[1]] = F::from_u64(u64::from(lane1_value));
            write_u32_le_bytes(
                &mut wit,
                [
                    COL_LINEAR_MEM_LANE1_BYTE0,
                    COL_LINEAR_MEM_LANE1_BYTE1,
                    COL_LINEAR_MEM_LANE1_BYTE2,
                    COL_LINEAR_MEM_LANE1_BYTE3,
                ],
                lane1_value,
            );
            wit[COL_LINEAR_MEM_LANE_VALUE_BEFORE[1]] = F::from_u64(u64::from(lane1.value_before));
            write_u32_le_bytes(
                &mut wit,
                [
                    COL_LINEAR_MEM_LANE1_BYTE0_BEFORE,
                    COL_LINEAR_MEM_LANE1_BYTE1_BEFORE,
                    COL_LINEAR_MEM_LANE1_BYTE2_BEFORE,
                    COL_LINEAR_MEM_LANE1_BYTE3_BEFORE,
                ],
                lane1.value_before,
            );
        }
        if let Some(lane2) = access.lane2 {
            wit[COL_LINEAR_MEM_LANE_ADDR[2]] = F::from_u64(lane2.word_addr);
            let lane2_value = match trace.opcode {
                super::isa::WasmOpcode::I64Load => lane2.value_before,
                super::isa::WasmOpcode::I64Store => lane2.value_after,
                _ => lane2.value_after,
            };
            wit[COL_LINEAR_MEM_LANE_VALUE[2]] = F::from_u64(u64::from(lane2_value));
            write_u32_le_bytes(
                &mut wit,
                [
                    COL_LINEAR_MEM_LANE2_BYTE0,
                    COL_LINEAR_MEM_LANE2_BYTE1,
                    COL_LINEAR_MEM_LANE2_BYTE2,
                    COL_LINEAR_MEM_LANE2_BYTE3,
                ],
                lane2_value,
            );
            wit[COL_LINEAR_MEM_LANE_VALUE_BEFORE[2]] = F::from_u64(u64::from(lane2.value_before));
            write_u32_le_bytes(
                &mut wit,
                [
                    COL_LINEAR_MEM_LANE2_BYTE0_BEFORE,
                    COL_LINEAR_MEM_LANE2_BYTE1_BEFORE,
                    COL_LINEAR_MEM_LANE2_BYTE2_BEFORE,
                    COL_LINEAR_MEM_LANE2_BYTE3_BEFORE,
                ],
                lane2.value_before,
            );
        }
        // `access_bytes` is the direction-agnostic byte view of the
        // value being read or written: lo limb for both i32 and i64,
        // hi limb for i64 only. The byte-decomp constraints bind it
        // to `stack.write0_value{,_hi}` on loads and
        // `stack.read1_value{,_hi}` on stores.
        let (access_lo, access_hi) = match trace.opcode {
            super::isa::WasmOpcode::I32Load
            | super::isa::WasmOpcode::I32Load8S
            | super::isa::WasmOpcode::I32Load8U
            | super::isa::WasmOpcode::I32Load16S
            | super::isa::WasmOpcode::I32Load16U
            | super::isa::WasmOpcode::I64Load8U
            | super::isa::WasmOpcode::I64Load16U
            | super::isa::WasmOpcode::I64Load32U
            | super::isa::WasmOpcode::I64Load8S
            | super::isa::WasmOpcode::I64Load16S
            | super::isa::WasmOpcode::I64Load32S => (trace.stack_write0.map(|lane| lane.value_lo).unwrap_or(0), 0),
            super::isa::WasmOpcode::I32Store
            | super::isa::WasmOpcode::I32Store8
            | super::isa::WasmOpcode::I32Store16
            | super::isa::WasmOpcode::I64Store8
            | super::isa::WasmOpcode::I64Store16
            | super::isa::WasmOpcode::I64Store32 => (trace.stack_read1.map(|lane| lane.value_lo).unwrap_or(0), 0),
            super::isa::WasmOpcode::I64Load => (
                trace.stack_write0.map(|lane| lane.value_lo).unwrap_or(0),
                trace
                    .stack_write0
                    .and_then(|lane| lane.value_hi)
                    .unwrap_or(0),
            ),
            super::isa::WasmOpcode::I64Store => (
                trace.stack_read1.map(|lane| lane.value_lo).unwrap_or(0),
                trace
                    .stack_read1
                    .and_then(|lane| lane.value_hi)
                    .unwrap_or(0),
            ),
            _ if is_host_event_byte_memory => (
                u32::from(access.lane0.value_after.to_le_bytes()[usize::from(access.byte_offset)]),
                0,
            ),
            _ if is_host_event_half_memory => {
                let bytes = access.lane0.value_after.to_le_bytes();
                let offset = usize::from(access.byte_offset);
                (u32::from(u16::from_le_bytes([bytes[offset], bytes[offset + 1]])), 0)
            }
            _ => (0, 0),
        };
        write_u32_le_bytes(
            &mut wit,
            [
                COL_LINEAR_MEM_ACCESS_BYTE0,
                COL_LINEAR_MEM_ACCESS_BYTE1,
                COL_LINEAR_MEM_ACCESS_BYTE2,
                COL_LINEAR_MEM_ACCESS_BYTE3,
            ],
            access_lo,
        );
        write_u32_le_bytes(
            &mut wit,
            [
                COL_LINEAR_MEM_ACCESS_BYTE4,
                COL_LINEAR_MEM_ACCESS_BYTE5,
                COL_LINEAR_MEM_ACCESS_BYTE6,
                COL_LINEAR_MEM_ACCESS_BYTE7,
            ],
            access_hi,
        );
        // Sign-source byte index per signed load: byte 0 for *8_s, byte 1 for
        // *16_s, byte 3 for i64.load32_s (the top byte of the loaded word).
        let sign_source = match trace.opcode {
            super::isa::WasmOpcode::I32Load8S | super::isa::WasmOpcode::I64Load8S => Some(0),
            super::isa::WasmOpcode::I32Load16S | super::isa::WasmOpcode::I64Load16S => Some(1),
            super::isa::WasmOpcode::I64Load32S => Some(3),
            _ => None,
        };
        if let Some(byte_index) = sign_source {
            let byte = access_lo.to_le_bytes()[byte_index];
            wit[COL_SIGN_EXT_LOW7] = F::from_u64(u64::from(byte & 0x7f));
            wit[COL_SIGN_EXT_BIT] = if (byte & 0x80) != 0 { F::ONE } else { F::ZERO };
        }
    }
    let integer_sign_extend_source = match trace.opcode {
        super::isa::WasmOpcode::I32Extend8S | super::isa::WasmOpcode::I64Extend8S => Some(0),
        super::isa::WasmOpcode::I32Extend16S | super::isa::WasmOpcode::I64Extend16S => Some(1),
        super::isa::WasmOpcode::I64ExtendI32S | super::isa::WasmOpcode::I64Extend32S => Some(3),
        _ => None,
    };
    if let Some(byte_index) = integer_sign_extend_source {
        let value = trace.stack_read0.map(|lane| lane.value_lo).unwrap_or(0);
        write_u32_le_bytes(
            &mut wit,
            [
                COL_LINEAR_MEM_ACCESS_BYTE0,
                COL_LINEAR_MEM_ACCESS_BYTE1,
                COL_LINEAR_MEM_ACCESS_BYTE2,
                COL_LINEAR_MEM_ACCESS_BYTE3,
            ],
            value,
        );
        let sign_source_byte = value.to_le_bytes()[byte_index];
        wit[COL_SIGN_EXT_LOW7] = F::from_u64(u64::from(sign_source_byte & 0x7f));
        wit[COL_SIGN_EXT_BIT] = if (sign_source_byte & 0x80) != 0 {
            F::ONE
        } else {
            F::ZERO
        };
    }
    if let Some(op_table) = trace.info.op_table {
        wit[COL_OP_TABLE_ID] = F::from_u64(u64::from(op_table.op_table_id()));
        wit[COL_OP_TABLE_VALUE] = F::from_u64(
            trace
                .stack_write0
                .map(|w| u64::from(w.value_lo))
                .unwrap_or(0),
        );
    }

    if matches!(
        trace.opcode,
        super::isa::WasmOpcode::LocalGet | super::isa::WasmOpcode::LocalSet | super::isa::WasmOpcode::LocalTee
    ) || trace.row_kind.is_call_param_init()
        // Bootstrap gather rows write runtime inputs into the locals family.
        || trace.row_kind.is_host_event_gather()
    {
        if let Some(idx) = trace.local_index {
            wit[COL_LOCAL_INDEX] = F::from_u64(u64::from(idx));
        }
        let local_value_lo = trace
            .local_read_value
            .or(trace.local_write_value)
            .unwrap_or(0);
        wit[COL_LOCAL_VALUE] = F::from_u64(u64::from(local_value_lo));
        let local_value_hi = trace
            .local_read_value_hi
            .or(trace.local_write_value_hi)
            .unwrap_or(0);
        wit[COL_LOCAL_VALUE_HI] = F::from_u64(u64::from(local_value_hi));
    }

    if matches!(
        trace.opcode,
        super::isa::WasmOpcode::GlobalGet | super::isa::WasmOpcode::GlobalSet
    ) {
        if let Some(idx) = trace.global_index {
            wit[COL_GLOBAL_INDEX] = F::from_u64(u64::from(idx));
        }
        let global_value_lo = trace
            .global_read_value
            .or(trace.global_write_value)
            .unwrap_or(0);
        wit[COL_GLOBAL_VALUE] = F::from_u64(u64::from(global_value_lo));
        let global_value_hi = trace
            .global_read_value_hi
            .or(trace.global_write_value_hi)
            .unwrap_or(0);
        wit[COL_GLOBAL_VALUE_HI] = F::from_u64(u64::from(global_value_hi));
    }

    if let Some(table_id) = trace.table_id {
        wit[COL_TABLE_ID] = F::from_u64(u64::from(table_id));
    }
    if let Some(table_index) = trace.table_index {
        wit[COL_TABLE_INDEX] = F::from_u64(u64::from(table_index));
    }
    if let Some(table_value) = trace.table_value {
        wit[COL_TABLE_VALUE] = F::from_u64(u64::from(table_value));
    }
    if let Some(table_size) = trace.table_size {
        wit[COL_TABLE_SIZE] = F::from_u64(u64::from(table_size));
    }
    if let Some(function_ref) = trace.function_ref {
        wit[COL_FUNCTION_REF] = F::from_u64(u64::from(function_ref));
    }
    wit[COL_TARGET_FUNCTION_IS_GUEST] = if trace.target_function_is_guest {
        F::ONE
    } else {
        F::ZERO
    };
    if let Some(param_count) = trace.call_param_count {
        wit[COL_CALL_PARAM_COUNT] = F::from_u64(u64::from(param_count));
    }
    if let Some(result_count) = trace.call_result_count {
        wit[COL_CALL_RESULT_COUNT] = F::from_u64(u64::from(result_count));
    }
    wit[COL_CALL_TARGET_METADATA] = F::from_u64(pack_function_call_metadata(
        trace.call_param_count.unwrap_or(0),
        trace.call_result_count.unwrap_or(0),
        trace.target_function_is_guest,
    ));
    if let Some(function_type_id) = trace.function_type_id {
        wit[COL_FUNCTION_TYPE_ID] = F::from_u64(u64::from(function_type_id));
    }
    if let Some(type_index) = trace.call_indirect_type_index {
        wit[COL_CALL_INDIRECT_TYPE_INDEX] = F::from_u64(u64::from(type_index));
    }
    if let Some(expected_type_id) = trace.expected_type_id {
        wit[COL_EXPECTED_TYPE_ID] = F::from_u64(u64::from(expected_type_id));
    }

    // Mirror the CCS call_indirect trap gates (OOB index / null table entry /
    // callee type mismatch) and the derived read-activation gates. Placed
    // after the table/type columns above, which the comparison and zero tests
    // read.
    let sel_call_indirect = wit[selector_col(super::isa::WasmOpcode::CallIndirect).expect("call_indirect selector")]
        + wit[selector_col(super::isa::WasmOpcode::ReturnCallIndirect).expect("return_call_indirect selector")];
    // ge = (table_index >= table_size); the gadget is gated on call_indirect,
    // so the columns only carry meaning (and ci_oob) on those rows.
    if sel_call_indirect == F::ONE {
        let (cmp_low, cmp_ge) = unsigned_ge_witness(
            trace
                .table_index
                .expect("call_indirect binds a table index")
                .into(),
            trace
                .table_size
                .expect("call_indirect binds a table size")
                .into(),
        );
        wit[COL_CMP_LOW] = cmp_low;
        wit[COL_CMP_GE] = cmp_ge;
    }
    let ci_oob = sel_call_indirect * wit[COL_CMP_GE];
    wit[COL_CI_OOB] = ci_oob;
    // No table entry exists at an OOB index: the entry read is de-gated.
    wit[COL_TABLE_READ_ENABLED] -= ci_oob;

    let (entry_is_null, entry_null_inv) = zero_test_witness_field(wit[COL_TABLE_VALUE]);
    wit[COL_CI_ENTRY_IS_NULL] = entry_is_null;
    wit[COL_CI_ENTRY_NULL_INV] = entry_null_inv;
    let (type_eq, type_eq_inv) = zero_test_witness_field(wit[COL_FUNCTION_TYPE_ID] - wit[COL_EXPECTED_TYPE_ID]);
    wit[COL_CI_TYPE_EQ] = type_eq;
    wit[COL_CI_TYPE_EQ_INV] = type_eq_inv;

    // A call_indirect is clean exactly when it has a live non-null table entry
    // and that entry's callee type matches the instruction's expected type.
    let ci_type_lookup = (sel_call_indirect - ci_oob) * (F::ONE - entry_is_null);
    let ci_trap = sel_call_indirect - (ci_type_lookup * type_eq);
    wit[COL_CALL_INDIRECT_IS_TRAP] = ci_trap;
    wit[COL_FUNCTION_CALL_TYPE_LOOKUP_GATE] = ci_type_lookup;
    wit[COL_CALL_INDIRECT_IS_NOT_TRAP] = sel_call_indirect * (F::ONE - ci_trap);
    // Indirect host call: binds pc_after through the call site's return-pc
    // pc-ROM slot, so it pins the same choice a guest push would use.
    wit[COL_CI_HOST_CALL] = wit[COL_CALL_INDIRECT_IS_NOT_TRAP] * (F::ONE - wit[COL_TARGET_FUNCTION_IS_GUEST]);
    if wit[COL_CI_HOST_CALL] == F::ONE {
        wit[COL_PC_ROM_CALL_RETURN_CHOICE] = F::from_u64(PC_ROM_CALL_RETURN_CHOICE);
    }

    fill_event_absorb(&mut wit, trace);
    crate::ccs::host_event_chain::fill_witness(&mut wit, trace);

    match trace.opcode {
        super::isa::WasmOpcode::I32Add => {
            let lhs = u64::from(trace.stack_read0.map(|lane| lane.value_lo).unwrap_or(0));
            let rhs = u64::from(trace.stack_read1.map(|lane| lane.value_lo).unwrap_or(0));
            let carry = (lhs + rhs) >> 32;
            wit[COL_WIDE_AUX0] = F::from_u64(carry);
        }
        super::isa::WasmOpcode::I32Sub => {
            let lhs = trace.stack_read0.map(|lane| lane.value_lo).unwrap_or(0);
            let rhs = trace.stack_read1.map(|lane| lane.value_lo).unwrap_or(0);
            let borrow = u64::from(lhs < rhs);
            wit[COL_WIDE_AUX0] = F::from_u64(borrow);
        }
        super::isa::WasmOpcode::I64Add => {
            let lhs_lo = u64::from(trace.stack_read0.map(|lane| lane.value_lo).unwrap_or(0));
            let rhs_lo = u64::from(trace.stack_read1.map(|lane| lane.value_lo).unwrap_or(0));
            let lhs_hi = u64::from(
                trace
                    .stack_read0
                    .and_then(|lane| lane.value_hi)
                    .unwrap_or(0),
            );
            let rhs_hi = u64::from(
                trace
                    .stack_read1
                    .and_then(|lane| lane.value_hi)
                    .unwrap_or(0),
            );
            let carry0 = (lhs_lo + rhs_lo) >> 32;
            let carry1 = (lhs_hi + rhs_hi + carry0) >> 32;
            wit[COL_WIDE_AUX0] = F::from_u64(carry0);
            wit[COL_WIDE_AUX1] = F::from_u64(carry1);
        }
        super::isa::WasmOpcode::I64Sub => {
            let lhs_lo = trace.stack_read0.map(|lane| lane.value_lo).unwrap_or(0);
            let rhs_lo = trace.stack_read1.map(|lane| lane.value_lo).unwrap_or(0);
            let lhs_hi = trace
                .stack_read0
                .and_then(|lane| lane.value_hi)
                .unwrap_or(0);
            let rhs_hi = trace
                .stack_read1
                .and_then(|lane| lane.value_hi)
                .unwrap_or(0);
            let borrow0 = u64::from(lhs_lo < rhs_lo);
            let borrow1 = u64::from(u64::from(lhs_hi) < u64::from(rhs_hi) + borrow0);
            wit[COL_WIDE_AUX0] = F::from_u64(borrow0);
            wit[COL_WIDE_AUX1] = F::from_u64(borrow1);
        }
        _ => {}
    }

    let select_cond = trace.stack_read2.map(|lane| lane.value_lo).unwrap_or(0);
    let (select_cond_is_zero, select_cond_inv) = zero_test_witness_u64(u64::from(select_cond));
    let select_lhs = F::from_u64(u64::from(trace.stack_read0.map(|lane| lane.value_lo).unwrap_or(0)));
    let select_rhs = F::from_u64(u64::from(trace.stack_read1.map(|lane| lane.value_lo).unwrap_or(0)));
    let select_lhs_hi = F::from_u64(u64::from(
        trace
            .stack_read0
            .and_then(|lane| lane.value_hi)
            .unwrap_or(0),
    ));
    let select_rhs_hi = F::from_u64(u64::from(
        trace
            .stack_read1
            .and_then(|lane| lane.value_hi)
            .unwrap_or(0),
    ));
    wit[COL_SELECT_COND_IS_ZERO] = select_cond_is_zero;
    wit[COL_SELECT_SCRATCH_INV] = select_cond_inv;
    wit[COL_SELECT_OUT_DELTA_LO] = (F::ONE - select_cond_is_zero) * (select_lhs - select_rhs);
    wit[COL_SELECT_OUT_DELTA_HI] = (F::ONE - select_cond_is_zero) * (select_lhs_hi - select_rhs_hi);

    // Comparator zero-test scratch (see `push_comparator_constraints` in ccs.rs).
    // For non-comparator opcodes both diffs are 0 and the gadget pins both
    // is_zero flags to 1, so cmp_and = 1.
    let cmp_lo_diff = match trace.opcode {
        super::isa::WasmOpcode::I32Eqz | super::isa::WasmOpcode::I64Eqz => {
            F::from_u64(u64::from(trace.stack_read0.map(|l| l.value_lo).unwrap_or(0)))
        }
        super::isa::WasmOpcode::I32Eq
        | super::isa::WasmOpcode::I32Ne
        | super::isa::WasmOpcode::I64Eq
        | super::isa::WasmOpcode::I64Ne => {
            let lhs = F::from_u64(u64::from(trace.stack_read0.map(|l| l.value_lo).unwrap_or(0)));
            let rhs = F::from_u64(u64::from(trace.stack_read1.map(|l| l.value_lo).unwrap_or(0)));
            lhs - rhs
        }
        _ => F::ZERO,
    };
    let cmp_hi_diff = match trace.opcode {
        super::isa::WasmOpcode::I64Eqz => F::from_u64(u64::from(
            trace
                .stack_read0
                .and_then(|lane| lane.value_hi)
                .unwrap_or(0),
        )),
        super::isa::WasmOpcode::I64Eq | super::isa::WasmOpcode::I64Ne => {
            let lhs = F::from_u64(u64::from(
                trace
                    .stack_read0
                    .and_then(|lane| lane.value_hi)
                    .unwrap_or(0),
            ));
            let rhs = F::from_u64(u64::from(
                trace
                    .stack_read1
                    .and_then(|lane| lane.value_hi)
                    .unwrap_or(0),
            ));
            lhs - rhs
        }
        _ => F::ZERO,
    };
    let (cmp_lo_is_zero, cmp_lo_inv) = zero_test_witness_field(cmp_lo_diff);
    let (cmp_hi_is_zero, cmp_hi_inv) = zero_test_witness_field(cmp_hi_diff);
    wit[COL_CMP_LO_DIFF] = cmp_lo_diff;
    wit[COL_CMP_LO_INV] = cmp_lo_inv;
    wit[COL_CMP_LO_IS_ZERO] = cmp_lo_is_zero;
    wit[COL_CMP_HI_DIFF] = cmp_hi_diff;
    wit[COL_CMP_HI_INV] = cmp_hi_inv;
    wit[COL_CMP_HI_IS_ZERO] = cmp_hi_is_zero;
    wit[COL_CMP_AND] = cmp_lo_is_zero * cmp_hi_is_zero;

    crate::range_check::write_range_check_bits(&mut wit);
    wit
}

/// Host-event absorb machinery witness: carried state columns, the perm-row
/// position one-hot, gather decoding, and the S-box power columns (whose unconditional mult rows are witness-filled
/// with the powers of their linear input expression on every row).
/// Fill the named host-event absorb interface columns (carried state); the
/// gadget-internal block is filled by `ccs::host_event_chain::fill_witness`.
fn fill_event_absorb(wit: &mut [F], trace: &WasmVmStep) {
    use crate::layout::{
        COL_EVBUF_AFTER, COL_EVBUF_BEFORE, COL_GATHER_ACTIVE, COL_GATHER_LOCAL_WRITE, COL_GATHER_LOCAL_WRITE_LO,
        COL_HOST_EVENTS_REMAINING_AFTER, COL_HOST_EVENTS_REMAINING_BEFORE, COL_HOST_EVENTS_REMAINING_BEFORE_INV,
        COL_HOST_EVENTS_REMAINING_BEFORE_IS_ZERO, COL_HOST_EVENT_ARGS_BASE_AFTER, COL_HOST_EVENT_ARGS_BASE_BEFORE,
        COL_HOST_EVENT_EXIT_LATCH, COL_HOST_EVENT_EXIT_SCHEDULE_COUNT, COL_HOST_EVENT_INDEX_AFTER,
        COL_HOST_EVENT_INDEX_BEFORE, COL_HOST_EVENT_INITIAL_SCHEDULE_COUNT, COL_HOST_EVENT_SLOT_ARG,
        COL_HOST_EVENT_SLOT_CURSOR_AFTER, COL_HOST_EVENT_SLOT_CURSOR_BEFORE, COL_HOST_EVENT_SLOT_IMMEDIATE0,
        COL_HOST_EVENT_SLOT_IMMEDIATE1, COL_HOST_EVENT_SLOT_KIND, COL_HOST_EVENT_SLOT_VARIANT, COL_PERM_PENDING_AFTER,
        COL_PERM_PENDING_BEFORE, COL_PERM_ROUND_AFTER, COL_PERM_ROUND_BEFORE, COL_PERM_ROUND_BEFORE_INV,
        COL_PERM_ROUND_BEFORE_IS_ZERO, COL_PERM_STATE_AFTER, COL_PERM_STATE_BEFORE,
    };

    let bool_f = |flag: bool| if flag { F::ONE } else { F::ZERO };
    let before = trace.state_before.event_absorb;
    let after = trace.state_after.event_absorb;

    for j in 0..8 {
        wit[COL_EVBUF_BEFORE[j]] = F::from_u64(before.evbuf[j]);
        wit[COL_EVBUF_AFTER[j]] = F::from_u64(after.evbuf[j]);
    }
    wit[COL_PERM_PENDING_BEFORE] = bool_f(before.perm_pending);
    wit[COL_PERM_PENDING_AFTER] = bool_f(after.perm_pending);
    wit[COL_PERM_ROUND_BEFORE] = F::from_u64(u64::from(before.perm_round));
    wit[COL_PERM_ROUND_AFTER] = F::from_u64(u64::from(after.perm_round));
    let (round_is_zero, round_inv) = zero_test_witness_u64(u64::from(before.perm_round));
    wit[COL_PERM_ROUND_BEFORE_IS_ZERO] = round_is_zero;
    wit[COL_PERM_ROUND_BEFORE_INV] = round_inv;
    for lane in 0..12 {
        wit[COL_PERM_STATE_BEFORE[lane]] = F::from_u64(before.perm_state[lane]);
        wit[COL_PERM_STATE_AFTER[lane]] = F::from_u64(after.perm_state[lane]);
    }

    // Gather and host-call interface gates read by the gadget.
    wit[COL_GATHER_ACTIVE] = bool_f(trace.row_kind.is_host_event_gather());
    wit[crate::layout::COL_PC_FREF_ACTIVE] = bool_f(
        !trace.row_kind.is_host_event_gather()
            && !trace.row_kind.is_turn_boundary()
            && !trace.row_kind.is_padding()
            && trace.state_before.event_absorb.perm_round == 0
            && !trace.state_before.event_absorb.perm_pending,
    );
    let host_call_gate = wit[selector_col(super::isa::WasmOpcode::Call).expect("call selector")]
        + wit[selector_col(super::isa::WasmOpcode::ReturnCall).expect("return_call selector")]
        + wit[COL_CALL_INDIRECT_IS_NOT_TRAP]
        - wit[COL_GUEST_ENTRY_ACTIVE];
    wit[crate::layout::COL_HOST_CALL_ACTIVE] = host_call_gate;
    wit[COL_GATHER_LOCAL_WRITE] = if trace.row_kind.is_host_event_gather()
        && trace
            .host_event_rom_slot
            .is_some_and(|rom| rom.kind == WasmHostEventSlotKind::InputLocal)
    {
        F::ONE
    } else {
        F::ZERO
    };
    wit[COL_GATHER_LOCAL_WRITE_LO] = if trace.row_kind.is_host_event_gather()
        && trace
            .host_event_rom_slot
            .is_some_and(|rom| rom.kind == WasmHostEventSlotKind::InputLocal && rom.variant.is_low_limb())
    {
        F::ONE
    } else {
        F::ZERO
    };
    // Hi-word stack write port gate: ordinary write0 activity, plus
    // result-hi gather rows (which write only the pushed cell's hi lane).
    let result_hi_gather = trace.row_kind.is_host_event_gather()
        && trace
            .host_event_rom_slot
            .is_some_and(|rom| rom.kind == WasmHostEventSlotKind::Result && rom.variant.is_high_limb());
    wit[crate::layout::COL_STACK_WRITE0_HI_ACTIVE] =
        wit[crate::layout::COL_STACK_WRITE0_ACTIVE] + bool_f(result_hi_gather);
    wit[COL_HOST_EVENT_EXIT_LATCH] =
        bool_f(!trace.state_before.halted && trace.state_after.halted && !trace.state_after.trapped);
    wit[crate::layout::COL_TURN_BOUNDARY] = bool_f(trace.row_kind.is_turn_boundary());

    // Host-event gather machinery: carried schedule/cursor state plus
    // the per-row host-event ROM interface columns.
    let g_before = trace.state_before.host_events;
    let g_after = trace.state_after.host_events;
    wit[COL_HOST_EVENTS_REMAINING_BEFORE] = F::from_u64(u64::from(g_before.events_remaining));
    wit[COL_HOST_EVENTS_REMAINING_AFTER] = F::from_u64(u64::from(g_after.events_remaining));
    let (evrem_is_zero, evrem_inv) = zero_test_witness_u64(u64::from(g_before.events_remaining));
    wit[COL_HOST_EVENTS_REMAINING_BEFORE_IS_ZERO] = evrem_is_zero;
    wit[COL_HOST_EVENTS_REMAINING_BEFORE_INV] = evrem_inv;
    wit[COL_HOST_EVENT_INDEX_BEFORE] = F::from_u64(u64::from(g_before.event_index));
    wit[COL_HOST_EVENT_INDEX_AFTER] = F::from_u64(u64::from(g_after.event_index));
    wit[COL_HOST_EVENT_ARGS_BASE_BEFORE] = F::from_u64(g_before.args_base);
    wit[COL_HOST_EVENT_ARGS_BASE_AFTER] = F::from_u64(g_after.args_base);
    wit[COL_HOST_EVENT_SLOT_CURSOR_BEFORE] = F::from_u64(u64::from(g_before.slot_cursor));
    wit[COL_HOST_EVENT_SLOT_CURSOR_AFTER] = F::from_u64(u64::from(g_after.slot_cursor));
    if let Some(rom) = trace.host_event_rom_slot {
        wit[COL_HOST_EVENT_SLOT_KIND] =
            F::from_u64(u64::from(rom.kind.code()) + WasmHostEventSlotKind::COUNT as u64 * u64::from(rom.advice));
        wit[COL_HOST_EVENT_SLOT_ARG] = F::from_u64(u64::from(rom.arg));
        wit[COL_HOST_EVENT_SLOT_VARIANT] = F::from_u64(u64::from(rom.variant.encoded()));
        wit[COL_HOST_EVENT_SLOT_IMMEDIATE0] = F::from_u64(u64::from(rom.immediate0));
        wit[COL_HOST_EVENT_SLOT_IMMEDIATE1] = F::from_u64(u64::from(rom.immediate1));
    }
    if let Some(count) = trace.host_event_initial_schedule_count {
        wit[COL_HOST_EVENT_INITIAL_SCHEDULE_COUNT] = F::from_u64(u64::from(count));
    }
    if let Some(count) = trace.host_event_exit_schedule_count {
        wit[COL_HOST_EVENT_EXIT_SCHEDULE_COUNT] = F::from_u64(u64::from(count));
    }
}

fn write_param_init_state(wit: &mut [F], before: bool, state: super::ir::WasmCountdownState) {
    let (active_col, remaining_col) = if before {
        (COL_PARAM_INIT_ACTIVE_BEFORE, COL_PARAM_INIT_REMAINING_BEFORE)
    } else {
        (COL_PARAM_INIT_ACTIVE_AFTER, COL_PARAM_INIT_REMAINING_AFTER)
    };
    wit[active_col] = if state.active { F::ONE } else { F::ZERO };
    wit[remaining_col] = F::from_u64(u64::from(state.remaining));
}

fn write_u32_le_bytes(wit: &mut [F], columns: [usize; 4], value: u32) {
    let bytes = value.to_le_bytes();
    for (column, byte) in columns.into_iter().zip(bytes) {
        wit[column] = F::from_u64(u64::from(byte));
    }
}
