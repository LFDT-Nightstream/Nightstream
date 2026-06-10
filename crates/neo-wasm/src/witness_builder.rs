//! Owns packaging normalized WASM rows into `StepBuild`.

use super::gadgets::{zero_test_witness_field, zero_test_witness_u64};
use super::ir::{WasmRowKind, WasmStepTrace};
use super::layout::{
    selector_col, CALL_RETURN_PC_CHOICE, COL_CALL_INDIRECT_TYPE_INDEX, COL_CALL_PARAM_COUNT, COL_CALL_RESULT_COUNT,
    COL_CALL_STACK_ADDR, COL_CALL_STACK_DEPTH_AFTER, COL_CALL_STACK_DEPTH_BEFORE, COL_CALL_STACK_POP_CALLER_FBP,
    COL_CALL_STACK_POP_PRESENT, COL_CALL_STACK_POP_RETURN_PC, COL_CALL_STACK_PUSH_PRESENT,
    COL_CALL_STACK_RETURN_PC_CHOICE, COL_CONTROL_CHOICE, COL_CURRENT_FUNCTION_NUM_LOCALS, COL_CURRENT_FUNCTION_REF,
    COL_EXPECTED_TYPE_ID, COL_FUNCTION_REF, COL_FUNCTION_TYPE_ID, COL_GLOBAL_INDEX, COL_GLOBAL_VALUE,
    COL_GLOBAL_VALUE_HI, COL_HALTED, COL_IS_PROGRAM_ROW, COL_LINEAR_MEM_ACCESS_BYTE0, COL_LINEAR_MEM_ACCESS_BYTE1,
    COL_LINEAR_MEM_ACCESS_BYTE2, COL_LINEAR_MEM_ACCESS_BYTE3, COL_LINEAR_MEM_ACCESS_BYTE4, COL_LINEAR_MEM_ACCESS_BYTE5,
    COL_LINEAR_MEM_ACCESS_BYTE6, COL_LINEAR_MEM_ACCESS_BYTE7, COL_LINEAR_MEM_BYTE_OFFSET,
    COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_0, COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_1,
    COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_2, COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_3,
    COL_LINEAR_MEM_DOUBLE_WIDTH_OFFSET_IS_0, COL_LINEAR_MEM_DOUBLE_WIDTH_OFFSET_IS_1,
    COL_LINEAR_MEM_DOUBLE_WIDTH_OFFSET_IS_2, COL_LINEAR_MEM_DOUBLE_WIDTH_OFFSET_IS_3,
    COL_LINEAR_MEM_FULL_WIDTH_OFFSET_IS_0, COL_LINEAR_MEM_FULL_WIDTH_OFFSET_IS_1,
    COL_LINEAR_MEM_FULL_WIDTH_OFFSET_IS_2, COL_LINEAR_MEM_FULL_WIDTH_OFFSET_IS_3,
    COL_LINEAR_MEM_HALF_WIDTH_OFFSET_IS_0, COL_LINEAR_MEM_HALF_WIDTH_OFFSET_IS_1,
    COL_LINEAR_MEM_HALF_WIDTH_OFFSET_IS_2, COL_LINEAR_MEM_HALF_WIDTH_OFFSET_IS_3, COL_LINEAR_MEM_I64_LOAD_OFFSET_IS_0,
    COL_LINEAR_MEM_I64_LOAD_OFFSET_IS_1, COL_LINEAR_MEM_I64_LOAD_OFFSET_IS_2, COL_LINEAR_MEM_I64_LOAD_OFFSET_IS_3,
    COL_LINEAR_MEM_I64_STORE_OFFSET_IS_0, COL_LINEAR_MEM_I64_STORE_OFFSET_IS_1, COL_LINEAR_MEM_I64_STORE_OFFSET_IS_2,
    COL_LINEAR_MEM_I64_STORE_OFFSET_IS_3, COL_LINEAR_MEM_IMM_OFFSET, COL_LINEAR_MEM_IS_BYTE_WIDTH,
    COL_LINEAR_MEM_IS_DOUBLE_WIDTH, COL_LINEAR_MEM_IS_FULL_WIDTH, COL_LINEAR_MEM_IS_HALF_WIDTH,
    COL_LINEAR_MEM_LANE0_ADDR, COL_LINEAR_MEM_LANE0_BYTE0, COL_LINEAR_MEM_LANE0_BYTE0_BEFORE,
    COL_LINEAR_MEM_LANE0_BYTE1, COL_LINEAR_MEM_LANE0_BYTE1_BEFORE, COL_LINEAR_MEM_LANE0_BYTE2,
    COL_LINEAR_MEM_LANE0_BYTE2_BEFORE, COL_LINEAR_MEM_LANE0_BYTE3, COL_LINEAR_MEM_LANE0_BYTE3_BEFORE,
    COL_LINEAR_MEM_LANE0_LOAD_ACTIVE, COL_LINEAR_MEM_LANE0_STORE_ACTIVE, COL_LINEAR_MEM_LANE0_VALUE,
    COL_LINEAR_MEM_LANE0_VALUE_BEFORE, COL_LINEAR_MEM_LANE1_ADDR, COL_LINEAR_MEM_LANE1_BYTE0,
    COL_LINEAR_MEM_LANE1_BYTE0_BEFORE, COL_LINEAR_MEM_LANE1_BYTE1, COL_LINEAR_MEM_LANE1_BYTE1_BEFORE,
    COL_LINEAR_MEM_LANE1_BYTE2, COL_LINEAR_MEM_LANE1_BYTE2_BEFORE, COL_LINEAR_MEM_LANE1_BYTE3,
    COL_LINEAR_MEM_LANE1_BYTE3_BEFORE, COL_LINEAR_MEM_LANE1_LOAD_ACTIVE, COL_LINEAR_MEM_LANE1_STORE_ACTIVE,
    COL_LINEAR_MEM_LANE1_VALUE, COL_LINEAR_MEM_LANE1_VALUE_BEFORE, COL_LINEAR_MEM_LANE2_ADDR,
    COL_LINEAR_MEM_LANE2_BYTE0, COL_LINEAR_MEM_LANE2_BYTE0_BEFORE, COL_LINEAR_MEM_LANE2_BYTE1,
    COL_LINEAR_MEM_LANE2_BYTE1_BEFORE, COL_LINEAR_MEM_LANE2_BYTE2, COL_LINEAR_MEM_LANE2_BYTE2_BEFORE,
    COL_LINEAR_MEM_LANE2_BYTE3, COL_LINEAR_MEM_LANE2_BYTE3_BEFORE, COL_LINEAR_MEM_LANE2_LOAD_ACTIVE,
    COL_LINEAR_MEM_LANE2_STORE_ACTIVE, COL_LINEAR_MEM_LANE2_VALUE, COL_LINEAR_MEM_LANE2_VALUE_BEFORE,
    COL_LINEAR_MEM_OFFSET_IS_0, COL_LINEAR_MEM_OFFSET_IS_1, COL_LINEAR_MEM_OFFSET_IS_2, COL_LINEAR_MEM_OFFSET_IS_3,
    COL_LINEAR_MEM_USE_LANE0, COL_LINEAR_MEM_USE_LANE1, COL_LINEAR_MEM_USE_LANE2, COL_LOCALS_FBP_AFTER,
    COL_LOCALS_FBP_BEFORE, COL_LOCAL_INDEX, COL_LOCAL_VALUE, COL_LOCAL_VALUE_HI, COL_LOCAL_WRITE_ENABLED,
    COL_MEMORY_PAGES_AFTER, COL_MEMORY_PAGES_BEFORE, COL_ONE, COL_OPCODE_CODE, COL_OP_TABLE_ENABLED, COL_OP_TABLE_ID,
    COL_OP_TABLE_VALUE, COL_OUTPUT_CAPTURED, COL_OUTPUT_ENABLED_AFTER, COL_OUTPUT_ENABLED_BEFORE,
    COL_OUTPUT_VALUE_HI_AFTER, COL_OUTPUT_VALUE_HI_BEFORE, COL_OUTPUT_VALUE_LO_AFTER, COL_OUTPUT_VALUE_LO_BEFORE,
    COL_PADDING_ACTIVE, COL_PARAM_INIT_ACTIVE_AFTER, COL_PARAM_INIT_ACTIVE_BEFORE, COL_PARAM_INIT_REMAINING_AFTER,
    COL_PARAM_INIT_REMAINING_AFTER_INV, COL_PARAM_INIT_REMAINING_AFTER_IS_ZERO, COL_PARAM_INIT_REMAINING_BEFORE,
    COL_PC_AFTER, COL_PC_BEFORE, COL_PC_EDGE_KIND, COL_PC_EDGE_KIND_INV, COL_PC_EDGE_KIND_IS_STATIC, COL_PC_ROM_ACTIVE,
    COL_SELECT_OUT_DELTA_HI, COL_SELECT_OUT_DELTA_LO, COL_SIGN_EXT_BIT, COL_SIGN_EXT_LOW7, COL_SP_AFTER, COL_SP_BEFORE,
    COL_STACK_READ0_ACTIVE, COL_STACK_READ0_ADDR_HI, COL_STACK_READ0_ADDR_LO, COL_STACK_READ0_VALUE_HI,
    COL_STACK_READ0_VALUE_LO, COL_STACK_READ1_ACTIVE, COL_STACK_READ1_ADDR_HI, COL_STACK_READ1_ADDR_LO,
    COL_STACK_READ1_VALUE_HI, COL_STACK_READ1_VALUE_LO, COL_STACK_READ2_ACTIVE, COL_STACK_READ2_ADDR_HI,
    COL_STACK_READ2_ADDR_LO, COL_STACK_READ2_VALUE_HI, COL_STACK_READ2_VALUE_LO, COL_STACK_READS,
    COL_STACK_WRITE0_ACTIVE, COL_STACK_WRITE0_ADDR_HI, COL_STACK_WRITE0_ADDR_LO, COL_STACK_WRITE0_VALUE_HI,
    COL_STACK_WRITE0_VALUE_LO, COL_STACK_WRITES, COL_TABLE_ID, COL_TABLE_INDEX, COL_TABLE_READ_ENABLED, COL_TABLE_SIZE,
    COL_TABLE_VALUE, COL_TARGET_FUNCTION_IS_GUEST, COL_WIDE_AUX0, COL_WIDE_AUX1, COL_WIDE_VALUES_ENABLED,
    WITNESS_WIDTH,
};
use super::step_build::WasmStepBuild;
use crate::layout::{
    COL_CMP_AND, COL_CMP_HI_DIFF, COL_CMP_HI_INV, COL_CMP_HI_IS_ZERO, COL_CMP_LO_DIFF, COL_CMP_LO_INV,
    COL_CMP_LO_IS_ZERO, COL_SELECT_COND_IS_ZERO, COL_SELECT_SCRATCH_INV,
};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

/// Build one R1CS-satisfying assignment per normalized wasm step.
///
/// The R1CS-F' chain builder in `neo-fold-clean` bit-decomposes each
/// assignment during `compile_step` and constructs the foldable F'-encoded
/// `CcsInstance` internally — neo-wasm does not commit to the assignment.
pub fn build_steps(steps: &[WasmStepTrace]) -> Vec<WasmStepBuild> {
    steps
        .iter()
        .map(|step| WasmStepBuild {
            assignment: build_witness_vector(step),
        })
        .collect()
}

pub fn build_witness_vector(trace: &WasmStepTrace) -> Vec<F> {
    let mut wit = vec![F::ZERO; WITNESS_WIDTH];
    wit[COL_ONE] = F::ONE;
    // High-limb stack addresses are constrained unconditionally as
    // `addr_hi = addr_lo + 1`. Inactive low addresses default to 0 and
    // inactive memory specs are gated off, so 1 is the canonical inactive
    // high address. Active lanes overwrite this below with `addr + 1`.
    wit[COL_STACK_READ0_ADDR_HI] = F::ONE;
    wit[COL_STACK_READ1_ADDR_HI] = F::ONE;
    wit[COL_STACK_READ2_ADDR_HI] = F::ONE;
    wit[COL_STACK_WRITE0_ADDR_HI] = F::ONE;
    let opcode_code = if trace.row_kind.is_program() {
        trace.info.code
    } else {
        0
    };
    wit[COL_OPCODE_CODE] = F::from_u64(u64::from(opcode_code));
    wit[COL_PC_BEFORE] = F::from_u64(trace.state_before.pc);
    wit[COL_PC_AFTER] = F::from_u64(trace.state_after.pc);
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
    if let Some((return_pc, caller_fbp)) = trace.call_stack_push.or(trace.call_stack_pop) {
        wit[COL_CALL_STACK_POP_RETURN_PC] = F::from_u64(return_pc);
        wit[COL_CALL_STACK_POP_CALLER_FBP] = F::from_u64(caller_fbp);
    }
    if trace.call_stack_push.is_some() {
        wit[COL_CALL_STACK_ADDR] = F::from_u64(trace.state_before.call_stack_depth);
        wit[COL_CALL_STACK_RETURN_PC_CHOICE] = F::from_u64(CALL_RETURN_PC_CHOICE);
    } else if trace.call_stack_pop.is_some() {
        wit[COL_CALL_STACK_ADDR] = F::from_u64(trace.state_after.call_stack_depth);
    }
    if let Some(pages) = trace.state_before.memory_pages {
        wit[COL_MEMORY_PAGES_BEFORE] = F::from_u64(u64::from(pages));
    }
    if let Some(pages) = trace.state_after.memory_pages {
        wit[COL_MEMORY_PAGES_AFTER] = F::from_u64(u64::from(pages));
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
    wit[COL_STACK_READ0_ACTIVE] = if stack_reads >= 1 { F::ONE } else { F::ZERO };
    wit[COL_STACK_READ1_ACTIVE] = if stack_reads >= 2 { F::ONE } else { F::ZERO };
    wit[COL_STACK_READ2_ACTIVE] = if stack_reads >= 3 { F::ONE } else { F::ZERO };
    wit[COL_STACK_WRITE0_ACTIVE] = if stack_writes >= 1 { F::ONE } else { F::ZERO };
    wit[COL_OP_TABLE_ENABLED] = if trace.info.uses_op_table { F::ONE } else { F::ZERO };
    wit[COL_LINEAR_MEM_USE_LANE0] = if trace.linear_memory.is_some() { F::ONE } else { F::ZERO };
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
        super::isa::WasmOpcode::TableGet | super::isa::WasmOpcode::CallIndirect
    ) {
        F::ONE
    } else {
        F::ZERO
    };

    if matches!(trace.row_kind, WasmRowKind::Program) {
        if let Some(col) = selector_col(trace.opcode) {
            wit[col] = F::ONE;
        }
    }
    if let Some(read) = trace.stack_read0 {
        wit[COL_STACK_READ0_ADDR_LO] = F::from_u64(read.addr_lo);
        wit[COL_STACK_READ0_ADDR_HI] = F::from_u64(read.addr_lo + 1);
        wit[COL_STACK_READ0_VALUE_LO] = F::from_u64(u64::from(read.value_lo));
    }
    if trace.output_captured {
        debug_assert_eq!(
            trace.stack_reads_override.unwrap_or(trace.info.stack_reads),
            0,
            "output capture reuses inactive stack_read0 columns"
        );
        let output_addr = trace.state_before.sp.saturating_sub(1).saturating_mul(2);
        wit[COL_STACK_READ0_ADDR_LO] = F::from_u64(output_addr);
        wit[COL_STACK_READ0_ADDR_HI] = F::from_u64(output_addr + 1);
        wit[COL_STACK_READ0_VALUE_LO] = F::from_u64(u64::from(trace.state_after.output.value_lo));
        wit[COL_STACK_READ0_VALUE_HI] = F::from_u64(u64::from(trace.state_after.output.value_hi));
    }
    if trace.wide_values_enabled {
        if let Some(read0_value_hi) = trace.stack_read0.and_then(|read| read.value_hi) {
            wit[COL_STACK_READ0_VALUE_HI] = F::from_u64(u64::from(read0_value_hi));
        }
    }
    if let Some(read) = trace.stack_read1 {
        wit[COL_STACK_READ1_ADDR_LO] = F::from_u64(read.addr_lo);
        wit[COL_STACK_READ1_ADDR_HI] = F::from_u64(read.addr_lo + 1);
        wit[COL_STACK_READ1_VALUE_LO] = F::from_u64(u64::from(read.value_lo));
    }
    if trace.wide_values_enabled {
        if let Some(read1_value_hi) = trace.stack_read1.and_then(|read| read.value_hi) {
            wit[COL_STACK_READ1_VALUE_HI] = F::from_u64(u64::from(read1_value_hi));
        }
    }
    if let Some(read) = trace.stack_read2 {
        wit[COL_STACK_READ2_ADDR_LO] = F::from_u64(read.addr_lo);
        wit[COL_STACK_READ2_ADDR_HI] = F::from_u64(read.addr_lo + 1);
        wit[COL_STACK_READ2_VALUE_LO] = F::from_u64(u64::from(read.value_lo));
    }
    if trace.wide_values_enabled {
        if let Some(read2_value_hi) = trace.stack_read2.and_then(|read| read.value_hi) {
            wit[COL_STACK_READ2_VALUE_HI] = F::from_u64(u64::from(read2_value_hi));
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
    if let Some(access) = trace.linear_memory {
        wit[COL_LINEAR_MEM_IMM_OFFSET] = F::from_u64(trace.linear_memory_offset);
        wit[COL_LINEAR_MEM_BYTE_OFFSET] = F::from_u64(u64::from(access.byte_offset));
        wit[COL_LINEAR_MEM_USE_LANE1] = if access.lane1.is_some() { F::ONE } else { F::ZERO };
        wit[COL_LINEAR_MEM_USE_LANE2] = if access.lane2.is_some() { F::ONE } else { F::ZERO };
        // Witness the CCS-bound load/store lane gates used by the memory spec.
        let is_load = trace
            .opcode
            .memory_access_info()
            .is_some_and(|info| info.kind == super::isa::WasmMemoryAccessKind::Load);
        let is_store = trace
            .opcode
            .memory_access_info()
            .is_some_and(|info| info.kind == super::isa::WasmMemoryAccessKind::Store);
        let f_or = |cond: bool| if cond { F::ONE } else { F::ZERO };
        wit[COL_LINEAR_MEM_LANE0_LOAD_ACTIVE] = f_or(is_load);
        wit[COL_LINEAR_MEM_LANE1_LOAD_ACTIVE] = f_or(is_load && access.lane1.is_some());
        wit[COL_LINEAR_MEM_LANE2_LOAD_ACTIVE] = f_or(is_load && access.lane2.is_some());
        wit[COL_LINEAR_MEM_LANE0_STORE_ACTIVE] = f_or(is_store);
        wit[COL_LINEAR_MEM_LANE1_STORE_ACTIVE] = f_or(is_store && access.lane1.is_some());
        wit[COL_LINEAR_MEM_LANE2_STORE_ACTIVE] = f_or(is_store && access.lane2.is_some());
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
        wit[COL_LINEAR_MEM_LANE0_ADDR] = F::from_u64(access.lane0.word_addr);
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
        wit[COL_LINEAR_MEM_LANE0_VALUE] = F::from_u64(u64::from(lane0_value));
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
        wit[COL_LINEAR_MEM_LANE0_VALUE_BEFORE] = F::from_u64(u64::from(access.lane0.value_before));
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
            wit[COL_LINEAR_MEM_LANE1_ADDR] = F::from_u64(lane1.word_addr);
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
            wit[COL_LINEAR_MEM_LANE1_VALUE] = F::from_u64(u64::from(lane1_value));
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
            wit[COL_LINEAR_MEM_LANE1_VALUE_BEFORE] = F::from_u64(u64::from(lane1.value_before));
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
            wit[COL_LINEAR_MEM_LANE2_ADDR] = F::from_u64(lane2.word_addr);
            let lane2_value = match trace.opcode {
                super::isa::WasmOpcode::I64Load => lane2.value_before,
                super::isa::WasmOpcode::I64Store => lane2.value_after,
                _ => lane2.value_after,
            };
            wit[COL_LINEAR_MEM_LANE2_VALUE] = F::from_u64(u64::from(lane2_value));
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
            wit[COL_LINEAR_MEM_LANE2_VALUE_BEFORE] = F::from_u64(u64::from(lane2.value_before));
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
    if let Some(function_type_id) = trace.function_type_id {
        wit[COL_FUNCTION_TYPE_ID] = F::from_u64(u64::from(function_type_id));
    }
    if let Some(type_index) = trace.call_indirect_type_index {
        wit[COL_CALL_INDIRECT_TYPE_INDEX] = F::from_u64(u64::from(type_index));
    }
    if let Some(expected_type_id) = trace.expected_type_id {
        wit[COL_EXPECTED_TYPE_ID] = F::from_u64(u64::from(expected_type_id));
    }

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

    wit
}

fn write_param_init_state(wit: &mut [F], before: bool, state: super::ir::WasmParamInitState) {
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
