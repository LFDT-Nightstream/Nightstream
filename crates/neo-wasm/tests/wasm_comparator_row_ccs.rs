//! Row-level CCS tests for the CCS-native comparators
//! (i32.eqz / i64.eqz / i32.eq / i32.ne / i64.eq / i64.ne). See
//! `push_comparator_constraints` in `crates/neo-wasm/src/ccs.rs` for the
//! constraint shape.

mod common;

use common::{assert_rejected, assert_satisfied};
use neo_math::F;
use neo_wasm::layout::{
    COL_CMP_LO_DIFF, COL_CMP_LO_INV, COL_CMP_LO_IS_ZERO, COL_STACK_READ_VALUE_HI, COL_STACK_READ_VALUE_LO,
    COL_STACK_WRITE0_VALUE_LO,
};
use neo_wasm::witness_builder::build_witness_vector;
use neo_wasm::{
    opcode_code, opcode_info_from_code, StackValueAccess, WasmCountdownState, WasmOpcode, WasmOutputState,
    WasmPcEdgeKind, WasmRowKind, WasmStepState, WasmVmStep,
};
use p3_field::PrimeCharacteristicRing;

fn step(
    opcode: WasmOpcode,
    sp_before: u64,
    sp_after: u64,
    stack_read0: Option<StackValueAccess>,
    stack_read1: Option<StackValueAccess>,
    stack_write0: Option<StackValueAccess>,
    wide_values_enabled: bool,
) -> WasmVmStep {
    fn state(pc: u64, sp: u64) -> WasmStepState {
        WasmStepState {
            pc,
            sp,
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
        }
    }

    fn physical(access: Option<StackValueAccess>) -> Option<StackValueAccess> {
        access.map(|lane| StackValueAccess::new(lane.addr_lo * 2, lane.value_lo).with_optional_hi(lane.value_hi))
    }

    let code = opcode_code(opcode);
    WasmVmStep {
        cycle: 0,
        row_kind: WasmRowKind::Program,
        state_before: state(2, sp_before),
        state_after: state(3, sp_after),
        control_choice: 0,
        pc_edge_kind: WasmPcEdgeKind::Static,
        wide_values_enabled,
        opcode,
        info: opcode_info_from_code(code),
        stack_reads_override: None,
        stack_writes_override: None,
        output_captured: false,
        current_function_ref: 0,
        current_function_num_locals: 0,
        stack_read0: physical(stack_read0),
        stack_read1: physical(stack_read1),
        stack_read2: None,
        stack_write0: physical(stack_write0),
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
fn i32_eqz_row_accepts_zero_and_nonzero_inputs() {
    for (input, result) in [(0u32, 1u32), (1, 0), (0xFFFF_FFFF, 0)] {
        let row = build_witness_vector(&step(
            WasmOpcode::I32Eqz,
            1,
            1,
            Some(StackValueAccess::new(0, input)),
            None,
            Some(StackValueAccess::new(0, result)),
            false,
        ));
        assert_satisfied(&row, &format!("i32.eqz({input})"));
    }
}

#[test]
fn i32_eqz_row_rejects_tampered_output() {
    // Honest: eqz(5) = 0. Tampered: claim eqz(5) = 1.
    let mut row = build_witness_vector(&step(
        WasmOpcode::I32Eqz,
        1,
        1,
        Some(StackValueAccess::new(0, 5)),
        None,
        Some(StackValueAccess::new(0, 1)),
        false,
    ));
    row[COL_STACK_WRITE0_VALUE_LO] = F::ONE;
    assert_rejected(&row, "tampered i32.eqz output");
}

#[test]
fn i64_eqz_row_accepts_zero_and_nonzero_inputs() {
    for (lo, hi, result) in [(0u32, 0u32, 1u32), (1, 0, 0), (0, 1, 0), (0xFFFF_FFFF, 0xFFFF_FFFF, 0)] {
        let row = build_witness_vector(&step(
            WasmOpcode::I64Eqz,
            1,
            1,
            Some(StackValueAccess::with_hi(0, lo, hi)),
            None,
            Some(StackValueAccess::new(0, result)),
            true,
        ));
        assert_satisfied(&row, &format!("i64.eqz(lo={lo}, hi={hi})"));
    }
}

/// Regression: Goldilocks has q = 2^64 - 2^32 + 1, so the u64 value
/// 0xffff_ffff_0000_0001 = q is the field zero (not the i64's 0). An i64.eqz
/// zero-test over a single field element `lo + hi*2^32` would wrongly accept
/// `eqz(0xffff_ffff_0000_0001) = 1`. The split limb-by-limb zero-test must
/// reject this.
#[test]
fn i64_eqz_row_rejects_goldilocks_modulus_collision() {
    let mut row = build_witness_vector(&step(
        WasmOpcode::I64Eqz,
        1,
        1,
        Some(StackValueAccess::new(0, 1)),
        None,
        Some(StackValueAccess::new(0, 1)),
        true,
    ));
    row[COL_STACK_READ_VALUE_LO[0]] = F::from_u64(1);
    row[COL_STACK_READ_VALUE_HI[0]] = F::from_u64(0xFFFF_FFFFu64);
    row[COL_CMP_LO_DIFF] = F::ZERO;
    row[COL_CMP_LO_INV] = F::ZERO;
    row[COL_CMP_LO_IS_ZERO] = F::ONE;
    assert_rejected(&row, "i64.eqz must reject Goldilocks-modulus collision");
}

#[test]
fn i32_eq_and_ne_rows_accept_equal_and_distinct_inputs() {
    for (lhs, rhs, eq_out, ne_out) in [(0u32, 0u32, 1u32, 0u32), (5, 5, 1, 0), (5, 7, 0, 1), (7, 5, 0, 1)] {
        let eq_row = build_witness_vector(&step(
            WasmOpcode::I32Eq,
            2,
            1,
            Some(StackValueAccess::new(0, lhs)),
            Some(StackValueAccess::new(1, rhs)),
            Some(StackValueAccess::new(0, eq_out)),
            false,
        ));
        assert_satisfied(&eq_row, &format!("i32.eq({lhs}, {rhs})"));
        let ne_row = build_witness_vector(&step(
            WasmOpcode::I32Ne,
            2,
            1,
            Some(StackValueAccess::new(0, lhs)),
            Some(StackValueAccess::new(1, rhs)),
            Some(StackValueAccess::new(0, ne_out)),
            false,
        ));
        assert_satisfied(&ne_row, &format!("i32.ne({lhs}, {rhs})"));
    }
}

#[test]
fn i32_eq_row_rejects_tampered_output() {
    let mut row = build_witness_vector(&step(
        WasmOpcode::I32Eq,
        2,
        1,
        Some(StackValueAccess::new(0, 7)),
        Some(StackValueAccess::new(1, 7)),
        Some(StackValueAccess::new(0, 1)),
        false,
    ));
    row[COL_STACK_WRITE0_VALUE_LO] = F::ZERO;
    assert_rejected(&row, "tampered i32.eq output");
}

#[test]
fn i64_eq_and_ne_rows_accept_equal_and_distinct_inputs() {
    // (lhs_lo, lhs_hi, rhs_lo, rhs_hi, eq_out, ne_out)
    let cases = [
        (0u32, 0u32, 0u32, 0u32, 1u32, 0u32),
        (5, 0, 5, 0, 1, 0),
        (5, 0, 7, 0, 0, 1),
        (5, 1, 5, 1, 1, 0),
        (5, 1, 5, 2, 0, 1),
        (0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, 1, 0),
    ];
    for (l_lo, l_hi, r_lo, r_hi, eq_out, ne_out) in cases {
        for (opcode, expected) in [(WasmOpcode::I64Eq, eq_out), (WasmOpcode::I64Ne, ne_out)] {
            let row = build_witness_vector(&step(
                opcode,
                2,
                1,
                Some(StackValueAccess::with_hi(0, l_lo, l_hi)),
                Some(StackValueAccess::with_hi(1, r_lo, r_hi)),
                Some(StackValueAccess::new(0, expected)),
                true,
            ));
            assert_satisfied(&row, &format!("{opcode:?}(lhs=({l_lo},{l_hi}), rhs=({r_lo},{r_hi}))"));
        }
    }
}

/// Same Goldilocks-modulus collision concern as `i64_eqz_row_rejects_*`,
/// but applied to i64.eq: claim eq(lhs, rhs) = 1 when (lhs-rhs) =
/// 0xffff_ffff_0000_0001 ≡ 0 mod q. The split limb-by-limb gates must
/// still see a nonzero hi-limb diff and reject.
#[test]
fn i64_eq_row_rejects_goldilocks_modulus_collision() {
    let mut row = build_witness_vector(&step(
        WasmOpcode::I64Eq,
        2,
        1,
        Some(StackValueAccess::new(0, 1)),
        Some(StackValueAccess::new(1, 0)),
        Some(StackValueAccess::new(0, 1)),
        true,
    ));
    row[COL_STACK_READ_VALUE_LO[0]] = F::from_u64(1);
    row[COL_STACK_READ_VALUE_HI[0]] = F::from_u64(0xFFFF_FFFFu64);
    row[COL_STACK_READ_VALUE_LO[1]] = F::ZERO;
    row[COL_STACK_READ_VALUE_HI[1]] = F::ZERO;
    row[COL_CMP_LO_DIFF] = F::ZERO;
    row[COL_CMP_LO_INV] = F::ZERO;
    row[COL_CMP_LO_IS_ZERO] = F::ONE;
    assert_rejected(&row, "i64.eq must reject Goldilocks-modulus collision");
}

#[test]
fn i64_ne_row_rejects_tampered_output() {
    // Honest: ne(5, 5) = 0. Tampered: claim ne(5, 5) = 1.
    let mut row = build_witness_vector(&step(
        WasmOpcode::I64Ne,
        2,
        1,
        Some(StackValueAccess::new(0, 5)),
        Some(StackValueAccess::new(1, 5)),
        Some(StackValueAccess::new(0, 0)),
        true,
    ));
    row[COL_STACK_WRITE0_VALUE_LO] = F::ONE;
    assert_rejected(&row, "tampered i64.ne output");
}
