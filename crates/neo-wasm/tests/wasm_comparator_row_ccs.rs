//! Row-level CCS tests for the CCS-native comparators
//! (i32.eqz / i64.eqz / i32.eq / i32.ne / i64.eq / i64.ne). See
//! `push_comparator_constraints` in `crates/neo-wasm/src/ccs.rs` for the
//! constraint shape.

mod common;

use common::{assert_rejected, assert_satisfied};
use neo_math::F;
use neo_wasm::layout::{
    COL_CMP_AND, COL_CMP_HI_DIFF, COL_CMP_HI_INV, COL_CMP_HI_IS_ZERO, COL_CMP_LO_DIFF, COL_CMP_LO_INV,
    COL_CMP_LO_IS_ZERO, COL_STACK_READ0_VALUE, COL_STACK_READ0_VALUE_HI, COL_STACK_READ1_VALUE,
    COL_STACK_READ1_VALUE_HI, COL_STACK_WRITE0_VALUE,
};
use neo_wasm::witness_builder::build_witness_vector;
use neo_wasm::{
    opcode_code, opcode_info_from_code, StackLaneAccess, WasmOpcode, WasmParamInitState, WasmPcEdgeKind, WasmRowKind,
    WasmStepTrace,
};
use p3_field::{Field, PrimeCharacteristicRing};

fn step(
    opcode: WasmOpcode,
    sp_before: u64,
    sp_after: u64,
    stack_read0: Option<StackLaneAccess>,
    stack_read1: Option<StackLaneAccess>,
    stack_write0: Option<StackLaneAccess>,
    wide_values_enabled: bool,
) -> WasmStepTrace {
    fn physical(access: Option<StackLaneAccess>) -> Option<StackLaneAccess> {
        access.map(|lane| StackLaneAccess {
            addr: lane.addr * 2,
            value: lane.value,
        })
    }

    let code = opcode_code(opcode);
    WasmStepTrace {
        cycle: 0,
        row_kind: WasmRowKind::Program,
        pc_before: 2,
        pc_after: 3,
        control_choice: 0,
        pc_edge_kind: WasmPcEdgeKind::Static,
        param_init_before: WasmParamInitState::ZERO,
        param_init_after: WasmParamInitState::ZERO,
        wide_values_enabled,
        opcode_code: code,
        opcode,
        info: opcode_info_from_code(code),
        stack_reads_override: None,
        stack_writes_override: None,
        sp_before,
        sp_after,
        output_enabled_before: false,
        output_enabled_after: false,
        output_value_lo_before: 0,
        output_value_lo_after: 0,
        output_value_hi_before: 0,
        output_value_hi_after: 0,
        output_captured: false,
        call_stack_depth_before: 0,
        call_stack_depth_after: 0,
        current_function_ref: 0,
        current_function_num_locals: 0,
        stack_read0: physical(stack_read0),
        stack_read0_hi: None,
        stack_read1: physical(stack_read1),
        stack_read1_hi: None,
        stack_read2: None,
        stack_read2_hi: None,
        stack_write0: physical(stack_write0),
        stack_write0_hi: None,
        linear_memory: None,
        linear_memory_offset: 0,
        memory_pages_before: None,
        memory_pages_after: None,
        halted: false,
        locals_fbp: 0,
        locals_fbp_after: 0,
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
    }
}

fn set_i64_eqz_cmp_scratch(row: &mut [F], lo: u32, hi: u32) {
    let lo_diff = F::from_u64(u64::from(lo));
    let hi_diff = F::from_u64(u64::from(hi));
    row[COL_CMP_LO_DIFF] = lo_diff;
    row[COL_CMP_HI_DIFF] = hi_diff;
    let (lo_is_zero, lo_inv) = if lo_diff == F::ZERO {
        (F::ONE, F::ZERO)
    } else {
        (F::ZERO, lo_diff.try_inverse().unwrap())
    };
    let (hi_is_zero, hi_inv) = if hi_diff == F::ZERO {
        (F::ONE, F::ZERO)
    } else {
        (F::ZERO, hi_diff.try_inverse().unwrap())
    };
    row[COL_CMP_LO_IS_ZERO] = lo_is_zero;
    row[COL_CMP_LO_INV] = lo_inv;
    row[COL_CMP_HI_IS_ZERO] = hi_is_zero;
    row[COL_CMP_HI_INV] = hi_inv;
    row[COL_CMP_AND] = lo_is_zero * hi_is_zero;
}

#[test]
fn i32_eqz_row_accepts_zero_and_nonzero_inputs() {
    for (input, result) in [(0u32, 1u32), (1, 0), (0xFFFF_FFFF, 0)] {
        let row = build_witness_vector(&step(
            WasmOpcode::I32Eqz,
            1,
            1,
            Some(StackLaneAccess { addr: 0, value: input }),
            None,
            Some(StackLaneAccess { addr: 0, value: result }),
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
        Some(StackLaneAccess { addr: 0, value: 5 }),
        None,
        Some(StackLaneAccess { addr: 0, value: 1 }),
        false,
    ));
    row[COL_STACK_WRITE0_VALUE] = F::ONE;
    assert_rejected(&row, "tampered i32.eqz output");
}

#[test]
fn i64_eqz_row_accepts_zero_and_nonzero_inputs() {
    for (lo, hi, result) in [(0u32, 0u32, 1u32), (1, 0, 0), (0, 1, 0), (0xFFFF_FFFF, 0xFFFF_FFFF, 0)] {
        let mut row = build_witness_vector(&step(
            WasmOpcode::I64Eqz,
            1,
            1,
            Some(StackLaneAccess { addr: 0, value: lo }),
            None,
            Some(StackLaneAccess { addr: 0, value: result }),
            true,
        ));
        // The `step` helper doesn't populate stack_read0_hi; set it directly
        // and recompute the comparator scratch.
        row[COL_STACK_READ0_VALUE] = F::from_u64(u64::from(lo));
        row[COL_STACK_READ0_VALUE_HI] = F::from_u64(u64::from(hi));
        set_i64_eqz_cmp_scratch(&mut row, lo, hi);
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
        Some(StackLaneAccess { addr: 0, value: 1 }),
        None,
        Some(StackLaneAccess { addr: 0, value: 1 }),
        true,
    ));
    row[COL_STACK_READ0_VALUE] = F::from_u64(1);
    row[COL_STACK_READ0_VALUE_HI] = F::from_u64(0xFFFF_FFFFu64);
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
            Some(StackLaneAccess { addr: 0, value: lhs }),
            Some(StackLaneAccess { addr: 1, value: rhs }),
            Some(StackLaneAccess { addr: 0, value: eq_out }),
            false,
        ));
        assert_satisfied(&eq_row, &format!("i32.eq({lhs}, {rhs})"));
        let ne_row = build_witness_vector(&step(
            WasmOpcode::I32Ne,
            2,
            1,
            Some(StackLaneAccess { addr: 0, value: lhs }),
            Some(StackLaneAccess { addr: 1, value: rhs }),
            Some(StackLaneAccess { addr: 0, value: ne_out }),
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
        Some(StackLaneAccess { addr: 0, value: 7 }),
        Some(StackLaneAccess { addr: 1, value: 7 }),
        Some(StackLaneAccess { addr: 0, value: 1 }),
        false,
    ));
    row[COL_STACK_WRITE0_VALUE] = F::ZERO;
    assert_rejected(&row, "tampered i32.eq output");
}

fn set_i64_cmp_scratch(row: &mut [F], lo_diff: F, hi_diff: F) {
    row[COL_CMP_LO_DIFF] = lo_diff;
    row[COL_CMP_HI_DIFF] = hi_diff;
    let (lo_is_zero, lo_inv) = if lo_diff == F::ZERO {
        (F::ONE, F::ZERO)
    } else {
        (F::ZERO, lo_diff.try_inverse().unwrap())
    };
    let (hi_is_zero, hi_inv) = if hi_diff == F::ZERO {
        (F::ONE, F::ZERO)
    } else {
        (F::ZERO, hi_diff.try_inverse().unwrap())
    };
    row[COL_CMP_LO_IS_ZERO] = lo_is_zero;
    row[COL_CMP_LO_INV] = lo_inv;
    row[COL_CMP_HI_IS_ZERO] = hi_is_zero;
    row[COL_CMP_HI_INV] = hi_inv;
    row[COL_CMP_AND] = lo_is_zero * hi_is_zero;
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
            let mut row = build_witness_vector(&step(
                opcode,
                2,
                1,
                Some(StackLaneAccess { addr: 0, value: l_lo }),
                Some(StackLaneAccess { addr: 1, value: r_lo }),
                Some(StackLaneAccess {
                    addr: 0,
                    value: expected,
                }),
                true,
            ));
            // The `step` helper doesn't populate stack_read*_hi; set them
            // directly along with wide-values gate and comparator scratch.
            row[COL_STACK_READ0_VALUE] = F::from_u64(u64::from(l_lo));
            row[COL_STACK_READ0_VALUE_HI] = F::from_u64(u64::from(l_hi));
            row[COL_STACK_READ1_VALUE] = F::from_u64(u64::from(r_lo));
            row[COL_STACK_READ1_VALUE_HI] = F::from_u64(u64::from(r_hi));
            let lo_diff = F::from_u64(u64::from(l_lo)) - F::from_u64(u64::from(r_lo));
            let hi_diff = F::from_u64(u64::from(l_hi)) - F::from_u64(u64::from(r_hi));
            set_i64_cmp_scratch(&mut row, lo_diff, hi_diff);
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
        Some(StackLaneAccess { addr: 0, value: 1 }),
        Some(StackLaneAccess { addr: 1, value: 0 }),
        Some(StackLaneAccess { addr: 0, value: 1 }),
        true,
    ));
    row[COL_STACK_READ0_VALUE] = F::from_u64(1);
    row[COL_STACK_READ0_VALUE_HI] = F::from_u64(0xFFFF_FFFFu64);
    row[COL_STACK_READ1_VALUE] = F::ZERO;
    row[COL_STACK_READ1_VALUE_HI] = F::ZERO;
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
        Some(StackLaneAccess { addr: 0, value: 5 }),
        Some(StackLaneAccess { addr: 1, value: 5 }),
        Some(StackLaneAccess { addr: 0, value: 0 }),
        true,
    ));
    row[COL_STACK_READ0_VALUE] = F::from_u64(5);
    row[COL_STACK_READ1_VALUE] = F::from_u64(5);
    set_i64_cmp_scratch(&mut row, F::ZERO, F::ZERO);
    row[COL_STACK_WRITE0_VALUE] = F::ONE;
    assert_rejected(&row, "tampered i64.ne output");
}
