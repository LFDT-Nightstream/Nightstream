//! Row-level CCS tests for the trap transition: `unreachable`, div/rem by
//! zero, and signed division overflow as terminal states, plus rejection of
//! hidden, faked, or post-trap execution claims. End-to-end trap proving
//! lives in `wasm_trap.rs`.

mod common;

use common::{assert_rejected, assert_satisfied, step};
use neo_math::F;
use neo_wasm::layout::{COL_DIV_OVERFLOW, COL_DIV_TRAP, COL_TRAPPED_AFTER, COL_TRAPPED_BEFORE};
use neo_wasm::witness_builder::build_witness_vector;
use neo_wasm::{opcode_code, StackValueAccess, WasmOpcode};
use p3_field::PrimeCharacteristicRing;

#[test]
fn unreachable_row_requires_halted_boundary() {
    let accepted = build_witness_vector(&step(
        0,
        14,
        opcode_code(WasmOpcode::Unreachable),
        0,
        0,
        None,
        None,
        None,
        None,
        None,
        0,
        true,
    ));
    assert_satisfied(&accepted, "unreachable row");

    let rejected = build_witness_vector(&step(
        0,
        14,
        opcode_code(WasmOpcode::Unreachable),
        0,
        0,
        None,
        None,
        None,
        None,
        None,
        0,
        false,
    ));
    assert_rejected(&rejected, "unreachable row with halted=0");
}

#[test]
fn unreachable_row_rejects_clean_trapped_flag() {
    let mut witness = build_witness_vector(&step(
        0,
        14,
        opcode_code(WasmOpcode::Unreachable),
        0,
        0,
        None,
        None,
        None,
        None,
        None,
        0,
        true,
    ));
    assert_satisfied(&witness, "unreachable row with trapped_after=1");
    witness[COL_TRAPPED_AFTER] = F::ZERO;
    assert_rejected(&witness, "unreachable row claiming a clean (non-trapped) exit");
}

#[test]
fn non_trap_row_rejects_fake_trapped_flag() {
    let mut witness = build_witness_vector(&step(
        0,
        1,
        opcode_code(WasmOpcode::Nop),
        0,
        0,
        None,
        None,
        None,
        None,
        None,
        0,
        false,
    ));
    assert_satisfied(&witness, "nop row");
    witness[COL_TRAPPED_AFTER] = F::ONE;
    assert_rejected(&witness, "nop row claiming a trap");
}

#[test]
fn program_row_rejects_execution_after_trap() {
    let mut witness = build_witness_vector(&step(
        0,
        1,
        opcode_code(WasmOpcode::Nop),
        0,
        0,
        None,
        None,
        None,
        None,
        None,
        0,
        false,
    ));
    // trapped_before = trapped_after = 1 satisfies the transition row, so
    // the rejection must come from `is_program_row · trapped_before = 0`.
    witness[COL_TRAPPED_BEFORE] = F::ONE;
    witness[COL_TRAPPED_AFTER] = F::ONE;
    assert_rejected(&witness, "program row executing after a trap");
}

#[test]
fn div_by_zero_row_rejects_clean_claim() {
    let mut witness = build_witness_vector(&step(
        0,
        5,
        opcode_code(WasmOpcode::I32DivU),
        2,
        1,
        Some(StackValueAccess::new(0, 7)),
        Some(StackValueAccess::new(1, 0)),
        None,
        Some(StackValueAccess::new(0, 0)),
        None,
        0,
        true,
    ));
    assert_satisfied(&witness, "i32.div_u row trapping on a zero divisor");
    // Hiding the trap: div_trap = 0 contradicts (Σ div sel) · divisor_is_zero.
    witness[COL_DIV_TRAP] = F::ZERO;
    assert_rejected(&witness, "div-by-zero row claiming no trap");
}

#[test]
fn div_row_rejects_fake_trap_on_nonzero_divisor() {
    let mut witness = build_witness_vector(&step(
        0,
        5,
        opcode_code(WasmOpcode::I32DivU),
        2,
        1,
        Some(StackValueAccess::new(0, 7)),
        Some(StackValueAccess::new(1, 2)),
        None,
        Some(StackValueAccess::new(0, 3)),
        None,
        0,
        false,
    ));
    assert_satisfied(&witness, "i32.div_u row with a nonzero divisor");
    // Faking the trap: divisor_is_zero is pinned to 0 by the zero test, so
    // div_trap = (Σ div sel) · 0 must be 0.
    witness[COL_DIV_TRAP] = F::ONE;
    witness[COL_TRAPPED_AFTER] = F::ONE;
    assert_rejected(&witness, "div row faking a trap on a nonzero divisor");
}

#[test]
fn signed_div_overflow_row_rejects_clean_claim() {
    let mut witness = build_witness_vector(&step(
        0,
        5,
        opcode_code(WasmOpcode::I32DivS),
        2,
        1,
        Some(StackValueAccess::new(0, 0x8000_0000)),
        Some(StackValueAccess::new(1, u32::MAX)),
        None,
        Some(StackValueAccess::new(0, 0)),
        None,
        0,
        true,
    ));
    assert_satisfied(&witness, "i32.div_s row trapping on MIN / -1");
    // Hiding the trap: the zero tests pin both overflow flags to 1, so
    // div_overflow = (Σ div_s sel) · 1 forces div_trap = 1.
    witness[COL_DIV_TRAP] = F::ZERO;
    assert_rejected(&witness, "signed-overflow row claiming no trap");
}

#[test]
fn div_s_row_rejects_fake_overflow_trap() {
    // MIN dividend with a non-(−1) divisor divides cleanly: MIN / 2.
    let mut witness = build_witness_vector(&step(
        0,
        5,
        opcode_code(WasmOpcode::I32DivS),
        2,
        1,
        Some(StackValueAccess::new(0, 0x8000_0000)),
        Some(StackValueAccess::new(1, 2)),
        None,
        Some(StackValueAccess::new(0, 0xC000_0000)),
        None,
        0,
        false,
    ));
    assert_satisfied(&witness, "i32.div_s row with MIN dividend and divisor 2");
    // Faking the trap: divisor_is_neg1 is pinned to 0 by its zero test, so
    // overflow_cond = is_min · 0 must be 0 and div_overflow must follow.
    witness[COL_DIV_OVERFLOW] = F::ONE;
    witness[COL_DIV_TRAP] = F::ONE;
    witness[COL_TRAPPED_AFTER] = F::ONE;
    assert_rejected(&witness, "div_s row faking an overflow trap");
}
