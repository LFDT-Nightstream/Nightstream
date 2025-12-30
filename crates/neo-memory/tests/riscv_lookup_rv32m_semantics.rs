use neo_memory::riscv::lookups::{compute_op, RiscvOpcode};

#[test]
fn rv32m_div_rem_edge_cases() {
    let xlen = 32;
    let mask = (1u64 << xlen) - 1;

    // DIV/DIVU by 0 => all ones.
    assert_eq!(compute_op(RiscvOpcode::Div, 123, 0, xlen), mask);
    assert_eq!(compute_op(RiscvOpcode::Divu, 123, 0, xlen), mask);

    // REM/REMU by 0 => dividend.
    assert_eq!(compute_op(RiscvOpcode::Rem, 123, 0, xlen), 123);
    assert_eq!(compute_op(RiscvOpcode::Remu, 123, 0, xlen), 123);

    // Overflow: INT_MIN / -1 => INT_MIN, INT_MIN % -1 => 0.
    let int_min = i32::MIN as u32 as u64;
    let minus_one = (-1i32) as u32 as u64;
    assert_eq!(compute_op(RiscvOpcode::Div, int_min, minus_one, xlen), int_min);
    assert_eq!(compute_op(RiscvOpcode::Rem, int_min, minus_one, xlen), 0);
}

#[test]
fn rv32m_mulh_sign_behavior_smoke() {
    let xlen = 32;
    let a = (-2i32) as u32 as u64;
    let b = 3u64;
    assert_eq!(compute_op(RiscvOpcode::Mulh, a, b, xlen), u32::MAX as u64);
}

