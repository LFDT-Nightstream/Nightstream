use neo_memory::riscv::exec_table::Rv32ExecTable;
use neo_memory::riscv::lookups::{
    decode_program, encode_program, RiscvCpu, RiscvInstruction, RiscvMemory, RiscvOpcode, RiscvShoutTables, PROG_ID,
    REG_ID,
};
use neo_memory::riscv::trace::{RiscvTraceAir, Rv32TraceWitness};
use neo_vm_trace::trace_program;
use neo_vm_trace::Twist;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

#[test]
fn rv32_trace_air_satisfies_addi_halt() {
    // Program: ADDI x1, x0, 1; HALT
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let decoded_program = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    let twist = RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);

    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 16).expect("trace_program");
    assert_eq!(trace.steps.len(), 2, "expected ADDI + HALT trace");

    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 4).expect("from_trace_padded_pow2");
    exec.validate_cycle_chain().expect("cycle chain");
    exec.validate_pc_chain().expect("pc chain");
    exec.validate_halted_tail().expect("halted tail");
    exec.validate_inactive_rows_are_empty()
        .expect("inactive rows");

    let air = RiscvTraceAir::new();
    let wit = Rv32TraceWitness::from_exec_table(&air.layout, &exec).expect("trace witness");
    air.assert_satisfied(&wit).expect("trace AIR satisfied");
}

#[test]
fn rv32_trace_air_accepts_virtual_commit_with_rd_x0() {
    let program = vec![
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulh,
            rd: 0,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let decoded_program = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    cpu.set_runtime_decomposition_enabled(true);
    let mut twist =
        RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    twist.store(REG_ID, 1, 0x8000_0000);
    twist.store(REG_ID, 2, 0xFFFF_FFFF);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);

    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 64).expect("trace_program");
    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 8).expect("from_trace_padded_pow2");

    let air = RiscvTraceAir::new();
    let wit = Rv32TraceWitness::from_exec_table(&air.layout, &exec).expect("trace witness");
    air.assert_satisfied(&wit)
        .expect("rd=x0 decomposed commit must satisfy AIR");
}

#[test]
fn trace_air_accepts_rv64_mulw_virtual_decomposition() {
    let program = vec![
        RiscvInstruction::RAluw {
            op: RiscvOpcode::Mulw,
            rd: 5,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let decoded_program = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(/*xlen=*/ 64);
    cpu.load_program(/*base=*/ 0, decoded_program);
    cpu.set_runtime_decomposition_enabled(true);
    let mut twist =
        RiscvMemory::with_program_in_twist(/*xlen=*/ 64, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    twist.store(REG_ID, 1, 3);
    twist.store(REG_ID, 2, 5);
    let shout = RiscvShoutTables::new(/*xlen=*/ 64);

    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 64).expect("trace_program");
    let exec = Rv32ExecTable::from_trace_padded_pow2_with_xlen(&trace, /*min_len=*/ 8, /*machine_xlen=*/ 64)
        .expect("from_trace_padded_pow2_with_xlen");

    let air = RiscvTraceAir::new_with_xlen(/*machine_xlen=*/ 64);
    let wit = Rv32TraceWitness::from_exec_table(&air.layout, &exec).expect("trace witness");
    air.assert_satisfied(&wit)
        .expect("rv64 Mulw decomposed rows must satisfy AIR under machine_xlen=64");
}

#[test]
fn rv32_trace_air_rejects_halted_tail_reactivation() {
    // Program with at least one active transition after row 0.
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 1,
            imm: 1,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let decoded_program = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    let twist = RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);

    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 16).expect("trace_program");
    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 4).expect("from_trace_padded_pow2");

    let air = RiscvTraceAir::new();
    let mut wit = Rv32TraceWitness::from_exec_table(&air.layout, &exec).expect("trace witness");

    // Force halted=1 from row 0 onward, while keeping row 1 active=1 in the witness.
    // This should violate the halted tail quiescence transition check at row 0.
    for row in 0..wit.t {
        wit.cols[air.layout.halted][row] = F::ONE;
    }

    let err = air
        .assert_satisfied(&wit)
        .expect_err("mutated witness should violate halted tail quiescence");
    assert!(
        err.contains("halted tail quiescence violated"),
        "unexpected error: {err}"
    );
}

#[test]
fn rv32_trace_air_rejects_non_boolean_active() {
    // Program: ADDI x1, x0, 1; HALT
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let decoded_program = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    let twist = RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);

    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 16).expect("trace_program");
    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 4).expect("from_trace_padded_pow2");

    let air = RiscvTraceAir::new();
    let mut wit = Rv32TraceWitness::from_exec_table(&air.layout, &exec).expect("trace witness");
    wit.cols[air.layout.active][0] = F::from_u64(2);

    let err = air
        .assert_satisfied(&wit)
        .expect_err("mutated witness should violate bit booleanity");
    assert!(err.contains("active not boolean"), "unexpected error: {err}");
}

#[test]
fn rv32_trace_air_rejects_virtual_pc_advance() {
    // Program: ADDI x1, x0, 1; HALT
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let decoded_program = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    let twist = RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);

    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 16).expect("trace_program");
    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 4).expect("from_trace_padded_pow2");

    let air = RiscvTraceAir::new();
    let mut wit = Rv32TraceWitness::from_exec_table(&air.layout, &exec).expect("trace witness");
    wit.cols[air.layout.is_virtual][0] = F::ONE;

    let err = air
        .assert_satisfied(&wit)
        .expect_err("mutated witness should violate virtual PC rule");
    assert!(
        err.contains("virtual step must keep pc_after == pc_before"),
        "unexpected error: {err}"
    );
}

#[test]
fn rv32_trace_air_rejects_virtual_sequence_remaining_countdown_break() {
    let program = vec![
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulh,
            rd: 5,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let decoded_program = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    cpu.set_runtime_decomposition_enabled(true);
    let mut twist =
        RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    twist.store(REG_ID, 1, 0x8000_0000);
    twist.store(REG_ID, 2, 0xFFFF_FFFF);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);

    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 64).expect("trace_program");
    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 8).expect("from_trace_padded_pow2");

    let air = RiscvTraceAir::new();
    let mut wit = Rv32TraceWitness::from_exec_table(&air.layout, &exec).expect("trace witness");

    // Break the countdown invariant on the first virtual row:
    // remaining(i) should be remaining(i+1) + 1.
    wit.cols[air.layout.virtual_sequence_remaining][0] += F::ONE;

    let err = air
        .assert_satisfied(&wit)
        .expect_err("mutated witness should violate virtual sequence countdown");
    assert!(err.contains("virtual_sequence_remaining"), "unexpected error: {err}");
}

#[test]
fn rv32_trace_air_rejects_virtual_commit_value_mismatch() {
    let program = vec![
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulh,
            rd: 5,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let decoded_program = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    cpu.set_runtime_decomposition_enabled(true);
    let mut twist =
        RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    twist.store(REG_ID, 1, 0x8000_0000);
    twist.store(REG_ID, 2, 0xFFFF_FFFF);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);

    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 64).expect("trace_program");
    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 8).expect("from_trace_padded_pow2");

    let air = RiscvTraceAir::new();
    let mut wit = Rv32TraceWitness::from_exec_table(&air.layout, &exec).expect("trace witness");

    // Row 6 is the last virtual write in MULH decomposition and row 7 is the non-virtual commit.
    let commit_row = 7usize;
    wit.cols[air.layout.rd_val][commit_row] += F::ONE;

    let err = air
        .assert_satisfied(&wit)
        .expect_err("mutated witness should violate virtual commit linkage");
    assert!(err.contains("virtual commit value mismatch"), "unexpected error: {err}");
}

#[test]
fn rv32_trace_air_rejects_nonzero_rd_lane_when_rd_has_write_is_zero() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let decoded_program = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    let twist = RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);

    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 16).expect("trace_program");
    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 4).expect("from_trace_padded_pow2");

    let air = RiscvTraceAir::new();
    let mut wit = Rv32TraceWitness::from_exec_table(&air.layout, &exec).expect("trace witness");

    // HALT row has no write event, so rd lane must remain zero.
    wit.cols[air.layout.rd_addr][1] = F::from_u64(7);

    let err = air
        .assert_satisfied(&wit)
        .expect_err("mutated witness should violate rd write-lane quiescence");
    assert!(
        err.contains("rd_addr must be 0 when rd_has_write=0"),
        "unexpected error: {err}"
    );
}

#[test]
fn rv32_trace_air_rejects_virtual_row_marked_halted() {
    let program = vec![
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulh,
            rd: 5,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let decoded_program = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    cpu.set_runtime_decomposition_enabled(true);
    let mut twist =
        RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    twist.store(REG_ID, 1, 0x8000_0000);
    twist.store(REG_ID, 2, 0xFFFF_FFFF);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);

    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 64).expect("trace_program");
    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 8).expect("from_trace_padded_pow2");

    let air = RiscvTraceAir::new();
    let mut wit = Rv32TraceWitness::from_exec_table(&air.layout, &exec).expect("trace witness");

    let virtual_row = (0..wit.t)
        .find(|&i| wit.cols[air.layout.is_virtual][i] == F::ONE)
        .expect("expected at least one virtual row");
    wit.cols[air.layout.halted][virtual_row] = F::ONE;

    let err = air
        .assert_satisfied(&wit)
        .expect_err("mutated witness should violate virtual row halted invariant");
    assert!(err.contains("virtual row cannot be halted"), "unexpected error: {err}");
}

#[test]
fn rv32_trace_air_rejects_nonzero_shout_table_id_without_lookup() {
    let program = vec![
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulh,
            rd: 5,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let decoded_program = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    cpu.set_runtime_decomposition_enabled(true);
    let mut twist =
        RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    twist.store(REG_ID, 1, 0x8000_0000);
    twist.store(REG_ID, 2, 0xFFFF_FFFF);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);

    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 64).expect("trace_program");
    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 8).expect("from_trace_padded_pow2");

    let air = RiscvTraceAir::new();
    let mut wit = Rv32TraceWitness::from_exec_table(&air.layout, &exec).expect("trace witness");

    let virtual_row = (0..wit.t)
        .find(|&i| wit.cols[air.layout.is_virtual][i] == F::ONE)
        .expect("expected at least one virtual row");
    wit.cols[air.layout.shout_has_lookup][virtual_row] = F::ZERO;
    wit.cols[air.layout.shout_table_id][virtual_row] = F::from_u64(9);
    wit.cols[air.layout.shout_val][virtual_row] = F::ZERO;
    wit.cols[air.layout.shout_lhs][virtual_row] = F::ZERO;
    wit.cols[air.layout.shout_rhs][virtual_row] = F::ZERO;
    wit.cols[air.layout.shout_link_lhs][virtual_row] = F::ZERO;
    wit.cols[air.layout.shout_link_rhs][virtual_row] = F::ZERO;
    wit.cols[air.layout.shout_add_sub_key][virtual_row] = F::ZERO;

    let err = air
        .assert_satisfied(&wit)
        .expect_err("mutated witness should violate shout table-id quiescence invariant");
    assert!(
        err.contains("shout_table_id must be 0 when shout_has_lookup=0"),
        "unexpected error: {err}"
    );
}

#[test]
fn rv32_trace_air_rejects_virtual_instr_word_drift() {
    let program = vec![
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulh,
            rd: 5,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let decoded_program = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    cpu.set_runtime_decomposition_enabled(true);
    let mut twist =
        RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    twist.store(REG_ID, 1, 0x8000_0000);
    twist.store(REG_ID, 2, 0xFFFF_FFFF);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);

    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 64).expect("trace_program");
    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 8).expect("from_trace_padded_pow2");

    let air = RiscvTraceAir::new();
    let mut wit = Rv32TraceWitness::from_exec_table(&air.layout, &exec).expect("trace witness");

    let virtual_row = (0..wit.t.saturating_sub(1))
        .find(|&i| wit.cols[air.layout.is_virtual][i] == F::ONE)
        .expect("expected at least one virtual row with successor");
    wit.cols[air.layout.instr_word][virtual_row + 1] += F::ONE;

    let err = air
        .assert_satisfied(&wit)
        .expect_err("mutated witness should violate virtual instr-word continuity");
    assert!(
        err.contains("virtual-row decode failed")
            || err.contains("no decomposition sequence")
            || err.contains("virtual row must preserve instr_word across sequence"),
        "unexpected error: {err}"
    );
}

#[test]
fn rv32_trace_air_rejects_virtual_transition_without_last_virtual_write() {
    let program = vec![
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulh,
            rd: 5,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let decoded_program = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    cpu.set_runtime_decomposition_enabled(true);
    let mut twist =
        RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    twist.store(REG_ID, 1, 0x8000_0000);
    twist.store(REG_ID, 2, 0xFFFF_FFFF);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);

    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 64).expect("trace_program");
    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 8).expect("from_trace_padded_pow2");

    let air = RiscvTraceAir::new();
    let mut wit = Rv32TraceWitness::from_exec_table(&air.layout, &exec).expect("trace witness");

    let transition_row = (0..wit.t)
        .find(|&i| wit.cols[air.layout.virtual_transition][i] == F::ONE)
        .expect("expected transition row");
    wit.cols[air.layout.rd_has_write][transition_row] = F::ZERO;
    wit.cols[air.layout.rd_addr][transition_row] = F::ZERO;
    wit.cols[air.layout.rd_val][transition_row] = F::ZERO;
    wit.cols[air.layout.virtual_commit_link][transition_row] = F::ZERO;

    let err = air
        .assert_satisfied(&wit)
        .expect_err("mutated witness should violate virtual-transition write invariant");
    assert!(
        err.contains("expected register write") || err.contains("virtual_transition requires last virtual row write"),
        "unexpected error: {err}"
    );
}

#[test]
fn rv32_trace_air_rejects_virtual_commit_from_prev_linkage_break() {
    let program = vec![
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulh,
            rd: 5,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let decoded_program = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    cpu.set_runtime_decomposition_enabled(true);
    let mut twist =
        RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    twist.store(REG_ID, 1, 0x8000_0000);
    twist.store(REG_ID, 2, 0xFFFF_FFFF);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);

    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 64).expect("trace_program");
    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 8).expect("from_trace_padded_pow2");

    let air = RiscvTraceAir::new();
    let mut wit = Rv32TraceWitness::from_exec_table(&air.layout, &exec).expect("trace witness");

    let commit_row = (1..wit.t)
        .find(|&i| wit.cols[air.layout.virtual_commit_from_prev][i] == F::ONE)
        .expect("expected commit row with virtual_commit_from_prev");
    wit.cols[air.layout.virtual_commit_from_prev][commit_row] = F::ZERO;

    let err = air
        .assert_satisfied(&wit)
        .expect_err("mutated witness should violate virtual_commit_from_prev linkage");
    assert!(
        err.contains("virtual_commit_from_prev linkage mismatch"),
        "unexpected error: {err}"
    );
}

#[test]
fn rv32_trace_air_rejects_virtual_op_shape_mismatch() {
    let program = vec![
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulh,
            rd: 5,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let decoded_program = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    cpu.set_runtime_decomposition_enabled(true);
    let mut twist =
        RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    twist.store(REG_ID, 1, 0x8000_0001);
    twist.store(REG_ID, 2, 0x7FFF_FFFF);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);

    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 64).expect("trace_program");
    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 8).expect("from_trace_padded_pow2");

    let air = RiscvTraceAir::new();
    let mut wit = Rv32TraceWitness::from_exec_table(&air.layout, &exec).expect("trace witness");

    // First virtual MULH row is MOVSIGN(src=rs1). Tamper rs1_addr so op-shape check fails.
    let first_virtual_row = (0..wit.t)
        .find(|&i| wit.cols[air.layout.is_virtual][i] == F::ONE)
        .expect("expected virtual row");
    wit.cols[air.layout.rs1_addr][first_virtual_row] = F::from_u64(9);

    let err = air
        .assert_satisfied(&wit)
        .expect_err("mutated witness should violate virtual op read-shape semantics");
    assert!(err.contains("read addr mismatch"), "unexpected error: {err}");
}

#[test]
fn rv32_trace_air_rejects_virtual_assert_predicate_break() {
    let program = vec![
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Divu,
            rd: 5,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let decoded_program = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    cpu.set_runtime_decomposition_enabled(true);
    let mut twist =
        RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    twist.store(REG_ID, 1, 23);
    twist.store(REG_ID, 2, 5);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);

    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 64).expect("trace_program");
    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 16).expect("from_trace_padded_pow2");

    let air = RiscvTraceAir::new();
    let mut wit = Rv32TraceWitness::from_exec_table(&air.layout, &exec).expect("trace witness");

    // DIVU remaining=6 row is AssertMulUNoOverflow(v_q, rs2). Break overflow predicate.
    let assert_overflow_row = (0..wit.t)
        .find(|&i| {
            wit.cols[air.layout.is_virtual][i] == F::ONE
                && wit.cols[air.layout.virtual_sequence_remaining][i] == F::from_u64(6)
        })
        .expect("expected DIVU AssertMulUNoOverflow virtual row");
    wit.cols[air.layout.rs2_val][assert_overflow_row] = F::from_u64(u32::MAX as u64);

    let err = air
        .assert_satisfied(&wit)
        .expect_err("mutated witness should violate virtual assert predicate");
    assert!(
        err.contains("AssertMulUNoOverflow predicate failed"),
        "unexpected error: {err}"
    );
}
