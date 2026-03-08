use neo_memory::riscv::exec_table::Rv32ExecTable;
use neo_memory::riscv::lookups::{
    compute_op, decode_program, encode_program, RiscvCpu, RiscvInstruction, RiscvMemory, RiscvOpcode, RiscvShoutTables,
    PROG_ID,
};
use neo_vm_trace::{trace_program, Twist};

fn run_mulw_trace(lhs: u64, rhs: u64) -> neo_vm_trace::VmTrace<u64, u64, u128> {
    let program = vec![
        RiscvInstruction::RAluw {
            op: RiscvOpcode::Mulw,
            rd: 4,
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
    twist.store(neo_memory::riscv::lookups::REG_ID, 1, lhs);
    twist.store(neo_memory::riscv::lookups::REG_ID, 2, rhs);
    let shout = RiscvShoutTables::new(/*xlen=*/ 64);

    trace_program(cpu, twist, shout, /*max_steps=*/ 32).expect("trace_program")
}

fn run_divuw_trace(lhs: u64, rhs: u64) -> neo_vm_trace::VmTrace<u64, u64, u128> {
    let program = vec![
        RiscvInstruction::RAluw {
            op: RiscvOpcode::Divuw,
            rd: 4,
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
    twist.store(neo_memory::riscv::lookups::REG_ID, 1, lhs);
    twist.store(neo_memory::riscv::lookups::REG_ID, 2, rhs);
    let shout = RiscvShoutTables::new(/*xlen=*/ 64);

    trace_program(cpu, twist, shout, /*max_steps=*/ 32).expect("trace_program")
}

fn run_divw_trace(lhs: u64, rhs: u64) -> neo_vm_trace::VmTrace<u64, u64, u128> {
    let program = vec![
        RiscvInstruction::RAluw {
            op: RiscvOpcode::Divw,
            rd: 4,
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
    twist.store(neo_memory::riscv::lookups::REG_ID, 1, lhs);
    twist.store(neo_memory::riscv::lookups::REG_ID, 2, rhs);
    let shout = RiscvShoutTables::new(/*xlen=*/ 64);

    trace_program(cpu, twist, shout, /*max_steps=*/ 32).expect("trace_program")
}

fn run_remuw_trace(lhs: u64, rhs: u64) -> neo_vm_trace::VmTrace<u64, u64, u128> {
    let program = vec![
        RiscvInstruction::RAluw {
            op: RiscvOpcode::Remuw,
            rd: 4,
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
    twist.store(neo_memory::riscv::lookups::REG_ID, 1, lhs);
    twist.store(neo_memory::riscv::lookups::REG_ID, 2, rhs);
    let shout = RiscvShoutTables::new(/*xlen=*/ 64);

    trace_program(cpu, twist, shout, /*max_steps=*/ 32).expect("trace_program")
}

fn run_remw_trace(lhs: u64, rhs: u64) -> neo_vm_trace::VmTrace<u64, u64, u128> {
    let program = vec![
        RiscvInstruction::RAluw {
            op: RiscvOpcode::Remw,
            rd: 4,
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
    twist.store(neo_memory::riscv::lookups::REG_ID, 1, lhs);
    twist.store(neo_memory::riscv::lookups::REG_ID, 2, rhs);
    let shout = RiscvShoutTables::new(/*xlen=*/ 64);

    trace_program(cpu, twist, shout, /*max_steps=*/ 32).expect("trace_program")
}

fn assert_no_direct_w_arith_shout_events(trace: &neo_vm_trace::VmTrace<u64, u64, u128>) {
    let shout = RiscvShoutTables::new(/*xlen=*/ 64);
    for step in &trace.steps {
        for ev in &step.shout_events {
            let op = shout.id_to_opcode(ev.shout_id);
            assert!(
                !matches!(
                    op,
                    Some(
                        RiscvOpcode::Mulw
                            | RiscvOpcode::Divw
                            | RiscvOpcode::Divuw
                            | RiscvOpcode::Remw
                            | RiscvOpcode::Remuw
                    )
                ),
                "RV64 W arithmetic must be helper-owned in the trace, found direct shout event {:?}",
                op
            );
        }
    }
}

#[test]
fn mulw_runtime_trace_decomposes_positive_rv64_word_result() {
    let lhs = 7u64;
    let rhs = 9u64;
    let trace = run_mulw_trace(lhs, rhs);
    assert_no_direct_w_arith_shout_events(&trace);

    assert_eq!(trace.steps.len(), 5);

    let mul_row = &trace.steps[0];
    assert!(mul_row.is_virtual);
    assert_eq!(mul_row.pc_before, 0);
    assert_eq!(mul_row.pc_after, 0);
    assert_eq!(mul_row.virtual_sequence_remaining, Some(3));

    let sign_row = &trace.steps[1];
    assert!(sign_row.is_virtual);
    assert_eq!(sign_row.pc_before, 0);
    assert_eq!(sign_row.pc_after, 0);
    assert_eq!(sign_row.virtual_sequence_remaining, Some(2));

    let compose_row = &trace.steps[2];
    assert!(compose_row.is_virtual);
    assert_eq!(compose_row.pc_before, 0);
    assert_eq!(compose_row.pc_after, 0);
    assert_eq!(compose_row.virtual_sequence_remaining, Some(1));

    let commit_row = &trace.steps[3];
    assert!(!commit_row.is_virtual);
    assert_eq!(commit_row.pc_before, 0);
    assert_eq!(commit_row.pc_after, 4);
    assert_eq!(commit_row.virtual_sequence_remaining, None);

    let halt_row = trace.steps.last().expect("halt row");
    assert!(!halt_row.is_virtual);
    let expected = compute_op(RiscvOpcode::Mulw, lhs, rhs, /*xlen=*/ 64);
    assert_eq!(halt_row.regs_after[4], expected);

    Rv32ExecTable::from_trace_padded_with_xlen(&trace, trace.steps.len(), /*machine_xlen=*/ 64)
        .expect("exec table must accept RV64 MULW virtual rows");
}

#[test]
fn mulw_runtime_trace_decomposes_negative_sign_extended_rv64_word_result() {
    let lhs = 0xffff_ffffu64;
    let rhs = 2u64;
    let trace = run_mulw_trace(lhs, rhs);
    assert_no_direct_w_arith_shout_events(&trace);

    assert_eq!(trace.steps.len(), 5);
    assert!(trace.steps[0].is_virtual);
    assert_eq!(trace.steps[0].virtual_sequence_remaining, Some(3));
    assert!(trace.steps[1].is_virtual);
    assert_eq!(trace.steps[1].virtual_sequence_remaining, Some(2));
    assert!(trace.steps[2].is_virtual);
    assert_eq!(trace.steps[2].virtual_sequence_remaining, Some(1));

    let expected = compute_op(RiscvOpcode::Mulw, lhs, rhs, /*xlen=*/ 64);
    let halt_row = trace.steps.last().expect("halt row");
    assert_eq!(halt_row.regs_after[4], expected);
    assert_eq!(expected, 0xffff_ffff_ffff_fffeu64);

    Rv32ExecTable::from_trace_padded_with_xlen(&trace, trace.steps.len(), /*machine_xlen=*/ 64)
        .expect("exec table must validate negative RV64 MULW virtual rows");
}

#[test]
fn divuw_runtime_trace_decomposes_positive_rv64_word_result() {
    let lhs = 21u64;
    let rhs = 5u64;
    let trace = run_divuw_trace(lhs, rhs);
    assert_no_direct_w_arith_shout_events(&trace);

    assert_eq!(trace.steps.len(), 5);
    assert!(trace.steps[0].is_virtual);
    assert_eq!(trace.steps[0].virtual_sequence_remaining, Some(3));
    assert!(trace.steps[1].is_virtual);
    assert_eq!(trace.steps[1].virtual_sequence_remaining, Some(2));
    assert!(trace.steps[2].is_virtual);
    assert_eq!(trace.steps[2].virtual_sequence_remaining, Some(1));

    let expected = compute_op(RiscvOpcode::Divuw, lhs, rhs, /*xlen=*/ 64);
    let halt_row = trace.steps.last().expect("halt row");
    assert_eq!(halt_row.regs_after[4], expected);

    Rv32ExecTable::from_trace_padded_with_xlen(&trace, trace.steps.len(), /*machine_xlen=*/ 64)
        .expect("exec table must accept RV64 DIVUW virtual rows");
}

#[test]
fn divw_runtime_trace_decomposes_positive_rv64_word_result() {
    let lhs = 21u64;
    let rhs = 5u64;
    let trace = run_divw_trace(lhs, rhs);

    assert_eq!(trace.steps.len(), 5);
    assert!(trace.steps[0].is_virtual);
    assert_eq!(trace.steps[0].virtual_sequence_remaining, Some(3));
    assert!(trace.steps[1].is_virtual);
    assert_eq!(trace.steps[1].virtual_sequence_remaining, Some(2));
    assert!(trace.steps[2].is_virtual);
    assert_eq!(trace.steps[2].virtual_sequence_remaining, Some(1));

    let expected = compute_op(RiscvOpcode::Divw, lhs, rhs, /*xlen=*/ 64);
    let halt_row = trace.steps.last().expect("halt row");
    assert_eq!(halt_row.regs_after[4], expected);

    Rv32ExecTable::from_trace_padded_with_xlen(&trace, trace.steps.len(), /*machine_xlen=*/ 64)
        .expect("exec table must accept RV64 DIVW virtual rows");
}

#[test]
fn remuw_runtime_trace_decomposes_positive_rv64_word_result() {
    let lhs = 21u64;
    let rhs = 5u64;
    let trace = run_remuw_trace(lhs, rhs);

    assert_eq!(trace.steps.len(), 5);
    assert!(trace.steps[0].is_virtual);
    assert_eq!(trace.steps[0].virtual_sequence_remaining, Some(3));
    assert!(trace.steps[1].is_virtual);
    assert_eq!(trace.steps[1].virtual_sequence_remaining, Some(2));
    assert!(trace.steps[2].is_virtual);
    assert_eq!(trace.steps[2].virtual_sequence_remaining, Some(1));

    let expected = compute_op(RiscvOpcode::Remuw, lhs, rhs, /*xlen=*/ 64);
    let halt_row = trace.steps.last().expect("halt row");
    assert_eq!(halt_row.regs_after[4], expected);

    Rv32ExecTable::from_trace_padded_with_xlen(&trace, trace.steps.len(), /*machine_xlen=*/ 64)
        .expect("exec table must accept RV64 REMUW virtual rows");
}

#[test]
fn remw_runtime_trace_decomposes_positive_rv64_word_result() {
    let lhs = 21u64;
    let rhs = 5u64;
    let trace = run_remw_trace(lhs, rhs);

    assert_eq!(trace.steps.len(), 5);
    assert!(trace.steps[0].is_virtual);
    assert_eq!(trace.steps[0].virtual_sequence_remaining, Some(3));
    assert!(trace.steps[1].is_virtual);
    assert_eq!(trace.steps[1].virtual_sequence_remaining, Some(2));
    assert!(trace.steps[2].is_virtual);
    assert_eq!(trace.steps[2].virtual_sequence_remaining, Some(1));

    let expected = compute_op(RiscvOpcode::Remw, lhs, rhs, /*xlen=*/ 64);
    let halt_row = trace.steps.last().expect("halt row");
    assert_eq!(halt_row.regs_after[4], expected);

    Rv32ExecTable::from_trace_padded_with_xlen(&trace, trace.steps.len(), /*machine_xlen=*/ 64)
        .expect("exec table must accept RV64 REMW virtual rows");
}
