use neo_memory::riscv::exec_table::{Rv32ExecTable, Rv32ShoutEventTable};
use neo_memory::riscv::lookups::{
    decode_program, encode_program, interleave_bits, RiscvCpu, RiscvInstruction, RiscvMemory, RiscvOpcode,
    RiscvShoutTables, PROG_ID,
};
use neo_vm_trace::trace_program;

fn rv32m_exec_table() -> Rv32ExecTable {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 3,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 5,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulh,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Divu,
            rd: 4,
            rs1: 2,
            rs2: 1,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Remu,
            rd: 5,
            rs1: 2,
            rs2: 1,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);
    let decoded_program = decode_program(&program_bytes).expect("decode_program");

    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    let twist = RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);
    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 32).expect("trace_program");

    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 8).expect("from_trace_padded_pow2");
    exec.validate_cycle_chain().expect("cycle chain");
    exec.validate_pc_chain().expect("pc chain");
    exec.validate_halted_tail().expect("halted tail");
    exec.validate_inactive_rows_are_empty()
        .expect("inactive rows");
    exec
}

#[test]
fn rv32_trace_shout_event_table_includes_rv32m_rows() {
    let exec = rv32m_exec_table();
    let events = Rv32ShoutEventTable::from_exec_table(&exec).expect("Rv32ShoutEventTable::from_exec_table");

    assert!(
        events
            .rows
            .iter()
            .any(|row| row.opcode == Some(RiscvOpcode::Mulh)),
        "expected MULH shout event row"
    );
    assert!(
        exec.rows.iter().any(|row| matches!(
            row.decoded,
            Some(RiscvInstruction::RAlu {
                op: RiscvOpcode::Divu,
                ..
            })
        )),
        "expected DIVU step in execution table"
    );
    assert!(
        exec.rows.iter().any(|row| matches!(
            row.decoded,
            Some(RiscvInstruction::RAlu {
                op: RiscvOpcode::Remu,
                ..
            })
        )),
        "expected REMU step in execution table"
    );
}

#[test]
fn rv32_trace_shout_event_table_mulh_key_matches_operands() {
    let exec = rv32m_exec_table();
    let events = Rv32ShoutEventTable::from_exec_table(&exec).expect("Rv32ShoutEventTable::from_exec_table");

    let mulh_row = events
        .rows
        .iter()
        .find(|row| row.opcode == Some(RiscvOpcode::Mulh))
        .expect("expected MULH row");

    let expected_key = interleave_bits(/*lhs=*/ 3, /*rhs=*/ 5);
    assert_eq!(mulh_row.key, expected_key, "MULH key must encode rs1/rs2 values");
}
