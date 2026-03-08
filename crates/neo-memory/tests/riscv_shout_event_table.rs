use std::collections::HashMap;

use neo_memory::riscv::exec_table::{Rv32ExecTable, Rv32ShoutEventTable};
use neo_memory::riscv::lookups::{
    decode_program, encode_program, RiscvCpu, RiscvInstruction, RiscvMemory, RiscvOpcode, RiscvShoutTables, PROG_ID,
};
use neo_memory::riscv::trace::extract_shout_lanes_over_time;
use neo_vm_trace::trace_program;

#[test]
fn rv32_shout_event_table_matches_fixed_lane_extract() {
    // Program:
    // - ADDI x1,x0,0x1234
    // - ADDI x2,x0,37
    // - SLL  x3,x1,x2          (shamt uses low 5 bits => 5)
    // - OR   x4,x1,x0
    // - HALT
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 0x1234,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 37,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Sll,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Or,
            rd: 4,
            rs1: 1,
            rs2: 0,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let decoded_program = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);

    let twist = RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);
    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 64).expect("trace_program");
    assert!(trace.did_halt(), "expected program to halt");

    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 8).expect("from_trace_padded_pow2");
    exec.validate_inactive_rows_are_empty()
        .expect("inactive rows");

    let shout_table_ids = vec![
        RiscvShoutTables::new(32).opcode_to_id(RiscvOpcode::Add).0,
        RiscvShoutTables::new(32).opcode_to_id(RiscvOpcode::Sll).0,
        RiscvShoutTables::new(32).opcode_to_id(RiscvOpcode::Or).0,
    ];
    let lanes = extract_shout_lanes_over_time(&exec, &shout_table_ids).expect("extract shout lanes");
    assert_eq!(lanes.len(), shout_table_ids.len());

    let table = Rv32ShoutEventTable::from_exec_table(&exec).expect("Rv32ShoutEventTable::from_exec_table");

    // Index events by (row_idx, shout_id); fixed-lane policy should make this unique.
    let mut by_row: HashMap<(usize, u32), (u128, u64)> = HashMap::new();
    for e in table.rows.iter() {
        assert!(
            by_row
                .insert((e.row_idx, e.shout_id), (e.key, e.value))
                .is_none(),
            "duplicate shout event at row_idx={} shout_id={}",
            e.row_idx,
            e.shout_id
        );
    }

    // For each provisioned shout lane, ensure the per-row key/value matches the event table.
    let t = exec.rows.len();
    let mut expected_event_count = 0usize;
    for (lane_idx, &shout_id) in shout_table_ids.iter().enumerate() {
        let lane = &lanes[lane_idx];
        for row_idx in 0..t {
            if lane.has_lookup[row_idx] {
                expected_event_count += 1;
                let (key, value) = by_row
                    .get(&(row_idx, shout_id))
                    .copied()
                    .unwrap_or_else(|| panic!("missing shout event row_idx={row_idx} shout_id={shout_id}"));
                assert_eq!(
                    key, lane.key[row_idx],
                    "key mismatch at row_idx={row_idx} shout_id={shout_id}"
                );
                assert_eq!(
                    value, lane.value[row_idx],
                    "value mismatch at row_idx={row_idx} shout_id={shout_id}"
                );
            }
        }
    }
    assert_eq!(table.rows.len(), expected_event_count, "unexpected shout event count");

    // Shift canonicalization: the SLL event rhs should be masked to 5 bits.
    let sll_id = RiscvShoutTables::new(32).opcode_to_id(RiscvOpcode::Sll).0;
    let sll_ev = table
        .rows
        .iter()
        .find(|e| e.shout_id == sll_id)
        .expect("expected SLL shout event");
    assert!(sll_ev.rhs <= 31, "expected canonicalized SLL rhs <= 31");
}

#[test]
fn rv32_shout_event_table_is_empty_for_halt_only_program() {
    let program = vec![RiscvInstruction::Halt];
    let program_bytes = encode_program(&program);

    let decoded_program = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);

    let twist = RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);
    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 8).expect("trace_program");
    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 4).expect("from_trace_padded_pow2");

    let table = Rv32ShoutEventTable::from_exec_table(&exec).expect("Rv32ShoutEventTable::from_exec_table");
    assert!(table.rows.is_empty(), "expected no shout events for HALT-only program");
}

#[test]
fn rv32_shout_event_table_single_row_for_single_xor() {
    // Program: XOR x1,x0,x0; HALT
    let program = vec![
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Xor,
            rd: 1,
            rs1: 0,
            rs2: 0,
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

    let table = Rv32ShoutEventTable::from_exec_table(&exec).expect("Rv32ShoutEventTable::from_exec_table");
    assert_eq!(table.rows.len(), 1, "expected exactly one shout event");

    let ev = &table.rows[0];
    let xor_id = RiscvShoutTables::new(32).opcode_to_id(RiscvOpcode::Xor).0;
    assert_eq!(ev.shout_id, xor_id);
    assert_eq!(ev.lhs, 0);
    assert_eq!(ev.rhs, 0);
    assert_eq!(ev.value, 0);
}
