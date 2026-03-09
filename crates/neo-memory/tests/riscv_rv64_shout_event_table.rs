use std::collections::HashMap;

use neo_memory::riscv::exec_table::{RiscvExecTable, Rv64ShoutEventTable};
use neo_memory::riscv::lookups::{
    decode_program, encode_program, RiscvCpu, RiscvInstruction, RiscvMemory, RiscvOpcode, RiscvShoutTables, PROG_ID,
};
use neo_memory::riscv::trace::extract_shout_lanes_over_time;
use neo_vm_trace::trace_program;

#[test]
fn rv64_shout_event_table_matches_fixed_lane_extract() {
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
            imm: 70,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Sll,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 4,
            rs1: 0,
            imm: 37,
        },
        RiscvInstruction::RAluw {
            op: RiscvOpcode::Sllw,
            rd: 5,
            rs1: 1,
            rs2: 4,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Or,
            rd: 6,
            rs1: 1,
            rs2: 0,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let decoded_program = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(/*xlen=*/ 64);
    cpu.load_program(/*base=*/ 0, decoded_program);

    let twist = RiscvMemory::with_program_in_twist(/*xlen=*/ 64, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    let shout = RiscvShoutTables::new(/*xlen=*/ 64);
    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 64).expect("trace_program");
    assert!(trace.did_halt(), "expected program to halt");

    let exec = RiscvExecTable::from_trace_padded_pow2_with_xlen(&trace, /*min_len=*/ 8, /*machine_xlen=*/ 64)
        .expect("from_trace_padded_pow2_with_xlen");
    exec.validate_inactive_rows_are_empty()
        .expect("inactive rows");

    let shout_tables = RiscvShoutTables::new(64);
    let shout_table_ids = vec![
        shout_tables.opcode_to_id(RiscvOpcode::Add).0,
        shout_tables.opcode_to_id(RiscvOpcode::Sll).0,
        shout_tables.opcode_to_id(RiscvOpcode::Sllw).0,
        shout_tables.opcode_to_id(RiscvOpcode::Or).0,
    ];
    let lanes = extract_shout_lanes_over_time(&exec, &shout_table_ids, /*xlen=*/ 64).expect("extract shout lanes");
    assert_eq!(lanes.len(), shout_table_ids.len());

    let table = Rv64ShoutEventTable::from_exec_table(&exec).expect("Rv64ShoutEventTable::from_exec_table");

    let mut by_row: HashMap<(usize, u32), (u128, u64)> = HashMap::new();
    for e in &table.rows {
        assert!(
            by_row
                .insert((e.row_idx, e.shout_id), (e.key, e.value))
                .is_none(),
            "duplicate shout event at row_idx={} shout_id={}",
            e.row_idx,
            e.shout_id
        );
    }

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

    let sll_id = shout_tables.opcode_to_id(RiscvOpcode::Sll).0;
    let sll_ev = table
        .rows
        .iter()
        .find(|e| e.shout_id == sll_id)
        .expect("expected SLL shout event");
    assert!(sll_ev.rhs <= 63, "expected canonicalized RV64 SLL rhs <= 63");

    let sllw_id = shout_tables.opcode_to_id(RiscvOpcode::Sllw).0;
    let sllw_ev = table
        .rows
        .iter()
        .find(|e| e.shout_id == sllw_id)
        .expect("expected SLLW shout event");
    assert!(sllw_ev.rhs <= 31, "expected canonicalized RV64 SLLW rhs <= 31");
}

#[test]
fn rv64_width_lookup_group_has_addr_dispatch_without_shared_ownership() {
    use neo_memory::riscv::trace::rv64_width_sidecar::{
        rv64_width_lookup_addr_group_for_table_id, RV64_TRACE_WIDTH_LOOKUP_GROUPED_TABLE_ID,
    };
    use neo_memory::riscv::trace::{
        riscv_trace_shared_width_lookup_addr_group_for_table_id, riscv_trace_uses_shared_width_lookup_table_id,
    };

    assert_eq!(
        riscv_trace_shared_width_lookup_addr_group_for_table_id(RV64_TRACE_WIDTH_LOOKUP_GROUPED_TABLE_ID),
        rv64_width_lookup_addr_group_for_table_id(RV64_TRACE_WIDTH_LOOKUP_GROUPED_TABLE_ID)
    );
    assert!(
        !riscv_trace_uses_shared_width_lookup_table_id(RV64_TRACE_WIDTH_LOOKUP_GROUPED_TABLE_ID),
        "RV64 width tables should route to the dedicated RV64 width stage, not the shared RV32 width stage"
    );
}
