use std::collections::HashMap;

use neo_memory::riscv::exec_table::Rv32ExecTable;
use neo_memory::riscv::lookups::{
    decode_program, encode_program, RiscvCpu, RiscvInstruction, RiscvMemOp, RiscvMemory, RiscvOpcode, RiscvShoutTables,
    PROG_ID,
};
use neo_memory::riscv::trace::{extract_shout_lanes_over_time, extract_twist_lanes_over_time};
use neo_vm_trace::trace_program;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

fn build_exec_table() -> (Rv32ExecTable, Vec<u32>) {
    // Program:
    // - ADDI x1, x0, 1
    // - SW x1, 0(x0)
    // - LW x2, 0(x0)
    // - HALT
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Store {
            op: RiscvMemOp::Sw,
            rs1: 0,
            rs2: 1,
            imm: 0,
        },
        RiscvInstruction::Load {
            op: RiscvMemOp::Lw,
            rd: 2,
            rs1: 0,
            imm: 0,
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

    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 4).expect("from_trace_padded_pow2");
    exec.validate_cycle_chain().expect("cycle chain");
    exec.validate_pc_chain().expect("pc chain");
    exec.validate_halted_tail().expect("halted tail");
    exec.validate_inactive_rows_are_empty()
        .expect("inactive rows");

    let shout_table_ids = vec![RiscvShoutTables::new(32).opcode_to_id(RiscvOpcode::Add).0];
    (exec, shout_table_ids)
}

#[test]
fn trace_sidecar_extract_smoke() {
    let (exec, shout_table_ids) = build_exec_table();

    // Keep it tiny: RAM addresses in this program are 0, so ell_addr=2 is enough.
    let init_regs: HashMap<u64, u64> = HashMap::new();
    let init_ram: HashMap<u64, u64> = HashMap::new();
    let twist =
        extract_twist_lanes_over_time(&exec, &init_regs, &init_ram, /*ram_ell_addr=*/ 2).expect("twist extract");
    let shout = extract_shout_lanes_over_time(&exec, &shout_table_ids, /*xlen=*/ 32).expect("shout extract");

    assert_eq!(twist.prog.has_read.len(), exec.rows.len());
    assert_eq!(twist.reg_lane0.has_read.len(), exec.rows.len());
    assert_eq!(twist.reg_lane1.has_read.len(), exec.rows.len());
    assert_eq!(twist.ram.has_read.len(), exec.rows.len());
    assert_eq!(shout.len(), 1);
    assert_eq!(shout[0].has_lookup.len(), exec.rows.len());

    // Inactive tail must be all zero/false.
    for (i, row) in exec.rows.iter().enumerate() {
        if row.active {
            continue;
        }
        assert!(!twist.prog.has_read[i]);
        assert!(!twist.reg_lane0.has_read[i]);
        assert!(!twist.reg_lane1.has_read[i]);
        assert!(!twist.reg_lane0.has_write[i]);
        assert!(!twist.ram.has_read[i]);
        assert!(!twist.ram.has_write[i]);
        assert_eq!(twist.reg_lane0.inc_at_write_addr[i], F::ZERO);
        assert_eq!(twist.ram.inc_at_write_addr[i], F::ZERO);
        assert!(!shout[0].has_lookup[i]);
        assert_eq!(shout[0].key[i], 0);
        assert_eq!(shout[0].value[i], 0);
    }

    // Sanity: should contain at least one RAM write (SW) and one RAM read (LW).
    assert!(twist.ram.has_write.iter().any(|&b| b), "expected a RAM write");
    assert!(twist.ram.has_read.iter().any(|&b| b), "expected a RAM read");
}

#[test]
fn trace_sidecar_extract_rejects_multiple_shout_events() {
    let (mut exec, shout_table_ids) = build_exec_table();

    let first_active = exec
        .rows
        .iter()
        .position(|r| r.active)
        .expect("must have active rows");
    let ev = neo_vm_trace::ShoutEvent::<u128, u64> {
        shout_id: neo_vm_trace::ShoutId(shout_table_ids[0]),
        key: 0,
        value: 0,
    };
    exec.rows[first_active].shout_events.push(ev.clone());
    exec.rows[first_active].shout_events.push(ev);

    let err = extract_shout_lanes_over_time(&exec, &shout_table_ids, /*xlen=*/ 32).unwrap_err();
    assert!(err.contains("multiple Shout events"), "{err}");
}

#[test]
fn trace_sidecar_extract_rejects_multiple_ram_writes() {
    let (mut exec, _shout_table_ids) = build_exec_table();

    let sw_row = exec
        .rows
        .iter()
        .position(|r| {
            r.ram_events
                .iter()
                .any(|e| matches!(e.kind, neo_vm_trace::TwistOpKind::Write))
        })
        .expect("expected a RAM write row");
    let write_ev = exec.rows[sw_row]
        .ram_events
        .iter()
        .find(|e| matches!(e.kind, neo_vm_trace::TwistOpKind::Write))
        .cloned()
        .expect("write event");
    exec.rows[sw_row].ram_events.push(write_ev);

    let init_regs: HashMap<u64, u64> = HashMap::new();
    let init_ram: HashMap<u64, u64> = HashMap::new();
    let err = extract_twist_lanes_over_time(&exec, &init_regs, &init_ram, /*ram_ell_addr=*/ 2).unwrap_err();
    assert!(err.contains("multiple RAM writes"), "{err}");
}

#[test]
fn trace_sidecar_extract_rejects_read_write_addr_mismatch() {
    let (mut exec, _shout_table_ids) = build_exec_table();

    let sw_row = exec
        .rows
        .iter()
        .position(|r| {
            r.ram_events
                .iter()
                .any(|e| matches!(e.kind, neo_vm_trace::TwistOpKind::Write))
        })
        .expect("expected a RAM write row");
    let write_ev = exec.rows[sw_row]
        .ram_events
        .iter()
        .find(|e| matches!(e.kind, neo_vm_trace::TwistOpKind::Write))
        .cloned()
        .expect("write event");

    let mut read_ev = write_ev.clone();
    read_ev.kind = neo_vm_trace::TwistOpKind::Read;
    read_ev.addr = write_ev.addr + 1;
    exec.rows[sw_row].ram_events.push(read_ev);

    let init_regs: HashMap<u64, u64> = HashMap::new();
    let init_ram: HashMap<u64, u64> = HashMap::new();
    let err = extract_twist_lanes_over_time(&exec, &init_regs, &init_ram, /*ram_ell_addr=*/ 2).unwrap_err();
    assert!(err.contains("RAM read/write addr mismatch"), "{err}");
}
