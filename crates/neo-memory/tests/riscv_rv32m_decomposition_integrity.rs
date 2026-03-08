use neo_memory::riscv::exec_table::{Rv32ExecRow, Rv32ExecTable, Rv32ShoutEventRow, Rv32ShoutEventTable};
use neo_memory::riscv::instruction::encode_lookup_key;
use neo_memory::riscv::lookups::{
    compute_op, decode_program, encode_program, RiscvCpu, RiscvInstruction, RiscvMemory, RiscvOpcode, RiscvShoutTables,
    PROG_ID, REG_ID,
};
use neo_vm_trace::trace_program;
use neo_vm_trace::Twist;

fn is_rv32m_opcode(op: RiscvOpcode) -> bool {
    matches!(
        op,
        RiscvOpcode::Mul
            | RiscvOpcode::Mulh
            | RiscvOpcode::Mulhu
            | RiscvOpcode::Mulhsu
            | RiscvOpcode::Div
            | RiscvOpcode::Divu
            | RiscvOpcode::Rem
            | RiscvOpcode::Remu
    )
}

fn trace_with_runtime_decomposition(
    program: Vec<RiscvInstruction>,
    reg_init: &[(u64, u64)],
) -> Result<Rv32ExecTable, String> {
    let program_bytes = encode_program(&program);
    let decoded_program = decode_program(&program_bytes)?;

    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    cpu.set_runtime_decomposition_enabled(true);

    let mut twist =
        RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    for &(reg, val) in reg_init {
        twist.store(REG_ID, reg, val);
    }
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);

    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 256)?;
    let exec = Rv32ExecTable::from_trace(&trace)?;
    exec.validate_cycle_chain()?;
    exec.validate_pc_chain()?;
    exec.validate_halted_tail()?;
    Ok(exec)
}

fn rv32m_commit_rows(exec: &Rv32ExecTable) -> Vec<(usize, &Rv32ExecRow, RiscvOpcode)> {
    exec.rows
        .iter()
        .enumerate()
        .filter_map(|(row_idx, row)| {
            if !row.active || row.is_virtual {
                return None;
            }
            let op = match row.decoded.as_ref() {
                Some(RiscvInstruction::RAlu { op, .. }) => *op,
                _ => return None,
            };
            if !is_rv32m_opcode(op) {
                return None;
            }
            Some((row_idx, row, op))
        })
        .collect()
}

fn find_shout_event<'a>(
    table: &'a Rv32ShoutEventTable,
    row_idx: usize,
    op: RiscvOpcode,
) -> Result<&'a Rv32ShoutEventRow, String> {
    let mut matches = table
        .rows
        .iter()
        .filter(|ev| ev.row_idx == row_idx && ev.opcode == Some(op));
    let Some(first) = matches.next() else {
        return Err(format!("missing shout event for row_idx={row_idx}, opcode={op:?}"));
    };
    if matches.next().is_some() {
        return Err(format!("duplicate shout events for row_idx={row_idx}, opcode={op:?}"));
    }
    Ok(first)
}

fn assert_commit_row_semantics(
    row_idx: usize,
    row: &Rv32ExecRow,
    op: RiscvOpcode,
    expect_write: bool,
    expected_rd: u8,
    expected_rs1: u8,
    expected_rs2: u8,
) -> Result<(), String> {
    if row.is_virtual {
        return Err(format!("row {row_idx} unexpectedly marked virtual"));
    }
    if row.virtual_sequence_remaining.is_some() {
        return Err(format!(
            "row {row_idx} commit row should have no virtual_sequence_remaining"
        ));
    }
    if row.fields.rd != expected_rd || row.fields.rs1 != expected_rs1 || row.fields.rs2 != expected_rs2 {
        return Err(format!(
            "row {row_idx} decoded field mismatch: got rd/rs1/rs2={}/{}/{} expected={}/{}/{}",
            row.fields.rd, row.fields.rs1, row.fields.rs2, expected_rd, expected_rs1, expected_rs2
        ));
    }

    let rs1 = row
        .reg_read_lane0
        .as_ref()
        .ok_or_else(|| format!("row {row_idx} missing reg_read_lane0"))?;
    let rs2 = row
        .reg_read_lane1
        .as_ref()
        .ok_or_else(|| format!("row {row_idx} missing reg_read_lane1"))?;
    if rs1.addr != expected_rs1 as u64 || rs2.addr != expected_rs2 as u64 {
        return Err(format!(
            "row {row_idx} read addr mismatch: got lane0/lane1={}/{} expected={}/{}",
            rs1.addr, rs2.addr, expected_rs1, expected_rs2
        ));
    }

    let expected = compute_op(op, rs1.value, rs2.value, /*xlen=*/ 32);
    match (&row.reg_write_lane0, expect_write) {
        (Some(w), true) => {
            if w.addr != expected_rd as u64 || w.value != expected {
                return Err(format!(
                    "row {row_idx} write mismatch: got addr/value={:#x}/{:#x} expected={:#x}/{:#x}",
                    w.addr, w.value, expected_rd, expected
                ));
            }
        }
        (None, false) => {}
        (Some(_), false) => {
            return Err(format!("row {row_idx} unexpectedly has a write event"));
        }
        (None, true) => {
            return Err(format!("row {row_idx} missing expected write event"));
        }
    }

    if row.shout_events.len() != 1 {
        return Err(format!(
            "row {row_idx} expected exactly one shout event (got {})",
            row.shout_events.len()
        ));
    }
    let ev = &row.shout_events[0];
    let shout_id = RiscvShoutTables::new(32).opcode_to_id(op).0;
    let expected_key = encode_lookup_key(op, rs1.value, rs2.value, /*xlen=*/ 32);
    if ev.shout_id.0 != shout_id || ev.key != expected_key || ev.value != expected {
        return Err(format!(
            "row {row_idx} shout mismatch: got id/key/value={}/{:#x}/{:#x} expected={}/{:#x}/{:#x}",
            ev.shout_id.0, ev.key, ev.value, shout_id, expected_key, expected
        ));
    }
    Ok(())
}

#[test]
fn rv32m_decomposition_commit_rows_match_semantics_unsigned_and_x0_write_policy() {
    let program = vec![
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mul,
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
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mul,
            rd: 0,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];

    let exec = trace_with_runtime_decomposition(program, &[(1, 3), (2, 5)]).expect("trace+exec");
    let commits = rv32m_commit_rows(&exec);
    assert_eq!(commits.len(), 4, "expected one non-virtual commit row per RV32M op");

    let expected = [
        (RiscvOpcode::Mul, true, 3u8, 1u8, 2u8),
        (RiscvOpcode::Divu, true, 4u8, 2u8, 1u8),
        (RiscvOpcode::Remu, true, 5u8, 2u8, 1u8),
        (RiscvOpcode::Mul, false, 0u8, 1u8, 2u8),
    ];
    for ((row_idx, row, op), (exp_op, exp_write, exp_rd, exp_rs1, exp_rs2)) in commits.iter().zip(expected.iter()) {
        assert_eq!(*op, *exp_op, "unexpected opcode at commit row {}", row_idx);
        assert_commit_row_semantics(*row_idx, row, *op, *exp_write, *exp_rd, *exp_rs1, *exp_rs2)
            .expect("commit semantics");
    }

    // Cross-check via shout-event table: each RV32M commit row should contribute exactly one
    // row with matching opcode/operands/result.
    let table = Rv32ShoutEventTable::from_exec_table(&exec).expect("shout event table");
    for (row_idx, row, op) in commits {
        let rs1 = row.reg_read_lane0.as_ref().expect("lane0 read");
        let rs2 = row.reg_read_lane1.as_ref().expect("lane1 read");
        let expected_val = compute_op(op, rs1.value, rs2.value, /*xlen=*/ 32);
        let ev = find_shout_event(&table, row_idx, op).expect("missing/duplicate shout event");
        assert_eq!(ev.lhs, rs1.value);
        assert_eq!(ev.rhs, rs2.value);
        assert_eq!(ev.value, expected_val);
    }
}

#[test]
fn rv32m_decomposition_commit_rows_match_semantics_signed_mix() {
    let program = vec![
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulh,
            rd: 6,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulhsu,
            rd: 7,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Div,
            rd: 8,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Rem,
            rd: 9,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];

    // rs1 = -7 (two's complement RV32), rs2 = +3.
    let exec = trace_with_runtime_decomposition(program, &[(1, 0xFFFF_FFF9), (2, 3)]).expect("trace+exec");
    let commits = rv32m_commit_rows(&exec);
    assert_eq!(commits.len(), 4, "expected one non-virtual commit row per RV32M op");

    let expected = [
        (RiscvOpcode::Mulh, 6u8),
        (RiscvOpcode::Mulhsu, 7u8),
        (RiscvOpcode::Div, 8u8),
        (RiscvOpcode::Rem, 9u8),
    ];
    for ((row_idx, row, op), (exp_op, exp_rd)) in commits.iter().zip(expected.iter()) {
        assert_eq!(*op, *exp_op, "unexpected opcode at commit row {}", row_idx);
        assert_commit_row_semantics(*row_idx, row, *op, /*expect_write=*/ true, *exp_rd, 1, 2)
            .expect("commit semantics");
    }

    let table = Rv32ShoutEventTable::from_exec_table(&exec).expect("shout event table");
    for (row_idx, row, op) in commits {
        let rs1 = row.reg_read_lane0.as_ref().expect("lane0 read");
        let rs2 = row.reg_read_lane1.as_ref().expect("lane1 read");
        let expected_val = compute_op(op, rs1.value, rs2.value, /*xlen=*/ 32);
        let ev = find_shout_event(&table, row_idx, op).expect("missing/duplicate shout event");
        assert_eq!(ev.lhs, rs1.value);
        assert_eq!(ev.rhs, rs2.value);
        assert_eq!(ev.value, expected_val);
    }
}

#[test]
fn rv32m_decomposition_commit_rows_absent_for_non_rv32m_program() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Halt,
    ];

    let exec = trace_with_runtime_decomposition(program, &[]).expect("trace+exec");
    let commits = rv32m_commit_rows(&exec);
    assert!(commits.is_empty(), "expected no RV32M commit rows");

    let table = Rv32ShoutEventTable::from_exec_table(&exec).expect("shout event table");
    assert!(
        table
            .rows
            .iter()
            .all(|ev| !ev.opcode.is_some_and(is_rv32m_opcode)),
        "expected no RV32M shout events"
    );
}

#[test]
fn rv32m_decomposition_virtual_rows_stay_non_arch_and_aliasing_reads_are_stable() {
    let init_x1 = 0x8000_0001u64;
    let init_x2 = 0x7FFF_FFFEu64;
    let program = vec![
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulh,
            rd: 1,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulhsu,
            rd: 2,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];

    let exec = trace_with_runtime_decomposition(program, &[(1, init_x1), (2, init_x2)]).expect("trace+exec");

    for (row_idx, row) in exec.rows.iter().enumerate() {
        if !row.active || !row.is_virtual {
            continue;
        }
        if let Some(w) = row.reg_write_lane0.as_ref() {
            assert!(
                w.addr >= 32,
                "virtual row {} wrote architectural register addr={}",
                row_idx,
                w.addr
            );
        }
    }

    let commits = rv32m_commit_rows(&exec);
    assert_eq!(commits.len(), 2, "expected one non-virtual commit row per RV32M op");

    let (mulh_row_idx, mulh_row, mulh_op) = commits[0];
    assert_eq!(mulh_op, RiscvOpcode::Mulh);
    let mulh_rs1 = mulh_row.reg_read_lane0.as_ref().expect("mulh lane0 read");
    assert_eq!(mulh_rs1.addr, 1);
    assert_eq!(
        mulh_rs1.value, init_x1,
        "mulh commit row must read original rs1 value even when rd==rs1"
    );
    assert_commit_row_semantics(mulh_row_idx, mulh_row, mulh_op, /*expect_write=*/ true, 1, 1, 2)
        .expect("mulh commit semantics");

    let (mulhsu_row_idx, mulhsu_row, mulhsu_op) = commits[1];
    assert_eq!(mulhsu_op, RiscvOpcode::Mulhsu);
    let mulhsu_rs2 = mulhsu_row
        .reg_read_lane1
        .as_ref()
        .expect("mulhsu lane1 read");
    assert_eq!(mulhsu_rs2.addr, 2);
    assert_eq!(
        mulhsu_rs2.value, init_x2,
        "mulhsu commit row must read original rs2 value even when rd==rs2"
    );
    assert_commit_row_semantics(
        mulhsu_row_idx,
        mulhsu_row,
        mulhsu_op,
        /*expect_write=*/ true,
        2,
        1,
        2,
    )
    .expect("mulhsu commit semantics");
}

#[test]
fn rv32m_decomposition_commit_rows_link_to_last_virtual_write_value() {
    let program = vec![
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulh,
            rd: 6,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulhsu,
            rd: 7,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Divu,
            rd: 8,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Remu,
            rd: 9,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Div,
            rd: 10,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Rem,
            rd: 11,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];

    let exec = trace_with_runtime_decomposition(program, &[(1, 0xFFFF_FFF9), (2, 3)]).expect("trace+exec");
    let commits = rv32m_commit_rows(&exec);
    assert_eq!(
        commits.len(),
        6,
        "expected one non-virtual commit row per decomposed RV32M op"
    );

    for (row_idx, row, op) in commits {
        assert!(row_idx > 0, "commit row index must have a predecessor");
        let prev = &exec.rows[row_idx - 1];
        assert!(
            prev.active && prev.is_virtual,
            "commit row {row_idx} ({op:?}) must be preceded by an active virtual row"
        );
        let prev_write = prev
            .reg_write_lane0
            .as_ref()
            .unwrap_or_else(|| panic!("commit row {row_idx} ({op:?}) predecessor must write a virtual value"));
        let commit_write = row
            .reg_write_lane0
            .as_ref()
            .unwrap_or_else(|| panic!("commit row {row_idx} ({op:?}) must write architectural rd"));

        assert!(
            prev_write.addr >= 32,
            "predecessor write for commit row {row_idx} ({op:?}) must target virtual addr"
        );
        assert!(
            commit_write.addr < 32,
            "commit row {row_idx} ({op:?}) must target architectural rd"
        );
        assert_eq!(
            prev_write.value, commit_write.value,
            "commit row {row_idx} ({op:?}) must copy last virtual write value into rd"
        );
    }
}

#[test]
fn rv32m_virtual_semantics_validator_rejects_tampered_exec_row() {
    let program = vec![
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulh,
            rd: 5,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let mut exec =
        trace_with_runtime_decomposition(program, &[(1, 0x8000_0001), (2, 0x7FFF_FFFF)]).expect("trace+exec");

    let first_virtual_idx = exec
        .rows
        .iter()
        .enumerate()
        .find_map(|(i, r)| if r.active && r.is_virtual { Some(i) } else { None })
        .expect("expected virtual row");
    let lane0 = exec.rows[first_virtual_idx]
        .reg_read_lane0
        .as_mut()
        .expect("virtual row lane0 read");
    lane0.addr = lane0.addr.wrapping_add(1);

    let err = exec
        .validate_virtual_decomposition_semantics(/*machine_xlen=*/ 32)
        .expect_err("tampered virtual row should fail semantic validation");
    assert!(err.contains("read addr mismatch"), "unexpected error: {err}");
}

#[test]
fn rv32m_from_trace_rejects_tampered_virtual_micro_op_shape() {
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
    let mut trace = trace_program(cpu, twist, shout, /*max_steps=*/ 64).expect("trace_program");

    let first_virtual_idx = trace
        .steps
        .iter()
        .enumerate()
        .find_map(|(i, s)| if s.is_virtual { Some(i) } else { None })
        .expect("expected virtual step");
    let read_lane0 = trace.steps[first_virtual_idx]
        .twist_events
        .iter_mut()
        .find(|ev| ev.twist_id == REG_ID && matches!(ev.kind, neo_vm_trace::TwistOpKind::Read) && ev.lane == Some(0))
        .expect("virtual step lane0 read");
    read_lane0.addr = read_lane0.addr.wrapping_add(1);

    let err = Rv32ExecTable::from_trace(&trace).expect_err("tampered trace should be rejected by from_trace");
    assert!(err.contains("read addr mismatch"), "unexpected error: {err}");
}
