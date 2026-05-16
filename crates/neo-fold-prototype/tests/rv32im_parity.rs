//! Focused tests for the sharded RV32IM exact-parity corpus.

use neo_fold_prototype::rv32im::build_all_parity_cases;
use std::collections::BTreeMap;

const INLINE_SCRATCH_REGISTER_BASE: u8 = 40;

#[test]
fn parity_cases_reach_expected_halted_states() {
    let cases = build_all_parity_cases().expect("RV32IM parity cases");

    for (_source, derived) in cases {
        assert!(derived.kernel.halted, "{} should halt on ECALL", derived.manifest.name);
        let mut rows_by_step = BTreeMap::new();
        for row in &derived.execution_rows {
            rows_by_step
                .entry(row.step_index)
                .or_insert_with(Vec::new)
                .push(row);
        }
        for rows in rows_by_step.values() {
            assert_eq!(rows.iter().filter(|row| row.is_effect_row).count(), 1);
            assert_eq!(rows.iter().filter(|row| row.is_commit_row).count(), 1);
            assert_eq!(rows.iter().filter(|row| row.is_real).count(), 1);
            let effect_index = rows
                .iter()
                .find(|row| row.is_effect_row)
                .map(|row| row.sequence_index)
                .expect("effect row");
            let commit_row = rows
                .iter()
                .find(|row| row.is_commit_row)
                .expect("commit row");
            assert!(effect_index <= commit_row.sequence_index);
            assert!(commit_row.is_real);
        }

        match derived.manifest.name.as_str() {
            "vertical_add_sw_lw_ecall" => {
                assert_eq!(derived.kernel.final_pc, 20);
                assert_eq!(derived.kernel.final_registers[1], 5);
                assert_eq!(derived.kernel.final_registers[2], 10);
                assert_eq!(derived.kernel.final_registers[3], 10);
                assert_eq!(derived.kernel.final_memory.len(), 1);
                assert_eq!(derived.kernel.final_memory[0].addr, 0x1000);
                assert_eq!(derived.kernel.final_memory[0].value, 10);
            }
            "native_add_chain_x0_ecall" => {
                assert_eq!(derived.kernel.final_pc, 20);
                assert_eq!(derived.kernel.final_registers[0], 0, "x0 sink must be preserved");
                assert_eq!(derived.kernel.final_registers[1], 7);
                assert_eq!(derived.kernel.final_registers[2], 16);
                assert_eq!(derived.kernel.final_registers[3], 23);
                assert!(derived.kernel.final_memory.is_empty());
            }
            "native_logic_compare_chain_ecall" => {
                assert_eq!(derived.kernel.final_pc, 56);
                assert_eq!(derived.kernel.final_registers[1], 5);
                assert_eq!(derived.kernel.final_registers[2], 3);
                assert_eq!(derived.kernel.final_registers[3], 1);
                assert_eq!(derived.kernel.final_registers[4], 4);
                assert_eq!(derived.kernel.final_registers[5], 7);
                assert_eq!(derived.kernel.final_registers[6], 11);
                assert_eq!(derived.kernel.final_registers[7], 6);
                assert_eq!(derived.kernel.final_registers[8], 2);
                assert_eq!(derived.kernel.final_registers[9], 1);
                assert_eq!(derived.kernel.final_registers[10], 1);
                assert_eq!(derived.kernel.final_registers[11], 1);
                assert_eq!(derived.kernel.final_registers[12], 0);
                assert!(derived.kernel.final_memory.is_empty());
            }
            "native_shift_chain_ecall" => {
                assert_eq!(derived.kernel.final_pc, 40);
                assert_eq!(derived.kernel.final_registers[1], 1);
                assert_eq!(derived.kernel.final_registers[2], 16);
                assert_eq!(derived.kernel.final_registers[3], u32::MAX - 15);
                assert_eq!(derived.kernel.final_registers[4], 4);
                assert_eq!(derived.kernel.final_registers[5], u32::MAX - 3);
                assert_eq!(derived.kernel.final_registers[6], 3);
                assert_eq!(derived.kernel.final_registers[7], 8);
                assert_eq!(derived.kernel.final_registers[8], 2);
                assert_eq!(derived.kernel.final_registers[9], u32::MAX - 1);
                assert!(derived.kernel.final_memory.is_empty());
            }
            "native_rv32_wrap_chain_ecall" => {
                assert_eq!(derived.kernel.final_pc, 20);
                assert_eq!(derived.kernel.final_registers[1], u32::MAX);
                assert_eq!(derived.kernel.final_registers[2], 1);
                assert_eq!(derived.kernel.final_registers[3], 0x7fff_ffff);
                assert_eq!(derived.kernel.final_registers[4], 2);
                assert_eq!(derived.kernel.final_registers[5], 0);
                assert_eq!(derived.kernel.final_registers[6], 1);
                assert_eq!(derived.kernel.final_registers[7], 0x8000_0001);
                assert_eq!(derived.kernel.final_registers[8], u32::MAX);
                assert!(derived.kernel.final_memory.is_empty());
            }
            "native_rv32_shift_mask_chain_ecall" => {
                assert_eq!(derived.kernel.final_pc, 28);
                assert_eq!(derived.kernel.final_registers[1], 1);
                assert_eq!(derived.kernel.final_registers[2], 0x8000_0000);
                assert_eq!(derived.kernel.final_registers[3], 0x8000_0000);
                assert_eq!(derived.kernel.final_registers[4], 0x0800_0000);
                assert_eq!(derived.kernel.final_registers[5], 0xf800_0000);
                assert_eq!(derived.kernel.final_registers[6], 40);
                assert_eq!(derived.kernel.final_registers[7], 0x100);
                assert_eq!(derived.kernel.final_registers[8], 0x0080_0000);
                assert_eq!(derived.kernel.final_registers[9], 0xff80_0000);
                assert!(derived.kernel.final_memory.is_empty());
            }
            "native_sub_lui_auipc_fence_ecall" => {
                assert_eq!(derived.kernel.final_pc, 28);
                assert_eq!(derived.kernel.final_registers[1], 9);
                assert_eq!(derived.kernel.final_registers[2], 4);
                assert_eq!(derived.kernel.final_registers[3], 5);
                assert_eq!(derived.kernel.final_registers[4], 0x1234_5000);
                assert_eq!(derived.kernel.final_registers[5], 0x2010);
                assert!(derived.kernel.final_memory.is_empty());
            }
            "narrow_memory_load_extract_extend_ecall" => {
                assert_eq!(derived.kernel.final_pc, 24);
                assert_eq!(derived.kernel.final_registers[1], u32::MAX);
                assert_eq!(derived.kernel.final_registers[2], 0x80);
                assert_eq!(derived.kernel.final_registers[3], 0xffff_80ff);
                assert_eq!(derived.kernel.final_registers[4], 0x807f);
                assert_eq!(derived.kernel.final_registers[5], 0x807f_80ff);
                assert_eq!(derived.kernel.final_registers[6], 0);
                assert_eq!(derived.kernel.final_registers[10], 0x3000);
                assert_eq!(derived.kernel.final_memory.len(), 1);
                assert_eq!(derived.kernel.final_memory[0].addr, 0x3000);
                assert_eq!(derived.kernel.final_memory[0].value, 0x807f_80ff);
            }
            "narrow_memory_store_blend_ecall" => {
                assert_eq!(derived.kernel.final_pc, 16);
                assert_eq!(derived.kernel.final_registers[1], u32::MAX);
                assert_eq!(derived.kernel.final_registers[2], 0x0123);
                assert_eq!(derived.kernel.final_registers[3], 0x1234_5067);
                assert_eq!(derived.kernel.final_registers[10], 0x4000);
                assert_eq!(derived.kernel.final_memory.len(), 2);
                assert_eq!(derived.kernel.final_memory[0].addr, 0x4000);
                assert_eq!(derived.kernel.final_memory[0].value, 0x0123_ff11);
                assert_eq!(derived.kernel.final_memory[1].addr, 0x4004);
                assert_eq!(derived.kernel.final_memory[1].value, 0x1234_5067);
            }
            "multiply_low_mul_ecall" => {
                assert_eq!(derived.execution_rows.len(), 4);
                assert_eq!(derived.stage1.rows.len(), derived.execution_rows.len());
                assert_eq!(derived.stage2.twist_links.len(), derived.execution_rows.len());
                assert_eq!(
                    derived.execution_rows[0].trace_opcode,
                    Some(neo_fold_prototype::rv32im::Rv32Opcode::Mul)
                );
                assert_eq!(derived.execution_rows[0].virtual_sequence_remaining, None);
                assert_eq!(
                    derived.execution_rows[1].trace_opcode,
                    Some(neo_fold_prototype::rv32im::Rv32Opcode::Mul)
                );
                assert_eq!(derived.execution_rows[1].virtual_sequence_remaining, Some(1));
                assert_eq!(
                    derived.execution_rows[2].trace_virtual_opcode,
                    Some(neo_fold_prototype::rv32im::Rv32TraceVirtualOpcode::SignExtendWord)
                );
                assert!(derived.execution_rows[2].is_effect_row);
                assert!(derived.execution_rows[2].is_commit_row);
                assert!(derived.execution_rows[2].is_real);
                assert_eq!(derived.kernel.final_pc, 12);
                assert_eq!(derived.kernel.final_registers[1], 3);
                assert_eq!(derived.kernel.final_registers[2], 5);
                assert_eq!(derived.kernel.final_registers[3], u32::MAX);
                assert_eq!(derived.kernel.final_registers[4], 5);
                assert_eq!(derived.kernel.final_registers[5], 15);
                assert_eq!(derived.kernel.final_registers[6], u32::MAX - 4);
                assert!(derived.kernel.final_memory.is_empty());
            }
            "multiply_high_mulh_mulhu_mulhsu_ecall" => {
                assert_eq!(derived.execution_rows.len(), 20);
                assert_eq!(derived.stage1.rows.len(), derived.execution_rows.len());
                assert_eq!(derived.stage2.twist_links.len(), derived.execution_rows.len());
                assert!(derived
                    .stage2
                    .register_writes
                    .iter()
                    .any(|event| event.reg >= INLINE_SCRATCH_REGISTER_BASE));
                assert!(derived
                    .stage2
                    .register_reads
                    .iter()
                    .any(|event| event.reg >= INLINE_SCRATCH_REGISTER_BASE));
                assert_eq!(
                    derived
                        .execution_rows
                        .iter()
                        .filter(|row| row.trace_virtual_opcode
                            == Some(neo_fold_prototype::rv32im::Rv32TraceVirtualOpcode::Movsign))
                        .count(),
                    3
                );
                assert_eq!(
                    derived
                        .execution_rows
                        .iter()
                        .filter(
                            |row| row.trace_opcode == Some(neo_fold_prototype::rv32im::Rv32Opcode::Mulh) && row.is_real
                        )
                        .count(),
                    0
                );
                assert_eq!(
                    derived
                        .execution_rows
                        .iter()
                        .filter(
                            |row| row.trace_opcode == Some(neo_fold_prototype::rv32im::Rv32Opcode::Mulhsu)
                                && row.is_real
                        )
                        .count(),
                    0
                );
                assert_eq!(
                    derived
                        .execution_rows
                        .iter()
                        .filter(|row| {
                            row.opcode == neo_fold_prototype::rv32im::Rv32Opcode::Mulhu
                                && row.virtual_sequence_remaining.is_none()
                        })
                        .count(),
                    1
                );
                assert!(!derived.execution_rows.iter().any(|row| {
                    row.trace_opcode == Some(neo_fold_prototype::rv32im::Rv32Opcode::Addi)
                        && row.rd >= INLINE_SCRATCH_REGISTER_BASE
                }));
                assert_eq!(derived.kernel.final_pc, 16);
                assert_eq!(derived.kernel.final_registers[1], (1u64 << 40) as u32);
                assert_eq!(derived.kernel.final_registers[2], (1u64 << 35) as u32);
                assert_eq!(derived.kernel.final_registers[3], (1u64 << 41) as u32);
                assert_eq!(derived.kernel.final_registers[4], (1u64 << 34) as u32);
                assert_eq!(derived.kernel.final_registers[5], (1u64 << 42) as u32);
                assert_eq!(derived.kernel.final_registers[6], (1u64 << 33) as u32);
                assert_eq!(derived.kernel.final_registers[7], 2048);
                assert_eq!(derived.kernel.final_registers[8], 2048);
                assert_eq!(derived.kernel.final_registers[9], 2048);
                assert!(derived.kernel.final_memory.is_empty());
            }
            "unsigned_divrem_chain_ecall" => {
                assert!(derived
                    .execution_rows
                    .iter()
                    .any(|row| row.trace_virtual_opcode.is_some()));
                assert!(derived
                    .execution_rows
                    .iter()
                    .any(|row| row.trace_virtual_opcode
                        == Some(neo_fold_prototype::rv32im::Rv32TraceVirtualOpcode::Advice)));
                assert!(!derived.execution_rows.iter().any(|row| {
                    row.trace_virtual_opcode
                        == Some(neo_fold_prototype::rv32im::Rv32TraceVirtualOpcode::AssertValidUnsignedRemainder)
                }));
                assert!(derived
                    .stage2
                    .register_writes
                    .iter()
                    .any(|event| event.reg >= INLINE_SCRATCH_REGISTER_BASE));
                assert!(derived
                    .stage2
                    .register_reads
                    .iter()
                    .any(|event| event.reg >= INLINE_SCRATCH_REGISTER_BASE));
                assert!(!derived.execution_rows.iter().any(|row| {
                    row.trace_opcode == Some(neo_fold_prototype::rv32im::Rv32Opcode::Addi)
                        && row.rd >= INLINE_SCRATCH_REGISTER_BASE
                }));
                assert_eq!(derived.kernel.final_pc, 36);
                assert_eq!(derived.kernel.final_registers[1], 20);
                assert_eq!(derived.kernel.final_registers[2], 6);
                assert_eq!(derived.kernel.final_registers[5], 3);
                assert_eq!(derived.kernel.final_registers[6], 2);
                assert_eq!(derived.kernel.final_registers[7], 0x5555_5555);
                assert_eq!(derived.kernel.final_registers[8], 0);
                assert_eq!(derived.kernel.final_registers[11], u32::MAX);
                assert_eq!(derived.kernel.final_registers[12], 9);
                assert_eq!(derived.kernel.final_registers[15], u32::MAX);
                assert_eq!(derived.kernel.final_registers[16], 0x8000_0001);
                assert!(derived.kernel.final_memory.is_empty());
            }
            "signed_divrem_chain_ecall" => {
                assert!(derived
                    .execution_rows
                    .iter()
                    .any(|row| row.trace_virtual_opcode.is_some()));
                assert!(derived
                    .execution_rows
                    .iter()
                    .any(|row| row.trace_virtual_opcode
                        == Some(neo_fold_prototype::rv32im::Rv32TraceVirtualOpcode::ChangeDivisor)));
                assert!(!derived.execution_rows.iter().any(|row| {
                    row.trace_virtual_opcode
                        == Some(neo_fold_prototype::rv32im::Rv32TraceVirtualOpcode::AssertSignedDivIdentity)
                }));
                assert!(derived
                    .stage2
                    .register_writes
                    .iter()
                    .any(|event| event.reg >= INLINE_SCRATCH_REGISTER_BASE));
                assert!(derived
                    .stage2
                    .register_reads
                    .iter()
                    .any(|event| event.reg >= INLINE_SCRATCH_REGISTER_BASE));
                assert!(!derived.execution_rows.iter().any(|row| {
                    row.trace_opcode == Some(neo_fold_prototype::rv32im::Rv32Opcode::Addi)
                        && row.rd >= INLINE_SCRATCH_REGISTER_BASE
                }));
                assert_eq!(derived.kernel.final_pc, 44);
                assert_eq!(derived.kernel.final_registers[5], u32::MAX - 2);
                assert_eq!(derived.kernel.final_registers[6], u32::MAX - 1);
                assert_eq!(derived.kernel.final_registers[7], 0x8000_0000);
                assert_eq!(derived.kernel.final_registers[8], 0);
                assert_eq!(derived.kernel.final_registers[11], u32::MAX - 1);
                assert_eq!(derived.kernel.final_registers[12], u32::MAX);
                assert_eq!(derived.kernel.final_registers[15], u32::MAX);
                assert_eq!(derived.kernel.final_registers[16], 7);
                assert_eq!(derived.kernel.final_registers[19], u32::MAX);
                assert_eq!(derived.kernel.final_registers[20], 0x8000_0001);
                assert!(derived.kernel.final_memory.is_empty());
            }
            "aligned_negative_offset_roundtrip" => {
                assert_eq!(derived.kernel.final_pc, 16);
                assert_eq!(derived.kernel.final_registers[1], 42);
                assert_eq!(derived.kernel.final_registers[2], 42);
                assert_eq!(derived.kernel.final_registers[10], 0x2008);
                assert_eq!(derived.kernel.final_memory.len(), 2);
                assert_eq!(derived.kernel.final_memory[0].addr, 0x2000);
                assert_eq!(derived.kernel.final_memory[0].value, 42);
                assert_eq!(derived.kernel.final_memory[1].addr, 0x2008);
                assert_eq!(derived.kernel.final_memory[1].value, 99);
            }
            "control_flow_ecall_only" => {
                assert_eq!(derived.kernel.final_pc, 4);
                assert!(derived.kernel.final_memory.is_empty());
                assert!(derived
                    .kernel
                    .final_registers
                    .iter()
                    .all(|value| *value == 0));
            }
            "control_flow_jal_skip_ecall" => {
                assert_eq!(derived.execution_rows.len(), 2);
                assert_eq!(derived.execution_rows[0].pc, 0);
                assert_eq!(derived.execution_rows[0].next_pc, 8);
                assert_eq!(derived.execution_rows[1].pc, 8);
                assert_eq!(derived.kernel.final_pc, 12);
                assert_eq!(derived.kernel.final_registers[1], 4);
                assert!(derived.kernel.final_memory.is_empty());
            }
            "control_flow_jalr_skip_ecall" => {
                assert_eq!(derived.execution_rows.len(), 2);
                assert_eq!(derived.execution_rows[0].pc, 0);
                assert_eq!(derived.execution_rows[0].next_pc, 8);
                assert_eq!(derived.execution_rows[1].pc, 8);
                assert_eq!(derived.kernel.final_pc, 12);
                assert_eq!(derived.kernel.final_registers[1], 4);
                assert_eq!(derived.kernel.final_registers[5], 8);
                assert!(derived.kernel.final_memory.is_empty());
            }
            "control_flow_beq_taken_skip_ecall" => {
                assert_eq!(derived.execution_rows.len(), 2);
                assert_eq!(derived.execution_rows[0].pc, 0);
                assert_eq!(derived.execution_rows[0].next_pc, 8);
                assert_eq!(derived.execution_rows[1].pc, 8);
                assert_eq!(derived.execution_rows[0].rs1, 1);
                assert_eq!(derived.execution_rows[0].rs2, 2);
                assert_eq!(derived.execution_rows[0].alu_result, 1);
                assert_eq!(derived.kernel.final_pc, 12);
                assert_eq!(derived.kernel.final_registers[1], 11);
                assert_eq!(derived.kernel.final_registers[2], 11);
                assert!(derived.kernel.final_memory.is_empty());
            }
            "control_flow_bne_taken_skip_ecall" => {
                assert_eq!(derived.execution_rows.len(), 2);
                assert_eq!(derived.execution_rows[0].pc, 0);
                assert_eq!(derived.execution_rows[0].next_pc, 8);
                assert_eq!(derived.execution_rows[1].pc, 8);
                assert_eq!(derived.execution_rows[0].rs1, 1);
                assert_eq!(derived.execution_rows[0].rs2, 2);
                assert_eq!(derived.execution_rows[0].alu_result, 1);
                assert_eq!(derived.kernel.final_pc, 12);
                assert_eq!(derived.kernel.final_registers[1], 11);
                assert_eq!(derived.kernel.final_registers[2], 12);
                assert!(derived.kernel.final_memory.is_empty());
            }
            "control_flow_blt_taken_skip_ecall" => {
                assert_eq!(derived.execution_rows.len(), 4);
                assert_eq!(derived.execution_rows[2].pc, 8);
                assert_eq!(derived.execution_rows[2].next_pc, 16);
                assert_eq!(derived.execution_rows[2].alu_result, 1);
                assert_eq!(derived.kernel.final_pc, 20);
                assert_eq!(derived.kernel.final_registers[1], u32::MAX);
                assert_eq!(derived.kernel.final_registers[2], 1);
                assert!(derived.kernel.final_memory.is_empty());
            }
            "control_flow_bge_taken_skip_ecall" => {
                assert_eq!(derived.execution_rows.len(), 4);
                assert_eq!(derived.execution_rows[2].pc, 8);
                assert_eq!(derived.execution_rows[2].next_pc, 16);
                assert_eq!(derived.execution_rows[2].alu_result, 1);
                assert_eq!(derived.kernel.final_pc, 20);
                assert_eq!(derived.kernel.final_registers[1], 1);
                assert_eq!(derived.kernel.final_registers[2], u32::MAX);
                assert!(derived.kernel.final_memory.is_empty());
            }
            "control_flow_bltu_taken_skip_ecall" => {
                assert_eq!(derived.execution_rows.len(), 4);
                assert_eq!(derived.execution_rows[2].pc, 8);
                assert_eq!(derived.execution_rows[2].next_pc, 16);
                assert_eq!(derived.execution_rows[2].alu_result, 1);
                assert_eq!(derived.kernel.final_pc, 20);
                assert_eq!(derived.kernel.final_registers[1], 1);
                assert_eq!(derived.kernel.final_registers[2], 2);
                assert!(derived.kernel.final_memory.is_empty());
            }
            "control_flow_bgeu_taken_skip_ecall" => {
                assert_eq!(derived.execution_rows.len(), 4);
                assert_eq!(derived.execution_rows[2].pc, 8);
                assert_eq!(derived.execution_rows[2].next_pc, 16);
                assert_eq!(derived.execution_rows[2].alu_result, 1);
                assert_eq!(derived.kernel.final_pc, 20);
                assert_eq!(derived.kernel.final_registers[1], 2);
                assert_eq!(derived.kernel.final_registers[2], 1);
                assert!(derived.kernel.final_memory.is_empty());
            }
            name => panic!("unexpected RV32IM parity case {name}"),
        }
    }
}

#[test]
fn parity_case_artifacts_are_deterministic() {
    let first = build_all_parity_cases().expect("first parity case set");
    let second = build_all_parity_cases().expect("second parity case set");

    assert_eq!(first, second, "RV32IM parity corpus should be deterministic");
}
