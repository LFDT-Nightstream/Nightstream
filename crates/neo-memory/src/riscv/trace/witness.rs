use neo_vm_trace::TwistOpKind;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

use crate::riscv::exec_table::Rv32ExecTable;
use crate::riscv::instruction::{
    opcode_uses_combined_lookup_key, operand_mode_keys_enabled, try_decode_lookup_operands,
};
use crate::riscv::lookups::{RiscvInstruction, RiscvOpcode, RiscvShoutTables};

use super::layout::Rv32TraceLayout;

#[inline]
fn sign_extend_to_u32(value: u32, bits: u32) -> u32 {
    let shift = 32 - bits;
    (((value << shift) as i32) >> shift) as u32
}

#[inline]
fn imm_i_from_word(instr_word: u32) -> u32 {
    sign_extend_to_u32((instr_word >> 20) & 0x0fff, 12)
}

#[inline]
fn lookup_operands_from_decoded(decoded: &RiscvInstruction, rs1_val: u64, rs2_val: u64, pc_before: u64) -> (u64, u64) {
    match decoded {
        RiscvInstruction::RAlu { .. } | RiscvInstruction::RAluw { .. } => (rs1_val, rs2_val),
        RiscvInstruction::IAlu { imm, .. } | RiscvInstruction::IAluw { imm, .. } => (rs1_val, *imm as u32 as u64),
        RiscvInstruction::Load { imm, .. } | RiscvInstruction::Jalr { imm, .. } => (rs1_val, *imm as u32 as u64),
        RiscvInstruction::Store { imm, .. } => (rs1_val, *imm as u32 as u64),
        RiscvInstruction::Auipc { imm, .. } => (pc_before, ((*imm as i64 as u64) << 12) as u32 as u64),
        RiscvInstruction::Branch { .. } => (rs1_val, rs2_val),
        _ => (rs1_val, rs2_val),
    }
}

#[derive(Clone, Debug)]
pub struct Rv32TraceWitness {
    pub t: usize,
    /// Column-major: `cols[col][row]`.
    pub cols: Vec<Vec<F>>,
}

impl Rv32TraceWitness {
    pub fn new_zero(layout: &Rv32TraceLayout, t: usize) -> Self {
        Self {
            t,
            cols: vec![vec![F::ZERO; t]; layout.cols],
        }
    }

    pub fn from_exec_table(layout: &Rv32TraceLayout, exec: &Rv32ExecTable) -> Result<Self, String> {
        let cols = exec.to_columns();
        let t = cols.len();
        let mut wit = Self::new_zero(layout, t);

        for i in 0..t {
            wit.cols[layout.one][i] = F::ONE;

            // Control / fetch
            wit.cols[layout.active][i] = if cols.active[i] { F::ONE } else { F::ZERO };
            wit.cols[layout.halted][i] = if cols.halted[i] { F::ONE } else { F::ZERO };
            wit.cols[layout.is_virtual][i] = if cols.is_virtual[i] { F::ONE } else { F::ZERO };
            wit.cols[layout.virtual_sequence_remaining][i] = F::from_u64(cols.virtual_sequence_remaining[i]);
            wit.cols[layout.virtual_transition][i] = if cols.virtual_transition[i] { F::ONE } else { F::ZERO };
            wit.cols[layout.virtual_commit_link][i] = if cols.virtual_commit_link[i] { F::ONE } else { F::ZERO };
            wit.cols[layout.virtual_commit_from_prev][i] = if cols.virtual_commit_from_prev[i] {
                F::ONE
            } else {
                F::ZERO
            };
            wit.cols[layout.cycle][i] = F::from_u64(cols.cycle[i]);
            wit.cols[layout.pc_before][i] = F::from_u64(cols.pc_before[i]);
            wit.cols[layout.pc_after][i] = F::from_u64(cols.pc_after[i]);
            wit.cols[layout.instr_word][i] = F::from_u64(cols.instr_word[i] as u64);
            if !cols.active[i] {
                // Inactive rows stay quiescent; WB/WP sidecars enforce these zeros.
                continue;
            }

            // REG view
            wit.cols[layout.rs1_addr][i] = F::from_u64(cols.rs1_addr[i]);
            wit.cols[layout.rs1_val][i] = F::from_u64(cols.rs1_val[i]);
            wit.cols[layout.rs2_addr][i] = F::from_u64(cols.rs2_addr[i]);
            wit.cols[layout.rs2_val][i] = F::from_u64(cols.rs2_val[i]);
            wit.cols[layout.rd_addr][i] = F::from_u64(cols.rd_addr[i]);
            wit.cols[layout.rd_has_write][i] = if cols.rd_has_write[i] { F::ONE } else { F::ZERO };
            wit.cols[layout.rd_val][i] = F::from_u64(cols.rd_val[i]);
            if cols.opcode[i] == 0x67 {
                let rs1 = cols.rs1_val[i] as u32;
                let imm_i = imm_i_from_word(cols.instr_word[i]);
                let drop = rs1.wrapping_add(imm_i) & 1;
                wit.cols[layout.jalr_drop_bit][i] = F::from_u64(drop as u64);
            }
        }

        // Normalize RAM events per row: at most one read + one write.
        for (i, r) in exec.rows.iter().enumerate() {
            if !r.active {
                continue;
            }

            let mut read: Option<(u64, u64)> = None;
            let mut write: Option<(u64, u64)> = None;
            for e in &r.ram_events {
                match e.kind {
                    TwistOpKind::Read => {
                        if read.is_some() {
                            return Err(format!("multiple RAM reads in one cycle={}", r.cycle));
                        }
                        read = Some((e.addr, e.value));
                    }
                    TwistOpKind::Write => {
                        if write.is_some() {
                            return Err(format!("multiple RAM writes in one cycle={}", r.cycle));
                        }
                        write = Some((e.addr, e.value));
                    }
                }
            }

            match (read, write) {
                (Some((ra, rv)), Some((wa, wv))) => {
                    if ra != wa {
                        return Err(format!(
                            "RAM read/write addr mismatch in one cycle {}: ra={:#x} wa={:#x}",
                            r.cycle, ra, wa
                        ));
                    }
                    wit.cols[layout.ram_addr][i] = F::from_u64(ra);
                    wit.cols[layout.ram_rv][i] = F::from_u64(rv);
                    wit.cols[layout.ram_wv][i] = F::from_u64(wv);
                }
                (Some((ra, rv)), None) => {
                    wit.cols[layout.ram_addr][i] = F::from_u64(ra);
                    wit.cols[layout.ram_rv][i] = F::from_u64(rv);
                }
                (None, Some((wa, wv))) => {
                    wit.cols[layout.ram_addr][i] = F::from_u64(wa);
                    wit.cols[layout.ram_wv][i] = F::from_u64(wv);
                }
                (None, None) => {}
            }
        }

        // Normalize fixed-lane Shout view for the main trace.
        //
        // Shared-bus mode may carry auxiliary lookup families in addition to
        // opcode-backed Shout events. The fixed-lane CPU shout glue must only
        // bind to canonical RV32 opcode tables.
        let shout_tables = RiscvShoutTables::new(/*xlen=*/ 32);
        for (i, r) in exec.rows.iter().enumerate() {
            if !r.active {
                continue;
            }
            let primary = r
                .shout_events
                .iter()
                .find(|ev| shout_tables.id_to_opcode(ev.shout_id).is_some());

            if let Some(ev) = primary {
                wit.cols[layout.shout_has_lookup][i] = F::ONE;
                wit.cols[layout.shout_table_id][i] = F::from_u64(ev.shout_id.0 as u64);
                wit.cols[layout.shout_val][i] = F::from_u64(ev.value);
                let fallback_lhs = cols.rs1_val[i];
                let fallback_rhs = cols.rs2_val[i];
                let op = shout_tables.id_to_opcode(ev.shout_id);
                let decoded_fallback = r.decoded.as_ref().map(|decoded| {
                    lookup_operands_from_decoded(decoded, cols.rs1_val[i], cols.rs2_val[i], cols.pc_before[i])
                });
                let (lhs, rhs) = if let Some(op) = op {
                    try_decode_lookup_operands(op, ev.key as u128, operand_mode_keys_enabled(), /*xlen=*/ 32)
                        .or(decoded_fallback)
                        .unwrap_or((fallback_lhs, fallback_rhs))
                } else {
                    (fallback_lhs, fallback_rhs)
                };
                wit.cols[layout.shout_lhs][i] = F::from_u64(lhs);
                // Canonicalize shift keys: RISC-V shifts use only the low 5 bits of `rhs`.
                let rhs = if let Some(op) = shout_tables.id_to_opcode(ev.shout_id) {
                    if matches!(op, RiscvOpcode::Sll | RiscvOpcode::Srl | RiscvOpcode::Sra) {
                        rhs & 0x1F
                    } else {
                        rhs
                    }
                } else {
                    rhs
                };
                wit.cols[layout.shout_rhs][i] = F::from_u64(rhs);

                let is_combined_mode = op.map(opcode_uses_combined_lookup_key).unwrap_or(false);
                if is_combined_mode {
                    let key = u64::try_from(ev.key)
                        .map_err(|_| format!("combined shout key does not fit u64 at row {i}: {}", ev.key))?;
                    wit.cols[layout.shout_add_sub_key][i] = F::from_u64(key);
                } else {
                    wit.cols[layout.shout_link_lhs][i] = F::from_u64(lhs);
                    wit.cols[layout.shout_link_rhs][i] = F::from_u64(rhs);
                }
            }
        }
        Ok(wit)
    }
}
