use neo_vm_trace::TwistOpKind;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks as F;

use crate::riscv::exec_table::Rv32ExecTable;
use crate::riscv::instruction::{
    opcode_uses_combined_lookup_key, operand_mode_keys_enabled, try_decode_lookup_operands,
};
use crate::riscv::lookups::{RiscvInstruction, RiscvOpcode, RiscvShoutTables};

#[derive(Clone, Debug)]
pub struct Rv64TraceLayout {
    pub cols: usize,

    // Core control / fetch.
    pub one: usize,
    pub active: usize,
    pub halted: usize,
    pub is_virtual: usize,
    pub virtual_sequence_remaining: usize,
    pub cycle: usize,
    pub pc_before: usize,
    pub pc_after: usize,
    pub instr_word: usize,

    // Regfile view (REG Twist).
    pub rs1_addr: usize,
    pub rs1_val: usize,
    pub rs2_addr: usize,
    pub rs2_val: usize,
    pub rd_addr: usize,
    pub rd_has_write: usize,
    pub rd_val: usize,

    // RAM view (RAM Twist, normalized to at most 1R + 1W per row).
    pub ram_addr: usize,
    pub ram_rv: usize,
    pub ram_wv: usize,

    // Shout view (single fixed-lane per row; output-only for now).
    pub shout_has_lookup: usize,
    pub shout_table_id: usize,
    pub shout_val: usize,
    pub shout_lhs: usize,
    pub shout_rhs: usize,
    pub shout_link_lhs: usize,
    pub shout_link_rhs: usize,
    pub shout_add_sub_key: usize,
    pub shout_add_sub_key_lo32: usize,
    pub shout_add_sub_key_hi32: usize,
    pub jalr_drop_bit: usize,
    pub virtual_transition: usize,
    pub virtual_commit_link: usize,
    pub virtual_commit_from_prev: usize,

    // Exact low/high-32 transport for RV64 value-carrying columns.
    pub rs1_val_lo32: usize,
    pub rs1_val_hi32: usize,
    pub rs2_val_lo32: usize,
    pub rs2_val_hi32: usize,
    pub rd_val_lo32: usize,
    pub rd_val_hi32: usize,
    pub ram_rv_lo32: usize,
    pub ram_rv_hi32: usize,
    pub ram_wv_lo32: usize,
    pub ram_wv_hi32: usize,
    pub shout_val_lo32: usize,
    pub shout_val_hi32: usize,
    pub shout_lhs_lo32: usize,
    pub shout_lhs_hi32: usize,
    pub shout_rhs_lo32: usize,
    pub shout_rhs_hi32: usize,
}

impl Rv64TraceLayout {
    pub fn new() -> Self {
        let mut next = 0usize;
        let mut take = || {
            let out = next;
            next += 1;
            out
        };

        let one = take();
        let active = take();
        let halted = take();
        let is_virtual = take();
        let virtual_sequence_remaining = take();
        let cycle = take();
        let pc_before = take();
        let pc_after = take();
        let instr_word = take();

        let rs1_addr = take();
        let rs1_val = take();
        let rs2_addr = take();
        let rs2_val = take();
        let rd_addr = take();
        let rd_has_write = take();
        let rd_val = take();

        let ram_addr = take();
        let ram_rv = take();
        let ram_wv = take();

        let shout_has_lookup = take();
        let shout_table_id = take();
        let shout_val = take();
        let shout_lhs = take();
        let shout_rhs = take();
        let shout_link_lhs = take();
        let shout_link_rhs = take();
        let shout_add_sub_key = take();
        // Preserve the RV32 trace prefix exactly through virtual_commit_from_prev.
        let jalr_drop_bit = take();
        let virtual_transition = take();
        let virtual_commit_link = take();
        let virtual_commit_from_prev = take();

        let shout_add_sub_key_lo32 = take();
        let shout_add_sub_key_hi32 = take();
        let rs1_val_lo32 = take();
        let rs1_val_hi32 = take();
        let rs2_val_lo32 = take();
        let rs2_val_hi32 = take();
        let rd_val_lo32 = take();
        let rd_val_hi32 = take();
        let ram_rv_lo32 = take();
        let ram_rv_hi32 = take();
        let ram_wv_lo32 = take();
        let ram_wv_hi32 = take();
        let shout_val_lo32 = take();
        let shout_val_hi32 = take();
        let shout_lhs_lo32 = take();
        let shout_lhs_hi32 = take();
        let shout_rhs_lo32 = take();
        let shout_rhs_hi32 = take();

        debug_assert_eq!(next, 49, "RV64 trace width drift after exact transport columns");

        Self {
            cols: next,
            one,
            active,
            halted,
            is_virtual,
            virtual_sequence_remaining,
            cycle,
            pc_before,
            pc_after,
            instr_word,
            rs1_addr,
            rs1_val,
            rs2_addr,
            rs2_val,
            rd_addr,
            rd_has_write,
            rd_val,
            ram_addr,
            ram_rv,
            ram_wv,
            shout_has_lookup,
            shout_table_id,
            shout_val,
            shout_lhs,
            shout_rhs,
            shout_link_lhs,
            shout_link_rhs,
            shout_add_sub_key,
            shout_add_sub_key_lo32,
            shout_add_sub_key_hi32,
            jalr_drop_bit,
            virtual_transition,
            virtual_commit_link,
            virtual_commit_from_prev,
            rs1_val_lo32,
            rs1_val_hi32,
            rs2_val_lo32,
            rs2_val_hi32,
            rd_val_lo32,
            rd_val_hi32,
            ram_rv_lo32,
            ram_rv_hi32,
            ram_wv_lo32,
            ram_wv_hi32,
            shout_val_lo32,
            shout_val_hi32,
            shout_lhs_lo32,
            shout_lhs_hi32,
            shout_rhs_lo32,
            shout_rhs_hi32,
        }
    }
}

#[inline]
fn sign_extend_to_u64(value: u32, bits: u32) -> u64 {
    let shift = 64 - bits;
    (((value as u64) << shift) as i64 >> shift) as u64
}

#[inline]
fn imm_i_from_word(instr_word: u32) -> u64 {
    sign_extend_to_u64((instr_word >> 20) & 0x0fff, 12)
}

#[inline]
fn auipc_imm_u64(imm: i32) -> u64 {
    (((imm as u32) << 12) as i32 as i64) as u64
}

#[inline]
fn lookup_operands_from_decoded(decoded: &RiscvInstruction, rs1_val: u64, rs2_val: u64, pc_before: u64) -> (u64, u64) {
    match decoded {
        RiscvInstruction::RAlu { .. } | RiscvInstruction::RAluw { .. } => (rs1_val, rs2_val),
        RiscvInstruction::IAlu { imm, .. } | RiscvInstruction::IAluw { imm, .. } => (rs1_val, *imm as i64 as u64),
        RiscvInstruction::Load { imm, .. } | RiscvInstruction::Jalr { imm, .. } => (rs1_val, *imm as i64 as u64),
        RiscvInstruction::Store { imm, .. } => (rs1_val, *imm as i64 as u64),
        RiscvInstruction::Auipc { imm, .. } => (pc_before, auipc_imm_u64(*imm)),
        RiscvInstruction::Branch { .. } => (rs1_val, rs2_val),
        _ => (rs1_val, rs2_val),
    }
}

#[inline]
fn field_from_u64_injective(value: u64) -> Result<F, String> {
    if value >= <F as PrimeField64>::ORDER_U64 {
        return Err(format!(
            "RV64 trace witness cannot injectively encode machine word {value:#x} into Goldilocks"
        ));
    }
    Ok(F::from_u64(value))
}

#[inline]
fn split_u64(value: u64) -> (u64, u64) {
    ((value as u32) as u64, (value >> 32) as u64)
}

#[inline]
fn field_from_u64_exact_transport(value: u64) -> F {
    let (lo, hi) = split_u64(value);
    F::from_u64(lo) + F::from_u64(hi) * F::from_u64(1u64 << 32)
}

#[inline]
fn set_exact_u64_cols(
    cols: &mut [Vec<F>],
    row: usize,
    scalar_col: usize,
    lo_col: usize,
    hi_col: usize,
    value: u64,
) -> Result<(), String> {
    let (lo, hi) = split_u64(value);
    cols[scalar_col][row] = field_from_u64_exact_transport(value);
    cols[lo_col][row] = F::from_u64(lo);
    cols[hi_col][row] = F::from_u64(hi);
    Ok(())
}

#[derive(Clone, Debug)]
pub struct Rv64TraceWitness {
    pub t: usize,
    pub cols: Vec<Vec<F>>,
}

impl Rv64TraceWitness {
    pub fn new_zero(layout: &Rv64TraceLayout, t: usize) -> Self {
        Self {
            t,
            cols: vec![vec![F::ZERO; t]; layout.cols],
        }
    }

    pub fn from_exec_table(layout: &Rv64TraceLayout, exec: &Rv32ExecTable) -> Result<Self, String> {
        let cols = exec.to_columns();
        let t = cols.len();
        let mut wit = Self::new_zero(layout, t);

        for i in 0..t {
            wit.cols[layout.one][i] = F::ONE;
            wit.cols[layout.active][i] = if cols.active[i] { F::ONE } else { F::ZERO };
            wit.cols[layout.halted][i] = if cols.halted[i] { F::ONE } else { F::ZERO };
            wit.cols[layout.is_virtual][i] = if cols.is_virtual[i] { F::ONE } else { F::ZERO };
            wit.cols[layout.virtual_sequence_remaining][i] =
                field_from_u64_injective(cols.virtual_sequence_remaining[i])?;
            wit.cols[layout.virtual_transition][i] = if cols.virtual_transition[i] { F::ONE } else { F::ZERO };
            wit.cols[layout.virtual_commit_link][i] = if cols.virtual_commit_link[i] { F::ONE } else { F::ZERO };
            wit.cols[layout.virtual_commit_from_prev][i] = if cols.virtual_commit_from_prev[i] {
                F::ONE
            } else {
                F::ZERO
            };
            wit.cols[layout.cycle][i] = field_from_u64_injective(cols.cycle[i])?;
            wit.cols[layout.pc_before][i] = field_from_u64_injective(cols.pc_before[i])?;
            wit.cols[layout.pc_after][i] = field_from_u64_injective(cols.pc_after[i])?;
            wit.cols[layout.instr_word][i] = F::from_u64(cols.instr_word[i] as u64);
            if !cols.active[i] {
                continue;
            }

            wit.cols[layout.rs1_addr][i] = field_from_u64_injective(cols.rs1_addr[i])?;
            set_exact_u64_cols(
                &mut wit.cols,
                i,
                layout.rs1_val,
                layout.rs1_val_lo32,
                layout.rs1_val_hi32,
                cols.rs1_val[i],
            )?;
            wit.cols[layout.rs2_addr][i] = field_from_u64_injective(cols.rs2_addr[i])?;
            set_exact_u64_cols(
                &mut wit.cols,
                i,
                layout.rs2_val,
                layout.rs2_val_lo32,
                layout.rs2_val_hi32,
                cols.rs2_val[i],
            )?;
            wit.cols[layout.rd_addr][i] = field_from_u64_injective(cols.rd_addr[i])?;
            wit.cols[layout.rd_has_write][i] = if cols.rd_has_write[i] { F::ONE } else { F::ZERO };
            set_exact_u64_cols(
                &mut wit.cols,
                i,
                layout.rd_val,
                layout.rd_val_lo32,
                layout.rd_val_hi32,
                cols.rd_val[i],
            )?;
            if cols.opcode[i] == 0x67 {
                let drop = cols.rs1_val[i].wrapping_add(imm_i_from_word(cols.instr_word[i])) & 1;
                wit.cols[layout.jalr_drop_bit][i] = F::from_u64(drop);
            }
        }

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
                    wit.cols[layout.ram_addr][i] = field_from_u64_injective(ra)?;
                    set_exact_u64_cols(
                        &mut wit.cols,
                        i,
                        layout.ram_rv,
                        layout.ram_rv_lo32,
                        layout.ram_rv_hi32,
                        rv,
                    )?;
                    set_exact_u64_cols(
                        &mut wit.cols,
                        i,
                        layout.ram_wv,
                        layout.ram_wv_lo32,
                        layout.ram_wv_hi32,
                        wv,
                    )?;
                }
                (Some((ra, rv)), None) => {
                    wit.cols[layout.ram_addr][i] = field_from_u64_injective(ra)?;
                    set_exact_u64_cols(
                        &mut wit.cols,
                        i,
                        layout.ram_rv,
                        layout.ram_rv_lo32,
                        layout.ram_rv_hi32,
                        rv,
                    )?;
                }
                (None, Some((wa, wv))) => {
                    wit.cols[layout.ram_addr][i] = field_from_u64_injective(wa)?;
                    set_exact_u64_cols(
                        &mut wit.cols,
                        i,
                        layout.ram_wv,
                        layout.ram_wv_lo32,
                        layout.ram_wv_hi32,
                        wv,
                    )?;
                }
                (None, None) => {}
            }
        }

        let shout_tables = RiscvShoutTables::new(/*xlen=*/ 64);
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
                set_exact_u64_cols(
                    &mut wit.cols,
                    i,
                    layout.shout_val,
                    layout.shout_val_lo32,
                    layout.shout_val_hi32,
                    ev.value,
                )?;
                let fallback_lhs = cols.rs1_val[i];
                let fallback_rhs = cols.rs2_val[i];
                let op = shout_tables.id_to_opcode(ev.shout_id);
                let decoded_fallback = r.decoded.as_ref().map(|decoded| {
                    lookup_operands_from_decoded(decoded, cols.rs1_val[i], cols.rs2_val[i], cols.pc_before[i])
                });
                let (lhs, rhs) = if let Some(op) = op {
                    try_decode_lookup_operands(op, ev.key as u128, operand_mode_keys_enabled(), /*xlen=*/ 64)
                        .or(decoded_fallback)
                        .unwrap_or((fallback_lhs, fallback_rhs))
                } else {
                    (fallback_lhs, fallback_rhs)
                };
                set_exact_u64_cols(
                    &mut wit.cols,
                    i,
                    layout.shout_lhs,
                    layout.shout_lhs_lo32,
                    layout.shout_lhs_hi32,
                    lhs,
                )?;
                let rhs = if let Some(op) = shout_tables.id_to_opcode(ev.shout_id) {
                    if matches!(op, RiscvOpcode::Sll | RiscvOpcode::Srl | RiscvOpcode::Sra) {
                        rhs & 0x3F
                    } else {
                        rhs
                    }
                } else {
                    rhs
                };
                set_exact_u64_cols(
                    &mut wit.cols,
                    i,
                    layout.shout_rhs,
                    layout.shout_rhs_lo32,
                    layout.shout_rhs_hi32,
                    rhs,
                )?;

                let is_combined_mode = op.map(opcode_uses_combined_lookup_key).unwrap_or(false);
                if is_combined_mode {
                    let key = u64::try_from(ev.key)
                        .map_err(|_| format!("combined shout key does not fit u64 at row {i}: {}", ev.key))?;
                    set_exact_u64_cols(
                        &mut wit.cols,
                        i,
                        layout.shout_add_sub_key,
                        layout.shout_add_sub_key_lo32,
                        layout.shout_add_sub_key_hi32,
                        key,
                    )?;
                } else {
                    wit.cols[layout.shout_link_lhs][i] = field_from_u64_exact_transport(lhs);
                    wit.cols[layout.shout_link_rhs][i] = field_from_u64_exact_transport(rhs);
                }
            }
        }
        Ok(wit)
    }
}
