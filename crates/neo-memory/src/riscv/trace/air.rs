use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks as F;

use crate::riscv::decomposition_semantics::{expected_virtual_decomposed_op, validate_virtual_row_semantics};

use super::{layout::Rv32TraceLayout, witness::Rv32TraceWitness};

#[derive(Clone, Debug)]
pub struct RiscvTraceAir {
    pub layout: Rv32TraceLayout,
    pub machine_xlen: usize,
}

impl RiscvTraceAir {
    pub fn new() -> Self {
        Self::new_with_xlen(/*machine_xlen=*/ 32)
    }

    pub fn new_with_xlen(machine_xlen: usize) -> Self {
        Self {
            layout: Rv32TraceLayout::new(),
            machine_xlen,
        }
    }

    #[inline]
    fn is_zero(x: F) -> bool {
        x == F::ZERO
    }

    #[inline]
    fn bool_check(x: F) -> F {
        x * (x - F::ONE)
    }

    #[inline]
    fn gated_zero(gate: F, x: F) -> F {
        gate * x
    }

    pub fn assert_satisfied(&self, wit: &Rv32TraceWitness) -> Result<(), String> {
        let l = &self.layout;
        if wit.cols.len() != l.cols {
            return Err(format!(
                "trace witness width mismatch: got {} cols, expected {}",
                wit.cols.len(),
                l.cols
            ));
        }
        for (c, col) in wit.cols.iter().enumerate() {
            if col.len() != wit.t {
                return Err(format!(
                    "trace witness column length mismatch at col {c}: got {}, expected {}",
                    col.len(),
                    wit.t
                ));
            }
        }

        let col = |c: usize, i: usize| -> F { wit.cols[c][i] };

        // Row-wise constraints.
        for i in 0..wit.t {
            let one = col(l.one, i);
            if one != F::ONE {
                return Err(format!("row {i}: one != 1"));
            }

            let active = col(l.active, i);
            let halted = col(l.halted, i);
            let is_virtual = col(l.is_virtual, i);
            let virtual_sequence_remaining = col(l.virtual_sequence_remaining, i);
            let virtual_transition = col(l.virtual_transition, i);
            let virtual_commit_link = col(l.virtual_commit_link, i);
            let virtual_commit_from_prev = col(l.virtual_commit_from_prev, i);
            let rd_has_write = col(l.rd_has_write, i);
            let shout_has_lookup = col(l.shout_has_lookup, i);

            // Booleans.
            for (name, v) in [
                ("active", active),
                ("halted", halted),
                ("is_virtual", is_virtual),
                ("virtual_transition", virtual_transition),
                ("virtual_commit_link", virtual_commit_link),
                ("virtual_commit_from_prev", virtual_commit_from_prev),
                ("rd_has_write", rd_has_write),
                ("shout_has_lookup", shout_has_lookup),
            ] {
                let e = Self::bool_check(v);
                if !Self::is_zero(e) {
                    return Err(format!("row {i}: {name} not boolean"));
                }
            }
            if !Self::is_zero(Self::gated_zero(F::ONE - is_virtual, virtual_sequence_remaining)) {
                return Err(format!(
                    "row {i}: virtual_sequence_remaining must be 0 when is_virtual=0"
                ));
            }
            if !Self::is_zero(Self::gated_zero(F::ONE - rd_has_write, col(l.rd_addr, i))) {
                return Err(format!("row {i}: rd_addr must be 0 when rd_has_write=0"));
            }
            if !Self::is_zero(Self::gated_zero(F::ONE - rd_has_write, col(l.rd_val, i))) {
                return Err(format!("row {i}: rd_val must be 0 when rd_has_write=0"));
            }
            if !Self::is_zero(Self::gated_zero(is_virtual, col(l.pc_after, i) - col(l.pc_before, i))) {
                return Err(format!("row {i}: virtual step must keep pc_after == pc_before"));
            }
            if !Self::is_zero(Self::gated_zero(is_virtual, halted)) {
                return Err(format!("row {i}: virtual row cannot be halted"));
            }
            if i == 0 && !Self::is_zero(virtual_commit_from_prev) {
                return Err("row 0: virtual_commit_from_prev must be 0".into());
            }
            if is_virtual == F::ONE {
                let instr_word_u64 = col(l.instr_word, i).as_canonical_u64();
                let instr_word = u32::try_from(instr_word_u64).map_err(|_| {
                    format!("row {i}: instr_word does not fit u32 for virtual decomposition: {instr_word_u64:#x}")
                })?;
                let remaining_u64 = virtual_sequence_remaining.as_canonical_u64();
                let remaining = u32::try_from(remaining_u64)
                    .map_err(|_| format!("row {i}: virtual_sequence_remaining does not fit u32: {remaining_u64}"))?;
                let op = expected_virtual_decomposed_op(instr_word, remaining, self.machine_xlen)
                    .map_err(|e| format!("row {i}: {e}"))?;
                validate_virtual_row_semantics(
                    op,
                    col(l.rs1_addr, i).as_canonical_u64(),
                    col(l.rs1_val, i).as_canonical_u64(),
                    col(l.rs2_addr, i).as_canonical_u64(),
                    col(l.rs2_val, i).as_canonical_u64(),
                    rd_has_write == F::ONE,
                    col(l.rd_addr, i).as_canonical_u64(),
                    col(l.rd_val, i).as_canonical_u64(),
                    self.machine_xlen,
                )
                .map_err(|e| format!("row {i}: {e}"))?;
            }
            // Padding invariants: inactive rows must not carry "hidden" values.
            let inv_active = F::ONE - active;
            for (name, c) in [
                ("instr_word", l.instr_word),
                ("is_virtual", l.is_virtual),
                ("virtual_sequence_remaining", l.virtual_sequence_remaining),
                ("virtual_transition", l.virtual_transition),
                ("virtual_commit_link", l.virtual_commit_link),
                ("virtual_commit_from_prev", l.virtual_commit_from_prev),
                ("rs1_addr", l.rs1_addr),
                ("rs1_val", l.rs1_val),
                ("rs2_addr", l.rs2_addr),
                ("rs2_val", l.rs2_val),
                ("rd_addr", l.rd_addr),
                ("rd_has_write", l.rd_has_write),
                ("rd_val", l.rd_val),
                ("ram_addr", l.ram_addr),
                ("ram_rv", l.ram_rv),
                ("ram_wv", l.ram_wv),
                ("shout_has_lookup", l.shout_has_lookup),
                ("shout_table_id", l.shout_table_id),
                ("shout_val", l.shout_val),
                ("shout_lhs", l.shout_lhs),
                ("shout_rhs", l.shout_rhs),
                ("shout_link_lhs", l.shout_link_lhs),
                ("shout_link_rhs", l.shout_link_rhs),
                ("shout_add_sub_key", l.shout_add_sub_key),
                ("jalr_drop_bit", l.jalr_drop_bit),
            ] {
                let e = Self::gated_zero(inv_active, col(c, i));
                if !Self::is_zero(e) {
                    return Err(format!("row {i}: inactive padding violated ({name} != 0)"));
                }
            }
            if is_virtual == F::ONE && rd_has_write == F::ONE && col(l.rd_addr, i).as_canonical_u64() < 32 {
                return Err(format!(
                    "row {i}: virtual row attempted architectural write (rd_addr={})",
                    col(l.rd_addr, i).as_canonical_u64()
                ));
            }
            if is_virtual == F::ZERO {
                let rs1_addr = col(l.rs1_addr, i).as_canonical_u64();
                let rs2_addr = col(l.rs2_addr, i).as_canonical_u64();
                let rd_addr = col(l.rd_addr, i).as_canonical_u64();
                if rs1_addr >= 32 || rs2_addr >= 32 {
                    return Err(format!(
                        "row {i}: non-virtual row uses virtual read addr (rs1_addr={rs1_addr}, rs2_addr={rs2_addr})"
                    ));
                }
                if rd_has_write == F::ONE && rd_addr >= 32 {
                    return Err(format!(
                        "row {i}: non-virtual row uses virtual write addr (rd_addr={rd_addr})"
                    ));
                }
            }

            // Shout padding: if no lookup, the lookup output must be 0.
            {
                if !Self::is_zero(Self::gated_zero(F::ONE - shout_has_lookup, col(l.shout_table_id, i))) {
                    return Err(format!("row {i}: shout_table_id must be 0 when shout_has_lookup=0"));
                }
                if !Self::is_zero(Self::gated_zero(F::ONE - shout_has_lookup, col(l.shout_val, i))) {
                    return Err(format!("row {i}: shout_val must be 0 when shout_has_lookup=0"));
                }
                if !Self::is_zero(Self::gated_zero(F::ONE - shout_has_lookup, col(l.shout_lhs, i))) {
                    return Err(format!("row {i}: shout_lhs must be 0 when shout_has_lookup=0"));
                }
                if !Self::is_zero(Self::gated_zero(F::ONE - shout_has_lookup, col(l.shout_rhs, i))) {
                    return Err(format!("row {i}: shout_rhs must be 0 when shout_has_lookup=0"));
                }
                if !Self::is_zero(Self::gated_zero(F::ONE - shout_has_lookup, col(l.shout_link_lhs, i))) {
                    return Err(format!("row {i}: shout_link_lhs must be 0 when shout_has_lookup=0"));
                }
                if !Self::is_zero(Self::gated_zero(F::ONE - shout_has_lookup, col(l.shout_link_rhs, i))) {
                    return Err(format!("row {i}: shout_link_rhs must be 0 when shout_has_lookup=0"));
                }
                if !Self::is_zero(Self::gated_zero(F::ONE - shout_has_lookup, col(l.shout_add_sub_key, i))) {
                    return Err(format!("row {i}: shout_add_sub_key must be 0 when shout_has_lookup=0"));
                }
            }
        }

        // Transition constraints.
        for i in 0..wit.t.saturating_sub(1) {
            let e = col(l.pc_after, i) - col(l.pc_before, i + 1);
            if !Self::is_zero(e) {
                return Err(format!("pc chain mismatch at row {i}"));
            }

            let e = col(l.cycle, i + 1) - (col(l.cycle, i) + F::ONE);
            if !Self::is_zero(e) {
                return Err(format!("cycle chain mismatch at row {i}"));
            }

            let is_virtual = col(l.is_virtual, i);
            let remaining = col(l.virtual_sequence_remaining, i);
            let remaining_next = col(l.virtual_sequence_remaining, i + 1);
            let active_next = col(l.active, i + 1);
            let is_virtual_next = col(l.is_virtual, i + 1);
            let virtual_transition = col(l.virtual_transition, i);
            let virtual_commit_link = col(l.virtual_commit_link, i);
            let virtual_commit_from_prev = col(l.virtual_commit_from_prev, i);
            let virtual_commit_from_prev_next = col(l.virtual_commit_from_prev, i + 1);
            let rd_has_write = col(l.rd_has_write, i);
            let rd_has_write_next = col(l.rd_has_write, i + 1);
            let rd_val = col(l.rd_val, i);
            let rd_val_next = col(l.rd_val, i + 1);
            if !Self::is_zero(is_virtual * (F::ONE - active_next)) {
                return Err(format!("virtual row must be followed by an active row at row {i}"));
            }
            if !Self::is_zero(is_virtual * (col(l.instr_word, i + 1) - col(l.instr_word, i))) {
                return Err(format!(
                    "virtual row must preserve instr_word across sequence at row {i}"
                ));
            }
            if !Self::is_zero(is_virtual * (remaining - remaining_next - F::ONE)) {
                return Err(format!("virtual_sequence_remaining countdown mismatch at row {i}"));
            }
            if !Self::is_zero(is_virtual * (active_next - is_virtual_next - virtual_transition)) {
                return Err(format!("virtual_transition linkage mismatch at row {i}"));
            }
            if !Self::is_zero((F::ONE - is_virtual) * virtual_transition) {
                return Err(format!("virtual_transition must be zero on non-virtual row {i}"));
            }
            if !Self::is_zero(virtual_transition * (F::ONE - rd_has_write)) {
                return Err(format!("virtual_transition requires last virtual row write at row {i}"));
            }
            if !Self::is_zero(virtual_transition * (virtual_commit_link - rd_has_write_next)) {
                return Err(format!("virtual_commit_link write gating mismatch at row {i}"));
            }
            if !Self::is_zero((F::ONE - virtual_transition) * virtual_commit_link) {
                return Err(format!(
                    "virtual_commit_link must be zero when virtual_transition=0 at row {i}"
                ));
            }
            if !Self::is_zero(virtual_commit_link * (F::ONE - rd_has_write_next)) {
                return Err(format!("virtual_commit_link requires next-row rd write at row {i}"));
            }
            if !Self::is_zero(virtual_commit_link * (rd_val_next - rd_val)) {
                return Err(format!("virtual commit value mismatch at row {i}"));
            }
            if !Self::is_zero(virtual_commit_from_prev_next - virtual_commit_link) {
                return Err(format!("virtual_commit_from_prev linkage mismatch at row {i}"));
            }
            if !Self::is_zero(virtual_commit_from_prev * is_virtual) {
                return Err(format!("virtual_commit_from_prev must be 0 on virtual row {i}"));
            }

            // Once inactive, remain inactive.
            let a0 = col(l.active, i);
            let a1 = col(l.active, i + 1);
            if !Self::is_zero(a1 * (F::ONE - a0)) {
                return Err(format!("active monotonicity violated at row {i}"));
            }

            // Once halted, remain halted.
            let h0 = col(l.halted, i);
            let h1 = col(l.halted, i + 1);
            if !Self::is_zero(h0 * (F::ONE - h1)) {
                return Err(format!("halted monotonicity violated at row {i}"));
            }

            // HALT terminates execution: halted[i] => active[i+1] == 0.
            if !Self::is_zero(h0 * a1) {
                return Err(format!("halted tail quiescence violated at row {i}"));
            }
        }

        if wit.t > 0 {
            let last = wit.t - 1;
            if !Self::is_zero(col(l.virtual_transition, last)) {
                return Err("last row: virtual_transition must be 0".into());
            }
            if !Self::is_zero(col(l.virtual_commit_link, last)) {
                return Err("last row: virtual_commit_link must be 0".into());
            }
        }

        Ok(())
    }
}

pub type Rv32TraceAir = RiscvTraceAir;
