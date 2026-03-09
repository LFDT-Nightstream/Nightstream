use neo_vm_trace::{ShoutEvent, StepTrace, TwistEvent, TwistOpKind, VmTrace};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use p3_symmetric::Permutation;

use crate::riscv::decomposition_semantics::{expected_virtual_decomposed_op, validate_virtual_row_semantics};
use crate::riscv::instruction::{encode_lookup_key, operand_mode_keys_enabled, try_decode_lookup_operands};
use crate::riscv::lookups::{
    compute_op, decode_instruction_with_xlen, RiscvInstruction, RiscvOpcode, RiscvShoutTables, PROG_ID, RAM_ID, REG_ID,
};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RiscvInstrFields {
    pub opcode: u32,
    pub rd: u8,
    pub funct3: u32,
    pub rs1: u8,
    pub rs2: u8,
    pub funct7: u32,
}

impl RiscvInstrFields {
    pub fn from_word(instr_word: u32) -> Self {
        Self {
            opcode: instr_word & 0x7f,
            rd: ((instr_word >> 7) & 0x1f) as u8,
            funct3: (instr_word >> 12) & 0x7,
            rs1: ((instr_word >> 15) & 0x1f) as u8,
            rs2: ((instr_word >> 20) & 0x1f) as u8,
            funct7: (instr_word >> 25) & 0x7f,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RiscvRegLaneIo {
    pub addr: u64,
    pub value: u64,
}

#[derive(Clone, Debug)]
pub struct RiscvExecRow {
    /// True for real trace rows; false for padded/inactive rows.
    pub active: bool,

    /// True when this row is a virtual/decomposed instruction step.
    ///
    /// Step 2 scaffold: currently always `false` until decomposition is wired in.
    pub is_virtual: bool,

    /// Remaining virtual sequence length (inclusive countdown), if any.
    ///
    /// Step 2 scaffold: currently always `None` until decomposition is wired in.
    pub virtual_sequence_remaining: Option<u32>,

    pub cycle: u64,
    pub pc_before: u64,
    pub pc_after: u64,
    pub instr_word: u32,
    pub fields: RiscvInstrFields,
    pub halted: bool,

    /// Decoded instruction (for semantic context; derived from `instr_word`).
    pub decoded: Option<crate::riscv::lookups::RiscvInstruction>,

    /// PROG ROM fetch (`PROG_ID`) for this step.
    pub prog_read: Option<TwistEvent<u64, u64>>,

    /// REG lane 0 read (`REG_ID`, lane=0): rs1_field → rs1_val.
    pub reg_read_lane0: Option<RiscvRegLaneIo>,

    /// REG lane 1 read (`REG_ID`, lane=1): rs2_field → rs2_val.
    pub reg_read_lane1: Option<RiscvRegLaneIo>,

    /// Optional REG lane 0 write (`REG_ID`, lane=0): rd_field → rd_write_val.
    pub reg_write_lane0: Option<RiscvRegLaneIo>,

    /// RAM twist events (`RAM_ID`) for this step.
    pub ram_events: Vec<TwistEvent<u64, u64>>,

    /// Shout events for this step.
    pub shout_events: Vec<ShoutEvent<u128, u64>>,
}

#[derive(Clone, Debug)]
pub struct RiscvExecColumns {
    pub active: Vec<bool>,
    pub is_virtual: Vec<bool>,
    pub virtual_sequence_remaining: Vec<u64>,
    pub virtual_transition: Vec<bool>,
    pub virtual_commit_link: Vec<bool>,
    pub virtual_commit_from_prev: Vec<bool>,
    pub cycle: Vec<u64>,
    pub pc_before: Vec<u64>,
    pub pc_after: Vec<u64>,
    pub instr_word: Vec<u32>,
    pub opcode: Vec<u32>,
    pub rd: Vec<u8>,
    pub funct3: Vec<u32>,
    pub rs1: Vec<u8>,
    pub rs2: Vec<u8>,
    pub funct7: Vec<u32>,
    pub halted: Vec<bool>,
    pub prog_addr: Vec<u64>,
    pub prog_value: Vec<u64>,
    pub rs1_addr: Vec<u64>,
    pub rs1_val: Vec<u64>,
    pub rs2_addr: Vec<u64>,
    pub rs2_val: Vec<u64>,
    pub rd_has_write: Vec<bool>,
    pub rd_addr: Vec<u64>,
    pub rd_val: Vec<u64>,
}

impl RiscvExecColumns {
    pub fn len(&self) -> usize {
        self.cycle.len()
    }
}

#[derive(Clone, Debug)]
pub struct RiscvExecTable {
    pub rows: Vec<RiscvExecRow>,
}

// Compatibility aliases while the shared RISC-V trace layer is migrated away
// from legacy RV32 naming.
pub type Rv32InstrFields = RiscvInstrFields;
pub type Rv32RegLaneIo = RiscvRegLaneIo;
pub type Rv32ExecRow = RiscvExecRow;
pub type Rv32ExecColumns = RiscvExecColumns;
pub type Rv32ExecTable = RiscvExecTable;

impl RiscvExecTable {
    pub fn from_trace<Key>(trace: &VmTrace<u64, u64, Key>) -> Result<Self, String>
    where
        Key: Copy + TryInto<u128>,
        <Key as TryInto<u128>>::Error: std::fmt::Debug,
    {
        Self::from_trace_with_xlen(trace, /*machine_xlen=*/ 32)
    }

    pub fn from_trace_with_xlen<Key>(trace: &VmTrace<u64, u64, Key>, machine_xlen: usize) -> Result<Self, String>
    where
        Key: Copy + TryInto<u128>,
        <Key as TryInto<u128>>::Error: std::fmt::Debug,
    {
        let mut rows = Vec::with_capacity(trace.steps.len());
        for step in &trace.steps {
            rows.push(RiscvExecRow::from_step_with_xlen(step, machine_xlen)?);
        }
        let out = Self { rows };
        out.validate_virtual_decomposition_semantics(machine_xlen)?;
        Ok(out)
    }

    pub fn from_trace_padded<Key>(trace: &VmTrace<u64, u64, Key>, padded_len: usize) -> Result<Self, String>
    where
        Key: Copy + TryInto<u128>,
        <Key as TryInto<u128>>::Error: std::fmt::Debug,
    {
        Self::from_trace_padded_with_xlen(trace, padded_len, /*machine_xlen=*/ 32)
    }

    pub fn from_trace_padded_with_xlen<Key>(
        trace: &VmTrace<u64, u64, Key>,
        padded_len: usize,
        machine_xlen: usize,
    ) -> Result<Self, String>
    where
        Key: Copy + TryInto<u128>,
        <Key as TryInto<u128>>::Error: std::fmt::Debug,
    {
        if padded_len < trace.steps.len() {
            return Err(format!(
                "padded_len must be >= trace length (padded_len={} trace_len={})",
                padded_len,
                trace.steps.len()
            ));
        }

        let mut rows = Vec::with_capacity(padded_len);
        for step in &trace.steps {
            rows.push(RiscvExecRow::from_step_with_xlen(step, machine_xlen)?);
        }
        if rows.is_empty() {
            if padded_len == 0 {
                return Ok(Self { rows });
            }
            return Err("cannot pad empty trace without an initial pc".into());
        }

        let last = rows.last().expect("rows non-empty");
        let mut cycle = last.cycle;
        let pad_pc = last.pc_after;
        let pad_halted = last.halted;

        while rows.len() < padded_len {
            cycle = cycle
                .checked_add(1)
                .ok_or_else(|| "cycle overflow while padding".to_string())?;
            rows.push(RiscvExecRow::inactive(cycle, pad_pc, pad_halted));
        }

        let out = Self { rows };
        out.validate_virtual_decomposition_semantics(machine_xlen)?;
        Ok(out)
    }

    pub fn from_trace_padded_pow2<Key>(trace: &VmTrace<u64, u64, Key>, min_len: usize) -> Result<Self, String>
    where
        Key: Copy + TryInto<u128>,
        <Key as TryInto<u128>>::Error: std::fmt::Debug,
    {
        Self::from_trace_padded_pow2_with_xlen(trace, min_len, /*machine_xlen=*/ 32)
    }

    pub fn from_trace_padded_pow2_with_xlen<Key>(
        trace: &VmTrace<u64, u64, Key>,
        min_len: usize,
        machine_xlen: usize,
    ) -> Result<Self, String>
    where
        Key: Copy + TryInto<u128>,
        <Key as TryInto<u128>>::Error: std::fmt::Debug,
    {
        let steps = trace.steps.len();
        let target = steps.max(min_len).next_power_of_two();
        Self::from_trace_padded_with_xlen(trace, target, machine_xlen)
    }

    pub fn validate_pc_chain(&self) -> Result<(), String> {
        for w in self.rows.windows(2) {
            let a = &w[0];
            let b = &w[1];
            if a.pc_after != b.pc_before {
                return Err(format!(
                    "pc chain mismatch: cycle {} pc_after={:#x} != cycle {} pc_before={:#x}",
                    a.cycle, a.pc_after, b.cycle, b.pc_before
                ));
            }
        }
        Ok(())
    }

    /// Validate that cycles are consecutive (`cycle[t+1] = cycle[t] + 1`).
    pub fn validate_cycle_chain(&self) -> Result<(), String> {
        for w in self.rows.windows(2) {
            let a = &w[0];
            let b = &w[1];
            if b.cycle != a.cycle + 1 {
                return Err(format!(
                    "cycle chain mismatch: cycle {} then {} (expected {})",
                    a.cycle,
                    b.cycle,
                    a.cycle + 1
                ));
            }
        }
        Ok(())
    }

    /// Validate that inactive rows contain no events and no decoded instruction.
    pub fn validate_inactive_rows_are_empty(&self) -> Result<(), String> {
        for r in &self.rows {
            if r.active {
                continue;
            }
            if r.decoded.is_some()
                || r.prog_read.is_some()
                || r.reg_read_lane0.is_some()
                || r.reg_read_lane1.is_some()
                || r.reg_write_lane0.is_some()
                || !r.ram_events.is_empty()
                || !r.shout_events.is_empty()
            {
                return Err(format!("inactive row has events/decoded at cycle {}", r.cycle));
            }
        }
        Ok(())
    }

    /// Validate that once `halted` becomes true, it stays true and the PC stops changing.
    pub fn validate_halted_tail(&self) -> Result<(), String> {
        let mut saw_halt = false;
        let mut halt_pc: Option<u64> = None;
        for r in &self.rows {
            if !saw_halt {
                if r.halted {
                    saw_halt = true;
                    // In our trace semantics, the HALT row itself can advance the PC (default +4),
                    // but after that the machine is halted and PC should stop changing.
                    halt_pc = Some(r.pc_after);
                }
                continue;
            }

            if !r.halted {
                return Err(format!(
                    "halted tail violated: halted dropped to false at cycle {} (pc_before={:#x})",
                    r.cycle, r.pc_before
                ));
            }

            let pc0 = halt_pc.expect("halt_pc set");
            if r.pc_before != pc0 || r.pc_after != pc0 {
                return Err(format!(
                    "halted tail violated: pc changed after halt at cycle {} (pc_before={:#x} pc_after={:#x}, expected {:#x})",
                    r.cycle, r.pc_before, r.pc_after, pc0
                ));
            }
        }
        Ok(())
    }

    /// Validate virtual decomposition micro-op semantics row-by-row.
    ///
    /// This is a trace extraction hardening check and mirrors the virtual-op
    /// semantics enforced by `RiscvTraceAir`.
    pub fn validate_virtual_decomposition_semantics(&self, machine_xlen: usize) -> Result<(), String> {
        for (row_idx, r) in self.rows.iter().enumerate() {
            if !r.active || !r.is_virtual {
                continue;
            }
            let remaining = r
                .virtual_sequence_remaining
                .ok_or_else(|| format!("row {row_idx}: virtual row missing virtual_sequence_remaining"))?;
            let op = expected_virtual_decomposed_op(r.instr_word, remaining, machine_xlen)
                .map_err(|e| format!("row {row_idx}: {e}"))?;
            let rs1 = r
                .reg_read_lane0
                .as_ref()
                .ok_or_else(|| format!("row {row_idx}: virtual row missing REG lane0 read"))?;
            let rs2 = r
                .reg_read_lane1
                .as_ref()
                .ok_or_else(|| format!("row {row_idx}: virtual row missing REG lane1 read"))?;
            let (rd_has_write, rd_addr, rd_val) = if let Some(wr) = &r.reg_write_lane0 {
                (true, wr.addr, wr.value)
            } else {
                (false, 0, 0)
            };
            validate_virtual_row_semantics(
                op,
                rs1.addr,
                rs1.value,
                rs2.addr,
                rs2.value,
                rd_has_write,
                rd_addr,
                rd_val,
                machine_xlen,
            )
            .map_err(|e| format!("row {row_idx}: {e}"))?;
        }
        Ok(())
    }

    /// Validate strict JALR next-PC policy used by trace-wiring control claims.
    ///
    /// Current trace-wiring control stage enforces `pc_after = rs1_val + imm_i` for JALR rows
    /// (no committed drop-bit helper columns). Under this policy, traces requiring ISA-level
    /// JALR masking are out of scope and must be rejected during trace construction.
    pub fn validate_jalr_strict_alignment_policy(&self) -> Result<(), String> {
        for r in &self.rows {
            if !r.active {
                continue;
            }
            let Some(crate::riscv::lookups::RiscvInstruction::Jalr { imm, .. }) = r.decoded.as_ref() else {
                continue;
            };
            let rs1_val = r.reg_read_lane0.as_ref().map(|io| io.value).unwrap_or(0);
            let imm_u32 = *imm as u32 as u64;
            let expected_pc_after = rs1_val.wrapping_add(imm_u32);
            if r.pc_after != expected_pc_after {
                return Err(format!(
                    "strict JALR policy violated at cycle {}: pc_after={:#x}, expected rs1+imm={:#x} (rs1={:#x}, imm={:#x})",
                    r.cycle, r.pc_after, expected_pc_after, rs1_val, imm_u32
                ));
            }
        }
        Ok(())
    }

    /// Validate REG lane semantics by replaying the register file from an initial state.
    ///
    /// - `init_regs` maps `reg_idx (0..31)` → value (u32 stored in u64).
    /// - Unspecified registers default to 0.
    /// - Reads happen before the optional lane0 write in each cycle.
    pub fn validate_regfile_semantics(&self, init_regs: &HashMap<u64, u64>) -> Result<(), String> {
        let mut regs: HashMap<u64, u64> = HashMap::new();
        for (&addr, &value) in init_regs {
            if addr == 0 && value != 0 {
                return Err("reg init must keep x0 == 0".into());
            }
            if value != 0 {
                regs.insert(addr, value);
            }
        }

        for r in &self.rows {
            if !r.active {
                continue;
            }

            let Some(rs1) = &r.reg_read_lane0 else {
                return Err(format!("missing REG lane0 read at cycle {}", r.cycle));
            };
            let Some(rs2) = &r.reg_read_lane1 else {
                return Err(format!("missing REG lane1 read at cycle {}", r.cycle));
            };
            let exp_rs1 = regs.get(&rs1.addr).copied().unwrap_or(0);
            let exp_rs2 = regs.get(&rs2.addr).copied().unwrap_or(0);
            if rs1.value != exp_rs1 {
                return Err(format!(
                    "REG lane0 read value mismatch at cycle {} pc={:#x}: addr={} got={:#x} expected={:#x}",
                    r.cycle, r.pc_before, rs1.addr, rs1.value, exp_rs1
                ));
            }
            if rs2.value != exp_rs2 {
                return Err(format!(
                    "REG lane1 read value mismatch at cycle {} pc={:#x}: addr={} got={:#x} expected={:#x}",
                    r.cycle, r.pc_before, rs2.addr, rs2.value, exp_rs2
                ));
            }

            if let Some(w) = &r.reg_write_lane0 {
                if w.addr == 0 {
                    return Err(format!(
                        "unexpected x0 write at cycle {} pc={:#x}",
                        r.cycle, r.pc_before
                    ));
                }
                if w.value == 0 {
                    regs.remove(&w.addr);
                } else {
                    regs.insert(w.addr, w.value);
                }
            }

            // x0 is always 0.
            regs.remove(&0);
        }

        Ok(())
    }

    /// Validate RAM twist semantics by replaying the RAM state from an initial state.
    ///
    /// - `init_ram` maps `byte_addr` → word value (u32 stored in u64) under the RV32 trace convention.
    /// - Unspecified addresses default to 0.
    /// - Multiple RAM events in a cycle are applied in trace order (e.g. SB/SH read-modify-write).
    pub fn validate_ram_semantics(&self, init_ram: &HashMap<u64, u64>) -> Result<(), String> {
        let mut mem: HashMap<u64, u64> = HashMap::new();
        for (&addr, &value) in init_ram {
            if value == 0 {
                continue;
            }
            mem.insert(addr, value);
        }

        for r in &self.rows {
            if !r.active {
                continue;
            }

            for e in &r.ram_events {
                match e.kind {
                    TwistOpKind::Read => {
                        let exp = mem.get(&e.addr).copied().unwrap_or(0);
                        if e.value != exp {
                            return Err(format!(
                                "RAM read value mismatch at cycle {} pc={:#x}: addr={:#x} got={:#x} expected={:#x}",
                                r.cycle, r.pc_before, e.addr, e.value, exp
                            ));
                        }
                    }
                    TwistOpKind::Write => {
                        if e.value == 0 {
                            mem.remove(&e.addr);
                        } else {
                            mem.insert(e.addr, e.value);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    pub fn to_columns(&self) -> RiscvExecColumns {
        let n = self.rows.len();

        let mut out = RiscvExecColumns {
            active: Vec::with_capacity(n),
            is_virtual: Vec::with_capacity(n),
            virtual_sequence_remaining: Vec::with_capacity(n),
            virtual_transition: Vec::with_capacity(n),
            virtual_commit_link: Vec::with_capacity(n),
            virtual_commit_from_prev: Vec::with_capacity(n),
            cycle: Vec::with_capacity(n),
            pc_before: Vec::with_capacity(n),
            pc_after: Vec::with_capacity(n),
            instr_word: Vec::with_capacity(n),
            opcode: Vec::with_capacity(n),
            rd: Vec::with_capacity(n),
            funct3: Vec::with_capacity(n),
            rs1: Vec::with_capacity(n),
            rs2: Vec::with_capacity(n),
            funct7: Vec::with_capacity(n),
            halted: Vec::with_capacity(n),
            prog_addr: Vec::with_capacity(n),
            prog_value: Vec::with_capacity(n),
            rs1_addr: Vec::with_capacity(n),
            rs1_val: Vec::with_capacity(n),
            rs2_addr: Vec::with_capacity(n),
            rs2_val: Vec::with_capacity(n),
            rd_has_write: Vec::with_capacity(n),
            rd_addr: Vec::with_capacity(n),
            rd_val: Vec::with_capacity(n),
        };

        for r in &self.rows {
            out.active.push(r.active);
            out.is_virtual.push(r.is_virtual);
            out.virtual_sequence_remaining
                .push(r.virtual_sequence_remaining.map(u64::from).unwrap_or(0));
            out.cycle.push(r.cycle);
            out.pc_before.push(r.pc_before);
            out.pc_after.push(r.pc_after);
            out.instr_word.push(r.instr_word);
            out.opcode.push(r.fields.opcode);
            out.rd.push(r.fields.rd);
            out.funct3.push(r.fields.funct3);
            out.rs1.push(r.fields.rs1);
            out.rs2.push(r.fields.rs2);
            out.funct7.push(r.fields.funct7);
            out.halted.push(r.halted);

            match &r.prog_read {
                Some(e) => {
                    out.prog_addr.push(e.addr);
                    out.prog_value.push(e.value);
                }
                None => {
                    out.prog_addr.push(0);
                    out.prog_value.push(0);
                }
            }

            match &r.reg_read_lane0 {
                Some(io) => {
                    out.rs1_addr.push(io.addr);
                    out.rs1_val.push(io.value);
                }
                None => {
                    out.rs1_addr.push(0);
                    out.rs1_val.push(0);
                }
            }

            match &r.reg_read_lane1 {
                Some(io) => {
                    out.rs2_addr.push(io.addr);
                    out.rs2_val.push(io.value);
                }
                None => {
                    out.rs2_addr.push(0);
                    out.rs2_val.push(0);
                }
            }

            match &r.reg_write_lane0 {
                Some(io) => {
                    out.rd_has_write.push(true);
                    out.rd_addr.push(io.addr);
                    out.rd_val.push(io.value);
                }
                None => {
                    out.rd_has_write.push(false);
                    out.rd_addr.push(0);
                    out.rd_val.push(0);
                }
            }
        }

        for i in 0..n {
            let next_active = if i + 1 < n { out.active[i + 1] } else { false };
            let next_is_virtual = if i + 1 < n { out.is_virtual[i + 1] } else { false };
            let transition = out.active[i] && out.is_virtual[i] && next_active && !next_is_virtual;
            out.virtual_transition.push(transition);
            let next_has_write = if i + 1 < n { out.rd_has_write[i + 1] } else { false };
            out.virtual_commit_link.push(transition && next_has_write);
        }
        for i in 0..n {
            let from_prev = if i > 0 { out.virtual_commit_link[i - 1] } else { false };
            out.virtual_commit_from_prev.push(from_prev);
        }

        out
    }
}

impl RiscvExecRow {
    pub fn from_step<Key>(step: &StepTrace<u64, u64, Key>) -> Result<Self, String>
    where
        Key: Copy + TryInto<u128>,
        <Key as TryInto<u128>>::Error: std::fmt::Debug,
    {
        Self::from_step_with_xlen(step, /*machine_xlen=*/ 32)
    }

    pub fn from_step_with_xlen<Key>(step: &StepTrace<u64, u64, Key>, machine_xlen: usize) -> Result<Self, String>
    where
        Key: Copy + TryInto<u128>,
        <Key as TryInto<u128>>::Error: std::fmt::Debug,
    {
        let instr_word = step.opcode;
        let fields = RiscvInstrFields::from_word(instr_word);
        let decoded = decode_instruction_with_xlen(instr_word, machine_xlen).map_err(|e| {
            format!(
                "decode_instruction failed at cycle {} pc={:#x} word={:#x}: {e}",
                step.cycle, step.pc_before, instr_word
            )
        })?;

        // PROG fetch
        let prog_read = {
            let mut reads = step
                .twist_events
                .iter()
                .filter(|e| e.twist_id == PROG_ID && matches!(e.kind, TwistOpKind::Read))
                .cloned();
            let first = reads.next().ok_or_else(|| {
                format!(
                    "missing PROG_ID read event at cycle {} pc={:#x}",
                    step.cycle, step.pc_before
                )
            })?;
            if reads.next().is_some() {
                return Err(format!(
                    "expected exactly 1 PROG_ID read event at cycle {} pc={:#x}",
                    step.cycle, step.pc_before
                ));
            }
            first
        };
        if prog_read.addr != step.pc_before {
            return Err(format!(
                "PROG_ID read addr mismatch at cycle {}: got={:#x} expected pc_before={:#x}",
                step.cycle, prog_read.addr, step.pc_before
            ));
        }
        if prog_read.value != instr_word as u64 {
            return Err(format!(
                "PROG_ID read value mismatch at cycle {} pc={:#x}: got={:#x} expected instr_word={:#x}",
                step.cycle, step.pc_before, prog_read.value, instr_word
            ));
        }
        if prog_read.lane.is_some() {
            return Err(format!(
                "unexpected PROG_ID lane hint at cycle {} pc={:#x}: lane={:?}",
                step.cycle, step.pc_before, prog_read.lane
            ));
        }

        // REG reads (lane 0 and lane 1)
        let mut reg_read_lane0: Option<RiscvRegLaneIo> = None;
        let mut reg_read_lane1: Option<RiscvRegLaneIo> = None;
        let mut reg_write_lane0: Option<RiscvRegLaneIo> = None;
        for e in step.twist_events.iter().filter(|e| e.twist_id == REG_ID) {
            match e.kind {
                TwistOpKind::Read => match e.lane {
                    Some(0) => {
                        if reg_read_lane0.is_some() {
                            return Err(format!(
                                "duplicate REG_ID lane 0 read at cycle {} pc={:#x}",
                                step.cycle, step.pc_before
                            ));
                        }
                        reg_read_lane0 = Some(RiscvRegLaneIo {
                            addr: e.addr,
                            value: e.value,
                        });
                    }
                    Some(1) => {
                        if reg_read_lane1.is_some() {
                            return Err(format!(
                                "duplicate REG_ID lane 1 read at cycle {} pc={:#x}",
                                step.cycle, step.pc_before
                            ));
                        }
                        reg_read_lane1 = Some(RiscvRegLaneIo {
                            addr: e.addr,
                            value: e.value,
                        });
                    }
                    other => {
                        return Err(format!(
                            "unexpected REG_ID read lane {:?} at cycle {} pc={:#x}",
                            other, step.cycle, step.pc_before
                        ));
                    }
                },
                TwistOpKind::Write => match e.lane {
                    Some(0) => {
                        if reg_write_lane0.is_some() {
                            return Err(format!(
                                "duplicate REG_ID lane 0 write at cycle {} pc={:#x}",
                                step.cycle, step.pc_before
                            ));
                        }
                        reg_write_lane0 = Some(RiscvRegLaneIo {
                            addr: e.addr,
                            value: e.value,
                        });
                    }
                    other => {
                        return Err(format!(
                            "unexpected REG_ID write lane {:?} at cycle {} pc={:#x}",
                            other, step.cycle, step.pc_before
                        ));
                    }
                },
            }
        }
        let reg_read_lane0 = reg_read_lane0.ok_or_else(|| {
            format!(
                "missing REG_ID lane 0 read at cycle {} pc={:#x}",
                step.cycle, step.pc_before
            )
        })?;
        let reg_read_lane1 = reg_read_lane1.ok_or_else(|| {
            format!(
                "missing REG_ID lane 1 read at cycle {} pc={:#x}",
                step.cycle, step.pc_before
            )
        })?;
        let has_virtual_reg_addr = reg_read_lane0.addr >= 32
            || reg_read_lane1.addr >= 32
            || reg_write_lane0
                .as_ref()
                .map(|w| w.addr >= 32)
                .unwrap_or(false);
        if has_virtual_reg_addr && !step.is_virtual {
            return Err(format!(
                "non-virtual row uses virtual register address at cycle {} pc={:#x}",
                step.cycle, step.pc_before
            ));
        }
        if step.is_virtual
            && reg_write_lane0
                .as_ref()
                .map(|w| w.addr < 32)
                .unwrap_or(false)
        {
            return Err(format!(
                "virtual row attempted architectural register write at cycle {} pc={:#x}",
                step.cycle, step.pc_before
            ));
        }
        let relax_reg_field_checks = step.is_virtual;
        if let Some(w) = &reg_write_lane0 {
            if w.addr == 0 {
                return Err(format!(
                    "unexpected REG_ID lane 0 write to x0 at cycle {} pc={:#x}",
                    step.cycle, step.pc_before
                ));
            }
            if !relax_reg_field_checks && fields.rd == 0 {
                return Err(format!(
                    "unexpected REG_ID lane 0 write to x0 at cycle {} pc={:#x}",
                    step.cycle, step.pc_before
                ));
            }
            if !relax_reg_field_checks && w.addr != fields.rd as u64 {
                return Err(format!(
                    "REG lane0 write addr mismatch at cycle {} pc={:#x}: got={} expected rd_field={}",
                    step.cycle, step.pc_before, w.addr, fields.rd
                ));
            }
        }

        // Light sanity check: make sure the trace's lane policy matches RV32 trace conventions.
        //
        // - lane0 reads rs1_field always
        // - lane1 reads rs2_field
        if !relax_reg_field_checks {
            let rs2_expected = fields.rs2 as u64;
            if reg_read_lane0.addr != fields.rs1 as u64 {
                return Err(format!(
                    "REG lane0 read addr mismatch at cycle {} pc={:#x}: got={} expected rs1_field={}",
                    step.cycle, step.pc_before, reg_read_lane0.addr, fields.rs1
                ));
            }
            if reg_read_lane1.addr != rs2_expected {
                return Err(format!(
                    "REG lane1 read addr mismatch at cycle {} pc={:#x}: got={} expected={}",
                    step.cycle, step.pc_before, reg_read_lane1.addr, rs2_expected
                ));
            }
        }

        // RAM events
        let ram_events: Vec<TwistEvent<u64, u64>> = step
            .twist_events
            .iter()
            .filter(|e| e.twist_id == RAM_ID)
            .cloned()
            .collect();

        // Shout events
        let mut shout_events: Vec<ShoutEvent<u128, u64>> = step
            .shout_events
            .iter()
            .map(|ev| {
                Ok(ShoutEvent {
                    shout_id: ev.shout_id,
                    key: ev.key.try_into().map_err(|_| {
                        format!(
                            "shout key does not fit u128 at cycle {} pc={:#x}",
                            step.cycle, step.pc_before
                        )
                    })?,
                    value: ev.value,
                })
            })
            .collect::<Result<_, String>>()?;
        if shout_events.is_empty() && !relax_reg_field_checks {
            // Backfill RV32M shout events for trace/event-table consumers.
            //
            // Some trace builders currently omit explicit Shout events for RV32M rows even when
            // the operation is semantically Shout-backed. Reconstruct the canonical event from the
            // decoded op and the architectural operands.
            if let RiscvInstruction::RAlu { op, .. } = &decoded {
                let is_rv32m = matches!(
                    op,
                    RiscvOpcode::Mul
                        | RiscvOpcode::Mulh
                        | RiscvOpcode::Mulhu
                        | RiscvOpcode::Mulhsu
                        | RiscvOpcode::Div
                        | RiscvOpcode::Divu
                        | RiscvOpcode::Rem
                        | RiscvOpcode::Remu
                );
                if is_rv32m {
                    let rs1_val = reg_read_lane0.value;
                    let rs2_val = reg_read_lane1.value;
                    let shout_id = RiscvShoutTables::new(/*xlen=*/ 32).opcode_to_id(*op);
                    let key = encode_lookup_key(*op, rs1_val, rs2_val, /*xlen=*/ 32);
                    let value = compute_op(*op, rs1_val, rs2_val, /*xlen=*/ 32);
                    shout_events.push(ShoutEvent { shout_id, key, value });
                }
            }
        }

        Ok(Self {
            active: true,
            is_virtual: step.is_virtual,
            virtual_sequence_remaining: step.virtual_sequence_remaining,
            cycle: step.cycle,
            pc_before: step.pc_before,
            pc_after: step.pc_after,
            instr_word,
            fields,
            halted: step.halted,
            decoded: Some(decoded),
            prog_read: Some(prog_read),
            reg_read_lane0: Some(reg_read_lane0),
            reg_read_lane1: Some(reg_read_lane1),
            reg_write_lane0,
            ram_events,
            shout_events,
        })
    }

    pub fn inactive(cycle: u64, pc: u64, halted: bool) -> Self {
        Self {
            active: false,
            is_virtual: false,
            virtual_sequence_remaining: None,
            cycle,
            pc_before: pc,
            pc_after: pc,
            instr_word: 0,
            fields: RiscvInstrFields::from_word(0),
            halted,
            decoded: None,
            prog_read: None,
            reg_read_lane0: None,
            reg_read_lane1: None,
            reg_write_lane0: None,
            ram_events: Vec::new(),
            shout_events: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Rv32ShoutEventRow {
    /// Row index within the padded exec table (0..t).
    pub row_idx: usize,
    pub cycle: u64,
    pub pc: u64,
    pub shout_id: u32,
    pub opcode: Option<RiscvOpcode>,
    /// Canonicalized key: for shift ops, `rhs` is masked to 5 bits.
    pub key: u128,
    pub lhs: u64,
    pub rhs: u64,
    pub value: u64,
}

#[derive(Clone, Debug)]
pub struct Rv32ShoutEventTable {
    pub rows: Vec<Rv32ShoutEventRow>,
}

impl Rv32ShoutEventTable {
    pub fn from_exec_table(exec: &RiscvExecTable) -> Result<Self, String> {
        let shout_tables = RiscvShoutTables::new(/*xlen=*/ 32);
        let mut rows = Vec::new();

        for (row_idx, r) in exec.rows.iter().enumerate() {
            if !r.active {
                continue;
            }
            for ev in r.shout_events.iter() {
                let opcode = shout_tables.id_to_opcode(ev.shout_id);
                let fallback_lhs = r.reg_read_lane0.as_ref().map(|io| io.value).unwrap_or(0);
                let fallback_rhs = r.reg_read_lane1.as_ref().map(|io| io.value).unwrap_or(0);
                let (lhs, rhs_raw) = if let Some(op) = opcode {
                    try_decode_lookup_operands(op, ev.key, operand_mode_keys_enabled(), /*xlen=*/ 32)
                        .unwrap_or((fallback_lhs, fallback_rhs))
                } else {
                    (fallback_lhs, fallback_rhs)
                };
                let rhs = if matches!(opcode, Some(RiscvOpcode::Sll | RiscvOpcode::Srl | RiscvOpcode::Sra)) {
                    rhs_raw & 0x1F
                } else {
                    rhs_raw
                };
                let key = if rhs != rhs_raw {
                    if let Some(op) = opcode {
                        encode_lookup_key(op, lhs, rhs, /*xlen=*/ 32)
                    } else {
                        ev.key
                    }
                } else {
                    ev.key
                };

                rows.push(Rv32ShoutEventRow {
                    row_idx,
                    cycle: r.cycle,
                    pc: r.pc_before,
                    shout_id: ev.shout_id.0,
                    opcode,
                    key,
                    lhs,
                    rhs,
                    value: ev.value,
                });
            }
        }

        Ok(Self { rows })
    }
}

#[derive(Clone, Debug)]
pub struct Rv64ShoutEventRow {
    /// Row index within the padded exec table (0..t).
    pub row_idx: usize,
    pub cycle: u64,
    pub pc: u64,
    pub shout_id: u32,
    pub opcode: Option<RiscvOpcode>,
    /// Canonicalized key: base shifts use 6-bit rhs masking, W-shifts use 5-bit rhs masking.
    pub key: u128,
    pub lhs: u64,
    pub rhs: u64,
    pub value: u64,
}

#[derive(Clone, Debug)]
pub struct Rv64ShoutEventTable {
    pub rows: Vec<Rv64ShoutEventRow>,
}

impl Rv64ShoutEventTable {
    pub fn from_exec_table(exec: &RiscvExecTable) -> Result<Self, String> {
        let shout_tables = RiscvShoutTables::new(/*xlen=*/ 64);
        let mut rows = Vec::new();

        for (row_idx, r) in exec.rows.iter().enumerate() {
            if !r.active {
                continue;
            }
            for ev in &r.shout_events {
                let opcode = shout_tables.id_to_opcode(ev.shout_id);
                let fallback_lhs = r.reg_read_lane0.as_ref().map(|io| io.value).unwrap_or(0);
                let fallback_rhs = r.reg_read_lane1.as_ref().map(|io| io.value).unwrap_or(0);
                let (lhs, rhs_raw) = if let Some(op) = opcode {
                    try_decode_lookup_operands(op, ev.key, operand_mode_keys_enabled(), /*xlen=*/ 64)
                        .unwrap_or((fallback_lhs, fallback_rhs))
                } else {
                    (fallback_lhs, fallback_rhs)
                };
                let rhs = match opcode {
                    Some(RiscvOpcode::Sll | RiscvOpcode::Srl | RiscvOpcode::Sra) => rhs_raw & 0x3F,
                    Some(RiscvOpcode::Sllw | RiscvOpcode::Srlw | RiscvOpcode::Sraw) => rhs_raw & 0x1F,
                    _ => rhs_raw,
                };
                let key = if rhs != rhs_raw {
                    if let Some(op) = opcode {
                        encode_lookup_key(op, lhs, rhs, /*xlen=*/ 64)
                    } else {
                        ev.key
                    }
                } else {
                    ev.key
                };

                rows.push(Rv64ShoutEventRow {
                    row_idx,
                    cycle: r.cycle,
                    pc: r.pc_before,
                    shout_id: ev.shout_id.0,
                    opcode,
                    key,
                    lhs,
                    rhs,
                    value: ev.value,
                });
            }
        }

        Ok(Self { rows })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rv32RegEventKind {
    ReadLane0,
    ReadLane1,
    WriteLane0,
}

#[derive(Clone, Debug)]
pub struct Rv32RegEventRow {
    pub cycle: u64,
    pub pc: u64,
    pub kind: Rv32RegEventKind,
    pub addr: u8,
    pub prev_val: u64,
    pub next_val: u64,
}

#[derive(Clone, Debug)]
pub struct Rv32RegEventTable {
    pub rows: Vec<Rv32RegEventRow>,
}

impl Rv32RegEventTable {
    pub fn from_exec_table(exec: &RiscvExecTable, init_regs: &HashMap<u64, u64>) -> Result<Self, String> {
        let mut regs: HashMap<u64, u64> = HashMap::new();
        for (&addr, &value) in init_regs {
            if addr == 0 && value != 0 {
                return Err("reg init must keep x0 == 0".into());
            }
            if value != 0 {
                regs.insert(addr, value);
            }
        }

        let mut rows: Vec<Rv32RegEventRow> = Vec::new();
        for r in &exec.rows {
            if !r.active {
                continue;
            }

            let Some(rs1) = &r.reg_read_lane0 else {
                return Err(format!("missing REG lane0 read at cycle {}", r.cycle));
            };
            let Some(rs2) = &r.reg_read_lane1 else {
                return Err(format!("missing REG lane1 read at cycle {}", r.cycle));
            };

            // Reads happen before the optional write.
            let rs1_prev = regs.get(&rs1.addr).copied().unwrap_or(0);
            let rs2_prev = regs.get(&rs2.addr).copied().unwrap_or(0);
            if rs1.value != rs1_prev {
                return Err(format!(
                    "REG lane0 read value mismatch at cycle {} pc={:#x}: addr={} got={:#x} expected={:#x}",
                    r.cycle, r.pc_before, rs1.addr, rs1.value, rs1_prev
                ));
            }
            if rs2.value != rs2_prev {
                return Err(format!(
                    "REG lane1 read value mismatch at cycle {} pc={:#x}: addr={} got={:#x} expected={:#x}",
                    r.cycle, r.pc_before, rs2.addr, rs2.value, rs2_prev
                ));
            }
            let rs1_addr = u8::try_from(rs1.addr)
                .map_err(|_| format!("REG lane0 addr does not fit u8 at cycle {}: {}", r.cycle, rs1.addr))?;
            let rs2_addr = u8::try_from(rs2.addr)
                .map_err(|_| format!("REG lane1 addr does not fit u8 at cycle {}: {}", r.cycle, rs2.addr))?;

            rows.push(Rv32RegEventRow {
                cycle: r.cycle,
                pc: r.pc_before,
                kind: Rv32RegEventKind::ReadLane0,
                addr: rs1_addr,
                prev_val: rs1_prev,
                next_val: rs1_prev,
            });
            rows.push(Rv32RegEventRow {
                cycle: r.cycle,
                pc: r.pc_before,
                kind: Rv32RegEventKind::ReadLane1,
                addr: rs2_addr,
                prev_val: rs2_prev,
                next_val: rs2_prev,
            });

            if let Some(w) = &r.reg_write_lane0 {
                if w.addr == 0 {
                    return Err(format!(
                        "unexpected x0 write at cycle {} pc={:#x}",
                        r.cycle, r.pc_before
                    ));
                }

                let prev = regs.get(&w.addr).copied().unwrap_or(0);
                let next = w.value;
                if next == 0 {
                    regs.remove(&w.addr);
                } else {
                    regs.insert(w.addr, next);
                }
                regs.remove(&0);
                let w_addr = u8::try_from(w.addr)
                    .map_err(|_| format!("REG write addr does not fit u8 at cycle {}: {}", r.cycle, w.addr))?;

                rows.push(Rv32RegEventRow {
                    cycle: r.cycle,
                    pc: r.pc_before,
                    kind: Rv32RegEventKind::WriteLane0,
                    addr: w_addr,
                    prev_val: prev,
                    next_val: next,
                });
            }
        }

        Ok(Self { rows })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rv32RamEventKind {
    Read,
    Write,
}

#[derive(Clone, Debug)]
pub struct Rv32RamEventRow {
    pub cycle: u64,
    pub pc: u64,
    pub kind: Rv32RamEventKind,
    pub addr: u64,
    pub prev_val: u64,
    pub next_val: u64,
}

#[derive(Clone, Debug)]
pub struct Rv32RamEventTable {
    pub rows: Vec<Rv32RamEventRow>,
}

impl Rv32RamEventTable {
    pub fn from_exec_table(exec: &RiscvExecTable, init_ram: &HashMap<u64, u64>) -> Result<Self, String> {
        let mut mem: HashMap<u64, u64> = HashMap::new();
        for (&addr, &value) in init_ram {
            if value == 0 {
                continue;
            }
            mem.insert(addr, value);
        }

        let mut rows: Vec<Rv32RamEventRow> = Vec::new();
        for r in &exec.rows {
            if !r.active {
                continue;
            }

            for e in &r.ram_events {
                match e.kind {
                    TwistOpKind::Read => {
                        let prev = mem.get(&e.addr).copied().unwrap_or(0);
                        let next = prev;
                        if e.value != prev {
                            return Err(format!(
                                "RAM read value mismatch at cycle {} pc={:#x}: addr={:#x} got={:#x} expected={:#x}",
                                r.cycle, r.pc_before, e.addr, e.value, prev
                            ));
                        }
                        rows.push(Rv32RamEventRow {
                            cycle: r.cycle,
                            pc: r.pc_before,
                            kind: Rv32RamEventKind::Read,
                            addr: e.addr,
                            prev_val: prev,
                            next_val: next,
                        });
                    }
                    TwistOpKind::Write => {
                        let prev = mem.get(&e.addr).copied().unwrap_or(0);
                        let next = e.value;
                        if next == 0 {
                            mem.remove(&e.addr);
                        } else {
                            mem.insert(e.addr, next);
                        }
                        rows.push(Rv32RamEventRow {
                            cycle: r.cycle,
                            pc: r.pc_before,
                            kind: Rv32RamEventKind::Write,
                            addr: e.addr,
                            prev_val: prev,
                            next_val: next,
                        });
                    }
                }
            }
        }

        Ok(Self { rows })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PoseidonSidecarMode {
    Absorbing,
    Finalized,
}

#[derive(Clone, Debug)]
pub struct RiscvPoseidonCycleEventRow {
    pub cycle: u64,
    pub op_absorb: bool,
    pub op_finalize: bool,
    pub op_squeeze: bool,
    /// Pre-instruction mode flag: true when the sponge is in `Finalized` mode.
    pub mode_finalized: bool,
    pub call_ctr: u64,
    pub cursor_before: u8,
    pub cursor_after: u8,
    pub do_perm_slot0: bool,
    pub do_perm_slot1: bool,
    pub absorb_lo32: u32,
    pub absorb_hi32: u32,
    pub squeeze_idx: u8,
    pub squeeze_word_u32: u32,
    pub state_pre: [u64; 8],
    pub state_post: [u64; 8],
    pub canonical_lo_sum: u32,
    pub canonical_hi_sum: u32,
    pub canonical_c0: u32,
    pub canonical_c1: u32,
}

#[derive(Clone, Debug)]
pub struct RiscvPoseidonPermSlotMetaRow {
    pub cycle: u64,
    pub slot: u8,
    pub call_ctr: u64,
    pub state_in: [u64; 8],
    pub state_out: [u64; 8],
}

#[derive(Clone, Debug)]
pub struct RiscvPoseidonSidecarTable {
    pub cycle_rows: Vec<RiscvPoseidonCycleEventRow>,
    pub perm_rows: Vec<RiscvPoseidonPermSlotMetaRow>,
}

#[inline]
fn poseidon_state_to_u64(state: &[Goldilocks; 8]) -> [u64; 8] {
    let mut out = [0u64; 8];
    for (i, x) in state.iter().enumerate() {
        out[i] = x.as_canonical_u64();
    }
    out
}

#[inline]
fn canonical_u64_lt_goldilocks_aux(v: u64) -> (u32, u32, u32, u32) {
    let lo = v as u32;
    let hi = (v >> 32) as u32;
    let (lo_sum, c0) = lo.overflowing_add(0xFFFF_FFFF);
    let (hi_sum, c1) = hi.overflowing_add(if c0 { 1 } else { 0 });
    (lo_sum, hi_sum, u32::from(c0), u32::from(c1))
}

impl RiscvPoseidonSidecarTable {
    pub fn from_exec_table(exec: &RiscvExecTable) -> Result<Self, String> {
        const WIDTH: usize = neo_ccs::crypto::poseidon2_goldilocks::WIDTH;
        const RATE: usize = neo_ccs::crypto::poseidon2_goldilocks::RATE;
        const DIGEST_LEN: usize = neo_ccs::crypto::poseidon2_goldilocks::DIGEST_LEN;

        let perm = neo_ccs::crypto::poseidon2_goldilocks::permutation();
        let mut mode = PoseidonSidecarMode::Absorbing;
        let mut state = [Goldilocks::ZERO; WIDTH];
        let mut absorb_cursor: usize = 0;
        let mut digest_words = [0u32; DIGEST_LEN * 2];
        let mut call_ctr = 0u64;

        let mut cycle_rows = Vec::with_capacity(exec.rows.len());
        let mut perm_rows: Vec<RiscvPoseidonPermSlotMetaRow> = Vec::new();

        for row in exec.rows.iter().filter(|r| r.active) {
            let state_pre = poseidon_state_to_u64(&state);
            let mut out = RiscvPoseidonCycleEventRow {
                cycle: row.cycle,
                op_absorb: false,
                op_finalize: false,
                op_squeeze: false,
                mode_finalized: mode == PoseidonSidecarMode::Finalized,
                call_ctr,
                cursor_before: absorb_cursor as u8,
                cursor_after: absorb_cursor as u8,
                do_perm_slot0: false,
                do_perm_slot1: false,
                absorb_lo32: 0,
                absorb_hi32: 0,
                squeeze_idx: 0,
                squeeze_word_u32: 0,
                state_pre,
                state_post: state_pre,
                canonical_lo_sum: 0,
                canonical_hi_sum: 0,
                canonical_c0: 0,
                canonical_c1: 0,
            };

            if let Some(decoded) = row.decoded.as_ref() {
                match decoded {
                    RiscvInstruction::Poseidon2AbsorbElem { .. } => {
                        out.op_absorb = true;
                        if mode == PoseidonSidecarMode::Finalized {
                            // Start a new message context.
                            state.fill(Goldilocks::ZERO);
                            absorb_cursor = 0;
                            mode = PoseidonSidecarMode::Absorbing;
                            digest_words.fill(0);
                            call_ctr = call_ctr.wrapping_add(1);
                            out.call_ctr = call_ctr;
                            out.cursor_before = 0;
                            out.state_pre = poseidon_state_to_u64(&state);
                        }

                        let rs1 = row
                            .reg_read_lane0
                            .as_ref()
                            .ok_or_else(|| format!("poseidon absorb: missing rs1 read at cycle {}", row.cycle))?
                            .value as u32;
                        let rs2 = row
                            .reg_read_lane1
                            .as_ref()
                            .ok_or_else(|| format!("poseidon absorb: missing rs2 read at cycle {}", row.cycle))?
                            .value as u32;
                        out.absorb_lo32 = rs1;
                        out.absorb_hi32 = rs2;

                        let elem_u64 = (rs1 as u64) | ((rs2 as u64) << 32);
                        state[absorb_cursor] += Goldilocks::from_u64(elem_u64);
                        absorb_cursor += 1;

                        if absorb_cursor == RATE {
                            out.do_perm_slot0 = true;
                            let in_state = poseidon_state_to_u64(&state);
                            state = perm.permute(state);
                            let out_state = poseidon_state_to_u64(&state);
                            perm_rows.push(RiscvPoseidonPermSlotMetaRow {
                                cycle: row.cycle,
                                slot: 0,
                                call_ctr,
                                state_in: in_state,
                                state_out: out_state,
                            });
                            absorb_cursor = 0;
                        }
                        out.cursor_after = absorb_cursor as u8;
                    }
                    RiscvInstruction::Poseidon2Finalize => {
                        out.op_finalize = true;
                        if mode == PoseidonSidecarMode::Finalized {
                            return Err(format!(
                                "poseidon finalize called in Finalized mode at cycle {}",
                                row.cycle
                            ));
                        }
                        if absorb_cursor > 0 {
                            out.do_perm_slot0 = true;
                            let in_state = poseidon_state_to_u64(&state);
                            state = perm.permute(state);
                            let out_state = poseidon_state_to_u64(&state);
                            perm_rows.push(RiscvPoseidonPermSlotMetaRow {
                                cycle: row.cycle,
                                slot: 0,
                                call_ctr,
                                state_in: in_state,
                                state_out: out_state,
                            });
                            absorb_cursor = 0;
                        }

                        state[0] += Goldilocks::ONE;
                        out.do_perm_slot1 = true;
                        let in_state = poseidon_state_to_u64(&state);
                        state = perm.permute(state);
                        let out_state = poseidon_state_to_u64(&state);
                        perm_rows.push(RiscvPoseidonPermSlotMetaRow {
                            cycle: row.cycle,
                            slot: 1,
                            call_ctr,
                            state_in: in_state,
                            state_out: out_state,
                        });

                        for i in 0..DIGEST_LEN {
                            let v = state[i].as_canonical_u64();
                            digest_words[2 * i] = v as u32;
                            digest_words[2 * i + 1] = (v >> 32) as u32;
                        }
                        mode = PoseidonSidecarMode::Finalized;
                        out.cursor_after = 0;
                    }
                    RiscvInstruction::Poseidon2SqueezeWord { rd, idx } => {
                        out.op_squeeze = true;
                        if mode != PoseidonSidecarMode::Finalized {
                            return Err(format!(
                                "poseidon squeeze called before finalize at cycle {}",
                                row.cycle
                            ));
                        }
                        let idx_usize = *idx as usize;
                        if idx_usize >= digest_words.len() {
                            return Err(format!(
                                "poseidon squeeze idx out of range at cycle {}: idx={}",
                                row.cycle, idx
                            ));
                        }
                        out.squeeze_idx = *idx;
                        let word = digest_words[idx_usize];
                        out.squeeze_word_u32 = word;
                        if *rd != 0 {
                            let write_word = row
                                .reg_write_lane0
                                .as_ref()
                                .ok_or_else(|| format!("poseidon squeeze: missing rd write at cycle {}", row.cycle))?
                                .value as u32;
                            if write_word != word {
                                return Err(format!(
                                    "poseidon squeeze word mismatch at cycle {}: got={:#x}, expected={:#x}",
                                    row.cycle, write_word, word
                                ));
                            }
                        }

                        let digest_elem_idx = idx_usize / 2;
                        let digest_elem = state[digest_elem_idx].as_canonical_u64();
                        let (lo_sum, hi_sum, c0, c1) = canonical_u64_lt_goldilocks_aux(digest_elem);
                        out.canonical_lo_sum = lo_sum;
                        out.canonical_hi_sum = hi_sum;
                        out.canonical_c0 = c0;
                        out.canonical_c1 = c1;
                        out.cursor_after = absorb_cursor as u8;
                    }
                    _ => {
                        out.cursor_after = absorb_cursor as u8;
                    }
                }
            }

            out.state_post = poseidon_state_to_u64(&state);
            cycle_rows.push(out);
        }

        Ok(Self { cycle_rows, perm_rows })
    }
}

pub type Rv32PoseidonCycleEventRow = RiscvPoseidonCycleEventRow;
pub type Rv32PoseidonPermSlotMetaRow = RiscvPoseidonPermSlotMetaRow;
pub type Rv32PoseidonSidecarTable = RiscvPoseidonSidecarTable;
