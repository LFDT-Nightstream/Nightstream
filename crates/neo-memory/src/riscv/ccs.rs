//! RV32 "B1" RISC-V step CCS (shared-bus compatible).
//!
//! This module provides a **sound, shared-bus-compatible** step circuit for a small,
//! MVP RV32 subset. The circuit is expressed as an identity-first, square R1CS→CCS:
//! - `M0 = I_n` (required by the Ajtai/NC pipeline)
//! - `A(z) * B(z) = C(z)` with `C = 0` for all rows
//!
//! The witness `z` includes a **reserved bus tail** whose column schema matches
//! `cpu::bus_layout::BusLayout`. The bus tail itself is written from `StepTrace`
//! events by `R1csCpu` (shared-bus mode), and is verified by the Twist/Shout sidecars.
//!
//! The CCS here constrains the **CPU glue**:
//! - ROM fetch binding (`PROG_ID`) via shared-bus bindings
//! - instruction decode from a committed 32-bit instruction word
//! - register-file update pattern
//! - RAM load/store binding (`RAM_ID`) via shared-bus bindings
//! - Shout key wiring for `ADD` lookups (table id 3)
//!
//! MVP instruction subset (RV32, no compressed):
//! - `ADD`, `ADDI`, `LW`, `SW`, `LUI`, `AUIPC`, `ECALL(imm=0)` (treated as `Halt`)

use std::collections::HashMap;

use neo_ccs::matrix::Mat;
use neo_ccs::poly::{SparsePoly, Term};
use neo_ccs::relations::CcsStructure;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

use crate::cpu::bus_layout::{build_bus_layout_for_instances, BusLayout};
use crate::cpu::constraints::{ShoutCpuBinding, TwistCpuBinding};
use crate::cpu::r1cs_adapter::SharedCpuBusConfig;
use crate::plain::PlainMemLayout;
use crate::riscv::lookups::{decode_instruction, RiscvInstruction, RiscvMemOp, RiscvOpcode, RAM_ID, PROG_ID};
use neo_vm_trace::{StepTrace, TwistOpKind};

const RV32_XLEN: usize = 32;
const ADD_TABLE_ID: u32 = 3; // `RiscvOpcode::Add` in `RiscvShoutTables::opcode_to_id`

#[derive(Clone, Debug)]
struct Constraint<Ff: PrimeCharacteristicRing + Copy> {
    condition_col: usize,
    negate_condition: bool,
    additional_condition_cols: Vec<usize>,
    b_terms: Vec<(usize, Ff)>,
}

impl<Ff: PrimeCharacteristicRing + Copy> Constraint<Ff> {
    fn eq_const(condition_col: usize, const_one_col: usize, left: usize, c: u64) -> Self {
        Self {
            condition_col,
            negate_condition: false,
            additional_condition_cols: Vec::new(),
            b_terms: vec![(left, Ff::ONE), (const_one_col, -Ff::from_u64(c))],
        }
    }

    fn zero(condition_col: usize, col: usize) -> Self {
        Self {
            condition_col,
            negate_condition: false,
            additional_condition_cols: Vec::new(),
            b_terms: vec![(col, Ff::ONE)],
        }
    }

    fn terms(condition_col: usize, negate_condition: bool, b_terms: Vec<(usize, Ff)>) -> Self {
        Self {
            condition_col,
            negate_condition,
            additional_condition_cols: Vec::new(),
            b_terms,
        }
    }

    fn terms_or(condition_cols: &[usize], negate_condition: bool, b_terms: Vec<(usize, Ff)>) -> Self {
        assert!(!condition_cols.is_empty(), "need at least one condition column");
        Self {
            condition_col: condition_cols[0],
            negate_condition,
            additional_condition_cols: condition_cols[1..].to_vec(),
            b_terms,
        }
    }
}

fn build_identity_first_r1cs_ccs(constraints: &[Constraint<F>], m: usize, const_one_col: usize) -> Result<CcsStructure<F>, String> {
    let n = m;
    if constraints.len() > n {
        return Err(format!(
            "RV32 B1 CCS: too many constraints ({}) for square CCS with m=n={}",
            constraints.len(),
            n
        ));
    }

    let mut a_data = vec![F::ZERO; n * m];
    let mut b_data = vec![F::ZERO; n * m];
    let c_data = vec![F::ZERO; n * m];

    for (row, c) in constraints.iter().enumerate() {
        if c.negate_condition {
            a_data[row * m + const_one_col] = F::ONE;
            a_data[row * m + c.condition_col] += -F::ONE;
            for &col in &c.additional_condition_cols {
                a_data[row * m + col] += -F::ONE;
            }
        } else {
            a_data[row * m + c.condition_col] += F::ONE;
            for &col in &c.additional_condition_cols {
                a_data[row * m + col] += F::ONE;
            }
        }

        for &(col, coeff) in &c.b_terms {
            b_data[row * m + col] += coeff;
        }
    }

    let i_n = Mat::identity(n);
    let a = Mat::from_row_major(n, m, a_data);
    let b = Mat::from_row_major(n, m, b_data);
    let c = Mat::from_row_major(n, m, c_data);

    let f = SparsePoly::new(
        4,
        vec![
            Term {
                coeff: F::ONE,
                exps: vec![0, 1, 1, 0],
            },
            Term {
                coeff: -F::ONE,
                exps: vec![0, 0, 0, 1],
            },
        ],
    );

    CcsStructure::new(vec![i_n, a, b, c], f).map_err(|e| format!("RV32 B1 CCS: invalid structure: {e:?}"))
}

fn injection_constraint_count(shout_ell_addrs: &[usize], twist_ell_addrs: &[usize]) -> usize {
    // Shout per-instance:
    // - value binding, selector binding, key binding, padding val = 4
    // - padding bits = ell_addr
    let shout: usize = shout_ell_addrs.iter().map(|&ell_addr| 4 + ell_addr).sum();
    // Twist per-instance (no inc binding):
    // - 2 value bindings + 2 selector bindings + 2 address bindings = 6
    // - 3 padding values = 3
    // - 2*ell_addr padding bits = 2*ell_addr
    let twist: usize = twist_ell_addrs.iter().map(|&ell_addr| 9 + 2 * ell_addr).sum();
    shout + twist
}

fn pow2_u64(i: usize) -> u64 {
    1u64 << i
}

/// Witness/column layout for the RV32 B1 step circuit.
#[derive(Clone, Debug)]
pub struct Rv32B1Layout {
    pub m_in: usize,
    pub m: usize,
    pub const_one: usize,

    pub pc_in: usize,
    pub pc_out: usize,
    pub instr_word: usize,

    pub regs_in_start: usize,
    pub regs_out_start: usize,

    pub instr_bits_start: usize, // 32 bits

    pub opcode: usize,
    pub funct3: usize,
    pub funct7: usize,
    pub rd_field: usize,
    pub rs1_field: usize,
    pub rs2_field: usize,

    pub imm12_raw: usize,
    pub imm_i: usize,
    pub imm_s: usize,
    pub imm_u: usize,

    pub is_add: usize,
    pub is_addi: usize,
    pub is_lw: usize,
    pub is_sw: usize,
    pub is_lui: usize,
    pub is_auipc: usize,
    pub is_halt: usize,

    pub rs1_sel_start: usize, // 32
    pub rs2_sel_start: usize, // 32
    pub rd_sel_start: usize,  // 32

    pub rs1_val: usize,
    pub rs2_val: usize,

    pub eff_addr: usize,
    pub rd_write_val: usize,

    pub add_has_lookup: usize,
    pub lookup_key: usize,

    pub bus: BusLayout,
    pub ram_twist_idx: usize,
    pub prog_twist_idx: usize,
    pub add_shout_idx: usize,
}

impl Rv32B1Layout {
    pub fn reg_in(&self, r: usize) -> usize {
        assert!(r < 32);
        self.regs_in_start + r
    }

    pub fn reg_out(&self, r: usize) -> usize {
        assert!(r < 32);
        self.regs_out_start + r
    }

    pub fn instr_bit(&self, i: usize) -> usize {
        assert!(i < 32);
        self.instr_bits_start + i
    }

    pub fn rs1_sel(&self, r: usize) -> usize {
        assert!(r < 32);
        self.rs1_sel_start + r
    }

    pub fn rs2_sel(&self, r: usize) -> usize {
        assert!(r < 32);
        self.rs2_sel_start + r
    }

    pub fn rd_sel(&self, r: usize) -> usize {
        assert!(r < 32);
        self.rd_sel_start + r
    }
}

fn derive_mem_ids_and_ell_addrs(mem_layouts: &HashMap<u32, PlainMemLayout>) -> Result<(Vec<u32>, Vec<usize>), String> {
    let mut mem_ids: Vec<u32> = mem_layouts.keys().copied().collect();
    mem_ids.sort_unstable();

    let mut twist_ell_addrs = Vec::with_capacity(mem_ids.len());
    for mem_id in &mem_ids {
        let layout = mem_layouts
            .get(mem_id)
            .ok_or_else(|| format!("missing mem_layout for mem_id={mem_id}"))?;
        if layout.n_side == 0 || !layout.n_side.is_power_of_two() {
            return Err(format!("mem_id={mem_id}: n_side={} must be power of two", layout.n_side));
        }
        let ell = layout.n_side.trailing_zeros() as usize;
        twist_ell_addrs.push(layout.d * ell);
    }

    Ok((mem_ids, twist_ell_addrs))
}

fn build_layout_with_m(m: usize, mem_layouts: &HashMap<u32, PlainMemLayout>) -> Result<Rv32B1Layout, String> {
    let m_in = 1usize;
    let const_one = 0usize;

    // Fixed CPU column allocation (CPU region only). All indices must be < bus.bus_base.
    let mut col = m_in;
    let pc_in = col;
    col += 1;
    let pc_out = col;
    col += 1;
    let instr_word = col;
    col += 1;

    let regs_in_start = col;
    col += 32;
    let regs_out_start = col;
    col += 32;

    let instr_bits_start = col;
    col += 32;

    let opcode = col;
    col += 1;
    let funct3 = col;
    col += 1;
    let funct7 = col;
    col += 1;
    let rd_field = col;
    col += 1;
    let rs1_field = col;
    col += 1;
    let rs2_field = col;
    col += 1;

    let imm12_raw = col;
    col += 1;
    let imm_i = col;
    col += 1;
    let imm_s = col;
    col += 1;
    let imm_u = col;
    col += 1;

    let is_add = col;
    col += 1;
    let is_addi = col;
    col += 1;
    let is_lw = col;
    col += 1;
    let is_sw = col;
    col += 1;
    let is_lui = col;
    col += 1;
    let is_auipc = col;
    col += 1;
    let is_halt = col;
    col += 1;

    let rs1_sel_start = col;
    col += 32;
    let rs2_sel_start = col;
    col += 32;
    let rd_sel_start = col;
    col += 32;

    let rs1_val = col;
    col += 1;
    let rs2_val = col;
    col += 1;

    let eff_addr = col;
    col += 1;
    let rd_write_val = col;
    col += 1;

    let add_has_lookup = col;
    col += 1;
    let lookup_key = col;
    col += 1;

    let cpu_cols_used = col;

    let (mem_ids, twist_ell_addrs) = derive_mem_ids_and_ell_addrs(mem_layouts)?;
    let shout_ell_addrs = [2 * RV32_XLEN]; // d=64, ell=1

    let bus = build_bus_layout_for_instances(m, m_in, 1, shout_ell_addrs, twist_ell_addrs.clone())?;
    if cpu_cols_used > bus.bus_base {
        return Err(format!(
            "RV32 B1 layout: CPU columns end at {cpu_cols_used}, but bus_base={} (need more padding columns before bus tail)",
            bus.bus_base
        ));
    }

    // Determine which twist instance index corresponds to RAM/PROG in the sorted mem_ids order.
    let ram_id = RAM_ID.0;
    let prog_id = PROG_ID.0;
    let ram_twist_idx = mem_ids
        .iter()
        .position(|&id| id == ram_id)
        .ok_or_else(|| format!("mem_layouts missing RAM_ID={ram_id}"))?;
    let prog_twist_idx = mem_ids
        .iter()
        .position(|&id| id == prog_id)
        .ok_or_else(|| format!("mem_layouts missing PROG_ID={prog_id}"))?;

    Ok(Rv32B1Layout {
        m_in,
        m,
        const_one,
        pc_in,
        pc_out,
        instr_word,
        regs_in_start,
        regs_out_start,
        instr_bits_start,
        opcode,
        funct3,
        funct7,
        rd_field,
        rs1_field,
        rs2_field,
        imm12_raw,
        imm_i,
        imm_s,
        imm_u,
        is_add,
        is_addi,
        is_lw,
        is_sw,
        is_lui,
        is_auipc,
        is_halt,
        rs1_sel_start,
        rs2_sel_start,
        rd_sel_start,
        rs1_val,
        rs2_val,
        eff_addr,
        rd_write_val,
        add_has_lookup,
        lookup_key,
        bus,
        ram_twist_idx,
        prog_twist_idx,
        add_shout_idx: 0, // only ADD table in MVP
    })
}

fn semantic_constraints(layout: &Rv32B1Layout, mem_layouts: &HashMap<u32, PlainMemLayout>) -> Result<Vec<Constraint<F>>, String> {
    let one = layout.const_one;

    let mut constraints = Vec::<Constraint<F>>::new();

    // x0 hardwired.
    constraints.push(Constraint::zero(one, layout.reg_in(0)));
    constraints.push(Constraint::zero(one, layout.reg_out(0)));

    // PC update: pc_out = pc_in + 4.
    constraints.push(Constraint::terms(
        one,
        false,
        vec![(layout.pc_out, F::ONE), (layout.pc_in, -F::ONE), (one, -F::from_u64(4))],
    ));

    // Instruction bits: boolean.
    for i in 0..32 {
        let b = layout.instr_bit(i);
        constraints.push(Constraint::terms(b, false, vec![(b, F::ONE), (one, -F::ONE)]));
    }

    // Pack instr_word = Σ 2^i bit[i]
    {
        let mut terms = vec![(layout.instr_word, F::ONE)];
        for i in 0..32 {
            terms.push((layout.instr_bit(i), -F::from_u64(pow2_u64(i))));
        }
        constraints.push(Constraint::terms(one, false, terms));
    }

    // Pack opcode/funct/fields from bits.
    {
        // opcode = bits[0..6]
        let mut terms = vec![(layout.opcode, F::ONE)];
        for i in 0..7 {
            terms.push((layout.instr_bit(i), -F::from_u64(pow2_u64(i))));
        }
        constraints.push(Constraint::terms(one, false, terms));
    }
    {
        // rd_field = bits[7..11]
        let mut terms = vec![(layout.rd_field, F::ONE)];
        for i in 0..5 {
            terms.push((layout.instr_bit(7 + i), -F::from_u64(pow2_u64(i))));
        }
        constraints.push(Constraint::terms(one, false, terms));
    }
    {
        // funct3 = bits[12..14]
        let mut terms = vec![(layout.funct3, F::ONE)];
        for i in 0..3 {
            terms.push((layout.instr_bit(12 + i), -F::from_u64(pow2_u64(i))));
        }
        constraints.push(Constraint::terms(one, false, terms));
    }
    {
        // rs1_field = bits[15..19]
        let mut terms = vec![(layout.rs1_field, F::ONE)];
        for i in 0..5 {
            terms.push((layout.instr_bit(15 + i), -F::from_u64(pow2_u64(i))));
        }
        constraints.push(Constraint::terms(one, false, terms));
    }
    {
        // rs2_field = bits[20..24]
        let mut terms = vec![(layout.rs2_field, F::ONE)];
        for i in 0..5 {
            terms.push((layout.instr_bit(20 + i), -F::from_u64(pow2_u64(i))));
        }
        constraints.push(Constraint::terms(one, false, terms));
    }
    {
        // funct7 = bits[25..31]
        let mut terms = vec![(layout.funct7, F::ONE)];
        for i in 0..7 {
            terms.push((layout.instr_bit(25 + i), -F::from_u64(pow2_u64(i))));
        }
        constraints.push(Constraint::terms(one, false, terms));
    }

    // imm12_raw = bits[20..31] (unsigned 12-bit)
    {
        let mut terms = vec![(layout.imm12_raw, F::ONE)];
        for i in 0..12 {
            terms.push((layout.instr_bit(20 + i), -F::from_u64(pow2_u64(i))));
        }
        constraints.push(Constraint::terms(one, false, terms));
    }

    // imm_i (u32 representation): imm12_raw + sign*(2^32 - 2^12)
    {
        let sign = layout.instr_bit(31);
        let bias = (1u64 << 32) - (1u64 << 12);
        constraints.push(Constraint::terms(
            one,
            false,
            vec![
                (layout.imm_i, F::ONE),
                (layout.imm12_raw, -F::ONE),
                (sign, -F::from_u64(bias)),
            ],
        ));
    }

    // imm_s (u32 representation):
    //   low5 = bits[7..11]  (already packed as rd_field)
    //   high7 = bits[25..31] at positions [5..11]
    //   imm_s = low5 + Σ 2^(5+i)*bits[25+i] + sign*(2^32 - 2^12)
    {
        let sign = layout.instr_bit(31);
        let bias = (1u64 << 32) - (1u64 << 12);
        let mut terms = vec![(layout.imm_s, F::ONE), (layout.rd_field, -F::ONE), (sign, -F::from_u64(bias))];
        for i in 0..7 {
            terms.push((layout.instr_bit(25 + i), -F::from_u64(pow2_u64(5 + i))));
        }
        constraints.push(Constraint::terms(one, false, terms));
    }

    // imm_u (already << 12): Σ_{i=12..31} 2^i * bit[i]
    {
        let mut terms = vec![(layout.imm_u, F::ONE)];
        for i in 12..32 {
            terms.push((layout.instr_bit(i), -F::from_u64(pow2_u64(i))));
        }
        constraints.push(Constraint::terms(one, false, terms));
    }

    // Flags: boolean + one-hot.
    let flags = [
        layout.is_add,
        layout.is_addi,
        layout.is_lw,
        layout.is_sw,
        layout.is_lui,
        layout.is_auipc,
        layout.is_halt,
    ];
    for &f in &flags {
        constraints.push(Constraint::terms(f, false, vec![(f, F::ONE), (one, -F::ONE)]));
    }
    {
        let mut terms = Vec::with_capacity(flags.len() + 1);
        for &f in &flags {
            terms.push((f, F::ONE));
        }
        terms.push((one, -F::ONE));
        constraints.push(Constraint::terms(one, false, terms));
    }

    // Decode constraints for supported subset.
    constraints.push(Constraint::eq_const(layout.is_add, one, layout.opcode, 0x33));
    constraints.push(Constraint::zero(layout.is_add, layout.funct3));
    constraints.push(Constraint::zero(layout.is_add, layout.funct7));

    constraints.push(Constraint::eq_const(layout.is_addi, one, layout.opcode, 0x13));
    constraints.push(Constraint::zero(layout.is_addi, layout.funct3));

    constraints.push(Constraint::eq_const(layout.is_lw, one, layout.opcode, 0x03));
    constraints.push(Constraint::eq_const(layout.is_lw, one, layout.funct3, 0x2));

    constraints.push(Constraint::eq_const(layout.is_sw, one, layout.opcode, 0x23));
    constraints.push(Constraint::eq_const(layout.is_sw, one, layout.funct3, 0x2));

    constraints.push(Constraint::eq_const(layout.is_lui, one, layout.opcode, 0x37));
    constraints.push(Constraint::eq_const(layout.is_auipc, one, layout.opcode, 0x17));

    constraints.push(Constraint::eq_const(layout.is_halt, one, layout.opcode, 0x73));
    constraints.push(Constraint::zero(layout.is_halt, layout.imm12_raw));
    constraints.push(Constraint::zero(layout.is_halt, layout.rd_field));
    constraints.push(Constraint::zero(layout.is_halt, layout.rs1_field));
    constraints.push(Constraint::zero(layout.is_halt, layout.funct3));

    // Selector one-hots (rs1/rs2 always derived from rs1_field/rs2_field).
    for r in 0..32 {
        let b1 = layout.rs1_sel(r);
        let b2 = layout.rs2_sel(r);
        let bd = layout.rd_sel(r);
        constraints.push(Constraint::terms(b1, false, vec![(b1, F::ONE), (one, -F::ONE)]));
        constraints.push(Constraint::terms(b2, false, vec![(b2, F::ONE), (one, -F::ONE)]));
        constraints.push(Constraint::terms(bd, false, vec![(bd, F::ONE), (one, -F::ONE)]));
    }
    for start in [layout.rs1_sel_start, layout.rs2_sel_start, layout.rd_sel_start] {
        let mut terms = Vec::with_capacity(33);
        for r in 0..32 {
            terms.push((start + r, F::ONE));
        }
        terms.push((one, -F::ONE));
        constraints.push(Constraint::terms(one, false, terms));
    }

    // rs1_field == Σ r * rs1_sel[r]
    {
        let mut terms = vec![(layout.rs1_field, F::ONE)];
        for r in 0..32 {
            terms.push((layout.rs1_sel(r), -F::from_u64(r as u64)));
        }
        constraints.push(Constraint::terms(one, false, terms));
    }
    // rs2_field == Σ r * rs2_sel[r]
    {
        let mut terms = vec![(layout.rs2_field, F::ONE)];
        for r in 0..32 {
            terms.push((layout.rs2_sel(r), -F::from_u64(r as u64)));
        }
        constraints.push(Constraint::terms(one, false, terms));
    }

    // rd_field == Σ r * rd_sel[r] when instruction writes rd (ADD/ADDI/LW/LUI/AUIPC).
    let writes_rd_flags = [layout.is_add, layout.is_addi, layout.is_lw, layout.is_lui, layout.is_auipc];
    {
        let mut terms = vec![(layout.rd_field, F::ONE)];
        for r in 0..32 {
            terms.push((layout.rd_sel(r), -F::from_u64(r as u64)));
        }
        constraints.push(Constraint::terms_or(&writes_rd_flags, false, terms));
    }

    // If NOT writing rd (SW/HALT), force rd_sel[1..] = 0 (so rd_sel == x0).
    for r in 1..32 {
        constraints.push(Constraint::terms_or(
            &writes_rd_flags,
            true, // (1 - writes_rd)
            vec![(layout.rd_sel(r), F::ONE)],
        ));
    }

    // Bind rs1_val / rs2_val to regs_in via one-hot selectors.
    for r in 0..32 {
        constraints.push(Constraint::terms(
            layout.rs1_sel(r),
            false,
            vec![(layout.rs1_val, F::ONE), (layout.reg_in(r), -F::ONE)],
        ));
        constraints.push(Constraint::terms(
            layout.rs2_sel(r),
            false,
            vec![(layout.rs2_val, F::ONE), (layout.reg_in(r), -F::ONE)],
        ));
    }

    // Register update pattern for r=1..31:
    //  - if rd_sel[r]=1 then reg_out[r] = rd_write_val
    //  - else reg_out[r] = reg_in[r]
    for r in 1..32 {
        constraints.push(Constraint::terms(
            layout.rd_sel(r),
            false,
            vec![(layout.reg_out(r), F::ONE), (layout.rd_write_val, -F::ONE)],
        ));
        constraints.push(Constraint::terms(
            layout.rd_sel(r),
            true,
            vec![(layout.reg_out(r), F::ONE), (layout.reg_in(r), -F::ONE)],
        ));
    }

    // RAM effective address: eff_addr = rs1_val + imm_* (field addition; MVP assumes no overflow).
    constraints.push(Constraint::terms(
        layout.is_lw,
        false,
        vec![
            (layout.eff_addr, F::ONE),
            (layout.rs1_val, -F::ONE),
            (layout.imm_i, -F::ONE),
        ],
    ));
    constraints.push(Constraint::terms(
        layout.is_sw,
        false,
        vec![
            (layout.eff_addr, F::ONE),
            (layout.rs1_val, -F::ONE),
            (layout.imm_s, -F::ONE),
        ],
    ));

    // Shout selector for ADD table: add_has_lookup = is_add + is_addi.
    constraints.push(Constraint::terms(
        one,
        false,
        vec![
            (layout.add_has_lookup, F::ONE),
            (layout.is_add, -F::ONE),
            (layout.is_addi, -F::ONE),
        ],
    ));

    // LUI/AUIPC writeback (when not covered by Shout/RAM bindings).
    constraints.push(Constraint::terms(
        layout.is_lui,
        false,
        vec![(layout.rd_write_val, F::ONE), (layout.imm_u, -F::ONE)],
    ));
    constraints.push(Constraint::terms(
        layout.is_auipc,
        false,
        vec![
            (layout.rd_write_val, F::ONE),
            (layout.pc_in, -F::ONE),
            (layout.imm_u, -F::ONE),
        ],
    ));

    // --- Shout key correctness (ADD table bus addr bits interleaving) ---
    let shout_cols = &layout.bus.shout_cols[layout.add_shout_idx];
    let ell_addr = 2 * RV32_XLEN;
    if shout_cols.addr_bits.end - shout_cols.addr_bits.start != ell_addr {
        return Err(format!(
            "ADD shout bus layout mismatch: expected ell_addr={ell_addr}, got {}",
            shout_cols.addr_bits.end - shout_cols.addr_bits.start
        ));
    }

    let mut even_terms = vec![(layout.rs1_val, F::ONE)];
    for i in 0..RV32_XLEN {
        let bit_col_id = shout_cols.addr_bits.start + 2 * i;
        let bit = layout.bus.bus_cell(bit_col_id, 0);
        even_terms.push((bit, -F::from_u64(pow2_u64(i))));
    }
    constraints.push(Constraint::terms(layout.is_add, false, even_terms.clone()));
    constraints.push(Constraint::terms(layout.is_addi, false, even_terms));

    let mut odd_terms_add = vec![(layout.rs2_val, F::ONE)];
    for i in 0..RV32_XLEN {
        let bit_col_id = shout_cols.addr_bits.start + 2 * i + 1;
        let bit = layout.bus.bus_cell(bit_col_id, 0);
        odd_terms_add.push((bit, -F::from_u64(pow2_u64(i))));
    }
    constraints.push(Constraint::terms(layout.is_add, false, odd_terms_add));

    let mut odd_terms_addi = vec![(layout.imm_i, F::ONE)];
    for i in 0..RV32_XLEN {
        let bit_col_id = shout_cols.addr_bits.start + 2 * i + 1;
        let bit = layout.bus.bus_cell(bit_col_id, 0);
        odd_terms_addi.push((bit, -F::from_u64(pow2_u64(i))));
    }
    constraints.push(Constraint::terms(layout.is_addi, false, odd_terms_addi));

    // --- Alignment constraints (MVP) ---
    // ROM fetch is always 32-bit, so enforce pc_in % 4 == 0 via PROG read address bits.
    {
        let prog_id = PROG_ID.0;
        let prog_layout = mem_layouts
            .get(&prog_id)
            .ok_or_else(|| format!("mem_layouts missing PROG_ID={prog_id}"))?;
        if prog_layout.n_side != 2 {
            return Err("RV32 B1: PROG_ID must use n_side=2 (bit addressing)".into());
        }
        if prog_layout.d < 2 {
            return Err("RV32 B1: PROG_ID layout.d must be >= 2 for alignment checks".into());
        }
        let prog = &layout.bus.twist_cols[layout.prog_twist_idx];
        let bit0 = layout.bus.bus_cell(prog.ra_bits.start + 0, 0);
        let bit1 = layout.bus.bus_cell(prog.ra_bits.start + 1, 0);
        constraints.push(Constraint::zero(one, bit0));
        constraints.push(Constraint::zero(one, bit1));
    }

    // Enforce word alignment for LW/SW via RAM bus addr bits.
    {
        let ram_id = RAM_ID.0;
        let ram_layout = mem_layouts
            .get(&ram_id)
            .ok_or_else(|| format!("mem_layouts missing RAM_ID={ram_id}"))?;
        if ram_layout.n_side != 2 {
            return Err("RV32 B1: RAM_ID must use n_side=2 (bit addressing)".into());
        }
        if ram_layout.d < 2 {
            return Err("RV32 B1: RAM_ID layout.d must be >= 2 for alignment checks".into());
        }

        let ram = &layout.bus.twist_cols[layout.ram_twist_idx];
        let ra0 = layout.bus.bus_cell(ram.ra_bits.start + 0, 0);
        let ra1 = layout.bus.bus_cell(ram.ra_bits.start + 1, 0);
        let wa0 = layout.bus.bus_cell(ram.wa_bits.start + 0, 0);
        let wa1 = layout.bus.bus_cell(ram.wa_bits.start + 1, 0);
        constraints.push(Constraint::zero(layout.is_lw, ra0));
        constraints.push(Constraint::zero(layout.is_lw, ra1));
        constraints.push(Constraint::zero(layout.is_sw, wa0));
        constraints.push(Constraint::zero(layout.is_sw, wa1));
    }

    Ok(constraints)
}

/// Build the RV32 B1 step CCS and its witness layout.
///
/// Requirements:
/// - `mem_layouts` must include `RAM_ID` and `PROG_ID`.
/// - `mem_layouts[PROG_ID]` is byte-addressed (`n_side=2`, `ell=1`).
pub fn build_rv32_b1_step_ccs(mem_layouts: &HashMap<u32, PlainMemLayout>) -> Result<(CcsStructure<F>, Rv32B1Layout), String> {
    let ram_id = RAM_ID.0;
    let prog_id = PROG_ID.0;
    if !mem_layouts.contains_key(&ram_id) {
        return Err(format!("RV32 B1: mem_layouts missing RAM_ID={ram_id}"));
    }
    if !mem_layouts.contains_key(&prog_id) {
        return Err(format!("RV32 B1: mem_layouts missing PROG_ID={prog_id}"));
    }

    let (mem_ids, twist_ell_addrs) = derive_mem_ids_and_ell_addrs(mem_layouts)?;
    if mem_ids.len() != twist_ell_addrs.len() {
        return Err("RV32 B1: internal error (twist ell addrs mismatch)".into());
    }
    let shout_ell_addrs = [2 * RV32_XLEN];
    let bus_cols_total: usize = (shout_ell_addrs[0] + 2)
        + twist_ell_addrs
            .iter()
            .map(|&ell_addr| 2 * ell_addr + 5)
            .sum::<usize>();

    // First pass: minimal width.
    let layout = build_layout_with_m(/*m=*/ 1 + 512 + bus_cols_total, mem_layouts)?;
    // The `build_layout_with_m` call above intentionally over-allocates (512 CPU padding)
    // to avoid a fragile multi-pass layout rebuild loop while we iterate on constraints.
    // We still ensure slack is sufficient below.

    let constraints = semantic_constraints(&layout, mem_layouts)?;
    let needed_injection = injection_constraint_count(&shout_ell_addrs, &twist_ell_addrs);
    if constraints.len() + needed_injection > layout.m {
        return Err(format!(
            "RV32 B1: insufficient padding: constraints={} + injected={} > m={}",
            constraints.len(),
            needed_injection,
            layout.m
        ));
    }

    let ccs = build_identity_first_r1cs_ccs(&constraints, layout.m, layout.const_one)?;
    Ok((ccs, layout))
}

/// Shared CPU-bus bindings for the RV32 B1 step circuit.
///
/// This config:
/// - binds `PROG_ID` reads to `pc_in` / `instr_word`, forces no ROM writes,
/// - binds `RAM_ID` reads/writes to `eff_addr` / `rd_write_val` / `rs2_val`,
/// - binds the `ADD` Shout instance (table id 3) to `lookup_key` / `rd_write_val`.
pub fn rv32_b1_shared_cpu_bus_config(
    layout: &Rv32B1Layout,
    mem_layouts: HashMap<u32, PlainMemLayout>,
    initial_mem: HashMap<(u32, u64), F>,
) -> SharedCpuBusConfig<F> {
    let mut shout_cpu = HashMap::new();
    shout_cpu.insert(
        ADD_TABLE_ID,
        ShoutCpuBinding {
            has_lookup: layout.add_has_lookup,
            addr: layout.lookup_key,
            val: layout.rd_write_val,
        },
    );

    let mut twist_cpu = HashMap::new();
    twist_cpu.insert(
        RAM_ID.0,
        TwistCpuBinding {
            has_read: layout.is_lw,
            has_write: layout.is_sw,
            read_addr: layout.eff_addr,
            write_addr: layout.eff_addr,
            rv: layout.rd_write_val,
            wv: layout.rs2_val,
            inc: None,
        },
    );
    twist_cpu.insert(
        PROG_ID.0,
        TwistCpuBinding {
            has_read: layout.const_one,
            has_write: layout.reg_in(0),
            read_addr: layout.pc_in,
            write_addr: layout.reg_in(0),
            rv: layout.instr_word,
            wv: layout.reg_in(0),
            inc: None,
        },
    );

    SharedCpuBusConfig {
        mem_layouts,
        initial_mem,
        const_one_col: layout.const_one,
        shout_cpu,
        twist_cpu,
    }
}

/// Build a CPU witness vector `z` (CPU region only; the bus tail is written by `R1csCpu`).
pub fn rv32_b1_step_to_witness(layout: Rv32B1Layout) -> Box<dyn Fn(&StepTrace<u64, u64>) -> Vec<F> + Send + Sync> {
    Box::new(move |step: &StepTrace<u64, u64>| {
        let mut z = vec![F::ZERO; layout.bus.bus_base];

        z[layout.const_one] = F::ONE;
        z[layout.pc_in] = F::from_u64(step.pc_before);
        z[layout.pc_out] = F::from_u64(step.pc_after);

        // Registers.
        for r in 0..32 {
            z[layout.reg_in(r)] = F::from_u64(step.regs_before[r]);
            z[layout.reg_out(r)] = F::from_u64(step.regs_after[r]);
        }

        // Instruction word: read from PROG_ID Twist event (commitment-bound source).
        let mut instr_word_u32: u32 = step.opcode;
        for ev in &step.twist_events {
            if ev.twist_id == PROG_ID && ev.kind == TwistOpKind::Read {
                instr_word_u32 = ev.value as u32;
                break;
            }
        }
        z[layout.instr_word] = F::from_u64(instr_word_u32 as u64);

        // Bits.
        for i in 0..32 {
            z[layout.instr_bit(i)] = if ((instr_word_u32 >> i) & 1) == 1 { F::ONE } else { F::ZERO };
        }

        // Packed fields.
        z[layout.opcode] = F::from_u64((instr_word_u32 & 0x7f) as u64);
        z[layout.rd_field] = F::from_u64(((instr_word_u32 >> 7) & 0x1f) as u64);
        z[layout.funct3] = F::from_u64(((instr_word_u32 >> 12) & 0x7) as u64);
        z[layout.rs1_field] = F::from_u64(((instr_word_u32 >> 15) & 0x1f) as u64);
        z[layout.rs2_field] = F::from_u64(((instr_word_u32 >> 20) & 0x1f) as u64);
        z[layout.funct7] = F::from_u64(((instr_word_u32 >> 25) & 0x7f) as u64);

        // Immediates.
        let imm12_raw = (instr_word_u32 >> 20) & 0x0fff;
        z[layout.imm12_raw] = F::from_u64(imm12_raw as u64);
        // u32 sign-extended immediate value (matches VM `sign_extend_imm` for xlen=32).
        let imm_i_signed = ((imm12_raw as i32) << 20) >> 20;
        z[layout.imm_i] = F::from_u64((imm_i_signed as u32) as u64);

        // S-type immediate (u32 sign-extended).
        let imm_s_raw = ((instr_word_u32 >> 7) & 0x1f) | (((instr_word_u32 >> 25) & 0x7f) << 5);
        let imm_s_signed = ((imm_s_raw as i32) << 20) >> 20;
        z[layout.imm_s] = F::from_u64((imm_s_signed as u32) as u64);

        // U-type immediate already shifted by 12.
        z[layout.imm_u] = F::from_u64((instr_word_u32 & 0xffff_f000) as u64);

        // Decode instruction for flags.
        let decoded = decode_instruction(instr_word_u32).expect("RV32 B1: decode failed");
        let mut is_add = false;
        let mut is_addi = false;
        let mut is_lw = false;
        let mut is_sw = false;
        let mut is_lui = false;
        let mut is_auipc = false;
        let mut is_halt = false;

        match decoded {
            RiscvInstruction::RAlu { op, .. } => {
                is_add = op == RiscvOpcode::Add;
            }
            RiscvInstruction::IAlu { op, .. } => {
                is_addi = op == RiscvOpcode::Add;
            }
            RiscvInstruction::Load { op, .. } => {
                is_lw = op == RiscvMemOp::Lw;
            }
            RiscvInstruction::Store { op, .. } => {
                is_sw = op == RiscvMemOp::Sw;
            }
            RiscvInstruction::Lui { .. } => {
                is_lui = true;
            }
            RiscvInstruction::Auipc { .. } => {
                is_auipc = true;
            }
            RiscvInstruction::Halt => {
                is_halt = true;
            }
            _ => {}
        }

        z[layout.is_add] = if is_add { F::ONE } else { F::ZERO };
        z[layout.is_addi] = if is_addi { F::ONE } else { F::ZERO };
        z[layout.is_lw] = if is_lw { F::ONE } else { F::ZERO };
        z[layout.is_sw] = if is_sw { F::ONE } else { F::ZERO };
        z[layout.is_lui] = if is_lui { F::ONE } else { F::ZERO };
        z[layout.is_auipc] = if is_auipc { F::ONE } else { F::ZERO };
        z[layout.is_halt] = if is_halt { F::ONE } else { F::ZERO };

        // Operand selectors from raw fields.
        let rs1_idx = ((instr_word_u32 >> 15) & 0x1f) as usize;
        let rs2_idx = ((instr_word_u32 >> 20) & 0x1f) as usize;
        let rd_idx = ((instr_word_u32 >> 7) & 0x1f) as usize;

        for r in 0..32 {
            z[layout.rs1_sel(r)] = if r == rs1_idx { F::ONE } else { F::ZERO };
            z[layout.rs2_sel(r)] = if r == rs2_idx { F::ONE } else { F::ZERO };
            z[layout.rd_sel(r)] = F::ZERO;
        }
        // rd_sel is meaningful only for writing instructions; otherwise forced to x0 by constraints.
        if is_add || is_addi || is_lw || is_lui || is_auipc {
            z[layout.rd_sel(rd_idx)] = F::ONE;
        } else {
            z[layout.rd_sel(0)] = F::ONE;
        }

        // Selected operand values.
        z[layout.rs1_val] = F::from_u64(step.regs_before[rs1_idx]);
        z[layout.rs2_val] = F::from_u64(step.regs_before[rs2_idx]);

        // Shared-bus bound values: lookup_key / rd_write_val / eff_addr.
        z[layout.add_has_lookup] = if is_add || is_addi { F::ONE } else { F::ZERO };

        // Default zeros.
        z[layout.lookup_key] = F::ZERO;
        z[layout.eff_addr] = F::ZERO;
        z[layout.rd_write_val] = F::ZERO;

        // Shout (ADD) event if present.
        for ev in &step.shout_events {
            if ev.shout_id.0 == ADD_TABLE_ID {
                z[layout.lookup_key] = F::from_u64(ev.key);
                z[layout.rd_write_val] = F::from_u64(ev.value);
            }
        }

        // RAM read/write events if present.
        for ev in &step.twist_events {
            if ev.twist_id == RAM_ID {
                match ev.kind {
                    TwistOpKind::Read => {
                        z[layout.eff_addr] = F::from_u64(ev.addr);
                        z[layout.rd_write_val] = F::from_u64(ev.value);
                    }
                    TwistOpKind::Write => {
                        z[layout.eff_addr] = F::from_u64(ev.addr);
                    }
                }
            }
        }

        // For LUI/AUIPC, compute expected rd_write_val (not bus-bound).
        if is_lui {
            z[layout.rd_write_val] = z[layout.imm_u];
        }
        if is_auipc {
            let pc = step.pc_before as u64;
            let imm = (instr_word_u32 & 0xffff_f000) as u64;
            z[layout.rd_write_val] = F::from_u64(pc.wrapping_add(imm) as u32 as u64);
        }

        z
    })
}
