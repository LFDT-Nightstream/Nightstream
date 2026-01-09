//! RV32 "B1" RISC-V step CCS (shared-bus compatible).
//!
//! This module provides a **sound, shared-bus-compatible** step circuit for a small,
//! MVP RV32 subset. The circuit is expressed as an identity-first, square R1CS→CCS:
//! - `M0 = I_n` (required by the Ajtai/NC pipeline)
//! - `A(z) * B(z) = C(z)` with `C = 0` for almost all rows
//!
//! The witness `z` includes a **reserved bus tail** whose column schema matches
//! `cpu::bus_layout::BusLayout`. The bus tail itself is written from `StepTrace`
//! events by `R1csCpu` (shared-bus mode), and is verified by the Twist/Shout sidecars.
//!
//! ## Execution model (Phase 1)
//! - Each lane `j` in a chunk is one architectural step.
//! - `is_active[j] ∈ {0,1}` gates padding; inactive lanes keep `(pc, regs)` constant and perform
//!   no bus activity (enforced via shared-bus padding constraints).
//! - Intra-chunk continuity is enforced: `pc_in[j+1] = pc_out[j]` and `regs_in[j+1] = regs_out[j]`.
//! - The chunk exposes public boundary state (`pc0/regs0` at lane 0 and `pc_final/regs_final` at the
//!   last lane). Multi-chunk executions must chain these boundary values across chunks at a higher layer.
//!
//! The CCS here constrains the **CPU glue**:
//! - ROM fetch binding (`PROG_ID`) via shared-bus bindings
//! - instruction decode from a committed 32-bit instruction word
//! - register-file update pattern
//! - RAM load/store binding (`RAM_ID`) via shared-bus bindings
//! - Shout key wiring for `ADD` lookups (table id 3)
//!
//! Supported RV32IMA subset (RV32, word-only memory, no compressed):
//! - ALU (R-type): `ADD`, `SUB`, `SLL`, `SLT`, `SLTU`, `XOR`, `SRL`, `SRA`, `OR`, `AND`
//! - M (R-type): `MUL`, `MULH`, `MULHU`, `MULHSU`, `DIV`, `DIVU`, `REM`, `REMU`
//! - ALU (I-type): `ADDI`, `SLTI`, `SLTIU`, `XORI`, `ORI`, `ANDI`, `SLLI`, `SRLI`, `SRAI`
//! - Memory (word): `LW`, `SW`
//! - Atomics (word): `AMOADD.W`, `AMOAND.W`, `AMOOR.W`, `AMOXOR.W`, `AMOSWAP.W`
//! - Branch: `BEQ`, `BNE`, `BLT`, `BGE`, `BLTU`, `BGEU`
//! - Jump: `JAL`, `JALR`
//! - U-type: `LUI`, `AUIPC`
//! - `ECALL(imm=0)` (treated as `Halt`)

use std::collections::HashMap;

use neo_ccs::relations::CcsStructure;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

use crate::plain::PlainMemLayout;
use crate::riscv::lookups::{RAM_ID, PROG_ID};

mod bus_bindings;
mod config;
mod constants;
mod constraint_builder;
mod layout;
mod witness;

pub use bus_bindings::rv32_b1_shared_cpu_bus_config;
pub use layout::Rv32B1Layout;
pub use witness::{rv32_b1_chunk_to_witness, rv32_b1_chunk_to_witness_checked};

/// Verifier-side step-linking pairs for chaining multi-chunk executions.
///
/// For each adjacent pair of shard steps (chunks) `i` and `i+1`, require:
/// - `pc_final[i] == pc0[i+1]`
/// - `regs_final[i][r] == regs0[i+1][r]` for `r ∈ [0..32)`
/// - `halted_out[i] == halted_in[i+1]`
///
/// This is the minimal glue needed to make `chunk_size` a semantic no-op: the CPU state must form
/// one contiguous execution across chunks.
pub fn rv32_b1_step_linking_pairs(layout: &Rv32B1Layout) -> Vec<(usize, usize)> {
    let mut pairs = Vec::with_capacity(34);
    pairs.push((layout.pc_final, layout.pc0));
    for r in 0..32 {
        pairs.push((layout.regs_final_start + r, layout.regs0_start + r));
    }
    pairs.push((layout.halted_out, layout.halted_in));
    pairs
}

/// Minimal Shout table set intended for small RV32 programs that only need:
/// - `ADD` (address/ALU wiring), and
/// - `EQ`/`NEQ` (BEQ/BNE).
pub const RV32_B1_SHOUT_PROFILE_MIN3: &[u32] = &[ADD_TABLE_ID, EQ_TABLE_ID, NEQ_TABLE_ID];

/// Full RV32I Shout table set (ids 0..=11).
pub const RV32_B1_SHOUT_PROFILE_FULL12: &[u32] = &[
    AND_TABLE_ID,
    XOR_TABLE_ID,
    OR_TABLE_ID,
    ADD_TABLE_ID,
    SUB_TABLE_ID,
    SLT_TABLE_ID,
    SLTU_TABLE_ID,
    SLL_TABLE_ID,
    SRL_TABLE_ID,
    SRA_TABLE_ID,
    EQ_TABLE_ID,
    NEQ_TABLE_ID,
];

/// Full RV32IM Shout table set (ids 0..=19).
pub const RV32_B1_SHOUT_PROFILE_FULL20: &[u32] = &[
    AND_TABLE_ID,
    XOR_TABLE_ID,
    OR_TABLE_ID,
    ADD_TABLE_ID,
    SUB_TABLE_ID,
    SLT_TABLE_ID,
    SLTU_TABLE_ID,
    SLL_TABLE_ID,
    SRL_TABLE_ID,
    SRA_TABLE_ID,
    EQ_TABLE_ID,
    NEQ_TABLE_ID,
    MUL_TABLE_ID,
    MULH_TABLE_ID,
    MULHU_TABLE_ID,
    MULHSU_TABLE_ID,
    DIV_TABLE_ID,
    DIVU_TABLE_ID,
    REM_TABLE_ID,
    REMU_TABLE_ID,
];

use bus_bindings::injected_bus_constraints_len;
use config::{derive_mem_ids_and_ell_addrs, derive_shout_ids_and_ell_addrs};
use constraint_builder::{build_identity_first_r1cs_ccs, Constraint};
use layout::build_layout_with_m;

use constants::{
    ADD_TABLE_ID, AND_TABLE_ID, DIV_TABLE_ID, DIVU_TABLE_ID, EQ_TABLE_ID, MULH_TABLE_ID, MULHSU_TABLE_ID,
    MULHU_TABLE_ID, MUL_TABLE_ID, NEQ_TABLE_ID, OR_TABLE_ID, REMU_TABLE_ID, REM_TABLE_ID, RV32_XLEN, SLL_TABLE_ID,
    SLT_TABLE_ID, SLTU_TABLE_ID, SRA_TABLE_ID, SRL_TABLE_ID, SUB_TABLE_ID, XOR_TABLE_ID,
};

fn pow2_u64(i: usize) -> u64 {
    1u64 << i
}

fn semantic_constraints(layout: &Rv32B1Layout, mem_layouts: &HashMap<u32, PlainMemLayout>) -> Result<Vec<Constraint<F>>, String> {
    let one = layout.const_one;

    let mut constraints = Vec::<Constraint<F>>::new();

    let shout_cols =
        |table_id: u32| layout.table_ids.binary_search(&table_id).ok().map(|idx| &layout.bus.shout_cols[idx].lanes[0]);

    // The ADD table is required because this circuit uses it for address/ALU wiring (LW/SW/AUIPC/JALR).
    let add_shout_idx = layout.shout_idx(ADD_TABLE_ID)?;
    let add_cols = &layout.bus.shout_cols[add_shout_idx].lanes[0];

    let and_cols = shout_cols(AND_TABLE_ID);
    let xor_cols = shout_cols(XOR_TABLE_ID);
    let or_cols = shout_cols(OR_TABLE_ID);
    let sub_cols = shout_cols(SUB_TABLE_ID);
    let slt_cols = shout_cols(SLT_TABLE_ID);
    let sltu_cols = shout_cols(SLTU_TABLE_ID);
    let sll_cols = shout_cols(SLL_TABLE_ID);
    let srl_cols = shout_cols(SRL_TABLE_ID);
    let sra_cols = shout_cols(SRA_TABLE_ID);
    let eq_cols = shout_cols(EQ_TABLE_ID);
    let neq_cols = shout_cols(NEQ_TABLE_ID);
    let mul_cols = shout_cols(MUL_TABLE_ID);
    let mulh_cols = shout_cols(MULH_TABLE_ID);
    let mulhu_cols = shout_cols(MULHU_TABLE_ID);
    let mulhsu_cols = shout_cols(MULHSU_TABLE_ID);
    let div_cols = shout_cols(DIV_TABLE_ID);
    let divu_cols = shout_cols(DIVU_TABLE_ID);
    let rem_cols = shout_cols(REM_TABLE_ID);
    let remu_cols = shout_cols(REMU_TABLE_ID);

    let ell_addr = 2 * RV32_XLEN;
    for (name, cols_opt) in [
        ("ADD", Some(add_cols)),
        ("AND", and_cols),
        ("XOR", xor_cols),
        ("OR", or_cols),
        ("SUB", sub_cols),
        ("SLT", slt_cols),
        ("SLTU", sltu_cols),
        ("SLL", sll_cols),
        ("SRL", srl_cols),
        ("SRA", sra_cols),
        ("EQ", eq_cols),
        ("NEQ", neq_cols),
        ("MUL", mul_cols),
        ("MULH", mulh_cols),
        ("MULHU", mulhu_cols),
        ("MULHSU", mulhsu_cols),
        ("DIV", div_cols),
        ("DIVU", divu_cols),
        ("REM", rem_cols),
        ("REMU", remu_cols),
    ] {
        if let Some(cols) = cols_opt {
            if cols.addr_bits.end - cols.addr_bits.start != ell_addr {
                return Err(format!(
                    "{name} shout bus layout mismatch: expected ell_addr={ell_addr}, got {}",
                    cols.addr_bits.end - cols.addr_bits.start
                ));
            }
        }
    }

    // If a Shout table isn't included, forbid the corresponding instruction variants.
    if and_cols.is_none() {
        for j in 0..layout.chunk_size {
            constraints.push(Constraint::zero(one, layout.is_and(j)));
            constraints.push(Constraint::zero(one, layout.is_andi(j)));
        }
    }
    if or_cols.is_none() {
        for j in 0..layout.chunk_size {
            constraints.push(Constraint::zero(one, layout.is_or(j)));
            constraints.push(Constraint::zero(one, layout.is_ori(j)));
        }
    }
    if xor_cols.is_none() {
        for j in 0..layout.chunk_size {
            constraints.push(Constraint::zero(one, layout.is_xor(j)));
            constraints.push(Constraint::zero(one, layout.is_xori(j)));
        }
    }
    if sub_cols.is_none() {
        for j in 0..layout.chunk_size {
            constraints.push(Constraint::zero(one, layout.is_sub(j)));
        }
    }
    if sll_cols.is_none() {
        for j in 0..layout.chunk_size {
            constraints.push(Constraint::zero(one, layout.is_sll(j)));
            constraints.push(Constraint::zero(one, layout.is_slli(j)));
        }
    }
    if srl_cols.is_none() {
        for j in 0..layout.chunk_size {
            constraints.push(Constraint::zero(one, layout.is_srl(j)));
            constraints.push(Constraint::zero(one, layout.is_srli(j)));
        }
    }
    if sra_cols.is_none() {
        for j in 0..layout.chunk_size {
            constraints.push(Constraint::zero(one, layout.is_sra(j)));
            constraints.push(Constraint::zero(one, layout.is_srai(j)));
        }
    }
    if slt_cols.is_none() {
        for j in 0..layout.chunk_size {
            constraints.push(Constraint::zero(one, layout.is_slt(j)));
            constraints.push(Constraint::zero(one, layout.is_slti(j)));
            constraints.push(Constraint::zero(one, layout.is_blt(j)));
            constraints.push(Constraint::zero(one, layout.is_bge(j)));
        }
    }
    if sltu_cols.is_none() {
        for j in 0..layout.chunk_size {
            constraints.push(Constraint::zero(one, layout.is_sltu(j)));
            constraints.push(Constraint::zero(one, layout.is_sltiu(j)));
            constraints.push(Constraint::zero(one, layout.is_bltu(j)));
            constraints.push(Constraint::zero(one, layout.is_bgeu(j)));
        }
    }
    if eq_cols.is_none() {
        for j in 0..layout.chunk_size {
            constraints.push(Constraint::zero(one, layout.is_beq(j)));
        }
    }
    if neq_cols.is_none() {
        for j in 0..layout.chunk_size {
            constraints.push(Constraint::zero(one, layout.is_bne(j)));
        }
    }
    if mul_cols.is_none() {
        for j in 0..layout.chunk_size {
            constraints.push(Constraint::zero(one, layout.is_mul(j)));
        }
    }
    if mulh_cols.is_none() {
        for j in 0..layout.chunk_size {
            constraints.push(Constraint::zero(one, layout.is_mulh(j)));
        }
    }
    if mulhu_cols.is_none() {
        for j in 0..layout.chunk_size {
            constraints.push(Constraint::zero(one, layout.is_mulhu(j)));
        }
    }
    if mulhsu_cols.is_none() {
        for j in 0..layout.chunk_size {
            constraints.push(Constraint::zero(one, layout.is_mulhsu(j)));
        }
    }
    if div_cols.is_none() {
        for j in 0..layout.chunk_size {
            constraints.push(Constraint::zero(one, layout.is_div(j)));
        }
    }
    if divu_cols.is_none() {
        for j in 0..layout.chunk_size {
            constraints.push(Constraint::zero(one, layout.is_divu(j)));
        }
    }
    if rem_cols.is_none() {
        for j in 0..layout.chunk_size {
            constraints.push(Constraint::zero(one, layout.is_rem(j)));
        }
    }
    if remu_cols.is_none() {
        for j in 0..layout.chunk_size {
            constraints.push(Constraint::zero(one, layout.is_remu(j)));
        }
    }

    // Alignment constraints require bit-addressed memories (n_side=2).
    let prog_id = PROG_ID.0;
    let prog_layout = mem_layouts
        .get(&prog_id)
        .ok_or_else(|| format!("mem_layouts missing PROG_ID={prog_id}"))?;
    if prog_layout.n_side != 2 || prog_layout.d < 2 {
        return Err("RV32 B1: PROG_ID must use n_side=2 and d>=2 (bit addressing)".into());
    }
    let ram_id = RAM_ID.0;
    let ram_layout = mem_layouts
        .get(&ram_id)
        .ok_or_else(|| format!("mem_layouts missing RAM_ID={ram_id}"))?;
    if ram_layout.n_side != 2 || ram_layout.d < 2 {
        return Err("RV32 B1: RAM_ID must use n_side=2 and d>=2 (bit addressing)".into());
    }

    let prog = &layout.bus.twist_cols[layout.prog_twist_idx].lanes[0];
    let ram = &layout.bus.twist_cols[layout.ram_twist_idx].lanes[0];

    let pack_interleaved_operand = |addr_bits_start: usize, j: usize, parity: usize, value_col: usize| -> Vec<(usize, F)> {
        debug_assert!(parity == 0 || parity == 1, "parity must be 0 (even) or 1 (odd)");
        let mut terms = vec![(value_col, F::ONE)];
        for i in 0..RV32_XLEN {
            let bit_col_id = addr_bits_start + 2 * i + parity;
            let bit = layout.bus.bus_cell(bit_col_id, j);
            terms.push((bit, -F::from_u64(pow2_u64(i))));
        }
        terms
    };

    // --- Public I/O binding (initial + final architectural state) ---
    // Initial state binds to lane 0.
    let j0 = 0usize;
    constraints.push(Constraint::terms(
        one,
        false,
        vec![(layout.pc_in(j0), F::ONE), (layout.pc0, -F::ONE)],
    ));
    for r in 0..32 {
        constraints.push(Constraint::terms(
            one,
            false,
            vec![
                (layout.reg_in(r, j0), F::ONE),
                (layout.regs0_start + r, -F::ONE),
            ],
        ));
    }

    // Final state binds to the last lane.
    let j_last = layout.chunk_size - 1;
    constraints.push(Constraint::terms(
        one,
        false,
        vec![
            (layout.pc_out(j_last), F::ONE),
            (layout.pc_final, -F::ONE),
        ],
    ));
    for r in 0..32 {
        constraints.push(Constraint::terms(
            one,
            false,
            vec![
                (layout.reg_out(r, j_last), F::ONE),
                (layout.regs_final_start + r, -F::ONE),
            ],
        ));
    }

    // --- Cross-chunk halting / padding semantics (L1-style) ---
    // halted_in/out are booleans.
    constraints.push(Constraint::terms(
        layout.halted_in,
        false,
        vec![(layout.halted_in, F::ONE), (one, -F::ONE)],
    ));
    constraints.push(Constraint::terms(
        layout.halted_out,
        false,
        vec![(layout.halted_out, F::ONE), (one, -F::ONE)],
    ));

    // halted_in + is_active[0] = 1 (chunk starts active iff not halted).
    constraints.push(Constraint::terms(
        one,
        false,
        vec![
            (layout.halted_in, F::ONE),
            (layout.is_active(j0), F::ONE),
            (one, -F::ONE),
        ],
    ));

    // halted_out = 1 - is_active[last] + is_halt[last].
    constraints.push(Constraint::terms(
        one,
        false,
        vec![
            (layout.halted_out, F::ONE),
            (layout.is_active(j_last), F::ONE),
            (layout.is_halt(j_last), -F::ONE),
            (one, -F::ONE),
        ],
    ));

    for j in 0..layout.chunk_size {
        let is_active = layout.is_active(j);
        let pc_in = layout.pc_in(j);
        let pc_out = layout.pc_out(j);
        let instr_word = layout.instr_word(j);
        let add_a0 = layout.bus.bus_cell(add_cols.addr_bits.start + 0, j);
        let add_b0 = layout.bus.bus_cell(add_cols.addr_bits.start + 1, j);

        // x0 hardwired.
        constraints.push(Constraint::zero(one, layout.reg_in(0, j)));
        constraints.push(Constraint::zero(one, layout.reg_out(0, j)));

        // is_active is boolean.
        constraints.push(Constraint::terms(
            is_active,
            false,
            vec![(is_active, F::ONE), (one, -F::ONE)],
        ));

        // Inactive rows keep PC constant: (1 - is_active) * (pc_out - pc_in) = 0.
        constraints.push(Constraint::terms(
            is_active,
            true,
            vec![(pc_out, F::ONE), (pc_in, -F::ONE)],
        ));

        // Instruction bits:
        // - If is_active=0, force all bits to 0.
        // - If is_active=1, force bits to be boolean.
        for i in 0..32 {
            let b = layout.instr_bit(i, j);
            constraints.push(Constraint::terms(
                b,
                false,
                vec![(b, F::ONE), (is_active, -F::ONE)],
            ));
        }

        // Pack instr_word = Σ 2^i bit[i]
        {
            let mut terms = vec![(instr_word, F::ONE)];
            for i in 0..32 {
                terms.push((layout.instr_bit(i, j), -F::from_u64(pow2_u64(i))));
            }
            constraints.push(Constraint::terms(one, false, terms));
        }

        // Pack opcode/funct/fields from bits.
        {
            // opcode = bits[0..6]
            let mut terms = vec![(layout.opcode(j), F::ONE)];
            for i in 0..7 {
                terms.push((layout.instr_bit(i, j), -F::from_u64(pow2_u64(i))));
            }
            constraints.push(Constraint::terms(one, false, terms));
        }
        {
            // rd_field = bits[7..11]
            let mut terms = vec![(layout.rd_field(j), F::ONE)];
            for i in 0..5 {
                terms.push((layout.instr_bit(7 + i, j), -F::from_u64(pow2_u64(i))));
            }
            constraints.push(Constraint::terms(one, false, terms));
        }
        {
            // funct3 = bits[12..14]
            let mut terms = vec![(layout.funct3(j), F::ONE)];
            for i in 0..3 {
                terms.push((layout.instr_bit(12 + i, j), -F::from_u64(pow2_u64(i))));
            }
            constraints.push(Constraint::terms(one, false, terms));
        }
        {
            // rs1_field = bits[15..19]
            let mut terms = vec![(layout.rs1_field(j), F::ONE)];
            for i in 0..5 {
                terms.push((layout.instr_bit(15 + i, j), -F::from_u64(pow2_u64(i))));
            }
            constraints.push(Constraint::terms(one, false, terms));
        }
        {
            // rs2_field = bits[20..24]
            let mut terms = vec![(layout.rs2_field(j), F::ONE)];
            for i in 0..5 {
                terms.push((layout.instr_bit(20 + i, j), -F::from_u64(pow2_u64(i))));
            }
            constraints.push(Constraint::terms(one, false, terms));
        }
        {
            // funct7 = bits[25..31]
            let mut terms = vec![(layout.funct7(j), F::ONE)];
            for i in 0..7 {
                terms.push((layout.instr_bit(25 + i, j), -F::from_u64(pow2_u64(i))));
            }
            constraints.push(Constraint::terms(one, false, terms));
        }

        // imm12_raw = bits[20..31] (unsigned 12-bit)
        {
            let mut terms = vec![(layout.imm12_raw(j), F::ONE)];
            for i in 0..12 {
                terms.push((layout.instr_bit(20 + i, j), -F::from_u64(pow2_u64(i))));
            }
            constraints.push(Constraint::terms(one, false, terms));
        }

        // imm_i (u32 representation): imm12_raw + sign*(2^32 - 2^12)
        {
            let sign = layout.instr_bit(31, j);
            let bias = (1u64 << 32) - (1u64 << 12);
            constraints.push(Constraint::terms(
                one,
                false,
                vec![
                    (layout.imm_i(j), F::ONE),
                    (layout.imm12_raw(j), -F::ONE),
                    (sign, -F::from_u64(bias)),
                ],
            ));
        }

        // imm_s (u32 representation):
        //   low5 = bits[7..11]  (already packed as rd_field)
        //   high7 = bits[25..31] at positions [5..11]
        //   imm_s = low5 + Σ 2^(5+i)*bits[25+i] + sign*(2^32 - 2^12)
        {
            let sign = layout.instr_bit(31, j);
            let bias = (1u64 << 32) - (1u64 << 12);
            let mut terms = vec![
                (layout.imm_s(j), F::ONE),
                (layout.rd_field(j), -F::ONE),
                (sign, -F::from_u64(bias)),
            ];
            for i in 0..7 {
                terms.push((layout.instr_bit(25 + i, j), -F::from_u64(pow2_u64(5 + i))));
            }
            constraints.push(Constraint::terms(one, false, terms));
        }

        // imm_u (already << 12): Σ_{i=12..31} 2^i * bit[i]
        {
            let mut terms = vec![(layout.imm_u(j), F::ONE)];
            for i in 12..32 {
                terms.push((layout.instr_bit(i, j), -F::from_u64(pow2_u64(i))));
            }
            constraints.push(Constraint::terms(one, false, terms));
        }

        // imm_b_raw (unsigned 13-bit, bit 0 is 0):
        //   imm[12]   = bit31
        //   imm[11]   = bit7
        //   imm[10:5] = bits[30:25]
        //   imm[4:1]  = bits[11:8]
        {
            let mut terms = vec![(layout.imm_b_raw(j), F::ONE)];
            terms.push((layout.instr_bit(31, j), -F::from_u64(pow2_u64(12))));
            terms.push((layout.instr_bit(7, j), -F::from_u64(pow2_u64(11))));
            for i in 0..6 {
                terms.push((layout.instr_bit(25 + i, j), -F::from_u64(pow2_u64(5 + i))));
            }
            for i in 0..4 {
                terms.push((layout.instr_bit(8 + i, j), -F::from_u64(pow2_u64(1 + i))));
            }
            constraints.push(Constraint::terms(one, false, terms));
        }

        // imm_b (signed i32, as field element): imm_b = imm_b_raw - sign*2^13.
        {
            let sign = layout.instr_bit(31, j);
            constraints.push(Constraint::terms(
                one,
                false,
                vec![
                    (layout.imm_b(j), F::ONE),
                    (layout.imm_b_raw(j), -F::ONE),
                    (sign, F::from_u64(pow2_u64(13))),
                ],
            ));
        }

        // imm_j_raw (unsigned 21-bit, bit 0 is 0):
        //   imm[20]    = bit31
        //   imm[19:12] = bits[19:12]
        //   imm[11]    = bit20
        //   imm[10:1]  = bits[30:21]
        {
            let mut terms = vec![(layout.imm_j_raw(j), F::ONE)];
            terms.push((layout.instr_bit(31, j), -F::from_u64(pow2_u64(20))));
            for i in 12..20 {
                terms.push((layout.instr_bit(i, j), -F::from_u64(pow2_u64(i))));
            }
            terms.push((layout.instr_bit(20, j), -F::from_u64(pow2_u64(11))));
            for i in 21..31 {
                terms.push((layout.instr_bit(i, j), -F::from_u64(pow2_u64(i - 20))));
            }
            constraints.push(Constraint::terms(one, false, terms));
        }

        // imm_j (signed i32, as field element): imm_j = imm_j_raw - sign*2^21.
        {
            let sign = layout.instr_bit(31, j);
            constraints.push(Constraint::terms(
                one,
                false,
                vec![
                    (layout.imm_j(j), F::ONE),
                    (layout.imm_j_raw(j), -F::ONE),
                    (sign, F::from_u64(pow2_u64(21))),
                ],
            ));
        }

        // Flags: boolean + one-hot.
        let flags = [
            layout.is_add(j),
            layout.is_sub(j),
            layout.is_sll(j),
            layout.is_slt(j),
            layout.is_sltu(j),
            layout.is_xor(j),
            layout.is_srl(j),
            layout.is_sra(j),
            layout.is_or(j),
            layout.is_and(j),
            layout.is_mul(j),
            layout.is_mulh(j),
            layout.is_mulhu(j),
            layout.is_mulhsu(j),
            layout.is_div(j),
            layout.is_divu(j),
            layout.is_rem(j),
            layout.is_remu(j),
            layout.is_addi(j),
            layout.is_slti(j),
            layout.is_sltiu(j),
            layout.is_xori(j),
            layout.is_ori(j),
            layout.is_andi(j),
            layout.is_slli(j),
            layout.is_srli(j),
            layout.is_srai(j),
            layout.is_lw(j),
            layout.is_sw(j),
            layout.is_amoswap_w(j),
            layout.is_amoadd_w(j),
            layout.is_amoxor_w(j),
            layout.is_amoor_w(j),
            layout.is_amoand_w(j),
            layout.is_lui(j),
            layout.is_auipc(j),
            layout.is_beq(j),
            layout.is_bne(j),
            layout.is_blt(j),
            layout.is_bge(j),
            layout.is_bltu(j),
            layout.is_bgeu(j),
            layout.is_jal(j),
            layout.is_jalr(j),
            layout.is_halt(j),
        ];
        for &f in &flags {
            constraints.push(Constraint::terms(
                f,
                false,
                vec![(f, F::ONE), (is_active, -F::ONE)],
            ));
        }
        {
            let mut terms = Vec::with_capacity(flags.len() + 1);
            for &f in &flags {
                terms.push((f, F::ONE));
            }
            terms.push((is_active, -F::ONE));
            constraints.push(Constraint::terms(one, false, terms));
        }

        // Decode constraints for the supported RV32I core subset.
        constraints.push(Constraint::eq_const(layout.is_add(j), one, layout.opcode(j), 0x33));
        constraints.push(Constraint::zero(layout.is_add(j), layout.funct3(j)));
        constraints.push(Constraint::zero(layout.is_add(j), layout.funct7(j)));

        constraints.push(Constraint::eq_const(layout.is_sub(j), one, layout.opcode(j), 0x33));
        constraints.push(Constraint::zero(layout.is_sub(j), layout.funct3(j)));
        constraints.push(Constraint::eq_const(layout.is_sub(j), one, layout.funct7(j), 0x20));

        constraints.push(Constraint::eq_const(layout.is_sll(j), one, layout.opcode(j), 0x33));
        constraints.push(Constraint::eq_const(layout.is_sll(j), one, layout.funct3(j), 0x1));
        constraints.push(Constraint::zero(layout.is_sll(j), layout.funct7(j)));

        constraints.push(Constraint::eq_const(layout.is_slt(j), one, layout.opcode(j), 0x33));
        constraints.push(Constraint::eq_const(layout.is_slt(j), one, layout.funct3(j), 0x2));
        constraints.push(Constraint::zero(layout.is_slt(j), layout.funct7(j)));

        constraints.push(Constraint::eq_const(layout.is_sltu(j), one, layout.opcode(j), 0x33));
        constraints.push(Constraint::eq_const(layout.is_sltu(j), one, layout.funct3(j), 0x3));
        constraints.push(Constraint::zero(layout.is_sltu(j), layout.funct7(j)));

        constraints.push(Constraint::eq_const(layout.is_xor(j), one, layout.opcode(j), 0x33));
        constraints.push(Constraint::eq_const(layout.is_xor(j), one, layout.funct3(j), 0x4));
        constraints.push(Constraint::zero(layout.is_xor(j), layout.funct7(j)));

        constraints.push(Constraint::eq_const(layout.is_srl(j), one, layout.opcode(j), 0x33));
        constraints.push(Constraint::eq_const(layout.is_srl(j), one, layout.funct3(j), 0x5));
        constraints.push(Constraint::zero(layout.is_srl(j), layout.funct7(j)));

        constraints.push(Constraint::eq_const(layout.is_sra(j), one, layout.opcode(j), 0x33));
        constraints.push(Constraint::eq_const(layout.is_sra(j), one, layout.funct3(j), 0x5));
        constraints.push(Constraint::eq_const(layout.is_sra(j), one, layout.funct7(j), 0x20));

        constraints.push(Constraint::eq_const(layout.is_or(j), one, layout.opcode(j), 0x33));
        constraints.push(Constraint::eq_const(layout.is_or(j), one, layout.funct3(j), 0x6));
        constraints.push(Constraint::zero(layout.is_or(j), layout.funct7(j)));

        constraints.push(Constraint::eq_const(layout.is_and(j), one, layout.opcode(j), 0x33));
        constraints.push(Constraint::eq_const(layout.is_and(j), one, layout.funct3(j), 0x7));
        constraints.push(Constraint::zero(layout.is_and(j), layout.funct7(j)));

        // RV32M (funct7 = 0b0000001).
        constraints.push(Constraint::eq_const(layout.is_mul(j), one, layout.opcode(j), 0x33));
        constraints.push(Constraint::zero(layout.is_mul(j), layout.funct3(j)));
        constraints.push(Constraint::eq_const(layout.is_mul(j), one, layout.funct7(j), 0x1));

        constraints.push(Constraint::eq_const(layout.is_mulh(j), one, layout.opcode(j), 0x33));
        constraints.push(Constraint::eq_const(layout.is_mulh(j), one, layout.funct3(j), 0x1));
        constraints.push(Constraint::eq_const(layout.is_mulh(j), one, layout.funct7(j), 0x1));

        constraints.push(Constraint::eq_const(layout.is_mulhsu(j), one, layout.opcode(j), 0x33));
        constraints.push(Constraint::eq_const(layout.is_mulhsu(j), one, layout.funct3(j), 0x2));
        constraints.push(Constraint::eq_const(layout.is_mulhsu(j), one, layout.funct7(j), 0x1));

        constraints.push(Constraint::eq_const(layout.is_mulhu(j), one, layout.opcode(j), 0x33));
        constraints.push(Constraint::eq_const(layout.is_mulhu(j), one, layout.funct3(j), 0x3));
        constraints.push(Constraint::eq_const(layout.is_mulhu(j), one, layout.funct7(j), 0x1));

        constraints.push(Constraint::eq_const(layout.is_div(j), one, layout.opcode(j), 0x33));
        constraints.push(Constraint::eq_const(layout.is_div(j), one, layout.funct3(j), 0x4));
        constraints.push(Constraint::eq_const(layout.is_div(j), one, layout.funct7(j), 0x1));

        constraints.push(Constraint::eq_const(layout.is_divu(j), one, layout.opcode(j), 0x33));
        constraints.push(Constraint::eq_const(layout.is_divu(j), one, layout.funct3(j), 0x5));
        constraints.push(Constraint::eq_const(layout.is_divu(j), one, layout.funct7(j), 0x1));

        constraints.push(Constraint::eq_const(layout.is_rem(j), one, layout.opcode(j), 0x33));
        constraints.push(Constraint::eq_const(layout.is_rem(j), one, layout.funct3(j), 0x6));
        constraints.push(Constraint::eq_const(layout.is_rem(j), one, layout.funct7(j), 0x1));

        constraints.push(Constraint::eq_const(layout.is_remu(j), one, layout.opcode(j), 0x33));
        constraints.push(Constraint::eq_const(layout.is_remu(j), one, layout.funct3(j), 0x7));
        constraints.push(Constraint::eq_const(layout.is_remu(j), one, layout.funct7(j), 0x1));

        constraints.push(Constraint::eq_const(layout.is_addi(j), one, layout.opcode(j), 0x13));
        constraints.push(Constraint::zero(layout.is_addi(j), layout.funct3(j)));

        constraints.push(Constraint::eq_const(layout.is_slti(j), one, layout.opcode(j), 0x13));
        constraints.push(Constraint::eq_const(layout.is_slti(j), one, layout.funct3(j), 0x2));

        constraints.push(Constraint::eq_const(layout.is_sltiu(j), one, layout.opcode(j), 0x13));
        constraints.push(Constraint::eq_const(layout.is_sltiu(j), one, layout.funct3(j), 0x3));

        constraints.push(Constraint::eq_const(layout.is_xori(j), one, layout.opcode(j), 0x13));
        constraints.push(Constraint::eq_const(layout.is_xori(j), one, layout.funct3(j), 0x4));

        constraints.push(Constraint::eq_const(layout.is_ori(j), one, layout.opcode(j), 0x13));
        constraints.push(Constraint::eq_const(layout.is_ori(j), one, layout.funct3(j), 0x6));

        constraints.push(Constraint::eq_const(layout.is_andi(j), one, layout.opcode(j), 0x13));
        constraints.push(Constraint::eq_const(layout.is_andi(j), one, layout.funct3(j), 0x7));

        constraints.push(Constraint::eq_const(layout.is_slli(j), one, layout.opcode(j), 0x13));
        constraints.push(Constraint::eq_const(layout.is_slli(j), one, layout.funct3(j), 0x1));
        constraints.push(Constraint::zero(layout.is_slli(j), layout.funct7(j)));

        constraints.push(Constraint::eq_const(layout.is_srli(j), one, layout.opcode(j), 0x13));
        constraints.push(Constraint::eq_const(layout.is_srli(j), one, layout.funct3(j), 0x5));
        constraints.push(Constraint::zero(layout.is_srli(j), layout.funct7(j)));

        constraints.push(Constraint::eq_const(layout.is_srai(j), one, layout.opcode(j), 0x13));
        constraints.push(Constraint::eq_const(layout.is_srai(j), one, layout.funct3(j), 0x5));
        constraints.push(Constraint::eq_const(layout.is_srai(j), one, layout.funct7(j), 0x20));

        constraints.push(Constraint::eq_const(layout.is_lw(j), one, layout.opcode(j), 0x03));
        constraints.push(Constraint::eq_const(layout.is_lw(j), one, layout.funct3(j), 0x2));

        constraints.push(Constraint::eq_const(layout.is_sw(j), one, layout.opcode(j), 0x23));
        constraints.push(Constraint::eq_const(layout.is_sw(j), one, layout.funct3(j), 0x2));

        // RV32A atomics (AMO*, word only): opcode=0x2F, funct3=010, funct5 in bits [31:27].
        constraints.push(Constraint::eq_const(layout.is_amoswap_w(j), one, layout.opcode(j), 0x2f));
        constraints.push(Constraint::eq_const(layout.is_amoswap_w(j), one, layout.funct3(j), 0x2));
        constraints.push(Constraint::terms(
            layout.is_amoswap_w(j),
            false,
            vec![(layout.instr_bit(27, j), F::ONE), (one, -F::ONE)],
        ));
        constraints.push(Constraint::zero(layout.is_amoswap_w(j), layout.instr_bit(28, j)));
        constraints.push(Constraint::zero(layout.is_amoswap_w(j), layout.instr_bit(29, j)));
        constraints.push(Constraint::zero(layout.is_amoswap_w(j), layout.instr_bit(30, j)));
        constraints.push(Constraint::zero(layout.is_amoswap_w(j), layout.instr_bit(31, j)));

        constraints.push(Constraint::eq_const(layout.is_amoadd_w(j), one, layout.opcode(j), 0x2f));
        constraints.push(Constraint::eq_const(layout.is_amoadd_w(j), one, layout.funct3(j), 0x2));
        constraints.push(Constraint::zero(layout.is_amoadd_w(j), layout.instr_bit(27, j)));
        constraints.push(Constraint::zero(layout.is_amoadd_w(j), layout.instr_bit(28, j)));
        constraints.push(Constraint::zero(layout.is_amoadd_w(j), layout.instr_bit(29, j)));
        constraints.push(Constraint::zero(layout.is_amoadd_w(j), layout.instr_bit(30, j)));
        constraints.push(Constraint::zero(layout.is_amoadd_w(j), layout.instr_bit(31, j)));

        constraints.push(Constraint::eq_const(layout.is_amoxor_w(j), one, layout.opcode(j), 0x2f));
        constraints.push(Constraint::eq_const(layout.is_amoxor_w(j), one, layout.funct3(j), 0x2));
        constraints.push(Constraint::zero(layout.is_amoxor_w(j), layout.instr_bit(27, j)));
        constraints.push(Constraint::zero(layout.is_amoxor_w(j), layout.instr_bit(28, j)));
        constraints.push(Constraint::terms(
            layout.is_amoxor_w(j),
            false,
            vec![(layout.instr_bit(29, j), F::ONE), (one, -F::ONE)],
        ));
        constraints.push(Constraint::zero(layout.is_amoxor_w(j), layout.instr_bit(30, j)));
        constraints.push(Constraint::zero(layout.is_amoxor_w(j), layout.instr_bit(31, j)));

        constraints.push(Constraint::eq_const(layout.is_amoor_w(j), one, layout.opcode(j), 0x2f));
        constraints.push(Constraint::eq_const(layout.is_amoor_w(j), one, layout.funct3(j), 0x2));
        constraints.push(Constraint::zero(layout.is_amoor_w(j), layout.instr_bit(27, j)));
        constraints.push(Constraint::zero(layout.is_amoor_w(j), layout.instr_bit(28, j)));
        constraints.push(Constraint::zero(layout.is_amoor_w(j), layout.instr_bit(29, j)));
        constraints.push(Constraint::terms(
            layout.is_amoor_w(j),
            false,
            vec![(layout.instr_bit(30, j), F::ONE), (one, -F::ONE)],
        ));
        constraints.push(Constraint::zero(layout.is_amoor_w(j), layout.instr_bit(31, j)));

        constraints.push(Constraint::eq_const(layout.is_amoand_w(j), one, layout.opcode(j), 0x2f));
        constraints.push(Constraint::eq_const(layout.is_amoand_w(j), one, layout.funct3(j), 0x2));
        constraints.push(Constraint::zero(layout.is_amoand_w(j), layout.instr_bit(27, j)));
        constraints.push(Constraint::zero(layout.is_amoand_w(j), layout.instr_bit(28, j)));
        constraints.push(Constraint::terms(
            layout.is_amoand_w(j),
            false,
            vec![(layout.instr_bit(29, j), F::ONE), (one, -F::ONE)],
        ));
        constraints.push(Constraint::terms(
            layout.is_amoand_w(j),
            false,
            vec![(layout.instr_bit(30, j), F::ONE), (one, -F::ONE)],
        ));
        constraints.push(Constraint::zero(layout.is_amoand_w(j), layout.instr_bit(31, j)));

        constraints.push(Constraint::eq_const(layout.is_lui(j), one, layout.opcode(j), 0x37));
        constraints.push(Constraint::eq_const(layout.is_auipc(j), one, layout.opcode(j), 0x17));

        constraints.push(Constraint::eq_const(layout.is_beq(j), one, layout.opcode(j), 0x63));
        constraints.push(Constraint::zero(layout.is_beq(j), layout.funct3(j)));

        constraints.push(Constraint::eq_const(layout.is_bne(j), one, layout.opcode(j), 0x63));
        constraints.push(Constraint::eq_const(layout.is_bne(j), one, layout.funct3(j), 0x1));

        constraints.push(Constraint::eq_const(layout.is_blt(j), one, layout.opcode(j), 0x63));
        constraints.push(Constraint::eq_const(layout.is_blt(j), one, layout.funct3(j), 0x4));

        constraints.push(Constraint::eq_const(layout.is_bge(j), one, layout.opcode(j), 0x63));
        constraints.push(Constraint::eq_const(layout.is_bge(j), one, layout.funct3(j), 0x5));

        constraints.push(Constraint::eq_const(layout.is_bltu(j), one, layout.opcode(j), 0x63));
        constraints.push(Constraint::eq_const(layout.is_bltu(j), one, layout.funct3(j), 0x6));

        constraints.push(Constraint::eq_const(layout.is_bgeu(j), one, layout.opcode(j), 0x63));
        constraints.push(Constraint::eq_const(layout.is_bgeu(j), one, layout.funct3(j), 0x7));

        constraints.push(Constraint::eq_const(layout.is_jal(j), one, layout.opcode(j), 0x6f));

        constraints.push(Constraint::eq_const(layout.is_jalr(j), one, layout.opcode(j), 0x67));
        constraints.push(Constraint::zero(layout.is_jalr(j), layout.funct3(j)));

        constraints.push(Constraint::eq_const(layout.is_halt(j), one, layout.opcode(j), 0x73));
        constraints.push(Constraint::zero(layout.is_halt(j), layout.imm12_raw(j)));
        constraints.push(Constraint::zero(layout.is_halt(j), layout.rd_field(j)));
        constraints.push(Constraint::zero(layout.is_halt(j), layout.rs1_field(j)));
        constraints.push(Constraint::zero(layout.is_halt(j), layout.funct3(j)));

        // Selector one-hots (rs1/rs2 always derived from rs1_field/rs2_field).
        for r in 0..32 {
            let b1 = layout.rs1_sel(r, j);
            let b2 = layout.rs2_sel(r, j);
            let bd = layout.rd_sel(r, j);
            constraints.push(Constraint::terms(
                b1,
                false,
                vec![(b1, F::ONE), (is_active, -F::ONE)],
            ));
            constraints.push(Constraint::terms(
                b2,
                false,
                vec![(b2, F::ONE), (is_active, -F::ONE)],
            ));
            constraints.push(Constraint::terms(
                bd,
                false,
                vec![(bd, F::ONE), (is_active, -F::ONE)],
            ));
        }
        for sels in ["rs1", "rs2", "rd"] {
            let mut terms = Vec::with_capacity(33);
            for r in 0..32 {
                let col = match sels {
                    "rs1" => layout.rs1_sel(r, j),
                    "rs2" => layout.rs2_sel(r, j),
                    _ => layout.rd_sel(r, j),
                };
                terms.push((col, F::ONE));
            }
            terms.push((is_active, -F::ONE));
            constraints.push(Constraint::terms(one, false, terms));
        }

        // rs1_field == Σ r * rs1_sel[r]
        {
            let mut terms = vec![(layout.rs1_field(j), F::ONE)];
            for r in 0..32 {
                terms.push((layout.rs1_sel(r, j), -F::from_u64(r as u64)));
            }
            constraints.push(Constraint::terms(one, false, terms));
        }
        // rs2_field == Σ r * rs2_sel[r]
        {
            let mut terms = vec![(layout.rs2_field(j), F::ONE)];
            for r in 0..32 {
                terms.push((layout.rs2_sel(r, j), -F::from_u64(r as u64)));
            }
            constraints.push(Constraint::terms(one, false, terms));
        }

        // rd_field == Σ r * rd_sel[r] when instruction writes rd.
        let writes_rd_flags = [
            layout.is_add(j),
            layout.is_sub(j),
            layout.is_sll(j),
            layout.is_slt(j),
            layout.is_sltu(j),
            layout.is_xor(j),
            layout.is_srl(j),
            layout.is_sra(j),
            layout.is_or(j),
            layout.is_and(j),
            layout.is_mul(j),
            layout.is_mulh(j),
            layout.is_mulhu(j),
            layout.is_mulhsu(j),
            layout.is_div(j),
            layout.is_divu(j),
            layout.is_rem(j),
            layout.is_remu(j),
            layout.is_addi(j),
            layout.is_slti(j),
            layout.is_sltiu(j),
            layout.is_xori(j),
            layout.is_ori(j),
            layout.is_andi(j),
            layout.is_slli(j),
            layout.is_srli(j),
            layout.is_srai(j),
            layout.is_lw(j),
            layout.is_amoswap_w(j),
            layout.is_amoadd_w(j),
            layout.is_amoxor_w(j),
            layout.is_amoor_w(j),
            layout.is_amoand_w(j),
            layout.is_lui(j),
            layout.is_auipc(j),
            layout.is_jal(j),
            layout.is_jalr(j),
        ];
        {
            let mut terms = vec![(layout.rd_field(j), F::ONE)];
            for r in 0..32 {
                terms.push((layout.rd_sel(r, j), -F::from_u64(r as u64)));
            }
            constraints.push(Constraint::terms_or(&writes_rd_flags, false, terms));
        }

        // If NOT writing rd, force rd_sel[1..] = 0 (so rd_sel == x0).
        for r in 1..32 {
            constraints.push(Constraint::terms_or(
                &writes_rd_flags,
                true, // (1 - writes_rd)
                vec![(layout.rd_sel(r, j), F::ONE)],
            ));
        }

        // Bind rs1_val / rs2_val to regs_in via one-hot selectors.
        for r in 0..32 {
            constraints.push(Constraint::terms(
                layout.rs1_sel(r, j),
                false,
                vec![(layout.rs1_val(j), F::ONE), (layout.reg_in(r, j), -F::ONE)],
            ));
            constraints.push(Constraint::terms(
                layout.rs2_sel(r, j),
                false,
                vec![(layout.rs2_val(j), F::ONE), (layout.reg_in(r, j), -F::ONE)],
            ));
        }

        // Register update pattern for r=1..31:
        //  - if rd_sel[r]=1 then reg_out[r] = rd_write_val
        //  - else reg_out[r] = reg_in[r]
        for r in 1..32 {
            constraints.push(Constraint::terms(
                layout.rd_sel(r, j),
                false,
                vec![(layout.reg_out(r, j), F::ONE), (layout.rd_write_val(j), -F::ONE)],
            ));
            constraints.push(Constraint::terms(
                layout.rd_sel(r, j),
                true,
                vec![(layout.reg_out(r, j), F::ONE), (layout.reg_in(r, j), -F::ONE)],
            ));
        }

        // RAM effective address is computed via the ADD Shout lookup (mod 2^32 semantics).
        constraints.push(Constraint::terms(
            layout.is_lw(j),
            false,
            vec![(layout.eff_addr(j), F::ONE), (layout.alu_out(j), -F::ONE)],
        ));
        constraints.push(Constraint::terms(
            layout.is_sw(j),
            false,
            vec![(layout.eff_addr(j), F::ONE), (layout.alu_out(j), -F::ONE)],
        ));

        // Atomics use rs1 as the effective address (no immediate).
        constraints.push(Constraint::terms_or(
            &[
                layout.is_amoswap_w(j),
                layout.is_amoadd_w(j),
                layout.is_amoxor_w(j),
                layout.is_amoor_w(j),
                layout.is_amoand_w(j),
            ],
            false,
            vec![(layout.eff_addr(j), F::ONE), (layout.rs1_val(j), -F::ONE)],
        ));

        // RAM bus selectors must be derived from instruction flags to avoid bypassing Twist.
        constraints.push(Constraint::terms(
            one,
            false,
            vec![
                (layout.ram_has_read(j), F::ONE),
                (layout.is_lw(j), -F::ONE),
                (layout.is_amoswap_w(j), -F::ONE),
                (layout.is_amoadd_w(j), -F::ONE),
                (layout.is_amoxor_w(j), -F::ONE),
                (layout.is_amoor_w(j), -F::ONE),
                (layout.is_amoand_w(j), -F::ONE),
            ],
        ));
        constraints.push(Constraint::terms(
            one,
            false,
            vec![
                (layout.ram_has_write(j), F::ONE),
                (layout.is_sw(j), -F::ONE),
                (layout.is_amoswap_w(j), -F::ONE),
                (layout.is_amoadd_w(j), -F::ONE),
                (layout.is_amoxor_w(j), -F::ONE),
                (layout.is_amoor_w(j), -F::ONE),
                (layout.is_amoand_w(j), -F::ONE),
            ],
        ));

        // RAM write value (WV): SW and AMOSWAP use rs2, other AMOs use a Shout output.
        constraints.push(Constraint::terms_or(
            &[layout.is_sw(j), layout.is_amoswap_w(j)],
            false,
            vec![(layout.ram_wv(j), F::ONE), (layout.rs2_val(j), -F::ONE)],
        ));
        for &f in &[
            layout.is_amoadd_w(j),
            layout.is_amoxor_w(j),
            layout.is_amoor_w(j),
            layout.is_amoand_w(j),
        ] {
            constraints.push(Constraint::terms(
                f,
                false,
                vec![(layout.ram_wv(j), F::ONE), (layout.alu_out(j), -F::ONE)],
            ));
        }

        // Shout selectors.
        // ADD table: add_has_lookup = is_add + is_addi + is_lw + is_sw + is_amoadd_w + is_auipc + is_jalr.
        constraints.push(Constraint::terms(
            one,
            false,
            vec![
                (layout.add_has_lookup(j), F::ONE),
                (layout.is_add(j), -F::ONE),
                (layout.is_addi(j), -F::ONE),
                (layout.is_lw(j), -F::ONE),
                (layout.is_sw(j), -F::ONE),
                (layout.is_amoadd_w(j), -F::ONE),
                (layout.is_auipc(j), -F::ONE),
                (layout.is_jalr(j), -F::ONE),
            ],
        ));

        // AND/XOR/OR tables (R-type + I-type + AMO ops).
        constraints.push(Constraint::terms(
            one,
            false,
            vec![
                (layout.and_has_lookup(j), F::ONE),
                (layout.is_and(j), -F::ONE),
                (layout.is_andi(j), -F::ONE),
                (layout.is_amoand_w(j), -F::ONE),
            ],
        ));
        constraints.push(Constraint::terms(
            one,
            false,
            vec![
                (layout.xor_has_lookup(j), F::ONE),
                (layout.is_xor(j), -F::ONE),
                (layout.is_xori(j), -F::ONE),
                (layout.is_amoxor_w(j), -F::ONE),
            ],
        ));
        constraints.push(Constraint::terms(
            one,
            false,
            vec![
                (layout.or_has_lookup(j), F::ONE),
                (layout.is_or(j), -F::ONE),
                (layout.is_ori(j), -F::ONE),
                (layout.is_amoor_w(j), -F::ONE),
            ],
        ));

        // Shift tables.
        constraints.push(Constraint::terms(
            one,
            false,
            vec![
                (layout.sll_has_lookup(j), F::ONE),
                (layout.is_sll(j), -F::ONE),
                (layout.is_slli(j), -F::ONE),
            ],
        ));
        constraints.push(Constraint::terms(
            one,
            false,
            vec![
                (layout.srl_has_lookup(j), F::ONE),
                (layout.is_srl(j), -F::ONE),
                (layout.is_srli(j), -F::ONE),
            ],
        ));
        constraints.push(Constraint::terms(
            one,
            false,
            vec![
                (layout.sra_has_lookup(j), F::ONE),
                (layout.is_sra(j), -F::ONE),
                (layout.is_srai(j), -F::ONE),
            ],
        ));

        // SLT/SLTU tables (ALU + branch comparisons).
        constraints.push(Constraint::terms(
            one,
            false,
            vec![
                (layout.slt_has_lookup(j), F::ONE),
                (layout.is_slt(j), -F::ONE),
                (layout.is_slti(j), -F::ONE),
                (layout.is_blt(j), -F::ONE),
                (layout.is_bge(j), -F::ONE),
            ],
        ));
        constraints.push(Constraint::terms(
            one,
            false,
            vec![
                (layout.sltu_has_lookup(j), F::ONE),
                (layout.is_sltu(j), -F::ONE),
                (layout.is_sltiu(j), -F::ONE),
                (layout.is_bltu(j), -F::ONE),
                (layout.is_bgeu(j), -F::ONE),
            ],
        ));

        // Instruction-specific writeback:
        // - RV32I ALU ops + AUIPC: rd_write_val = alu_out (verified via Shout)
        // - LW: rd_write_val = mem_rv (verified via Twist)
        // - LUI: rd_write_val = imm_u (pure)
        for &f in &[
            layout.is_add(j),
            layout.is_sub(j),
            layout.is_sll(j),
            layout.is_slt(j),
            layout.is_sltu(j),
            layout.is_xor(j),
            layout.is_srl(j),
            layout.is_sra(j),
            layout.is_or(j),
            layout.is_and(j),
            layout.is_mul(j),
            layout.is_mulh(j),
            layout.is_mulhu(j),
            layout.is_mulhsu(j),
            layout.is_div(j),
            layout.is_divu(j),
            layout.is_rem(j),
            layout.is_remu(j),
            layout.is_addi(j),
            layout.is_slti(j),
            layout.is_sltiu(j),
            layout.is_xori(j),
            layout.is_ori(j),
            layout.is_andi(j),
            layout.is_slli(j),
            layout.is_srli(j),
            layout.is_srai(j),
            layout.is_auipc(j),
        ] {
            constraints.push(Constraint::terms(
                f,
                false,
                vec![(layout.rd_write_val(j), F::ONE), (layout.alu_out(j), -F::ONE)],
            ));
        }
        constraints.push(Constraint::terms_or(
            &[
                layout.is_lw(j),
                layout.is_amoswap_w(j),
                layout.is_amoadd_w(j),
                layout.is_amoxor_w(j),
                layout.is_amoor_w(j),
                layout.is_amoand_w(j),
            ],
            false,
            vec![(layout.rd_write_val(j), F::ONE), (layout.mem_rv(j), -F::ONE)],
        ));
        constraints.push(Constraint::terms(
            layout.is_lui(j),
            false,
            vec![(layout.rd_write_val(j), F::ONE), (layout.imm_u(j), -F::ONE)],
        ));

        // JAL/JALR writeback: rd_write_val = pc_in + 4.
        constraints.push(Constraint::terms_or(
            &[layout.is_jal(j), layout.is_jalr(j)],
            false,
            vec![
                (layout.rd_write_val(j), F::ONE),
                (pc_in, -F::ONE),
                (one, -F::from_u64(4)),
            ],
        ));

        // PC update:
        // - Straight-line instructions: pc_out = pc_in + 4.
        for &f in &[
            layout.is_add(j),
            layout.is_sub(j),
            layout.is_sll(j),
            layout.is_slt(j),
            layout.is_sltu(j),
            layout.is_xor(j),
            layout.is_srl(j),
            layout.is_sra(j),
            layout.is_or(j),
            layout.is_and(j),
            layout.is_mul(j),
            layout.is_mulh(j),
            layout.is_mulhu(j),
            layout.is_mulhsu(j),
            layout.is_div(j),
            layout.is_divu(j),
            layout.is_rem(j),
            layout.is_remu(j),
            layout.is_addi(j),
            layout.is_slti(j),
            layout.is_sltiu(j),
            layout.is_xori(j),
            layout.is_ori(j),
            layout.is_andi(j),
            layout.is_slli(j),
            layout.is_srli(j),
            layout.is_srai(j),
            layout.is_lw(j),
            layout.is_sw(j),
            layout.is_amoswap_w(j),
            layout.is_amoadd_w(j),
            layout.is_amoxor_w(j),
            layout.is_amoor_w(j),
            layout.is_amoand_w(j),
            layout.is_lui(j),
            layout.is_auipc(j),
            layout.is_halt(j),
        ] {
            constraints.push(Constraint::terms(
                f,
                false,
                vec![(pc_out, F::ONE), (pc_in, -F::ONE), (one, -F::from_u64(4))],
            ));
        }

        // - JAL: pc_out = pc_in + imm_j.
        constraints.push(Constraint::terms(
            layout.is_jal(j),
            false,
            vec![(pc_out, F::ONE), (pc_in, -F::ONE), (layout.imm_j(j), -F::ONE)],
        ));

        // - JALR: pc_out = (rs1 + imm_i) & !1.
        //
        // The ADD-table Shout output `alu_out` is (rs1 + imm_i) mod 2^32.
        // Let a0,b0 be the operand LSBs (from the interleaved ADD key bits), and let a0b0 = a0*b0.
        // Then lsb(alu_out) = a0 XOR b0 = a0 + b0 - 2*a0b0, and pc_out = alu_out - lsb.
        constraints.push(Constraint::terms(
            layout.is_jalr(j),
            false,
            vec![
                (pc_out, F::ONE),
                (layout.alu_out(j), -F::ONE),
                (add_a0, F::ONE),
                (add_b0, F::ONE),
                (layout.add_a0b0(j), -F::from_u64(2)),
            ],
        ));

        let branch_flags = [
            layout.is_beq(j),
            layout.is_bne(j),
            layout.is_blt(j),
            layout.is_bge(j),
            layout.is_bltu(j),
            layout.is_bgeu(j),
        ];

        // Branch control: br_taken/br_not_taken are only set on branch rows.
        constraints.push(Constraint::terms_or(
            &branch_flags,
            true, // (1 - is_branch)
            vec![(layout.br_taken(j), F::ONE)],
        ));
        constraints.push(Constraint::terms_or(
            &branch_flags,
            true, // (1 - is_branch)
            vec![(layout.br_not_taken(j), F::ONE)],
        ));

        // Exactly one branch case on branch rows: br_taken + br_not_taken = is_branch.
        constraints.push(Constraint::terms(
            one,
            false,
            vec![
                (layout.br_taken(j), F::ONE),
                (layout.br_not_taken(j), F::ONE),
                (layout.is_beq(j), -F::ONE),
                (layout.is_bne(j), -F::ONE),
                (layout.is_blt(j), -F::ONE),
                (layout.is_bge(j), -F::ONE),
                (layout.is_bltu(j), -F::ONE),
                (layout.is_bgeu(j), -F::ONE),
            ],
        ));

        // Branch decision comes from the Shout output:
        // - BEQ/BNE/BLT/BLTU: br_taken = alu_out
        // - BGE/BGEU: br_taken = 1 - alu_out
        constraints.push(Constraint::terms_or(
            &[
                layout.is_beq(j),
                layout.is_bne(j),
                layout.is_blt(j),
                layout.is_bltu(j),
            ],
            false,
            vec![(layout.br_taken(j), F::ONE), (layout.alu_out(j), -F::ONE)],
        ));
        constraints.push(Constraint::terms_or(
            &[layout.is_bge(j), layout.is_bgeu(j)],
            false,
            vec![
                (layout.br_taken(j), F::ONE),
                (layout.alu_out(j), F::ONE),
                (one, -F::ONE),
            ],
        ));

        // Branch PC update:
        // - Taken: pc_out = pc_in + imm_b.
        // - Not taken: pc_out = pc_in + 4.
        constraints.push(Constraint::terms(
            layout.br_taken(j),
            false,
            vec![(pc_out, F::ONE), (pc_in, -F::ONE), (layout.imm_b(j), -F::ONE)],
        ));
        constraints.push(Constraint::terms(
            layout.br_not_taken(j),
            false,
            vec![(pc_out, F::ONE), (pc_in, -F::ONE), (one, -F::from_u64(4))],
        ));

        // Helper: bind the product of the ADD-table operand LSBs (used for JALR mask in Phase 2).
        constraints.push(Constraint::mul(add_a0, add_b0, layout.add_a0b0(j)));

        // --- Shout key correctness (ADD table bus addr bits interleaving) ---
        let mut even_terms = vec![(layout.rs1_val(j), F::ONE)];
        for i in 0..RV32_XLEN {
            let bit_col_id = add_cols.addr_bits.start + 2 * i;
            let bit = layout.bus.bus_cell(bit_col_id, j);
            even_terms.push((bit, -F::from_u64(pow2_u64(i))));
        }
        constraints.push(Constraint::terms_or(
            &[
                layout.is_add(j),
                layout.is_addi(j),
                layout.is_lw(j),
                layout.is_sw(j),
                layout.is_jalr(j),
            ],
            false,
            even_terms,
        ));

        // AMOADD.W uses mem_rv as the left ADD operand (old memory value).
        let mut even_terms_amoadd = vec![(layout.mem_rv(j), F::ONE)];
        for i in 0..RV32_XLEN {
            let bit_col_id = add_cols.addr_bits.start + 2 * i;
            let bit = layout.bus.bus_cell(bit_col_id, j);
            even_terms_amoadd.push((bit, -F::from_u64(pow2_u64(i))));
        }
        constraints.push(Constraint::terms(layout.is_amoadd_w(j), false, even_terms_amoadd));

        let mut even_terms_auipc = vec![(pc_in, F::ONE)];
        for i in 0..RV32_XLEN {
            let bit_col_id = add_cols.addr_bits.start + 2 * i;
            let bit = layout.bus.bus_cell(bit_col_id, j);
            even_terms_auipc.push((bit, -F::from_u64(pow2_u64(i))));
        }
        constraints.push(Constraint::terms(layout.is_auipc(j), false, even_terms_auipc));

        let mut odd_terms_add = vec![(layout.rs2_val(j), F::ONE)];
        for i in 0..RV32_XLEN {
            let bit_col_id = add_cols.addr_bits.start + 2 * i + 1;
            let bit = layout.bus.bus_cell(bit_col_id, j);
            odd_terms_add.push((bit, -F::from_u64(pow2_u64(i))));
        }
        constraints.push(Constraint::terms_or(
            &[layout.is_add(j), layout.is_amoadd_w(j)],
            false,
            odd_terms_add,
        ));

        let mut odd_terms_addi = vec![(layout.imm_i(j), F::ONE)];
        for i in 0..RV32_XLEN {
            let bit_col_id = add_cols.addr_bits.start + 2 * i + 1;
            let bit = layout.bus.bus_cell(bit_col_id, j);
            odd_terms_addi.push((bit, -F::from_u64(pow2_u64(i))));
        }
        constraints.push(Constraint::terms_or(
            &[layout.is_addi(j), layout.is_lw(j), layout.is_jalr(j)],
            false,
            odd_terms_addi,
        ));

        let mut odd_terms_sw = vec![(layout.imm_s(j), F::ONE)];
        for i in 0..RV32_XLEN {
            let bit_col_id = add_cols.addr_bits.start + 2 * i + 1;
            let bit = layout.bus.bus_cell(bit_col_id, j);
            odd_terms_sw.push((bit, -F::from_u64(pow2_u64(i))));
        }
        constraints.push(Constraint::terms(layout.is_sw(j), false, odd_terms_sw));

        let mut odd_terms_auipc = vec![(layout.imm_u(j), F::ONE)];
        for i in 0..RV32_XLEN {
            let bit_col_id = add_cols.addr_bits.start + 2 * i + 1;
            let bit = layout.bus.bus_cell(bit_col_id, j);
            odd_terms_auipc.push((bit, -F::from_u64(pow2_u64(i))));
        }
        constraints.push(Constraint::terms(layout.is_auipc(j), false, odd_terms_auipc));

        // --- Shout key correctness (EQ/NEQ table bus addr bits interleaving) ---
        if let Some(eq_cols) = eq_cols {
            let flag = layout.is_beq(j);
            let mut even = vec![(layout.rs1_val(j), F::ONE)];
            for i in 0..RV32_XLEN {
                let bit_col_id = eq_cols.addr_bits.start + 2 * i;
                let bit = layout.bus.bus_cell(bit_col_id, j);
                even.push((bit, -F::from_u64(pow2_u64(i))));
            }
            constraints.push(Constraint::terms(flag, false, even));

            let mut odd = vec![(layout.rs2_val(j), F::ONE)];
            for i in 0..RV32_XLEN {
                let bit_col_id = eq_cols.addr_bits.start + 2 * i + 1;
                let bit = layout.bus.bus_cell(bit_col_id, j);
                odd.push((bit, -F::from_u64(pow2_u64(i))));
            }
            constraints.push(Constraint::terms(flag, false, odd));
        }
        if let Some(neq_cols) = neq_cols {
            let flag = layout.is_bne(j);
            let mut even = vec![(layout.rs1_val(j), F::ONE)];
            for i in 0..RV32_XLEN {
                let bit_col_id = neq_cols.addr_bits.start + 2 * i;
                let bit = layout.bus.bus_cell(bit_col_id, j);
                even.push((bit, -F::from_u64(pow2_u64(i))));
            }
            constraints.push(Constraint::terms(flag, false, even));

            let mut odd = vec![(layout.rs2_val(j), F::ONE)];
            for i in 0..RV32_XLEN {
                let bit_col_id = neq_cols.addr_bits.start + 2 * i + 1;
                let bit = layout.bus.bus_cell(bit_col_id, j);
                odd.push((bit, -F::from_u64(pow2_u64(i))));
            }
            constraints.push(Constraint::terms(flag, false, odd));
        }

        // --- Shout key correctness (other opcode tables) ---
        // AND / OR / XOR (R-type uses rs2, I-type uses imm_i).
        if let Some(and_cols) = and_cols {
            constraints.push(Constraint::terms_or(
                &[layout.is_and(j), layout.is_andi(j)],
                false,
                pack_interleaved_operand(and_cols.addr_bits.start, j, 0, layout.rs1_val(j)),
            ));
            constraints.push(Constraint::terms(
                layout.is_amoand_w(j),
                false,
                pack_interleaved_operand(and_cols.addr_bits.start, j, 0, layout.mem_rv(j)),
            ));
            constraints.push(Constraint::terms(
                layout.is_and(j),
                false,
                pack_interleaved_operand(and_cols.addr_bits.start, j, 1, layout.rs2_val(j)),
            ));
            constraints.push(Constraint::terms(
                layout.is_amoand_w(j),
                false,
                pack_interleaved_operand(and_cols.addr_bits.start, j, 1, layout.rs2_val(j)),
            ));
            constraints.push(Constraint::terms(
                layout.is_andi(j),
                false,
                pack_interleaved_operand(and_cols.addr_bits.start, j, 1, layout.imm_i(j)),
            ));
        }

        if let Some(or_cols) = or_cols {
            constraints.push(Constraint::terms_or(
                &[layout.is_or(j), layout.is_ori(j)],
                false,
                pack_interleaved_operand(or_cols.addr_bits.start, j, 0, layout.rs1_val(j)),
            ));
            constraints.push(Constraint::terms(
                layout.is_amoor_w(j),
                false,
                pack_interleaved_operand(or_cols.addr_bits.start, j, 0, layout.mem_rv(j)),
            ));
            constraints.push(Constraint::terms(
                layout.is_or(j),
                false,
                pack_interleaved_operand(or_cols.addr_bits.start, j, 1, layout.rs2_val(j)),
            ));
            constraints.push(Constraint::terms(
                layout.is_amoor_w(j),
                false,
                pack_interleaved_operand(or_cols.addr_bits.start, j, 1, layout.rs2_val(j)),
            ));
            constraints.push(Constraint::terms(
                layout.is_ori(j),
                false,
                pack_interleaved_operand(or_cols.addr_bits.start, j, 1, layout.imm_i(j)),
            ));
        }

        if let Some(xor_cols) = xor_cols {
            constraints.push(Constraint::terms_or(
                &[layout.is_xor(j), layout.is_xori(j)],
                false,
                pack_interleaved_operand(xor_cols.addr_bits.start, j, 0, layout.rs1_val(j)),
            ));
            constraints.push(Constraint::terms(
                layout.is_amoxor_w(j),
                false,
                pack_interleaved_operand(xor_cols.addr_bits.start, j, 0, layout.mem_rv(j)),
            ));
            constraints.push(Constraint::terms(
                layout.is_xor(j),
                false,
                pack_interleaved_operand(xor_cols.addr_bits.start, j, 1, layout.rs2_val(j)),
            ));
            constraints.push(Constraint::terms(
                layout.is_amoxor_w(j),
                false,
                pack_interleaved_operand(xor_cols.addr_bits.start, j, 1, layout.rs2_val(j)),
            ));
            constraints.push(Constraint::terms(
                layout.is_xori(j),
                false,
                pack_interleaved_operand(xor_cols.addr_bits.start, j, 1, layout.imm_i(j)),
            ));
        }

        // SUB (R-type only).
        if let Some(sub_cols) = sub_cols {
            constraints.push(Constraint::terms(
                layout.is_sub(j),
                false,
                pack_interleaved_operand(sub_cols.addr_bits.start, j, 0, layout.rs1_val(j)),
            ));
            constraints.push(Constraint::terms(
                layout.is_sub(j),
                false,
                pack_interleaved_operand(sub_cols.addr_bits.start, j, 1, layout.rs2_val(j)),
            ));
        }

        // Shifts (R-type uses rs2, I-type uses shamt).
        if let Some(sll_cols) = sll_cols {
            constraints.push(Constraint::terms_or(
                &[layout.is_sll(j), layout.is_slli(j)],
                false,
                pack_interleaved_operand(sll_cols.addr_bits.start, j, 0, layout.rs1_val(j)),
            ));
            constraints.push(Constraint::terms(
                layout.is_sll(j),
                false,
                pack_interleaved_operand(sll_cols.addr_bits.start, j, 1, layout.rs2_val(j)),
            ));
            constraints.push(Constraint::terms(
                layout.is_slli(j),
                false,
                pack_interleaved_operand(sll_cols.addr_bits.start, j, 1, layout.shamt(j)),
            ));
        }

        if let Some(srl_cols) = srl_cols {
            constraints.push(Constraint::terms_or(
                &[layout.is_srl(j), layout.is_srli(j)],
                false,
                pack_interleaved_operand(srl_cols.addr_bits.start, j, 0, layout.rs1_val(j)),
            ));
            constraints.push(Constraint::terms(
                layout.is_srl(j),
                false,
                pack_interleaved_operand(srl_cols.addr_bits.start, j, 1, layout.rs2_val(j)),
            ));
            constraints.push(Constraint::terms(
                layout.is_srli(j),
                false,
                pack_interleaved_operand(srl_cols.addr_bits.start, j, 1, layout.shamt(j)),
            ));
        }

        if let Some(sra_cols) = sra_cols {
            constraints.push(Constraint::terms_or(
                &[layout.is_sra(j), layout.is_srai(j)],
                false,
                pack_interleaved_operand(sra_cols.addr_bits.start, j, 0, layout.rs1_val(j)),
            ));
            constraints.push(Constraint::terms(
                layout.is_sra(j),
                false,
                pack_interleaved_operand(sra_cols.addr_bits.start, j, 1, layout.rs2_val(j)),
            ));
            constraints.push(Constraint::terms(
                layout.is_srai(j),
                false,
                pack_interleaved_operand(sra_cols.addr_bits.start, j, 1, layout.shamt(j)),
            ));
        }

        // SLT/SLTU (ALU + branch comparisons).
        if let Some(slt_cols) = slt_cols {
            constraints.push(Constraint::terms_or(
                &[layout.is_slt(j), layout.is_slti(j), layout.is_blt(j), layout.is_bge(j)],
                false,
                pack_interleaved_operand(slt_cols.addr_bits.start, j, 0, layout.rs1_val(j)),
            ));
            constraints.push(Constraint::terms_or(
                &[layout.is_slt(j), layout.is_blt(j), layout.is_bge(j)],
                false,
                pack_interleaved_operand(slt_cols.addr_bits.start, j, 1, layout.rs2_val(j)),
            ));
            constraints.push(Constraint::terms(
                layout.is_slti(j),
                false,
                pack_interleaved_operand(slt_cols.addr_bits.start, j, 1, layout.imm_i(j)),
            ));
        }

        if let Some(sltu_cols) = sltu_cols {
            constraints.push(Constraint::terms_or(
                &[layout.is_sltu(j), layout.is_sltiu(j), layout.is_bltu(j), layout.is_bgeu(j)],
                false,
                pack_interleaved_operand(sltu_cols.addr_bits.start, j, 0, layout.rs1_val(j)),
            ));
            constraints.push(Constraint::terms_or(
                &[layout.is_sltu(j), layout.is_bltu(j), layout.is_bgeu(j)],
                false,
                pack_interleaved_operand(sltu_cols.addr_bits.start, j, 1, layout.rs2_val(j)),
            ));
            constraints.push(Constraint::terms(
                layout.is_sltiu(j),
                false,
                pack_interleaved_operand(sltu_cols.addr_bits.start, j, 1, layout.imm_i(j)),
            ));
        }

        // RV32M (all R-type, rs1/rs2 operands).
        for (flag, cols_opt) in [
            (layout.is_mul(j), mul_cols),
            (layout.is_mulh(j), mulh_cols),
            (layout.is_mulhu(j), mulhu_cols),
            (layout.is_mulhsu(j), mulhsu_cols),
            (layout.is_div(j), div_cols),
            (layout.is_divu(j), divu_cols),
            (layout.is_rem(j), rem_cols),
            (layout.is_remu(j), remu_cols),
        ] {
            if let Some(cols) = cols_opt {
                constraints.push(Constraint::terms(
                    flag,
                    false,
                    pack_interleaved_operand(cols.addr_bits.start, j, 0, layout.rs1_val(j)),
                ));
                constraints.push(Constraint::terms(
                    flag,
                    false,
                    pack_interleaved_operand(cols.addr_bits.start, j, 1, layout.rs2_val(j)),
                ));
            }
        }

        // --- Alignment constraints (MVP) ---
        // ROM fetch is always 32-bit, so enforce pc_in % 4 == 0 via PROG read address bits.
        let prog_bit0 = layout.bus.bus_cell(prog.ra_bits.start + 0, j);
        let prog_bit1 = layout.bus.bus_cell(prog.ra_bits.start + 1, j);
        constraints.push(Constraint::zero(one, prog_bit0));
        constraints.push(Constraint::zero(one, prog_bit1));

        // Enforce word alignment for LW/SW via RAM bus addr bits.
        let ra0 = layout.bus.bus_cell(ram.ra_bits.start + 0, j);
        let ra1 = layout.bus.bus_cell(ram.ra_bits.start + 1, j);
        let wa0 = layout.bus.bus_cell(ram.wa_bits.start + 0, j);
        let wa1 = layout.bus.bus_cell(ram.wa_bits.start + 1, j);
        let amo_flags = [
            layout.is_amoswap_w(j),
            layout.is_amoadd_w(j),
            layout.is_amoxor_w(j),
            layout.is_amoor_w(j),
            layout.is_amoand_w(j),
        ];
        constraints.push(Constraint::terms_or(
            &[layout.is_lw(j), amo_flags[0], amo_flags[1], amo_flags[2], amo_flags[3], amo_flags[4]],
            false,
            vec![(ra0, F::ONE)],
        ));
        constraints.push(Constraint::terms_or(
            &[layout.is_lw(j), amo_flags[0], amo_flags[1], amo_flags[2], amo_flags[3], amo_flags[4]],
            false,
            vec![(ra1, F::ONE)],
        ));
        constraints.push(Constraint::terms_or(
            &[layout.is_sw(j), amo_flags[0], amo_flags[1], amo_flags[2], amo_flags[3], amo_flags[4]],
            false,
            vec![(wa0, F::ONE)],
        ));
        constraints.push(Constraint::terms_or(
            &[layout.is_sw(j), amo_flags[0], amo_flags[1], amo_flags[2], amo_flags[3], amo_flags[4]],
            false,
            vec![(wa1, F::ONE)],
        ));
    }

    // --- Intra-chunk composition / padding semantics ---
    // Enforce monotone activity and state continuity:
    // - is_active[j+1] => is_active[j]
    // - pc_in[j+1] == pc_out[j] and regs_in[j+1] == regs_out[j] for all j
    //
    // The unconditional continuity ensures padding rows (is_active=0) *carry* the final
    // architectural state forward, making the final state unambiguous in an L1-style layout.
    for j in 0..layout.chunk_size.saturating_sub(1) {
        let a = layout.is_active(j);
        let b = layout.is_active(j + 1);

        // b * (1 - a) = 0
        constraints.push(Constraint::terms(
            b,
            false,
            vec![(one, F::ONE), (a, -F::ONE)],
        ));

        // HALT terminates execution within a chunk: is_halt[j] => is_active[j+1] == 0.
        constraints.push(Constraint::terms(
            layout.is_halt(j),
            false,
            vec![(layout.is_active(j + 1), F::ONE)],
        ));

        // pc_in[j+1] - pc_out[j] = 0
        constraints.push(Constraint::terms(
            one,
            false,
            vec![(layout.pc_in(j + 1), F::ONE), (layout.pc_out(j), -F::ONE)],
        ));

        // regs_in[j+1] - regs_out[j] = 0 for all regs
        for r in 0..32 {
            constraints.push(Constraint::terms(
                one,
                false,
                vec![(layout.reg_in(r, j + 1), F::ONE), (layout.reg_out(r, j), -F::ONE)],
            ));
        }
    }

    Ok(constraints)
}

/// Build the RV32 B1 step CCS and its witness layout.
///
/// Requirements:
/// - `mem_layouts` must include `RAM_ID` and `PROG_ID`.
/// - `mem_layouts[PROG_ID]` is byte-addressed (`n_side=2`, `ell=1`).
///
/// `shout_table_ids` must be non-empty and include the RV32 `ADD` table id (3). Any subset of the
/// base RV32I opcode tables (ids 0..=11) is allowed (unused tables can be bound to fixed-zero CPU
/// columns in the shared-bus config).
pub fn build_rv32_b1_step_ccs(
    mem_layouts: &HashMap<u32, PlainMemLayout>,
    shout_table_ids: &[u32],
    chunk_size: usize,
) -> Result<(CcsStructure<F>, Rv32B1Layout), String> {
    if chunk_size == 0 {
        return Err("RV32 B1: chunk_size must be >= 1".into());
    }
    let ram_id = RAM_ID.0;
    let prog_id = PROG_ID.0;
    if !mem_layouts.contains_key(&ram_id) {
        return Err(format!("RV32 B1: mem_layouts missing RAM_ID={ram_id}"));
    }
    if !mem_layouts.contains_key(&prog_id) {
        return Err(format!("RV32 B1: mem_layouts missing PROG_ID={prog_id}"));
    }

    // B1 circuit currently assumes only RISC-V opcode Shout tables (ell_addr = 2*xlen = 64).
    let (table_ids, shout_ell_addrs) = derive_shout_ids_and_ell_addrs(shout_table_ids)?;

    let (mem_ids, twist_ell_addrs) = derive_mem_ids_and_ell_addrs(mem_layouts)?;
    if mem_ids.len() != twist_ell_addrs.len() {
        return Err("RV32 B1: internal error (twist ell addrs mismatch)".into());
    }
    let bus_cols_per_step: usize = shout_ell_addrs.iter().sum::<usize>() + 2 * shout_ell_addrs.len()
        + twist_ell_addrs
            .iter()
            .map(|&ell_addr| 2 * ell_addr + 5)
            .sum::<usize>();
    let bus_region_len = bus_cols_per_step
        .checked_mul(chunk_size)
        .ok_or_else(|| "RV32 B1: bus_region_len overflow".to_string())?;

    // Probe layout to learn the CPU column footprint and count injected constraints.
    // We rebuild once with the minimal `m` after computing exact sizes.
    let mut probe_m = bus_region_len
        .checked_add(1)
        .ok_or_else(|| "RV32 B1: probe_m overflow".to_string())?;
    let probe = loop {
        match build_layout_with_m(probe_m, mem_layouts, &table_ids, chunk_size) {
            Ok(layout) => break layout,
            Err(e) if e.contains("need more padding columns before bus tail") || e.contains("overlaps public inputs") => {
                probe_m = probe_m
                    .checked_mul(2)
                    .ok_or_else(|| "RV32 B1: probe_m overflow".to_string())?;
            }
            Err(e) => return Err(e),
        }
    };
    let cpu_cols_used = probe.add_a0b0 + chunk_size;

    let semantic = semantic_constraints(&probe, mem_layouts)?;
    let injected = injected_bus_constraints_len(&probe, &table_ids, &mem_ids);

    let m_cols_min = cpu_cols_used + bus_region_len;
    let m_rows_min = semantic.len() + injected;
    let m = m_cols_min.max(m_rows_min);

    let layout = build_layout_with_m(m, mem_layouts, &table_ids, chunk_size)?;
    let constraints = semantic_constraints(&layout, mem_layouts)?;
    if constraints.len() + injected > layout.m {
        return Err(format!(
            "RV32 B1: internal error: constraints={} + injected={} > m={}",
            constraints.len(),
            injected,
            layout.m
        ));
    }

    let ccs = build_identity_first_r1cs_ccs(&constraints, layout.m, layout.const_one)?;
    Ok((ccs, layout))
}
