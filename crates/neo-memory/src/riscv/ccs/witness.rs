use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

use neo_vm_trace::{StepTrace, TwistOpKind};

use crate::riscv::lookups::{decode_instruction, BranchCondition, RiscvInstruction, RiscvMemOp, RiscvOpcode, RAM_ID, PROG_ID};

use super::Rv32B1Layout;

/// Build a CPU witness vector `z` (CPU region only; the bus tail is written by `R1csCpu`).
pub fn rv32_b1_chunk_to_witness(
    layout: Rv32B1Layout,
) -> Box<dyn Fn(&[StepTrace<u64, u64>]) -> Vec<F> + Send + Sync> {
    Box::new(move |chunk: &[StepTrace<u64, u64>]| {
        rv32_b1_chunk_to_witness_checked(&layout, chunk).unwrap_or_else(|e| {
            panic!("RV32 B1 witness build failed: {e}");
        })
    })
}

pub fn rv32_b1_chunk_to_witness_checked(layout: &Rv32B1Layout, chunk: &[StepTrace<u64, u64>]) -> Result<Vec<F>, String> {
    let mut z = vec![F::ZERO; layout.bus.bus_base];

    z[layout.const_one] = F::ONE;

    // Carry the architectural state forward through padding rows.
    // Initialize from the chunk's start state so fully-inactive chunks are well-defined.
    let mut carried_pc = 0u64;
    let mut carried_regs = [0u64; 32];

    if let Some(first) = chunk.first() {
        z[layout.pc0] = F::from_u64(first.pc_before);
        for r in 0..32 {
            z[layout.regs0_start + r] = F::from_u64(first.regs_before[r]);
            carried_regs[r] = first.regs_before[r];
        }
        z[layout.regs0_start] = F::ZERO;
        carried_regs[0] = 0;
        carried_pc = first.pc_before;
    }

    for j in 0..layout.chunk_size {
        if j >= chunk.len() {
            z[layout.is_active(j)] = F::ZERO;

            z[layout.pc_in(j)] = F::from_u64(carried_pc);
            z[layout.pc_out(j)] = F::from_u64(carried_pc);
            for r in 0..32 {
                z[layout.reg_in(r, j)] = F::from_u64(carried_regs[r]);
                z[layout.reg_out(r, j)] = F::from_u64(carried_regs[r]);
            }
            continue;
        }
        let step = &chunk[j];

        // A row is active iff it contains exactly one PROG_ID read (B1 instruction fetch).
        // Padding rows contain no Twist/Shout events and are treated as inactive.
        let mut prog_read: Option<(u64, u64)> = None;
        for ev in &step.twist_events {
            if ev.twist_id != PROG_ID {
                continue;
            }
            match ev.kind {
                TwistOpKind::Read => {
                    if prog_read.replace((ev.addr, ev.value)).is_some() {
                        return Err(format!(
                            "RV32 B1: multiple PROG_ID reads in one step at pc={:#x} (chunk j={j})",
                            step.pc_before
                        ));
                    }
                }
                TwistOpKind::Write => {
                    return Err(format!(
                        "RV32 B1: unexpected PROG_ID write at pc={:#x} (chunk j={j})",
                        step.pc_before
                    ));
                }
            }
        }

        if prog_read.is_none() {
            if !step.twist_events.is_empty() || !step.shout_events.is_empty() {
                return Err(format!(
                    "RV32 B1: non-empty events in inactive row at step cycle={} (chunk j={j})",
                    step.cycle
                ));
            }

            z[layout.is_active(j)] = F::ZERO;
            z[layout.pc_in(j)] = F::from_u64(carried_pc);
            z[layout.pc_out(j)] = F::from_u64(carried_pc);
            for r in 0..32 {
                z[layout.reg_in(r, j)] = F::from_u64(carried_regs[r]);
                z[layout.reg_out(r, j)] = F::from_u64(carried_regs[r]);
            }
            continue;
        }

        z[layout.is_active(j)] = F::ONE;
        z[layout.pc_in(j)] = F::from_u64(step.pc_before);
        z[layout.pc_out(j)] = F::from_u64(step.pc_after);

        // Registers.
        for r in 0..32 {
            z[layout.reg_in(r, j)] = F::from_u64(step.regs_before[r]);
            z[layout.reg_out(r, j)] = F::from_u64(step.regs_after[r]);
        }

        carried_pc = step.pc_after;
        carried_regs.copy_from_slice(&step.regs_after);
        carried_regs[0] = 0;

        // Instruction word: read from PROG_ID Twist event (commitment-bound source).
        let (prog_addr, prog_value) = prog_read.expect("checked prog_read is present");
        if prog_addr != step.pc_before {
            return Err(format!(
                "RV32 B1: PROG_ID read addr mismatch at pc={:#x} (chunk j={j}): read_addr={:#x}",
                step.pc_before, prog_addr
            ));
        }
        let instr_word_u32 = u32::try_from(prog_value).map_err(|_| {
            format!(
                "RV32 B1: PROG_ID read value does not fit in u32 at pc={:#x}: value={:#x}",
                step.pc_before, prog_value
            )
        })?;
        z[layout.instr_word(j)] = F::from_u64(instr_word_u32 as u64);

        // Bits.
        for i in 0..32 {
            z[layout.instr_bit(i, j)] = if ((instr_word_u32 >> i) & 1) == 1 { F::ONE } else { F::ZERO };
        }

        // Decode fields.
        let opcode = instr_word_u32 & 0x7f;
        let rd = (instr_word_u32 >> 7) & 0x1f;
        let funct3 = (instr_word_u32 >> 12) & 0x7;
        let rs1 = (instr_word_u32 >> 15) & 0x1f;
        let rs2 = (instr_word_u32 >> 20) & 0x1f;
        let funct7 = (instr_word_u32 >> 25) & 0x7f;

        z[layout.opcode(j)] = F::from_u64(opcode as u64);
        z[layout.funct3(j)] = F::from_u64(funct3 as u64);
        z[layout.funct7(j)] = F::from_u64(funct7 as u64);
        z[layout.rd_field(j)] = F::from_u64(rd as u64);
        z[layout.rs1_field(j)] = F::from_u64(rs1 as u64);
        z[layout.rs2_field(j)] = F::from_u64(rs2 as u64);

        // Helpers for immediate representations:
        // - `sx_u32` matches the CCS u32-style encoding used for imm_i / imm_s.
        // - `from_i32` matches the CCS signed encoding used for imm_b / imm_j.
        let sx_u32 = |x: i32| x as u32 as u64;
        let from_i32 = |v: i32| -> F {
            if v >= 0 {
                F::from_u64(v as u64)
            } else {
                -F::from_u64((-v) as u64)
            }
        };

        // Immediate raw fields.
        let imm12_raw = ((instr_word_u32 >> 20) & 0xfff) as u32;
        z[layout.imm12_raw(j)] = F::from_u64(imm12_raw as u64);

        // I-type immediate (sign-extended 12-bit).
        let imm_i = ((imm12_raw as i32) << 20) >> 20;
        z[layout.imm_i(j)] = F::from_u64(sx_u32(imm_i));

        // S-type immediate (sign-extended 12-bit).
        let imm_s_raw = (((instr_word_u32 >> 7) & 0x1f) | (((instr_word_u32 >> 25) & 0x7f) << 5)) as u32;
        let imm_s = ((imm_s_raw as i32) << 20) >> 20;
        z[layout.imm_s(j)] = F::from_u64(sx_u32(imm_s));

        // U-type immediate (upper 20 bits).
        let imm_u = (instr_word_u32 & 0xfffff000) as u64;
        z[layout.imm_u(j)] = F::from_u64(imm_u);

        // B-type immediate raw bits (before sign extension).
        let imm_b_raw = (((instr_word_u32 >> 7) & 0x1) << 11)
            | (((instr_word_u32 >> 8) & 0xf) << 1)
            | (((instr_word_u32 >> 25) & 0x3f) << 5)
            | (((instr_word_u32 >> 31) & 0x1) << 12);
        z[layout.imm_b_raw(j)] = F::from_u64(imm_b_raw as u64);
        let imm_b = ((imm_b_raw as i32) << 19) >> 19;
        z[layout.imm_b(j)] = from_i32(imm_b);

        // J-type immediate raw bits (before sign extension).
        let imm_j_raw = (((instr_word_u32 >> 21) & 0x3ff) << 1)
            | (((instr_word_u32 >> 20) & 0x1) << 11)
            | (((instr_word_u32 >> 12) & 0xff) << 12)
            | (((instr_word_u32 >> 31) & 0x1) << 20);
        z[layout.imm_j_raw(j)] = F::from_u64(imm_j_raw as u64);
        let imm_j = ((imm_j_raw as i32) << 11) >> 11;
        z[layout.imm_j(j)] = from_i32(imm_j);

        // One-hot flags: use the shared decoder as the single source of truth.
        let decoded = decode_instruction(instr_word_u32)
            .map_err(|e| format!("RV32 B1: decode failed at pc={:#x}: {e}", step.pc_before))?;

        let mut is_add = false;
        let mut is_sub = false;
        let mut is_sll = false;
        let mut is_slt = false;
        let mut is_sltu = false;
        let mut is_xor = false;
        let mut is_srl = false;
        let mut is_sra = false;
        let mut is_or = false;
        let mut is_and = false;
        let mut is_mul = false;
        let mut is_mulh = false;
        let mut is_mulhu = false;
        let mut is_mulhsu = false;
        let mut is_div = false;
        let mut is_divu = false;
        let mut is_rem = false;
        let mut is_remu = false;

        let mut is_addi = false;
        let mut is_slti = false;
        let mut is_sltiu = false;
        let mut is_xori = false;
        let mut is_ori = false;
        let mut is_andi = false;
        let mut is_slli = false;
        let mut is_srli = false;
        let mut is_srai = false;

        let mut is_lw = false;
        let mut is_sw = false;
        let mut is_amoswap_w = false;
        let mut is_amoadd_w = false;
        let mut is_amoxor_w = false;
        let mut is_amoor_w = false;
        let mut is_amoand_w = false;
        let mut is_lui = false;
        let mut is_auipc = false;
        let mut is_beq = false;
        let mut is_bne = false;
        let mut is_blt = false;
        let mut is_bge = false;
        let mut is_bltu = false;
        let mut is_bgeu = false;
        let mut is_jal = false;
        let mut is_jalr = false;
        let mut is_halt = false;

        match decoded {
            RiscvInstruction::RAlu { op, .. } => match op {
                RiscvOpcode::Add => is_add = true,
                RiscvOpcode::Sub => is_sub = true,
                RiscvOpcode::Sll => is_sll = true,
                RiscvOpcode::Slt => is_slt = true,
                RiscvOpcode::Sltu => is_sltu = true,
                RiscvOpcode::Xor => is_xor = true,
                RiscvOpcode::Srl => is_srl = true,
                RiscvOpcode::Sra => is_sra = true,
                RiscvOpcode::Or => is_or = true,
                RiscvOpcode::And => is_and = true,
                RiscvOpcode::Mul => is_mul = true,
                RiscvOpcode::Mulh => is_mulh = true,
                RiscvOpcode::Mulhu => is_mulhu = true,
                RiscvOpcode::Mulhsu => is_mulhsu = true,
                RiscvOpcode::Div => is_div = true,
                RiscvOpcode::Divu => is_divu = true,
                RiscvOpcode::Rem => is_rem = true,
                RiscvOpcode::Remu => is_remu = true,
                _ => {}
            },
            RiscvInstruction::IAlu { op, .. } => match op {
                RiscvOpcode::Add => is_addi = true,
                RiscvOpcode::Slt => is_slti = true,
                RiscvOpcode::Sltu => is_sltiu = true,
                RiscvOpcode::Xor => is_xori = true,
                RiscvOpcode::Or => is_ori = true,
                RiscvOpcode::And => is_andi = true,
                RiscvOpcode::Sll => is_slli = true,
                RiscvOpcode::Srl => is_srli = true,
                RiscvOpcode::Sra => is_srai = true,
                _ => {}
            },
            RiscvInstruction::Load { op, .. } => match op {
                RiscvMemOp::Lw => is_lw = true,
                _ => {}
            },
            RiscvInstruction::Store { op, .. } => match op {
                RiscvMemOp::Sw => is_sw = true,
                _ => {}
            },
            RiscvInstruction::Amo { op, .. } => match op {
                RiscvMemOp::AmoswapW => is_amoswap_w = true,
                RiscvMemOp::AmoaddW => is_amoadd_w = true,
                RiscvMemOp::AmoxorW => is_amoxor_w = true,
                RiscvMemOp::AmoorW => is_amoor_w = true,
                RiscvMemOp::AmoandW => is_amoand_w = true,
                _ => {}
            },
            RiscvInstruction::Lui { .. } => is_lui = true,
            RiscvInstruction::Auipc { .. } => is_auipc = true,
            RiscvInstruction::Branch { cond, .. } => match cond {
                BranchCondition::Eq => is_beq = true,
                BranchCondition::Ne => is_bne = true,
                BranchCondition::Lt => is_blt = true,
                BranchCondition::Ge => is_bge = true,
                BranchCondition::Ltu => is_bltu = true,
                BranchCondition::Geu => is_bgeu = true,
            },
            RiscvInstruction::Jal { .. } => is_jal = true,
            RiscvInstruction::Jalr { .. } => is_jalr = true,
            RiscvInstruction::Halt => is_halt = true,
            _ => {}
        }

        // Reject unsupported instructions.
        if !(is_add
            || is_sub
            || is_sll
            || is_slt
            || is_sltu
            || is_xor
            || is_srl
            || is_sra
            || is_or
            || is_and
            || is_mul
            || is_mulh
            || is_mulhu
            || is_mulhsu
            || is_div
            || is_divu
            || is_rem
            || is_remu
            || is_addi
            || is_slti
            || is_sltiu
            || is_xori
            || is_ori
            || is_andi
            || is_slli
            || is_srli
            || is_srai
            || is_lw
            || is_sw
            || is_amoswap_w
            || is_amoadd_w
            || is_amoxor_w
            || is_amoor_w
            || is_amoand_w
            || is_lui
            || is_auipc
            || is_beq
            || is_bne
            || is_blt
            || is_bge
            || is_bltu
            || is_bgeu
            || is_jal
            || is_jalr
            || is_halt)
        {
            return Err(format!(
                "RV32 B1: unsupported instruction at pc={:#x}: word={:#x}",
                step.pc_before, instr_word_u32
            ));
        }

        z[layout.is_add(j)] = if is_add { F::ONE } else { F::ZERO };
        z[layout.is_sub(j)] = if is_sub { F::ONE } else { F::ZERO };
        z[layout.is_sll(j)] = if is_sll { F::ONE } else { F::ZERO };
        z[layout.is_slt(j)] = if is_slt { F::ONE } else { F::ZERO };
        z[layout.is_sltu(j)] = if is_sltu { F::ONE } else { F::ZERO };
        z[layout.is_xor(j)] = if is_xor { F::ONE } else { F::ZERO };
        z[layout.is_srl(j)] = if is_srl { F::ONE } else { F::ZERO };
        z[layout.is_sra(j)] = if is_sra { F::ONE } else { F::ZERO };
        z[layout.is_or(j)] = if is_or { F::ONE } else { F::ZERO };
        z[layout.is_and(j)] = if is_and { F::ONE } else { F::ZERO };
        z[layout.is_mul(j)] = if is_mul { F::ONE } else { F::ZERO };
        z[layout.is_mulh(j)] = if is_mulh { F::ONE } else { F::ZERO };
        z[layout.is_mulhu(j)] = if is_mulhu { F::ONE } else { F::ZERO };
        z[layout.is_mulhsu(j)] = if is_mulhsu { F::ONE } else { F::ZERO };
        z[layout.is_div(j)] = if is_div { F::ONE } else { F::ZERO };
        z[layout.is_divu(j)] = if is_divu { F::ONE } else { F::ZERO };
        z[layout.is_rem(j)] = if is_rem { F::ONE } else { F::ZERO };
        z[layout.is_remu(j)] = if is_remu { F::ONE } else { F::ZERO };
        z[layout.is_addi(j)] = if is_addi { F::ONE } else { F::ZERO };
        z[layout.is_slti(j)] = if is_slti { F::ONE } else { F::ZERO };
        z[layout.is_sltiu(j)] = if is_sltiu { F::ONE } else { F::ZERO };
        z[layout.is_xori(j)] = if is_xori { F::ONE } else { F::ZERO };
        z[layout.is_ori(j)] = if is_ori { F::ONE } else { F::ZERO };
        z[layout.is_andi(j)] = if is_andi { F::ONE } else { F::ZERO };
        z[layout.is_slli(j)] = if is_slli { F::ONE } else { F::ZERO };
        z[layout.is_srli(j)] = if is_srli { F::ONE } else { F::ZERO };
        z[layout.is_srai(j)] = if is_srai { F::ONE } else { F::ZERO };
        z[layout.is_lw(j)] = if is_lw { F::ONE } else { F::ZERO };
        z[layout.is_sw(j)] = if is_sw { F::ONE } else { F::ZERO };
        z[layout.is_amoswap_w(j)] = if is_amoswap_w { F::ONE } else { F::ZERO };
        z[layout.is_amoadd_w(j)] = if is_amoadd_w { F::ONE } else { F::ZERO };
        z[layout.is_amoxor_w(j)] = if is_amoxor_w { F::ONE } else { F::ZERO };
        z[layout.is_amoor_w(j)] = if is_amoor_w { F::ONE } else { F::ZERO };
        z[layout.is_amoand_w(j)] = if is_amoand_w { F::ONE } else { F::ZERO };
        z[layout.is_lui(j)] = if is_lui { F::ONE } else { F::ZERO };
        z[layout.is_auipc(j)] = if is_auipc { F::ONE } else { F::ZERO };
        z[layout.is_beq(j)] = if is_beq { F::ONE } else { F::ZERO };
        z[layout.is_bne(j)] = if is_bne { F::ONE } else { F::ZERO };
        z[layout.is_blt(j)] = if is_blt { F::ONE } else { F::ZERO };
        z[layout.is_bge(j)] = if is_bge { F::ONE } else { F::ZERO };
        z[layout.is_bltu(j)] = if is_bltu { F::ONE } else { F::ZERO };
        z[layout.is_bgeu(j)] = if is_bgeu { F::ONE } else { F::ZERO };
        z[layout.is_jal(j)] = if is_jal { F::ONE } else { F::ZERO };
        z[layout.is_jalr(j)] = if is_jalr { F::ONE } else { F::ZERO };
        z[layout.is_halt(j)] = if is_halt { F::ONE } else { F::ZERO };

        // One-hot register selectors.
        let rs1_idx = rs1 as usize;
        let rs2_idx = rs2 as usize;
        let rd_idx = rd as usize;
        z[layout.rs1_sel(rs1_idx, j)] = F::ONE;
        z[layout.rs2_sel(rs2_idx, j)] = F::ONE;

        // rd_sel: writes set rd_sel[rd] = 1, non-writes set rd_sel[0] = 1 (x0).
        let writes_rd = is_add
            || is_sub
            || is_sll
            || is_slt
            || is_sltu
            || is_xor
            || is_srl
            || is_sra
            || is_or
            || is_and
            || is_mul
            || is_mulh
            || is_mulhu
            || is_mulhsu
            || is_div
            || is_divu
            || is_rem
            || is_remu
            || is_addi
            || is_slti
            || is_sltiu
            || is_xori
            || is_ori
            || is_andi
            || is_slli
            || is_srli
            || is_srai
            || is_lw
            || is_amoswap_w
            || is_amoadd_w
            || is_amoxor_w
            || is_amoor_w
            || is_amoand_w
            || is_lui
            || is_auipc
            || is_jal
            || is_jalr;
        if writes_rd {
            z[layout.rd_sel(rd_idx, j)] = F::ONE;
        } else {
            z[layout.rd_sel(0, j)] = F::ONE;
        }

        // Selected operand values.
        z[layout.rs1_val(j)] = F::from_u64(step.regs_before[rs1_idx]);
        z[layout.rs2_val(j)] = F::from_u64(step.regs_before[rs2_idx]);

        // Shared-bus bound values: lookup_key / alu_out / mem_rv / eff_addr.
        z[layout.add_has_lookup(j)] = if is_add || is_addi || is_lw || is_sw || is_amoadd_w || is_auipc || is_jalr {
            F::ONE
        } else {
            F::ZERO
        };
        z[layout.and_has_lookup(j)] = if is_and || is_andi || is_amoand_w { F::ONE } else { F::ZERO };
        z[layout.xor_has_lookup(j)] = if is_xor || is_xori || is_amoxor_w { F::ONE } else { F::ZERO };
        z[layout.or_has_lookup(j)] = if is_or || is_ori || is_amoor_w { F::ONE } else { F::ZERO };
        z[layout.sll_has_lookup(j)] = if is_sll || is_slli { F::ONE } else { F::ZERO };
        z[layout.srl_has_lookup(j)] = if is_srl || is_srli { F::ONE } else { F::ZERO };
        z[layout.sra_has_lookup(j)] = if is_sra || is_srai { F::ONE } else { F::ZERO };
        z[layout.slt_has_lookup(j)] = if is_slt || is_slti || is_blt || is_bge { F::ONE } else { F::ZERO };
        z[layout.sltu_has_lookup(j)] = if is_sltu || is_sltiu || is_bltu || is_bgeu { F::ONE } else { F::ZERO };

        let is_amo = is_amoswap_w || is_amoadd_w || is_amoxor_w || is_amoor_w || is_amoand_w;
        z[layout.ram_has_read(j)] = if is_lw || is_amo { F::ONE } else { F::ZERO };
        z[layout.ram_has_write(j)] = if is_sw || is_amo { F::ONE } else { F::ZERO };

        // Default zeros.
        z[layout.lookup_key(j)] = F::ZERO;
        z[layout.alu_out(j)] = F::ZERO;
        z[layout.add_a0b0(j)] = F::ZERO;
        z[layout.mem_rv(j)] = F::ZERO;
        z[layout.eff_addr(j)] = F::ZERO;
        z[layout.ram_wv(j)] = F::ZERO;
        z[layout.rd_write_val(j)] = F::ZERO;
        z[layout.br_taken(j)] = F::ZERO;
        z[layout.br_not_taken(j)] = F::ZERO;

        // Shout events (this circuit exposes only one Shout port).
        let mut shout_events: Vec<(u32, u64, u64)> = Vec::new();
        for ev in &step.shout_events {
            let id = ev.shout_id.0;
            if layout.shout_idx(id).is_ok() {
                shout_events.push((id, ev.key, ev.value));
            } else {
                return Err(format!(
                    "RV32 B1: unsupported shout table id={id} at pc={:#x} (chunk j={j})",
                    step.pc_before
                ));
            }
        }
        if shout_events.len() > 1 {
            let ids: Vec<u32> = shout_events.iter().map(|(id, _, _)| *id).collect();
            return Err(format!(
                "RV32 B1: multiple shout events in one step (pc={:#x}, chunk j={j}): shout_ids={ids:?}; this circuit has 1 Shout port (no lanes) so you must either provision multiple Shout lanes in the shared CPU bus or split into micro-steps",
                step.pc_before
            ));
        }
        if let Some((_id, key, value)) = shout_events.pop() {
            z[layout.lookup_key(j)] = F::from_u64(key);
            z[layout.alu_out(j)] = F::from_u64(value);
            if z[layout.add_has_lookup(j)] == F::ONE {
                let a0 = key & 1;
                let b0 = (key >> 1) & 1;
                z[layout.add_a0b0(j)] = F::from_u64(a0 & b0);
            }
        }

        // RAM read/write events if present.
        for ev in &step.twist_events {
            if ev.twist_id == RAM_ID {
                match ev.kind {
                    TwistOpKind::Read => {
                        z[layout.eff_addr(j)] = F::from_u64(ev.addr);
                        z[layout.mem_rv(j)] = F::from_u64(ev.value);
                    }
                    TwistOpKind::Write => {
                        z[layout.eff_addr(j)] = F::from_u64(ev.addr);
                    }
                }
            }
        }

        // Writeback value.
        if is_add
            || is_sub
            || is_sll
            || is_slt
            || is_sltu
            || is_xor
            || is_srl
            || is_sra
            || is_or
            || is_and
            || is_mul
            || is_mulh
            || is_mulhsu
            || is_mulhu
            || is_div
            || is_divu
            || is_rem
            || is_remu
            || is_addi
            || is_slti
            || is_sltiu
            || is_xori
            || is_ori
            || is_andi
            || is_slli
            || is_srli
            || is_srai
            || is_auipc
        {
            z[layout.rd_write_val(j)] = z[layout.alu_out(j)];
        }
        if is_lw || is_amoswap_w || is_amoadd_w || is_amoxor_w || is_amoor_w || is_amoand_w {
            z[layout.rd_write_val(j)] = z[layout.mem_rv(j)];
        }
        if is_lui {
            z[layout.rd_write_val(j)] = z[layout.imm_u(j)];
        }
        if is_jal || is_jalr {
            z[layout.rd_write_val(j)] = F::from_u64(step.pc_before.wrapping_add(4));
        }
        if is_beq || is_bne || is_blt || is_bltu {
            z[layout.br_taken(j)] = z[layout.alu_out(j)];
            z[layout.br_not_taken(j)] = F::ONE - z[layout.br_taken(j)];
        }
        if is_bge || is_bgeu {
            z[layout.br_taken(j)] = F::ONE - z[layout.alu_out(j)];
            z[layout.br_not_taken(j)] = F::ONE - z[layout.br_taken(j)];
        }

        if is_sw || is_amoswap_w {
            z[layout.ram_wv(j)] = z[layout.rs2_val(j)];
        }
        if is_amoadd_w || is_amoxor_w || is_amoor_w || is_amoand_w {
            z[layout.ram_wv(j)] = z[layout.alu_out(j)];
        }
    }

    z[layout.pc_final] = F::from_u64(carried_pc);
    for r in 0..32 {
        z[layout.regs_final_start + r] = F::from_u64(carried_regs[r]);
    }
    z[layout.regs_final_start] = F::ZERO;

    // Chunk-level halting state used for cross-chunk padding semantics.
    z[layout.halted_in] = F::ONE - z[layout.is_active(0)];
    let j_last = layout.chunk_size - 1;
    z[layout.halted_out] = F::ONE - z[layout.is_active(j_last)] + z[layout.is_halt(j_last)];

    Ok(z)
}
