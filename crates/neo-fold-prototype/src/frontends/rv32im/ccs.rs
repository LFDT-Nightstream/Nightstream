//! Owns the RV32IM protocol ids plus the exact 27-field root main-lane CCS embedding.

use neo_ccs::CcsStructure;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::rv32im::isa::Rv32Opcode;
use crate::rv32im::lower::Rv32ExpandedRow;
use crate::vm::r1cs_builder::R1csBuilder;

pub const RV32IM_PARITY_TRANSCRIPT_APP_LABEL: &[u8] = b"neo.fold.next/rv32im/parity_kernel_v1";
pub const RV32IM_PARITY_TRANSCRIPT_SEED_LABEL: &[u8] = b"rv32im/kernel/transcript_seed";
pub const RV32IM_PARITY_CASE_NAME_LABEL: &[u8] = b"rv32im/kernel/case_name";
pub const RV32IM_PARITY_PROGRAM_WORDS_LABEL: &[u8] = b"rv32im/kernel/program_words";
pub const RV32IM_PARITY_INITIAL_REGS_LABEL: &[u8] = b"rv32im/kernel/initial_regs";
pub const RV32IM_PARITY_INITIAL_MEMORY_LABEL: &[u8] = b"rv32im/kernel/initial_memory";
pub const RV32IM_PARITY_ROOT0_DIGEST_LABEL: &[u8] = b"rv32im/kernel/root0_digest";
pub const RV32IM_PARITY_STAGE1_DIGEST_LABEL: &[u8] = b"rv32im/kernel/stage1_digest";
pub const RV32IM_PARITY_STAGE2_DIGEST_LABEL: &[u8] = b"rv32im/kernel/stage2_digest";
pub const RV32IM_PARITY_STAGE3_DIGEST_LABEL: &[u8] = b"rv32im/kernel/stage3_digest";
pub const RV32IM_PARITY_EXECUTION_DIGEST_LABEL: &[u8] = b"rv32im/kernel/execution_digest";
pub const RV32IM_PARITY_FINAL_STATE_DIGEST_LABEL: &[u8] = b"rv32im/kernel/final_state_digest";
pub const RV32IM_PARITY_STAGE1_MIX_LABEL: &[u8] = b"rv32im/stage1/row_mix";
pub const RV32IM_PARITY_STAGE2_REG_MIX_LABEL: &[u8] = b"rv32im/stage2/reg_mix";
pub const RV32IM_PARITY_STAGE2_RAM_MIX_LABEL: &[u8] = b"rv32im/stage2/ram_mix";
pub const RV32IM_PARITY_STAGE3_CONTINUITY_MIX_LABEL: &[u8] = b"rv32im/stage3/continuity_mix";
pub const RV32IM_PARITY_KERNEL_FINAL_MIX_LABEL: &[u8] = b"rv32im/kernel/final_mix";

pub const RV32IM_ROOT_ROW_WIDTH: usize = 27;
pub const RV32IM_ROOT_PUBLIC_INPUTS: usize = 1;

const COL_ONE: usize = 0;
const COL_PC: usize = 1;
const COL_PC_NEXT: usize = 2;
const COL_RS1: usize = 3;
const COL_RS2: usize = 4;
const COL_RD_NEXT: usize = 5;
const COL_IMM: usize = 6;
const COL_ALU_OUT: usize = 7;
const COL_STEP_PC: usize = 8;
const COL_JUMP_TARGET: usize = 9;
const COL_MEM_ADDR: usize = 10;
const COL_MEM_VAL: usize = 11;
const COL_RD_IDX: usize = 12;
const COL_RS1_IDX: usize = 13;
const COL_RS2_IDX: usize = 14;
const COL_WRITES_ALU_TO_RD: usize = 15;
const COL_WRITES_MEM_TO_RD: usize = 16;
const COL_PRESERVES_RD: usize = 17;
const COL_IS_JAL: usize = 18;
const COL_IS_JALR: usize = 19;
const COL_IS_BRANCH: usize = 20;
const COL_BRANCH_TAKEN: usize = 21;
const COL_BRANCH_TAKEN_MUX: usize = 22;
const COL_IS_LOAD: usize = 23;
const COL_IS_STORE: usize = 24;
const COL_USES_RS2: usize = 25;
const COL_ADVANCE_ARCH_PC: usize = 26;

fn encode_u32(row: &mut [F; RV32IM_ROOT_ROW_WIDTH], idx: usize, value: u32) {
    row[idx] = F::from_u64(u64::from(value));
}

fn bool_field(value: bool) -> F {
    if value {
        F::ONE
    } else {
        F::ZERO
    }
}

fn is_real_branch(opcode: Rv32Opcode) -> bool {
    matches!(
        opcode,
        Rv32Opcode::Beq | Rv32Opcode::Bne | Rv32Opcode::Blt | Rv32Opcode::Bge | Rv32Opcode::Bltu | Rv32Opcode::Bgeu
    )
}

fn is_real_load(opcode: Rv32Opcode) -> bool {
    matches!(
        opcode,
        Rv32Opcode::Lb | Rv32Opcode::Lbu | Rv32Opcode::Lh | Rv32Opcode::Lhu | Rv32Opcode::Lw
    )
}

fn is_real_store(opcode: Rv32Opcode) -> bool {
    matches!(opcode, Rv32Opcode::Sb | Rv32Opcode::Sh | Rv32Opcode::Sw)
}

fn real_opcode_uses_rs2(opcode: Rv32Opcode) -> bool {
    matches!(
        opcode,
        Rv32Opcode::Add
            | Rv32Opcode::Sub
            | Rv32Opcode::And
            | Rv32Opcode::Or
            | Rv32Opcode::Xor
            | Rv32Opcode::Slt
            | Rv32Opcode::Sltu
            | Rv32Opcode::Sll
            | Rv32Opcode::Srl
            | Rv32Opcode::Sra
            | Rv32Opcode::Mul
            | Rv32Opcode::Mulh
            | Rv32Opcode::Mulhsu
            | Rv32Opcode::Mulhu
            | Rv32Opcode::Div
            | Rv32Opcode::Divu
            | Rv32Opcode::Rem
            | Rv32Opcode::Remu
            | Rv32Opcode::Sb
            | Rv32Opcode::Sh
            | Rv32Opcode::Sw
            | Rv32Opcode::Beq
            | Rv32Opcode::Bne
            | Rv32Opcode::Blt
            | Rv32Opcode::Bge
            | Rv32Opcode::Bltu
            | Rv32Opcode::Bgeu
    )
}

fn narrow_store_value(opcode: Rv32Opcode, rs2_value: u32) -> u32 {
    match opcode {
        Rv32Opcode::Sb => rs2_value & 0xff,
        Rv32Opcode::Sh => rs2_value & 0xffff,
        Rv32Opcode::Sw => rs2_value,
        _ => 0,
    }
}

fn sign_extend_bits(raw: u32, bits: u32) -> u32 {
    (((raw << (32 - bits)) as i32) >> (32 - bits)) as u32
}

fn narrow_load_value(row: &Rv32ExpandedRow, opcode: Rv32Opcode) -> u32 {
    let value = row.memory_before.unwrap_or(0);
    let addr = row.effective_addr.unwrap_or(0);
    let byte_offset = addr & 0x3;
    match opcode {
        Rv32Opcode::Lb => sign_extend_bits((value >> (byte_offset * 8)) & 0xff, 8),
        Rv32Opcode::Lbu => (value >> (byte_offset * 8)) & 0xff,
        Rv32Opcode::Lh => sign_extend_bits((value >> (byte_offset * 8)) & 0xffff, 16),
        Rv32Opcode::Lhu => (value >> (byte_offset * 8)) & 0xffff,
        Rv32Opcode::Lw => value,
        _ => row.rd_after,
    }
}

fn memory_transfer_value(row: &Rv32ExpandedRow, opcode: Option<Rv32Opcode>) -> u32 {
    match opcode {
        Some(real) if is_real_load(real) => narrow_load_value(row, real),
        Some(real) if is_real_store(real) => narrow_store_value(real, row.rs2_value),
        _ => 0,
    }
}

pub fn semantic_row_from_execution_row(row: &Rv32ExpandedRow) -> [F; RV32IM_ROOT_ROW_WIDTH] {
    let mut out = [F::ZERO; RV32IM_ROOT_ROW_WIDTH];
    let real_opcode = row.trace_opcode;
    let is_load = real_opcode.is_some_and(is_real_load);
    let is_store = real_opcode.is_some_and(is_real_store);
    let writes_mem_to_rd = is_load && row.writes_rd;
    let writes_alu_to_rd = row.writes_rd && !writes_mem_to_rd;
    let preserves_rd = !writes_alu_to_rd && !writes_mem_to_rd;
    let is_jal = real_opcode == Some(Rv32Opcode::Jal);
    let is_jalr = real_opcode == Some(Rv32Opcode::Jalr);
    let is_branch = real_opcode.is_some_and(is_real_branch);
    let step_pc = row.pc.wrapping_add(4);
    let branch_taken = is_branch && row.next_pc != step_pc;
    let branch_taken_mux = is_branch && branch_taken;
    let jump_target = if is_jal || is_jalr || branch_taken_mux {
        row.next_pc
    } else {
        0
    };
    let mem_addr = if is_load || is_store {
        row.effective_addr.unwrap_or(0)
    } else {
        0
    };
    let mem_val = memory_transfer_value(row, real_opcode);
    let rd_next = if writes_alu_to_rd || writes_mem_to_rd {
        row.rd_after
    } else {
        0
    };
    let uses_rs2 = match real_opcode {
        Some(opcode) => real_opcode_uses_rs2(opcode),
        None => row.rs2 != 0,
    };

    out[COL_ONE] = F::ONE;
    encode_u32(&mut out, COL_PC, row.pc);
    encode_u32(&mut out, COL_PC_NEXT, row.next_pc);
    encode_u32(&mut out, COL_RS1, row.rs1_value);
    encode_u32(&mut out, COL_RS2, row.rs2_value);
    encode_u32(&mut out, COL_RD_NEXT, rd_next);
    encode_u32(&mut out, COL_IMM, row.imm as u32);
    encode_u32(&mut out, COL_ALU_OUT, row.alu_result);
    encode_u32(&mut out, COL_STEP_PC, step_pc);
    encode_u32(&mut out, COL_JUMP_TARGET, jump_target);
    encode_u32(&mut out, COL_MEM_ADDR, mem_addr);
    encode_u32(&mut out, COL_MEM_VAL, mem_val);
    out[COL_RD_IDX] = F::from_u64(row.rd as u64);
    out[COL_RS1_IDX] = F::from_u64(row.rs1 as u64);
    out[COL_RS2_IDX] = F::from_u64(row.rs2 as u64);
    out[COL_WRITES_ALU_TO_RD] = bool_field(writes_alu_to_rd);
    out[COL_WRITES_MEM_TO_RD] = bool_field(writes_mem_to_rd);
    out[COL_PRESERVES_RD] = bool_field(preserves_rd);
    out[COL_IS_JAL] = bool_field(is_jal);
    out[COL_IS_JALR] = bool_field(is_jalr);
    out[COL_IS_BRANCH] = bool_field(is_branch);
    out[COL_BRANCH_TAKEN] = bool_field(branch_taken);
    out[COL_BRANCH_TAKEN_MUX] = bool_field(branch_taken_mux);
    out[COL_IS_LOAD] = bool_field(is_load);
    out[COL_IS_STORE] = bool_field(is_store);
    out[COL_USES_RS2] = bool_field(uses_rs2);
    out[COL_ADVANCE_ARCH_PC] = bool_field(row.is_commit_row);
    out
}

pub fn rv32im_root_main_lane_ccs() -> Result<CcsStructure<F>, String> {
    let mut b = R1csBuilder::new(RV32IM_ROOT_ROW_WIDTH, COL_ONE)?;

    for &col in &[
        COL_WRITES_ALU_TO_RD,
        COL_WRITES_MEM_TO_RD,
        COL_PRESERVES_RD,
        COL_IS_JAL,
        COL_IS_JALR,
        COL_IS_BRANCH,
        COL_BRANCH_TAKEN,
        COL_IS_LOAD,
        COL_IS_STORE,
        COL_USES_RS2,
        COL_ADVANCE_ARCH_PC,
    ] {
        b.push_boolean(col);
    }

    b.push_row(
        [(COL_IS_BRANCH, F::ONE)],
        [(COL_BRANCH_TAKEN, F::ONE)],
        [(COL_BRANCH_TAKEN_MUX, F::ONE)],
    );
    b.push_linear_zero(
        [
            (COL_WRITES_ALU_TO_RD, F::ONE),
            (COL_WRITES_MEM_TO_RD, F::ONE),
            (COL_PRESERVES_RD, F::ONE),
            (COL_ONE, -F::ONE),
        ]
        .into_iter(),
    );
    b.push_row(
        [(COL_WRITES_ALU_TO_RD, F::ONE)],
        [(COL_RD_NEXT, F::ONE), (COL_ALU_OUT, -F::ONE)],
        [],
    );
    b.push_row(
        [(COL_WRITES_MEM_TO_RD, F::ONE)],
        [(COL_RD_NEXT, F::ONE), (COL_MEM_VAL, -F::ONE)],
        [],
    );
    b.push_row([(COL_PRESERVES_RD, F::ONE)], [(COL_RD_NEXT, F::ONE)], []);
    b.push_row(
        [
            (COL_IS_JAL, F::ONE),
            (COL_IS_JALR, F::ONE),
            (COL_BRANCH_TAKEN_MUX, F::ONE),
        ],
        [(COL_PC_NEXT, F::ONE), (COL_JUMP_TARGET, -F::ONE)],
        [],
    );
    b.push_row(
        [
            (COL_ADVANCE_ARCH_PC, F::ONE),
            (COL_IS_JAL, -F::ONE),
            (COL_IS_JALR, -F::ONE),
            (COL_BRANCH_TAKEN_MUX, -F::ONE),
        ],
        [(COL_PC_NEXT, F::ONE), (COL_STEP_PC, -F::ONE)],
        [],
    );
    b.push_row(
        [(COL_ONE, F::ONE), (COL_ADVANCE_ARCH_PC, -F::ONE)],
        [(COL_PC_NEXT, F::ONE), (COL_PC, -F::ONE)],
        [],
    );
    b.push_row(
        [(COL_ONE, F::ONE), (COL_IS_LOAD, -F::ONE), (COL_IS_STORE, -F::ONE)],
        [(COL_MEM_ADDR, F::ONE)],
        [],
    );
    b.push_row(
        [(COL_ONE, F::ONE), (COL_IS_LOAD, -F::ONE), (COL_IS_STORE, -F::ONE)],
        [(COL_MEM_VAL, F::ONE)],
        [],
    );

    Ok(b.build()?)
}
