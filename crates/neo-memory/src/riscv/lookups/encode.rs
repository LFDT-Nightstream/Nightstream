use super::isa::{BranchCondition, RiscvInstruction, RiscvMemOp, RiscvOpcode};

/// Assemble a single RISC-V instruction to its 32-bit encoding.
///
/// This is the inverse of `decode_instruction`.
pub fn encode_instruction(instr: &RiscvInstruction) -> u32 {
    match instr {
        RiscvInstruction::RAlu { op, rd, rs1, rs2 } => {
            let (funct3, funct7) = match op {
                RiscvOpcode::Add => (0b000, 0b0000000),
                RiscvOpcode::Sub => (0b000, 0b0100000),
                RiscvOpcode::Sll => (0b001, 0b0000000),
                RiscvOpcode::Slt => (0b010, 0b0000000),
                RiscvOpcode::Sltu => (0b011, 0b0000000),
                RiscvOpcode::Xor => (0b100, 0b0000000),
                RiscvOpcode::Srl => (0b101, 0b0000000),
                RiscvOpcode::Sra => (0b101, 0b0100000),
                RiscvOpcode::Or => (0b110, 0b0000000),
                RiscvOpcode::And => (0b111, 0b0000000),
                RiscvOpcode::Mul => (0b000, 0b0000001),
                RiscvOpcode::Mulh => (0b001, 0b0000001),
                RiscvOpcode::Mulhsu => (0b010, 0b0000001),
                RiscvOpcode::Mulhu => (0b011, 0b0000001),
                RiscvOpcode::Div => (0b100, 0b0000001),
                RiscvOpcode::Divu => (0b101, 0b0000001),
                RiscvOpcode::Rem => (0b110, 0b0000001),
                RiscvOpcode::Remu => (0b111, 0b0000001),
                _ => (0, 0), // Not R-type
            };
            0b0110011
                | ((*rd as u32) << 7)
                | (funct3 << 12)
                | ((*rs1 as u32) << 15)
                | ((*rs2 as u32) << 20)
                | (funct7 << 25)
        }

        RiscvInstruction::IAlu { op, rd, rs1, imm } => {
            let funct3 = match op {
                RiscvOpcode::Add => 0b000,
                RiscvOpcode::Slt => 0b010,
                RiscvOpcode::Sltu => 0b011,
                RiscvOpcode::Xor => 0b100,
                RiscvOpcode::Or => 0b110,
                RiscvOpcode::And => 0b111,
                RiscvOpcode::Sll => 0b001,
                RiscvOpcode::Srl => 0b101,
                RiscvOpcode::Sra => 0b101,
                _ => 0,
            };
            let imm_bits = (*imm as u32) & 0xFFF;
            // For SRA, set the special bit
            let imm_bits = if *op == RiscvOpcode::Sra {
                imm_bits | 0x400
            } else {
                imm_bits
            };
            0b0010011 | ((*rd as u32) << 7) | (funct3 << 12) | ((*rs1 as u32) << 15) | (imm_bits << 20)
        }

        RiscvInstruction::Load { op, rd, rs1, imm } => {
            let funct3 = match op {
                RiscvMemOp::Lb => 0b000,
                RiscvMemOp::Lh => 0b001,
                RiscvMemOp::Lw => 0b010,
                RiscvMemOp::Ld => 0b011,
                RiscvMemOp::Lbu => 0b100,
                RiscvMemOp::Lhu => 0b101,
                RiscvMemOp::Lwu => 0b110,
                _ => 0,
            };
            let imm_bits = (*imm as u32) & 0xFFF;
            0b0000011 | ((*rd as u32) << 7) | (funct3 << 12) | ((*rs1 as u32) << 15) | (imm_bits << 20)
        }

        RiscvInstruction::Store { op, rs1, rs2, imm } => {
            let funct3 = match op {
                RiscvMemOp::Sb => 0b000,
                RiscvMemOp::Sh => 0b001,
                RiscvMemOp::Sw => 0b010,
                RiscvMemOp::Sd => 0b011,
                _ => 0,
            };
            let imm_bits = *imm as u32;
            let imm_4_0 = imm_bits & 0x1F;
            let imm_11_5 = (imm_bits >> 5) & 0x7F;
            0b0100011
                | (imm_4_0 << 7)
                | (funct3 << 12)
                | ((*rs1 as u32) << 15)
                | ((*rs2 as u32) << 20)
                | (imm_11_5 << 25)
        }

        RiscvInstruction::Branch { cond, rs1, rs2, imm } => {
            let funct3 = match cond {
                BranchCondition::Eq => 0b000,
                BranchCondition::Ne => 0b001,
                BranchCondition::Lt => 0b100,
                BranchCondition::Ge => 0b101,
                BranchCondition::Ltu => 0b110,
                BranchCondition::Geu => 0b111,
            };
            let imm_bits = *imm as u32;
            let imm_11 = (imm_bits >> 11) & 1;
            let imm_4_1 = (imm_bits >> 1) & 0xF;
            let imm_10_5 = (imm_bits >> 5) & 0x3F;
            let imm_12 = (imm_bits >> 12) & 1;
            0b1100011
                | (imm_11 << 7)
                | (imm_4_1 << 8)
                | (funct3 << 12)
                | ((*rs1 as u32) << 15)
                | ((*rs2 as u32) << 20)
                | (imm_10_5 << 25)
                | (imm_12 << 31)
        }

        RiscvInstruction::Jal { rd, imm } => {
            let imm_bits = *imm as u32;
            let imm_20 = (imm_bits >> 20) & 1;
            let imm_10_1 = (imm_bits >> 1) & 0x3FF;
            let imm_11 = (imm_bits >> 11) & 1;
            let imm_19_12 = (imm_bits >> 12) & 0xFF;
            0b1101111 | ((*rd as u32) << 7) | (imm_19_12 << 12) | (imm_11 << 20) | (imm_10_1 << 21) | (imm_20 << 31)
        }

        RiscvInstruction::Jalr { rd, rs1, imm } => {
            let imm_bits = (*imm as u32) & 0xFFF;
            0b1100111 | ((*rd as u32) << 7) | (0b000 << 12) | ((*rs1 as u32) << 15) | (imm_bits << 20)
        }

        RiscvInstruction::Lui { rd, imm } => {
            let imm_bits = (*imm as u32) & 0xFFFFF;
            0b0110111 | ((*rd as u32) << 7) | (imm_bits << 12)
        }

        RiscvInstruction::Auipc { rd, imm } => {
            let imm_bits = (*imm as u32) & 0xFFFFF;
            0b0010111 | ((*rd as u32) << 7) | (imm_bits << 12)
        }

        RiscvInstruction::Halt => {
            // ECALL
            0b1110011
        }

        RiscvInstruction::Nop => {
            // ADDI x0, x0, 0
            0b0010011
        }

        // === RV64 W-suffix Operations ===
        RiscvInstruction::RAluw { op, rd, rs1, rs2 } => {
            let (funct3, funct7) = match op {
                RiscvOpcode::Addw => (0b000, 0b0000000),
                RiscvOpcode::Subw => (0b000, 0b0100000),
                RiscvOpcode::Sllw => (0b001, 0b0000000),
                RiscvOpcode::Srlw => (0b101, 0b0000000),
                RiscvOpcode::Sraw => (0b101, 0b0100000),
                RiscvOpcode::Mulw => (0b000, 0b0000001),
                RiscvOpcode::Divw => (0b100, 0b0000001),
                RiscvOpcode::Divuw => (0b101, 0b0000001),
                RiscvOpcode::Remw => (0b110, 0b0000001),
                RiscvOpcode::Remuw => (0b111, 0b0000001),
                _ => (0, 0),
            };
            0b0111011
                | ((*rd as u32) << 7)
                | (funct3 << 12)
                | ((*rs1 as u32) << 15)
                | ((*rs2 as u32) << 20)
                | (funct7 << 25)
        }

        RiscvInstruction::IAluw { op, rd, rs1, imm } => {
            let funct3 = match op {
                RiscvOpcode::Addw => 0b000,
                RiscvOpcode::Sllw => 0b001,
                RiscvOpcode::Srlw => 0b101,
                RiscvOpcode::Sraw => 0b101,
                _ => 0,
            };
            let imm_bits = (*imm as u32) & 0xFFF;
            let imm_bits = if *op == RiscvOpcode::Sraw {
                imm_bits | 0x400
            } else {
                imm_bits
            };
            0b0011011 | ((*rd as u32) << 7) | (funct3 << 12) | ((*rs1 as u32) << 15) | (imm_bits << 20)
        }

        // === A Extension: Atomics ===
        RiscvInstruction::LoadReserved { op, rd, rs1 } => {
            let (funct3, funct5) = match op {
                RiscvMemOp::LrW => (0b010, 0b00010),
                RiscvMemOp::LrD => (0b011, 0b00010),
                _ => (0, 0),
            };
            // AMO format: funct5 | aq | rl | rs2 | rs1 | funct3 | rd | opcode
            0b0101111 | ((*rd as u32) << 7) | (funct3 << 12) | ((*rs1 as u32) << 15) | (0 << 20) | (funct5 << 27)
        }

        RiscvInstruction::StoreConditional { op, rd, rs1, rs2 } => {
            let (funct3, funct5) = match op {
                RiscvMemOp::ScW => (0b010, 0b00011),
                RiscvMemOp::ScD => (0b011, 0b00011),
                _ => (0, 0),
            };
            0b0101111
                | ((*rd as u32) << 7)
                | (funct3 << 12)
                | ((*rs1 as u32) << 15)
                | ((*rs2 as u32) << 20)
                | (funct5 << 27)
        }

        RiscvInstruction::Amo { op, rd, rs1, rs2 } => {
            let (funct3, funct5) = match op {
                RiscvMemOp::AmoswapW => (0b010, 0b00001),
                RiscvMemOp::AmoswapD => (0b011, 0b00001),
                RiscvMemOp::AmoaddW => (0b010, 0b00000),
                RiscvMemOp::AmoaddD => (0b011, 0b00000),
                RiscvMemOp::AmoxorW => (0b010, 0b00100),
                RiscvMemOp::AmoxorD => (0b011, 0b00100),
                RiscvMemOp::AmoandW => (0b010, 0b01100),
                RiscvMemOp::AmoandD => (0b011, 0b01100),
                RiscvMemOp::AmoorW => (0b010, 0b01000),
                RiscvMemOp::AmoorD => (0b011, 0b01000),
                RiscvMemOp::AmominW => (0b010, 0b10000),
                RiscvMemOp::AmominD => (0b011, 0b10000),
                RiscvMemOp::AmomaxW => (0b010, 0b10100),
                RiscvMemOp::AmomaxD => (0b011, 0b10100),
                RiscvMemOp::AmominuW => (0b010, 0b11000),
                RiscvMemOp::AmominuD => (0b011, 0b11000),
                RiscvMemOp::AmomaxuW => (0b010, 0b11100),
                RiscvMemOp::AmomaxuD => (0b011, 0b11100),
                _ => (0, 0),
            };
            0b0101111
                | ((*rd as u32) << 7)
                | (funct3 << 12)
                | ((*rs1 as u32) << 15)
                | ((*rs2 as u32) << 20)
                | (funct5 << 27)
        }

        // === System Instructions ===
        RiscvInstruction::Ecall => {
            // ECALL: imm=0
            0b1110011
        }

        RiscvInstruction::Ebreak => {
            // EBREAK: imm=1
            0b1110011 | (1 << 20)
        }

        RiscvInstruction::Fence { pred, succ } => {
            // FENCE: funct3=0, imm encodes pred/succ
            let imm = ((*pred as u32) << 4) | (*succ as u32);
            0b0001111 | (imm << 20)
        }

        RiscvInstruction::FenceI => {
            // FENCE.I: funct3=1
            0b0001111 | (0b001 << 12)
        }
    }
}

/// Assemble a program to bytes.
pub fn encode_program(instructions: &[RiscvInstruction]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(instructions.len() * 4);
    for instr in instructions {
        let encoded = encode_instruction(instr);
        bytes.extend_from_slice(&encoded.to_le_bytes());
    }
    bytes
}
