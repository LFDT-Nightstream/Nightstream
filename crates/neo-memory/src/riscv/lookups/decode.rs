use super::isa::{BranchCondition, RiscvInstruction, RiscvMemOp, RiscvOpcode};

/// RISC-V instruction format types.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RiscvFormat {
    /// R-type: register-register operations
    R,
    /// I-type: immediate operations, loads, JALR
    I,
    /// S-type: stores
    S,
    /// B-type: branches
    B,
    /// U-type: LUI, AUIPC
    U,
    /// J-type: JAL
    J,
}

/// Decode a 32-bit RISC-V instruction into our RiscvInstruction enum.
///
/// Supports RV32I/RV64I base integer instruction set and M extension.
///
/// # Arguments
/// * `instr` - The 32-bit instruction word
///
/// # Returns
/// * `Ok(RiscvInstruction)` - Decoded instruction
/// * `Err(String)` - Decoding error with description
///
/// # Example
/// ```ignore
/// // ADDI x1, x0, 42  (x1 = 0 + 42)
/// let instr = 0x02a00093;
/// let decoded = decode_instruction(instr)?;
/// assert!(matches!(decoded, RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 1, rs1: 0, imm: 42 }));
/// ```
pub fn decode_instruction(instr: u32) -> Result<RiscvInstruction, String> {
    // Extract common fields
    let opcode = instr & 0x7F;
    let rd = ((instr >> 7) & 0x1F) as u8;
    let funct3 = (instr >> 12) & 0x7;
    let rs1 = ((instr >> 15) & 0x1F) as u8;
    let rs2 = ((instr >> 20) & 0x1F) as u8;
    let funct7 = (instr >> 25) & 0x7F;

    match opcode {
        // R-type: OP (0110011)
        0b0110011 => {
            let op = match (funct3, funct7) {
                (0b000, 0b0000000) => RiscvOpcode::Add,
                (0b000, 0b0100000) => RiscvOpcode::Sub,
                (0b001, 0b0000000) => RiscvOpcode::Sll,
                (0b010, 0b0000000) => RiscvOpcode::Slt,
                (0b011, 0b0000000) => RiscvOpcode::Sltu,
                (0b100, 0b0000000) => RiscvOpcode::Xor,
                (0b101, 0b0000000) => RiscvOpcode::Srl,
                (0b101, 0b0100000) => RiscvOpcode::Sra,
                (0b110, 0b0000000) => RiscvOpcode::Or,
                (0b111, 0b0000000) => RiscvOpcode::And,
                // M extension
                (0b000, 0b0000001) => RiscvOpcode::Mul,
                (0b001, 0b0000001) => RiscvOpcode::Mulh,
                (0b010, 0b0000001) => RiscvOpcode::Mulhsu,
                (0b011, 0b0000001) => RiscvOpcode::Mulhu,
                (0b100, 0b0000001) => RiscvOpcode::Div,
                (0b101, 0b0000001) => RiscvOpcode::Divu,
                (0b110, 0b0000001) => RiscvOpcode::Rem,
                (0b111, 0b0000001) => RiscvOpcode::Remu,
                _ => return Err(format!("Unknown R-type: funct3={:#x}, funct7={:#x}", funct3, funct7)),
            };
            Ok(RiscvInstruction::RAlu { op, rd, rs1, rs2 })
        }

        // I-type: OP-IMM (0010011)
        0b0010011 => {
            let imm = sign_extend_i_imm(instr);
            let op = match funct3 {
                0b000 => RiscvOpcode::Add,  // ADDI
                0b010 => RiscvOpcode::Slt,  // SLTI
                0b011 => RiscvOpcode::Sltu, // SLTIU
                0b100 => RiscvOpcode::Xor,  // XORI
                0b110 => RiscvOpcode::Or,   // ORI
                0b111 => RiscvOpcode::And,  // ANDI
                0b001 => {
                    // SLLI
                    RiscvOpcode::Sll
                }
                0b101 => {
                    // SRLI or SRAI
                    let shamt_funct = (instr >> 26) & 0x3F;
                    if shamt_funct == 0b010000 {
                        RiscvOpcode::Sra
                    } else {
                        RiscvOpcode::Srl
                    }
                }
                _ => return Err(format!("Unknown I-type OP-IMM: funct3={:#x}", funct3)),
            };
            // For shifts, extract shamt properly
            let imm = if funct3 == 0b001 || funct3 == 0b101 {
                (instr >> 20) & 0x3F // shamt for shifts
            } else {
                imm as u32
            };
            Ok(RiscvInstruction::IAlu {
                op,
                rd,
                rs1,
                imm: imm as i32,
            })
        }

        // Load (0000011)
        0b0000011 => {
            let imm = sign_extend_i_imm(instr);
            let op = match funct3 {
                0b000 => RiscvMemOp::Lb,
                0b001 => RiscvMemOp::Lh,
                0b010 => RiscvMemOp::Lw,
                0b100 => RiscvMemOp::Lbu,
                0b101 => RiscvMemOp::Lhu,
                0b011 => RiscvMemOp::Ld,  // RV64
                0b110 => RiscvMemOp::Lwu, // RV64 (LWU)
                _ => return Err(format!("Unknown load: funct3={:#x}", funct3)),
            };
            Ok(RiscvInstruction::Load { op, rd, rs1, imm })
        }

        // Store (0100011)
        0b0100011 => {
            let imm = sign_extend_s_imm(instr);
            let op = match funct3 {
                0b000 => RiscvMemOp::Sb,
                0b001 => RiscvMemOp::Sh,
                0b010 => RiscvMemOp::Sw,
                0b011 => RiscvMemOp::Sd, // RV64
                _ => return Err(format!("Unknown store: funct3={:#x}", funct3)),
            };
            Ok(RiscvInstruction::Store { op, rs1, rs2, imm })
        }

        // Branch (1100011)
        0b1100011 => {
            let imm = sign_extend_b_imm(instr);
            let cond = match funct3 {
                0b000 => BranchCondition::Eq,
                0b001 => BranchCondition::Ne,
                0b100 => BranchCondition::Lt,
                0b101 => BranchCondition::Ge,
                0b110 => BranchCondition::Ltu,
                0b111 => BranchCondition::Geu,
                _ => return Err(format!("Unknown branch: funct3={:#x}", funct3)),
            };
            Ok(RiscvInstruction::Branch { cond, rs1, rs2, imm })
        }

        // JAL (1101111)
        0b1101111 => {
            let imm = sign_extend_j_imm(instr);
            Ok(RiscvInstruction::Jal { rd, imm })
        }

        // JALR (1100111)
        0b1100111 => {
            let imm = sign_extend_i_imm(instr);
            Ok(RiscvInstruction::Jalr { rd, rs1, imm })
        }

        // LUI (0110111)
        0b0110111 => {
            let imm = (instr >> 12) as i32;
            Ok(RiscvInstruction::Lui { rd, imm })
        }

        // AUIPC (0010111)
        0b0010111 => {
            let imm = (instr >> 12) as i32;
            Ok(RiscvInstruction::Auipc { rd, imm })
        }

        // SYSTEM (1110011) - ECALL, EBREAK
        0b1110011 => {
            let imm = (instr >> 20) & 0xFFF;
            match imm {
                // Note: We use Halt for ECALL in our simplified model.
                // Real RISC-V would trap to the OS/hypervisor.
                // Our step() function checks if a0==0 for ECALL and halts,
                // otherwise continues. For testing, we use Halt to unconditionally halt.
                0 => Ok(RiscvInstruction::Halt), // ECALL -> Halt for our test programs
                1 => Ok(RiscvInstruction::Ebreak),
                _ => Err(format!("Unknown SYSTEM instruction: imm={:#x}", imm)),
            }
        }

        // MISC-MEM (0001111) - FENCE, FENCE.I
        0b0001111 => {
            if funct3 == 0b001 {
                Ok(RiscvInstruction::FenceI)
            } else {
                let pred = ((instr >> 24) & 0xF) as u8;
                let succ = ((instr >> 20) & 0xF) as u8;
                Ok(RiscvInstruction::Fence { pred, succ })
            }
        }

        // OP-32 (0111011) - RV64 W-suffix R-type operations
        0b0111011 => {
            let op = match (funct3, funct7) {
                (0b000, 0b0000000) => RiscvOpcode::Addw,
                (0b000, 0b0100000) => RiscvOpcode::Subw,
                (0b001, 0b0000000) => RiscvOpcode::Sllw,
                (0b101, 0b0000000) => RiscvOpcode::Srlw,
                (0b101, 0b0100000) => RiscvOpcode::Sraw,
                // M extension W-suffix
                (0b000, 0b0000001) => RiscvOpcode::Mulw,
                (0b100, 0b0000001) => RiscvOpcode::Divw,
                (0b101, 0b0000001) => RiscvOpcode::Divuw,
                (0b110, 0b0000001) => RiscvOpcode::Remw,
                (0b111, 0b0000001) => RiscvOpcode::Remuw,
                _ => return Err(format!("Unknown OP-32: funct3={:#x}, funct7={:#x}", funct3, funct7)),
            };
            Ok(RiscvInstruction::RAluw { op, rd, rs1, rs2 })
        }

        // OP-IMM-32 (0011011) - RV64 W-suffix I-type operations
        0b0011011 => {
            let op = match funct3 {
                0b000 => RiscvOpcode::Addw, // ADDIW
                0b001 => RiscvOpcode::Sllw, // SLLIW
                0b101 => {
                    let shamt_funct = (instr >> 25) & 0x7F;
                    if shamt_funct == 0b0100000 {
                        RiscvOpcode::Sraw // SRAIW
                    } else {
                        RiscvOpcode::Srlw // SRLIW
                    }
                }
                _ => return Err(format!("Unknown OP-IMM-32: funct3={:#x}", funct3)),
            };
            let imm = if funct3 == 0b001 || funct3 == 0b101 {
                ((instr >> 20) & 0x1F) as i32 // shamt for W shifts (5 bits)
            } else {
                sign_extend_i_imm(instr)
            };
            Ok(RiscvInstruction::IAluw { op, rd, rs1, imm })
        }

        // AMO (0101111) - Atomic Memory Operations
        0b0101111 => {
            let funct5 = (instr >> 27) & 0x1F;
            match funct5 {
                // LR
                0b00010 => {
                    let op = if funct3 == 0b010 {
                        RiscvMemOp::LrW
                    } else {
                        RiscvMemOp::LrD
                    };
                    Ok(RiscvInstruction::LoadReserved { op, rd, rs1 })
                }
                // SC
                0b00011 => {
                    let op = if funct3 == 0b010 {
                        RiscvMemOp::ScW
                    } else {
                        RiscvMemOp::ScD
                    };
                    Ok(RiscvInstruction::StoreConditional { op, rd, rs1, rs2 })
                }
                // AMOSWAP
                0b00001 => {
                    let op = if funct3 == 0b010 {
                        RiscvMemOp::AmoswapW
                    } else {
                        RiscvMemOp::AmoswapD
                    };
                    Ok(RiscvInstruction::Amo { op, rd, rs1, rs2 })
                }
                // AMOADD
                0b00000 => {
                    let op = if funct3 == 0b010 {
                        RiscvMemOp::AmoaddW
                    } else {
                        RiscvMemOp::AmoaddD
                    };
                    Ok(RiscvInstruction::Amo { op, rd, rs1, rs2 })
                }
                // AMOXOR
                0b00100 => {
                    let op = if funct3 == 0b010 {
                        RiscvMemOp::AmoxorW
                    } else {
                        RiscvMemOp::AmoxorD
                    };
                    Ok(RiscvInstruction::Amo { op, rd, rs1, rs2 })
                }
                // AMOAND
                0b01100 => {
                    let op = if funct3 == 0b010 {
                        RiscvMemOp::AmoandW
                    } else {
                        RiscvMemOp::AmoandD
                    };
                    Ok(RiscvInstruction::Amo { op, rd, rs1, rs2 })
                }
                // AMOOR
                0b01000 => {
                    let op = if funct3 == 0b010 {
                        RiscvMemOp::AmoorW
                    } else {
                        RiscvMemOp::AmoorD
                    };
                    Ok(RiscvInstruction::Amo { op, rd, rs1, rs2 })
                }
                // AMOMIN
                0b10000 => {
                    let op = if funct3 == 0b010 {
                        RiscvMemOp::AmominW
                    } else {
                        RiscvMemOp::AmominD
                    };
                    Ok(RiscvInstruction::Amo { op, rd, rs1, rs2 })
                }
                // AMOMAX
                0b10100 => {
                    let op = if funct3 == 0b010 {
                        RiscvMemOp::AmomaxW
                    } else {
                        RiscvMemOp::AmomaxD
                    };
                    Ok(RiscvInstruction::Amo { op, rd, rs1, rs2 })
                }
                // AMOMINU
                0b11000 => {
                    let op = if funct3 == 0b010 {
                        RiscvMemOp::AmominuW
                    } else {
                        RiscvMemOp::AmominuD
                    };
                    Ok(RiscvInstruction::Amo { op, rd, rs1, rs2 })
                }
                // AMOMAXU
                0b11100 => {
                    let op = if funct3 == 0b010 {
                        RiscvMemOp::AmomaxuW
                    } else {
                        RiscvMemOp::AmomaxuD
                    };
                    Ok(RiscvInstruction::Amo { op, rd, rs1, rs2 })
                }
                _ => Err(format!("Unknown AMO: funct5={:#x}", funct5)),
            }
        }

        _ => Err(format!("Unknown opcode: {:#09b}", opcode)),
    }
}

/// Sign-extend I-type immediate (bits [31:20] -> 12 bits)
fn sign_extend_i_imm(instr: u32) -> i32 {
    let imm = (instr >> 20) as i32;
    // Sign-extend from bit 11
    if imm & 0x800 != 0 {
        imm | !0xFFF
    } else {
        imm
    }
}

/// Sign-extend S-type immediate (bits [31:25] and [11:7])
fn sign_extend_s_imm(instr: u32) -> i32 {
    let imm_11_5 = (instr >> 25) & 0x7F;
    let imm_4_0 = (instr >> 7) & 0x1F;
    let imm = ((imm_11_5 << 5) | imm_4_0) as i32;
    // Sign-extend from bit 11
    if imm & 0x800 != 0 {
        imm | !0xFFF
    } else {
        imm
    }
}

/// Sign-extend B-type immediate (branch offset)
fn sign_extend_b_imm(instr: u32) -> i32 {
    let imm_12 = (instr >> 31) & 1;
    let imm_11 = (instr >> 7) & 1;
    let imm_10_5 = (instr >> 25) & 0x3F;
    let imm_4_1 = (instr >> 8) & 0xF;
    let imm = ((imm_12 << 12) | (imm_11 << 11) | (imm_10_5 << 5) | (imm_4_1 << 1)) as i32;
    // Sign-extend from bit 12
    if imm & 0x1000 != 0 {
        imm | !0x1FFF
    } else {
        imm
    }
}

/// Sign-extend J-type immediate (JAL offset)
fn sign_extend_j_imm(instr: u32) -> i32 {
    let imm_20 = (instr >> 31) & 1;
    let imm_19_12 = (instr >> 12) & 0xFF;
    let imm_11 = (instr >> 20) & 1;
    let imm_10_1 = (instr >> 21) & 0x3FF;
    let imm = ((imm_20 << 20) | (imm_19_12 << 12) | (imm_11 << 11) | (imm_10_1 << 1)) as i32;
    // Sign-extend from bit 20
    if imm & 0x100000 != 0 {
        imm | !0x1FFFFF
    } else {
        imm
    }
}

/// Decode a sequence of bytes into RISC-V instructions.
///
/// Supports both 32-bit and 16-bit (compressed) instructions.
///
/// # Arguments
/// * `bytes` - Program bytes (must be 2-byte aligned)
///
/// # Returns
/// * `Vec<RiscvInstruction>` - List of decoded instructions
///
/// # Errors
/// Returns an error if any instruction cannot be decoded.
pub fn decode_program(bytes: &[u8]) -> Result<Vec<RiscvInstruction>, String> {
    if bytes.len() % 2 != 0 {
        return Err("Program bytes must be 2-byte aligned".to_string());
    }

    let mut instructions = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let first_half = u16::from_le_bytes([bytes[i], bytes[i + 1]]);

        // Check if this is a compressed instruction (C extension)
        // Compressed instructions have the two lowest bits != 0b11
        if (first_half & 0b11) != 0b11 {
            // 16-bit compressed instruction
            instructions.push(decode_compressed_instruction(first_half)?);
            i += 2;
        } else {
            // 32-bit standard instruction
            if i + 4 > bytes.len() {
                return Err(format!("Incomplete 32-bit instruction at offset {}", i));
            }
            let instr = u32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]);
            instructions.push(decode_instruction(instr)?);
            i += 4;
        }
    }
    Ok(instructions)
}

/// Decode a 16-bit compressed RISC-V instruction (C extension).
///
/// Converts the compressed instruction to its equivalent 32-bit instruction.
pub fn decode_compressed_instruction(instr: u16) -> Result<RiscvInstruction, String> {
    let op = instr & 0b11;
    let funct3 = (instr >> 13) & 0b111;

    match (op, funct3) {
        // Quadrant 0
        (0b00, 0b000) => {
            // C.ADDI4SPN: addi rd', x2, nzuimm
            let rd = ((instr >> 2) & 0b111) as u8 + 8; // rd' = x8-x15
            let nzuimm = decode_ciw_imm(instr);
            if nzuimm == 0 {
                return Err("C.ADDI4SPN with nzuimm=0 is reserved".to_string());
            }
            Ok(RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd,
                rs1: 2,
                imm: nzuimm as i32,
            })
        }
        (0b00, 0b010) => {
            // C.LW: lw rd', offset(rs1')
            let rd = ((instr >> 2) & 0b111) as u8 + 8;
            let rs1 = ((instr >> 7) & 0b111) as u8 + 8;
            let offset = decode_cl_lw_imm(instr);
            Ok(RiscvInstruction::Load {
                op: RiscvMemOp::Lw,
                rd,
                rs1,
                imm: offset as i32,
            })
        }
        (0b00, 0b011) => {
            // C.LD (RV64) or C.FLW (RV32): ld rd', offset(rs1')
            let rd = ((instr >> 2) & 0b111) as u8 + 8;
            let rs1 = ((instr >> 7) & 0b111) as u8 + 8;
            let offset = decode_cl_ld_imm(instr);
            Ok(RiscvInstruction::Load {
                op: RiscvMemOp::Ld,
                rd,
                rs1,
                imm: offset as i32,
            })
        }
        (0b00, 0b110) => {
            // C.SW: sw rs2', offset(rs1')
            let rs2 = ((instr >> 2) & 0b111) as u8 + 8;
            let rs1 = ((instr >> 7) & 0b111) as u8 + 8;
            let offset = decode_cl_lw_imm(instr);
            Ok(RiscvInstruction::Store {
                op: RiscvMemOp::Sw,
                rs1,
                rs2,
                imm: offset as i32,
            })
        }
        (0b00, 0b111) => {
            // C.SD (RV64): sd rs2', offset(rs1')
            let rs2 = ((instr >> 2) & 0b111) as u8 + 8;
            let rs1 = ((instr >> 7) & 0b111) as u8 + 8;
            let offset = decode_cl_ld_imm(instr);
            Ok(RiscvInstruction::Store {
                op: RiscvMemOp::Sd,
                rs1,
                rs2,
                imm: offset as i32,
            })
        }

        // Quadrant 1
        (0b01, 0b000) => {
            // C.ADDI / C.NOP
            let rd = ((instr >> 7) & 0b11111) as u8;
            let imm = decode_ci_imm(instr);
            if rd == 0 {
                Ok(RiscvInstruction::Nop) // C.NOP
            } else {
                Ok(RiscvInstruction::IAlu {
                    op: RiscvOpcode::Add,
                    rd,
                    rs1: rd,
                    imm,
                })
            }
        }
        (0b01, 0b001) => {
            // C.ADDIW (RV64) / C.JAL (RV32)
            let rd = ((instr >> 7) & 0b11111) as u8;
            let imm = decode_ci_imm(instr);
            // Assuming RV64
            Ok(RiscvInstruction::IAluw {
                op: RiscvOpcode::Addw,
                rd,
                rs1: rd,
                imm,
            })
        }
        (0b01, 0b010) => {
            // C.LI: addi rd, x0, imm
            let rd = ((instr >> 7) & 0b11111) as u8;
            let imm = decode_ci_imm(instr);
            Ok(RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd,
                rs1: 0,
                imm,
            })
        }
        (0b01, 0b011) => {
            let rd = ((instr >> 7) & 0b11111) as u8;
            if rd == 2 {
                // C.ADDI16SP: addi x2, x2, nzimm
                let imm = decode_ci_addi16sp_imm(instr);
                Ok(RiscvInstruction::IAlu {
                    op: RiscvOpcode::Add,
                    rd: 2,
                    rs1: 2,
                    imm,
                })
            } else {
                // C.LUI: lui rd, nzimm
                let imm = decode_ci_lui_imm(instr);
                Ok(RiscvInstruction::Lui { rd, imm })
            }
        }
        (0b01, 0b100) => {
            // Various ALU operations on compressed registers
            let funct2 = (instr >> 10) & 0b11;
            let rd = ((instr >> 7) & 0b111) as u8 + 8;
            match funct2 {
                0b00 => {
                    // C.SRLI
                    let shamt = decode_ci_shamt(instr);
                    Ok(RiscvInstruction::IAlu {
                        op: RiscvOpcode::Srl,
                        rd,
                        rs1: rd,
                        imm: shamt as i32,
                    })
                }
                0b01 => {
                    // C.SRAI
                    let shamt = decode_ci_shamt(instr);
                    Ok(RiscvInstruction::IAlu {
                        op: RiscvOpcode::Sra,
                        rd,
                        rs1: rd,
                        imm: shamt as i32,
                    })
                }
                0b10 => {
                    // C.ANDI
                    let imm = decode_ci_imm(instr);
                    Ok(RiscvInstruction::IAlu {
                        op: RiscvOpcode::And,
                        rd,
                        rs1: rd,
                        imm,
                    })
                }
                0b11 => {
                    // C.SUB, C.XOR, C.OR, C.AND, C.SUBW, C.ADDW
                    let funct2_b = (instr >> 5) & 0b11;
                    let rs2 = ((instr >> 2) & 0b111) as u8 + 8;
                    let bit12 = (instr >> 12) & 1;
                    match (bit12, funct2_b) {
                        (0, 0b00) => Ok(RiscvInstruction::RAlu {
                            op: RiscvOpcode::Sub,
                            rd,
                            rs1: rd,
                            rs2,
                        }),
                        (0, 0b01) => Ok(RiscvInstruction::RAlu {
                            op: RiscvOpcode::Xor,
                            rd,
                            rs1: rd,
                            rs2,
                        }),
                        (0, 0b10) => Ok(RiscvInstruction::RAlu {
                            op: RiscvOpcode::Or,
                            rd,
                            rs1: rd,
                            rs2,
                        }),
                        (0, 0b11) => Ok(RiscvInstruction::RAlu {
                            op: RiscvOpcode::And,
                            rd,
                            rs1: rd,
                            rs2,
                        }),
                        (1, 0b00) => Ok(RiscvInstruction::RAluw {
                            op: RiscvOpcode::Subw,
                            rd,
                            rs1: rd,
                            rs2,
                        }),
                        (1, 0b01) => Ok(RiscvInstruction::RAluw {
                            op: RiscvOpcode::Addw,
                            rd,
                            rs1: rd,
                            rs2,
                        }),
                        _ => Err(format!("Unknown C.ALU: funct2_b={:#x}", funct2_b)),
                    }
                }
                _ => Err(format!("Unknown C.ALU funct2: {:#x}", funct2)),
            }
        }
        (0b01, 0b101) => {
            // C.J: jal x0, offset
            let offset = decode_cj_imm(instr);
            Ok(RiscvInstruction::Jal { rd: 0, imm: offset })
        }
        (0b01, 0b110) => {
            // C.BEQZ: beq rs1', x0, offset
            let rs1 = ((instr >> 7) & 0b111) as u8 + 8;
            let offset = decode_cb_imm(instr);
            Ok(RiscvInstruction::Branch {
                cond: BranchCondition::Eq,
                rs1,
                rs2: 0,
                imm: offset,
            })
        }
        (0b01, 0b111) => {
            // C.BNEZ: bne rs1', x0, offset
            let rs1 = ((instr >> 7) & 0b111) as u8 + 8;
            let offset = decode_cb_imm(instr);
            Ok(RiscvInstruction::Branch {
                cond: BranchCondition::Ne,
                rs1,
                rs2: 0,
                imm: offset,
            })
        }

        // Quadrant 2
        (0b10, 0b000) => {
            // C.SLLI
            let rd = ((instr >> 7) & 0b11111) as u8;
            let shamt = decode_ci_shamt(instr);
            Ok(RiscvInstruction::IAlu {
                op: RiscvOpcode::Sll,
                rd,
                rs1: rd,
                imm: shamt as i32,
            })
        }
        (0b10, 0b010) => {
            // C.LWSP: lw rd, offset(x2)
            let rd = ((instr >> 7) & 0b11111) as u8;
            let offset = decode_ci_lwsp_imm(instr);
            Ok(RiscvInstruction::Load {
                op: RiscvMemOp::Lw,
                rd,
                rs1: 2,
                imm: offset as i32,
            })
        }
        (0b10, 0b011) => {
            // C.LDSP (RV64): ld rd, offset(x2)
            let rd = ((instr >> 7) & 0b11111) as u8;
            let offset = decode_ci_ldsp_imm(instr);
            Ok(RiscvInstruction::Load {
                op: RiscvMemOp::Ld,
                rd,
                rs1: 2,
                imm: offset as i32,
            })
        }
        (0b10, 0b100) => {
            let bit12 = ((instr >> 12) & 1) != 0;
            let rs1 = ((instr >> 7) & 0b11111) as u8;
            let rs2 = ((instr >> 2) & 0b11111) as u8;
            match (bit12, rs2) {
                (false, 0) => {
                    // C.JR: jalr x0, rs1, 0
                    Ok(RiscvInstruction::Jalr { rd: 0, rs1, imm: 0 })
                }
                (false, _) => {
                    // C.MV: add rd, x0, rs2
                    Ok(RiscvInstruction::RAlu {
                        op: RiscvOpcode::Add,
                        rd: rs1,
                        rs1: 0,
                        rs2,
                    })
                }
                (true, 0) => {
                    if rs1 == 0 {
                        // C.EBREAK
                        Ok(RiscvInstruction::Ebreak)
                    } else {
                        // C.JALR: jalr x1, rs1, 0
                        Ok(RiscvInstruction::Jalr { rd: 1, rs1, imm: 0 })
                    }
                }
                (true, _) => {
                    // C.ADD: add rd, rd, rs2
                    Ok(RiscvInstruction::RAlu {
                        op: RiscvOpcode::Add,
                        rd: rs1,
                        rs1,
                        rs2,
                    })
                }
            }
        }
        (0b10, 0b110) => {
            // C.SWSP: sw rs2, offset(x2)
            let rs2 = ((instr >> 2) & 0b11111) as u8;
            let offset = decode_css_sw_imm(instr);
            Ok(RiscvInstruction::Store {
                op: RiscvMemOp::Sw,
                rs1: 2,
                rs2,
                imm: offset as i32,
            })
        }
        (0b10, 0b111) => {
            // C.SDSP (RV64): sd rs2, offset(x2)
            let rs2 = ((instr >> 2) & 0b11111) as u8;
            let offset = decode_css_sd_imm(instr);
            Ok(RiscvInstruction::Store {
                op: RiscvMemOp::Sd,
                rs1: 2,
                rs2,
                imm: offset as i32,
            })
        }

        _ => Err(format!(
            "Unknown compressed instruction: op={:#x}, funct3={:#x}",
            op, funct3
        )),
    }
}

// Compressed instruction immediate decoders
fn decode_ciw_imm(instr: u16) -> u32 {
    // C.ADDI4SPN: nzuimm[5:4|9:6|2|3] => scaled by 4
    let bits =
        ((instr >> 5) & 1) << 3 | ((instr >> 6) & 1) << 2 | ((instr >> 7) & 0xF) << 6 | ((instr >> 11) & 0x3) << 4;
    bits as u32
}

fn decode_cl_lw_imm(instr: u16) -> u32 {
    // C.LW/C.SW: offset[5:3|2|6]
    let bits = ((instr >> 5) & 1) << 6 | ((instr >> 6) & 1) << 2 | ((instr >> 10) & 0x7) << 3;
    bits as u32
}

fn decode_cl_ld_imm(instr: u16) -> u32 {
    // C.LD/C.SD: offset[5:3|7:6]
    let bits = ((instr >> 5) & 0x3) << 6 | ((instr >> 10) & 0x7) << 3;
    bits as u32
}

fn decode_ci_imm(instr: u16) -> i32 {
    // CI format: imm[5|4:0], sign-extended
    let imm5 = ((instr >> 12) & 1) as i32;
    let imm4_0 = ((instr >> 2) & 0x1F) as i32;
    let imm = (imm5 << 5) | imm4_0;
    // Sign-extend from bit 5
    if imm5 != 0 {
        imm | !0x3F
    } else {
        imm
    }
}

fn decode_ci_shamt(instr: u16) -> u32 {
    // Shift amount: shamt[5|4:0]
    let shamt5 = ((instr >> 12) & 1) as u32;
    let shamt4_0 = ((instr >> 2) & 0x1F) as u32;
    (shamt5 << 5) | shamt4_0
}

fn decode_ci_addi16sp_imm(instr: u16) -> i32 {
    // C.ADDI16SP: nzimm[9|4|6|8:7|5] scaled by 16
    let bit9 = ((instr >> 12) & 1) as i32;
    let bit4 = ((instr >> 6) & 1) as i32;
    let bit6 = ((instr >> 5) & 1) as i32;
    let bit8_7 = ((instr >> 3) & 0x3) as i32;
    let bit5 = ((instr >> 2) & 1) as i32;
    let imm = (bit9 << 9) | (bit8_7 << 7) | (bit6 << 6) | (bit5 << 5) | (bit4 << 4);
    // Sign-extend from bit 9
    if bit9 != 0 {
        imm | !0x3FF
    } else {
        imm
    }
}

fn decode_ci_lui_imm(instr: u16) -> i32 {
    // C.LUI: nzimm[17|16:12]
    let bit17 = ((instr >> 12) & 1) as i32;
    let bits16_12 = ((instr >> 2) & 0x1F) as i32;
    let imm = (bit17 << 5) | bits16_12;
    // Sign-extend from bit 5
    if bit17 != 0 {
        imm | !0x3F
    } else {
        imm
    }
}

fn decode_cj_imm(instr: u16) -> i32 {
    // C.J/C.JAL: offset[11|4|9:8|10|6|7|3:1|5]
    let bit11 = ((instr >> 12) & 1) as i32;
    let bit4 = ((instr >> 11) & 1) as i32;
    let bit9_8 = ((instr >> 9) & 0x3) as i32;
    let bit10 = ((instr >> 8) & 1) as i32;
    let bit6 = ((instr >> 7) & 1) as i32;
    let bit7 = ((instr >> 6) & 1) as i32;
    let bit3_1 = ((instr >> 3) & 0x7) as i32;
    let bit5 = ((instr >> 2) & 1) as i32;
    let imm = (bit11 << 11)
        | (bit10 << 10)
        | (bit9_8 << 8)
        | (bit7 << 7)
        | (bit6 << 6)
        | (bit5 << 5)
        | (bit4 << 4)
        | (bit3_1 << 1);
    // Sign-extend from bit 11
    if bit11 != 0 {
        imm | !0xFFF
    } else {
        imm
    }
}

fn decode_cb_imm(instr: u16) -> i32 {
    // C.BEQZ/C.BNEZ: offset[8|4:3|7:6|2:1|5]
    let bit8 = ((instr >> 12) & 1) as i32;
    let bit4_3 = ((instr >> 10) & 0x3) as i32;
    let bit7_6 = ((instr >> 5) & 0x3) as i32;
    let bit2_1 = ((instr >> 3) & 0x3) as i32;
    let bit5 = ((instr >> 2) & 1) as i32;
    let imm = (bit8 << 8) | (bit7_6 << 6) | (bit5 << 5) | (bit4_3 << 3) | (bit2_1 << 1);
    // Sign-extend from bit 8
    if bit8 != 0 {
        imm | !0x1FF
    } else {
        imm
    }
}

fn decode_ci_lwsp_imm(instr: u16) -> u32 {
    // C.LWSP: offset[5|4:2|7:6]
    let bit5 = ((instr >> 12) & 1) as u32;
    let bit4_2 = ((instr >> 4) & 0x7) as u32;
    let bit7_6 = ((instr >> 2) & 0x3) as u32;
    (bit7_6 << 6) | (bit5 << 5) | (bit4_2 << 2)
}

fn decode_ci_ldsp_imm(instr: u16) -> u32 {
    // C.LDSP: offset[5|4:3|8:6]
    let bit5 = ((instr >> 12) & 1) as u32;
    let bit4_3 = ((instr >> 5) & 0x3) as u32;
    let bit8_6 = ((instr >> 2) & 0x7) as u32;
    (bit8_6 << 6) | (bit5 << 5) | (bit4_3 << 3)
}

fn decode_css_sw_imm(instr: u16) -> u32 {
    // C.SWSP: offset[5:2|7:6]
    let bit5_2 = ((instr >> 9) & 0xF) as u32;
    let bit7_6 = ((instr >> 7) & 0x3) as u32;
    (bit7_6 << 6) | (bit5_2 << 2)
}

fn decode_css_sd_imm(instr: u16) -> u32 {
    // C.SDSP: offset[5:3|8:6]
    let bit5_3 = ((instr >> 10) & 0x7) as u32;
    let bit8_6 = ((instr >> 7) & 0x7) as u32;
    (bit8_6 << 6) | (bit5_3 << 3)
}
