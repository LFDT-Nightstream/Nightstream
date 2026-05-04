//! Owns the compact RV32IM parity-slice machine types, instruction encoding, and decode.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::layout::RV32_REGISTER_COUNT;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Rv32BuildError {
    Decode(String),
    Program(String),
    Memory(String),
}

impl core::fmt::Display for Rv32BuildError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Decode(msg) | Self::Program(msg) | Self::Memory(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for Rv32BuildError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Rv32Opcode {
    Addi,
    Add,
    Sub,
    Andi,
    And,
    Ori,
    Or,
    Xori,
    Xor,
    Slti,
    Slt,
    Sltiu,
    Sltu,
    Slli,
    Sll,
    Srli,
    Srl,
    Srai,
    Sra,
    Lui,
    Auipc,
    Fence,
    Mul,
    Mulh,
    Mulhsu,
    Mulhu,
    Div,
    Divu,
    Rem,
    Remu,
    Lb,
    Lbu,
    Lh,
    Lhu,
    Lw,
    Sb,
    Sh,
    Sw,
    Jal,
    Jalr,
    Beq,
    Bne,
    Blt,
    Bge,
    Bltu,
    Bgeu,
    Ecall,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv32DecodedInstruction {
    pub opcode: Rv32Opcode,
    pub rd: u8,
    pub rs1: u8,
    pub rs2: u8,
    pub imm: i32,
    pub word: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MemoryWord {
    pub addr: u32,
    pub value: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv32Program {
    pub start_pc: u32,
    pub words: Vec<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Rv32State {
    pub pc: u32,
    pub regs: [u32; RV32_REGISTER_COUNT],
    pub memory: BTreeMap<u32, u32>,
    pub halted: bool,
}

impl Rv32Program {
    pub fn new(start_pc: u32, words: Vec<u32>) -> Self {
        Self { start_pc, words }
    }

    pub fn fetch_word(&self, pc: u32) -> Result<u32, Rv32BuildError> {
        if pc < self.start_pc {
            return Err(Rv32BuildError::Program(format!(
                "pc 0x{pc:08x} is below program base 0x{:08x}",
                self.start_pc
            )));
        }
        if (pc - self.start_pc) % 4 != 0 {
            return Err(Rv32BuildError::Program(format!("pc 0x{pc:08x} is not 4-byte aligned")));
        }
        let idx = ((pc - self.start_pc) / 4) as usize;
        self.words
            .get(idx)
            .copied()
            .ok_or_else(|| Rv32BuildError::Program(format!("no instruction at pc 0x{pc:08x}")))
    }
}

impl Rv32State {
    pub fn new(pc: u32, regs: [u32; RV32_REGISTER_COUNT], memory_words: &[MemoryWord]) -> Self {
        let mut state = Self {
            pc,
            regs,
            memory: memory_words
                .iter()
                .map(|word| (word.addr, word.value))
                .collect(),
            halted: false,
        };
        state.regs[0] = 0;
        state
    }

    pub fn read_reg(&self, idx: u8) -> u32 {
        self.regs[idx as usize]
    }

    pub fn write_reg(&mut self, idx: u8, value: u32) {
        if idx != 0 {
            self.regs[idx as usize] = value;
        }
        self.regs[0] = 0;
    }

    pub fn read_memory_word(&self, addr: u32) -> u32 {
        self.memory.get(&addr).copied().unwrap_or(0)
    }

    pub fn write_memory_word(&mut self, addr: u32, value: u32) {
        self.memory.insert(addr, value);
    }

    pub fn memory_words(&self) -> Vec<MemoryWord> {
        self.memory
            .iter()
            .map(|(&addr, &value)| MemoryWord { addr, value })
            .collect()
    }
}

fn field(word: u32, shift: u32, width: u32) -> u32 {
    (word >> shift) & ((1u32 << width) - 1)
}

fn sign_extend(value: u32, bits: u32) -> i32 {
    let shift = 32 - bits;
    ((value as i32) << shift) >> shift
}

fn decode_i_imm(word: u32) -> i32 {
    sign_extend(field(word, 20, 12), 12)
}

fn decode_b_imm(word: u32) -> i32 {
    let imm11 = field(word, 7, 1);
    let imm4_1 = field(word, 8, 4);
    let imm10_5 = field(word, 25, 6);
    let imm12 = field(word, 31, 1);
    let imm = (imm12 << 12) | (imm11 << 11) | (imm10_5 << 5) | (imm4_1 << 1);
    sign_extend(imm, 13)
}

fn decode_u_imm(word: u32) -> i32 {
    sign_extend(word & 0xffff_f000, 32)
}

pub fn decode_instruction(word: u32) -> Result<Rv32DecodedInstruction, Rv32BuildError> {
    if word == 0x0000_0073 {
        return Ok(Rv32DecodedInstruction {
            opcode: Rv32Opcode::Ecall,
            rd: 0,
            rs1: 0,
            rs2: 0,
            imm: 0,
            word,
        });
    }

    let opcode = field(word, 0, 7);
    let rd = field(word, 7, 5) as u8;
    let funct3 = field(word, 12, 3);
    let rs1 = field(word, 15, 5) as u8;
    let rs2 = field(word, 20, 5) as u8;
    let funct7 = field(word, 25, 7);
    let shamt5 = field(word, 20, 5) as i32;

    match opcode {
        0x13 => match funct3 {
            0b000 => Ok(Rv32DecodedInstruction {
                opcode: Rv32Opcode::Addi,
                rd,
                rs1,
                rs2: 0,
                imm: decode_i_imm(word),
                word,
            }),
            0b010 => Ok(Rv32DecodedInstruction {
                opcode: Rv32Opcode::Slti,
                rd,
                rs1,
                rs2: 0,
                imm: decode_i_imm(word),
                word,
            }),
            0b011 => Ok(Rv32DecodedInstruction {
                opcode: Rv32Opcode::Sltiu,
                rd,
                rs1,
                rs2: 0,
                imm: decode_i_imm(word),
                word,
            }),
            0b100 => Ok(Rv32DecodedInstruction {
                opcode: Rv32Opcode::Xori,
                rd,
                rs1,
                rs2: 0,
                imm: decode_i_imm(word),
                word,
            }),
            0b110 => Ok(Rv32DecodedInstruction {
                opcode: Rv32Opcode::Ori,
                rd,
                rs1,
                rs2: 0,
                imm: decode_i_imm(word),
                word,
            }),
            0b111 => Ok(Rv32DecodedInstruction {
                opcode: Rv32Opcode::Andi,
                rd,
                rs1,
                rs2: 0,
                imm: decode_i_imm(word),
                word,
            }),
            0b001 if funct7 == 0 => Ok(Rv32DecodedInstruction {
                opcode: Rv32Opcode::Slli,
                rd,
                rs1,
                rs2: 0,
                imm: shamt5,
                word,
            }),
            0b101 if funct7 == 0 => Ok(Rv32DecodedInstruction {
                opcode: Rv32Opcode::Srli,
                rd,
                rs1,
                rs2: 0,
                imm: shamt5,
                word,
            }),
            0b101 if funct7 == 0b0100000 => Ok(Rv32DecodedInstruction {
                opcode: Rv32Opcode::Srai,
                rd,
                rs1,
                rs2: 0,
                imm: shamt5,
                word,
            }),
            _ => Err(Rv32BuildError::Decode(format!(
                "unsupported RV32 parity-slice instruction 0x{word:08x}"
            ))),
        },
        0x33 => match (funct3, funct7) {
            (0b000, 0b0000000) => Ok(Rv32DecodedInstruction {
                opcode: Rv32Opcode::Add,
                rd,
                rs1,
                rs2,
                imm: 0,
                word,
            }),
            (0b000, 0b0100000) => Ok(Rv32DecodedInstruction {
                opcode: Rv32Opcode::Sub,
                rd,
                rs1,
                rs2,
                imm: 0,
                word,
            }),
            (0b001, 0b0000000) => Ok(Rv32DecodedInstruction {
                opcode: Rv32Opcode::Sll,
                rd,
                rs1,
                rs2,
                imm: 0,
                word,
            }),
            (0b010, 0b0000000) => Ok(Rv32DecodedInstruction {
                opcode: Rv32Opcode::Slt,
                rd,
                rs1,
                rs2,
                imm: 0,
                word,
            }),
            (0b011, 0b0000000) => Ok(Rv32DecodedInstruction {
                opcode: Rv32Opcode::Sltu,
                rd,
                rs1,
                rs2,
                imm: 0,
                word,
            }),
            (0b100, 0b0000000) => Ok(Rv32DecodedInstruction {
                opcode: Rv32Opcode::Xor,
                rd,
                rs1,
                rs2,
                imm: 0,
                word,
            }),
            (0b101, 0b0000000) => Ok(Rv32DecodedInstruction {
                opcode: Rv32Opcode::Srl,
                rd,
                rs1,
                rs2,
                imm: 0,
                word,
            }),
            (0b101, 0b0100000) => Ok(Rv32DecodedInstruction {
                opcode: Rv32Opcode::Sra,
                rd,
                rs1,
                rs2,
                imm: 0,
                word,
            }),
            (0b110, 0b0000000) => Ok(Rv32DecodedInstruction {
                opcode: Rv32Opcode::Or,
                rd,
                rs1,
                rs2,
                imm: 0,
                word,
            }),
            (0b111, 0b0000000) => Ok(Rv32DecodedInstruction {
                opcode: Rv32Opcode::And,
                rd,
                rs1,
                rs2,
                imm: 0,
                word,
            }),
            (0b000, 0b0000001) => Ok(Rv32DecodedInstruction {
                opcode: Rv32Opcode::Mul,
                rd,
                rs1,
                rs2,
                imm: 0,
                word,
            }),
            (0b001, 0b0000001) => Ok(Rv32DecodedInstruction {
                opcode: Rv32Opcode::Mulh,
                rd,
                rs1,
                rs2,
                imm: 0,
                word,
            }),
            (0b010, 0b0000001) => Ok(Rv32DecodedInstruction {
                opcode: Rv32Opcode::Mulhsu,
                rd,
                rs1,
                rs2,
                imm: 0,
                word,
            }),
            (0b011, 0b0000001) => Ok(Rv32DecodedInstruction {
                opcode: Rv32Opcode::Mulhu,
                rd,
                rs1,
                rs2,
                imm: 0,
                word,
            }),
            (0b100, 0b0000001) => Ok(Rv32DecodedInstruction {
                opcode: Rv32Opcode::Div,
                rd,
                rs1,
                rs2,
                imm: 0,
                word,
            }),
            (0b101, 0b0000001) => Ok(Rv32DecodedInstruction {
                opcode: Rv32Opcode::Divu,
                rd,
                rs1,
                rs2,
                imm: 0,
                word,
            }),
            (0b110, 0b0000001) => Ok(Rv32DecodedInstruction {
                opcode: Rv32Opcode::Rem,
                rd,
                rs1,
                rs2,
                imm: 0,
                word,
            }),
            (0b111, 0b0000001) => Ok(Rv32DecodedInstruction {
                opcode: Rv32Opcode::Remu,
                rd,
                rs1,
                rs2,
                imm: 0,
                word,
            }),
            _ => Err(Rv32BuildError::Decode(format!(
                "unsupported RV32 parity-slice instruction 0x{word:08x}"
            ))),
        },
        0x17 => Ok(Rv32DecodedInstruction {
            opcode: Rv32Opcode::Auipc,
            rd,
            rs1: 0,
            rs2: 0,
            imm: decode_u_imm(word),
            word,
        }),
        0x37 => Ok(Rv32DecodedInstruction {
            opcode: Rv32Opcode::Lui,
            rd,
            rs1: 0,
            rs2: 0,
            imm: decode_u_imm(word),
            word,
        }),
        0x0f if funct3 == 0 => Ok(Rv32DecodedInstruction {
            opcode: Rv32Opcode::Fence,
            rd: 0,
            rs1: 0,
            rs2: 0,
            imm: 0,
            word,
        }),
        0x03 if matches!(funct3, 0 | 1 | 2 | 4 | 5) => Ok(Rv32DecodedInstruction {
            opcode: match funct3 {
                0 => Rv32Opcode::Lb,
                1 => Rv32Opcode::Lh,
                2 => Rv32Opcode::Lw,
                4 => Rv32Opcode::Lbu,
                5 => Rv32Opcode::Lhu,
                _ => unreachable!(),
            },
            rd,
            rs1,
            rs2: 0,
            imm: decode_i_imm(word),
            word,
        }),
        0x23 if matches!(funct3, 0 | 1 | 2) => {
            let imm_lo = field(word, 7, 5);
            let imm_hi = field(word, 25, 7);
            let imm = (imm_hi << 5) | imm_lo;
            Ok(Rv32DecodedInstruction {
                opcode: match funct3 {
                    0 => Rv32Opcode::Sb,
                    1 => Rv32Opcode::Sh,
                    2 => Rv32Opcode::Sw,
                    _ => unreachable!(),
                },
                rd: 0,
                rs1,
                rs2,
                imm: sign_extend(imm, 12),
                word,
            })
        }
        0x6f => {
            let imm20 = field(word, 31, 1);
            let imm10_1 = field(word, 21, 10);
            let imm11 = field(word, 20, 1);
            let imm19_12 = field(word, 12, 8);
            let imm = (imm20 << 20) | (imm19_12 << 12) | (imm11 << 11) | (imm10_1 << 1);
            Ok(Rv32DecodedInstruction {
                opcode: Rv32Opcode::Jal,
                rd,
                rs1: 0,
                rs2: 0,
                imm: sign_extend(imm, 21),
                word,
            })
        }
        0x67 if funct3 == 0 => Ok(Rv32DecodedInstruction {
            opcode: Rv32Opcode::Jalr,
            rd,
            rs1,
            rs2: 0,
            imm: decode_i_imm(word),
            word,
        }),
        0x63 if matches!(funct3, 0 | 1 | 4 | 5 | 6 | 7) => Ok(Rv32DecodedInstruction {
            opcode: match funct3 {
                0 => Rv32Opcode::Beq,
                1 => Rv32Opcode::Bne,
                4 => Rv32Opcode::Blt,
                5 => Rv32Opcode::Bge,
                6 => Rv32Opcode::Bltu,
                7 => Rv32Opcode::Bgeu,
                _ => unreachable!(),
            },
            rd: 0,
            rs1,
            rs2,
            imm: decode_b_imm(word),
            word,
        }),
        _ => Err(Rv32BuildError::Decode(format!(
            "unsupported RV32 parity-slice instruction 0x{word:08x}"
        ))),
    }
}

fn encode_i_op(rd: u8, rs1: u8, imm: i16, funct3: u32) -> u32 {
    let imm12 = (imm as i32 as u32) & 0x0fff;
    (imm12 << 20) | ((rs1 as u32) << 15) | (funct3 << 12) | ((rd as u32) << 7) | 0x13
}

fn encode_shift_i_op(rd: u8, rs1: u8, shamt: u8, funct3: u32, funct6: u32) -> u32 {
    let shamt5 = (shamt as u32) & 0x1f;
    (funct6 << 25) | (shamt5 << 20) | ((rs1 as u32) << 15) | (funct3 << 12) | ((rd as u32) << 7) | 0x13
}

fn encode_r_op(rd: u8, rs1: u8, rs2: u8, funct3: u32, funct7: u32) -> u32 {
    (funct7 << 25) | ((rs2 as u32) << 20) | ((rs1 as u32) << 15) | (funct3 << 12) | ((rd as u32) << 7) | 0x33
}

fn encode_u_op(rd: u8, imm: i32, opcode: u32) -> u32 {
    let imm32 = imm as u32;
    (imm32 & 0xffff_f000) | ((rd as u32) << 7) | opcode
}

pub fn encode_addi(rd: u8, rs1: u8, imm: i16) -> u32 {
    encode_i_op(rd, rs1, imm, 0b000)
}

pub fn encode_add(rd: u8, rs1: u8, rs2: u8) -> u32 {
    encode_r_op(rd, rs1, rs2, 0b000, 0b0000000)
}

pub fn encode_sub(rd: u8, rs1: u8, rs2: u8) -> u32 {
    encode_r_op(rd, rs1, rs2, 0b000, 0b0100000)
}

pub fn encode_andi(rd: u8, rs1: u8, imm: i16) -> u32 {
    encode_i_op(rd, rs1, imm, 0b111)
}

pub fn encode_and(rd: u8, rs1: u8, rs2: u8) -> u32 {
    encode_r_op(rd, rs1, rs2, 0b111, 0b0000000)
}

pub fn encode_ori(rd: u8, rs1: u8, imm: i16) -> u32 {
    encode_i_op(rd, rs1, imm, 0b110)
}

pub fn encode_or(rd: u8, rs1: u8, rs2: u8) -> u32 {
    encode_r_op(rd, rs1, rs2, 0b110, 0b0000000)
}

pub fn encode_xori(rd: u8, rs1: u8, imm: i16) -> u32 {
    encode_i_op(rd, rs1, imm, 0b100)
}

pub fn encode_xor(rd: u8, rs1: u8, rs2: u8) -> u32 {
    encode_r_op(rd, rs1, rs2, 0b100, 0b0000000)
}

pub fn encode_slti(rd: u8, rs1: u8, imm: i16) -> u32 {
    encode_i_op(rd, rs1, imm, 0b010)
}

pub fn encode_slt(rd: u8, rs1: u8, rs2: u8) -> u32 {
    encode_r_op(rd, rs1, rs2, 0b010, 0b0000000)
}

pub fn encode_sltiu(rd: u8, rs1: u8, imm: i16) -> u32 {
    encode_i_op(rd, rs1, imm, 0b011)
}

pub fn encode_sltu(rd: u8, rs1: u8, rs2: u8) -> u32 {
    encode_r_op(rd, rs1, rs2, 0b011, 0b0000000)
}

pub fn encode_slli(rd: u8, rs1: u8, shamt: u8) -> u32 {
    encode_shift_i_op(rd, rs1, shamt, 0b001, 0b000000)
}

pub fn encode_sll(rd: u8, rs1: u8, rs2: u8) -> u32 {
    encode_r_op(rd, rs1, rs2, 0b001, 0b0000000)
}

pub fn encode_srli(rd: u8, rs1: u8, shamt: u8) -> u32 {
    encode_shift_i_op(rd, rs1, shamt, 0b101, 0b000000)
}

pub fn encode_srl(rd: u8, rs1: u8, rs2: u8) -> u32 {
    encode_r_op(rd, rs1, rs2, 0b101, 0b0000000)
}

pub fn encode_srai(rd: u8, rs1: u8, shamt: u8) -> u32 {
    encode_shift_i_op(rd, rs1, shamt, 0b101, 0b010000)
}

pub fn encode_sra(rd: u8, rs1: u8, rs2: u8) -> u32 {
    encode_r_op(rd, rs1, rs2, 0b101, 0b0100000)
}

pub fn encode_lui(rd: u8, imm: i32) -> u32 {
    encode_u_op(rd, imm, 0x37)
}

pub fn encode_auipc(rd: u8, imm: i32) -> u32 {
    encode_u_op(rd, imm, 0x17)
}

pub fn encode_fence() -> u32 {
    0x0000_000f
}

pub fn encode_mul(rd: u8, rs1: u8, rs2: u8) -> u32 {
    encode_r_op(rd, rs1, rs2, 0b000, 0b0000001)
}

pub fn encode_mulh(rd: u8, rs1: u8, rs2: u8) -> u32 {
    encode_r_op(rd, rs1, rs2, 0b001, 0b0000001)
}

pub fn encode_mulhsu(rd: u8, rs1: u8, rs2: u8) -> u32 {
    encode_r_op(rd, rs1, rs2, 0b010, 0b0000001)
}

pub fn encode_mulhu(rd: u8, rs1: u8, rs2: u8) -> u32 {
    encode_r_op(rd, rs1, rs2, 0b011, 0b0000001)
}

pub fn encode_div(rd: u8, rs1: u8, rs2: u8) -> u32 {
    encode_r_op(rd, rs1, rs2, 0b100, 0b0000001)
}

pub fn encode_divu(rd: u8, rs1: u8, rs2: u8) -> u32 {
    encode_r_op(rd, rs1, rs2, 0b101, 0b0000001)
}

pub fn encode_rem(rd: u8, rs1: u8, rs2: u8) -> u32 {
    encode_r_op(rd, rs1, rs2, 0b110, 0b0000001)
}

pub fn encode_remu(rd: u8, rs1: u8, rs2: u8) -> u32 {
    encode_r_op(rd, rs1, rs2, 0b111, 0b0000001)
}

fn encode_load(rd: u8, rs1: u8, imm: i16, funct3: u32) -> u32 {
    let imm12 = (imm as i32 as u32) & 0x0fff;
    (imm12 << 20) | ((rs1 as u32) << 15) | (funct3 << 12) | ((rd as u32) << 7) | 0x03
}

fn encode_store(rs2: u8, rs1: u8, imm: i16, funct3: u32) -> u32 {
    let imm12 = (imm as i32 as u32) & 0x0fff;
    let imm_lo = imm12 & 0x1f;
    let imm_hi = (imm12 >> 5) & 0x7f;
    (imm_hi << 25) | ((rs2 as u32) << 20) | ((rs1 as u32) << 15) | (funct3 << 12) | (imm_lo << 7) | 0x23
}

pub fn encode_lb(rd: u8, rs1: u8, imm: i16) -> u32 {
    encode_load(rd, rs1, imm, 0b000)
}

pub fn encode_lh(rd: u8, rs1: u8, imm: i16) -> u32 {
    encode_load(rd, rs1, imm, 0b001)
}

pub fn encode_lw(rd: u8, rs1: u8, imm: i16) -> u32 {
    encode_load(rd, rs1, imm, 0b010)
}

pub fn encode_lbu(rd: u8, rs1: u8, imm: i16) -> u32 {
    encode_load(rd, rs1, imm, 0b100)
}

pub fn encode_lhu(rd: u8, rs1: u8, imm: i16) -> u32 {
    encode_load(rd, rs1, imm, 0b101)
}

pub fn encode_sb(rs2: u8, rs1: u8, imm: i16) -> u32 {
    encode_store(rs2, rs1, imm, 0b000)
}

pub fn encode_sh(rs2: u8, rs1: u8, imm: i16) -> u32 {
    encode_store(rs2, rs1, imm, 0b001)
}

pub fn encode_sw(rs2: u8, rs1: u8, imm: i16) -> u32 {
    encode_store(rs2, rs1, imm, 0b010)
}

pub fn encode_jal(rd: u8, imm: i32) -> u32 {
    let imm21 = (imm as u32) & 0x1f_ffff;
    let imm20 = (imm21 >> 20) & 0x1;
    let imm10_1 = (imm21 >> 1) & 0x03ff;
    let imm11 = (imm21 >> 11) & 0x1;
    let imm19_12 = (imm21 >> 12) & 0xff;
    (imm20 << 31) | (imm10_1 << 21) | (imm11 << 20) | (imm19_12 << 12) | ((rd as u32) << 7) | 0x6f
}

pub fn encode_jalr(rd: u8, rs1: u8, imm: i16) -> u32 {
    let imm12 = (imm as i32 as u32) & 0x0fff;
    (imm12 << 20) | ((rs1 as u32) << 15) | ((rd as u32) << 7) | 0x67
}

pub fn encode_beq(rs1: u8, rs2: u8, imm: i16) -> u32 {
    encode_b_branch(rs1, rs2, imm, 0)
}

pub fn encode_bne(rs1: u8, rs2: u8, imm: i16) -> u32 {
    encode_b_branch(rs1, rs2, imm, 0b001)
}

pub fn encode_blt(rs1: u8, rs2: u8, imm: i16) -> u32 {
    encode_b_branch(rs1, rs2, imm, 0b100)
}

pub fn encode_bge(rs1: u8, rs2: u8, imm: i16) -> u32 {
    encode_b_branch(rs1, rs2, imm, 0b101)
}

pub fn encode_bltu(rs1: u8, rs2: u8, imm: i16) -> u32 {
    encode_b_branch(rs1, rs2, imm, 0b110)
}

pub fn encode_bgeu(rs1: u8, rs2: u8, imm: i16) -> u32 {
    encode_b_branch(rs1, rs2, imm, 0b111)
}

fn encode_b_branch(rs1: u8, rs2: u8, imm: i16, funct3: u32) -> u32 {
    let imm13 = (imm as i32 as u32) & 0x1fff;
    let imm11 = (imm13 >> 11) & 0x1;
    let imm4_1 = (imm13 >> 1) & 0x0f;
    let imm10_5 = (imm13 >> 5) & 0x3f;
    let imm12 = (imm13 >> 12) & 0x1;
    (imm12 << 31)
        | (imm10_5 << 25)
        | ((rs2 as u32) << 20)
        | ((rs1 as u32) << 15)
        | (funct3 << 12)
        | (imm4_1 << 8)
        | (imm11 << 7)
        | 0x63
}

pub fn encode_ecall() -> u32 {
    0x0000_0073
}
