use serde::{Deserialize, Serialize};
use std::fmt;

use super::alu::sign_extend;

/// RISC-V ALU operations that use lookup tables (Shout).
///
/// Based on Jolt's instruction semantics (MIT/Apache-2.0 license).
/// Credit: <https://github.com/a16z/jolt>
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RiscvOpcode {
    // === Bitwise Operations (interleaved index) ===
    /// Bitwise AND: rd = rs1 & rs2
    And,
    /// Bitwise XOR: rd = rs1 ^ rs2
    Xor,
    /// Bitwise OR: rd = rs1 | rs2
    Or,

    // === Arithmetic Operations ===
    /// Subtraction: rd = rs1 - rs2 (with wraparound)
    Sub,
    /// Addition: rd = rs1 + rs2 (with wraparound)
    Add,

    // === M Extension: Multiply/Divide Operations ===
    /// Multiply: rd = (rs1 * rs2)[xlen-1:0]
    Mul,
    /// Multiply High (signed × signed): rd = (rs1 * rs2)[2*xlen-1:xlen]
    Mulh,
    /// Multiply High (unsigned × unsigned): rd = (rs1 * rs2)[2*xlen-1:xlen]
    Mulhu,
    /// Multiply High (signed × unsigned): rd = (rs1 * rs2)[2*xlen-1:xlen]
    Mulhsu,
    /// Divide (signed): rd = rs1 / rs2
    Div,
    /// Divide (unsigned): rd = rs1 / rs2
    Divu,
    /// Remainder (signed): rd = rs1 % rs2
    Rem,
    /// Remainder (unsigned): rd = rs1 % rs2
    Remu,

    // === Comparison Operations (interleaved index) ===
    /// Set Less Than (unsigned): rd = (rs1 < rs2) ? 1 : 0
    Sltu,
    /// Set Less Than (signed): rd = (rs1 < rs2) ? 1 : 0
    Slt,
    /// Equality check: rd = (rs1 == rs2) ? 1 : 0
    Eq,
    /// Inequality check: rd = (rs1 != rs2) ? 1 : 0
    Neq,

    // === Shift Operations (specialized tables) ===
    /// Shift Left Logical: rd = rs1 << rs2[log2(xlen)-1:0]
    Sll,
    /// Shift Right Logical: rd = rs1 >> rs2[log2(xlen)-1:0]
    Srl,
    /// Shift Right Arithmetic: rd = rs1 >>> rs2[log2(xlen)-1:0]
    Sra,

    // === RV64 W-suffix Operations (32-bit ops on 64-bit, sign-extended) ===
    /// Add Word: rd = sext((rs1 + rs2)[31:0])
    Addw,
    /// Subtract Word: rd = sext((rs1 - rs2)[31:0])
    Subw,
    /// Shift Left Logical Word: rd = sext((rs1 << shamt)[31:0])
    Sllw,
    /// Shift Right Logical Word: rd = sext((rs1 >> shamt)[31:0])
    Srlw,
    /// Shift Right Arithmetic Word: rd = sext((rs1 >>> shamt)[31:0])
    Sraw,
    /// Multiply Word: rd = sext((rs1 * rs2)[31:0])
    Mulw,
    /// Divide Word (signed): rd = sext((rs1 / rs2)[31:0])
    Divw,
    /// Divide Word (unsigned): rd = sext((rs1 / rs2)[31:0])
    Divuw,
    /// Remainder Word (signed): rd = sext((rs1 % rs2)[31:0])
    Remw,
    /// Remainder Word (unsigned): rd = sext((rs1 % rs2)[31:0])
    Remuw,

    // === Bitmanip (Zbb subset, as used by Jolt) ===
    /// AND with NOT: rd = rs1 & ~rs2
    Andn,
}

impl fmt::Display for RiscvOpcode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RiscvOpcode::And => write!(f, "AND"),
            RiscvOpcode::Xor => write!(f, "XOR"),
            RiscvOpcode::Or => write!(f, "OR"),
            RiscvOpcode::Sub => write!(f, "SUB"),
            RiscvOpcode::Add => write!(f, "ADD"),
            RiscvOpcode::Mul => write!(f, "MUL"),
            RiscvOpcode::Mulh => write!(f, "MULH"),
            RiscvOpcode::Mulhu => write!(f, "MULHU"),
            RiscvOpcode::Mulhsu => write!(f, "MULHSU"),
            RiscvOpcode::Div => write!(f, "DIV"),
            RiscvOpcode::Divu => write!(f, "DIVU"),
            RiscvOpcode::Rem => write!(f, "REM"),
            RiscvOpcode::Remu => write!(f, "REMU"),
            RiscvOpcode::Sltu => write!(f, "SLTU"),
            RiscvOpcode::Slt => write!(f, "SLT"),
            RiscvOpcode::Eq => write!(f, "EQ"),
            RiscvOpcode::Neq => write!(f, "NEQ"),
            RiscvOpcode::Sll => write!(f, "SLL"),
            RiscvOpcode::Srl => write!(f, "SRL"),
            RiscvOpcode::Sra => write!(f, "SRA"),
            RiscvOpcode::Addw => write!(f, "ADDW"),
            RiscvOpcode::Subw => write!(f, "SUBW"),
            RiscvOpcode::Sllw => write!(f, "SLLW"),
            RiscvOpcode::Srlw => write!(f, "SRLW"),
            RiscvOpcode::Sraw => write!(f, "SRAW"),
            RiscvOpcode::Mulw => write!(f, "MULW"),
            RiscvOpcode::Divw => write!(f, "DIVW"),
            RiscvOpcode::Divuw => write!(f, "DIVUW"),
            RiscvOpcode::Remw => write!(f, "REMW"),
            RiscvOpcode::Remuw => write!(f, "REMUW"),
            RiscvOpcode::Andn => write!(f, "ANDN"),
        }
    }
}

/// RISC-V memory operations that use read/write memory (Twist).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum RiscvMemOp {
    /// Load Word (32-bit)
    Lw,
    /// Load Half-word (16-bit, sign-extended)
    Lh,
    /// Load Half-word Unsigned (16-bit, zero-extended)
    Lhu,
    /// Load Byte (8-bit, sign-extended)
    Lb,
    /// Load Byte Unsigned (8-bit, zero-extended)
    Lbu,
    /// Load Double-word (64-bit, RV64 only)
    Ld,
    /// Load Word Unsigned (32-bit, zero-extended, RV64 only)
    Lwu,
    /// Store Word (32-bit)
    Sw,
    /// Store Half-word (16-bit)
    Sh,
    /// Store Byte (8-bit)
    Sb,
    /// Store Double-word (64-bit, RV64 only)
    Sd,

    // === A Extension: Atomic Operations ===
    /// Load-Reserved Word (32-bit)
    LrW,
    /// Load-Reserved Double-word (64-bit)
    LrD,
    /// Store-Conditional Word (32-bit)
    ScW,
    /// Store-Conditional Double-word (64-bit)
    ScD,
    /// Atomic Swap Word
    AmoswapW,
    /// Atomic Swap Double-word
    AmoswapD,
    /// Atomic Add Word
    AmoaddW,
    /// Atomic Add Double-word
    AmoaddD,
    /// Atomic XOR Word
    AmoxorW,
    /// Atomic XOR Double-word
    AmoxorD,
    /// Atomic AND Word
    AmoandW,
    /// Atomic AND Double-word
    AmoandD,
    /// Atomic OR Word
    AmoorW,
    /// Atomic OR Double-word
    AmoorD,
    /// Atomic Min Word (signed)
    AmominW,
    /// Atomic Min Double-word (signed)
    AmominD,
    /// Atomic Max Word (signed)
    AmomaxW,
    /// Atomic Max Double-word (signed)
    AmomaxD,
    /// Atomic Min Word (unsigned)
    AmominuW,
    /// Atomic Min Double-word (unsigned)
    AmominuD,
    /// Atomic Max Word (unsigned)
    AmomaxuW,
    /// Atomic Max Double-word (unsigned)
    AmomaxuD,
}

impl fmt::Display for RiscvMemOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RiscvMemOp::Lw => write!(f, "LW"),
            RiscvMemOp::Lh => write!(f, "LH"),
            RiscvMemOp::Lhu => write!(f, "LHU"),
            RiscvMemOp::Lb => write!(f, "LB"),
            RiscvMemOp::Lbu => write!(f, "LBU"),
            RiscvMemOp::Ld => write!(f, "LD"),
            RiscvMemOp::Lwu => write!(f, "LWU"),
            RiscvMemOp::Sw => write!(f, "SW"),
            RiscvMemOp::Sh => write!(f, "SH"),
            RiscvMemOp::Sb => write!(f, "SB"),
            RiscvMemOp::Sd => write!(f, "SD"),
            RiscvMemOp::LrW => write!(f, "LR.W"),
            RiscvMemOp::LrD => write!(f, "LR.D"),
            RiscvMemOp::ScW => write!(f, "SC.W"),
            RiscvMemOp::ScD => write!(f, "SC.D"),
            RiscvMemOp::AmoswapW => write!(f, "AMOSWAP.W"),
            RiscvMemOp::AmoswapD => write!(f, "AMOSWAP.D"),
            RiscvMemOp::AmoaddW => write!(f, "AMOADD.W"),
            RiscvMemOp::AmoaddD => write!(f, "AMOADD.D"),
            RiscvMemOp::AmoxorW => write!(f, "AMOXOR.W"),
            RiscvMemOp::AmoxorD => write!(f, "AMOXOR.D"),
            RiscvMemOp::AmoandW => write!(f, "AMOAND.W"),
            RiscvMemOp::AmoandD => write!(f, "AMOAND.D"),
            RiscvMemOp::AmoorW => write!(f, "AMOOR.W"),
            RiscvMemOp::AmoorD => write!(f, "AMOOR.D"),
            RiscvMemOp::AmominW => write!(f, "AMOMIN.W"),
            RiscvMemOp::AmominD => write!(f, "AMOMIN.D"),
            RiscvMemOp::AmomaxW => write!(f, "AMOMAX.W"),
            RiscvMemOp::AmomaxD => write!(f, "AMOMAX.D"),
            RiscvMemOp::AmominuW => write!(f, "AMOMINU.W"),
            RiscvMemOp::AmominuD => write!(f, "AMOMINU.D"),
            RiscvMemOp::AmomaxuW => write!(f, "AMOMAXU.W"),
            RiscvMemOp::AmomaxuD => write!(f, "AMOMAXU.D"),
        }
    }
}

impl RiscvMemOp {
    /// Returns true if this is a load operation.
    pub fn is_load(&self) -> bool {
        matches!(
            self,
            RiscvMemOp::Lw
                | RiscvMemOp::Lh
                | RiscvMemOp::Lhu
                | RiscvMemOp::Lb
                | RiscvMemOp::Lbu
                | RiscvMemOp::Ld
                | RiscvMemOp::Lwu
                | RiscvMemOp::LrW
                | RiscvMemOp::LrD
        )
    }

    /// Returns true if this is a store operation.
    pub fn is_store(&self) -> bool {
        matches!(
            self,
            RiscvMemOp::Sw | RiscvMemOp::Sh | RiscvMemOp::Sb | RiscvMemOp::Sd | RiscvMemOp::ScW | RiscvMemOp::ScD
        )
    }

    /// Returns true if this is an atomic operation.
    pub fn is_atomic(&self) -> bool {
        matches!(
            self,
            RiscvMemOp::LrW
                | RiscvMemOp::LrD
                | RiscvMemOp::ScW
                | RiscvMemOp::ScD
                | RiscvMemOp::AmoswapW
                | RiscvMemOp::AmoswapD
                | RiscvMemOp::AmoaddW
                | RiscvMemOp::AmoaddD
                | RiscvMemOp::AmoxorW
                | RiscvMemOp::AmoxorD
                | RiscvMemOp::AmoandW
                | RiscvMemOp::AmoandD
                | RiscvMemOp::AmoorW
                | RiscvMemOp::AmoorD
                | RiscvMemOp::AmominW
                | RiscvMemOp::AmominD
                | RiscvMemOp::AmomaxW
                | RiscvMemOp::AmomaxD
                | RiscvMemOp::AmominuW
                | RiscvMemOp::AmominuD
                | RiscvMemOp::AmomaxuW
                | RiscvMemOp::AmomaxuD
        )
    }

    /// Returns the access width in bytes.
    pub fn width_bytes(&self) -> usize {
        match self {
            RiscvMemOp::Lb | RiscvMemOp::Lbu | RiscvMemOp::Sb => 1,
            RiscvMemOp::Lh | RiscvMemOp::Lhu | RiscvMemOp::Sh => 2,
            RiscvMemOp::Lw
            | RiscvMemOp::Lwu
            | RiscvMemOp::Sw
            | RiscvMemOp::LrW
            | RiscvMemOp::ScW
            | RiscvMemOp::AmoswapW
            | RiscvMemOp::AmoaddW
            | RiscvMemOp::AmoxorW
            | RiscvMemOp::AmoandW
            | RiscvMemOp::AmoorW
            | RiscvMemOp::AmominW
            | RiscvMemOp::AmomaxW
            | RiscvMemOp::AmominuW
            | RiscvMemOp::AmomaxuW => 4,
            RiscvMemOp::Ld
            | RiscvMemOp::Sd
            | RiscvMemOp::LrD
            | RiscvMemOp::ScD
            | RiscvMemOp::AmoswapD
            | RiscvMemOp::AmoaddD
            | RiscvMemOp::AmoxorD
            | RiscvMemOp::AmoandD
            | RiscvMemOp::AmoorD
            | RiscvMemOp::AmominD
            | RiscvMemOp::AmomaxD
            | RiscvMemOp::AmominuD
            | RiscvMemOp::AmomaxuD => 8,
        }
    }

    /// Returns true if this load should sign-extend.
    pub fn is_sign_extend(&self) -> bool {
        matches!(self, RiscvMemOp::Lh | RiscvMemOp::Lb | RiscvMemOp::Lw | RiscvMemOp::LrW)
    }
}

/// Branch condition types.
///
/// Based on Jolt's implementation (MIT/Apache-2.0 license).
/// Credit: <https://github.com/a16z/jolt>
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BranchCondition {
    /// Branch if Equal: pc = (rs1 == rs2) ? pc + imm : pc + 4
    Eq,
    /// Branch if Not Equal: pc = (rs1 != rs2) ? pc + imm : pc + 4
    Ne,
    /// Branch if Less Than (signed): pc = (rs1 < rs2) ? pc + imm : pc + 4
    Lt,
    /// Branch if Greater or Equal (signed): pc = (rs1 >= rs2) ? pc + imm : pc + 4
    Ge,
    /// Branch if Less Than (unsigned): pc = (rs1 < rs2) ? pc + imm : pc + 4
    Ltu,
    /// Branch if Greater or Equal (unsigned): pc = (rs1 >= rs2) ? pc + imm : pc + 4
    Geu,
}

impl fmt::Display for BranchCondition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BranchCondition::Eq => write!(f, "BEQ"),
            BranchCondition::Ne => write!(f, "BNE"),
            BranchCondition::Lt => write!(f, "BLT"),
            BranchCondition::Ge => write!(f, "BGE"),
            BranchCondition::Ltu => write!(f, "BLTU"),
            BranchCondition::Geu => write!(f, "BGEU"),
        }
    }
}

impl BranchCondition {
    /// Evaluate the branch condition.
    ///
    /// Returns true if the branch should be taken.
    pub fn evaluate(&self, rs1: u64, rs2: u64, xlen: usize) -> bool {
        match self {
            BranchCondition::Eq => rs1 == rs2,
            BranchCondition::Ne => rs1 != rs2,
            BranchCondition::Lt => {
                let rs1_signed = sign_extend(rs1, xlen);
                let rs2_signed = sign_extend(rs2, xlen);
                rs1_signed < rs2_signed
            }
            BranchCondition::Ge => {
                let rs1_signed = sign_extend(rs1, xlen);
                let rs2_signed = sign_extend(rs2, xlen);
                rs1_signed >= rs2_signed
            }
            BranchCondition::Ltu => rs1 < rs2,
            BranchCondition::Geu => rs1 >= rs2,
        }
    }

    /// Get the corresponding Shout opcode for this branch condition.
    ///
    /// Branch conditions use the same comparison operations as ALU.
    pub fn to_shout_opcode(&self) -> RiscvOpcode {
        match self {
            BranchCondition::Eq => RiscvOpcode::Eq,
            BranchCondition::Ne => RiscvOpcode::Neq,
            BranchCondition::Lt => RiscvOpcode::Slt,
            BranchCondition::Ge => RiscvOpcode::Slt, // BGE = !(rs1 < rs2)
            BranchCondition::Ltu => RiscvOpcode::Sltu,
            BranchCondition::Geu => RiscvOpcode::Sltu, // BGEU = !(rs1 < rs2)
        }
    }
}

/// A complete RISC-V instruction (decoded).
///
/// Based on Jolt's instruction representation (MIT/Apache-2.0 license).
/// Credit: <https://github.com/a16z/jolt>
#[derive(Clone, Debug)]
pub enum RiscvInstruction {
    /// R-type ALU operation: rd = rs1 op rs2
    RAlu {
        op: RiscvOpcode,
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    /// I-type ALU operation: rd = rs1 op imm
    IAlu {
        op: RiscvOpcode,
        rd: u8,
        rs1: u8,
        imm: i32,
    },
    /// Load operation: rd = mem[rs1 + imm]
    Load {
        op: RiscvMemOp,
        rd: u8,
        rs1: u8,
        imm: i32,
    },
    /// Store operation: mem[rs1 + imm] = rs2
    Store {
        op: RiscvMemOp,
        rs1: u8,
        rs2: u8,
        imm: i32,
    },
    /// Branch operation: if cond(rs1, rs2) then pc = pc + imm
    Branch {
        cond: BranchCondition,
        rs1: u8,
        rs2: u8,
        imm: i32,
    },
    /// Jump and Link: rd = pc + 4; pc = pc + imm
    Jal { rd: u8, imm: i32 },
    /// Jump and Link Register: rd = pc + 4; pc = (rs1 + imm) & ~1
    Jalr { rd: u8, rs1: u8, imm: i32 },
    /// Load Upper Immediate: rd = imm << 12
    Lui { rd: u8, imm: i32 },
    /// Add Upper Immediate to PC: rd = pc + (imm << 12)
    Auipc { rd: u8, imm: i32 },
    /// Halt (ECALL with a0 = 0)
    Halt,
    /// No-op
    Nop,

    // === RV64 W-suffix Operations ===
    /// R-type ALU Word operation: rd = sext((rs1 op rs2)[31:0])
    RAluw {
        op: RiscvOpcode,
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    /// I-type ALU Word operation: rd = sext((rs1 op imm)[31:0])
    IAluw {
        op: RiscvOpcode,
        rd: u8,
        rs1: u8,
        imm: i32,
    },

    // === A Extension: Atomics ===
    /// Load-Reserved: rd = mem[rs1]; reserve address
    LoadReserved { op: RiscvMemOp, rd: u8, rs1: u8 },
    /// Store-Conditional: if reserved, mem[rs1] = rs2, rd = 0; else rd = 1
    StoreConditional {
        op: RiscvMemOp,
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    /// Atomic Memory Operation: rd = mem[rs1]; mem[rs1] = op(mem[rs1], rs2)
    Amo {
        op: RiscvMemOp,
        rd: u8,
        rs1: u8,
        rs2: u8,
    },

    // === System Instructions ===
    /// Environment Call (syscall)
    Ecall,
    /// Environment Break (debugger trap)
    Ebreak,
    /// Memory Fence
    Fence { pred: u8, succ: u8 },
    /// Instruction Fence
    FenceI,
}
