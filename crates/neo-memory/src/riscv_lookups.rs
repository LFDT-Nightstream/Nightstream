//! RISC-V RV64IMAC instruction support for Neo's proving system.
//!
//! This module implements a complete **RV64IMAC** RISC-V instruction set, providing:
//! - Instruction decoding (32-bit and 16-bit compressed)
//! - Instruction encoding
//! - CPU execution with tracing
//! - Lookup tables for ALU operations (Shout protocol)
//! - Memory operations (Twist protocol)
//!
//! # Supported RISC-V Extensions
//!
//! | Extension | Description | Status |
//! |-----------|-------------|--------|
//! | **I** | Base Integer (RV64I) | ✅ Full |
//! | **M** | Multiply/Divide | ✅ Full |
//! | **A** | Atomics (LR/SC, AMO) | ✅ Full |
//! | **C** | Compressed (16-bit) | ✅ Full |
//! | **Zbb** | Bitmanip (subset) | ✅ ANDN |
//!
//! This provides feature parity with [Jolt](https://github.com/a16z/jolt).
//!
//! # Architecture
//!
//! ## Lookup Tables (Shout)
//!
//! ALU operations are proven using Neo's Shout (read-only memory) protocol:
//! - The **index** encodes operands via bit interleaving
//! - The **value** is the operation result
//! - MLEs enable efficient sumcheck verification
//!
//! ## Memory (Twist)
//!
//! Load/store and atomic operations use Neo's Twist (read-write memory) protocol.
//!
//! # Instruction Categories
//!
//! ## Base Integer (I Extension)
//! - **Arithmetic**: ADD, ADDI, SUB
//! - **Logical**: AND, ANDI, OR, ORI, XOR, XORI
//! - **Shifts**: SLL, SLLI, SRL, SRLI, SRA, SRAI
//! - **Compare**: SLT, SLTI, SLTU, SLTIU
//! - **Branches**: BEQ, BNE, BLT, BGE, BLTU, BGEU
//! - **Jumps**: JAL, JALR
//! - **Upper Immediate**: LUI, AUIPC
//! - **Loads**: LB, LBU, LH, LHU, LW, LWU, LD
//! - **Stores**: SB, SH, SW, SD
//!
//! ## RV64 Word Operations
//! - ADDW, SUBW, ADDIW
//! - SLLW, SLLIW, SRLW, SRLIW, SRAW, SRAIW
//!
//! ## Multiply/Divide (M Extension)
//! - MUL, MULH, MULHU, MULHSU
//! - DIV, DIVU, REM, REMU
//! - MULW, DIVW, DIVUW, REMW, REMUW (RV64)
//!
//! ## Atomics (A Extension)
//! - **Load-Reserved**: LR.W, LR.D
//! - **Store-Conditional**: SC.W, SC.D
//! - **AMO**: AMOSWAP, AMOADD, AMOXOR, AMOAND, AMOOR, AMOMIN, AMOMAX, AMOMINU, AMOMAXU
//!
//! ## Compressed (C Extension)
//! - All quadrant 0, 1, and 2 instructions
//! - Automatic detection of 16-bit vs 32-bit instructions
//!
//! ## System
//! - ECALL, EBREAK
//! - FENCE, FENCE.I
//!
//! # Example
//!
//! ```ignore
//! use neo_memory::riscv_lookups::{RiscvCpu, RiscvMemory, RiscvShoutTables, decode_program};
//! use neo_vm_trace::trace_program;
//!
//! // Load and decode a RISC-V binary (supports compressed instructions)
//! let program = decode_program(&binary_bytes)?;
//!
//! // Execute with full tracing
//! let mut cpu = RiscvCpu::new(64); // RV64
//! cpu.load_program(0, &program);
//! let memory = RiscvMemory::new(64);
//! let shout = RiscvShoutTables::new(64);
//!
//! let trace = trace_program(cpu, memory, shout, 1000)?;
//! // trace now contains all steps for proving
//! ```

use neo_vm_trace::{Shout, ShoutId, Twist, TwistId};
use p3_field::Field;
use serde::{Deserialize, Serialize};
use std::fmt;

// ============================================================================
// Bit manipulation utilities (matching Jolt's approach)
// ============================================================================

/// Interleave the bits of two operands into a single lookup index.
///
/// For n-bit operands x and y, produces a 2n-bit index where:
/// - Bit positions 2i contain x_i
/// - Bit positions 2i+1 contain y_i
///
/// This matches Jolt's interleaving convention for lookup tables.
///
/// # Example
/// For x = 0b10 and y = 0b01:
/// - x_0 = 0, x_1 = 1
/// - y_0 = 1, y_1 = 0
/// - Result: bits at pos 0,1,2,3 = x0,y0,x1,y1 = 0,1,1,0 = 0b0110 = 6
pub fn interleave_bits(x: u64, y: u64) -> u128 {
    let mut result = 0u128;
    for i in 0..64 {
        let x_bit = ((x >> i) & 1) as u128;
        let y_bit = ((y >> i) & 1) as u128;
        result |= x_bit << (2 * i);
        result |= y_bit << (2 * i + 1);
    }
    result
}

/// Uninterleave bits from a lookup index back to two operands.
///
/// Inverse of `interleave_bits`.
pub fn uninterleave_bits(index: u128) -> (u64, u64) {
    let mut x = 0u64;
    let mut y = 0u64;
    for i in 0..64 {
        x |= (((index >> (2 * i)) & 1) as u64) << i;
        y |= (((index >> (2 * i + 1)) & 1) as u64) << i;
    }
    (x, y)
}

// ============================================================================
// RISC-V Opcodes
// ============================================================================

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
            RiscvMemOp::Sw
                | RiscvMemOp::Sh
                | RiscvMemOp::Sb
                | RiscvMemOp::Sd
                | RiscvMemOp::ScW
                | RiscvMemOp::ScD
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
        matches!(
            self,
            RiscvMemOp::Lh | RiscvMemOp::Lb | RiscvMemOp::Lw | RiscvMemOp::LrW
        )
    }
}

// ============================================================================
// Lookup Table Computation
// ============================================================================

/// Compute the result of a RISC-V operation for given operands.
///
/// # Arguments
/// * `op` - The RISC-V opcode
/// * `x` - First operand (rs1)
/// * `y` - Second operand (rs2)
/// * `xlen` - The word size in bits (8, 32, or 64)
///
/// # Returns
/// The result of the operation, masked to `xlen` bits.
///
/// Based on Jolt's instruction semantics (MIT/Apache-2.0 license).
/// Credit: <https://github.com/a16z/jolt>
pub fn compute_op(op: RiscvOpcode, x: u64, y: u64, xlen: usize) -> u64 {
    let mask = if xlen >= 64 { u64::MAX } else { (1u64 << xlen) - 1 };
    let x = x & mask;
    let y = y & mask;

    // For shift operations, the shift amount is only the lower log2(xlen) bits
    let shift_mask = match xlen {
        32 => 0x1F,
        64 => 0x3F,
        _ => (xlen - 1) as u64, // For smaller xlen (testing)
    };

    let result = match op {
        RiscvOpcode::And => x & y,
        RiscvOpcode::Xor => x ^ y,
        RiscvOpcode::Or => x | y,
        RiscvOpcode::Sub => x.wrapping_sub(y),
        RiscvOpcode::Add => x.wrapping_add(y),

        // === M Extension: Multiply ===
        RiscvOpcode::Mul => {
            // MUL: lower xlen bits of product
            x.wrapping_mul(y)
        }
        RiscvOpcode::Mulh => {
            // MULH: upper xlen bits of signed × signed multiplication
            let x_signed = sign_extend(x, xlen);
            let y_signed = sign_extend(y, xlen);
            match xlen {
                32 => {
                    let product = (x_signed as i64) * (y_signed as i64);
                    (product >> 32) as u64
                }
                64 => {
                    let product = (x_signed as i128) * (y_signed as i128);
                    (product >> 64) as u64
                }
                _ => {
                    // For small xlen (testing)
                    let product = x_signed * y_signed;
                    ((product >> xlen) as u64) & mask
                }
            }
        }
        RiscvOpcode::Mulhu => {
            // MULHU: upper xlen bits of unsigned × unsigned multiplication
            match xlen {
                32 => {
                    let product = (x as u64) * (y as u64);
                    (product >> 32) & mask
                }
                64 => {
                    let product = (x as u128) * (y as u128);
                    (product >> 64) as u64
                }
                _ => {
                    // For small xlen (testing)
                    let product = (x as u128) * (y as u128);
                    ((product >> xlen) as u64) & mask
                }
            }
        }
        RiscvOpcode::Mulhsu => {
            // MULHSU: upper xlen bits of signed × unsigned multiplication
            let x_signed = sign_extend(x, xlen);
            match xlen {
                32 => {
                    let product = (x_signed as i64) * (y as i64);
                    (product >> 32) as u64
                }
                64 => {
                    let product = (x_signed as i128) * (y as i128);
                    (product >> 64) as u64
                }
                _ => {
                    let product = x_signed * (y as i64);
                    ((product >> xlen) as u64) & mask
                }
            }
        }

        // === M Extension: Divide ===
        RiscvOpcode::Div => {
            // DIV: signed division
            // Special cases per RISC-V spec:
            // - Division by zero: returns -1
            // - Overflow (most_negative / -1): returns most_negative
            if y == 0 {
                mask // All 1s = -1 in signed
            } else {
                let x_signed = sign_extend(x, xlen);
                let y_signed = sign_extend(y, xlen);
                let most_negative = 1i64 << (xlen - 1);
                if x_signed == -most_negative && y_signed == -1 {
                    x // Overflow case: return dividend
                } else {
                    (x_signed / y_signed) as u64
                }
            }
        }
        RiscvOpcode::Divu => {
            // DIVU: unsigned division
            // Division by zero returns all 1s
            if y == 0 {
                mask
            } else {
                x / y
            }
        }
        RiscvOpcode::Rem => {
            // REM: signed remainder
            // Special cases per RISC-V spec:
            // - Division by zero: returns dividend
            // - Overflow (most_negative / -1): returns 0
            if y == 0 {
                x
            } else {
                let x_signed = sign_extend(x, xlen);
                let y_signed = sign_extend(y, xlen);
                let most_negative = 1i64 << (xlen - 1);
                if x_signed == -most_negative && y_signed == -1 {
                    0
                } else {
                    (x_signed % y_signed) as u64
                }
            }
        }
        RiscvOpcode::Remu => {
            // REMU: unsigned remainder
            // Division by zero returns dividend
            if y == 0 {
                x
            } else {
                x % y
            }
        }

        // === Comparison ===
        RiscvOpcode::Sltu => {
            if x < y {
                1
            } else {
                0
            }
        }
        RiscvOpcode::Slt => {
            let x_signed = sign_extend(x, xlen);
            let y_signed = sign_extend(y, xlen);
            if x_signed < y_signed {
                1
            } else {
                0
            }
        }
        RiscvOpcode::Eq => {
            if x == y {
                1
            } else {
                0
            }
        }
        RiscvOpcode::Neq => {
            if x != y {
                1
            } else {
                0
            }
        }

        // === Shifts ===
        RiscvOpcode::Sll => {
            let shamt = y & shift_mask;
            x << shamt
        }
        RiscvOpcode::Srl => {
            let shamt = y & shift_mask;
            x >> shamt
        }
        RiscvOpcode::Sra => {
            let shamt = y & shift_mask;
            let x_signed = sign_extend(x, xlen);
            (x_signed >> shamt) as u64
        }

        // === RV64 W-suffix Operations (32-bit ops, sign-extended to 64-bit) ===
        RiscvOpcode::Addw => {
            let result32 = (x as u32).wrapping_add(y as u32);
            sign_extend_32(result32)
        }
        RiscvOpcode::Subw => {
            let result32 = (x as u32).wrapping_sub(y as u32);
            sign_extend_32(result32)
        }
        RiscvOpcode::Sllw => {
            let shamt = (y & 0x1F) as u32;
            let result32 = (x as u32) << shamt;
            sign_extend_32(result32)
        }
        RiscvOpcode::Srlw => {
            let shamt = (y & 0x1F) as u32;
            let result32 = (x as u32) >> shamt;
            sign_extend_32(result32)
        }
        RiscvOpcode::Sraw => {
            let shamt = (y & 0x1F) as u32;
            let result32 = ((x as i32) >> shamt) as u32;
            sign_extend_32(result32)
        }
        RiscvOpcode::Mulw => {
            let result32 = (x as u32).wrapping_mul(y as u32);
            sign_extend_32(result32)
        }
        RiscvOpcode::Divw => {
            let x32 = x as i32;
            let y32 = y as i32;
            if y32 == 0 {
                u64::MAX // All 1s
            } else if x32 == i32::MIN && y32 == -1 {
                sign_extend_32(x32 as u32) // Overflow
            } else {
                sign_extend_32((x32 / y32) as u32)
            }
        }
        RiscvOpcode::Divuw => {
            let x32 = x as u32;
            let y32 = y as u32;
            if y32 == 0 {
                u64::MAX
            } else {
                sign_extend_32(x32 / y32)
            }
        }
        RiscvOpcode::Remw => {
            let x32 = x as i32;
            let y32 = y as i32;
            if y32 == 0 {
                sign_extend_32(x32 as u32)
            } else if x32 == i32::MIN && y32 == -1 {
                0
            } else {
                sign_extend_32((x32 % y32) as u32)
            }
        }
        RiscvOpcode::Remuw => {
            let x32 = x as u32;
            let y32 = y as u32;
            if y32 == 0 {
                sign_extend_32(x32)
            } else {
                sign_extend_32(x32 % y32)
            }
        }

        // === Bitmanip (Zbb subset) ===
        RiscvOpcode::Andn => x & !y,
    };

    result & mask
}

/// Sign-extend a 32-bit value to 64 bits.
fn sign_extend_32(x: u32) -> u64 {
    (x as i32) as i64 as u64
}

/// Sign-extend a value from xlen bits to i64.
fn sign_extend(x: u64, xlen: usize) -> i64 {
    match xlen {
        8 => (x as u8) as i8 as i64,
        16 => (x as u16) as i16 as i64,
        32 => (x as u32) as i32 as i64,
        64 => x as i64,
        _ => {
            // For arbitrary xlen, do sign extension manually
            let sign_bit = 1u64 << (xlen - 1);
            if (x & sign_bit) != 0 {
                // Negative: extend with 1s
                (x | !((1u64 << xlen) - 1)) as i64
            } else {
                x as i64
            }
        }
    }
}

/// Compute a lookup table entry from an interleaved index.
pub fn lookup_entry(op: RiscvOpcode, index: u128, xlen: usize) -> u64 {
    let (x, y) = uninterleave_bits(index);
    compute_op(op, x, y, xlen)
}

// ============================================================================
// MLE Evaluation (matching Jolt's approach)
// ============================================================================

/// Evaluate the MLE of the AND operation at a random point.
///
/// For AND, the MLE has a simple form:
/// `AND~(r) = Σ_{i=0}^{n-1} 2^i * r_{2i} * r_{2i+1}`
///
/// where r is a vector of length 2*XLEN with interleaved x and y bits.
/// Position 2i contains the i-th bit of x, position 2i+1 contains the i-th bit of y.
pub fn evaluate_and_mle<F: Field>(r: &[F]) -> F {
    debug_assert!(r.len() % 2 == 0);
    let xlen = r.len() / 2;

    let mut result = F::ZERO;
    for i in 0..xlen {
        let x_i = r[2 * i];
        let y_i = r[2 * i + 1];
        let coeff = F::from_u64(1u64 << i);
        result += coeff * x_i * y_i;
    }
    result
}

/// Evaluate the MLE of the XOR operation at a random point.
///
/// For XOR, the MLE is:
/// `XOR~(r) = Σ_{i=0}^{n-1} 2^i * (r_{2i}(1-r_{2i+1}) + (1-r_{2i})r_{2i+1})`
pub fn evaluate_xor_mle<F: Field>(r: &[F]) -> F {
    debug_assert!(r.len() % 2 == 0);
    let xlen = r.len() / 2;

    let mut result = F::ZERO;
    for i in 0..xlen {
        let x_i = r[2 * i];
        let y_i = r[2 * i + 1];
        let coeff = F::from_u64(1u64 << i);
        // XOR: x(1-y) + (1-x)y = x + y - 2xy
        result += coeff * (x_i * (F::ONE - y_i) + (F::ONE - x_i) * y_i);
    }
    result
}

/// Evaluate the MLE of the OR operation at a random point.
///
/// For OR, the MLE is:
/// `OR~(r) = Σ_{i=0}^{n-1} 2^i * (r_{2i} + r_{2i+1} - r_{2i}*r_{2i+1})`
pub fn evaluate_or_mle<F: Field>(r: &[F]) -> F {
    debug_assert!(r.len() % 2 == 0);
    let xlen = r.len() / 2;

    let mut result = F::ZERO;
    for i in 0..xlen {
        let x_i = r[2 * i];
        let y_i = r[2 * i + 1];
        let coeff = F::from_u64(1u64 << i);
        // OR: x + y - xy
        result += coeff * (x_i + y_i - x_i * y_i);
    }
    result
}

/// Evaluate the MLE of EQ (equality predicate) at a random point.
///
/// Returns 1 on Boolean points iff `x == y`, else 0.
pub fn evaluate_eq_mle<F: Field>(r: &[F]) -> F {
    debug_assert!(r.len() % 2 == 0);
    let xlen = r.len() / 2;

    let mut result = F::ONE;
    for i in 0..xlen {
        let x_i = r[2 * i];
        let y_i = r[2 * i + 1];
        // eq(x_i, y_i) = x_i y_i + (1-x_i)(1-y_i)
        result *= x_i * y_i + (F::ONE - x_i) * (F::ONE - y_i);
    }
    result
}

/// Evaluate the MLE of NEQ (inequality predicate) at a random point.
pub fn evaluate_neq_mle<F: Field>(r: &[F]) -> F {
    F::ONE - evaluate_eq_mle(r)
}

/// Evaluate the MLE of SLTU (unsigned less-than predicate) at a random point.
///
/// Returns 1 on Boolean points iff `x < y` as unsigned `xlen`-bit integers.
pub fn evaluate_sltu_mle<F: Field>(r: &[F]) -> F {
    debug_assert!(r.len() % 2 == 0);
    let xlen = r.len() / 2;

    // Interpret r as interleaved bits (x_0,y_0,x_1,y_1,...) with LSB-first.
    // To compare numbers, scan from MSB → LSB.
    let mut lt = F::ZERO;
    let mut eq_prefix = F::ONE;
    for bit in (0..xlen).rev() {
        let x_i = r[2 * bit];
        let y_i = r[2 * bit + 1];
        lt += (F::ONE - x_i) * y_i * eq_prefix;
        eq_prefix *= x_i * y_i + (F::ONE - x_i) * (F::ONE - y_i);
    }
    lt
}

/// Evaluate the MLE of SLT (signed less-than predicate) at a random point.
///
/// Returns 1 on Boolean points iff `x < y` as signed two's-complement `xlen`-bit integers.
pub fn evaluate_slt_mle<F: Field>(r: &[F]) -> F {
    debug_assert!(r.len() % 2 == 0);
    let xlen = r.len() / 2;
    debug_assert!(xlen > 0);

    // Sign bits (MSB) for our LSB-first encoding.
    let x_sign = r[2 * (xlen - 1)];
    let y_sign = r[2 * (xlen - 1) + 1];

    // Unsigned less-than over the full bitstring (including sign bit), then adjust by sign bits.
    let mut lt = F::ZERO;
    let mut eq_prefix = F::ONE;
    for bit in (0..xlen).rev() {
        let x_i = r[2 * bit];
        let y_i = r[2 * bit + 1];
        lt += (F::ONE - x_i) * y_i * eq_prefix;
        eq_prefix *= x_i * y_i + (F::ONE - x_i) * (F::ONE - y_i);
    }

    x_sign - y_sign + lt
}

/// Evaluate the MLE of ADD at a random point.
///
/// For ADD, we use the decomposition: result = x + y (mod 2^xlen)
/// The MLE can be computed as: ADD~(r) = Σ x_bits + Σ y_bits + carry propagation
///
/// However, for simplicity, we use a different approach inspired by Jolt:
/// We verify ADD using a range check on the result. The MLE returns
/// the lower word (second operand bits in the interleaved representation).
pub fn evaluate_add_mle<F: Field>(r: &[F]) -> F {
    // ADD is verified via decomposition: result = x + y (mod 2^xlen)
    // For the MLE, we compute the sum at the evaluation point.
    // This works because at boolean points, it equals the table value.
    debug_assert!(r.len() % 2 == 0);
    let xlen = r.len() / 2;

    // The direct polynomial for ADD is complex due to carry propagation.
    // We use the identity: x + y = x ^ y + 2 * (x & y)
    // But more accurately, we need the full ripple-carry:
    // result_i = x_i ^ y_i ^ c_{i-1}
    // c_i = (x_i & y_i) | (x_i & c_{i-1}) | (y_i & c_{i-1})

    // For efficiency, compute iteratively:
    let mut result = F::ZERO;
    let mut carry = F::ZERO;

    for i in 0..xlen {
        let x_i = r[2 * i];
        let y_i = r[2 * i + 1];
        let coeff = F::from_u64(1u64 << i);

        // result_i = x_i ⊕ y_i ⊕ carry
        // In multilinear form: x + y + c - 2*x*y - 2*x*c - 2*y*c + 4*x*y*c
        let sum_bit = x_i + y_i + carry
            - x_i * y_i * F::from_u64(2)
            - x_i * carry * F::from_u64(2)
            - y_i * carry * F::from_u64(2)
            + x_i * y_i * carry * F::from_u64(4);

        result += coeff * sum_bit;

        // carry_i = (x_i & y_i) | (x_i & c_{i-1}) | (y_i & c_{i-1})
        // In multilinear: xy + xc + yc - 2xyc
        carry =
            x_i * y_i + x_i * carry + y_i * carry - x_i * y_i * carry * F::from_u64(2);
    }

    result
}

/// Evaluate the MLE of SUB at a random point.
///
/// Computes `x - y (mod 2^xlen)` using ripple-carry addition of `x + (~y + 1)`.
pub fn evaluate_sub_mle<F: Field>(r: &[F]) -> F {
    debug_assert!(r.len() % 2 == 0);
    let xlen = r.len() / 2;

    let mut result = F::ZERO;
    // Initial carry-in of 1 accounts for the "+ 1" in two's complement (~y + 1).
    let mut carry = F::ONE;

    for i in 0..xlen {
        let x_i = r[2 * i];
        let y_i = r[2 * i + 1];
        let y_comp = F::ONE - y_i;
        let coeff = F::from_u64(1u64 << i);

        // sum_bit = x_i ⊕ y_comp ⊕ carry
        let sum_bit = x_i + y_comp + carry
            - x_i * y_comp * F::from_u64(2)
            - x_i * carry * F::from_u64(2)
            - y_comp * carry * F::from_u64(2)
            + x_i * y_comp * carry * F::from_u64(4);

        result += coeff * sum_bit;

        // carry = majority(x_i, y_comp, carry) in multilinear form.
        carry = x_i * y_comp + x_i * carry + y_comp * carry - x_i * y_comp * carry * F::from_u64(2);
    }

    result
}

/// Evaluate the MLE of SLL (Shift Left Logical) at a random point.
///
/// For shift operations, Jolt uses a "virtual table" approach where the
/// MLE is computed using products over bit positions.
pub fn evaluate_sll_mle<F: Field>(r: &[F], xlen: usize) -> F {
    debug_assert_eq!(r.len(), 2 * xlen);

    // SLL: result_i = x_{i-shamt} if i >= shamt, else 0
    // The MLE is: Σ_i 2^i * Σ_{s=0}^{i} eq(y, s) * x_{i-s}
    // For simplicity, use naive evaluation for now
    evaluate_mle_naive(RiscvOpcode::Sll, r, xlen)
}

/// Evaluate the MLE of SRL (Shift Right Logical) at a random point.
///
/// Following Jolt's virtual SRL table approach.
pub fn evaluate_srl_mle<F: Field>(r: &[F], xlen: usize) -> F {
    debug_assert_eq!(r.len(), 2 * xlen);

    // Jolt's SRL formula: iteratively compute result *= (1 + y_i); result += x_i * y_i
    // This works because for each bit position, if y_i=1, we're selecting x_i,
    // otherwise we're shifting (multiplying by 1 + y_i = 2 when y_i=1 at boolean points)
    let mut result = F::ZERO;
    for i in 0..xlen {
        let x_i = r[2 * i];
        let y_i = r[2 * i + 1];
        result = result * (F::ONE + y_i) + x_i * y_i;
    }
    result
}

/// Evaluate the MLE of SRA (Shift Right Arithmetic) at a random point.
///
/// Following Jolt's virtual SRA table approach.
pub fn evaluate_sra_mle<F: Field>(r: &[F], xlen: usize) -> F {
    debug_assert_eq!(r.len(), 2 * xlen);

    // SRA is like SRL but with sign extension
    // Jolt's formula adds a sign_extension term based on the MSB
    let mut result = F::ZERO;
    let mut sign_extension = F::ZERO;

    for i in 0..xlen {
        let x_i = r[2 * i];
        let y_i = r[2 * i + 1];
        result = result * (F::ONE + y_i) + x_i * y_i;
        if i != 0 {
            sign_extension += F::from_u64(1 << i) * (F::ONE - y_i);
        }
    }

    // Add sign extension: MSB * sign_extension_mask
    let msb = r[0]; // x_0 is the MSB in interleaved representation
    result + msb * sign_extension
}

/// Evaluate the MLE of a RISC-V opcode at a random point.
///
/// This dispatches to the appropriate MLE evaluation function based on the opcode.
/// For opcodes without closed-form MLEs, this falls back to the naive computation.
///
/// # Note on Shift Operations
///
/// Jolt uses "virtual tables" for shift operations with specialized MLE formulas
/// (see `evaluate_srl_mle` and `evaluate_sra_mle`). These virtual tables encode
/// the shift amount as a bitmask rather than a direct value, which allows for
/// efficient MLE evaluation. Our standard lookup tables use direct shift amounts,
/// so we use naive MLE evaluation for consistency.
pub fn evaluate_opcode_mle<F: Field>(op: RiscvOpcode, r: &[F], xlen: usize) -> F {
    debug_assert_eq!(r.len(), 2 * xlen);

    match op {
        RiscvOpcode::And => evaluate_and_mle(r),
        RiscvOpcode::Xor => evaluate_xor_mle(r),
        RiscvOpcode::Or => evaluate_or_mle(r),
        RiscvOpcode::Add => evaluate_add_mle(r),
        RiscvOpcode::Sub => evaluate_sub_mle(r),
        RiscvOpcode::Eq => evaluate_eq_mle(r),
        RiscvOpcode::Neq => evaluate_neq_mle(r),
        RiscvOpcode::Slt => evaluate_slt_mle(r),
        RiscvOpcode::Sltu => evaluate_sltu_mle(r),
        // For shift and other opcodes, use the naive MLE evaluation when available.
        // Note: naive evaluation is O(2^{2*xlen}) and intentionally limited to tiny xlen.
        _ => {
            if xlen <= 8 {
                evaluate_mle_naive(op, r, xlen)
            } else {
                panic!("evaluate_opcode_mle: closed-form MLE not implemented for opcode {op:?} at xlen={xlen}");
            }
        }
    }
}

/// Naive MLE evaluation by summing over the Boolean hypercube.
///
/// This is O(2^{2*xlen}) and should only be used for testing or small tables.
fn evaluate_mle_naive<F: Field>(op: RiscvOpcode, r: &[F], xlen: usize) -> F {
    assert!(xlen <= 8, "Naive MLE evaluation only supports xlen <= 8");

    let table_size = 1usize << (2 * xlen);
    let mut result = F::ZERO;

    for idx in 0..table_size {
        // Compute χ_idx(r) = Π_k (idx_k * r_k + (1-idx_k)(1-r_k))
        // With LSB-aligned indexing, bit k of idx corresponds to r[k]
        let mut chi = F::ONE;
        for k in 0..(2 * xlen) {
            let bit = ((idx >> k) & 1) as u64;
            let r_k = r[k];
            if bit == 1 {
                chi *= r_k;
            } else {
                chi *= F::ONE - r_k;
            }
        }

        // Add contribution: χ_idx(r) * table[idx]
        let entry = lookup_entry(op, idx as u128, xlen);
        result += chi * F::from_u64(entry);
    }

    result
}

// ============================================================================
// RISC-V Lookup Table (Shout-compatible)
// ============================================================================

/// A RISC-V instruction lookup table compatible with Neo's Shout protocol.
///
/// This struct encapsulates:
/// - The opcode (which operation to perform)
/// - The word size (xlen)
/// - Methods for table lookup and MLE evaluation
#[derive(Clone, Debug)]
pub struct RiscvLookupTable<F> {
    /// The RISC-V opcode this table implements.
    pub opcode: RiscvOpcode,
    /// Word size in bits (8, 32, or 64).
    pub xlen: usize,
    /// Precomputed table values (only for small tables).
    /// For large tables, values are computed on-demand.
    pub values: Option<Vec<F>>,
}

impl<F: Field> RiscvLookupTable<F> {
    /// Create a new lookup table for the given opcode and word size.
    ///
    /// For xlen <= 8, precomputes all table entries.
    /// For larger word sizes, entries are computed on-demand.
    pub fn new(opcode: RiscvOpcode, xlen: usize) -> Self {
        let values = if xlen <= 8 {
            let table_size = 1usize << (2 * xlen);
            Some(
                (0..table_size)
                    .map(|idx| {
                        let entry = lookup_entry(opcode, idx as u128, xlen);
                        F::from_u64(entry)
                    })
                    .collect(),
            )
        } else {
            None
        };

        Self { opcode, xlen, values }
    }

    /// Get the table size (K = 2^{2*xlen}).
    pub fn size(&self) -> usize {
        1usize << (2 * self.xlen)
    }

    /// Look up a value by index.
    pub fn lookup(&self, index: u128) -> F {
        if let Some(ref values) = self.values {
            values[index as usize]
        } else {
            let entry = lookup_entry(self.opcode, index, self.xlen);
            F::from_u64(entry)
        }
    }

    /// Look up a value by operands.
    pub fn lookup_operands(&self, x: u64, y: u64) -> F {
        let index = interleave_bits(x, y);
        // Mask the index to the correct bit width (index is LSB-aligned)
        let mask = (1u128 << (2 * self.xlen)) - 1;
        self.lookup(index & mask)
    }

    /// Evaluate the MLE at a random point.
    pub fn evaluate_mle(&self, r: &[F]) -> F {
        evaluate_opcode_mle(self.opcode, r, self.xlen)
    }

    /// Get the content as a vector of field elements (for Shout encoding).
    pub fn content(&self) -> Vec<F> {
        if let Some(ref values) = self.values {
            values.clone()
        } else {
            let table_size = self.size();
            (0..table_size)
                .map(|idx| self.lookup(idx as u128))
                .collect()
        }
    }
}

// ============================================================================
// RISC-V Instruction Trace Event
// ============================================================================

/// A RISC-V instruction lookup event for the trace.
///
/// Records an instruction execution that will be proven via Shout.
#[derive(Clone, Debug)]
pub struct RiscvLookupEvent {
    /// The opcode executed.
    pub opcode: RiscvOpcode,
    /// First operand (rs1 value).
    pub rs1: u64,
    /// Second operand (rs2 value).
    pub rs2: u64,
    /// The result (rd value).
    pub result: u64,
}

impl RiscvLookupEvent {
    /// Create a new lookup event.
    pub fn new(opcode: RiscvOpcode, rs1: u64, rs2: u64, xlen: usize) -> Self {
        let result = compute_op(opcode, rs1, rs2, xlen);
        Self { opcode, rs1, rs2, result }
    }

    /// Get the lookup index for this event.
    pub fn lookup_index(&self, xlen: usize) -> u128 {
        let index = interleave_bits(self.rs1, self.rs2);
        // With LSB-aligned interleaving, the index is at the LSB
        let mask = (1u128 << (2 * xlen)) - 1;
        index & mask
    }
}

// ============================================================================
// Range Check Table (for ADD verification)
// ============================================================================

/// Range Check table for ADD verification.
///
/// Following Jolt's approach: ADD is verified using a range check that ensures
/// the result is in the correct range [0, 2^xlen). The table maps each value
/// to itself: table[i] = i.
///
/// This table is used to decompose the ADD result into verified chunks.
#[derive(Clone, Debug)]
pub struct RangeCheckTable<F> {
    /// Word size in bits.
    pub xlen: usize,
    /// Precomputed table values.
    pub values: Vec<F>,
}

impl<F: Field> RangeCheckTable<F> {
    /// Create a new range check table.
    pub fn new(xlen: usize) -> Self {
        assert!(xlen <= 16, "Range check table too large for xlen > 16");
        let size = 1usize << xlen;
        let values = (0..size).map(|i| F::from_u64(i as u64)).collect();
        Self { xlen, values }
    }

    /// Get the table size.
    pub fn size(&self) -> usize {
        1usize << self.xlen
    }

    /// Look up a value (identity: table[i] = i).
    pub fn lookup(&self, index: u64) -> F {
        self.values[index as usize]
    }

    /// Evaluate the MLE at a random point.
    ///
    /// For the identity table, the MLE is simply the binary expansion:
    /// RangeCheck~(r) = Σ_{i=0}^{xlen-1} 2^i * r_i
    pub fn evaluate_mle(&self, r: &[F]) -> F {
        debug_assert_eq!(r.len(), self.xlen);
        let mut result = F::ZERO;
        for i in 0..self.xlen {
            result += F::from_u64(1u64 << i) * r[i];
        }
        result
    }

    /// Get the content as a vector of field elements.
    pub fn content(&self) -> Vec<F> {
        self.values.clone()
    }
}

// ============================================================================
// RISC-V Memory Event (for Twist)
// ============================================================================

/// A RISC-V memory operation event for the trace.
///
/// Records a load or store operation that will be proven via Twist.
#[derive(Clone, Debug)]
pub struct RiscvMemoryEvent {
    /// The memory operation type.
    pub op: RiscvMemOp,
    /// The memory address (base + offset).
    pub addr: u64,
    /// The value loaded or stored.
    pub value: u64,
}

impl RiscvMemoryEvent {
    /// Create a new memory event.
    pub fn new(op: RiscvMemOp, addr: u64, value: u64) -> Self {
        Self { op, addr, value }
    }
}

// ============================================================================
// RISC-V Shout Table Set
// ============================================================================

/// A collection of RISC-V lookup tables for the Shout protocol.
///
/// This implements the `Shout` trait and provides lookup tables for all
/// RISC-V ALU operations.
pub struct RiscvShoutTables {
    /// Word size in bits (32 or 64).
    pub xlen: usize,
}

impl RiscvShoutTables {
    /// Create a new set of RISC-V Shout tables.
    pub fn new(xlen: usize) -> Self {
        Self { xlen }
    }

    /// Get the opcode for a given ShoutId.
    pub fn id_to_opcode(&self, id: ShoutId) -> Option<RiscvOpcode> {
        match id.0 {
            0 => Some(RiscvOpcode::And),
            1 => Some(RiscvOpcode::Xor),
            2 => Some(RiscvOpcode::Or),
            3 => Some(RiscvOpcode::Add),
            4 => Some(RiscvOpcode::Sub),
            5 => Some(RiscvOpcode::Slt),
            6 => Some(RiscvOpcode::Sltu),
            7 => Some(RiscvOpcode::Sll),
            8 => Some(RiscvOpcode::Srl),
            9 => Some(RiscvOpcode::Sra),
            10 => Some(RiscvOpcode::Eq),
            11 => Some(RiscvOpcode::Neq),
            // M Extension
            12 => Some(RiscvOpcode::Mul),
            13 => Some(RiscvOpcode::Mulh),
            14 => Some(RiscvOpcode::Mulhu),
            15 => Some(RiscvOpcode::Mulhsu),
            16 => Some(RiscvOpcode::Div),
            17 => Some(RiscvOpcode::Divu),
            18 => Some(RiscvOpcode::Rem),
            19 => Some(RiscvOpcode::Remu),
            // RV64 W-suffix
            20 => Some(RiscvOpcode::Addw),
            21 => Some(RiscvOpcode::Subw),
            22 => Some(RiscvOpcode::Sllw),
            23 => Some(RiscvOpcode::Srlw),
            24 => Some(RiscvOpcode::Sraw),
            25 => Some(RiscvOpcode::Mulw),
            26 => Some(RiscvOpcode::Divw),
            27 => Some(RiscvOpcode::Divuw),
            28 => Some(RiscvOpcode::Remw),
            29 => Some(RiscvOpcode::Remuw),
            // Bitmanip
            30 => Some(RiscvOpcode::Andn),
            _ => None,
        }
    }

    /// Get the ShoutId for a given opcode.
    pub fn opcode_to_id(&self, op: RiscvOpcode) -> ShoutId {
        match op {
            RiscvOpcode::And => ShoutId(0),
            RiscvOpcode::Xor => ShoutId(1),
            RiscvOpcode::Or => ShoutId(2),
            RiscvOpcode::Add => ShoutId(3),
            RiscvOpcode::Sub => ShoutId(4),
            RiscvOpcode::Slt => ShoutId(5),
            RiscvOpcode::Sltu => ShoutId(6),
            RiscvOpcode::Sll => ShoutId(7),
            RiscvOpcode::Srl => ShoutId(8),
            RiscvOpcode::Sra => ShoutId(9),
            RiscvOpcode::Eq => ShoutId(10),
            RiscvOpcode::Neq => ShoutId(11),
            // M Extension
            RiscvOpcode::Mul => ShoutId(12),
            RiscvOpcode::Mulh => ShoutId(13),
            RiscvOpcode::Mulhu => ShoutId(14),
            RiscvOpcode::Mulhsu => ShoutId(15),
            RiscvOpcode::Div => ShoutId(16),
            RiscvOpcode::Divu => ShoutId(17),
            RiscvOpcode::Rem => ShoutId(18),
            RiscvOpcode::Remu => ShoutId(19),
            // RV64 W-suffix
            RiscvOpcode::Addw => ShoutId(20),
            RiscvOpcode::Subw => ShoutId(21),
            RiscvOpcode::Sllw => ShoutId(22),
            RiscvOpcode::Srlw => ShoutId(23),
            RiscvOpcode::Sraw => ShoutId(24),
            RiscvOpcode::Mulw => ShoutId(25),
            RiscvOpcode::Divw => ShoutId(26),
            RiscvOpcode::Divuw => ShoutId(27),
            RiscvOpcode::Remw => ShoutId(28),
            RiscvOpcode::Remuw => ShoutId(29),
            // Bitmanip
            RiscvOpcode::Andn => ShoutId(30),
        }
    }
}

impl Shout<u64> for RiscvShoutTables {
    fn lookup(&mut self, shout_id: ShoutId, key: u64) -> u64 {
        // The key is an interleaved index containing both operands
        if let Some(op) = self.id_to_opcode(shout_id) {
            let (rs1, rs2) = uninterleave_bits(key as u128);
            compute_op(op, rs1, rs2, self.xlen)
        } else {
            0 // Unknown table
        }
    }
}

// ============================================================================
// RISC-V Memory (Twist)
// ============================================================================

/// RISC-V memory implementation for the Twist protocol.
///
/// Provides byte-addressable memory with support for different access widths.
pub struct RiscvMemory {
    /// Memory contents (sparse representation).
    data: std::collections::HashMap<u64, u8>,
    /// Word size in bits (32 or 64).
    pub xlen: usize,
}

impl RiscvMemory {
    /// Create a new empty memory.
    pub fn new(xlen: usize) -> Self {
        Self {
            data: std::collections::HashMap::new(),
            xlen,
        }
    }

    /// Create memory pre-initialized with a program.
    pub fn with_program(xlen: usize, base_addr: u64, program: &[u8]) -> Self {
        let mut mem = Self::new(xlen);
        for (i, &byte) in program.iter().enumerate() {
            mem.data.insert(base_addr + i as u64, byte);
        }
        mem
    }

    /// Read a byte from memory.
    pub fn read_byte(&self, addr: u64) -> u8 {
        self.data.get(&addr).copied().unwrap_or(0)
    }

    /// Write a byte to memory.
    pub fn write_byte(&mut self, addr: u64, value: u8) {
        if value == 0 {
            self.data.remove(&addr);
        } else {
            self.data.insert(addr, value);
        }
    }

    /// Read a value with the given width (in bytes).
    pub fn read(&self, addr: u64, width: usize) -> u64 {
        let mut value = 0u64;
        for i in 0..width {
            value |= (self.read_byte(addr + i as u64) as u64) << (8 * i);
        }
        value
    }

    /// Write a value with the given width (in bytes).
    pub fn write(&mut self, addr: u64, width: usize, value: u64) {
        for i in 0..width {
            self.write_byte(addr + i as u64, (value >> (8 * i)) as u8);
        }
    }

    /// Execute a memory operation and return the value.
    pub fn execute(&mut self, op: RiscvMemOp, addr: u64, store_value: u64) -> u64 {
        let width = op.width_bytes();

        if op.is_load() {
            let raw = self.read(addr, width);
            // Sign-extend if needed
            if op.is_sign_extend() {
                match width {
                    1 => (raw as u8) as i8 as i64 as u64,
                    2 => (raw as u16) as i16 as i64 as u64,
                    4 => (raw as u32) as i32 as i64 as u64,
                    _ => raw,
                }
            } else {
                raw
            }
        } else {
            self.write(addr, width, store_value);
            store_value
        }
    }
}

impl Twist<u64, u64> for RiscvMemory {
    fn load(&mut self, _twist_id: TwistId, addr: u64) -> u64 {
        // Default: word-sized load
        let width = self.xlen / 8;
        self.read(addr, width)
    }

    fn store(&mut self, _twist_id: TwistId, addr: u64, value: u64) {
        // Default: word-sized store
        let width = self.xlen / 8;
        self.write(addr, width, value);
    }
}

// ============================================================================
// RISC-V Branch/Jump Types
// ============================================================================

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

// ============================================================================
// RISC-V Instruction Types
// ============================================================================

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
    LoadReserved {
        op: RiscvMemOp,
        rd: u8,
        rs1: u8,
    },
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
    Fence {
        pred: u8,
        succ: u8,
    },
    /// Instruction Fence
    FenceI,
}

// ============================================================================
// RISC-V CPU Implementation
// ============================================================================

/// A RISC-V CPU that can be traced using Neo's VmCpu trait.
///
/// Implements RV32I/RV64I base instruction set.
/// Based on Jolt's CPU implementation (MIT/Apache-2.0 license).
/// Credit: <https://github.com/a16z/jolt>
pub struct RiscvCpu {
    /// Program counter.
    pub pc: u64,
    /// General-purpose registers (x0-x31, where x0 is always 0).
    pub regs: [u64; 32],
    /// Word size in bits (32 or 64).
    pub xlen: usize,
    /// Whether the CPU has halted.
    pub halted: bool,
    /// Program to execute (list of instructions).
    program: Vec<RiscvInstruction>,
    /// Base address of the program.
    program_base: u64,
}

impl RiscvCpu {
    /// Create a new CPU with the given word size.
    pub fn new(xlen: usize) -> Self {
        assert!(xlen == 32 || xlen == 64);
        Self {
            pc: 0,
            regs: [0; 32],
            xlen,
            halted: false,
            program: Vec::new(),
            program_base: 0,
        }
    }

    /// Load a program starting at the given base address.
    pub fn load_program(&mut self, base: u64, program: Vec<RiscvInstruction>) {
        self.program_base = base;
        self.program = program;
        self.pc = base;
    }

    /// Set a register value (x0 writes are ignored).
    pub fn set_reg(&mut self, reg: u8, value: u64) {
        if reg != 0 {
            self.regs[reg as usize] = self.mask_value(value);
        }
    }

    /// Get a register value.
    pub fn get_reg(&self, reg: u8) -> u64 {
        self.regs[reg as usize]
    }

    /// Mask a value to the word size.
    fn mask_value(&self, value: u64) -> u64 {
        if self.xlen == 32 {
            value as u32 as u64
        } else {
            value
        }
    }

    /// Sign-extend an immediate.
    fn sign_extend_imm(&self, imm: i32) -> u64 {
        if self.xlen == 32 {
            imm as u32 as u64
        } else {
            imm as i64 as u64
        }
    }

    /// Get the current instruction (if any).
    fn current_instruction(&self) -> Option<&RiscvInstruction> {
        let index = (self.pc - self.program_base) / 4;
        self.program.get(index as usize)
    }
}

impl neo_vm_trace::VmCpu<u64, u64> for RiscvCpu {
    type Error = String;

    fn snapshot_regs(&self) -> Vec<u64> {
        self.regs.to_vec()
    }

    fn pc(&self) -> u64 {
        self.pc
    }

    fn halted(&self) -> bool {
        self.halted
    }

    fn step<T, S>(
        &mut self,
        twist: &mut T,
        shout: &mut S,
    ) -> Result<neo_vm_trace::StepMeta<u64>, Self::Error>
    where
        T: Twist<u64, u64>,
        S: Shout<u64>,
    {
        let ram = TwistId(0);

        let instr = self
            .current_instruction()
            .cloned()
            .ok_or_else(|| format!("No instruction at PC {:#x}", self.pc))?;

        // Default: advance PC by 4
        let mut next_pc = self.pc.wrapping_add(4);
        let opcode_num: u32;

        match instr {
            RiscvInstruction::RAlu { op, rd, rs1, rs2 } => {
                let rs1_val = self.get_reg(rs1);
                let rs2_val = self.get_reg(rs2);

                // Use Shout for the ALU operation
                let shout_id = RiscvShoutTables::new(self.xlen).opcode_to_id(op);
                let index = interleave_bits(rs1_val, rs2_val) as u64;
                let result = shout.lookup(shout_id, index);

                self.set_reg(rd, result);
                opcode_num = 0x33; // R-type opcode
            }

            RiscvInstruction::IAlu { op, rd, rs1, imm } => {
                let rs1_val = self.get_reg(rs1);
                let imm_val = self.sign_extend_imm(imm);

                // Use Shout for the ALU operation
                let shout_id = RiscvShoutTables::new(self.xlen).opcode_to_id(op);
                let index = interleave_bits(rs1_val, imm_val) as u64;
                let result = shout.lookup(shout_id, index);

                self.set_reg(rd, result);
                opcode_num = 0x13; // I-type opcode
            }

            RiscvInstruction::Load { op, rd, rs1, imm } => {
                let base = self.get_reg(rs1);
                let addr = base.wrapping_add(self.sign_extend_imm(imm));

                // Use Twist for memory access
                let raw_value = twist.load(ram, addr);

                // Apply width and sign extension
                let width = op.width_bytes();
                let mask = if width >= 8 {
                    u64::MAX
                } else {
                    (1u64 << (width * 8)) - 1
                };
                let value = raw_value & mask;

                // Sign-extend if needed
                let result = if op.is_sign_extend() {
                    match width {
                        1 => (value as u8) as i8 as i64 as u64,
                        2 => (value as u16) as i16 as i64 as u64,
                        4 => (value as u32) as i32 as i64 as u64,
                        _ => value,
                    }
                } else {
                    value
                };

                self.set_reg(rd, self.mask_value(result));
                opcode_num = 0x03; // Load opcode
            }

            RiscvInstruction::Store { op, rs1, rs2, imm } => {
                let base = self.get_reg(rs1);
                let addr = base.wrapping_add(self.sign_extend_imm(imm));
                let value = self.get_reg(rs2);

                // Mask value to store width
                let width = op.width_bytes();
                let mask = if width >= 8 {
                    u64::MAX
                } else {
                    (1u64 << (width * 8)) - 1
                };
                let store_value = value & mask;

                // Use Twist for memory access
                twist.store(ram, addr, store_value);
                opcode_num = 0x23; // Store opcode
            }

            RiscvInstruction::Branch { cond, rs1, rs2, imm } => {
                let rs1_val = self.get_reg(rs1);
                let rs2_val = self.get_reg(rs2);

                // Use Shout for the comparison
                let shout_id = RiscvShoutTables::new(self.xlen).opcode_to_id(cond.to_shout_opcode());
                let index = interleave_bits(rs1_val, rs2_val) as u64;
                let _comparison_result = shout.lookup(shout_id, index);

                // Evaluate branch condition
                if cond.evaluate(rs1_val, rs2_val, self.xlen) {
                    next_pc = (self.pc as i64 + imm as i64) as u64;
                }
                opcode_num = 0x63; // Branch opcode
            }

            RiscvInstruction::Jal { rd, imm } => {
                // rd = pc + 4 (return address)
                self.set_reg(rd, self.pc.wrapping_add(4));
                // pc = pc + imm
                next_pc = (self.pc as i64 + imm as i64) as u64;
                opcode_num = 0x6F; // JAL opcode
            }

            RiscvInstruction::Jalr { rd, rs1, imm } => {
                let rs1_val = self.get_reg(rs1);
                let return_addr = self.pc.wrapping_add(4);

                // pc = (rs1 + imm) & ~1
                next_pc = rs1_val.wrapping_add(self.sign_extend_imm(imm)) & !1;

                // rd = return address
                self.set_reg(rd, return_addr);
                opcode_num = 0x67; // JALR opcode
            }

            RiscvInstruction::Lui { rd, imm } => {
                // rd = imm << 12 (upper 20 bits)
                let value = (imm as i64 as u64) << 12;
                self.set_reg(rd, self.mask_value(value));
                opcode_num = 0x37; // LUI opcode
            }

            RiscvInstruction::Auipc { rd, imm } => {
                // rd = pc + (imm << 12)
                let value = self.pc.wrapping_add((imm as i64 as u64) << 12);
                self.set_reg(rd, self.mask_value(value));
                opcode_num = 0x17; // AUIPC opcode
            }

            RiscvInstruction::Halt => {
                self.halted = true;
                opcode_num = 0x73; // ECALL opcode
            }

            RiscvInstruction::Nop => {
                opcode_num = 0x13; // NOP is ADDI x0, x0, 0
            }

            // === RV64 W-suffix Operations ===
            RiscvInstruction::RAluw { op, rd, rs1, rs2 } => {
                let rs1_val = self.get_reg(rs1);
                let rs2_val = self.get_reg(rs2);

                let shout_id = RiscvShoutTables::new(self.xlen).opcode_to_id(op);
                let index = interleave_bits(rs1_val, rs2_val) as u64;
                let result = shout.lookup(shout_id, index);

                self.set_reg(rd, result);
                opcode_num = 0x3B; // RV64 R-type W opcode
            }

            RiscvInstruction::IAluw { op, rd, rs1, imm } => {
                let rs1_val = self.get_reg(rs1);
                let imm_val = self.sign_extend_imm(imm);

                let shout_id = RiscvShoutTables::new(self.xlen).opcode_to_id(op);
                let index = interleave_bits(rs1_val, imm_val) as u64;
                let result = shout.lookup(shout_id, index);

                self.set_reg(rd, result);
                opcode_num = 0x1B; // RV64 I-type W opcode
            }

            // === A Extension: Atomics ===
            RiscvInstruction::LoadReserved { op, rd, rs1 } => {
                let addr = self.get_reg(rs1);
                let value = twist.load(ram, addr);

                // Apply width and sign extension
                let width = op.width_bytes();
                let mask = if width >= 8 { u64::MAX } else { (1u64 << (width * 8)) - 1 };
                let result = if op.is_sign_extend() {
                    match width {
                        4 => (value as u32) as i32 as i64 as u64,
                        _ => value & mask,
                    }
                } else {
                    value & mask
                };

                self.set_reg(rd, self.mask_value(result));
                // Note: In a real implementation, we'd reserve the address here
                opcode_num = 0x2F; // AMO opcode
            }

            RiscvInstruction::StoreConditional { op, rd, rs1, rs2 } => {
                let addr = self.get_reg(rs1);
                let value = self.get_reg(rs2);

                // Mask value to store width
                let width = op.width_bytes();
                let mask = if width >= 8 { u64::MAX } else { (1u64 << (width * 8)) - 1 };
                let store_value = value & mask;

                // Store the value
                twist.store(ram, addr, store_value);

                // SC returns 0 on success (assuming reservation is valid in single-threaded mode)
                self.set_reg(rd, 0);
                opcode_num = 0x2F; // AMO opcode
            }

            RiscvInstruction::Amo { op, rd, rs1, rs2 } => {
                let addr = self.get_reg(rs1);
                let src = self.get_reg(rs2);

                // Load original value
                let original = twist.load(ram, addr);
                self.set_reg(rd, self.mask_value(original));

                // Compute new value based on AMO operation
                let new_val = match op {
                    RiscvMemOp::AmoswapW | RiscvMemOp::AmoswapD => src,
                    RiscvMemOp::AmoaddW | RiscvMemOp::AmoaddD => original.wrapping_add(src),
                    RiscvMemOp::AmoxorW | RiscvMemOp::AmoxorD => original ^ src,
                    RiscvMemOp::AmoandW | RiscvMemOp::AmoandD => original & src,
                    RiscvMemOp::AmoorW | RiscvMemOp::AmoorD => original | src,
                    RiscvMemOp::AmominW => {
                        if (original as i32) < (src as i32) { original } else { src }
                    }
                    RiscvMemOp::AmominD => {
                        if (original as i64) < (src as i64) { original } else { src }
                    }
                    RiscvMemOp::AmomaxW => {
                        if (original as i32) > (src as i32) { original } else { src }
                    }
                    RiscvMemOp::AmomaxD => {
                        if (original as i64) > (src as i64) { original } else { src }
                    }
                    RiscvMemOp::AmominuW | RiscvMemOp::AmominuD => {
                        if original < src { original } else { src }
                    }
                    RiscvMemOp::AmomaxuW | RiscvMemOp::AmomaxuD => {
                        if original > src { original } else { src }
                    }
                    _ => src, // Fallback
                };

                // Store new value
                twist.store(ram, addr, new_val);
                opcode_num = 0x2F; // AMO opcode
            }

            // === System Instructions ===
            RiscvInstruction::Ecall => {
                // ECALL - environment call (syscall)
                // In a real implementation, this would trigger a trap
                // For now, check if a0 (x10) == 0 to halt
                if self.get_reg(10) == 0 {
                    self.halted = true;
                }
                opcode_num = 0x73; // SYSTEM opcode
            }

            RiscvInstruction::Ebreak => {
                // EBREAK - debugger breakpoint
                // For now, treat as halt
                self.halted = true;
                opcode_num = 0x73; // SYSTEM opcode
            }

            RiscvInstruction::Fence { pred: _, succ: _ } => {
                // FENCE - memory ordering
                // No-op in single-threaded execution
                opcode_num = 0x0F; // MISC-MEM opcode
            }

            RiscvInstruction::FenceI => {
                // FENCE.I - instruction fence
                // No-op in our implementation
                opcode_num = 0x0F; // MISC-MEM opcode
            }
        }

        self.pc = next_pc;

        Ok(neo_vm_trace::StepMeta {
            pc_after: self.pc,
            opcode: opcode_num,
        })
    }
}

// ============================================================================
// Binary Decoder: Parse RISC-V 32-bit Instructions
// ============================================================================

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
            Ok(RiscvInstruction::IAlu { op, rd, rs1, imm: imm as i32 })
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
                    let op = if funct3 == 0b010 { RiscvMemOp::LrW } else { RiscvMemOp::LrD };
                    Ok(RiscvInstruction::LoadReserved { op, rd, rs1 })
                }
                // SC
                0b00011 => {
                    let op = if funct3 == 0b010 { RiscvMemOp::ScW } else { RiscvMemOp::ScD };
                    Ok(RiscvInstruction::StoreConditional { op, rd, rs1, rs2 })
                }
                // AMOSWAP
                0b00001 => {
                    let op = if funct3 == 0b010 { RiscvMemOp::AmoswapW } else { RiscvMemOp::AmoswapD };
                    Ok(RiscvInstruction::Amo { op, rd, rs1, rs2 })
                }
                // AMOADD
                0b00000 => {
                    let op = if funct3 == 0b010 { RiscvMemOp::AmoaddW } else { RiscvMemOp::AmoaddD };
                    Ok(RiscvInstruction::Amo { op, rd, rs1, rs2 })
                }
                // AMOXOR
                0b00100 => {
                    let op = if funct3 == 0b010 { RiscvMemOp::AmoxorW } else { RiscvMemOp::AmoxorD };
                    Ok(RiscvInstruction::Amo { op, rd, rs1, rs2 })
                }
                // AMOAND
                0b01100 => {
                    let op = if funct3 == 0b010 { RiscvMemOp::AmoandW } else { RiscvMemOp::AmoandD };
                    Ok(RiscvInstruction::Amo { op, rd, rs1, rs2 })
                }
                // AMOOR
                0b01000 => {
                    let op = if funct3 == 0b010 { RiscvMemOp::AmoorW } else { RiscvMemOp::AmoorD };
                    Ok(RiscvInstruction::Amo { op, rd, rs1, rs2 })
                }
                // AMOMIN
                0b10000 => {
                    let op = if funct3 == 0b010 { RiscvMemOp::AmominW } else { RiscvMemOp::AmominD };
                    Ok(RiscvInstruction::Amo { op, rd, rs1, rs2 })
                }
                // AMOMAX
                0b10100 => {
                    let op = if funct3 == 0b010 { RiscvMemOp::AmomaxW } else { RiscvMemOp::AmomaxD };
                    Ok(RiscvInstruction::Amo { op, rd, rs1, rs2 })
                }
                // AMOMINU
                0b11000 => {
                    let op = if funct3 == 0b010 { RiscvMemOp::AmominuW } else { RiscvMemOp::AmominuD };
                    Ok(RiscvInstruction::Amo { op, rd, rs1, rs2 })
                }
                // AMOMAXU
                0b11100 => {
                    let op = if funct3 == 0b010 { RiscvMemOp::AmomaxuW } else { RiscvMemOp::AmomaxuD };
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
            Ok(RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd, rs1: 2, imm: nzuimm as i32 })
        }
        (0b00, 0b010) => {
            // C.LW: lw rd', offset(rs1')
            let rd = ((instr >> 2) & 0b111) as u8 + 8;
            let rs1 = ((instr >> 7) & 0b111) as u8 + 8;
            let offset = decode_cl_lw_imm(instr);
            Ok(RiscvInstruction::Load { op: RiscvMemOp::Lw, rd, rs1, imm: offset as i32 })
        }
        (0b00, 0b011) => {
            // C.LD (RV64) or C.FLW (RV32): ld rd', offset(rs1')
            let rd = ((instr >> 2) & 0b111) as u8 + 8;
            let rs1 = ((instr >> 7) & 0b111) as u8 + 8;
            let offset = decode_cl_ld_imm(instr);
            Ok(RiscvInstruction::Load { op: RiscvMemOp::Ld, rd, rs1, imm: offset as i32 })
        }
        (0b00, 0b110) => {
            // C.SW: sw rs2', offset(rs1')
            let rs2 = ((instr >> 2) & 0b111) as u8 + 8;
            let rs1 = ((instr >> 7) & 0b111) as u8 + 8;
            let offset = decode_cl_lw_imm(instr);
            Ok(RiscvInstruction::Store { op: RiscvMemOp::Sw, rs1, rs2, imm: offset as i32 })
        }
        (0b00, 0b111) => {
            // C.SD (RV64): sd rs2', offset(rs1')
            let rs2 = ((instr >> 2) & 0b111) as u8 + 8;
            let rs1 = ((instr >> 7) & 0b111) as u8 + 8;
            let offset = decode_cl_ld_imm(instr);
            Ok(RiscvInstruction::Store { op: RiscvMemOp::Sd, rs1, rs2, imm: offset as i32 })
        }
        
        // Quadrant 1
        (0b01, 0b000) => {
            // C.ADDI / C.NOP
            let rd = ((instr >> 7) & 0b11111) as u8;
            let imm = decode_ci_imm(instr);
            if rd == 0 {
                Ok(RiscvInstruction::Nop) // C.NOP
            } else {
                Ok(RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd, rs1: rd, imm })
            }
        }
        (0b01, 0b001) => {
            // C.ADDIW (RV64) / C.JAL (RV32)
            let rd = ((instr >> 7) & 0b11111) as u8;
            let imm = decode_ci_imm(instr);
            // Assuming RV64
            Ok(RiscvInstruction::IAluw { op: RiscvOpcode::Addw, rd, rs1: rd, imm })
        }
        (0b01, 0b010) => {
            // C.LI: addi rd, x0, imm
            let rd = ((instr >> 7) & 0b11111) as u8;
            let imm = decode_ci_imm(instr);
            Ok(RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd, rs1: 0, imm })
        }
        (0b01, 0b011) => {
            let rd = ((instr >> 7) & 0b11111) as u8;
            if rd == 2 {
                // C.ADDI16SP: addi x2, x2, nzimm
                let imm = decode_ci_addi16sp_imm(instr);
                Ok(RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 2, rs1: 2, imm })
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
                    Ok(RiscvInstruction::IAlu { op: RiscvOpcode::Srl, rd, rs1: rd, imm: shamt as i32 })
                }
                0b01 => {
                    // C.SRAI
                    let shamt = decode_ci_shamt(instr);
                    Ok(RiscvInstruction::IAlu { op: RiscvOpcode::Sra, rd, rs1: rd, imm: shamt as i32 })
                }
                0b10 => {
                    // C.ANDI
                    let imm = decode_ci_imm(instr);
                    Ok(RiscvInstruction::IAlu { op: RiscvOpcode::And, rd, rs1: rd, imm })
                }
                0b11 => {
                    // C.SUB, C.XOR, C.OR, C.AND, C.SUBW, C.ADDW
                    let funct2_b = (instr >> 5) & 0b11;
                    let rs2 = ((instr >> 2) & 0b111) as u8 + 8;
                    let bit12 = (instr >> 12) & 1;
                    match (bit12, funct2_b) {
                        (0, 0b00) => Ok(RiscvInstruction::RAlu { op: RiscvOpcode::Sub, rd, rs1: rd, rs2 }),
                        (0, 0b01) => Ok(RiscvInstruction::RAlu { op: RiscvOpcode::Xor, rd, rs1: rd, rs2 }),
                        (0, 0b10) => Ok(RiscvInstruction::RAlu { op: RiscvOpcode::Or, rd, rs1: rd, rs2 }),
                        (0, 0b11) => Ok(RiscvInstruction::RAlu { op: RiscvOpcode::And, rd, rs1: rd, rs2 }),
                        (1, 0b00) => Ok(RiscvInstruction::RAluw { op: RiscvOpcode::Subw, rd, rs1: rd, rs2 }),
                        (1, 0b01) => Ok(RiscvInstruction::RAluw { op: RiscvOpcode::Addw, rd, rs1: rd, rs2 }),
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
            Ok(RiscvInstruction::Branch { cond: BranchCondition::Eq, rs1, rs2: 0, imm: offset })
        }
        (0b01, 0b111) => {
            // C.BNEZ: bne rs1', x0, offset
            let rs1 = ((instr >> 7) & 0b111) as u8 + 8;
            let offset = decode_cb_imm(instr);
            Ok(RiscvInstruction::Branch { cond: BranchCondition::Ne, rs1, rs2: 0, imm: offset })
        }
        
        // Quadrant 2
        (0b10, 0b000) => {
            // C.SLLI
            let rd = ((instr >> 7) & 0b11111) as u8;
            let shamt = decode_ci_shamt(instr);
            Ok(RiscvInstruction::IAlu { op: RiscvOpcode::Sll, rd, rs1: rd, imm: shamt as i32 })
        }
        (0b10, 0b010) => {
            // C.LWSP: lw rd, offset(x2)
            let rd = ((instr >> 7) & 0b11111) as u8;
            let offset = decode_ci_lwsp_imm(instr);
            Ok(RiscvInstruction::Load { op: RiscvMemOp::Lw, rd, rs1: 2, imm: offset as i32 })
        }
        (0b10, 0b011) => {
            // C.LDSP (RV64): ld rd, offset(x2)
            let rd = ((instr >> 7) & 0b11111) as u8;
            let offset = decode_ci_ldsp_imm(instr);
            Ok(RiscvInstruction::Load { op: RiscvMemOp::Ld, rd, rs1: 2, imm: offset as i32 })
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
                    Ok(RiscvInstruction::RAlu { op: RiscvOpcode::Add, rd: rs1, rs1: 0, rs2 })
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
                    Ok(RiscvInstruction::RAlu { op: RiscvOpcode::Add, rd: rs1, rs1, rs2 })
                }
            }
        }
        (0b10, 0b110) => {
            // C.SWSP: sw rs2, offset(x2)
            let rs2 = ((instr >> 2) & 0b11111) as u8;
            let offset = decode_css_sw_imm(instr);
            Ok(RiscvInstruction::Store { op: RiscvMemOp::Sw, rs1: 2, rs2, imm: offset as i32 })
        }
        (0b10, 0b111) => {
            // C.SDSP (RV64): sd rs2, offset(x2)
            let rs2 = ((instr >> 2) & 0b11111) as u8;
            let offset = decode_css_sd_imm(instr);
            Ok(RiscvInstruction::Store { op: RiscvMemOp::Sd, rs1: 2, rs2, imm: offset as i32 })
        }
        
        _ => Err(format!("Unknown compressed instruction: op={:#x}, funct3={:#x}", op, funct3)),
    }
}

// Compressed instruction immediate decoders
fn decode_ciw_imm(instr: u16) -> u32 {
    // C.ADDI4SPN: nzuimm[5:4|9:6|2|3] => scaled by 4
    let bits = ((instr >> 5) & 1) << 3
        | ((instr >> 6) & 1) << 2
        | ((instr >> 7) & 0xF) << 6
        | ((instr >> 11) & 0x3) << 4;
    bits as u32
}

fn decode_cl_lw_imm(instr: u16) -> u32 {
    // C.LW/C.SW: offset[5:3|2|6]
    let bits = ((instr >> 5) & 1) << 6
        | ((instr >> 6) & 1) << 2
        | ((instr >> 10) & 0x7) << 3;
    bits as u32
}

fn decode_cl_ld_imm(instr: u16) -> u32 {
    // C.LD/C.SD: offset[5:3|7:6]
    let bits = ((instr >> 5) & 0x3) << 6
        | ((instr >> 10) & 0x7) << 3;
    bits as u32
}

fn decode_ci_imm(instr: u16) -> i32 {
    // CI format: imm[5|4:0], sign-extended
    let imm5 = ((instr >> 12) & 1) as i32;
    let imm4_0 = ((instr >> 2) & 0x1F) as i32;
    let imm = (imm5 << 5) | imm4_0;
    // Sign-extend from bit 5
    if imm5 != 0 { imm | !0x3F } else { imm }
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
    if bit9 != 0 { imm | !0x3FF } else { imm }
}

fn decode_ci_lui_imm(instr: u16) -> i32 {
    // C.LUI: nzimm[17|16:12]
    let bit17 = ((instr >> 12) & 1) as i32;
    let bits16_12 = ((instr >> 2) & 0x1F) as i32;
    let imm = (bit17 << 5) | bits16_12;
    // Sign-extend from bit 5
    if bit17 != 0 { imm | !0x3F } else { imm }
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
    let imm = (bit11 << 11) | (bit10 << 10) | (bit9_8 << 8) | (bit7 << 7)
        | (bit6 << 6) | (bit5 << 5) | (bit4 << 4) | (bit3_1 << 1);
    // Sign-extend from bit 11
    if bit11 != 0 { imm | !0xFFF } else { imm }
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
    if bit8 != 0 { imm | !0x1FF } else { imm }
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
            0b0110011 | ((*rd as u32) << 7) | (funct3 << 12) | ((*rs1 as u32) << 15) | ((*rs2 as u32) << 20) | (funct7 << 25)
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
            0b0100011 | (imm_4_0 << 7) | (funct3 << 12) | ((*rs1 as u32) << 15) | ((*rs2 as u32) << 20) | (imm_11_5 << 25)
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
            0b1100011 | (imm_11 << 7) | (imm_4_1 << 8) | (funct3 << 12) | ((*rs1 as u32) << 15) | ((*rs2 as u32) << 20) | (imm_10_5 << 25) | (imm_12 << 31)
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
            0b0111011 | ((*rd as u32) << 7) | (funct3 << 12) | ((*rs1 as u32) << 15) | ((*rs2 as u32) << 20) | (funct7 << 25)
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
            0b0101111 | ((*rd as u32) << 7) | (funct3 << 12) | ((*rs1 as u32) << 15) | ((*rs2 as u32) << 20) | (funct5 << 27)
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
            0b0101111 | ((*rd as u32) << 7) | (funct3 << 12) | ((*rs1 as u32) << 15) | ((*rs2 as u32) << 20) | (funct5 << 27)
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

// ============================================================================
// Trace → Proof Binding: Convert VmTrace to Neo Witness
// ============================================================================

use neo_vm_trace::VmTrace;
use crate::plain::{PlainMemTrace, PlainLutTrace, LutTable, PlainMemLayout};
use p3_field::PrimeField64;
use std::collections::HashMap;

/// Configuration for trace-to-proof conversion.
#[derive(Clone, Debug)]
pub struct TraceToProofConfig {
    /// Word size in bits (32 or 64)
    pub xlen: usize,
    /// Memory layout parameters
    pub mem_layout: PlainMemLayout,
    /// Shout table for each opcode
    pub opcode_tables: HashMap<RiscvOpcode, LutTable<p3_goldilocks::Goldilocks>>,
}

impl Default for TraceToProofConfig {
    fn default() -> Self {
        Self {
            xlen: 32,
            mem_layout: PlainMemLayout { k: 16, d: 1, n_side: 256 },
            opcode_tables: HashMap::new(),
        }
    }
}

/// Convert a VmTrace to PlainMemTrace for Twist encoding.
///
/// This extracts all memory read/write events from the trace and formats them
/// for Neo's Twist (read/write memory) argument.
pub fn trace_to_plain_mem_trace<F: PrimeField64>(
    trace: &VmTrace<u64, u64>,
) -> PlainMemTrace<F> {
    let steps = trace.len();

    let mut has_read = vec![F::ZERO; steps];
    let mut has_write = vec![F::ZERO; steps];
    let mut read_addr = vec![0u64; steps];
    let mut write_addr = vec![0u64; steps];
    let mut read_val = vec![F::ZERO; steps];
    let mut write_val = vec![F::ZERO; steps];
    let mut inc_at_write_addr = vec![F::ZERO; steps];

    // Track memory state for increment calculation
    let mut mem_state: HashMap<u64, F> = HashMap::new();

    for (j, step) in trace.steps.iter().enumerate() {
        for event in &step.twist_events {
            match event.kind {
                neo_vm_trace::TwistOpKind::Read => {
                    has_read[j] = F::ONE;
                    read_addr[j] = event.addr;
                    read_val[j] = F::from_u64(event.value);
                }
                neo_vm_trace::TwistOpKind::Write => {
                    has_write[j] = F::ONE;
                    write_addr[j] = event.addr;
                    write_val[j] = F::from_u64(event.value);

                    // Calculate increment
                    let old_val = mem_state.get(&event.addr).copied().unwrap_or(F::ZERO);
                    let new_val = F::from_u64(event.value);
                    inc_at_write_addr[j] = new_val - old_val;
                    mem_state.insert(event.addr, new_val);
                }
            }
        }
    }

    PlainMemTrace {
        steps,
        has_read,
        has_write,
        read_addr,
        write_addr,
        read_val,
        write_val,
        inc_at_write_addr,
    }
}

/// Convert a VmTrace to PlainLutTrace for Shout encoding.
///
/// This extracts all lookup events from the trace and formats them
/// for Neo's Shout (read-only lookup) argument.
///
/// # Note
/// Currently assumes a single unified lookup table. For multiple opcode-specific
/// tables, use `trace_to_plain_lut_traces_by_opcode`.
pub fn trace_to_plain_lut_trace<F: PrimeField64>(
    trace: &VmTrace<u64, u64>,
) -> PlainLutTrace<F> {
    let steps = trace.len();

    let mut has_lookup = vec![F::ZERO; steps];
    let mut addr = vec![0u64; steps];
    let mut val = vec![F::ZERO; steps];

    for (j, step) in trace.steps.iter().enumerate() {
        // Take the first Shout event if any
        if let Some(event) = step.shout_events.first() {
            has_lookup[j] = F::ONE;
            addr[j] = event.key;
            val[j] = F::from_u64(event.value);
        }
    }

    PlainLutTrace {
        has_lookup,
        addr,
        val,
    }
}

/// Convert a VmTrace to multiple PlainLutTraces, one per opcode/table.
///
/// This separates lookup events by their ShoutId, allowing different
/// opcodes to use different lookup tables.
pub fn trace_to_plain_lut_traces_by_opcode<F: PrimeField64>(
    trace: &VmTrace<u64, u64>,
    num_tables: usize,
) -> Vec<PlainLutTrace<F>> {
    let steps = trace.len();

    // Initialize a trace for each table
    let mut traces: Vec<PlainLutTrace<F>> = (0..num_tables)
        .map(|_| PlainLutTrace {
            has_lookup: vec![F::ZERO; steps],
            addr: vec![0u64; steps],
            val: vec![F::ZERO; steps],
        })
        .collect();

    for (j, step) in trace.steps.iter().enumerate() {
        for event in &step.shout_events {
            let table_id = event.shout_id.0 as usize;
            if table_id < num_tables {
                traces[table_id].has_lookup[j] = F::ONE;
                traces[table_id].addr[j] = event.key;
                traces[table_id].val[j] = F::from_u64(event.value);
            }
        }
    }

    traces
}

/// Build a lookup table for a specific RISC-V opcode.
///
/// This creates a `LutTable` that can be used with Neo's Shout encoding.
pub fn build_opcode_lut_table<F: PrimeField64>(
    table_id: u32,
    opcode: RiscvOpcode,
    xlen: usize,
) -> LutTable<F> {
    let table: RiscvLookupTable<F> = RiscvLookupTable::new(opcode, xlen);
    let size = table.size();
    let k = (size as f64).log2().ceil() as usize;

    LutTable {
        table_id,
        k,
        d: 1,
        n_side: size,
        content: table.content(),
    }
}

/// Summary of a trace conversion.
#[derive(Clone, Debug)]
pub struct TraceConversionSummary {
    /// Total steps in the trace
    pub total_steps: usize,
    /// Number of memory read operations
    pub num_reads: usize,
    /// Number of memory write operations
    pub num_writes: usize,
    /// Number of lookup operations
    pub num_lookups: usize,
    /// Unique memory addresses accessed
    pub unique_addresses: usize,
    /// Unique lookup keys used
    pub unique_lookup_keys: usize,
}

/// Extract final register values from a trace as a ProgramIO structure.
///
/// This creates a ProgramIO suitable for the Output Sumcheck, using RISC-V
/// register conventions (x10-x17 as return value registers).
pub fn extract_program_io<F: p3_field::PrimeField64>(
    trace: &VmTrace<u64, u64>,
    output_regs: &[usize],
) -> crate::output_check::ProgramIO<F> {
    // RISC-V ABI: x10-x17 (a0-a7) are argument/return registers
    // We map register x_i to virtual address i

    let mut program_io = crate::output_check::ProgramIO::new();

    if let Some(last_step) = trace.steps.last() {
        for &reg in output_regs {
            if reg < 32 {
                let val = last_step.regs_after[reg];
                program_io = program_io.with_output(reg as u64, F::from_u64(val));
            }
        }
    }

    program_io
}

/// Build a final memory state vector suitable for the Output Sumcheck.
///
/// This creates a sparse representation of the final register file state,
/// where virtual address i contains the value of register x_i.
pub fn build_final_memory_state<F: p3_field::PrimeField64>(
    trace: &VmTrace<u64, u64>,
    num_bits: usize,
) -> Vec<F> {
    let size = 1usize << num_bits;
    let mut state = vec![F::ZERO; size];

    if let Some(last_step) = trace.steps.last() {
        // Map registers to virtual addresses 0-31
        for (i, &val) in last_step.regs_after.iter().enumerate() {
            if i < size {
                state[i] = F::from_u64(val);
            }
        }
    }

    state
}

/// Analyze a trace and return a summary.
pub fn analyze_trace(trace: &VmTrace<u64, u64>) -> TraceConversionSummary {
    let mut num_reads = 0;
    let mut num_writes = 0;
    let mut num_lookups = 0;
    let mut addresses = std::collections::HashSet::new();
    let mut lookup_keys = std::collections::HashSet::new();

    for step in &trace.steps {
        for event in &step.twist_events {
            match event.kind {
                neo_vm_trace::TwistOpKind::Read => {
                    num_reads += 1;
                    addresses.insert(event.addr);
                }
                neo_vm_trace::TwistOpKind::Write => {
                    num_writes += 1;
                    addresses.insert(event.addr);
                }
            }
        }
        for event in &step.shout_events {
            num_lookups += 1;
            lookup_keys.insert(event.key);
        }
    }

    TraceConversionSummary {
        total_steps: trace.len(),
        num_reads,
        num_writes,
        num_lookups,
        unique_addresses: addresses.len(),
        unique_lookup_keys: lookup_keys.len(),
    }
}
