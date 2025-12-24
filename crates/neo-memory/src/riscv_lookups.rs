//! RISC-V instruction lookup tables for Neo's Shout protocol.
//!
//! This module provides lookup tables for RISC-V ALU operations (AND, XOR, OR, etc.)
//! that can be proven using Neo's Shout (read-only memory) argument.
//!
//! ## Design
//!
//! Jolt represents each instruction's semantics via a lookup table where:
//! - The **index** encodes the operands (typically interleaved bits of x and y)
//! - The **value** is the operation result
//!
//! For Neo's Shout protocol, we need:
//! - A table of values `Val[k]` for k in [0, K)
//! - The MLE `Val~(r)` that can be evaluated at random points
//!
//! ## Supported Operations
//!
//! ### Bitwise Operations (Shout - uses bit interleaving)
//! - `AND`: Bitwise AND
//! - `XOR`: Bitwise XOR
//! - `OR`: Bitwise OR
//!
//! ### Comparison Operations (Shout - uses bit interleaving)
//! - `SUB`: Subtraction (with wraparound)
//! - `SLT`: Set Less Than (signed)
//! - `SLTU`: Set Less Than (unsigned)
//! - `EQ`: Equality check
//! - `NEQ`: Inequality check
//!
//! ### Arithmetic Operations (Shout - uses RangeCheck approach)
//! - `ADD/ADDI`: Addition with carry detection via range check
//!
//! ### Shift Operations (Shout - specialized virtual tables)
//! - `SLL`: Shift Left Logical
//! - `SRL`: Shift Right Logical
//! - `SRA`: Shift Right Arithmetic
//!
//! ### Memory Operations (Twist - read/write memory)
//! - `LW/LH/LB`: Load Word/Half/Byte
//! - `SW/SH/SB`: Store Word/Half/Byte
//!
//! ## Example
//!
//! ```ignore
//! use neo_memory::riscv_lookups::{RiscvOpcode, create_lookup_table};
//!
//! // Create an 8-bit XOR lookup table
//! let xor_table = create_lookup_table(RiscvOpcode::Xor, 8);
//!
//! // Verify a specific lookup: 0xAB ^ 0xCD = 0x66
//! let index = interleave_bits(0xAB, 0xCD);
//! assert_eq!(xor_table[index], 0x66);
//! ```

use neo_vm_trace::{Shout, ShoutId, Twist, TwistId};
use p3_field::Field;
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
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
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
        )
    }

    /// Returns true if this is a store operation.
    pub fn is_store(&self) -> bool {
        matches!(
            self,
            RiscvMemOp::Sw | RiscvMemOp::Sh | RiscvMemOp::Sb | RiscvMemOp::Sd
        )
    }

    /// Returns the access width in bytes.
    pub fn width_bytes(&self) -> usize {
        match self {
            RiscvMemOp::Lb | RiscvMemOp::Lbu | RiscvMemOp::Sb => 1,
            RiscvMemOp::Lh | RiscvMemOp::Lhu | RiscvMemOp::Sh => 2,
            RiscvMemOp::Lw | RiscvMemOp::Lwu | RiscvMemOp::Sw => 4,
            RiscvMemOp::Ld | RiscvMemOp::Sd => 8,
        }
    }

    /// Returns true if this load should sign-extend.
    pub fn is_sign_extend(&self) -> bool {
        matches!(self, RiscvMemOp::Lh | RiscvMemOp::Lb | RiscvMemOp::Lw)
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
    };

    result & mask
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
        // For shift and other opcodes, use the naive MLE evaluation
        // Note: Jolt's virtual table approach (evaluate_srl_mle, evaluate_sra_mle)
        // uses a different encoding that doesn't match our standard tables.
        _ => evaluate_mle_naive(op, r, xlen),
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
    fn id_to_opcode(&self, id: ShoutId) -> Option<RiscvOpcode> {
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
                0 => Ok(RiscvInstruction::Halt), // ECALL -> Halt for now
                1 => Ok(RiscvInstruction::Halt), // EBREAK -> Halt for now
                _ => Err(format!("Unknown SYSTEM instruction: imm={:#x}", imm)),
            }
        }

        // MISC-MEM (0001111) - FENCE
        0b0001111 => {
            // FENCE - treat as NOP for now
            Ok(RiscvInstruction::Nop)
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
/// # Arguments
/// * `bytes` - Program bytes (must be 4-byte aligned)
///
/// # Returns
/// * `Vec<RiscvInstruction>` - List of decoded instructions
///
/// # Errors
/// Returns an error if any instruction cannot be decoded.
pub fn decode_program(bytes: &[u8]) -> Result<Vec<RiscvInstruction>, String> {
    if bytes.len() % 4 != 0 {
        return Err("Program bytes must be 4-byte aligned".to_string());
    }

    let mut instructions = Vec::new();
    for chunk in bytes.chunks(4) {
        let instr = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        instructions.push(decode_instruction(instr)?);
    }
    Ok(instructions)
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
