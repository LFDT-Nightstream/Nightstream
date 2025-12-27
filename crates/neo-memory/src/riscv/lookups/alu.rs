use super::bits::uninterleave_bits;
use super::isa::RiscvOpcode;

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
pub(super) fn sign_extend(x: u64, xlen: usize) -> i64 {
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
