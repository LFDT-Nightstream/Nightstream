//! RISC-V instruction module layout (Jolt-style organization).
//!
//! Step 1 migration goal: provide a per-instruction file structure without
//! changing runtime behavior. Existing execution/proving code still lives under
//! `riscv::lookups` and is reused here through thin wrappers.

use crate::riscv::lookups::{interleave_bits, uninterleave_bits, RiscvOpcode};

pub mod add;
pub mod and;
pub mod andn;
pub mod eq;
pub mod or;
pub mod sll;
pub mod slt;
pub mod sltu;
pub mod sra;
pub mod srl;
pub mod sub;
pub mod xor;

pub mod mul;
pub mod mulhu;
pub mod mulw;

pub mod div;
pub mod divu;
pub mod divuw;
pub mod divw;
pub mod mulh;
pub mod mulhsu;
pub mod rem;
pub mod remu;
pub mod remuw;
pub mod remw;

pub mod virtual_advice;
pub mod virtual_assert_eq;
pub mod virtual_assert_eq_signs;
pub mod virtual_assert_lt_abs;
pub mod virtual_assert_lte;
pub mod virtual_assert_ltu;
pub mod virtual_move;
pub mod virtual_movsign;

pub mod auipc;
pub mod branch;
pub mod jal;
pub mod jalr;
pub mod lui;

pub mod load;
pub mod store;

pub mod tables;

pub use crate::riscv::lookups::{BranchCondition, RiscvInstruction, RiscvMemOp};

/// Rollout switch for operand-mode lookup keying.
///
/// `true` enables combined operand keying for selected opcodes (ADD/SUB).
/// `false` falls back to canonical interleaved `(lhs, rhs)` keys for all opcodes.
pub const ENABLE_OPERAND_MODE_KEYS: bool = true;

#[inline]
pub fn operand_mode_keys_enabled() -> bool {
    ENABLE_OPERAND_MODE_KEYS
}

#[inline]
pub fn mask_to_xlen(value: u64, xlen: usize) -> u64 {
    if xlen >= 64 {
        value
    } else {
        let mask = (1u64 << xlen) - 1;
        value & mask
    }
}

#[inline]
fn sign_extend_32(x: u32) -> u64 {
    (x as i32) as i64 as u64
}

#[inline]
fn sign_mask_word(x: u64) -> u64 {
    if ((x >> 31) & 1) == 1 {
        0xFFFF_FFFF
    } else {
        0
    }
}

/// Sign-extend a value from `xlen` bits to `i64`.
pub fn sign_extend(x: u64, xlen: usize) -> i64 {
    match xlen {
        8 => (x as u8) as i8 as i64,
        16 => (x as u16) as i16 as i64,
        32 => (x as u32) as i32 as i64,
        64 => x as i64,
        _ => {
            let sign_bit = 1u64 << (xlen - 1);
            if (x & sign_bit) != 0 {
                (x | !((1u64 << xlen) - 1)) as i64
            } else {
                x as i64
            }
        }
    }
}

/// Canonical opcode semantics.
///
/// Step-8 cutover source of truth: lookup-layer helpers delegate here.
pub fn compute_op(op: RiscvOpcode, x: u64, y: u64, xlen: usize) -> u64 {
    let mask = mask_to_xlen(u64::MAX, xlen);
    let x = x & mask;
    let y = y & mask;

    let shift_mask = match xlen {
        32 => 0x1F,
        64 => 0x3F,
        _ => (xlen - 1) as u64,
    };

    let result = match op {
        RiscvOpcode::And => x & y,
        RiscvOpcode::Xor => x ^ y,
        RiscvOpcode::Or => x | y,
        RiscvOpcode::Sub => x.wrapping_sub(y),
        RiscvOpcode::Add => x.wrapping_add(y),

        RiscvOpcode::Mul => x.wrapping_mul(y),
        RiscvOpcode::Mulh => {
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
                    let product = x_signed * y_signed;
                    ((product >> xlen) as u64) & mask
                }
            }
        }
        RiscvOpcode::Mulhu => match xlen {
            32 => {
                let product = x.wrapping_mul(y);
                (product >> 32) & mask
            }
            64 => {
                let product = (x as u128) * (y as u128);
                (product >> 64) as u64
            }
            _ => {
                let product = (x as u128) * (y as u128);
                ((product >> xlen) as u64) & mask
            }
        },
        RiscvOpcode::Mulhsu => {
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

        RiscvOpcode::Div => {
            if y == 0 {
                mask
            } else {
                let x_signed = sign_extend(x, xlen);
                let y_signed = sign_extend(y, xlen);
                let most_negative = 1i64 << (xlen - 1);
                if x_signed == -most_negative && y_signed == -1 {
                    x
                } else {
                    (x_signed / y_signed) as u64
                }
            }
        }
        RiscvOpcode::Divu => {
            if y == 0 {
                mask
            } else {
                x / y
            }
        }
        RiscvOpcode::Rem => {
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
            if y == 0 {
                x
            } else {
                x % y
            }
        }

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
                u64::MAX
            } else if x32 == i32::MIN && y32 == -1 {
                sign_extend_32(x32 as u32)
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

        RiscvOpcode::VirtualMulWord => ((x as u32).wrapping_mul(y as u32)) as u64,
        RiscvOpcode::VirtualDivuWord => {
            let x32 = x as u32;
            let y32 = y as u32;
            if y32 == 0 {
                u32::MAX as u64
            } else {
                (x32 / y32) as u64
            }
        }
        RiscvOpcode::VirtualRemuWord => {
            let x32 = x as u32;
            let y32 = y as u32;
            if y32 == 0 {
                x32 as u64
            } else {
                (x32 % y32) as u64
            }
        }
        RiscvOpcode::VirtualDivWord => {
            let x32 = x as i32;
            let y32 = y as i32;
            if y32 == 0 {
                u32::MAX as u64
            } else if x32 == i32::MIN && y32 == -1 {
                (x32 as u32) as u64
            } else {
                ((x32 / y32) as u32) as u64
            }
        }
        RiscvOpcode::VirtualRemWord => {
            let x32 = x as i32;
            let y32 = y as i32;
            if y32 == 0 {
                (x32 as u32) as u64
            } else if x32 == i32::MIN && y32 == -1 {
                0
            } else {
                ((x32 % y32) as u32) as u64
            }
        }
        RiscvOpcode::VirtualMovsignWord => sign_mask_word(x),

        RiscvOpcode::Andn => x & !y,
    };

    result & mask
}

/// Architectural register count (`x0..x31`).
pub const ARCH_REG_COUNT: u64 = 32;
/// First virtual register address used by decomposition plans.
pub const VIRTUAL_REG_BASE: u64 = ARCH_REG_COUNT;

#[derive(Clone, Debug, Default)]
pub struct VirtualRegisterAllocator {
    next: u64,
}

impl VirtualRegisterAllocator {
    pub fn new() -> Self {
        Self { next: VIRTUAL_REG_BASE }
    }

    pub fn allocate(&mut self) -> u64 {
        let out = self.next;
        self.next = self
            .next
            .checked_add(1)
            .expect("virtual register allocator overflow");
        out
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecomposedOp {
    Advice {
        dst: u64,
    },
    AdviceRemainderAbs {
        dst: u64,
        dividend: u64,
        divisor: u64,
    },
    AdviceQuotient {
        dst: u64,
        op: RiscvOpcode,
        lhs: u64,
        rhs: u64,
    },
    MovSignWord {
        dst: u64,
        src: u64,
    },
    MovSign {
        dst: u64,
        src: u64,
    },
    ComposeU64FromLoHi32 {
        dst: u64,
        lo_src: u64,
        hi_src: u64,
    },
    Move {
        dst: u64,
        src: u64,
    },
    Add {
        dst: u64,
        lhs: u64,
        rhs: u64,
    },
    Sub {
        dst: u64,
        lhs: u64,
        rhs: u64,
    },
    Xor {
        dst: u64,
        lhs: u64,
        rhs: u64,
    },
    Mul {
        dst: u64,
        lhs: u64,
        rhs: u64,
    },
    Mulhu {
        dst: u64,
        lhs: u64,
        rhs: u64,
    },
    AssertEq {
        lhs: u64,
        rhs: u64,
    },
    AssertLtu {
        lhs: u64,
        rhs: u64,
    },
    AssertLte {
        lhs: u64,
        rhs: u64,
    },
    AssertLtAbs {
        lhs: u64,
        rhs: u64,
    },
    AssertEqSigns {
        lhs: u64,
        rhs: u64,
    },
    AssertValidDiv0 {
        divisor: u64,
        quotient: u64,
    },
    ChangeDivisor {
        dst: u64,
        dividend: u64,
        divisor: u64,
    },
    AssertMulUNoOverflow {
        lhs: u64,
        rhs: u64,
    },
    AssertValidUnsignedRemainder {
        remainder: u64,
        divisor: u64,
    },
}

/// Operand keying mode used by lookup-bound instructions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperandMode {
    Interleaved,
    AddOperands,
    SubtractOperands,
    MultiplyOperands,
    Advice,
}

/// Lightweight metadata interface for future decomposition plumbing.
pub trait InstructionDescriptor {
    fn opcode() -> Option<RiscvOpcode>;
    fn operand_mode() -> OperandMode {
        OperandMode::Interleaved
    }
}

pub fn rv64_word_helper_decomposition(
    rd: u8,
    rs1: u8,
    rs2: u8,
    helper_op: RiscvOpcode,
    alloc: &mut VirtualRegisterAllocator,
) -> Vec<DecomposedOp> {
    let rd = rd as u64;
    let rs1 = rs1 as u64;
    let rs2 = rs2 as u64;
    let v_out = alloc.allocate();
    let v_sign = alloc.allocate();

    vec![
        DecomposedOp::AdviceQuotient {
            dst: v_out,
            op: helper_op,
            lhs: rs1,
            rhs: rs2,
        },
        DecomposedOp::MovSignWord {
            dst: v_sign,
            src: v_out,
        },
        DecomposedOp::ComposeU64FromLoHi32 {
            dst: v_out,
            lo_src: v_out,
            hi_src: v_sign,
        },
        DecomposedOp::Move { dst: rd, src: v_out },
    ]
}

/// Operand mode for a concrete architectural opcode in the Step-1/2 scaffold.
pub fn opcode_operand_mode(op: RiscvOpcode) -> OperandMode {
    match op {
        RiscvOpcode::Add => OperandMode::AddOperands,
        RiscvOpcode::Sub => OperandMode::SubtractOperands,
        RiscvOpcode::Mul
        | RiscvOpcode::Mulh
        | RiscvOpcode::Mulhu
        | RiscvOpcode::Mulhsu
        | RiscvOpcode::Mulw
        | RiscvOpcode::Div
        | RiscvOpcode::Divu
        | RiscvOpcode::Rem
        | RiscvOpcode::Remu
        | RiscvOpcode::Divw
        | RiscvOpcode::Divuw
        | RiscvOpcode::Remw
        | RiscvOpcode::Remuw
        | RiscvOpcode::VirtualMulWord
        | RiscvOpcode::VirtualDivuWord
        | RiscvOpcode::VirtualRemuWord
        | RiscvOpcode::VirtualDivWord
        | RiscvOpcode::VirtualRemWord => OperandMode::MultiplyOperands,
        _ => OperandMode::Interleaved,
    }
}

/// Runtime decomposition plan for a decoded architectural instruction.
///
/// Returns the full decomposed sequence (including the final non-virtual commit step)
/// when decomposition is supported for this instruction.
pub fn decomposition_sequence_for_instruction(instr: &RiscvInstruction) -> Option<Vec<DecomposedOp>> {
    let mut alloc = VirtualRegisterAllocator::new();
    match instr {
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mul,
            rd,
            rs1,
            rs2,
        } => Some(crate::riscv::instruction::mul::decomposition_sequence(
            *rd, *rs1, *rs2, &mut alloc,
        )),
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulhu,
            rd,
            rs1,
            rs2,
        } => Some(crate::riscv::instruction::mulhu::decomposition_sequence(
            *rd, *rs1, *rs2, &mut alloc,
        )),
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulh,
            rd,
            rs1,
            rs2,
        } => Some(crate::riscv::instruction::mulh::decomposition_sequence(
            *rd, *rs1, *rs2, &mut alloc,
        )),
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulhsu,
            rd,
            rs1,
            rs2,
        } => Some(crate::riscv::instruction::mulhsu::decomposition_sequence(
            *rd, *rs1, *rs2, &mut alloc,
        )),
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Divu,
            rd,
            rs1,
            rs2,
        } => Some(crate::riscv::instruction::divu::decomposition_sequence(
            *rd, *rs1, *rs2, &mut alloc,
        )),
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Remu,
            rd,
            rs1,
            rs2,
        } => Some(crate::riscv::instruction::remu::decomposition_sequence(
            *rd, *rs1, *rs2, &mut alloc,
        )),
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Div,
            rd,
            rs1,
            rs2,
        } => Some(crate::riscv::instruction::div::decomposition_sequence(
            *rd, *rs1, *rs2, &mut alloc,
        )),
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Rem,
            rd,
            rs1,
            rs2,
        } => Some(crate::riscv::instruction::rem::decomposition_sequence(
            *rd, *rs1, *rs2, &mut alloc,
        )),
        RiscvInstruction::RAluw {
            op: RiscvOpcode::Mulw,
            rd,
            rs1,
            rs2,
        } => Some(crate::riscv::instruction::mulw::decomposition_sequence(
            *rd, *rs1, *rs2, &mut alloc,
        )),
        RiscvInstruction::RAluw {
            op: RiscvOpcode::Divw,
            rd,
            rs1,
            rs2,
        } => Some(crate::riscv::instruction::divw::decomposition_sequence(
            *rd, *rs1, *rs2, &mut alloc,
        )),
        RiscvInstruction::RAluw {
            op: RiscvOpcode::Divuw,
            rd,
            rs1,
            rs2,
        } => Some(crate::riscv::instruction::divuw::decomposition_sequence(
            *rd, *rs1, *rs2, &mut alloc,
        )),
        RiscvInstruction::RAluw {
            op: RiscvOpcode::Remw,
            rd,
            rs1,
            rs2,
        } => Some(crate::riscv::instruction::remw::decomposition_sequence(
            *rd, *rs1, *rs2, &mut alloc,
        )),
        RiscvInstruction::RAluw {
            op: RiscvOpcode::Remuw,
            rd,
            rs1,
            rs2,
        } => Some(crate::riscv::instruction::remuw::decomposition_sequence(
            *rd, *rs1, *rs2, &mut alloc,
        )),
        _ => None,
    }
}

/// Encode a Shout lookup key for a given opcode and operands.
///
/// Uses the current rollout toggle (`ENABLE_OPERAND_MODE_KEYS`).
/// When the toggle is off, this returns the canonical interleaved `(lhs, rhs)` key
/// for all opcodes.
#[inline]
pub fn encode_lookup_key(op: RiscvOpcode, lhs: u64, rhs: u64, xlen: usize) -> u128 {
    encode_lookup_key_with_mode(op, lhs, rhs, xlen, ENABLE_OPERAND_MODE_KEYS)
}

#[inline]
pub fn opcode_lookup_xlen(op: RiscvOpcode, xlen: usize) -> usize {
    match op {
        RiscvOpcode::Addw
        | RiscvOpcode::Subw
        | RiscvOpcode::Sllw
        | RiscvOpcode::Srlw
        | RiscvOpcode::Sraw
        | RiscvOpcode::Mulw
        | RiscvOpcode::Divw
        | RiscvOpcode::Divuw
        | RiscvOpcode::Remw
        | RiscvOpcode::Remuw
        | RiscvOpcode::VirtualMulWord
        | RiscvOpcode::VirtualDivuWord
        | RiscvOpcode::VirtualRemuWord
        | RiscvOpcode::VirtualDivWord
        | RiscvOpcode::VirtualRemWord
        | RiscvOpcode::VirtualMovsignWord => 32,
        _ => xlen,
    }
}

#[inline]
fn normalized_lookup_operands(op: RiscvOpcode, lhs: u64, rhs: u64, xlen: usize) -> (u64, u64, usize) {
    let key_xlen = opcode_lookup_xlen(op, xlen);
    (mask_to_xlen(lhs, key_xlen), mask_to_xlen(rhs, key_xlen), key_xlen)
}

/// Encode a Shout lookup key with an explicit rollout toggle.
#[inline]
pub fn encode_lookup_key_with_mode(
    op: RiscvOpcode,
    lhs: u64,
    rhs: u64,
    xlen: usize,
    use_operand_mode_keys: bool,
) -> u128 {
    let (lhs, rhs, key_xlen) = normalized_lookup_operands(op, lhs, rhs, xlen);

    if !use_operand_mode_keys {
        return interleave_bits(lhs, rhs);
    }

    match opcode_operand_mode(op) {
        OperandMode::Interleaved => interleave_bits(lhs, rhs),
        OperandMode::AddOperands => mask_to_xlen(lhs.wrapping_add(rhs), key_xlen) as u128,
        OperandMode::SubtractOperands => mask_to_xlen(lhs.wrapping_sub(rhs), key_xlen) as u128,
        OperandMode::MultiplyOperands => interleave_bits(lhs, rhs),
        OperandMode::Advice => interleave_bits(lhs, rhs),
    }
}

#[inline]
pub fn decode_interleaved_lookup_key(key: u128) -> (u64, u64) {
    uninterleave_bits(key)
}

/// Decode `(lhs, rhs)` operands from a key when the opcode still uses interleaved keying.
#[inline]
pub fn try_decode_lookup_operands(
    op: RiscvOpcode,
    key: u128,
    use_operand_mode_keys: bool,
    xlen: usize,
) -> Option<(u64, u64)> {
    let decode_interleaved_for_opcode = || {
        let (lhs, rhs) = decode_interleaved_lookup_key(key);
        let key_xlen = opcode_lookup_xlen(op, xlen);
        (mask_to_xlen(lhs, key_xlen), mask_to_xlen(rhs, key_xlen))
    };

    if !use_operand_mode_keys {
        return Some(decode_interleaved_for_opcode());
    }

    match opcode_operand_mode(op) {
        OperandMode::Interleaved => Some(decode_interleaved_for_opcode()),
        OperandMode::MultiplyOperands => Some(decode_interleaved_for_opcode()),
        OperandMode::AddOperands | OperandMode::SubtractOperands | OperandMode::Advice => None,
    }
}

#[inline]
pub fn opcode_uses_combined_lookup_key(op: RiscvOpcode) -> bool {
    if !operand_mode_keys_enabled() {
        return false;
    }
    matches!(op, RiscvOpcode::Add | RiscvOpcode::Sub)
}
