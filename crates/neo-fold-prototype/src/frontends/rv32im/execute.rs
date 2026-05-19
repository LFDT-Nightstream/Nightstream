//! Owns concrete RV32IM slice execution for the current parity corpus opcodes.

use super::isa::{decode_instruction, Rv32BuildError, Rv32DecodedInstruction, Rv32Opcode, Rv32Program, Rv32State};
use super::tables::{opcode_family, Rv32FamilyTag};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutedStep {
    pub step_index: usize,
    pub word: u32,
    pub decoded: Rv32DecodedInstruction,
    pub family: Rv32FamilyTag,
    pub prev: Rv32State,
    pub next: Rv32State,
    pub rs1_value: u32,
    pub rs2_value: u32,
    pub rd_before: u32,
    pub alu_result: u32,
    pub effective_addr: Option<u32>,
    pub memory_before: Option<u32>,
    pub memory_after: Option<u32>,
    pub terminated: bool,
}

fn wrapping_add_signed(base: u32, offset: i32) -> u32 {
    base.wrapping_add(offset as u32)
}

fn signed_lt(lhs: u32, rhs: u32) -> bool {
    (lhs as i32) < (rhs as i32)
}

fn signed_ge(lhs: u32, rhs: u32) -> bool {
    (lhs as i32) >= (rhs as i32)
}

fn shift_imm(decoded: Rv32DecodedInstruction) -> u32 {
    (decoded.imm as u32) & 0x1f
}

fn shift_reg(rs2_value: u32) -> u32 {
    rs2_value & 0x1f
}

fn narrow_access_spec(opcode: Rv32Opcode) -> Option<(u32, bool, bool)> {
    match opcode {
        Rv32Opcode::Lb => Some((1, true, false)),
        Rv32Opcode::Lbu => Some((1, false, false)),
        Rv32Opcode::Lh => Some((2, true, false)),
        Rv32Opcode::Lhu => Some((2, false, false)),
        Rv32Opcode::Lw => Some((4, true, false)),
        Rv32Opcode::Sb => Some((1, false, true)),
        Rv32Opcode::Sh => Some((2, false, true)),
        Rv32Opcode::Sw => Some((4, false, true)),
        _ => None,
    }
}

fn narrow_backing_addr(addr: u32, size_bytes: u32, opcode: Rv32Opcode) -> Result<(u32, u32), Rv32BuildError> {
    if addr % size_bytes != 0 {
        return Err(Rv32BuildError::Memory(format!(
            "{opcode:?} effective address 0x{addr:08x} is not naturally aligned for {size_bytes} bytes"
        )));
    }
    let byte_offset = addr & 0x3;
    if byte_offset + size_bytes > 4 {
        return Err(Rv32BuildError::Memory(format!(
            "{opcode:?} effective address 0x{addr:08x} crosses a 4-byte backing word"
        )));
    }
    Ok((addr & !0x3, byte_offset))
}

fn sign_extend_bits(raw: u32, bits: u32) -> u32 {
    (((raw << (32 - bits)) as i32) >> (32 - bits)) as u32
}

pub(crate) fn mul_low(lhs: u32, rhs: u32) -> u32 {
    lhs.wrapping_mul(rhs)
}

pub(crate) fn mul_high_signed(lhs: u32, rhs: u32) -> u32 {
    (((lhs as i32 as i64) * (rhs as i32 as i64)) >> 32) as u32
}

pub(crate) fn mul_high_signed_unsigned(lhs: u32, rhs: u32) -> u32 {
    (((lhs as i32 as i64) * (rhs as u64 as i64)) >> 32) as u32
}

pub(crate) fn mul_high_unsigned(lhs: u32, rhs: u32) -> u32 {
    (((lhs as u64) * (rhs as u64)) >> 32) as u32
}

fn div_signed_result(lhs: u32, rhs: u32) -> u32 {
    let lhs_signed = lhs as i32;
    let rhs_signed = rhs as i32;
    if rhs_signed == 0 {
        u32::MAX
    } else if lhs_signed == i32::MIN && rhs_signed == -1 {
        lhs
    } else {
        (lhs_signed / rhs_signed) as u32
    }
}

fn div_unsigned_result(lhs: u32, rhs: u32) -> u32 {
    if rhs == 0 {
        u32::MAX
    } else {
        lhs / rhs
    }
}

fn rem_signed_result(lhs: u32, rhs: u32) -> u32 {
    let lhs_signed = lhs as i32;
    let rhs_signed = rhs as i32;
    if rhs_signed == 0 {
        lhs
    } else if lhs_signed == i32::MIN && rhs_signed == -1 {
        0
    } else {
        (lhs_signed % rhs_signed) as u32
    }
}

fn rem_unsigned_result(lhs: u32, rhs: u32) -> u32 {
    if rhs == 0 {
        lhs
    } else {
        lhs % rhs
    }
}

fn extract_narrow(word: u32, byte_offset: u32, size_bytes: u32, signed: bool) -> u32 {
    let bits = size_bytes * 8;
    let mask = if bits == 32 { u32::MAX } else { (1u32 << bits) - 1 };
    let raw = (word >> (byte_offset * 8)) & mask;
    if signed {
        sign_extend_bits(raw, bits)
    } else {
        raw
    }
}

fn blend_narrow(word: u32, byte_offset: u32, size_bytes: u32, value: u32) -> u32 {
    let bits = size_bytes * 8;
    let field_mask = if bits == 32 { u32::MAX } else { (1u32 << bits) - 1 };
    let shifted_mask = field_mask << (byte_offset * 8);
    let shifted_value = (value & field_mask) << (byte_offset * 8);
    (word & !shifted_mask) | shifted_value
}

pub fn execute_step(
    program: &Rv32Program,
    prev: &Rv32State,
    step_index: usize,
) -> Result<ExecutedStep, Rv32BuildError> {
    if prev.halted {
        return Err(Rv32BuildError::Program(
            "cannot execute another step from a halted RV32 state".into(),
        ));
    }

    let word = program.fetch_word(prev.pc)?;
    let decoded = decode_instruction(word)?;
    let family = opcode_family(decoded.opcode);
    let rs1_value = prev.read_reg(decoded.rs1);
    let rs2_value = prev.read_reg(decoded.rs2);
    let rd_before = prev.read_reg(decoded.rd);
    let mut next = prev.clone();
    next.pc = prev.pc.wrapping_add(4);

    let (alu_result, effective_addr, memory_before, memory_after, terminated) = match decoded.opcode {
        Rv32Opcode::Addi => {
            let result = rs1_value.wrapping_add(decoded.imm as u32);
            next.write_reg(decoded.rd, result);
            (result, None, None, None, false)
        }
        Rv32Opcode::Add => {
            let result = rs1_value.wrapping_add(rs2_value);
            next.write_reg(decoded.rd, result);
            (result, None, None, None, false)
        }
        Rv32Opcode::Sub => {
            let result = rs1_value.wrapping_sub(rs2_value);
            next.write_reg(decoded.rd, result);
            (result, None, None, None, false)
        }
        Rv32Opcode::Andi => {
            let result = rs1_value & decoded.imm as u32;
            next.write_reg(decoded.rd, result);
            (result, None, None, None, false)
        }
        Rv32Opcode::And => {
            let result = rs1_value & rs2_value;
            next.write_reg(decoded.rd, result);
            (result, None, None, None, false)
        }
        Rv32Opcode::Ori => {
            let result = rs1_value | decoded.imm as u32;
            next.write_reg(decoded.rd, result);
            (result, None, None, None, false)
        }
        Rv32Opcode::Or => {
            let result = rs1_value | rs2_value;
            next.write_reg(decoded.rd, result);
            (result, None, None, None, false)
        }
        Rv32Opcode::Xori => {
            let result = rs1_value ^ decoded.imm as u32;
            next.write_reg(decoded.rd, result);
            (result, None, None, None, false)
        }
        Rv32Opcode::Xor => {
            let result = rs1_value ^ rs2_value;
            next.write_reg(decoded.rd, result);
            (result, None, None, None, false)
        }
        Rv32Opcode::Slti => {
            let result = signed_lt(rs1_value, decoded.imm as u32) as u32;
            next.write_reg(decoded.rd, result);
            (result, None, None, None, false)
        }
        Rv32Opcode::Slt => {
            let result = signed_lt(rs1_value, rs2_value) as u32;
            next.write_reg(decoded.rd, result);
            (result, None, None, None, false)
        }
        Rv32Opcode::Sltiu => {
            let result = (rs1_value < decoded.imm as u32) as u32;
            next.write_reg(decoded.rd, result);
            (result, None, None, None, false)
        }
        Rv32Opcode::Sltu => {
            let result = (rs1_value < rs2_value) as u32;
            next.write_reg(decoded.rd, result);
            (result, None, None, None, false)
        }
        Rv32Opcode::Slli => {
            let result = rs1_value << shift_imm(decoded);
            next.write_reg(decoded.rd, result);
            (result, None, None, None, false)
        }
        Rv32Opcode::Sll => {
            let result = rs1_value << shift_reg(rs2_value);
            next.write_reg(decoded.rd, result);
            (result, None, None, None, false)
        }
        Rv32Opcode::Srli => {
            let result = rs1_value >> shift_imm(decoded);
            next.write_reg(decoded.rd, result);
            (result, None, None, None, false)
        }
        Rv32Opcode::Srl => {
            let result = rs1_value >> shift_reg(rs2_value);
            next.write_reg(decoded.rd, result);
            (result, None, None, None, false)
        }
        Rv32Opcode::Srai => {
            let result = ((rs1_value as i32) >> shift_imm(decoded)) as u32;
            next.write_reg(decoded.rd, result);
            (result, None, None, None, false)
        }
        Rv32Opcode::Sra => {
            let result = ((rs1_value as i32) >> shift_reg(rs2_value)) as u32;
            next.write_reg(decoded.rd, result);
            (result, None, None, None, false)
        }
        Rv32Opcode::Lui => {
            let result = decoded.imm as u32;
            next.write_reg(decoded.rd, result);
            (result, None, None, None, false)
        }
        Rv32Opcode::Auipc => {
            let result = wrapping_add_signed(prev.pc, decoded.imm);
            next.write_reg(decoded.rd, result);
            (result, None, None, None, false)
        }
        Rv32Opcode::Fence => (0, None, None, None, false),
        Rv32Opcode::Mul => {
            let result = mul_low(rs1_value, rs2_value);
            next.write_reg(decoded.rd, result);
            (result, None, None, None, false)
        }
        Rv32Opcode::Mulh => {
            let result = mul_high_signed(rs1_value, rs2_value);
            next.write_reg(decoded.rd, result);
            (result, None, None, None, false)
        }
        Rv32Opcode::Mulhsu => {
            let result = mul_high_signed_unsigned(rs1_value, rs2_value);
            next.write_reg(decoded.rd, result);
            (result, None, None, None, false)
        }
        Rv32Opcode::Mulhu => {
            let result = mul_high_unsigned(rs1_value, rs2_value);
            next.write_reg(decoded.rd, result);
            (result, None, None, None, false)
        }
        Rv32Opcode::Div => {
            let result = div_signed_result(rs1_value, rs2_value);
            next.write_reg(decoded.rd, result);
            (result, None, None, None, false)
        }
        Rv32Opcode::Divu => {
            let result = div_unsigned_result(rs1_value, rs2_value);
            next.write_reg(decoded.rd, result);
            (result, None, None, None, false)
        }
        Rv32Opcode::Rem => {
            let result = rem_signed_result(rs1_value, rs2_value);
            next.write_reg(decoded.rd, result);
            (result, None, None, None, false)
        }
        Rv32Opcode::Remu => {
            let result = rem_unsigned_result(rs1_value, rs2_value);
            next.write_reg(decoded.rd, result);
            (result, None, None, None, false)
        }
        Rv32Opcode::Lb
        | Rv32Opcode::Lbu
        | Rv32Opcode::Lh
        | Rv32Opcode::Lhu
        | Rv32Opcode::Lw
        | Rv32Opcode::Sb
        | Rv32Opcode::Sh
        | Rv32Opcode::Sw => {
            let (size_bytes, signed, writes_ram) = narrow_access_spec(decoded.opcode).expect("RV32 memory opcode");
            let addr = wrapping_add_signed(rs1_value, decoded.imm);
            let (backing_addr, byte_offset) = narrow_backing_addr(addr, size_bytes, decoded.opcode)?;
            let backing_word = prev.read_memory_word(backing_addr);
            if writes_ram {
                let blended = blend_narrow(backing_word, byte_offset, size_bytes, rs2_value);
                next.write_memory_word(backing_addr, blended);
                (blended, Some(addr), Some(backing_word), Some(blended), false)
            } else {
                let value = extract_narrow(backing_word, byte_offset, size_bytes, signed);
                next.write_reg(decoded.rd, value);
                (value, Some(addr), Some(backing_word), Some(backing_word), false)
            }
        }
        Rv32Opcode::Jal => {
            let link = prev.pc.wrapping_add(4);
            next.pc = wrapping_add_signed(prev.pc, decoded.imm);
            next.write_reg(decoded.rd, link);
            (link, None, None, None, false)
        }
        Rv32Opcode::Jalr => {
            let link = prev.pc.wrapping_add(4);
            let target = wrapping_add_signed(rs1_value, decoded.imm) & !1;
            if target % 4 != 0 {
                return Err(Rv32BuildError::Program(format!(
                    "JALR target 0x{target:08x} is not 4-byte aligned"
                )));
            }
            next.pc = target;
            next.write_reg(decoded.rd, link);
            (link, None, None, None, false)
        }
        Rv32Opcode::Beq => {
            let taken = rs1_value == rs2_value;
            if taken {
                let target = wrapping_add_signed(prev.pc, decoded.imm);
                if target % 4 != 0 {
                    return Err(Rv32BuildError::Program(format!(
                        "BEQ target 0x{target:08x} is not 4-byte aligned"
                    )));
                }
                next.pc = target;
            }
            (taken as u32, None, None, None, false)
        }
        Rv32Opcode::Bne => {
            let taken = rs1_value != rs2_value;
            if taken {
                let target = wrapping_add_signed(prev.pc, decoded.imm);
                if target % 4 != 0 {
                    return Err(Rv32BuildError::Program(format!(
                        "BNE target 0x{target:08x} is not 4-byte aligned"
                    )));
                }
                next.pc = target;
            }
            (taken as u32, None, None, None, false)
        }
        Rv32Opcode::Blt => {
            let taken = signed_lt(rs1_value, rs2_value);
            if taken {
                let target = wrapping_add_signed(prev.pc, decoded.imm);
                if target % 4 != 0 {
                    return Err(Rv32BuildError::Program(format!(
                        "BLT target 0x{target:08x} is not 4-byte aligned"
                    )));
                }
                next.pc = target;
            }
            (taken as u32, None, None, None, false)
        }
        Rv32Opcode::Bge => {
            let taken = signed_ge(rs1_value, rs2_value);
            if taken {
                let target = wrapping_add_signed(prev.pc, decoded.imm);
                if target % 4 != 0 {
                    return Err(Rv32BuildError::Program(format!(
                        "BGE target 0x{target:08x} is not 4-byte aligned"
                    )));
                }
                next.pc = target;
            }
            (taken as u32, None, None, None, false)
        }
        Rv32Opcode::Bltu => {
            let taken = rs1_value < rs2_value;
            if taken {
                let target = wrapping_add_signed(prev.pc, decoded.imm);
                if target % 4 != 0 {
                    return Err(Rv32BuildError::Program(format!(
                        "BLTU target 0x{target:08x} is not 4-byte aligned"
                    )));
                }
                next.pc = target;
            }
            (taken as u32, None, None, None, false)
        }
        Rv32Opcode::Bgeu => {
            let taken = rs1_value >= rs2_value;
            if taken {
                let target = wrapping_add_signed(prev.pc, decoded.imm);
                if target % 4 != 0 {
                    return Err(Rv32BuildError::Program(format!(
                        "BGEU target 0x{target:08x} is not 4-byte aligned"
                    )));
                }
                next.pc = target;
            }
            (taken as u32, None, None, None, false)
        }
        Rv32Opcode::Ecall => {
            next.halted = true;
            (0, None, None, None, true)
        }
    };

    next.regs[0] = 0;

    Ok(ExecutedStep {
        step_index,
        word,
        decoded,
        family,
        prev: prev.clone(),
        next,
        rs1_value,
        rs2_value,
        rd_before,
        alu_result,
        effective_addr,
        memory_before,
        memory_after,
        terminated,
    })
}
