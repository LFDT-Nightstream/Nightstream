use neo_reductions::error::PiCcsError;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};

use crate::riscv::lookups::{compute_op, RiscvOpcode};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackedBitnessRole {
    HasLookup,
    Val,
    PackedCol(usize),
}

#[inline]
fn field_from_u64_injective<F: PrimeField64>(value: u64, label: &str) -> Result<F, PiCcsError> {
    if value >= F::ORDER_U64 {
        return Err(PiCcsError::InvalidInput(format!(
            "packed RISC-V requires injective scalar transport for {label} (value={value})"
        )));
    }
    Ok(F::from_u64(value))
}

#[inline]
pub fn rv_packed_supported_opcode(op: RiscvOpcode, xlen: usize) -> bool {
    match xlen {
        32 => rv32_packed_supported_opcode(op),
        64 => matches!(op, RiscvOpcode::Mul | RiscvOpcode::Mulhu),
        _ => false,
    }
}

#[inline]
pub fn rv_packed_rollout_opcode(op: RiscvOpcode, xlen: usize) -> bool {
    match xlen {
        32 => rv32_packed_rollout_opcode(op),
        64 => matches!(op, RiscvOpcode::Mul | RiscvOpcode::Mulhu),
        _ => false,
    }
}

#[inline]
pub fn rv32_packed_supported_opcode(op: RiscvOpcode) -> bool {
    matches!(
        op,
        RiscvOpcode::And
            | RiscvOpcode::Andn
            | RiscvOpcode::Or
            | RiscvOpcode::Xor
            | RiscvOpcode::Add
            | RiscvOpcode::Sub
            | RiscvOpcode::Eq
            | RiscvOpcode::Neq
            | RiscvOpcode::Slt
            | RiscvOpcode::Sll
            | RiscvOpcode::Srl
            | RiscvOpcode::Sra
            | RiscvOpcode::Sltu
            | RiscvOpcode::Mul
            | RiscvOpcode::VirtualMulWord
            | RiscvOpcode::Mulh
            | RiscvOpcode::Mulhu
            | RiscvOpcode::Mulhsu
            | RiscvOpcode::Div
            | RiscvOpcode::VirtualDivWord
            | RiscvOpcode::Divu
            | RiscvOpcode::VirtualDivuWord
            | RiscvOpcode::Rem
            | RiscvOpcode::VirtualRemWord
            | RiscvOpcode::Remu
            | RiscvOpcode::VirtualRemuWord
            | RiscvOpcode::VirtualMovsignWord
    )
}

#[inline]
pub fn rv32_packed_rollout_opcode(op: RiscvOpcode) -> bool {
    matches!(
        op,
        RiscvOpcode::Mul
            | RiscvOpcode::VirtualMulWord
            | RiscvOpcode::Mulh
            | RiscvOpcode::Mulhu
            | RiscvOpcode::Mulhsu
            | RiscvOpcode::Div
            | RiscvOpcode::VirtualDivWord
            | RiscvOpcode::Divu
            | RiscvOpcode::VirtualDivuWord
            | RiscvOpcode::Rem
            | RiscvOpcode::VirtualRemWord
            | RiscvOpcode::Remu
            | RiscvOpcode::VirtualRemuWord
            | RiscvOpcode::VirtualMovsignWord
    )
}

pub fn rv32_packed_d(op: RiscvOpcode) -> Result<usize, PiCcsError> {
    Ok(match op {
        RiscvOpcode::And | RiscvOpcode::Andn | RiscvOpcode::Or | RiscvOpcode::Xor => 34usize,
        RiscvOpcode::Add | RiscvOpcode::Sub => 3usize,
        RiscvOpcode::Eq | RiscvOpcode::Neq => 35usize,
        RiscvOpcode::Slt => 37usize,
        RiscvOpcode::Sll | RiscvOpcode::Srl | RiscvOpcode::Sra => 38usize,
        RiscvOpcode::VirtualMovsignWord => 38usize,
        RiscvOpcode::Sltu => 35usize,
        RiscvOpcode::Mul | RiscvOpcode::VirtualMulWord => 34usize,
        RiscvOpcode::Mulh => 38usize,
        RiscvOpcode::Mulhu => 34usize,
        RiscvOpcode::Mulhsu => 37usize,
        RiscvOpcode::Div | RiscvOpcode::VirtualDivWord | RiscvOpcode::Rem | RiscvOpcode::VirtualRemWord => 41usize,
        RiscvOpcode::Divu | RiscvOpcode::VirtualDivuWord | RiscvOpcode::Remu | RiscvOpcode::VirtualRemuWord => 37usize,
        _ => {
            return Err(PiCcsError::InvalidInput(format!(
                "packed RV32 opcode is unsupported: opcode={op:?}"
            )));
        }
    })
}

pub fn rv_packed_d(op: RiscvOpcode, xlen: usize) -> Result<usize, PiCcsError> {
    match xlen {
        32 => rv32_packed_d(op),
        64 => match op {
            RiscvOpcode::Mul | RiscvOpcode::Mulhu => Ok(66usize),
            _ => Err(PiCcsError::InvalidInput(format!(
                "packed RV64 opcode is unsupported: opcode={op:?}"
            ))),
        },
        _ => Err(PiCcsError::InvalidInput(format!(
            "packed RISC-V xlen is unsupported: xlen={xlen}"
        ))),
    }
}

fn push_col_range(out: &mut Vec<PackedBitnessRole>, start: usize, len: usize) {
    for idx in start..start + len {
        out.push(PackedBitnessRole::PackedCol(idx));
    }
}

pub fn rv32_packed_bitness_roles(op: RiscvOpcode) -> Result<Vec<PackedBitnessRole>, PiCcsError> {
    let mut out = Vec::new();
    match op {
        RiscvOpcode::And | RiscvOpcode::Andn | RiscvOpcode::Or | RiscvOpcode::Xor => {
            out.push(PackedBitnessRole::HasLookup);
        }
        RiscvOpcode::Add | RiscvOpcode::Sub => {
            out.push(PackedBitnessRole::PackedCol(2));
            out.push(PackedBitnessRole::HasLookup);
        }
        RiscvOpcode::Eq | RiscvOpcode::Neq => {
            out.push(PackedBitnessRole::HasLookup);
            out.push(PackedBitnessRole::Val);
            out.push(PackedBitnessRole::PackedCol(2));
            push_col_range(&mut out, 3, 32);
        }
        RiscvOpcode::Mul | RiscvOpcode::VirtualMulWord | RiscvOpcode::Mulhu => {
            out.push(PackedBitnessRole::HasLookup);
            push_col_range(&mut out, 2, 32);
        }
        RiscvOpcode::Mulh => {
            out.push(PackedBitnessRole::HasLookup);
            out.push(PackedBitnessRole::PackedCol(3));
            out.push(PackedBitnessRole::PackedCol(4));
            push_col_range(&mut out, 6, 32);
        }
        RiscvOpcode::Mulhsu => {
            out.push(PackedBitnessRole::HasLookup);
            out.push(PackedBitnessRole::PackedCol(3));
            out.push(PackedBitnessRole::PackedCol(4));
            push_col_range(&mut out, 5, 32);
        }
        RiscvOpcode::Slt => {
            out.push(PackedBitnessRole::Val);
            out.push(PackedBitnessRole::HasLookup);
            out.push(PackedBitnessRole::PackedCol(3));
            out.push(PackedBitnessRole::PackedCol(4));
            push_col_range(&mut out, 5, 32);
        }
        RiscvOpcode::Sll | RiscvOpcode::Srl => {
            out.push(PackedBitnessRole::HasLookup);
            push_col_range(&mut out, 1, 5);
            push_col_range(&mut out, 6, 32);
        }
        RiscvOpcode::Sra => {
            out.push(PackedBitnessRole::HasLookup);
            push_col_range(&mut out, 1, 5);
            out.push(PackedBitnessRole::PackedCol(6));
            push_col_range(&mut out, 7, 31);
        }
        RiscvOpcode::VirtualMovsignWord => {
            out.push(PackedBitnessRole::HasLookup);
            push_col_range(&mut out, 1, 5);
            out.push(PackedBitnessRole::PackedCol(6));
            push_col_range(&mut out, 7, 31);
        }
        RiscvOpcode::Sltu => {
            out.push(PackedBitnessRole::Val);
            out.push(PackedBitnessRole::HasLookup);
            push_col_range(&mut out, 3, 32);
        }
        RiscvOpcode::Divu | RiscvOpcode::VirtualDivuWord | RiscvOpcode::Remu | RiscvOpcode::VirtualRemuWord => {
            out.push(PackedBitnessRole::HasLookup);
            out.push(PackedBitnessRole::PackedCol(3));
            push_col_range(&mut out, 5, 32);
        }
        RiscvOpcode::Div | RiscvOpcode::VirtualDivWord | RiscvOpcode::Rem | RiscvOpcode::VirtualRemWord => {
            out.push(PackedBitnessRole::HasLookup);
            out.push(PackedBitnessRole::PackedCol(4));
            out.push(PackedBitnessRole::PackedCol(5));
            out.push(PackedBitnessRole::PackedCol(6));
            out.push(PackedBitnessRole::PackedCol(7));
            push_col_range(&mut out, 9, 32);
        }
        _ => {
            return Err(PiCcsError::InvalidInput(format!(
                "packed RV32 bitness roles are unsupported: opcode={op:?}"
            )));
        }
    }
    Ok(out)
}

pub fn rv32_collect_packed_bitness_terms<T: Clone>(
    op: RiscvOpcode,
    packed_cols: &[T],
    has_lookup: T,
    val: T,
) -> Result<Vec<T>, PiCcsError> {
    let roles = rv32_packed_bitness_roles(op)?;
    let mut out = Vec::with_capacity(roles.len());
    for role in roles {
        match role {
            PackedBitnessRole::HasLookup => out.push(has_lookup.clone()),
            PackedBitnessRole::Val => out.push(val.clone()),
            PackedBitnessRole::PackedCol(idx) => {
                let col = packed_cols.get(idx).ok_or_else(|| {
                    PiCcsError::InvalidInput(format!(
                        "packed RV32 bitness role index out of bounds for opcode={op:?}: idx={idx}, packed_cols={}",
                        packed_cols.len()
                    ))
                })?;
                out.push(col.clone());
            }
        }
    }
    Ok(out)
}

pub fn rv_packed_bitness_roles(op: RiscvOpcode, xlen: usize) -> Result<Vec<PackedBitnessRole>, PiCcsError> {
    if xlen == 32 {
        return rv32_packed_bitness_roles(op);
    }
    match (op, xlen) {
        (RiscvOpcode::Mul | RiscvOpcode::Mulhu, 64) => {
            let mut out = Vec::with_capacity(65);
            out.push(PackedBitnessRole::HasLookup);
            push_col_range(&mut out, 2, 64);
            Ok(out)
        }
        _ => Err(PiCcsError::InvalidInput(format!(
            "packed RISC-V bitness roles are unsupported: opcode={op:?}, xlen={xlen}"
        ))),
    }
}

pub fn rv_collect_packed_bitness_terms<T: Clone>(
    op: RiscvOpcode,
    xlen: usize,
    packed_cols: &[T],
    has_lookup: T,
    val: T,
) -> Result<Vec<T>, PiCcsError> {
    let roles = rv_packed_bitness_roles(op, xlen)?;
    let mut out = Vec::with_capacity(roles.len());
    for role in roles {
        match role {
            PackedBitnessRole::HasLookup => out.push(has_lookup.clone()),
            PackedBitnessRole::Val => out.push(val.clone()),
            PackedBitnessRole::PackedCol(idx) => {
                let col = packed_cols.get(idx).ok_or_else(|| {
                    PiCcsError::InvalidInput(format!(
                        "packed RISC-V bitness role index out of bounds for opcode={op:?}, xlen={xlen}: idx={idx}, packed_cols={}",
                        packed_cols.len()
                    ))
                })?;
                out.push(col.clone());
            }
        }
    }
    Ok(out)
}

#[inline]
fn f_bool<F: PrimeCharacteristicRing>(bit: bool) -> F {
    if bit {
        F::ONE
    } else {
        F::ZERO
    }
}

pub fn build_rv32_packed_cols<F: Field + PrimeCharacteristicRing>(
    op: RiscvOpcode,
    lhs: u32,
    rhs: u32,
    val: u32,
) -> Result<Vec<F>, PiCcsError> {
    if !rv32_packed_rollout_opcode(op) {
        return Err(PiCcsError::InvalidInput(format!(
            "packed RV32 col synthesis is unsupported for opcode={op:?}"
        )));
    }
    let expected = compute_op(op, lhs as u64, rhs as u64, 32) as u32;
    if val != expected {
        return Err(PiCcsError::InvalidInput(format!(
            "packed RV32 col synthesis value mismatch for opcode={op:?}: got={val:#x}, expected={expected:#x}"
        )));
    }

    match op {
        RiscvOpcode::Mul | RiscvOpcode::VirtualMulWord => {
            let wide = (lhs as u64) * (rhs as u64);
            let hi = (wide >> 32) as u32;
            let mut packed = Vec::with_capacity(34);
            packed.push(F::from_u64(lhs as u64));
            packed.push(F::from_u64(rhs as u64));
            for bit in 0..32usize {
                packed.push(f_bool::<F>(((hi >> bit) & 1) == 1));
            }
            Ok(packed)
        }
        RiscvOpcode::Mulhu => {
            let wide = (lhs as u64) * (rhs as u64);
            let lo = (wide & 0xffff_ffff) as u32;
            let mut packed = Vec::with_capacity(34);
            packed.push(F::from_u64(lhs as u64));
            packed.push(F::from_u64(rhs as u64));
            for bit in 0..32usize {
                packed.push(f_bool::<F>(((lo >> bit) & 1) == 1));
            }
            Ok(packed)
        }
        RiscvOpcode::Mulh => {
            let uprod = (lhs as u64) * (rhs as u64);
            let lo = (uprod & 0xffff_ffff) as u32;
            let hi = (uprod >> 32) as u32;
            let lhs_sign = (lhs >> 31) & 1;
            let rhs_sign = (rhs >> 31) & 1;

            let diff =
                (val as i128) - (hi as i128) + (lhs_sign as i128) * (rhs as i128) + (rhs_sign as i128) * (lhs as i128);
            let two32 = 1_i128 << 32;
            if diff < 0 || diff % two32 != 0 {
                return Err(PiCcsError::InvalidInput(format!(
                    "packed MULH helper: invalid k decomposition (diff={diff})"
                )));
            }
            let k = (diff / two32) as u32;
            if k > 2 {
                return Err(PiCcsError::InvalidInput(format!(
                    "packed MULH helper: k out of range (k={k})"
                )));
            }

            let mut packed = Vec::with_capacity(38);
            packed.push(F::from_u64(lhs as u64));
            packed.push(F::from_u64(rhs as u64));
            packed.push(F::from_u64(hi as u64));
            packed.push(f_bool::<F>(lhs_sign == 1));
            packed.push(f_bool::<F>(rhs_sign == 1));
            packed.push(F::from_u64(k as u64));
            for bit in 0..32usize {
                packed.push(f_bool::<F>(((lo >> bit) & 1) == 1));
            }
            Ok(packed)
        }
        RiscvOpcode::Mulhsu => {
            let uprod = (lhs as u64) * (rhs as u64);
            let lo = (uprod & 0xffff_ffff) as u32;
            let hi = (uprod >> 32) as u32;
            let lhs_sign = (lhs >> 31) & 1;

            let diff = (val as i128) - (hi as i128) + (lhs_sign as i128) * (rhs as i128);
            let two32 = 1_i128 << 32;
            if diff < 0 || diff % two32 != 0 {
                return Err(PiCcsError::InvalidInput(format!(
                    "packed MULHSU helper: invalid borrow decomposition (diff={diff})"
                )));
            }
            let borrow = (diff / two32) as u32;
            if borrow > 1 {
                return Err(PiCcsError::InvalidInput(format!(
                    "packed MULHSU helper: borrow out of range (borrow={borrow})"
                )));
            }

            let mut packed = Vec::with_capacity(37);
            packed.push(F::from_u64(lhs as u64));
            packed.push(F::from_u64(rhs as u64));
            packed.push(F::from_u64(hi as u64));
            packed.push(f_bool::<F>(lhs_sign == 1));
            packed.push(f_bool::<F>(borrow == 1));
            for bit in 0..32usize {
                packed.push(f_bool::<F>(((lo >> bit) & 1) == 1));
            }
            Ok(packed)
        }
        RiscvOpcode::Div | RiscvOpcode::VirtualDivWord => {
            let lhs_sign = (lhs >> 31) & 1;
            let rhs_sign = (rhs >> 31) & 1;
            let lhs_abs = if lhs_sign == 0 { lhs } else { lhs.wrapping_neg() };
            let rhs_abs = if rhs == 0 {
                0u32
            } else if rhs_sign == 0 {
                rhs
            } else {
                rhs.wrapping_neg()
            };

            let rhs_is_zero = rhs == 0;

            let (q_abs, r_abs) = if rhs == 0 {
                (0u32, 0u32)
            } else {
                (lhs_abs / rhs_abs, lhs_abs % rhs_abs)
            };
            let q_is_zero = q_abs == 0;
            let diff = if rhs == 0 { 0u32 } else { r_abs.wrapping_sub(rhs_abs) };

            let mut packed = Vec::with_capacity(41);
            packed.push(F::from_u64(lhs as u64));
            packed.push(F::from_u64(rhs as u64));
            packed.push(F::from_u64(q_abs as u64));
            packed.push(F::from_u64(r_abs as u64));
            packed.push(f_bool::<F>(rhs_is_zero));
            packed.push(f_bool::<F>(lhs_sign == 1));
            packed.push(f_bool::<F>(rhs_sign == 1));
            packed.push(f_bool::<F>(q_is_zero));
            packed.push(F::from_u64(diff as u64));
            for bit in 0..32usize {
                packed.push(f_bool::<F>(((diff >> bit) & 1) == 1));
            }
            Ok(packed)
        }
        RiscvOpcode::Divu | RiscvOpcode::VirtualDivuWord => {
            let rhs_is_zero = rhs == 0;

            let rem = if rhs == 0 {
                0u32
            } else {
                ((lhs as u64) % (rhs as u64)) as u32
            };
            if rhs != 0 {
                let rem_check = (lhs as u64).wrapping_sub((rhs as u64).wrapping_mul(val as u64)) as u32;
                if rem_check != rem {
                    return Err(PiCcsError::InvalidInput(format!(
                        "packed DIVU helper: invalid quotient/remainder relation (rem_check={rem_check:#x}, rem={rem:#x})"
                    )));
                }
            }
            let diff = rem.wrapping_sub(rhs);

            let mut packed = Vec::with_capacity(37);
            packed.push(F::from_u64(lhs as u64));
            packed.push(F::from_u64(rhs as u64));
            packed.push(F::from_u64(rem as u64));
            packed.push(f_bool::<F>(rhs_is_zero));
            packed.push(F::from_u64(diff as u64));
            for bit in 0..32usize {
                packed.push(f_bool::<F>(((diff >> bit) & 1) == 1));
            }
            Ok(packed)
        }
        RiscvOpcode::Rem | RiscvOpcode::VirtualRemWord => {
            let lhs_sign = (lhs >> 31) & 1;
            let rhs_sign = (rhs >> 31) & 1;
            let lhs_abs = if lhs_sign == 0 { lhs } else { lhs.wrapping_neg() };
            let rhs_abs = if rhs == 0 {
                0u32
            } else if rhs_sign == 0 {
                rhs
            } else {
                rhs.wrapping_neg()
            };

            let rhs_is_zero = rhs == 0;

            let (q_abs, r_abs) = if rhs == 0 {
                (0u32, 0u32)
            } else {
                (lhs_abs / rhs_abs, lhs_abs % rhs_abs)
            };
            let r_is_zero = r_abs == 0;
            let diff = if rhs == 0 { 0u32 } else { r_abs.wrapping_sub(rhs_abs) };

            let mut packed = Vec::with_capacity(41);
            packed.push(F::from_u64(lhs as u64));
            packed.push(F::from_u64(rhs as u64));
            packed.push(F::from_u64(q_abs as u64));
            packed.push(F::from_u64(r_abs as u64));
            packed.push(f_bool::<F>(rhs_is_zero));
            packed.push(f_bool::<F>(lhs_sign == 1));
            packed.push(f_bool::<F>(rhs_sign == 1));
            packed.push(f_bool::<F>(r_is_zero));
            packed.push(F::from_u64(diff as u64));
            for bit in 0..32usize {
                packed.push(f_bool::<F>(((diff >> bit) & 1) == 1));
            }
            Ok(packed)
        }
        RiscvOpcode::Remu | RiscvOpcode::VirtualRemuWord => {
            let rhs_is_zero = rhs == 0;

            let quot = if rhs == 0 {
                0u32
            } else {
                (lhs as u64 / rhs as u64) as u32
            };
            if rhs != 0 {
                let rem_check = ((lhs as u64) % (rhs as u64)) as u32;
                if rem_check != val {
                    return Err(PiCcsError::InvalidInput(format!(
                        "packed REMU helper: invalid remainder relation (rem_check={rem_check:#x}, val={val:#x})"
                    )));
                }
            }
            let diff = val.wrapping_sub(rhs);

            let mut packed = Vec::with_capacity(37);
            packed.push(F::from_u64(lhs as u64));
            packed.push(F::from_u64(rhs as u64));
            packed.push(F::from_u64(quot as u64));
            packed.push(f_bool::<F>(rhs_is_zero));
            packed.push(F::from_u64(diff as u64));
            for bit in 0..32usize {
                packed.push(f_bool::<F>(((diff >> bit) & 1) == 1));
            }
            Ok(packed)
        }
        RiscvOpcode::VirtualMovsignWord => {
            let shamt = 31u32;
            let expected = compute_op(RiscvOpcode::VirtualMovsignWord, lhs as u64, rhs as u64, 32) as u32;
            if val != expected {
                return Err(PiCcsError::InvalidInput(format!(
                    "packed VMOVSIGNW helper: value mismatch (got={val:#x}, expected={expected:#x})"
                )));
            }
            let sign = (lhs >> 31) & 1;
            let lhs_signed: i64 = if sign == 1 {
                (lhs as i64) - (1i64 << 32)
            } else {
                lhs as i64
            };
            let val_signed: i64 = (val as i64) - (sign as i64) * (1i64 << 32);
            let pow2: i64 = 1i64 << shamt;
            let rem_i64 = lhs_signed - val_signed * pow2;
            if rem_i64 < 0 {
                return Err(PiCcsError::InvalidInput(
                    "packed VMOVSIGNW helper: negative remainder".into(),
                ));
            }
            let rem = rem_i64 as u64;

            let mut packed = Vec::with_capacity(38);
            packed.push(F::from_u64(lhs as u64));
            for bit in 0..5usize {
                packed.push(f_bool::<F>(((shamt >> bit) & 1) == 1));
            }
            packed.push(f_bool::<F>(sign == 1));
            for bit in 0..31usize {
                packed.push(f_bool::<F>(((rem >> bit) & 1) == 1));
            }
            Ok(packed)
        }
        _ => Err(PiCcsError::InvalidInput(format!(
            "packed RV32 col synthesis is unsupported for opcode={op:?}"
        ))),
    }
}

pub fn build_rv_packed_cols<F: Field + PrimeCharacteristicRing + PrimeField64>(
    op: RiscvOpcode,
    lhs: u64,
    rhs: u64,
    val: u64,
    xlen: usize,
) -> Result<Vec<F>, PiCcsError> {
    match xlen {
        32 => {
            if lhs > u32::MAX as u64 || rhs > u32::MAX as u64 || val > u32::MAX as u64 {
                return Err(PiCcsError::InvalidInput(format!(
                    "packed RV32 col synthesis expects 32-bit operands/value (lhs={lhs}, rhs={rhs}, val={val})"
                )));
            }
            build_rv32_packed_cols(op, lhs as u32, rhs as u32, val as u32)
        }
        64 => match op {
            RiscvOpcode::Mul => {
                let expected = compute_op(op, lhs, rhs, 64);
                if val != expected {
                    return Err(PiCcsError::InvalidInput(format!(
                        "packed RV64 MUL col synthesis value mismatch: got={val:#x}, expected={expected:#x}"
                    )));
                }
                let wide = (lhs as u128) * (rhs as u128);
                let hi = (wide >> 64) as u64;
                let mut packed = Vec::with_capacity(66);
                packed.push(field_from_u64_injective::<F>(lhs, "rv64 packed mul lhs")?);
                packed.push(field_from_u64_injective::<F>(rhs, "rv64 packed mul rhs")?);
                let _ = field_from_u64_injective::<F>(val, "rv64 packed mul val")?;
                for bit in 0..64usize {
                    packed.push(f_bool::<F>(((hi >> bit) & 1) == 1));
                }
                Ok(packed)
            }
            RiscvOpcode::Mulhu => {
                let expected = compute_op(op, lhs, rhs, 64);
                if val != expected {
                    return Err(PiCcsError::InvalidInput(format!(
                        "packed RV64 MULHU col synthesis value mismatch: got={val:#x}, expected={expected:#x}"
                    )));
                }
                let wide = (lhs as u128) * (rhs as u128);
                let lo = wide as u64;
                let mut packed = Vec::with_capacity(66);
                packed.push(field_from_u64_injective::<F>(lhs, "rv64 packed mulhu lhs")?);
                packed.push(field_from_u64_injective::<F>(rhs, "rv64 packed mulhu rhs")?);
                let _ = field_from_u64_injective::<F>(val, "rv64 packed mulhu val")?;
                for bit in 0..64usize {
                    packed.push(f_bool::<F>(((lo >> bit) & 1) == 1));
                }
                Ok(packed)
            }
            _ => Err(PiCcsError::InvalidInput(format!(
                "packed RV64 col synthesis is unsupported for opcode={op:?}"
            ))),
        },
        _ => Err(PiCcsError::InvalidInput(format!(
            "packed RISC-V col synthesis is unsupported for xlen={xlen}"
        ))),
    }
}
