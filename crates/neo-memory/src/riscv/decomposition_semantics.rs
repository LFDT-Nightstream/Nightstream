use crate::riscv::instruction::{decomposition_sequence_for_instruction, DecomposedOp};
use crate::riscv::lookups::{compute_op, decode_instruction_with_xlen, RiscvOpcode};

#[inline]
fn mask_to_xlen(value: u64, xlen: usize) -> u64 {
    if xlen >= 64 {
        value
    } else {
        value & ((1u64 << xlen) - 1)
    }
}

#[inline]
/// Resolve the expected virtual micro-op for a row from `(instr_word, remaining)`.
///
/// `remaining` uses the same inclusive countdown as `virtual_sequence_remaining`.
pub fn expected_virtual_decomposed_op(
    instr_word: u32,
    remaining: u32,
    machine_xlen: usize,
) -> Result<DecomposedOp, String> {
    if machine_xlen != 32 && machine_xlen != 64 {
        return Err(format!(
            "unsupported machine_xlen for virtual decomposition: {machine_xlen}"
        ));
    }
    if remaining == 0 {
        return Err("virtual row must have virtual_sequence_remaining > 0".into());
    }
    let instr = decode_instruction_with_xlen(instr_word, machine_xlen)
        .map_err(|e| format!("virtual-row decode failed for instr_word={instr_word:#x}: {e}"))?;
    let seq = decomposition_sequence_for_instruction(&instr)
        .ok_or_else(|| format!("no decomposition sequence for virtual row instruction: {instr:?}"))?;
    if seq.len() < 2 {
        return Err(format!(
            "decomposition sequence must include virtual+commit rows (len={})",
            seq.len()
        ));
    }
    let remaining = remaining as usize;
    if remaining >= seq.len() {
        return Err(format!(
            "virtual_sequence_remaining out of range for sequence len {}: {}",
            seq.len(),
            remaining
        ));
    }
    let idx = seq.len() - remaining - 1;
    let op = seq
        .get(idx)
        .copied()
        .ok_or_else(|| format!("virtual row index out of range: idx={idx}, len={}", seq.len()))?;
    Ok(op)
}

/// Validate virtual micro-op shape + local semantics against row IO.
///
/// This mirrors runtime virtual execution semantics for decomposition rows.
#[allow(clippy::too_many_arguments)]
pub fn validate_virtual_row_semantics(
    op: DecomposedOp,
    rs1_addr: u64,
    rs1_val: u64,
    rs2_addr: u64,
    rs2_val: u64,
    rd_has_write: bool,
    rd_addr: u64,
    rd_val: u64,
    xlen: usize,
) -> Result<(), String> {
    let require_reads = |lhs: u64, rhs: u64| -> Result<(), String> {
        if rs1_addr != lhs || rs2_addr != rhs {
            return Err(format!(
                "virtual {:?} read addr mismatch: got (rs1={rs1_addr}, rs2={rs2_addr}), expected (lhs={lhs}, rhs={rhs})",
                op
            ));
        }
        Ok(())
    };
    let require_write = |dst: u64, expected_val: u64| -> Result<(), String> {
        if !rd_has_write {
            return Err(format!("virtual {:?} expected register write", op));
        }
        if rd_addr != dst {
            return Err(format!(
                "virtual {:?} write addr mismatch: got rd_addr={rd_addr}, expected {dst}",
                op
            ));
        }
        let expected_val = mask_to_xlen(expected_val, xlen);
        if rd_val != expected_val {
            return Err(format!(
                "virtual {:?} write value mismatch: got rd_val={rd_val:#x}, expected {expected_val:#x}",
                op
            ));
        }
        Ok(())
    };
    let require_no_write = || -> Result<(), String> {
        if rd_has_write {
            return Err(format!(
                "virtual {:?} should not write (rd_addr={rd_addr}, rd_val={rd_val:#x})",
                op
            ));
        }
        Ok(())
    };

    match op {
        DecomposedOp::Advice { dst } => {
            require_reads(0, 0)?;
            require_write(dst, 0)?;
        }
        DecomposedOp::AdviceRemainderAbs { dst, dividend, divisor } => {
            require_reads(dividend, divisor)?;
            let rem = compute_op(RiscvOpcode::Rem, rs1_val, rs2_val, xlen);
            let rem_abs = if xlen == 32 {
                (rem as u32 as i32).unsigned_abs() as u64
            } else {
                (rem as i64).unsigned_abs()
            };
            require_write(dst, rem_abs)?;
        }
        DecomposedOp::AdviceQuotient { dst, op, lhs, rhs } => {
            require_reads(lhs, rhs)?;
            let out = compute_op(op, rs1_val, rs2_val, xlen);
            require_write(dst, out)?;
        }
        DecomposedOp::MovSignWord { dst, src } => {
            require_reads(src, 0)?;
            let sign_mask = if ((rs1_val >> 31) & 1) == 1 { 0xFFFF_FFFF } else { 0 };
            require_write(dst, sign_mask)?;
        }
        DecomposedOp::MovSign { dst, src } => {
            require_reads(src, 0)?;
            let sign_bit = xlen - 1;
            let sign_set = ((rs1_val >> sign_bit) & 1) == 1;
            let sign_mask = if sign_set { mask_to_xlen(u64::MAX, xlen) } else { 0 };
            require_write(dst, sign_mask)?;
        }
        DecomposedOp::ComposeU64FromLoHi32 { dst, lo_src, hi_src } => {
            require_reads(lo_src, hi_src)?;
            let out = (rs1_val as u32 as u64) | ((rs2_val as u32 as u64) << 32);
            require_write(dst, out)?;
        }
        DecomposedOp::Move { dst, src } => {
            require_reads(src, 0)?;
            require_write(dst, rs1_val)?;
        }
        DecomposedOp::Add { dst, lhs, rhs } => {
            require_reads(lhs, rhs)?;
            require_write(dst, compute_op(RiscvOpcode::Add, rs1_val, rs2_val, xlen))?;
        }
        DecomposedOp::Sub { dst, lhs, rhs } => {
            require_reads(lhs, rhs)?;
            require_write(dst, compute_op(RiscvOpcode::Sub, rs1_val, rs2_val, xlen))?;
        }
        DecomposedOp::Xor { dst, lhs, rhs } => {
            require_reads(lhs, rhs)?;
            require_write(dst, compute_op(RiscvOpcode::Xor, rs1_val, rs2_val, xlen))?;
        }
        DecomposedOp::Mul { dst, lhs, rhs } => {
            require_reads(lhs, rhs)?;
            require_write(dst, compute_op(RiscvOpcode::Mul, rs1_val, rs2_val, xlen))?;
        }
        DecomposedOp::Mulhu { dst, lhs, rhs } => {
            require_reads(lhs, rhs)?;
            require_write(dst, compute_op(RiscvOpcode::Mulhu, rs1_val, rs2_val, xlen))?;
        }
        DecomposedOp::AssertEq { lhs, rhs } => {
            require_reads(lhs, rhs)?;
            require_no_write()?;
            if rs1_val != rs2_val {
                return Err(format!(
                    "virtual AssertEq predicate failed: lhs={rs1_val:#x}, rhs={rs2_val:#x}"
                ));
            }
        }
        DecomposedOp::AssertLtu { lhs, rhs } => {
            require_reads(lhs, rhs)?;
            require_no_write()?;
            if rs1_val >= rs2_val {
                return Err(format!(
                    "virtual AssertLtu predicate failed: lhs={rs1_val:#x}, rhs={rs2_val:#x}"
                ));
            }
        }
        DecomposedOp::AssertLte { lhs, rhs } => {
            require_reads(lhs, rhs)?;
            require_no_write()?;
            if rs1_val > rs2_val {
                return Err(format!(
                    "virtual AssertLte predicate failed: lhs={rs1_val:#x}, rhs={rs2_val:#x}"
                ));
            }
        }
        DecomposedOp::AssertLtAbs { lhs, rhs } => {
            require_reads(lhs, rhs)?;
            require_no_write()?;
            let lhs_abs = if xlen == 32 {
                (rs1_val as u32 as i32).unsigned_abs() as u64
            } else {
                (rs1_val as i64).unsigned_abs()
            };
            let rhs_abs = if xlen == 32 {
                (rs2_val as u32 as i32).unsigned_abs() as u64
            } else {
                (rs2_val as i64).unsigned_abs()
            };
            if lhs_abs >= rhs_abs {
                return Err(format!(
                    "virtual AssertLtAbs predicate failed: |lhs|={lhs_abs:#x}, |rhs|={rhs_abs:#x}"
                ));
            }
        }
        DecomposedOp::AssertEqSigns { lhs, rhs } => {
            require_reads(lhs, rhs)?;
            require_no_write()?;
            let sign_bit = xlen - 1;
            let lhs_sign = (rs1_val >> sign_bit) & 1;
            let rhs_sign = (rs2_val >> sign_bit) & 1;
            if lhs_sign != rhs_sign {
                return Err(format!(
                    "virtual AssertEqSigns predicate failed: lhs_sign={lhs_sign}, rhs_sign={rhs_sign}"
                ));
            }
        }
        DecomposedOp::AssertValidDiv0 { divisor, quotient } => {
            require_reads(divisor, quotient)?;
            require_no_write()?;
            let all_ones = mask_to_xlen(u64::MAX, xlen);
            if rs1_val == 0 && rs2_val != all_ones {
                return Err(format!(
                    "virtual AssertValidDiv0 predicate failed: divisor=0, quotient={rs2_val:#x}, expected={all_ones:#x}"
                ));
            }
        }
        DecomposedOp::ChangeDivisor { dst, dividend, divisor } => {
            require_reads(dividend, divisor)?;
            let out = if xlen == 32 {
                let dividend_i = rs1_val as u32 as i32;
                let divisor_i = rs2_val as u32 as i32;
                if dividend_i == i32::MIN && divisor_i == -1 {
                    1u64
                } else {
                    rs2_val
                }
            } else {
                let dividend_i = rs1_val as i64;
                let divisor_i = rs2_val as i64;
                if dividend_i == i64::MIN && divisor_i == -1 {
                    1u64
                } else {
                    rs2_val
                }
            };
            require_write(dst, out)?;
        }
        DecomposedOp::AssertMulUNoOverflow { lhs, rhs } => {
            require_reads(lhs, rhs)?;
            require_no_write()?;
            let hi = compute_op(RiscvOpcode::Mulhu, rs1_val, rs2_val, xlen);
            if hi != 0 {
                return Err(format!(
                    "virtual AssertMulUNoOverflow predicate failed: lhs={rs1_val:#x}, rhs={rs2_val:#x}, hi={hi:#x}"
                ));
            }
        }
        DecomposedOp::AssertValidUnsignedRemainder { remainder, divisor } => {
            require_reads(remainder, divisor)?;
            require_no_write()?;
            if rs2_val != 0 && rs1_val >= rs2_val {
                return Err(format!(
                    "virtual AssertValidUnsignedRemainder predicate failed: remainder={rs1_val:#x}, divisor={rs2_val:#x}"
                ));
            }
        }
    }

    Ok(())
}
