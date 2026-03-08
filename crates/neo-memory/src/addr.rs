use core::ops::Range;
use neo_reductions::error::PiCcsError;
use p3_field::PrimeCharacteristicRing;

use crate::cpu::BusLayout;
use crate::riscv::shout_oracle::RiscvAddressLookupOracleSparse;
use crate::riscv::trace::rv32_trace_lookup_n_vals_for_table_id;
use crate::witness::{LutInstance, LutTableSpec, MemInstance};

pub fn for_each_addr_bit_dim_major_le(addr: u64, d: usize, n_side: usize, ell: usize, mut f: impl FnMut(usize, bool)) {
    assert!(n_side > 0, "n_side must be > 0");
    let mut tmp = addr;
    for dim in 0..d {
        let comp = (tmp % (n_side as u64)) as u64;
        tmp /= n_side as u64;
        for bit in 0..ell {
            let idx = dim * ell + bit;
            let is_one = ((comp >> bit) & 1) == 1;
            f(idx, is_one);
        }
    }
}

pub fn for_each_addr_bit_dim_major_le_u128(
    addr: u128,
    d: usize,
    n_side: usize,
    ell: usize,
    mut f: impl FnMut(usize, bool),
) {
    assert!(n_side > 0, "n_side must be > 0");
    let base = n_side as u128;
    let mut tmp = addr;
    for dim in 0..d {
        let comp = tmp % base;
        tmp /= base;
        for bit in 0..ell {
            let idx = dim * ell + bit;
            let is_one = ((comp >> bit) & 1) == 1;
            f(idx, is_one);
        }
    }
}

pub fn write_addr_bits_dim_major_le<F: PrimeCharacteristicRing>(
    out_bits: &mut [F],
    addr: u64,
    d: usize,
    n_side: usize,
    ell: usize,
) {
    assert_eq!(out_bits.len(), d * ell, "addr_bits output length mismatch");
    for_each_addr_bit_dim_major_le(addr, d, n_side, ell, |idx, is_one| {
        out_bits[idx] = if is_one { F::ONE } else { F::ZERO };
    });
}

pub fn write_addr_bits_dim_major_le_into_bus<F: PrimeCharacteristicRing>(
    z: &mut [F],
    bus: &BusLayout,
    bit_cols: Range<usize>,
    j: usize,
    addr: u64,
    d: usize,
    n_side: usize,
    ell: usize,
) {
    assert_eq!(bit_cols.end - bit_cols.start, d * ell, "addr_bits bus range mismatch");
    for_each_addr_bit_dim_major_le(addr, d, n_side, ell, |idx, is_one| {
        let col_id = bit_cols.start + idx;
        z[bus.bus_cell(col_id, j)] = if is_one { F::ONE } else { F::ZERO };
    });
}

pub fn write_addr_bits_dim_major_le_u128_into_bus<F: PrimeCharacteristicRing>(
    z: &mut [F],
    bus: &BusLayout,
    bit_cols: Range<usize>,
    j: usize,
    addr: u128,
    d: usize,
    n_side: usize,
    ell: usize,
) {
    assert_eq!(bit_cols.end - bit_cols.start, d * ell, "addr_bits bus range mismatch");
    for_each_addr_bit_dim_major_le_u128(addr, d, n_side, ell, |idx, is_one| {
        let col_id = bit_cols.start + idx;
        z[bus.bus_cell(col_id, j)] = if is_one { F::ONE } else { F::ZERO };
    });
}

pub fn validate_pow2_bit_addressing(
    proto: &'static str,
    n_side: usize,
    d: usize,
    ell: usize,
    k: usize,
) -> Result<(), PiCcsError> {
    if n_side == 0 {
        return Err(PiCcsError::InvalidInput(format!("{proto}: n_side must be > 0")));
    }
    if !n_side.is_power_of_two() {
        return Err(PiCcsError::InvalidInput(format!(
            "{proto}: n_side={n_side} must be a power of two under bit addressing (otherwise a range proof is required)"
        )));
    }

    let expected_ell = n_side.trailing_zeros() as usize;
    if ell != expected_ell {
        return Err(PiCcsError::InvalidInput(format!(
            "{proto}: ell={ell} must equal log2(n_side)={expected_ell} for power-of-two n_side"
        )));
    }

    let expected_k = n_side
        .checked_pow(d as u32)
        .ok_or_else(|| PiCcsError::InvalidInput(format!("{proto}: n_side^d overflow")))?;
    if k != expected_k {
        return Err(PiCcsError::InvalidInput(format!(
            "{proto}: k={k} must equal n_side^d = {n_side}^{d} = {expected_k} for bit addressing"
        )));
    }

    Ok(())
}

pub fn validate_pow2_bit_addressing_shape(proto: &'static str, n_side: usize, ell: usize) -> Result<(), PiCcsError> {
    if n_side == 0 {
        return Err(PiCcsError::InvalidInput(format!("{proto}: n_side must be > 0")));
    }
    if !n_side.is_power_of_two() {
        return Err(PiCcsError::InvalidInput(format!(
            "{proto}: n_side={n_side} must be a power of two under bit addressing (otherwise a range proof is required)"
        )));
    }

    let expected_ell = n_side.trailing_zeros() as usize;
    if ell != expected_ell {
        return Err(PiCcsError::InvalidInput(format!(
            "{proto}: ell={ell} must equal log2(n_side)={expected_ell} for power-of-two n_side"
        )));
    }

    Ok(())
}

pub fn validate_shout_bit_addressing<Cmt, F>(inst: &LutInstance<Cmt, F>) -> Result<(), PiCcsError> {
    // Virtual/implicit tables may not have a materialized `k = n_side^d` table.
    if let Some(spec) = &inst.table_spec {
        validate_pow2_bit_addressing_shape("Shout", inst.n_side, inst.ell)?;
        if inst.k != 0 {
            return Err(PiCcsError::InvalidInput(
                "Shout: k must be 0 when table_spec is set (implicit table)".into(),
            ));
        }
        if !inst.table.is_empty() {
            return Err(PiCcsError::InvalidInput(
                "Shout: table must be empty when table_spec is set".into(),
            ));
        }

        match spec {
            LutTableSpec::RiscvOpcode { opcode, xlen } => {
                RiscvAddressLookupOracleSparse::validate_spec(*opcode, *xlen)?;
                if inst.n_side != 2 || inst.ell != 1 {
                    return Err(PiCcsError::InvalidInput(format!(
                        "Shout(RISC-V): expected n_side=2, ell=1, got n_side={}, ell={}",
                        inst.n_side, inst.ell
                    )));
                }
                let expected_d = xlen
                    .checked_mul(2)
                    .ok_or_else(|| PiCcsError::InvalidInput("Shout(RISC-V): 2*xlen overflow".into()))?;
                if inst.d != expected_d {
                    return Err(PiCcsError::InvalidInput(format!(
                        "Shout(RISC-V): expected d=2*xlen={}, got d={}",
                        expected_d, inst.d
                    )));
                }
            }
            LutTableSpec::RiscvOpcodePacked { opcode, xlen } => {
                let expected_d = crate::riscv::packed::rv_packed_d(*opcode, *xlen)?;
                // Packed-key Shout lanes are not bit-addressed: we repurpose the addr-bit slice as
                // `[lhs, rhs, aux...]` and keep `[has_lookup, val]`.
                if inst.n_side != 2 || inst.ell != 1 {
                    return Err(PiCcsError::InvalidInput(format!(
                        "Shout(RISC-V packed): expected n_side=2, ell=1, got n_side={}, ell={}",
                        inst.n_side, inst.ell
                    )));
                }
                if inst.d != expected_d {
                    return Err(PiCcsError::InvalidInput(format!(
                        "Shout(RISC-V packed): expected d={expected_d}, got d={}",
                        inst.d,
                    )));
                }
            }
            LutTableSpec::RiscvOpcodeEventTablePacked {
                opcode,
                xlen,
                time_bits,
            } => {
                if *xlen != 32 {
                    return Err(PiCcsError::InvalidInput(format!(
                        "Shout(RISC-V event-table packed): expected xlen=32, got xlen={xlen}"
                    )));
                }
                if *time_bits == 0 {
                    return Err(PiCcsError::InvalidInput(
                        "Shout(RISC-V event-table packed): time_bits must be >= 1".into(),
                    ));
                }
                if *time_bits > 64 {
                    return Err(PiCcsError::InvalidInput(format!(
                        "Shout(RISC-V event-table packed): time_bits={time_bits} too large (max 64)"
                    )));
                }
                let base_d = crate::riscv::packed::rv_packed_d(*opcode, *xlen)?;
                let expected_d = time_bits
                    .checked_add(base_d)
                    .ok_or_else(|| PiCcsError::InvalidInput("Shout(RISC-V event-table packed): d overflow".into()))?;

                // Event-table packed Shout lanes are not bit-addressed: addr_bits is repurposed as
                // `[time_bits_le, lhs_u32, rhs_u32, aux...]` and we keep `[has_lookup, val_u32]`.
                if inst.n_side != 2 || inst.ell != 1 {
                    return Err(PiCcsError::InvalidInput(format!(
                        "Shout(RISC-V event-table packed): expected n_side=2, ell=1, got n_side={}, ell={}",
                        inst.n_side, inst.ell
                    )));
                }
                if inst.d != expected_d {
                    return Err(PiCcsError::InvalidInput(format!(
                        "Shout(RISC-V event-table packed): expected d={expected_d} (= time_bits({time_bits}) + base_d({base_d})), got d={}",
                        inst.d,
                    )));
                }
            }
            LutTableSpec::IdentityU32 => {
                if inst.n_side != 2 || inst.ell != 1 {
                    return Err(PiCcsError::InvalidInput(format!(
                        "Shout(IdentityU32): expected n_side=2, ell=1, got n_side={}, ell={}",
                        inst.n_side, inst.ell
                    )));
                }
                if inst.d != 32 {
                    return Err(PiCcsError::InvalidInput(format!(
                        "Shout(IdentityU32): expected d=32, got d={}",
                        inst.d
                    )));
                }
            }
        }

        return Ok(());
    }

    // Explicit table mode (legacy).
    validate_pow2_bit_addressing("Shout", inst.n_side, inst.d, inst.ell, inst.k)?;
    let n_vals = rv32_trace_lookup_n_vals_for_table_id(inst.table_id);
    let expected_len = inst.k.checked_mul(n_vals).ok_or_else(|| {
        PiCcsError::InvalidInput(format!(
            "Shout: table length overflow for k={} and n_vals={n_vals}",
            inst.k
        ))
    })?;
    if inst.table.len() != expected_len {
        return Err(PiCcsError::InvalidInput(format!(
            "Shout: table.len()={} must equal k*n_vals={} (k={}, n_vals={n_vals}) for bit addressing",
            inst.table.len(),
            expected_len,
            inst.k,
        )));
    }
    Ok(())
}

pub fn validate_twist_bit_addressing<Cmt, F: PrimeCharacteristicRing>(
    inst: &MemInstance<Cmt, F>,
) -> Result<(), PiCcsError> {
    validate_pow2_bit_addressing("Twist", inst.n_side, inst.d, inst.ell, inst.k)?;
    inst.init.validate(inst.k)?;
    Ok(())
}
