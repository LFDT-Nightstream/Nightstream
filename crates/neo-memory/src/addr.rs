use core::ops::Range;
use neo_reductions::error::PiCcsError;
use p3_field::PrimeCharacteristicRing;

use crate::cpu::BusLayout;
use crate::witness::{LutInstance, MemInstance};

pub fn for_each_addr_bit_dim_major_le(
    addr: u64,
    d: usize,
    n_side: usize,
    ell: usize,
    mut f: impl FnMut(usize, bool),
) {
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

pub fn write_addr_bits_dim_major_le<F: PrimeCharacteristicRing>(
    out_bits: &mut [F],
    addr: u64,
    d: usize,
    n_side: usize,
    ell: usize,
) {
    assert_eq!(
        out_bits.len(),
        d * ell,
        "addr_bits output length mismatch"
    );
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
    assert_eq!(
        bit_cols.end - bit_cols.start,
        d * ell,
        "addr_bits bus range mismatch"
    );
    for_each_addr_bit_dim_major_le(addr, d, n_side, ell, |idx, is_one| {
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

pub fn validate_shout_bit_addressing<Cmt, F>(inst: &LutInstance<Cmt, F>) -> Result<(), PiCcsError> {
    validate_pow2_bit_addressing("Shout", inst.n_side, inst.d, inst.ell, inst.k)?;
    if inst.table.len() != inst.k {
        return Err(PiCcsError::InvalidInput(format!(
            "Shout: table.len()={} must equal k={} for bit addressing",
            inst.table.len(),
            inst.k
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
