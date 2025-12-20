use neo_reductions::error::PiCcsError;

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
