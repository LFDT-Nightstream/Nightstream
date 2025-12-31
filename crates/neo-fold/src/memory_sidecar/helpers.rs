use crate::PiCcsError;
use neo_math::K;
use p3_field::PrimeCharacteristicRing;

pub fn check_bitness_terminal(
    chi_cycle_at_r_time: K,
    bit_open: K,
    observed_final: K,
    ctx: &'static str,
) -> Result<(), PiCcsError> {
    let expected = chi_cycle_at_r_time * bit_open * (bit_open - K::ONE);
    if expected != observed_final {
        return Err(PiCcsError::ProtocolError(format!("{ctx}: bitness terminal mismatch")));
    }
    Ok(())
}
