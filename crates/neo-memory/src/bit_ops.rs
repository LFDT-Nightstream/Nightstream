use neo_math::K;
use neo_reductions::error::PiCcsError;
use p3_field::PrimeCharacteristicRing;

#[inline]
pub fn eq_bit_affine(bit: K, u: K) -> K {
    // eq(bit, u) = bit*(2u-1) + (1-u)
    bit * (u + u - K::ONE) + (K::ONE - u)
}

pub fn eq_bits_prod(bits_open: &[K], u: &[K]) -> Result<K, PiCcsError> {
    if bits_open.len() != u.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "eq_bits_prod: length mismatch (bits={}, u={})",
            bits_open.len(),
            u.len()
        )));
    }
    let mut acc = K::ONE;
    for (&b, &ui) in bits_open.iter().zip(u.iter()) {
        acc *= eq_bit_affine(b, ui);
    }
    Ok(acc)
}
