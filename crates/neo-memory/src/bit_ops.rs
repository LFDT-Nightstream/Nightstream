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

pub fn eq_bits_prod_table(bit_cols: &[Vec<K>], r_addr: &[K]) -> Result<Vec<K>, PiCcsError> {
    if bit_cols.len() != r_addr.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "eq_bits_prod_table: length mismatch (bit_cols={}, r_addr={})",
            bit_cols.len(),
            r_addr.len()
        )));
    }
    let n = bit_cols.first().map(|c| c.len()).unwrap_or(0);
    for (idx, col) in bit_cols.iter().enumerate() {
        if col.len() != n {
            return Err(PiCcsError::InvalidInput(format!(
                "eq_bits_prod_table: inconsistent column length at idx {} (got {}, expected {})",
                idx,
                col.len(),
                n
            )));
        }
    }

    let mut result = vec![K::ONE; n];
    for (col, &r) in bit_cols.iter().zip(r_addr.iter()) {
        for (i, &b) in col.iter().enumerate() {
            result[i] *= eq_bit_affine(b, r);
        }
    }
    Ok(result)
}
