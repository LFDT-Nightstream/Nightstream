use super::*;

pub(super) fn validate_source_bit_range(
    source: &DirectCcsFPrimeLowNormSourceImage,
    offset: usize,
    label: &str,
) -> Result<(), DirectCcsFPrimeSnarkError> {
    if offset
        .checked_add(CONSTRUCTION2_ENC_INST_BITS)
        .is_some_and(|end| end <= source.len())
    {
        Ok(())
    } else {
        Err(DirectCcsFPrimeSnarkError::Input(format!(
            "direct F' source {label} bits are outside the low-norm source image"
        )))
    }
}

pub(super) fn validate_source_u64_range(
    source: &DirectCcsFPrimeLowNormSourceImage,
    offset: usize,
    label: &str,
) -> Result<(), DirectCcsFPrimeSnarkError> {
    if offset
        .checked_add(64)
        .is_some_and(|end| end <= source.len())
    {
        Ok(())
    } else {
        Err(DirectCcsFPrimeSnarkError::Input(format!(
            "direct F' source {label} bits are outside the low-norm source image"
        )))
    }
}

pub(super) fn source_u64_at(
    source: &DirectCcsFPrimeLowNormSourceImage,
    offset: usize,
) -> Result<u64, DirectCcsFPrimeSnarkError> {
    validate_source_u64_range(source, offset, "u64")?;
    let mut out = 0u64;
    for bit_index in 0..64 {
        let value = source.values()[offset + bit_index];
        if value == F::ONE {
            out |= 1u64 << bit_index;
        } else if value != F::ZERO {
            return Err(DirectCcsFPrimeSnarkError::Input(
                "direct F' source u64 contains a non-binary value".into(),
            ));
        }
    }
    Ok(out)
}

pub(super) fn canonical_field_lane_aux_bits(
    source: &DirectCcsFPrimeLowNormSourceImage,
) -> Result<Vec<u8>, DirectCcsFPrimeSnarkError> {
    let mut out = Vec::with_capacity(source.field_lane_count() * GOLDILOCKS_CANONICAL_AUX_BITS_PER_LANE);
    for &offset in source.field_lane_bit_offsets() {
        let mut high_all =
            source_bit(source, offset + GOLDILOCKS_LOW_BITS)? & source_bit(source, offset + GOLDILOCKS_LOW_BITS + 1)?;
        out.push(high_all);
        for high_index in 2..GOLDILOCKS_HIGH_BITS {
            high_all &= source_bit(source, offset + GOLDILOCKS_LOW_BITS + high_index)?;
            out.push(high_all);
        }
    }
    Ok(out)
}

pub(super) fn source_bit(
    source: &DirectCcsFPrimeLowNormSourceImage,
    offset: usize,
) -> Result<u8, DirectCcsFPrimeSnarkError> {
    match source.values().get(offset).copied() {
        Some(value) if value == F::ZERO => Ok(0),
        Some(value) if value == F::ONE => Ok(1),
        Some(_) => Err(DirectCcsFPrimeSnarkError::Input(
            "direct F' source bit contains a non-binary value".into(),
        )),
        None => Err(DirectCcsFPrimeSnarkError::Input(
            "direct F' source bit offset is outside the low-norm source image".into(),
        )),
    }
}
