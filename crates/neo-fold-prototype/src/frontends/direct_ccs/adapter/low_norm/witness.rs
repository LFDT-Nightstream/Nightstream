//! Witness expansion for low-norm direct R1CS lowering.

use super::*;

pub(super) fn push_lanes_for_range(
    export: &DirectSparseR1csExport,
    layout: &DirectR1csLowNormLayout,
    range: std::ops::Range<usize>,
    witness: &mut Vec<F>,
    lanes: &mut [LaneMap],
) -> Result<usize, DirectCcsFPrimeSnarkError> {
    for col in range {
        let start = witness.len();
        let kind = layout.kinds[col];
        push_lane_bits(witness, export.witness[col], kind, col)?;
        lanes[col].bits_start_col = start;
        lanes[col].bit_len = kind.bit_len();
    }
    Ok(witness.len())
}

pub(super) fn push_canonical_aux_bits(
    export: &DirectSparseR1csExport,
    layout: &DirectR1csLowNormLayout,
    witness: &mut Vec<F>,
    lanes: &mut [LaneMap],
) -> Result<(), DirectCcsFPrimeSnarkError> {
    for (col, &kind) in layout.kinds.iter().enumerate() {
        if !kind.needs_canonical_field_check() {
            continue;
        }
        let start = witness.len();
        lanes[col].canonical_aux_start_col = Some(start);
        let value = export.witness[col].as_canonical_u64();
        let mut high_all = ((value >> GOLDILOCKS_LOW_BITS) & 1) & ((value >> (GOLDILOCKS_LOW_BITS + 1)) & 1);
        witness.push(F::from_u64(high_all));
        for high_index in 2..GOLDILOCKS_HIGH_BITS {
            high_all &= (value >> (GOLDILOCKS_LOW_BITS + high_index)) & 1;
            witness.push(F::from_u64(high_all));
        }
        if witness.len() != start + GOLDILOCKS_CANONICAL_AUX_BITS {
            return Err(DirectCcsFPrimeSnarkError::Input(
                "direct low-norm R1CS canonical aux layout mismatch".into(),
            ));
        }
    }
    Ok(())
}

fn push_lane_bits(
    witness: &mut Vec<F>,
    value: F,
    kind: DirectLowNormLaneKind,
    original_col: usize,
) -> Result<(), DirectCcsFPrimeSnarkError> {
    let raw = value.as_canonical_u64();
    match kind {
        DirectLowNormLaneKind::Bit => {
            if raw > 1 {
                return Err(DirectCcsFPrimeSnarkError::Input(format!(
                    "direct low-norm R1CS bit lane at original column {original_col} has value {raw}"
                )));
            }
            witness.push(F::from_u64(raw));
        }
        DirectLowNormLaneKind::U32 => {
            if raw > u32::MAX as u64 {
                return Err(DirectCcsFPrimeSnarkError::Input(format!(
                    "direct low-norm R1CS u32 lane at original column {original_col} has value {raw}"
                )));
            }
            push_bits(witness, raw, U32_BITS);
        }
        DirectLowNormLaneKind::Field => push_bits(witness, raw, FIELD_BITS),
    }
    Ok(())
}

fn push_bits(witness: &mut Vec<F>, value: u64, bit_len: usize) {
    for bit_index in 0..bit_len {
        witness.push(F::from_u64((value >> bit_index) & 1));
    }
}
