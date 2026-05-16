use super::*;

pub(super) fn validate_low_norm_source_r1cs_inputs(
    source: &DirectCcsFPrimeLowNormSourceImage,
    expected_kappa: u64,
) -> Result<(), DirectCcsFPrimeSnarkError> {
    if expected_kappa == 0 {
        return Err(DirectCcsFPrimeSnarkError::Input(
            "direct F' source R1CS requires nonzero Construction-2 commitment kappa".into(),
        ));
    }
    if source.is_empty() {
        return Err(DirectCcsFPrimeSnarkError::Input(
            "direct F' low-norm source R1CS requires a non-empty source image".into(),
        ));
    }
    validate_digest_source_ranges(source)?;
    validate_public_input_source_ranges(source)?;
    validate_counter_source_ranges(source)?;
    validate_nifs_source_ranges(source)?;
    validate_construction2_boundary_ranges(
        source,
        "input",
        source.construction2_u_in_fresh_digest_bit_offset(),
        source.construction2_u_in_commitment_digest_bit_offset(),
        source.construction2_u_in_commitment_d_bit_offset(),
        source.construction2_u_in_commitment_kappa_bit_offset(),
        source.construction2_u_in_x_i_bit_offset(),
    )?;
    validate_construction2_boundary_ranges(
        source,
        "output",
        source.construction2_u_out_fresh_digest_bit_offset(),
        source.construction2_u_out_commitment_digest_bit_offset(),
        source.construction2_u_out_commitment_d_bit_offset(),
        source.construction2_u_out_commitment_kappa_bit_offset(),
        source.construction2_u_out_x_i_bit_offset(),
    )?;
    validate_field_lane_ranges(source)?;
    Ok(())
}

fn validate_digest_source_ranges(source: &DirectCcsFPrimeLowNormSourceImage) -> Result<(), DirectCcsFPrimeSnarkError> {
    for (offset, label) in [
        (source.mat_digest_bit_offset(), "matrix digest"),
        (source.vk_fs_digest_bit_offset(), "vk_fs digest"),
        (source.initial_boundary_digest_bit_offset(), "initial boundary digest"),
        (
            source.current_boundary_in_digest_bit_offset(),
            "current boundary input digest",
        ),
        (
            source.current_boundary_out_digest_bit_offset(),
            "current boundary output digest",
        ),
        (source.public_trace_in_digest_bit_offset(), "public trace input digest"),
        (
            source.public_trace_out_digest_bit_offset(),
            "public trace output digest",
        ),
        (
            source.semantic_accumulator_in_digest_bit_offset(),
            "semantic accumulator input digest",
        ),
        (
            source.semantic_accumulator_out_digest_bit_offset(),
            "semantic accumulator output digest",
        ),
        (
            source.f_prime_accumulator_in_digest_bit_offset(),
            "F' accumulator input digest",
        ),
        (
            source.f_prime_accumulator_out_digest_bit_offset(),
            "F' accumulator output digest",
        ),
        (
            source.compact_construction2_u_in_digest_bit_offset(),
            "compact Construction-2 input digest",
        ),
        (
            source.compact_construction2_u_out_digest_bit_offset(),
            "compact Construction-2 output digest",
        ),
        (source.latest_chunk_digest_bit_offset(), "latest chunk digest"),
        (source.latest_fold_digest_bit_offset(), "latest fold digest"),
        (
            source.latest_chunk_relation_digest_bit_offset(),
            "latest chunk relation digest",
        ),
    ] {
        validate_source_bit_range(source, offset, label)?;
    }
    Ok(())
}

fn validate_public_input_source_ranges(
    source: &DirectCcsFPrimeLowNormSourceImage,
) -> Result<(), DirectCcsFPrimeSnarkError> {
    for (offset, label) in [
        (source.compact_x_in_bit_offset(), "compact x_in"),
        (source.compact_x_out_bit_offset(), "compact x_out"),
    ] {
        validate_source_bit_range(source, offset, label)?;
    }
    Ok(())
}

fn validate_counter_source_ranges(source: &DirectCcsFPrimeLowNormSourceImage) -> Result<(), DirectCcsFPrimeSnarkError> {
    for (offset, label) in [
        (source.pc_bit_offset(), "pc"),
        (source.chunk_count_in_bit_offset(), "chunk count input"),
        (source.chunk_count_out_bit_offset(), "chunk count output"),
        (source.step_count_in_bit_offset(), "step count input"),
        (source.step_count_out_bit_offset(), "step count output"),
        (source.fresh_claims_bit_offset(), "fresh claim count"),
        (source.incoming_ce_claims_bit_offset(), "incoming CE claim count"),
        (source.output_ce_claims_bit_offset(), "output CE claim count"),
        (source.final_ce_claims_bit_offset(), "final CE claim count"),
    ] {
        validate_source_u64_range(source, offset, label)?;
    }
    Ok(())
}

fn validate_construction2_boundary_ranges(
    source: &DirectCcsFPrimeLowNormSourceImage,
    label: &str,
    fresh_digest_offset: usize,
    commitment_digest_offset: usize,
    commitment_d_offset: usize,
    commitment_kappa_offset: usize,
    x_i_offset: usize,
) -> Result<(), DirectCcsFPrimeSnarkError> {
    validate_source_bit_range(
        source,
        fresh_digest_offset,
        &format!("Construction-2 {label} boundary fresh digest"),
    )?;
    validate_source_bit_range(
        source,
        commitment_digest_offset,
        &format!("Construction-2 {label} boundary commitment digest"),
    )?;
    validate_source_u64_range(
        source,
        commitment_d_offset,
        &format!("Construction-2 {label} boundary commitment d"),
    )?;
    validate_source_u64_range(
        source,
        commitment_kappa_offset,
        &format!("Construction-2 {label} boundary commitment kappa"),
    )?;
    validate_source_bit_range(source, x_i_offset, &format!("Construction-2 {label} boundary x_i"))
}

fn validate_field_lane_ranges(source: &DirectCcsFPrimeLowNormSourceImage) -> Result<(), DirectCcsFPrimeSnarkError> {
    for &offset in source.field_lane_bit_offsets() {
        validate_source_u64_range(source, offset, "canonical field lane")?;
    }
    Ok(())
}

fn validate_nifs_source_ranges(source: &DirectCcsFPrimeLowNormSourceImage) -> Result<(), DirectCcsFPrimeSnarkError> {
    for (offset, label) in [
        (source.nifs_chunk_index_bit_offset(), "NIFS chunk index"),
        (source.nifs_fresh_claims_bit_offset(), "NIFS fresh claim count"),
        (
            source.nifs_incoming_ce_claims_bit_offset(),
            "NIFS incoming CE claim count",
        ),
        (source.nifs_pi_ccs_outputs_bit_offset(), "NIFS Pi_CCS output count"),
        (source.nifs_final_ce_claims_bit_offset(), "NIFS final CE claim count"),
        (source.nifs_fe_sumcheck_rounds_bit_offset(), "NIFS FE sumcheck rounds"),
        (
            source.nifs_fe_sumcheck_messages_bit_offset(),
            "NIFS FE sumcheck messages",
        ),
        (source.nifs_nc_sumcheck_rounds_bit_offset(), "NIFS NC sumcheck rounds"),
        (
            source.nifs_nc_sumcheck_messages_bit_offset(),
            "NIFS NC sumcheck messages",
        ),
        (
            source.nifs_transcript_absorbed_in_bit_offset(),
            "NIFS transcript absorbed input",
        ),
        (
            source.nifs_transcript_absorbed_out_bit_offset(),
            "NIFS transcript absorbed output",
        ),
    ] {
        validate_source_u64_range(source, offset, label)?;
    }
    Ok(())
}

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
