use super::super::DirectCcsCompactFPrimeImage;
use super::*;

pub(super) const NIFS_AUTHORITY_U64_FIELDS: usize = 11;
pub(super) const NIFS_AUTHORITY_ROWS: usize = NIFS_AUTHORITY_U64_FIELDS * U64_BITS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct DirectCcsFPrimeNifsAuthoritySpec {
    chunk_index: u64,
    fresh_claims: u64,
    incoming_ce_claims: u64,
    pi_ccs_outputs: u64,
    final_ce_claims: u64,
    fe_sumcheck_rounds: u64,
    fe_sumcheck_messages: u64,
    nc_sumcheck_rounds: u64,
    nc_sumcheck_messages: u64,
    transcript_absorbed_in: u64,
    transcript_absorbed_out: u64,
}

impl DirectCcsFPrimeNifsAuthoritySpec {
    pub(super) fn from_compact_image(image: &DirectCcsCompactFPrimeImage) -> Result<Self, DirectCcsFPrimeSnarkError> {
        image.validate()?;
        if image.nifs_fe_sumcheck_rounds == 0
            || image.nifs_fe_sumcheck_messages == 0
            || image.nifs_nc_sumcheck_rounds == 0
            || image.nifs_nc_sumcheck_messages == 0
            || image.nifs_transcript_absorbed_out < image.nifs_transcript_absorbed_in
        {
            return Err(DirectCcsFPrimeSnarkError::Input(
                "direct F' NIFS authority requires non-empty verifier transcript material".into(),
            ));
        }
        Ok(Self {
            chunk_index: image.nifs_chunk_index,
            fresh_claims: image.nifs_fresh_claims,
            incoming_ce_claims: image.nifs_incoming_ce_claims,
            pi_ccs_outputs: image.nifs_pi_ccs_outputs,
            final_ce_claims: image.nifs_final_ce_claims,
            fe_sumcheck_rounds: image.nifs_fe_sumcheck_rounds,
            fe_sumcheck_messages: image.nifs_fe_sumcheck_messages,
            nc_sumcheck_rounds: image.nifs_nc_sumcheck_rounds,
            nc_sumcheck_messages: image.nifs_nc_sumcheck_messages,
            transcript_absorbed_in: image.nifs_transcript_absorbed_in,
            transcript_absorbed_out: image.nifs_transcript_absorbed_out,
        })
    }

    fn fields(self, source: &DirectCcsFPrimeLowNormSourceImage) -> [(usize, u64); NIFS_AUTHORITY_U64_FIELDS] {
        [
            (source.nifs_chunk_index_bit_offset(), self.chunk_index),
            (source.nifs_fresh_claims_bit_offset(), self.fresh_claims),
            (source.nifs_incoming_ce_claims_bit_offset(), self.incoming_ce_claims),
            (source.nifs_pi_ccs_outputs_bit_offset(), self.pi_ccs_outputs),
            (source.nifs_final_ce_claims_bit_offset(), self.final_ce_claims),
            (source.nifs_fe_sumcheck_rounds_bit_offset(), self.fe_sumcheck_rounds),
            (source.nifs_fe_sumcheck_messages_bit_offset(), self.fe_sumcheck_messages),
            (source.nifs_nc_sumcheck_rounds_bit_offset(), self.nc_sumcheck_rounds),
            (source.nifs_nc_sumcheck_messages_bit_offset(), self.nc_sumcheck_messages),
            (
                source.nifs_transcript_absorbed_in_bit_offset(),
                self.transcript_absorbed_in,
            ),
            (
                source.nifs_transcript_absorbed_out_bit_offset(),
                self.transcript_absorbed_out,
            ),
        ]
    }
}

pub(super) fn add_nifs_authority_constraints(
    a_trips: &mut Vec<(usize, usize, F)>,
    b_trips: &mut Vec<(usize, usize, F)>,
    row: &mut usize,
    source_start_col: usize,
    source: &DirectCcsFPrimeLowNormSourceImage,
    spec: DirectCcsFPrimeNifsAuthoritySpec,
) {
    for (offset, expected) in spec.fields(source) {
        add_source_u64_constant_constraints(a_trips, b_trips, row, source_start_col, offset, expected);
    }
}
