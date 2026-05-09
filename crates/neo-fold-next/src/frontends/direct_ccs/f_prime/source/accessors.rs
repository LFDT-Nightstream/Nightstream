//! Read-only accessors for the low-norm F' source layout.

use super::DirectCcsFPrimeLowNormSourceImage;
use neo_math::F;

impl DirectCcsFPrimeLowNormSourceImage {
    pub fn values(&self) -> &[F] {
        &self.values
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn compact_x_in_bit_offset(&self) -> usize {
        self.compact_x_in_bit_offset
    }

    pub fn mat_digest_bit_offset(&self) -> usize {
        self.mat_digest_bit_offset
    }

    pub fn vk_fs_digest_bit_offset(&self) -> usize {
        self.vk_fs_digest_bit_offset
    }

    pub fn pc_bit_offset(&self) -> usize {
        self.pc_bit_offset
    }

    pub fn chunk_count_in_bit_offset(&self) -> usize {
        self.chunk_count_in_bit_offset
    }

    pub fn step_count_in_bit_offset(&self) -> usize {
        self.step_count_in_bit_offset
    }

    pub fn chunk_count_out_bit_offset(&self) -> usize {
        self.chunk_count_out_bit_offset
    }

    pub fn step_count_out_bit_offset(&self) -> usize {
        self.step_count_out_bit_offset
    }

    pub fn initial_boundary_digest_bit_offset(&self) -> usize {
        self.initial_boundary_digest_bit_offset
    }

    pub fn current_boundary_in_digest_bit_offset(&self) -> usize {
        self.current_boundary_in_digest_bit_offset
    }

    pub fn current_boundary_out_digest_bit_offset(&self) -> usize {
        self.current_boundary_out_digest_bit_offset
    }

    pub fn public_trace_in_digest_bit_offset(&self) -> usize {
        self.public_trace_in_digest_bit_offset
    }

    pub fn public_trace_out_digest_bit_offset(&self) -> usize {
        self.public_trace_out_digest_bit_offset
    }

    pub fn semantic_accumulator_in_digest_bit_offset(&self) -> usize {
        self.semantic_accumulator_in_digest_bit_offset
    }

    pub fn semantic_accumulator_out_digest_bit_offset(&self) -> usize {
        self.semantic_accumulator_out_digest_bit_offset
    }

    pub fn f_prime_accumulator_in_digest_bit_offset(&self) -> usize {
        self.f_prime_accumulator_in_digest_bit_offset
    }

    pub fn f_prime_accumulator_out_digest_bit_offset(&self) -> usize {
        self.f_prime_accumulator_out_digest_bit_offset
    }

    pub fn compact_x_out_bit_offset(&self) -> usize {
        self.compact_x_out_bit_offset
    }

    pub fn compact_construction2_u_in_digest_bit_offset(&self) -> usize {
        self.compact_construction2_u_in_digest_bit_offset
    }

    pub fn compact_construction2_u_out_digest_bit_offset(&self) -> usize {
        self.compact_construction2_u_out_digest_bit_offset
    }

    pub fn latest_chunk_digest_bit_offset(&self) -> usize {
        self.latest_chunk_digest_bit_offset
    }

    pub fn latest_fold_digest_bit_offset(&self) -> usize {
        self.latest_fold_digest_bit_offset
    }

    pub fn latest_chunk_relation_digest_bit_offset(&self) -> usize {
        self.latest_chunk_relation_digest_bit_offset
    }

    pub fn fresh_claims_bit_offset(&self) -> usize {
        self.fresh_claims_bit_offset
    }

    pub fn incoming_ce_claims_bit_offset(&self) -> usize {
        self.incoming_ce_claims_bit_offset
    }

    pub fn output_ce_claims_bit_offset(&self) -> usize {
        self.output_ce_claims_bit_offset
    }

    pub fn final_ce_claims_bit_offset(&self) -> usize {
        self.final_ce_claims_bit_offset
    }

    pub fn nifs_chunk_index_bit_offset(&self) -> usize {
        self.nifs_chunk_index_bit_offset
    }

    pub fn nifs_fresh_claims_bit_offset(&self) -> usize {
        self.nifs_fresh_claims_bit_offset
    }

    pub fn nifs_incoming_ce_claims_bit_offset(&self) -> usize {
        self.nifs_incoming_ce_claims_bit_offset
    }

    pub fn nifs_pi_ccs_outputs_bit_offset(&self) -> usize {
        self.nifs_pi_ccs_outputs_bit_offset
    }

    pub fn nifs_final_ce_claims_bit_offset(&self) -> usize {
        self.nifs_final_ce_claims_bit_offset
    }

    pub fn nifs_fe_sumcheck_rounds_bit_offset(&self) -> usize {
        self.nifs_fe_sumcheck_rounds_bit_offset
    }

    pub fn nifs_fe_sumcheck_messages_bit_offset(&self) -> usize {
        self.nifs_fe_sumcheck_messages_bit_offset
    }

    pub fn nifs_nc_sumcheck_rounds_bit_offset(&self) -> usize {
        self.nifs_nc_sumcheck_rounds_bit_offset
    }

    pub fn nifs_nc_sumcheck_messages_bit_offset(&self) -> usize {
        self.nifs_nc_sumcheck_messages_bit_offset
    }

    pub fn nifs_transcript_absorbed_in_bit_offset(&self) -> usize {
        self.nifs_transcript_absorbed_in_bit_offset
    }

    pub fn nifs_transcript_absorbed_out_bit_offset(&self) -> usize {
        self.nifs_transcript_absorbed_out_bit_offset
    }

    pub fn construction2_u_in_fresh_digest_bit_offset(&self) -> usize {
        self.construction2_u_in_fresh_digest_bit_offset
    }

    pub fn construction2_u_in_commitment_digest_bit_offset(&self) -> usize {
        self.construction2_u_in_commitment_digest_bit_offset
    }

    pub fn construction2_u_in_commitment_d_bit_offset(&self) -> usize {
        self.construction2_u_in_commitment_d_bit_offset
    }

    pub fn construction2_u_in_commitment_kappa_bit_offset(&self) -> usize {
        self.construction2_u_in_commitment_kappa_bit_offset
    }

    pub fn construction2_u_in_x_i_bit_offset(&self) -> usize {
        self.construction2_u_in_x_i_bit_offset
    }

    pub fn construction2_u_out_fresh_digest_bit_offset(&self) -> usize {
        self.construction2_u_out_fresh_digest_bit_offset
    }

    pub fn construction2_u_out_commitment_digest_bit_offset(&self) -> usize {
        self.construction2_u_out_commitment_digest_bit_offset
    }

    pub fn construction2_u_out_commitment_d_bit_offset(&self) -> usize {
        self.construction2_u_out_commitment_d_bit_offset
    }

    pub fn construction2_u_out_commitment_kappa_bit_offset(&self) -> usize {
        self.construction2_u_out_commitment_kappa_bit_offset
    }

    pub fn construction2_u_out_x_i_bit_offset(&self) -> usize {
        self.construction2_u_out_x_i_bit_offset
    }

    pub fn digest_count(&self) -> usize {
        self.digest_count
    }

    pub fn u64_count(&self) -> usize {
        self.u64_count
    }

    pub fn encoded_public_input_count(&self) -> usize {
        self.encoded_public_input_count
    }

    pub fn construction2_commitment_fields(&self) -> usize {
        self.construction2_commitment_fields
    }

    pub fn field_lane_bit_offsets(&self) -> &[usize] {
        &self.field_lane_bit_offsets
    }

    pub fn field_lane_count(&self) -> usize {
        self.field_lane_count
    }
}
