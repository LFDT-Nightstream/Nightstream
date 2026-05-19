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
        self.offsets.public_inputs.compact_x_in
    }

    pub fn mat_digest_bit_offset(&self) -> usize {
        self.offsets.digests.mat_digest
    }

    pub fn vk_fs_digest_bit_offset(&self) -> usize {
        self.offsets.digests.vk_fs_digest
    }

    pub fn pc_bit_offset(&self) -> usize {
        self.offsets.counters.pc
    }

    pub fn chunk_count_in_bit_offset(&self) -> usize {
        self.offsets.counters.chunk_count_in
    }

    pub fn step_count_in_bit_offset(&self) -> usize {
        self.offsets.counters.step_count_in
    }

    pub fn chunk_count_out_bit_offset(&self) -> usize {
        self.offsets.counters.chunk_count_out
    }

    pub fn step_count_out_bit_offset(&self) -> usize {
        self.offsets.counters.step_count_out
    }

    pub fn initial_boundary_digest_bit_offset(&self) -> usize {
        self.offsets.digests.initial_boundary_digest
    }

    pub fn current_boundary_in_digest_bit_offset(&self) -> usize {
        self.offsets.digests.current_boundary_in_digest
    }

    pub fn current_boundary_out_digest_bit_offset(&self) -> usize {
        self.offsets.digests.current_boundary_out_digest
    }

    pub fn public_trace_in_digest_bit_offset(&self) -> usize {
        self.offsets.digests.public_trace_in_digest
    }

    pub fn public_trace_out_digest_bit_offset(&self) -> usize {
        self.offsets.digests.public_trace_out_digest
    }

    pub fn semantic_accumulator_in_digest_bit_offset(&self) -> usize {
        self.offsets.digests.semantic_accumulator_in_digest
    }

    pub fn semantic_accumulator_out_digest_bit_offset(&self) -> usize {
        self.offsets.digests.semantic_accumulator_out_digest
    }

    pub fn f_prime_accumulator_in_digest_bit_offset(&self) -> usize {
        self.offsets.digests.f_prime_accumulator_in_digest
    }

    pub fn f_prime_accumulator_out_digest_bit_offset(&self) -> usize {
        self.offsets.digests.f_prime_accumulator_out_digest
    }

    pub fn compact_x_out_bit_offset(&self) -> usize {
        self.offsets.public_inputs.compact_x_out
    }

    pub fn compact_construction2_u_in_digest_bit_offset(&self) -> usize {
        self.offsets.digests.compact_construction2_u_in_digest
    }

    pub fn compact_construction2_u_out_digest_bit_offset(&self) -> usize {
        self.offsets.digests.compact_construction2_u_out_digest
    }

    pub fn latest_chunk_digest_bit_offset(&self) -> usize {
        self.offsets.digests.latest_chunk_digest
    }

    pub fn latest_fold_digest_bit_offset(&self) -> usize {
        self.offsets.digests.latest_fold_digest
    }

    pub fn latest_chunk_relation_digest_bit_offset(&self) -> usize {
        self.offsets.digests.latest_chunk_relation_digest
    }

    pub fn fresh_claims_bit_offset(&self) -> usize {
        self.offsets.counters.fresh_claims
    }

    pub fn incoming_ce_claims_bit_offset(&self) -> usize {
        self.offsets.counters.incoming_ce_claims
    }

    pub fn output_ce_claims_bit_offset(&self) -> usize {
        self.offsets.counters.output_ce_claims
    }

    pub fn final_ce_claims_bit_offset(&self) -> usize {
        self.offsets.counters.final_ce_claims
    }

    pub fn nifs_chunk_index_bit_offset(&self) -> usize {
        self.offsets.nifs.chunk_index
    }

    pub fn nifs_fresh_claims_bit_offset(&self) -> usize {
        self.offsets.nifs.fresh_claims
    }

    pub fn nifs_incoming_ce_claims_bit_offset(&self) -> usize {
        self.offsets.nifs.incoming_ce_claims
    }

    pub fn nifs_pi_ccs_outputs_bit_offset(&self) -> usize {
        self.offsets.nifs.pi_ccs_outputs
    }

    pub fn nifs_final_ce_claims_bit_offset(&self) -> usize {
        self.offsets.nifs.final_ce_claims
    }

    pub fn nifs_fe_sumcheck_rounds_bit_offset(&self) -> usize {
        self.offsets.nifs.fe_sumcheck_rounds
    }

    pub fn nifs_fe_sumcheck_messages_bit_offset(&self) -> usize {
        self.offsets.nifs.fe_sumcheck_messages
    }

    pub fn nifs_nc_sumcheck_rounds_bit_offset(&self) -> usize {
        self.offsets.nifs.nc_sumcheck_rounds
    }

    pub fn nifs_nc_sumcheck_messages_bit_offset(&self) -> usize {
        self.offsets.nifs.nc_sumcheck_messages
    }

    pub fn nifs_transcript_absorbed_in_bit_offset(&self) -> usize {
        self.offsets.nifs.transcript_absorbed_in
    }

    pub fn nifs_transcript_absorbed_out_bit_offset(&self) -> usize {
        self.offsets.nifs.transcript_absorbed_out
    }

    pub fn construction2_u_in_fresh_digest_bit_offset(&self) -> usize {
        self.offsets.construction2_u_in.fresh_digest
    }

    pub fn construction2_u_in_commitment_digest_bit_offset(&self) -> usize {
        self.offsets.construction2_u_in.commitment_digest
    }

    pub fn construction2_u_in_commitment_d_bit_offset(&self) -> usize {
        self.offsets.construction2_u_in.commitment_d
    }

    pub fn construction2_u_in_commitment_kappa_bit_offset(&self) -> usize {
        self.offsets.construction2_u_in.commitment_kappa
    }

    pub fn construction2_u_in_x_i_bit_offset(&self) -> usize {
        self.offsets.construction2_u_in.x_i
    }

    pub fn construction2_u_out_fresh_digest_bit_offset(&self) -> usize {
        self.offsets.construction2_u_out.fresh_digest
    }

    pub fn construction2_u_out_commitment_digest_bit_offset(&self) -> usize {
        self.offsets.construction2_u_out.commitment_digest
    }

    pub fn construction2_u_out_commitment_d_bit_offset(&self) -> usize {
        self.offsets.construction2_u_out.commitment_d
    }

    pub fn construction2_u_out_commitment_kappa_bit_offset(&self) -> usize {
        self.offsets.construction2_u_out.commitment_kappa
    }

    pub fn construction2_u_out_x_i_bit_offset(&self) -> usize {
        self.offsets.construction2_u_out.x_i
    }

    pub fn digest_count(&self) -> usize {
        self.stats.digest_count
    }

    pub fn u64_count(&self) -> usize {
        self.stats.u64_count
    }

    pub fn encoded_public_input_count(&self) -> usize {
        self.stats.encoded_public_input_count
    }

    pub fn construction2_commitment_fields(&self) -> usize {
        self.stats.construction2_commitment_fields
    }

    pub fn field_lane_bit_offsets(&self) -> &[usize] {
        &self.field_lane_bit_offsets
    }

    pub fn field_lane_count(&self) -> usize {
        self.stats.field_lane_count
    }
}
