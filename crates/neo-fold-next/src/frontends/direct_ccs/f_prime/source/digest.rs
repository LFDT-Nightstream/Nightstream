//! Transcript digest for the low-norm F' source image.

use neo_transcript::{Poseidon2Transcript, Transcript};

use super::DirectCcsFPrimeLowNormSourceImage;

impl DirectCcsFPrimeLowNormSourceImage {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/direct_ccs/f_prime_low_norm_source");
        tr.append_message(b"neo.fold.next/direct_ccs/f_prime_low_norm_source/version", b"v1");
        tr.append_u64s(
            b"neo.fold.next/direct_ccs/f_prime_low_norm_source/counts",
            &[
                self.values.len() as u64,
                self.mat_digest_bit_offset as u64,
                self.vk_fs_digest_bit_offset as u64,
                self.pc_bit_offset as u64,
                self.chunk_count_in_bit_offset as u64,
                self.step_count_in_bit_offset as u64,
                self.chunk_count_out_bit_offset as u64,
                self.step_count_out_bit_offset as u64,
                self.initial_boundary_digest_bit_offset as u64,
                self.current_boundary_in_digest_bit_offset as u64,
                self.current_boundary_out_digest_bit_offset as u64,
                self.public_trace_in_digest_bit_offset as u64,
                self.public_trace_out_digest_bit_offset as u64,
                self.semantic_accumulator_in_digest_bit_offset as u64,
                self.semantic_accumulator_out_digest_bit_offset as u64,
                self.f_prime_accumulator_in_digest_bit_offset as u64,
                self.f_prime_accumulator_out_digest_bit_offset as u64,
                self.compact_x_in_bit_offset as u64,
                self.compact_x_out_bit_offset as u64,
                self.compact_construction2_u_in_digest_bit_offset as u64,
                self.compact_construction2_u_out_digest_bit_offset as u64,
                self.latest_chunk_digest_bit_offset as u64,
                self.latest_fold_digest_bit_offset as u64,
                self.latest_chunk_relation_digest_bit_offset as u64,
                self.fresh_claims_bit_offset as u64,
                self.incoming_ce_claims_bit_offset as u64,
                self.output_ce_claims_bit_offset as u64,
                self.final_ce_claims_bit_offset as u64,
                self.nifs_chunk_index_bit_offset as u64,
                self.nifs_fresh_claims_bit_offset as u64,
                self.nifs_incoming_ce_claims_bit_offset as u64,
                self.nifs_pi_ccs_outputs_bit_offset as u64,
                self.nifs_final_ce_claims_bit_offset as u64,
                self.nifs_fe_sumcheck_rounds_bit_offset as u64,
                self.nifs_fe_sumcheck_messages_bit_offset as u64,
                self.nifs_nc_sumcheck_rounds_bit_offset as u64,
                self.nifs_nc_sumcheck_messages_bit_offset as u64,
                self.nifs_transcript_absorbed_in_bit_offset as u64,
                self.nifs_transcript_absorbed_out_bit_offset as u64,
                self.construction2_u_in_fresh_digest_bit_offset as u64,
                self.construction2_u_in_commitment_digest_bit_offset as u64,
                self.construction2_u_in_commitment_d_bit_offset as u64,
                self.construction2_u_in_commitment_kappa_bit_offset as u64,
                self.construction2_u_in_x_i_bit_offset as u64,
                self.construction2_u_out_fresh_digest_bit_offset as u64,
                self.construction2_u_out_commitment_digest_bit_offset as u64,
                self.construction2_u_out_commitment_d_bit_offset as u64,
                self.construction2_u_out_commitment_kappa_bit_offset as u64,
                self.construction2_u_out_x_i_bit_offset as u64,
                self.digest_count as u64,
                self.u64_count as u64,
                self.encoded_public_input_count as u64,
                self.field_lane_count as u64,
                self.construction2_commitment_fields as u64,
            ],
        );
        tr.append_u64s(
            b"neo.fold.next/direct_ccs/f_prime_low_norm_source/field_lane_offsets",
            &self
                .field_lane_bit_offsets
                .iter()
                .copied()
                .map(|offset| offset as u64)
                .collect::<Vec<_>>(),
        );
        tr.append_fields(b"neo.fold.next/direct_ccs/f_prime_low_norm_source/values", &self.values);
        tr.digest32()
    }
}
