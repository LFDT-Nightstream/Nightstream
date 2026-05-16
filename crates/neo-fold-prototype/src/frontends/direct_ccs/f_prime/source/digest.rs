//! Transcript digest for the low-norm F' source image.

use neo_transcript::{Poseidon2Transcript, Transcript};

use super::DirectCcsFPrimeLowNormSourceImage;

impl DirectCcsFPrimeLowNormSourceImage {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/direct_ccs/f_prime_low_norm_source");
        tr.append_message(b"neo.fold.next/direct_ccs/f_prime_low_norm_source/version", b"v1");
        let digests = &self.offsets.digests;
        let counters = &self.offsets.counters;
        let public_inputs = &self.offsets.public_inputs;
        let nifs = &self.offsets.nifs;
        let construction2_u_in = &self.offsets.construction2_u_in;
        let construction2_u_out = &self.offsets.construction2_u_out;
        tr.append_u64s(
            b"neo.fold.next/direct_ccs/f_prime_low_norm_source/counts",
            &[
                self.values.len() as u64,
                digests.mat_digest as u64,
                digests.vk_fs_digest as u64,
                counters.pc as u64,
                counters.chunk_count_in as u64,
                counters.step_count_in as u64,
                counters.chunk_count_out as u64,
                counters.step_count_out as u64,
                digests.initial_boundary_digest as u64,
                digests.current_boundary_in_digest as u64,
                digests.current_boundary_out_digest as u64,
                digests.public_trace_in_digest as u64,
                digests.public_trace_out_digest as u64,
                digests.semantic_accumulator_in_digest as u64,
                digests.semantic_accumulator_out_digest as u64,
                digests.f_prime_accumulator_in_digest as u64,
                digests.f_prime_accumulator_out_digest as u64,
                public_inputs.compact_x_in as u64,
                public_inputs.compact_x_out as u64,
                digests.compact_construction2_u_in_digest as u64,
                digests.compact_construction2_u_out_digest as u64,
                digests.latest_chunk_digest as u64,
                digests.latest_fold_digest as u64,
                digests.latest_chunk_relation_digest as u64,
                counters.fresh_claims as u64,
                counters.incoming_ce_claims as u64,
                counters.output_ce_claims as u64,
                counters.final_ce_claims as u64,
                nifs.chunk_index as u64,
                nifs.fresh_claims as u64,
                nifs.incoming_ce_claims as u64,
                nifs.pi_ccs_outputs as u64,
                nifs.final_ce_claims as u64,
                nifs.fe_sumcheck_rounds as u64,
                nifs.fe_sumcheck_messages as u64,
                nifs.nc_sumcheck_rounds as u64,
                nifs.nc_sumcheck_messages as u64,
                nifs.transcript_absorbed_in as u64,
                nifs.transcript_absorbed_out as u64,
                construction2_u_in.fresh_digest as u64,
                construction2_u_in.commitment_digest as u64,
                construction2_u_in.commitment_d as u64,
                construction2_u_in.commitment_kappa as u64,
                construction2_u_in.x_i as u64,
                construction2_u_out.fresh_digest as u64,
                construction2_u_out.commitment_digest as u64,
                construction2_u_out.commitment_d as u64,
                construction2_u_out.commitment_kappa as u64,
                construction2_u_out.x_i as u64,
                self.stats.digest_count as u64,
                self.stats.u64_count as u64,
                self.stats.encoded_public_input_count as u64,
                self.stats.field_lane_count as u64,
                self.stats.construction2_commitment_fields as u64,
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
