//! Builder mechanics for the low-norm F' source image.

use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::construction2::{Construction2EncodedPublicInput, Construction2PublicBoundary};
use crate::frontends::direct_ccs::f_prime::DirectCcsNativeFPrimeAdvice;

use super::{
    DirectCcsFPrimeConstruction2BoundaryOffsets, DirectCcsFPrimeLowNormSourceImage,
    DirectCcsFPrimeLowNormSourceOffsets, DirectCcsFPrimeLowNormSourceStats,
};
use crate::frontends::direct_ccs::state::DirectCcsFPrimeSnarkError;

pub(super) fn build_low_norm_source_image_from_native_advice(
    advice: &DirectCcsNativeFPrimeAdvice,
) -> Result<DirectCcsFPrimeLowNormSourceImage, DirectCcsFPrimeSnarkError> {
    advice.validate()?;
    let mut builder = DirectCcsFPrimeLowNormSourceBuilder::default();
    let image = advice.compact_image();
    builder.offsets.digests.mat_digest = builder.values.len();
    for value in image.mat_digest {
        builder.push_field_u64(value);
    }
    builder.offsets.digests.vk_fs_digest = builder.push_digest(image.vk_fs_digest);
    builder.offsets.counters.pc = builder.push_u64(image.pc);
    builder.offsets.counters.chunk_count_in = builder.push_u64(image.chunk_count_in);
    builder.offsets.counters.step_count_in = builder.push_u64(image.step_count_in);
    builder.offsets.counters.chunk_count_out = builder.push_u64(image.chunk_count_out);
    builder.offsets.counters.step_count_out = builder.push_u64(image.step_count_out);
    builder.offsets.digests.initial_boundary_digest = builder.push_digest(image.initial_boundary_digest);
    builder.offsets.digests.current_boundary_in_digest = builder.push_digest(image.current_boundary_in_digest);
    builder.offsets.digests.current_boundary_out_digest = builder.push_digest(image.current_boundary_out_digest);
    builder.offsets.digests.public_trace_in_digest = builder.push_digest(image.public_trace_in_digest);
    builder.offsets.digests.public_trace_out_digest = builder.push_digest(image.public_trace_out_digest);
    builder.offsets.digests.semantic_accumulator_in_digest = builder.push_digest(image.semantic_accumulator_in_digest);
    builder.offsets.digests.semantic_accumulator_out_digest =
        builder.push_digest(image.semantic_accumulator_out_digest);
    builder.offsets.digests.f_prime_accumulator_in_digest = builder.push_digest(image.f_prime_accumulator_in_digest);
    builder.offsets.digests.f_prime_accumulator_out_digest = builder.push_digest(image.f_prime_accumulator_out_digest);
    builder.offsets.public_inputs.compact_x_in = builder.push_encoded_public_input(&image.x_in);
    builder.offsets.public_inputs.compact_x_out = builder.push_encoded_public_input(&image.x_out);
    builder.offsets.digests.compact_construction2_u_in_digest = builder.push_digest(image.construction2_u_in_digest);
    builder.offsets.digests.compact_construction2_u_out_digest = builder.push_digest(image.construction2_u_out_digest);
    builder.offsets.digests.latest_chunk_digest = builder.values.len();
    for value in image.latest_chunk_digest {
        builder.push_field_u64(value);
    }
    builder.offsets.digests.latest_fold_digest = builder.push_digest(image.latest_fold_digest);
    builder.offsets.digests.latest_chunk_relation_digest = builder.push_digest(image.latest_chunk_relation_digest);
    builder.offsets.counters.fresh_claims = builder.push_u64(image.fresh_claims);
    builder.offsets.counters.incoming_ce_claims = builder.push_u64(image.incoming_ce_claims);
    builder.offsets.counters.output_ce_claims = builder.push_u64(image.output_ce_claims);
    builder.offsets.counters.final_ce_claims = builder.push_u64(image.final_ce_claims);
    builder.offsets.nifs.chunk_index = builder.push_u64(image.nifs_chunk_index);
    builder.offsets.nifs.fresh_claims = builder.push_u64(image.nifs_fresh_claims);
    builder.offsets.nifs.incoming_ce_claims = builder.push_u64(image.nifs_incoming_ce_claims);
    builder.offsets.nifs.pi_ccs_outputs = builder.push_u64(image.nifs_pi_ccs_outputs);
    builder.offsets.nifs.final_ce_claims = builder.push_u64(image.nifs_final_ce_claims);
    builder.offsets.nifs.fe_sumcheck_rounds = builder.push_u64(image.nifs_fe_sumcheck_rounds);
    builder.offsets.nifs.fe_sumcheck_messages = builder.push_u64(image.nifs_fe_sumcheck_messages);
    builder.offsets.nifs.nc_sumcheck_rounds = builder.push_u64(image.nifs_nc_sumcheck_rounds);
    builder.offsets.nifs.nc_sumcheck_messages = builder.push_u64(image.nifs_nc_sumcheck_messages);
    builder.offsets.nifs.transcript_absorbed_in = builder.push_u64(image.nifs_transcript_absorbed_in);
    builder.offsets.nifs.transcript_absorbed_out = builder.push_u64(image.nifs_transcript_absorbed_out);
    let in_offsets = builder.push_construction2_boundary(advice.construction2_u_in());
    builder.offsets.construction2_u_in = in_offsets;
    let out_offsets = builder.push_construction2_boundary(advice.construction2_u_out());
    builder.offsets.construction2_u_out = out_offsets;
    Ok(builder.finish())
}

#[derive(Default)]
pub(super) struct DirectCcsFPrimeLowNormSourceBuilder {
    values: Vec<F>,
    offsets: DirectCcsFPrimeLowNormSourceOffsets,
    stats: DirectCcsFPrimeLowNormSourceStats,
    field_lane_bit_offsets: Vec<usize>,
}

impl DirectCcsFPrimeLowNormSourceBuilder {
    fn push_digest(&mut self, digest: [u8; 32]) -> usize {
        let start = self.values.len();
        self.stats.digest_count += 1;
        self.push_field_lane_offsets(start, 4);
        for byte in digest {
            for bit_index in 0..8 {
                self.push_bit((byte >> bit_index) & 1);
            }
        }
        start
    }

    fn push_encoded_public_input(&mut self, input: &Construction2EncodedPublicInput) -> usize {
        let start = self.values.len();
        self.stats.encoded_public_input_count += 1;
        self.push_field_lane_offsets(start, 4);
        for bit in input.bit_image() {
            self.push_bit(bit);
        }
        start
    }

    fn push_construction2_boundary(
        &mut self,
        boundary: &Construction2PublicBoundary,
    ) -> DirectCcsFPrimeConstruction2BoundaryOffsets {
        let fresh_offset = self.push_digest(boundary.fresh_instance_digest);
        let commitment_digest_offset = self.push_digest(boundary.commitment_digest);
        let commitment_d_offset = self.values.len();
        self.push_u64(boundary.commitment_d);
        let commitment_kappa_offset = self.values.len();
        self.push_u64(boundary.commitment_kappa);
        let x_i_offset = self.push_encoded_public_input(&boundary.x_i);
        DirectCcsFPrimeConstruction2BoundaryOffsets {
            fresh_digest: fresh_offset,
            commitment_digest: commitment_digest_offset,
            commitment_d: commitment_d_offset,
            commitment_kappa: commitment_kappa_offset,
            x_i: x_i_offset,
        }
    }

    fn push_field_u64(&mut self, value: F) {
        self.field_lane_bit_offsets.push(self.values.len());
        self.stats.field_lane_count += 1;
        self.push_u64(value.as_canonical_u64());
    }

    fn push_u64(&mut self, value: u64) -> usize {
        let start = self.values.len();
        self.stats.u64_count += 1;
        for bit_index in 0..64 {
            self.push_bit(((value >> bit_index) & 1) as u8);
        }
        start
    }

    fn push_bit(&mut self, bit: u8) {
        debug_assert!(bit <= 1);
        self.values.push(F::from_u64(bit as u64));
    }

    fn push_field_lane_offsets(&mut self, start: usize, count: usize) {
        self.stats.field_lane_count += count;
        self.field_lane_bit_offsets
            .extend((0..count).map(|idx| start + idx * 64));
    }

    fn finish(self) -> DirectCcsFPrimeLowNormSourceImage {
        DirectCcsFPrimeLowNormSourceImage {
            values: self.values,
            offsets: self.offsets,
            field_lane_bit_offsets: self.field_lane_bit_offsets,
            stats: self.stats,
        }
    }
}
