use neo_math::F;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde::{Deserialize, Serialize};

use super::super::ivc::DirectCcsFPrimeSnarkError;
use super::advice::DirectCcsNativeFPrimeAdvice;
use crate::construction2::{Construction2EncodedPublicInput, Construction2PublicBoundary};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectCcsFPrimeLowNormSourceImage {
    values: Vec<F>,
    mat_digest_bit_offset: usize,
    vk_fs_digest_bit_offset: usize,
    pc_bit_offset: usize,
    chunk_count_in_bit_offset: usize,
    step_count_in_bit_offset: usize,
    chunk_count_out_bit_offset: usize,
    step_count_out_bit_offset: usize,
    initial_boundary_digest_bit_offset: usize,
    current_boundary_in_digest_bit_offset: usize,
    current_boundary_out_digest_bit_offset: usize,
    public_trace_in_digest_bit_offset: usize,
    public_trace_out_digest_bit_offset: usize,
    semantic_accumulator_in_digest_bit_offset: usize,
    semantic_accumulator_out_digest_bit_offset: usize,
    f_prime_accumulator_in_digest_bit_offset: usize,
    f_prime_accumulator_out_digest_bit_offset: usize,
    compact_x_in_bit_offset: usize,
    compact_x_out_bit_offset: usize,
    compact_construction2_u_in_digest_bit_offset: usize,
    compact_construction2_u_out_digest_bit_offset: usize,
    latest_chunk_digest_bit_offset: usize,
    latest_fold_digest_bit_offset: usize,
    latest_chunk_relation_digest_bit_offset: usize,
    fresh_claims_bit_offset: usize,
    incoming_ce_claims_bit_offset: usize,
    output_ce_claims_bit_offset: usize,
    final_ce_claims_bit_offset: usize,
    nifs_chunk_index_bit_offset: usize,
    nifs_fresh_claims_bit_offset: usize,
    nifs_incoming_ce_claims_bit_offset: usize,
    nifs_pi_ccs_outputs_bit_offset: usize,
    nifs_final_ce_claims_bit_offset: usize,
    nifs_fe_sumcheck_rounds_bit_offset: usize,
    nifs_fe_sumcheck_messages_bit_offset: usize,
    nifs_nc_sumcheck_rounds_bit_offset: usize,
    nifs_nc_sumcheck_messages_bit_offset: usize,
    nifs_transcript_absorbed_in_bit_offset: usize,
    nifs_transcript_absorbed_out_bit_offset: usize,
    construction2_u_in_fresh_digest_bit_offset: usize,
    construction2_u_in_commitment_digest_bit_offset: usize,
    construction2_u_in_commitment_d_bit_offset: usize,
    construction2_u_in_commitment_kappa_bit_offset: usize,
    construction2_u_in_x_i_bit_offset: usize,
    construction2_u_out_fresh_digest_bit_offset: usize,
    construction2_u_out_commitment_digest_bit_offset: usize,
    construction2_u_out_commitment_d_bit_offset: usize,
    construction2_u_out_commitment_kappa_bit_offset: usize,
    construction2_u_out_x_i_bit_offset: usize,
    field_lane_bit_offsets: Vec<usize>,
    digest_count: usize,
    u64_count: usize,
    encoded_public_input_count: usize,
    field_lane_count: usize,
    construction2_commitment_fields: usize,
}

impl DirectCcsFPrimeLowNormSourceImage {
    pub fn from_native_advice(advice: &DirectCcsNativeFPrimeAdvice) -> Result<Self, DirectCcsFPrimeSnarkError> {
        advice.validate()?;
        let mut builder = DirectCcsFPrimeLowNormSourceBuilder::default();
        let image = advice.compact_image();
        builder.mat_digest_bit_offset = builder.values.len();
        for value in image.mat_digest {
            builder.push_field_u64(value);
        }
        builder.vk_fs_digest_bit_offset = builder.push_digest(image.vk_fs_digest);
        builder.pc_bit_offset = builder.push_u64(image.pc);
        builder.chunk_count_in_bit_offset = builder.push_u64(image.chunk_count_in);
        builder.step_count_in_bit_offset = builder.push_u64(image.step_count_in);
        builder.chunk_count_out_bit_offset = builder.push_u64(image.chunk_count_out);
        builder.step_count_out_bit_offset = builder.push_u64(image.step_count_out);
        builder.initial_boundary_digest_bit_offset = builder.push_digest(image.initial_boundary_digest);
        builder.current_boundary_in_digest_bit_offset = builder.push_digest(image.current_boundary_in_digest);
        builder.current_boundary_out_digest_bit_offset = builder.push_digest(image.current_boundary_out_digest);
        builder.public_trace_in_digest_bit_offset = builder.push_digest(image.public_trace_in_digest);
        builder.public_trace_out_digest_bit_offset = builder.push_digest(image.public_trace_out_digest);
        builder.semantic_accumulator_in_digest_bit_offset = builder.push_digest(image.semantic_accumulator_in_digest);
        builder.semantic_accumulator_out_digest_bit_offset = builder.push_digest(image.semantic_accumulator_out_digest);
        builder.f_prime_accumulator_in_digest_bit_offset = builder.push_digest(image.f_prime_accumulator_in_digest);
        builder.f_prime_accumulator_out_digest_bit_offset = builder.push_digest(image.f_prime_accumulator_out_digest);
        builder.compact_x_in_bit_offset = builder.push_encoded_public_input(&image.x_in);
        builder.compact_x_out_bit_offset = builder.push_encoded_public_input(&image.x_out);
        builder.compact_construction2_u_in_digest_bit_offset = builder.push_digest(image.construction2_u_in_digest);
        builder.compact_construction2_u_out_digest_bit_offset = builder.push_digest(image.construction2_u_out_digest);
        builder.latest_chunk_digest_bit_offset = builder.values.len();
        for value in image.latest_chunk_digest {
            builder.push_field_u64(value);
        }
        builder.latest_fold_digest_bit_offset = builder.push_digest(image.latest_fold_digest);
        builder.latest_chunk_relation_digest_bit_offset = builder.push_digest(image.latest_chunk_relation_digest);
        builder.fresh_claims_bit_offset = builder.push_u64(image.fresh_claims);
        builder.incoming_ce_claims_bit_offset = builder.push_u64(image.incoming_ce_claims);
        builder.output_ce_claims_bit_offset = builder.push_u64(image.output_ce_claims);
        builder.final_ce_claims_bit_offset = builder.push_u64(image.final_ce_claims);
        builder.nifs_chunk_index_bit_offset = builder.push_u64(image.nifs_chunk_index);
        builder.nifs_fresh_claims_bit_offset = builder.push_u64(image.nifs_fresh_claims);
        builder.nifs_incoming_ce_claims_bit_offset = builder.push_u64(image.nifs_incoming_ce_claims);
        builder.nifs_pi_ccs_outputs_bit_offset = builder.push_u64(image.nifs_pi_ccs_outputs);
        builder.nifs_final_ce_claims_bit_offset = builder.push_u64(image.nifs_final_ce_claims);
        builder.nifs_fe_sumcheck_rounds_bit_offset = builder.push_u64(image.nifs_fe_sumcheck_rounds);
        builder.nifs_fe_sumcheck_messages_bit_offset = builder.push_u64(image.nifs_fe_sumcheck_messages);
        builder.nifs_nc_sumcheck_rounds_bit_offset = builder.push_u64(image.nifs_nc_sumcheck_rounds);
        builder.nifs_nc_sumcheck_messages_bit_offset = builder.push_u64(image.nifs_nc_sumcheck_messages);
        builder.nifs_transcript_absorbed_in_bit_offset = builder.push_u64(image.nifs_transcript_absorbed_in);
        builder.nifs_transcript_absorbed_out_bit_offset = builder.push_u64(image.nifs_transcript_absorbed_out);
        let in_offsets = builder.push_construction2_boundary(advice.construction2_u_in());
        builder.construction2_u_in_fresh_digest_bit_offset = in_offsets.fresh_digest;
        builder.construction2_u_in_commitment_digest_bit_offset = in_offsets.commitment_digest;
        builder.construction2_u_in_commitment_d_bit_offset = in_offsets.commitment_d;
        builder.construction2_u_in_commitment_kappa_bit_offset = in_offsets.commitment_kappa;
        builder.construction2_u_in_x_i_bit_offset = in_offsets.x_i;
        let out_offsets = builder.push_construction2_boundary(advice.construction2_u_out());
        builder.construction2_u_out_fresh_digest_bit_offset = out_offsets.fresh_digest;
        builder.construction2_u_out_commitment_digest_bit_offset = out_offsets.commitment_digest;
        builder.construction2_u_out_commitment_d_bit_offset = out_offsets.commitment_d;
        builder.construction2_u_out_commitment_kappa_bit_offset = out_offsets.commitment_kappa;
        builder.construction2_u_out_x_i_bit_offset = out_offsets.x_i;
        Ok(builder.finish())
    }

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

#[derive(Default)]
struct DirectCcsFPrimeLowNormSourceBuilder {
    values: Vec<F>,
    mat_digest_bit_offset: usize,
    vk_fs_digest_bit_offset: usize,
    pc_bit_offset: usize,
    chunk_count_in_bit_offset: usize,
    step_count_in_bit_offset: usize,
    chunk_count_out_bit_offset: usize,
    step_count_out_bit_offset: usize,
    initial_boundary_digest_bit_offset: usize,
    current_boundary_in_digest_bit_offset: usize,
    current_boundary_out_digest_bit_offset: usize,
    public_trace_in_digest_bit_offset: usize,
    public_trace_out_digest_bit_offset: usize,
    semantic_accumulator_in_digest_bit_offset: usize,
    semantic_accumulator_out_digest_bit_offset: usize,
    f_prime_accumulator_in_digest_bit_offset: usize,
    f_prime_accumulator_out_digest_bit_offset: usize,
    compact_x_in_bit_offset: usize,
    compact_x_out_bit_offset: usize,
    compact_construction2_u_in_digest_bit_offset: usize,
    compact_construction2_u_out_digest_bit_offset: usize,
    latest_chunk_digest_bit_offset: usize,
    latest_fold_digest_bit_offset: usize,
    latest_chunk_relation_digest_bit_offset: usize,
    fresh_claims_bit_offset: usize,
    incoming_ce_claims_bit_offset: usize,
    output_ce_claims_bit_offset: usize,
    final_ce_claims_bit_offset: usize,
    nifs_chunk_index_bit_offset: usize,
    nifs_fresh_claims_bit_offset: usize,
    nifs_incoming_ce_claims_bit_offset: usize,
    nifs_pi_ccs_outputs_bit_offset: usize,
    nifs_final_ce_claims_bit_offset: usize,
    nifs_fe_sumcheck_rounds_bit_offset: usize,
    nifs_fe_sumcheck_messages_bit_offset: usize,
    nifs_nc_sumcheck_rounds_bit_offset: usize,
    nifs_nc_sumcheck_messages_bit_offset: usize,
    nifs_transcript_absorbed_in_bit_offset: usize,
    nifs_transcript_absorbed_out_bit_offset: usize,
    construction2_u_in_fresh_digest_bit_offset: usize,
    construction2_u_in_commitment_digest_bit_offset: usize,
    construction2_u_in_commitment_d_bit_offset: usize,
    construction2_u_in_commitment_kappa_bit_offset: usize,
    construction2_u_in_x_i_bit_offset: usize,
    construction2_u_out_fresh_digest_bit_offset: usize,
    construction2_u_out_commitment_digest_bit_offset: usize,
    construction2_u_out_commitment_d_bit_offset: usize,
    construction2_u_out_commitment_kappa_bit_offset: usize,
    construction2_u_out_x_i_bit_offset: usize,
    digest_count: usize,
    u64_count: usize,
    encoded_public_input_count: usize,
    field_lane_bit_offsets: Vec<usize>,
    field_lane_count: usize,
    construction2_commitment_fields: usize,
}

impl DirectCcsFPrimeLowNormSourceBuilder {
    fn push_digest(&mut self, digest: [u8; 32]) -> usize {
        let start = self.values.len();
        self.digest_count += 1;
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
        self.encoded_public_input_count += 1;
        self.push_field_lane_offsets(start, 4);
        for bit in input.bit_image() {
            self.push_bit(bit);
        }
        start
    }

    fn push_construction2_boundary(&mut self, boundary: &Construction2PublicBoundary) -> Construction2BoundaryOffsets {
        let fresh_offset = self.push_digest(boundary.fresh_instance_digest);
        let commitment_digest_offset = self.push_digest(boundary.commitment_digest);
        let commitment_d_offset = self.values.len();
        self.push_u64(boundary.commitment_d);
        let commitment_kappa_offset = self.values.len();
        self.push_u64(boundary.commitment_kappa);
        let x_i_offset = self.push_encoded_public_input(&boundary.x_i);
        Construction2BoundaryOffsets {
            fresh_digest: fresh_offset,
            commitment_digest: commitment_digest_offset,
            commitment_d: commitment_d_offset,
            commitment_kappa: commitment_kappa_offset,
            x_i: x_i_offset,
        }
    }

    fn push_field_u64(&mut self, value: F) {
        self.field_lane_bit_offsets.push(self.values.len());
        self.field_lane_count += 1;
        self.push_u64(value.as_canonical_u64());
    }

    fn push_u64(&mut self, value: u64) -> usize {
        let start = self.values.len();
        self.u64_count += 1;
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
        self.field_lane_count += count;
        self.field_lane_bit_offsets
            .extend((0..count).map(|idx| start + idx * 64));
    }

    fn finish(self) -> DirectCcsFPrimeLowNormSourceImage {
        DirectCcsFPrimeLowNormSourceImage {
            values: self.values,
            mat_digest_bit_offset: self.mat_digest_bit_offset,
            vk_fs_digest_bit_offset: self.vk_fs_digest_bit_offset,
            pc_bit_offset: self.pc_bit_offset,
            chunk_count_in_bit_offset: self.chunk_count_in_bit_offset,
            step_count_in_bit_offset: self.step_count_in_bit_offset,
            chunk_count_out_bit_offset: self.chunk_count_out_bit_offset,
            step_count_out_bit_offset: self.step_count_out_bit_offset,
            initial_boundary_digest_bit_offset: self.initial_boundary_digest_bit_offset,
            current_boundary_in_digest_bit_offset: self.current_boundary_in_digest_bit_offset,
            current_boundary_out_digest_bit_offset: self.current_boundary_out_digest_bit_offset,
            public_trace_in_digest_bit_offset: self.public_trace_in_digest_bit_offset,
            public_trace_out_digest_bit_offset: self.public_trace_out_digest_bit_offset,
            semantic_accumulator_in_digest_bit_offset: self.semantic_accumulator_in_digest_bit_offset,
            semantic_accumulator_out_digest_bit_offset: self.semantic_accumulator_out_digest_bit_offset,
            f_prime_accumulator_in_digest_bit_offset: self.f_prime_accumulator_in_digest_bit_offset,
            f_prime_accumulator_out_digest_bit_offset: self.f_prime_accumulator_out_digest_bit_offset,
            compact_x_in_bit_offset: self.compact_x_in_bit_offset,
            compact_x_out_bit_offset: self.compact_x_out_bit_offset,
            compact_construction2_u_in_digest_bit_offset: self.compact_construction2_u_in_digest_bit_offset,
            compact_construction2_u_out_digest_bit_offset: self.compact_construction2_u_out_digest_bit_offset,
            latest_chunk_digest_bit_offset: self.latest_chunk_digest_bit_offset,
            latest_fold_digest_bit_offset: self.latest_fold_digest_bit_offset,
            latest_chunk_relation_digest_bit_offset: self.latest_chunk_relation_digest_bit_offset,
            fresh_claims_bit_offset: self.fresh_claims_bit_offset,
            incoming_ce_claims_bit_offset: self.incoming_ce_claims_bit_offset,
            output_ce_claims_bit_offset: self.output_ce_claims_bit_offset,
            final_ce_claims_bit_offset: self.final_ce_claims_bit_offset,
            nifs_chunk_index_bit_offset: self.nifs_chunk_index_bit_offset,
            nifs_fresh_claims_bit_offset: self.nifs_fresh_claims_bit_offset,
            nifs_incoming_ce_claims_bit_offset: self.nifs_incoming_ce_claims_bit_offset,
            nifs_pi_ccs_outputs_bit_offset: self.nifs_pi_ccs_outputs_bit_offset,
            nifs_final_ce_claims_bit_offset: self.nifs_final_ce_claims_bit_offset,
            nifs_fe_sumcheck_rounds_bit_offset: self.nifs_fe_sumcheck_rounds_bit_offset,
            nifs_fe_sumcheck_messages_bit_offset: self.nifs_fe_sumcheck_messages_bit_offset,
            nifs_nc_sumcheck_rounds_bit_offset: self.nifs_nc_sumcheck_rounds_bit_offset,
            nifs_nc_sumcheck_messages_bit_offset: self.nifs_nc_sumcheck_messages_bit_offset,
            nifs_transcript_absorbed_in_bit_offset: self.nifs_transcript_absorbed_in_bit_offset,
            nifs_transcript_absorbed_out_bit_offset: self.nifs_transcript_absorbed_out_bit_offset,
            construction2_u_in_fresh_digest_bit_offset: self.construction2_u_in_fresh_digest_bit_offset,
            construction2_u_in_commitment_digest_bit_offset: self.construction2_u_in_commitment_digest_bit_offset,
            construction2_u_in_commitment_d_bit_offset: self.construction2_u_in_commitment_d_bit_offset,
            construction2_u_in_commitment_kappa_bit_offset: self.construction2_u_in_commitment_kappa_bit_offset,
            construction2_u_in_x_i_bit_offset: self.construction2_u_in_x_i_bit_offset,
            construction2_u_out_fresh_digest_bit_offset: self.construction2_u_out_fresh_digest_bit_offset,
            construction2_u_out_commitment_digest_bit_offset: self.construction2_u_out_commitment_digest_bit_offset,
            construction2_u_out_commitment_d_bit_offset: self.construction2_u_out_commitment_d_bit_offset,
            construction2_u_out_commitment_kappa_bit_offset: self.construction2_u_out_commitment_kappa_bit_offset,
            construction2_u_out_x_i_bit_offset: self.construction2_u_out_x_i_bit_offset,
            field_lane_bit_offsets: self.field_lane_bit_offsets,
            digest_count: self.digest_count,
            u64_count: self.u64_count,
            encoded_public_input_count: self.encoded_public_input_count,
            field_lane_count: self.field_lane_count,
            construction2_commitment_fields: self.construction2_commitment_fields,
        }
    }
}

struct Construction2BoundaryOffsets {
    fresh_digest: usize,
    commitment_digest: usize,
    commitment_d: usize,
    commitment_kappa: usize,
    x_i: usize,
}
