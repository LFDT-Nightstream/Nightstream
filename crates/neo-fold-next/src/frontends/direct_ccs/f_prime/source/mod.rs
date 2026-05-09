//! Low-norm source encoding for native direct-CCS F' advice.
//!
//! The encoded source is binary witness material for the R1CS shell. It is a
//! diagnostic/input surface until the verifier body proves F' authority.

use neo_math::F;
use serde::{Deserialize, Serialize};

use super::super::state::DirectCcsFPrimeSnarkError;
use super::DirectCcsNativeFPrimeAdvice;

mod accessors;
mod builder;
mod digest;

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
        builder::build_low_norm_source_image_from_native_advice(advice)
    }
}
