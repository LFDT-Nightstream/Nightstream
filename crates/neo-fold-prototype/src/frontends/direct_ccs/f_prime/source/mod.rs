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
    offsets: DirectCcsFPrimeLowNormSourceOffsets,
    field_lane_bit_offsets: Vec<usize>,
    stats: DirectCcsFPrimeLowNormSourceStats,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct DirectCcsFPrimeLowNormSourceOffsets {
    pub(super) digests: DirectCcsFPrimeDigestOffsets,
    pub(super) counters: DirectCcsFPrimeCounterOffsets,
    pub(super) public_inputs: DirectCcsFPrimePublicInputOffsets,
    pub(super) nifs: DirectCcsFPrimeNifsOffsets,
    pub(super) construction2_u_in: DirectCcsFPrimeConstruction2BoundaryOffsets,
    pub(super) construction2_u_out: DirectCcsFPrimeConstruction2BoundaryOffsets,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct DirectCcsFPrimeDigestOffsets {
    pub(super) mat_digest: usize,
    pub(super) vk_fs_digest: usize,
    pub(super) initial_boundary_digest: usize,
    pub(super) current_boundary_in_digest: usize,
    pub(super) current_boundary_out_digest: usize,
    pub(super) public_trace_in_digest: usize,
    pub(super) public_trace_out_digest: usize,
    pub(super) semantic_accumulator_in_digest: usize,
    pub(super) semantic_accumulator_out_digest: usize,
    pub(super) f_prime_accumulator_in_digest: usize,
    pub(super) f_prime_accumulator_out_digest: usize,
    pub(super) compact_construction2_u_in_digest: usize,
    pub(super) compact_construction2_u_out_digest: usize,
    pub(super) latest_chunk_digest: usize,
    pub(super) latest_fold_digest: usize,
    pub(super) latest_chunk_relation_digest: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct DirectCcsFPrimeCounterOffsets {
    pub(super) pc: usize,
    pub(super) chunk_count_in: usize,
    pub(super) step_count_in: usize,
    pub(super) chunk_count_out: usize,
    pub(super) step_count_out: usize,
    pub(super) fresh_claims: usize,
    pub(super) incoming_ce_claims: usize,
    pub(super) output_ce_claims: usize,
    pub(super) final_ce_claims: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct DirectCcsFPrimePublicInputOffsets {
    pub(super) compact_x_in: usize,
    pub(super) compact_x_out: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct DirectCcsFPrimeNifsOffsets {
    pub(super) chunk_index: usize,
    pub(super) fresh_claims: usize,
    pub(super) incoming_ce_claims: usize,
    pub(super) pi_ccs_outputs: usize,
    pub(super) final_ce_claims: usize,
    pub(super) fe_sumcheck_rounds: usize,
    pub(super) fe_sumcheck_messages: usize,
    pub(super) nc_sumcheck_rounds: usize,
    pub(super) nc_sumcheck_messages: usize,
    pub(super) transcript_absorbed_in: usize,
    pub(super) transcript_absorbed_out: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct DirectCcsFPrimeConstruction2BoundaryOffsets {
    pub(super) fresh_digest: usize,
    pub(super) commitment_digest: usize,
    pub(super) commitment_d: usize,
    pub(super) commitment_kappa: usize,
    pub(super) x_i: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct DirectCcsFPrimeLowNormSourceStats {
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
