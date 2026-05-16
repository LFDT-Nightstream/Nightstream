//! Read-only summaries for the direct-CCS recursive carrier.
//!
//! These types are diagnostic/reporting data only. They group counters by the
//! protocol surface that owns them instead of flattening every F' measurement
//! into one large struct.

use super::super::f_prime::chain::{
    DirectCcsFPrimeEncoderStatus, DIRECT_CCS_F_PRIME_EXACT_ENCODER_MAX_R1CS_CONSTRAINTS,
};
use super::super::f_prime::{
    DirectCcsFPrimeLowNormSourceR1csShape, DirectCcsFPrimeNifsPayloadShape, DirectCcsFPrimeVerifierBodyShape,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirectCcsRecursiveIvcSummary {
    pub semantic: DirectCcsRecursiveSemanticSummary,
    pub f_prime: DirectCcsRecursiveFPrimeSummary,
    pub proof: DirectCcsRecursiveProofSummary,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirectCcsRecursiveSemanticSummary {
    pub chunks: u64,
    pub steps: u64,
    pub terminal_chunks_synthesized: u64,
    pub carried_ce_claims: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirectCcsRecursiveFPrimeSummary {
    pub folded_r2_steps: u64,
    pub carried_ce_claims: usize,
    pub native_evaluator_available: bool,
    pub encoder_required: bool,
    pub encoder_available: bool,
    pub compact_image_digest: Option<[u8; 32]>,
    pub exact_encoder_row_cap: usize,
    pub low_norm_source: DirectCcsFPrimeLowNormSourceSummary,
    pub verifier_body: DirectCcsFPrimeVerifierBodySummary,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirectCcsFPrimeLowNormSourceSummary {
    pub available: bool,
    pub len: usize,
    pub digest: Option<[u8; 32]>,
    pub digest_count: usize,
    pub u64_count: usize,
    pub encoded_public_input_count: usize,
    pub field_lane_count: usize,
    pub construction2_commitment_fields: usize,
    pub nifs_payload_shape: Option<DirectCcsFPrimeNifsPayloadShape>,
    pub r1cs: DirectCcsFPrimeLowNormSourceR1csSummary,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DirectCcsFPrimeLowNormSourceR1csSummary {
    pub constraints: usize,
    pub variables: usize,
    pub nnz: usize,
    pub public_inputs: usize,
    pub private_bits: usize,
    pub counter_carry_bits: usize,
    pub shell_constraints: usize,
    pub bit_constraints: usize,
    pub x_out_link_constraints: usize,
    pub construction2_boundary_link_constraints: usize,
    pub construction2_instance_digest_link_constraints: usize,
    pub construction2_commitment_shape_constraints: usize,
    pub structural_counter_constraints: usize,
    pub structural_fixed_arity_constraints: usize,
    pub structural_counter_carry_bit_constraints: usize,
    pub canonical_field_lane_constraints: usize,
    pub canonical_field_lane_aux_bits: usize,
    pub poseidon_digest_recomputation_constraints: usize,
    pub nifs_v_verifier_constraints: usize,
    pub authority_constraints: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirectCcsFPrimeVerifierBodySummary {
    pub measured: bool,
    pub measure_skipped: bool,
    pub public_inputs: usize,
    pub constraints: usize,
    pub nifs: DirectCcsFPrimeVerifierNifsSummary,
    pub construction2_fold_constraints: usize,
    pub public_link_constraints: usize,
    pub chunk_done_constraints: usize,
    pub final_ce_relation_constraints: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DirectCcsFPrimeVerifierNifsSummary {
    pub constraints: usize,
    pub chunk_meta_constraints: usize,
    pub pi_ccs_constraints: usize,
    pub pi_rlc_constraints: usize,
    pub pi_dec_constraints: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirectCcsRecursiveProofSummary {
    pub standalone_authority_ready: bool,
    pub encoder_blocker: Option<&'static str>,
}

impl DirectCcsRecursiveFPrimeSummary {
    pub(crate) fn from_encoder_status(
        folded_r2_steps: u64,
        carried_ce_claims: usize,
        encoder_required: bool,
        status: DirectCcsFPrimeEncoderStatus,
    ) -> Self {
        Self {
            folded_r2_steps,
            carried_ce_claims,
            native_evaluator_available: status.native_evaluator_available,
            encoder_required,
            encoder_available: status.low_norm_relation_available,
            compact_image_digest: status.compact_image_digest,
            exact_encoder_row_cap: DIRECT_CCS_F_PRIME_EXACT_ENCODER_MAX_R1CS_CONSTRAINTS,
            low_norm_source: DirectCcsFPrimeLowNormSourceSummary::from_encoder_status(&status),
            verifier_body: DirectCcsFPrimeVerifierBodySummary::from_encoder_status(&status),
        }
    }
}

impl DirectCcsFPrimeLowNormSourceSummary {
    fn from_encoder_status(status: &DirectCcsFPrimeEncoderStatus) -> Self {
        Self {
            available: status.low_norm_source_available,
            len: status.low_norm_source_len,
            digest: status.low_norm_source_digest,
            digest_count: status.low_norm_source_digest_count,
            u64_count: status.low_norm_source_u64_count,
            encoded_public_input_count: status.low_norm_source_encoded_public_input_count,
            field_lane_count: status.low_norm_source_field_lane_count,
            construction2_commitment_fields: status.low_norm_source_construction2_commitment_fields,
            nifs_payload_shape: status.nifs_payload_shape,
            r1cs: DirectCcsFPrimeLowNormSourceR1csSummary::from_shape(status.low_norm_source_r1cs_shape),
        }
    }
}

impl DirectCcsFPrimeLowNormSourceR1csSummary {
    fn from_shape(shape: Option<DirectCcsFPrimeLowNormSourceR1csShape>) -> Self {
        let Some(shape) = shape else {
            return Self::default();
        };
        Self {
            constraints: shape.constraint_count,
            variables: shape.variable_count,
            nnz: shape.nonzero_entries,
            public_inputs: shape.public_input_len,
            private_bits: shape.source.private_bits,
            counter_carry_bits: shape.variables.counter_carry_bits,
            shell_constraints: shape.shell_constraints(),
            bit_constraints: shape.constraints.bitness,
            x_out_link_constraints: shape.constraints.x_out_link,
            construction2_boundary_link_constraints: shape.constraints.construction2_boundary_link,
            construction2_instance_digest_link_constraints: shape.constraints.construction2_instance_digest_link,
            construction2_commitment_shape_constraints: shape.constraints.construction2_commitment_shape,
            structural_counter_constraints: shape.constraints.structural_counter,
            structural_fixed_arity_constraints: shape.constraints.structural_fixed_arity,
            structural_counter_carry_bit_constraints: shape.constraints.structural_counter_carry_bitness,
            canonical_field_lane_constraints: shape.constraints.canonical_field_lane,
            canonical_field_lane_aux_bits: shape.variables.canonical_field_lane_aux_bits,
            poseidon_digest_recomputation_constraints: shape.constraints.poseidon_digest_recomputation,
            nifs_v_verifier_constraints: shape.constraints.nifs_v_verifier,
            authority_constraints: shape.authority_constraints(),
        }
    }
}

impl DirectCcsFPrimeVerifierBodySummary {
    fn from_encoder_status(status: &DirectCcsFPrimeEncoderStatus) -> Self {
        let shape = status.verifier_body_shape.as_ref();
        Self {
            measured: shape.is_some(),
            measure_skipped: status.verifier_body_measure_skipped,
            public_inputs: shape.map_or(0, |shape| shape.public_inputs),
            constraints: shape.map_or(0, |shape| shape.constraints),
            nifs: DirectCcsFPrimeVerifierNifsSummary::from_shape(shape),
            construction2_fold_constraints: shape.map_or(0, |shape| shape.construction2_fold_constraints),
            public_link_constraints: shape.map_or(0, |shape| shape.public_link_constraints),
            chunk_done_constraints: shape.map_or(0, |shape| shape.chunk_done_constraints),
            final_ce_relation_constraints: shape.map_or(0, |shape| shape.final_ce_relation_constraints),
        }
    }
}

impl DirectCcsFPrimeVerifierNifsSummary {
    fn from_shape(shape: Option<&DirectCcsFPrimeVerifierBodyShape>) -> Self {
        Self {
            constraints: shape.map_or(0, |shape| shape.nifs_constraints()),
            chunk_meta_constraints: shape.map_or(0, |shape| shape.nifs.chunk_meta_constraints),
            pi_ccs_constraints: shape.map_or(0, |shape| shape.nifs.pi_ccs_constraints),
            pi_rlc_constraints: shape.map_or(0, |shape| shape.nifs.pi_rlc_constraints),
            pi_dec_constraints: shape.map_or(0, |shape| shape.nifs.pi_dec_constraints),
        }
    }
}
