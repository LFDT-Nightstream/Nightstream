//! Owns the direct-CCS Construction-2 prover carrier.
//!
//! This type owns the non-VM direct CCS/R1CS append state. Standalone Spartan
//! compression proves the latest committed `F'` step and the final CE bundle
//! for the folded `F'` accumulator, matching the RV32IM two-part terminal
//! boundary without replaying historical chunks.

mod public_image;
mod snark;
mod state;

pub use snark::verify_direct_ccs_recursive_ivc_snark_public;

use std::time::Instant;

use neo_ajtai::{AjtaiSModule, Commitment};
use neo_ccs::{CeClaim, Mat};
use neo_math::{F, K};
use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};
use spartan2::traits::snark::DigestHelperTrait;

use super::ce_bundle::{
    canonical_direct_ce_claims, direct_ce_bundle_witnesses, measure_direct_ce_bundle_relation,
    prove_direct_ce_bundle_relation, setup_direct_ce_bundle_relation, verify_direct_ce_bundle_relation,
    DirectCcsCeBundleProof, DirectCcsCeBundleVerifierKey,
};
use super::circuit_util::{direct_accumulator_digest_from_claims, direct_accumulator_digest_from_claims_with_base};
use super::f_prime::DirectCcsFPrimeNifsPayloadShape;
use super::f_prime_chain::{
    DirectCcsFPrimeChain, DirectCcsFPrimeEncoderStatus, DIRECT_CCS_F_PRIME_EXACT_ENCODER_MAX_R1CS_CONSTRAINTS,
    DIRECT_CCS_F_PRIME_LOW_NORM_ENCODER_BLOCKER,
};
use super::ivc::{
    DirectCcsFPrimeSnarkError, DirectCcsFPrimeSnarkPerf, DirectCcsIvcState, DirectCcsProgram, DirectCcsStep,
};
use super::public_image::DirectCcsIvcPublicImage;
use super::snark::{DirectCcsIvcSnark, DirectCcsIvcSnarkVerifierKey};
use crate::ivc::SuperNeoIvcStepRelation;
use crate::prover::CommitmentMixers;

#[derive(Clone)]
pub struct DirectCcsRecursiveIvcState {
    direct: DirectCcsIvcState,
    f_prime_chain: DirectCcsFPrimeChain,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirectCcsRecursiveIvcSummary {
    pub semantic_chunks: u64,
    pub semantic_steps: u64,
    pub terminal_chunks_synthesized: u64,
    pub carried_semantic_ce_claims: usize,
    pub folded_f_prime_r2_steps: u64,
    pub carried_f_prime_ce_claims: usize,
    pub native_f_prime_evaluator_available: bool,
    pub f_prime_encoder_required: bool,
    pub f_prime_encoder_available: bool,
    pub compact_f_prime_image_digest: Option<[u8; 32]>,
    pub low_norm_f_prime_source_available: bool,
    pub low_norm_f_prime_source_len: usize,
    pub low_norm_f_prime_source_digest: Option<[u8; 32]>,
    pub low_norm_f_prime_source_r1cs_constraints: usize,
    pub low_norm_f_prime_source_r1cs_variables: usize,
    pub low_norm_f_prime_source_r1cs_nnz: usize,
    pub low_norm_f_prime_source_public_inputs: usize,
    pub low_norm_f_prime_source_private_bits: usize,
    pub low_norm_f_prime_source_counter_carry_bits: usize,
    pub low_norm_f_prime_source_digest_count: usize,
    pub low_norm_f_prime_source_u64_count: usize,
    pub low_norm_f_prime_source_encoded_public_input_count: usize,
    pub low_norm_f_prime_source_field_lane_count: usize,
    pub low_norm_f_prime_source_construction2_commitment_fields: usize,
    pub low_norm_f_prime_nifs_payload_shape: Option<DirectCcsFPrimeNifsPayloadShape>,
    pub f_prime_verifier_body_measured: bool,
    pub f_prime_verifier_body_measure_skipped: bool,
    pub f_prime_verifier_body_public_inputs: usize,
    pub f_prime_verifier_body_constraints: usize,
    pub f_prime_verifier_body_nifs_constraints: usize,
    pub f_prime_verifier_body_nifs_chunk_meta_constraints: usize,
    pub f_prime_verifier_body_nifs_pi_ccs_constraints: usize,
    pub f_prime_verifier_body_nifs_pi_rlc_constraints: usize,
    pub f_prime_verifier_body_nifs_pi_dec_constraints: usize,
    pub f_prime_verifier_body_construction2_fold_constraints: usize,
    pub f_prime_verifier_body_public_link_constraints: usize,
    pub f_prime_verifier_body_chunk_done_constraints: usize,
    pub f_prime_verifier_body_final_ce_relation_constraints: usize,
    pub f_prime_exact_encoder_row_cap: usize,
    pub low_norm_f_prime_source_shell_constraints: usize,
    pub low_norm_f_prime_source_bit_constraints: usize,
    pub low_norm_f_prime_source_x_out_link_constraints: usize,
    pub low_norm_f_prime_source_construction2_boundary_link_constraints: usize,
    pub low_norm_f_prime_source_construction2_instance_digest_link_constraints: usize,
    pub low_norm_f_prime_source_construction2_commitment_shape_constraints: usize,
    pub low_norm_f_prime_source_structural_counter_constraints: usize,
    pub low_norm_f_prime_source_structural_fixed_arity_constraints: usize,
    pub low_norm_f_prime_source_structural_counter_carry_bit_constraints: usize,
    pub low_norm_f_prime_source_canonical_field_lane_constraints: usize,
    pub low_norm_f_prime_source_canonical_field_lane_aux_bits: usize,
    pub low_norm_f_prime_source_poseidon_digest_recomputation_constraints: usize,
    pub low_norm_f_prime_source_nifs_v_verifier_constraints: usize,
    pub low_norm_f_prime_source_authority_constraints: usize,
    pub f_prime_encoder_blocker: Option<&'static str>,
    pub standalone_proof_authority_ready: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectCcsRecursiveIvcPublicImage {
    pub terminal_public_image: DirectCcsIvcPublicImage,
    pub proven_accumulator_digest: [u8; 32],
    pub proven_f_prime_accumulator_digest: [u8; 32],
    pub f_prime_accumulator_base: u32,
    pub proven_chunk_count: u64,
    pub proven_step_count: u64,
    pub folded_f_prime_r2_steps: u64,
    pub f_prime_final_ce_claims: u64,
}

pub struct DirectCcsRecursiveIvcSnarkVerifierKey {
    terminal: DirectCcsIvcSnarkVerifierKey,
    f_prime_chain: Option<DirectCcsIvcSnarkVerifierKey>,
    f_prime_final_ce: Option<DirectCcsCeBundleVerifierKey>,
    expected_f_prime_default_accumulator_digest: [u8; 32],
    expected_f_prime_accumulator_base: u32,
    expected_f_prime_final_ce_claims: u64,
}

impl DirectCcsRecursiveIvcSnarkVerifierKey {
    pub fn expected_digest(&self) -> Result<[u8; 32], DirectCcsFPrimeSnarkError> {
        let terminal_digest = self.terminal.expected_digest()?;
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/direct_ccs/recursive_ivc_snark_verifier_key");
        tr.append_message(
            b"neo.fold.next/direct_ccs/recursive_ivc_snark_verifier_key/version",
            b"v1",
        );
        tr.append_message(
            b"neo.fold.next/direct_ccs/recursive_ivc_snark_verifier_key/terminal",
            &terminal_digest,
        );
        tr.append_message(
            b"neo.fold.next/direct_ccs/recursive_ivc_snark_verifier_key/default_f_prime_accumulator",
            &self.expected_f_prime_default_accumulator_digest,
        );
        match self.f_prime_chain.as_ref() {
            Some(vk) => {
                let f_prime_chain_digest = vk.expected_digest()?;
                tr.append_u64s(
                    b"neo.fold.next/direct_ccs/recursive_ivc_snark_verifier_key/has_f_prime_chain",
                    &[1],
                );
                tr.append_message(
                    b"neo.fold.next/direct_ccs/recursive_ivc_snark_verifier_key/f_prime_chain",
                    &f_prime_chain_digest,
                );
            }
            None => {
                tr.append_u64s(
                    b"neo.fold.next/direct_ccs/recursive_ivc_snark_verifier_key/has_f_prime_chain",
                    &[0],
                );
            }
        }
        match self.f_prime_final_ce.as_ref() {
            Some(vk) => {
                let final_ce_digest = vk
                    .digest()
                    .map_err(|err| DirectCcsFPrimeSnarkError::Encode(err.to_string()))?;
                tr.append_u64s(
                    b"neo.fold.next/direct_ccs/recursive_ivc_snark_verifier_key/has_f_prime_final_ce",
                    &[1],
                );
                tr.append_message(
                    b"neo.fold.next/direct_ccs/recursive_ivc_snark_verifier_key/f_prime_final_ce",
                    &final_ce_digest,
                );
            }
            None => {
                tr.append_u64s(
                    b"neo.fold.next/direct_ccs/recursive_ivc_snark_verifier_key/has_f_prime_final_ce",
                    &[0],
                );
            }
        }
        tr.append_u64s(
            b"neo.fold.next/direct_ccs/recursive_ivc_snark_verifier_key/f_prime_shape",
            &[
                self.expected_f_prime_accumulator_base as u64,
                self.expected_f_prime_final_ce_claims,
            ],
        );
        Ok(tr.digest32())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DirectCcsRecursiveIvcSnark {
    terminal: DirectCcsIvcSnark,
    f_prime_chain: Option<DirectCcsIvcSnark>,
    f_prime_final_claims: Vec<CeClaim<Commitment, F, K>>,
    f_prime_final_ce_proof: Option<DirectCcsCeBundleProof>,
    public_image: DirectCcsRecursiveIvcPublicImage,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct DirectCcsRecursiveIvcSnarkPerf {
    pub terminal: DirectCcsFPrimeSnarkPerf,
    pub f_prime_chain: Option<DirectCcsFPrimeSnarkPerf>,
    pub f_prime_chain_setup_ms: f64,
    pub f_prime_chain_prove_ms: f64,
    pub f_prime_chain_verify_ms: f64,
    pub f_prime_chain_constraints: usize,
    pub f_prime_chain_proof_bytes: usize,
    pub f_prime_final_ce_setup_ms: f64,
    pub f_prime_final_ce_prove_ms: f64,
    pub f_prime_final_ce_verify_ms: f64,
    pub f_prime_final_ce_constraints: usize,
    pub f_prime_final_ce_digest_constraints: usize,
    pub f_prime_final_ce_digest_match_constraints: usize,
    pub f_prime_final_ce_relation_constraints: usize,
    pub f_prime_final_ce_public_inputs: usize,
    pub f_prime_final_ce_claims: usize,
    pub total_prove_ms: f64,
    pub total_verify_ms: f64,
    pub terminal_proof_bytes: usize,
    pub f_prime_final_ce_proof_bytes: usize,
    pub total_proof_bytes: usize,
}
