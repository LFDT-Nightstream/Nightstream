//! Owns the direct Spartan proof for the full RV64IM main relation.
//!
//! This module compiles the route-owned `R_main^SN` witness relation directly.
//! It does not route theorem meaning through the generic shell target.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use neo_ajtai::Commitment;
use neo_ccs::{build_superneo_ring_forms, CcsClaim, CcsStructure, CcsWitness, Mat, SModuleHomomorphism};
use neo_math::{balanced::to_balanced_i128, KExtensions, D, F, K};
use neo_params::NeoParams;
use neo_reductions::common::{
    compute_y_zcol_from_witness, compute_y_zcol_from_witness_digits, decode_superneo_coeffs_from_witness_mat,
};
use neo_reductions::engines::utils::{
    build_dims_and_policy, digest_ccs_matrices_with_sparse_cache, Dims, PI_CCS_SUMCHECK_FE_RAW_DOMAIN_TAG,
    PI_CCS_SUMCHECK_INITIAL_RAW_TAG, PI_CCS_SUMCHECK_NC_RAW_DOMAIN_TAG,
};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use serde::{Deserialize, Serialize};
use spartan2::{
    bellpepper::poseidon2::hash_packed_goldilocks_fields,
    provider::{goldi::F as SpartanF, GoldilocksP3MerkleMleEngine},
    spartan::R1CSSNARK,
    traits::{circuit::SpartanCircuit, snark::R1CSSNARKTrait},
};
use thiserror::Error;

use crate::finalize::digest32_as_fields;
use crate::proof::PublicChunk;
use crate::rv64im::chunk_relation::{
    trace_rv64im_chunk_relation_with_replay, verify_rv64im_chunk_relation_with_replay, Rv64imChunkRelationTrace,
    RV64IM_CHUNK_RELATION_DIGEST_RAW_TAG,
};
use crate::rv64im::decider_relation::{validate_rv64im_decider_relation_surface, Rv64imDeciderRelation};
use crate::rv64im::final_relation::{
    final_statement_digest, folded_statement_digest, Rv64imFinalProof, Rv64imFinalStatement, RV64IM_CHUNK_DONE_RAW_TAG,
    RV64IM_SESSION_RAW_DOMAIN_TAG,
};
use crate::rv64im::kernel::{
    rv64im_cached_root_main_lane_context, rv64im_cached_root_main_lane_optimized_cache,
    verify_rv64im_kernel_export_proof_with_output, Rv64imKernelExportProof, Rv64imKernelExportRelationResult,
    SimpleKernelError,
};
use crate::rv64im::main_relation_circuit::claim::{
    alloc_ce_claim_point_only_with_shared_point, alloc_ce_claim_public_surface_with_shared_point,
    alloc_ce_claim_with_shared_point, alloc_ce_claim_without_f_surface_with_shared_point, CeClaimVar,
};
use crate::rv64im::main_relation_circuit::initial_sum::claimed_initial_sum_from_me_inputs;
use crate::rv64im::main_relation_circuit::k_field::{alloc_constant_k, KNum, KNumVar};
use crate::rv64im::main_relation_circuit::output_binding::{
    embedded_fresh_x_values, enforce_me_outputs_against_inputs, set_fresh_output_constant_f_surface,
};
use crate::rv64im::main_relation_circuit::pi_ccs::{
    bind_header_and_instance_digest, bind_me_inputs, sample_challenges,
};
use crate::rv64im::main_relation_circuit::rho_sampling::{
    materialize_goldilocks_rot_matrices, sample_goldilocks_rot_rhos,
};
use crate::rv64im::main_relation_circuit::rlc_dec::{
    enforce_dec_public, enforce_rlc_dec_public_with_rho_coeffs_for_last_chunk,
    enforce_rlc_public_with_rho_coeffs_for_constant_children, enforce_rlc_public_with_rho_vars_constant_prefix,
};
use crate::rv64im::main_relation_circuit::structure::pad_ccs_structure_to_block_width;
use crate::rv64im::main_relation_circuit::sumcheck::{sumcheck_eval_gadget, sumcheck_round_gadget};
use crate::rv64im::main_relation_circuit::sumcheck_replay::verify_sumcheck_rounds;
use crate::rv64im::main_relation_circuit::terminal_identity::{
    enforce_terminal_identity_fe, enforce_terminal_identity_nc,
};
use crate::rv64im::main_relation_circuit::transcript::Poseidon2TranscriptCircuit;
mod debug;

const RV64IM_MAIN_RELATION_DELTA: u64 = 7;
const CHUNK_META_RAW_TAG: u64 = 14;
const STEP_INDEX_RAW_TAG: u64 = 15;
pub type Rv64imSpartan2DeciderEngine = GoldilocksP3MerkleMleEngine;
pub type Rv64imSpartan2DeciderSnark = R1CSSNARK<Rv64imSpartan2DeciderEngine>;
pub type Rv64imSpartan2DeciderProverKey = spartan2::spartan::SpartanProverKey<Rv64imSpartan2DeciderEngine>;
pub type Rv64imSpartan2DeciderVerifierKey = spartan2::spartan::SpartanVerifierKey<Rv64imSpartan2DeciderEngine>;
pub type Rv64imSpartan2DeciderKeyPair = Arc<(Rv64imSpartan2DeciderProverKey, Rv64imSpartan2DeciderVerifierKey)>;

static RV64IM_MAIN_RELATION_SETUP_CACHE: OnceLock<Mutex<HashMap<[u8; 32], Rv64imSpartan2DeciderKeyPair>>> =
    OnceLock::new();

use debug::append_k_to_transcript;
pub use debug::{
    debug_check_rv64im_spartan2_decider_circuit, inspect_rv64im_spartan2_decider_trace,
    measure_rv64im_spartan2_decider_circuit, Rv64imMainRelationCircuitMetrics, Rv64imMainRelationCountBucket,
    Rv64imMainRelationHotspotDetail, Rv64imMainRelationPhaseBucket, Rv64imMainRelationSumcheckBucket,
    Rv64imMainRelationSurfaceFamilyBucket, Rv64imMainRelationSurfaceMetrics, Rv64imMainRelationTraceStats,
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imSpartan2DeciderProof {
    pub snark_data: Vec<u8>,
}

impl Rv64imSpartan2DeciderProof {
    pub fn snark_bytes_len(&self) -> usize {
        self.snark_data.len()
    }
}

#[derive(Debug, Error)]
pub enum Rv64imSpartan2DeciderError {
    #[error("rv64im main relation setup failed: {0}")]
    Setup(String),
    #[error("rv64im main relation prepare failed: {0}")]
    Prepare(String),
    #[error("rv64im main relation prove failed: {0}")]
    Prove(String),
    #[error("rv64im main relation verify failed: {0}")]
    Verify(String),
    #[error("rv64im main relation proof encoding failed: {0}")]
    Encode(String),
    #[error("rv64im main relation proof decoding failed: {0}")]
    Decode(String),
    #[error("rv64im main relation public IO mismatch: {0}")]
    PublicIo(String),
}

#[derive(Clone)]
struct Rv64imMainRelationHandoff {
    public_chunk: PublicChunk,
    public_chunk_instance_digest: [F; 4],
    public_chunk_digest: [u8; 32],
    bridge_handoff_digest: [u8; 32],
    chunk_relation_digest: [u8; 32],
}

#[derive(Clone)]
struct Rv64imMainRelationChunkTrace {
    handoff: Rv64imMainRelationHandoff,
    fresh_claims: Vec<CcsClaim<Commitment, F>>,
    fresh_witnesses: Vec<CcsWitness<F>>,
    ccs_trace: Rv64imChunkRelationTrace,
    replay_public_challenges: neo_reductions::optimized_engine::Challenges,
    replay_row_chals: Vec<K>,
    replay_alpha_prime: Vec<K>,
    replay_s_col: Vec<K>,
    replay_alpha_prime_nc: Vec<K>,
}

#[derive(Clone)]
struct Rv64imMainRelationTrace {
    statement: Rv64imFinalStatement,
    chunk_traces: Vec<Rv64imMainRelationChunkTrace>,
}

#[derive(Clone)]
struct Rv64imMainRelationCircuit {
    public_statement_digest: [u8; 32],
    params: NeoParams,
    structure: CcsStructure<F>,
    dims: Dims,
    mat_digest: [Goldilocks; 4],
    trace: Rv64imMainRelationTrace,
}

impl Rv64imMainRelationCircuit {
    fn delta() -> SpartanF {
        SpartanF::from_canonical_u64(RV64IM_MAIN_RELATION_DELTA)
    }

    fn expected_public_values(&self) -> Vec<SpartanF> {
        let mut out = Vec::new();
        out.extend(
            digest32_as_fields(self.public_statement_digest)
                .into_iter()
                .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64())),
        );
        for chunk in &self.trace.chunk_traces {
            out.extend(
                digest32_as_fields(chunk.handoff.chunk_relation_digest)
                    .into_iter()
                    .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64())),
            );
        }
        out
    }
}

impl SpartanCircuit<Rv64imSpartan2DeciderEngine> for Rv64imMainRelationCircuit {
    fn public_values(&self) -> Result<Vec<SpartanF>, SynthesisError> {
        Ok(self.expected_public_values())
    }

    fn shared<CS: ConstraintSystem<SpartanF>>(
        &self,
        _: &mut CS,
    ) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
        Ok(Vec::new())
    }

    fn precommitted<CS: ConstraintSystem<SpartanF>>(
        &self,
        _: &mut CS,
        _: &[AllocatedNum<SpartanF>],
    ) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
        Ok(Vec::new())
    }

    fn num_challenges(&self) -> usize {
        0
    }

    fn synthesize<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        _: &[AllocatedNum<SpartanF>],
        _: &[AllocatedNum<SpartanF>],
        _: Option<&[SpartanF]>,
    ) -> Result<(), SynthesisError> {
        let public_inputs = self
            .expected_public_values()
            .into_iter()
            .enumerate()
            .map(|(idx, value)| AllocatedNum::alloc_input(cs.namespace(|| format!("public_input_{idx}")), || Ok(value)))
            .collect::<Result<Vec<_>, _>>()?;
        let mut public_cursor = 0usize;
        let _public_statement_digest =
            next_public_digest(&public_inputs, &mut public_cursor, "public_statement_digest")?;
        let mut transcript = Poseidon2TranscriptCircuit::new_raw_fields(
            cs.namespace(|| "session_transcript"),
            &[SpartanF::from_canonical_u64(RV64IM_SESSION_RAW_DOMAIN_TAG)],
        )?;
        let delta = Self::delta();
        let mut carried_claims: Vec<CeClaimVar> = Vec::new();

        for (chunk_idx, chunk) in self.trace.chunk_traces.iter().enumerate() {
            append_chunk_meta(
                &mut cs.namespace(|| format!("chunk_meta_{chunk_idx}")),
                &mut transcript,
                &chunk.handoff,
            )?;

            bind_header_and_instance_digest(
                &mut cs.namespace(|| format!("chunk_{chunk_idx}_bind_header")),
                &mut transcript,
                &self.params,
                self.structure.n,
                self.structure.m,
                self.structure.t(),
                &self.structure.f,
                self.dims,
                &self.mat_digest,
                &chunk
                    .handoff
                    .public_chunk_instance_digest
                    .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64())),
            )?;
            bind_me_inputs(
                &mut cs.namespace(|| format!("chunk_{chunk_idx}_bind_me_inputs")),
                &mut transcript,
                &carried_claims,
            )?;
            let public_challenges = sample_challenges(
                &mut cs.namespace(|| format!("chunk_{chunk_idx}_sample_challenges")),
                &mut transcript,
                self.dims,
            )?;

            let (initial_sum_fe, initial_sum_fe_value) = claimed_initial_sum_from_me_inputs(
                &mut cs.namespace(|| format!("chunk_{chunk_idx}_initial_sum_fe")),
                &self.structure,
                &public_challenges.alpha,
                &chunk.replay_public_challenges.alpha,
                &public_challenges.gamma,
                chunk.replay_public_challenges.gamma,
                chunk.fresh_claims.len(),
                &carried_claims,
                delta,
                &format!("chunk_{chunk_idx}_initial_sum_fe"),
            )?;
            transcript.append_const_fields_raw(
                cs.namespace(|| format!("chunk_{chunk_idx}_fe_sumcheck_domain")),
                &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_FE_RAW_DOMAIN_TAG)],
            )?;
            if carried_claims.is_empty() {
                let coeffs = initial_sum_fe_value.as_coeffs();
                transcript.append_const_fields_raw(
                    cs.namespace(|| format!("chunk_{chunk_idx}_fe_sumcheck_initial_tag")),
                    &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_INITIAL_RAW_TAG)],
                )?;
                transcript.append_const_fields_raw(
                    cs.namespace(|| format!("chunk_{chunk_idx}_fe_sumcheck_initial_append")),
                    &[
                        SpartanF::from_canonical_u64(coeffs[0].as_canonical_u64()),
                        SpartanF::from_canonical_u64(coeffs[1].as_canonical_u64()),
                    ],
                )?;
            } else {
                append_k_to_transcript(
                    &mut cs.namespace(|| format!("chunk_{chunk_idx}_fe_sumcheck_initial")),
                    &mut transcript,
                    PI_CCS_SUMCHECK_INITIAL_RAW_TAG,
                    &initial_sum_fe,
                    initial_sum_fe_value,
                    &format!("chunk_{chunk_idx}_fe_sumcheck_initial"),
                )?;
            }
            let fe_rounds = alloc_rounds(
                &mut cs.namespace(|| format!("chunk_{chunk_idx}_fe_rounds")),
                &chunk.ccs_trace.ccs_replay_proof.sumcheck_rounds,
                &format!("chunk_{chunk_idx}_fe_round"),
            )?;
            let fe_challenge_values = chunk_sumcheck_challenges(&chunk.replay_row_chals, &chunk.replay_alpha_prime);
            let (fe_challenges, sumcheck_final_fe) = verify_sumcheck_rounds(
                &mut cs.namespace(|| format!("chunk_{chunk_idx}_fe_sumcheck")),
                &mut transcript,
                max_degree(&chunk.ccs_trace.ccs_replay_proof.sumcheck_rounds),
                &initial_sum_fe,
                &fe_rounds,
                &chunk.ccs_trace.ccs_replay_proof.sumcheck_rounds,
                &fe_challenge_values,
                delta,
                &format!("chunk_{chunk_idx}_fe_sumcheck"),
            )?;
            let (r_prime_vars, alpha_prime_vars) = split_vec(&fe_challenges, self.dims.ell_n)?;

            let zero_nc = alloc_constant_k(
                &mut cs.namespace(|| format!("chunk_{chunk_idx}_initial_sum_nc_zero")),
                KNum::from_neo_k(K::ZERO),
                &format!("chunk_{chunk_idx}_initial_sum_nc_zero"),
            )?;
            transcript.append_const_fields_raw(
                cs.namespace(|| format!("chunk_{chunk_idx}_nc_sumcheck_domain")),
                &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_NC_RAW_DOMAIN_TAG)],
            )?;
            transcript.append_const_fields_raw(
                cs.namespace(|| format!("chunk_{chunk_idx}_nc_sumcheck_initial_tag")),
                &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_INITIAL_RAW_TAG)],
            )?;
            transcript.append_const_fields_raw(
                cs.namespace(|| format!("chunk_{chunk_idx}_nc_sumcheck_initial_append")),
                &[SpartanF::from_canonical_u64(0), SpartanF::from_canonical_u64(0)],
            )?;
            let nc_rounds = alloc_rounds(
                &mut cs.namespace(|| format!("chunk_{chunk_idx}_nc_rounds")),
                &chunk.ccs_trace.ccs_replay_proof.sumcheck_rounds_nc,
                &format!("chunk_{chunk_idx}_nc_round"),
            )?;
            let nc_challenge_values = chunk_sumcheck_challenges(&chunk.replay_s_col, &chunk.replay_alpha_prime_nc);
            let (nc_challenges, sumcheck_final_nc) = verify_sumcheck_rounds(
                &mut cs.namespace(|| format!("chunk_{chunk_idx}_nc_sumcheck")),
                &mut transcript,
                max_degree(&chunk.ccs_trace.ccs_replay_proof.sumcheck_rounds_nc),
                &zero_nc,
                &nc_rounds,
                &chunk.ccs_trace.ccs_replay_proof.sumcheck_rounds_nc,
                &nc_challenge_values,
                delta,
                &format!("chunk_{chunk_idx}_nc_sumcheck"),
            )?;
            let (s_col_prime_vars, alpha_prime_nc_vars) = split_vec(&nc_challenges, self.dims.ell_m)?;

            let fold_digest_is_constant = transcript.constant_snapshot().is_some();
            let fold_digest = transcript.digest32(cs.namespace(|| format!("chunk_{chunk_idx}_fold_digest")))?;
            let chunk_relation_digest_input = next_public_digest(
                &public_inputs,
                &mut public_cursor,
                &format!("chunk_{chunk_idx}_relation_digest"),
            )?;
            let fold_digest_values = digest32_as_fields(chunk.ccs_trace.terminal_state.fold_digest)
                .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()));
            if fold_digest_is_constant {
                let expected = chunk_relation_digest_values(
                    chunk.handoff.public_chunk_digest,
                    fold_digest_values,
                    chunk.handoff.bridge_handoff_digest,
                );
                for (idx, value) in expected.iter().enumerate() {
                    cs.enforce(
                        || format!("chunk_{chunk_idx}_relation_digest_const_{idx}"),
                        |lc| lc + chunk_relation_digest_input[idx].get_variable(),
                        |lc| lc + CS::one(),
                        |lc| lc + (*value, CS::one()),
                    );
                }
            } else {
                let chunk_relation_digest = chunk_relation_digest_circuit(
                    &mut cs.namespace(|| format!("chunk_{chunk_idx}_relation_digest")),
                    chunk.handoff.public_chunk_digest,
                    &fold_digest,
                    &fold_digest_values,
                    chunk.handoff.bridge_handoff_digest,
                )?;
                enforce_digest_eq(
                    &mut cs.namespace(|| format!("chunk_{chunk_idx}_relation_digest_eq")),
                    &chunk_relation_digest,
                    &chunk_relation_digest_input,
                    &format!("chunk_{chunk_idx}_relation_digest_eq"),
                )?;
            }

            let mut ccs_outputs = Vec::with_capacity(chunk.ccs_trace.ccs_outputs.len());
            for (idx, claim) in chunk.ccs_trace.ccs_outputs.iter().enumerate() {
                let output = if idx < chunk.fresh_claims.len() {
                    let fresh = &chunk.fresh_claims[idx];
                    alloc_ce_claim_without_f_surface_with_shared_point(
                        &mut cs.namespace(|| format!("chunk_{chunk_idx}_ccs_output_{idx}")),
                        claim,
                        &fresh.c.data,
                        &embedded_fresh_x_values(fresh),
                        &r_prime_vars,
                        &chunk.replay_row_chals,
                        &s_col_prime_vars,
                        &chunk.replay_s_col,
                        &format!("chunk_{chunk_idx}_ccs_output_{idx}"),
                    )?
                } else {
                    alloc_ce_claim_public_surface_with_shared_point(
                        &mut cs.namespace(|| format!("chunk_{chunk_idx}_ccs_output_{idx}")),
                        claim,
                        &r_prime_vars,
                        &chunk.replay_row_chals,
                        &s_col_prime_vars,
                        &chunk.replay_s_col,
                        &format!("chunk_{chunk_idx}_ccs_output_{idx}"),
                    )?
                };
                ccs_outputs.push(output);
            }
            for (idx, fresh) in chunk.fresh_claims.iter().enumerate() {
                set_fresh_output_constant_f_surface(&mut ccs_outputs[idx], fresh)?;
            }
            enforce_me_outputs_against_inputs(
                &mut cs.namespace(|| format!("chunk_{chunk_idx}_output_binding")),
                &self.structure,
                &self.params,
                &chunk.fresh_claims,
                &carried_claims,
                &ccs_outputs,
                &r_prime_vars,
                &chunk.replay_row_chals,
                &s_col_prime_vars,
                &chunk.replay_s_col,
                &format!("chunk_{chunk_idx}_output_binding"),
            )?;
            let me_inputs_r_vars = carried_claims.first().map(|claim| claim.r.as_slice());
            let me_inputs_r_values = carried_claims
                .first()
                .map(|claim| claim.r_values.as_slice());
            let _ = enforce_terminal_identity_fe(
                &mut cs.namespace(|| format!("chunk_{chunk_idx}_terminal_fe")),
                &sumcheck_final_fe,
                &self.structure,
                &chunk.replay_public_challenges,
                &public_challenges.alpha,
                &public_challenges.beta_a,
                &public_challenges.beta_r,
                &public_challenges.gamma,
                &r_prime_vars,
                &chunk.replay_row_chals,
                &alpha_prime_vars,
                &chunk.replay_alpha_prime,
                &ccs_outputs,
                chunk.fresh_claims.len(),
                me_inputs_r_vars,
                me_inputs_r_values,
                delta,
                &format!("chunk_{chunk_idx}_terminal_fe"),
            )?;
            let _ = enforce_terminal_identity_nc(
                &mut cs.namespace(|| format!("chunk_{chunk_idx}_terminal_nc")),
                &sumcheck_final_nc,
                &self.params,
                &chunk.replay_public_challenges,
                &public_challenges.beta_a,
                &public_challenges.beta_m,
                &public_challenges.gamma,
                &s_col_prime_vars,
                &chunk.replay_s_col,
                &alpha_prime_nc_vars,
                &chunk.replay_alpha_prime_nc,
                &ccs_outputs,
                delta,
                &format!("chunk_{chunk_idx}_terminal_nc"),
            )?;

            let is_last_chunk = chunk_idx + 1 == self.trace.chunk_traces.len();
            let parent_claim = if is_last_chunk && chunk.fresh_claims.len() == ccs_outputs.len() {
                alloc_ce_claim_point_only_with_shared_point(
                    &chunk.ccs_trace.parent,
                    &chunk.ccs_trace.parent.c.data,
                    chunk.ccs_trace.parent.X.as_slice(),
                    &r_prime_vars,
                    &chunk.replay_row_chals,
                    &s_col_prime_vars,
                    &chunk.replay_s_col,
                )?
            } else {
                alloc_ce_claim_public_surface_with_shared_point(
                    &mut cs.namespace(|| format!("chunk_{chunk_idx}_parent_claim")),
                    &chunk.ccs_trace.parent,
                    &r_prime_vars,
                    &chunk.replay_row_chals,
                    &s_col_prime_vars,
                    &chunk.replay_s_col,
                    &format!("chunk_{chunk_idx}_parent_claim"),
                )?
            };
            let child_claim_source = if is_last_chunk {
                &self
                    .trace
                    .statement
                    .folded
                    .final_accumulator
                    .final_main_claims
            } else {
                &chunk.ccs_trace.children
            };
            let child_claims = if is_last_chunk {
                None
            } else {
                Some(
                    child_claim_source
                        .iter()
                        .enumerate()
                        .map(|(idx, claim)| {
                            alloc_ce_claim_with_shared_point(
                                &mut cs.namespace(|| format!("chunk_{chunk_idx}_child_claim_{idx}")),
                                claim,
                                &r_prime_vars,
                                &chunk.replay_row_chals,
                                &s_col_prime_vars,
                                &chunk.replay_s_col,
                                &format!("chunk_{chunk_idx}_child_claim_{idx}"),
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                )
            };
            let rho_vars = sample_goldilocks_rot_rhos(
                &mut cs.namespace(|| format!("chunk_{chunk_idx}_rlc_rhos")),
                &mut transcript,
                ccs_outputs.len(),
                &format!("chunk_{chunk_idx}_rlc_rhos"),
            )?;
            if is_last_chunk && chunk.fresh_claims.len() == ccs_outputs.len() {
                enforce_rlc_dec_public_with_rho_coeffs_for_last_chunk(
                    &mut cs.namespace(|| format!("chunk_{chunk_idx}_rlc_public")),
                    &parent_claim,
                    &ccs_outputs,
                    child_claim_source,
                    &rho_vars,
                    self.params.b,
                    &format!("chunk_{chunk_idx}_rlc_public"),
                )?;
            } else if chunk.fresh_claims.len() == ccs_outputs.len() {
                enforce_rlc_public_with_rho_coeffs_for_constant_children(
                    &mut cs.namespace(|| format!("chunk_{chunk_idx}_rlc_public")),
                    &parent_claim,
                    &ccs_outputs,
                    &rho_vars,
                    &format!("chunk_{chunk_idx}_rlc_public"),
                )?;
            } else {
                let rho_mats = materialize_goldilocks_rot_matrices(
                    &mut cs.namespace(|| format!("chunk_{chunk_idx}_rlc_rho_mats")),
                    &rho_vars,
                    &format!("chunk_{chunk_idx}_rlc_rho_mats"),
                )?;
                enforce_rlc_public_with_rho_vars_constant_prefix(
                    &mut cs.namespace(|| format!("chunk_{chunk_idx}_rlc_public")),
                    &parent_claim,
                    &ccs_outputs,
                    &rho_mats,
                    chunk.fresh_claims.len(),
                    &format!("chunk_{chunk_idx}_rlc_public"),
                )?;
            }
            if let Some(child_claims) = child_claims {
                enforce_dec_public(
                    &mut cs.namespace(|| format!("chunk_{chunk_idx}_dec_public")),
                    &parent_claim,
                    &child_claims,
                    self.params.b,
                    &format!("chunk_{chunk_idx}_dec_public"),
                )?;
                carried_claims = child_claims;
            } else if !is_last_chunk || chunk.fresh_claims.len() != ccs_outputs.len() {
                // Non-fused constant-child paths still need direct DEC checks here.
                crate::rv64im::main_relation_circuit::rlc_dec::enforce_dec_public_with_constant_children(
                    &mut cs.namespace(|| format!("chunk_{chunk_idx}_dec_public")),
                    &parent_claim,
                    child_claim_source,
                    self.params.b,
                    &format!("chunk_{chunk_idx}_dec_public"),
                )?;
            }
            transcript.append_const_fields_raw(
                cs.namespace(|| format!("chunk_done_{chunk_idx}")),
                &[
                    SpartanF::from_canonical_u64(RV64IM_CHUNK_DONE_RAW_TAG),
                    SpartanF::from_canonical_u64(1),
                ],
            )?;
        }

        if public_cursor != public_inputs.len() {
            return Err(SynthesisError::Unsatisfiable);
        }
        Ok(())
    }
}

pub fn setup_rv64im_spartan2_decider(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalProof,
) -> Result<(Rv64imSpartan2DeciderProverKey, Rv64imSpartan2DeciderVerifierKey), SimpleKernelError> {
    let circuit = build_main_relation_circuit(statement, proof)?;
    Rv64imSpartan2DeciderSnark::setup(circuit)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM main relation setup failed: {err}")))
}

pub fn setup_rv64im_spartan2_decider_cached(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalProof,
) -> Result<Rv64imSpartan2DeciderKeyPair, SimpleKernelError> {
    let cache_key = rv64im_compact_main_decider_shape_digest(statement, proof)?;
    let cache = RV64IM_MAIN_RELATION_SETUP_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(keys) = cache
        .lock()
        .map_err(|_| SimpleKernelError::Bridge("RV64IM main relation setup cache poisoned".into()))?
        .get(&cache_key)
        .cloned()
    {
        return Ok(keys);
    }

    let keys = Arc::new(setup_rv64im_spartan2_decider(statement, proof)?);
    cache
        .lock()
        .map_err(|_| SimpleKernelError::Bridge("RV64IM main relation setup cache poisoned".into()))?
        .insert(cache_key, keys.clone());
    Ok(keys)
}

pub fn prove_rv64im_spartan2_decider(
    pk: &Rv64imSpartan2DeciderProverKey,
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalProof,
) -> Result<Rv64imSpartan2DeciderProof, SimpleKernelError> {
    let circuit = build_main_relation_circuit(statement, proof)?;
    let prep = Rv64imSpartan2DeciderSnark::prep_prove(pk, circuit.clone(), true)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM main relation prepare failed: {err}")))?;
    let proof = Rv64imSpartan2DeciderSnark::prove(pk, circuit, &prep, true)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM main relation prove failed: {err}")))?;
    let snark_data = bincode::serialize(&proof)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM main relation encode failed: {err}")))?;
    Ok(Rv64imSpartan2DeciderProof { snark_data })
}

pub fn verify_rv64im_spartan2_decider(
    vk: &Rv64imSpartan2DeciderVerifierKey,
    public_statement_digest: [u8; 32],
    relation: &Rv64imDeciderRelation,
    decider_proof: &Rv64imSpartan2DeciderProof,
) -> Result<(), SimpleKernelError> {
    validate_rv64im_decider_relation_surface(relation)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM main relation verify failed: {err}")))?;
    let proof: Rv64imSpartan2DeciderSnark = bincode::deserialize(&decider_proof.snark_data)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM main relation decode failed: {err}")))?;
    let public_values = proof
        .verify(vk)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM main relation verify failed: {err}")))?;
    verify_public_io(public_statement_digest, relation, &public_values)
}

fn build_main_relation_circuit(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalProof,
) -> Result<Rv64imMainRelationCircuit, SimpleKernelError> {
    let trace = build_main_relation_trace(statement, proof)?;
    let (params, _, structure) = rv64im_cached_root_main_lane_context()?;
    let optimized_cache = rv64im_cached_root_main_lane_optimized_cache()?;
    let dims = build_dims_and_policy(params, structure)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM main relation dims failed: {err}")))?;
    let mat_digest_vec = digest_ccs_matrices_with_sparse_cache(structure, Some(optimized_cache.sparse()));
    let mat_digest: [Goldilocks; 4] = mat_digest_vec
        .try_into()
        .map_err(|_| SimpleKernelError::Bridge("RV64IM main relation matrix digest length mismatch".into()))?;
    Ok(Rv64imMainRelationCircuit {
        public_statement_digest: statement.public_statement_digest,
        params: params.clone(),
        structure: structure.clone(),
        dims,
        mat_digest,
        trace,
    })
}

fn rv64im_compact_main_decider_shape_digest(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalProof,
) -> Result<[u8; 32], SimpleKernelError> {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/main_relation_spartan/compact_shape");
    tr.append_message(
        b"neo.fold.next/rv64im/main_relation_spartan/compact_shape/statement_digest",
        &statement.digest,
    );
    tr.append_message(
        b"neo.fold.next/rv64im/main_relation_spartan/compact_shape/proof_digest",
        &proof.proof_digest,
    );
    Ok(tr.digest32())
}

fn build_main_relation_trace(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalProof,
) -> Result<Rv64imMainRelationTrace, SimpleKernelError> {
    validate_final_statement_against_kernel_export(statement, &proof.kernel_export)?;
    let verified_kernel =
        verify_rv64im_kernel_export_proof_with_output(statement.folded.kernel_relation_digest, &proof.kernel_export)?;
    if proof.steps.len() != verified_kernel.chunk_handoffs.len() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main relation replay witness count does not match kernel export chunk count".into(),
        ));
    }
    if proof.chunk_summaries.len() != verified_kernel.chunk_handoffs.len() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main relation chunk summary count does not match kernel export chunk count".into(),
        ));
    }
    let (params, log, structure) = rv64im_cached_root_main_lane_context()?;
    let ce_structure = pad_ccs_structure_to_block_width(structure)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM padded CE structure failed: {err}")))?;
    let optimized_cache = rv64im_cached_root_main_lane_optimized_cache()?;
    let mut transcript = Poseidon2Transcript::new_raw_fields(&[F::from_u64(RV64IM_SESSION_RAW_DOMAIN_TAG)]);
    let dims = build_dims_and_policy(params, structure)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM main relation dims failed: {err}")))?;
    let mat_digest_vec = digest_ccs_matrices_with_sparse_cache(structure, Some(optimized_cache.sparse()));
    let mat_digest: [Goldilocks; 4] = mat_digest_vec
        .try_into()
        .map_err(|_| SimpleKernelError::Bridge("RV64IM main relation matrix digest length mismatch".into()))?;
    let mut incoming_main = crate::proof::Carry::default();
    let mut traces = Vec::with_capacity(proof.steps.len());

    for (chunk_idx, (handoff, step)) in verified_kernel
        .chunk_handoffs
        .iter()
        .zip(proof.steps.iter())
        .enumerate()
    {
        let fresh_claims = handoff
            .chunk_input
            .steps
            .iter()
            .map(|step| step.mcs.clone())
            .collect::<Vec<_>>();
        let fresh_witnesses = handoff
            .chunk_input
            .steps
            .iter()
            .map(|step| step.witness.clone())
            .collect::<Vec<_>>();
        let mut verify_transcript = transcript.clone();
        let (verified_next_main, verified_public_chunk_digest, _verified_chunk_relation_digest) =
            verify_rv64im_chunk_relation_with_replay(
                chunk_idx,
                handoff,
                &incoming_main,
                &step.replay_witness,
                &mut verify_transcript,
                params,
                structure,
                log,
                &optimized_cache,
            )?;
        let mut replay_transcript = transcript.clone();
        if handoff.public_chunk.steps.len() == 1 {
            replay_transcript.append_fields_raw(&[
                F::from_u64(STEP_INDEX_RAW_TAG),
                F::from_u64(handoff.public_chunk.start_index as u64),
            ]);
        } else {
            replay_transcript.append_fields_raw(&[
                F::from_u64(CHUNK_META_RAW_TAG),
                F::from_u64(handoff.public_chunk.start_index as u64),
                F::from_u64(handoff.public_chunk.steps.len() as u64),
            ]);
        }
        let replay_challenges = derive_replay_challenges_from_rounds(
            &mut replay_transcript,
            params,
            structure,
            dims,
            &mat_digest,
            &fresh_claims,
            &incoming_main.claims,
            &step.replay_witness.ccs_replay_proof,
            handoff.public_chunk_instance_digest,
        )?;
        let trace = trace_rv64im_chunk_relation_with_replay(
            chunk_idx,
            handoff,
            &incoming_main,
            &step.replay_witness,
            &mut transcript,
            params,
            structure,
            log,
            &optimized_cache,
        )?;
        if trace.ccs_outputs != trace.terminal_state.me_outputs {
            return Err(SimpleKernelError::Bridge(format!(
                "RV64IM main relation chunk {chunk_idx} replay outputs do not match the terminal state outputs"
            )));
        }
        check_claim_fold_digest_native(
            &trace.ccs_outputs,
            &trace.parent,
            &trace.children,
            &trace.terminal_state.fold_digest,
        )
        .map_err(|err| {
            SimpleKernelError::Bridge(format!(
                "RV64IM main relation chunk {chunk_idx} fold-digest binding failed: {err}"
            ))
        })?;
        check_output_binding_native(
            structure,
            &fresh_claims,
            &incoming_main.claims,
            &trace.ccs_outputs,
            &replay_challenges.row_chals,
            &replay_challenges.s_col,
        )
        .map_err(|err| {
            SimpleKernelError::Bridge(format!(
                "RV64IM main relation chunk {chunk_idx} output binding failed: {err}"
            ))
        })?;
        if trace.ccs_replay_proof.header_digest != trace.terminal_state.fold_digest {
            return Err(SimpleKernelError::Bridge(format!(
                "RV64IM main relation chunk {chunk_idx} replay header digest does not match the terminal fold digest"
            )));
        }
        let mut ccs_output_zs = fresh_witnesses
            .iter()
            .map(|witness| witness.Z.clone())
            .collect::<Vec<_>>();
        ccs_output_zs.extend(incoming_main.witnesses.iter().cloned());
        if trace.ccs_outputs.len() != ccs_output_zs.len() {
            return Err(SimpleKernelError::Bridge(format!(
                "RV64IM main relation chunk {chunk_idx} output/witness arity mismatch"
            )));
        }
        for (output_idx, (claim, z_matrix)) in trace
            .ccs_outputs
            .iter()
            .zip(ccs_output_zs.iter())
            .enumerate()
        {
            check_output_claim_consistency(params, structure, &ce_structure, claim, z_matrix)
            .map_err(|err| {
                SimpleKernelError::Bridge(format!(
                    "RV64IM main relation chunk {chunk_idx} backend consistency failed for ccs_output {output_idx}: {err}"
                ))
            })?;
        }
        if verified_public_chunk_digest != handoff.public_chunk_digest {
            return Err(SimpleKernelError::Bridge(format!(
                "RV64IM main relation chunk {chunk_idx} verified public chunk digest mismatch"
            )));
        }
        if verified_next_main.claims != trace.children || verified_next_main.witnesses != trace.z_split {
            return Err(SimpleKernelError::Bridge(format!(
                "RV64IM main relation chunk {chunk_idx} trace/verify next-main mismatch"
            )));
        }
        for (child_idx, (claim, z_matrix)) in trace.children.iter().zip(trace.z_split.iter()).enumerate() {
            check_dec_child_claim_consistency(params, structure, &ce_structure, log, claim, z_matrix).map_err(
                |err| {
                    SimpleKernelError::Bridge(format!(
                        "RV64IM main relation chunk {chunk_idx} child {child_idx} backend consistency failed: {err}"
                    ))
                },
            )?;
        }
        incoming_main = crate::proof::Carry {
            claims: trace.children.clone(),
            witnesses: trace.z_split.clone(),
        };
        traces.push(Rv64imMainRelationChunkTrace {
            handoff: Rv64imMainRelationHandoff {
                public_chunk: handoff.public_chunk.clone(),
                public_chunk_instance_digest: handoff.public_chunk_instance_digest,
                public_chunk_digest: handoff.public_chunk_digest,
                bridge_handoff_digest: handoff.bridge_handoff.digest,
                chunk_relation_digest: proof.chunk_summaries[chunk_idx].chunk_relation_digest,
            },
            fresh_claims,
            fresh_witnesses,
            replay_public_challenges: replay_challenges.public_challenges,
            replay_row_chals: replay_challenges.row_chals,
            replay_alpha_prime: replay_challenges.alpha_prime,
            replay_s_col: replay_challenges.s_col,
            replay_alpha_prime_nc: replay_challenges.alpha_prime_nc,
            ccs_trace: trace,
        });
        transcript.append_fields_raw(&[F::from_u64(RV64IM_CHUNK_DONE_RAW_TAG), F::ONE]);
    }

    if incoming_main.claims != statement.folded.final_accumulator.final_main_claims {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main relation final carried claims do not match the final statement".into(),
        ));
    }

    Ok(Rv64imMainRelationTrace {
        statement: statement.clone(),
        chunk_traces: traces,
    })
}

fn check_output_binding_native(
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Commitment, F>],
    me_inputs: &[neo_ccs::CeClaim<Commitment, F, K>],
    me_outputs: &[neo_ccs::CeClaim<Commitment, F, K>],
    r_prime: &[K],
    s_col_prime: &[K],
) -> Result<(), String> {
    if me_outputs.len() != fresh_claims.len() + me_inputs.len() {
        return Err("output arity mismatch".into());
    }

    for (idx, output) in me_outputs.iter().enumerate() {
        if output.r != r_prime {
            return Err(format!("output {idx} r mismatch"));
        }
        if output.s_col != s_col_prime {
            return Err(format!("output {idx} s_col mismatch"));
        }
        for matrix_idx in 0..structure.t() {
            if output.ct.get(matrix_idx).copied() != output.y_ring[matrix_idx].first().copied() {
                return Err(format!("output {idx} ct[{matrix_idx}] mismatch"));
            }
        }

        if idx < fresh_claims.len() {
            let fresh = &fresh_claims[idx];
            if output.c.data != fresh.c.data {
                return Err(format!("fresh output {idx} commitment mismatch"));
            }
            if output.m_in != fresh.m_in {
                return Err(format!("fresh output {idx} m_in mismatch"));
            }
            let expected_x = project_x_from_f_slice(&fresh.x, fresh.m_in)
                .map_err(|err| format!("fresh output {idx} X projection failed: {err}"))?;
            if output.X != expected_x {
                return Err(format!("fresh output {idx} X mismatch"));
            }
        } else {
            let me_idx = idx - fresh_claims.len();
            let input = &me_inputs[me_idx];
            if output.c.data != input.c.data {
                return Err(format!("me_input output {me_idx} commitment mismatch"));
            }
            if output.X != input.X {
                return Err(format!("me_input output {me_idx} X mismatch"));
            }
        }
    }

    Ok(())
}

fn check_claim_fold_digest_native(
    outputs: &[neo_ccs::CeClaim<Commitment, F, K>],
    parent: &neo_ccs::CeClaim<Commitment, F, K>,
    children: &[neo_ccs::CeClaim<Commitment, F, K>],
    terminal_fold_digest: &[u8; 32],
) -> Result<(), String> {
    for (idx, claim) in outputs.iter().enumerate() {
        if &claim.fold_digest != terminal_fold_digest {
            return Err(format!("ccs_output {idx} fold digest mismatch"));
        }
    }
    if &parent.fold_digest != terminal_fold_digest {
        return Err("parent fold digest mismatch".into());
    }
    for (idx, claim) in children.iter().enumerate() {
        if &claim.fold_digest != terminal_fold_digest {
            return Err(format!("child {idx} fold digest mismatch"));
        }
    }
    Ok(())
}

fn project_x_from_f_slice(values: &[F], m_in: usize) -> Result<Mat<F>, String> {
    if values.len() != m_in {
        return Err("x length mismatch".into());
    }
    let mut projected = Mat::zero(D, m_in, F::ZERO);
    for (col, value) in values.iter().copied().enumerate() {
        projected[(col % D, col)] = value;
    }
    Ok(projected)
}

fn check_output_claim_consistency(
    params: &NeoParams,
    base_structure: &CcsStructure<F>,
    ring_structure: &CcsStructure<F>,
    claim: &neo_ccs::CeClaim<Commitment, F, K>,
    z_matrix: &neo_ccs::Mat<F>,
) -> Result<(), String> {
    if !(claim.s_col.is_empty() && claim.y_zcol.is_empty()) {
        let chi_s = neo_ccs::tensor_point::<K>(&claim.s_col);
        let y_zcol = compute_y_zcol_from_witness_digits(params, z_matrix, base_structure.m, &chi_s, claim.y_zcol.len())
            .map_err(|err| err.to_string())?;
        if y_zcol != claim.y_zcol {
            return Err("y_zcol != Z_digits · χ_{s_col}".into());
        }
    }

    let z_coeffs =
        decode_superneo_coeffs_from_witness_mat(z_matrix, base_structure.m).map_err(|err| err.to_string())?;
    let ring_forms = build_superneo_ring_forms(ring_structure, &claim.r).map_err(|err| err.to_string())?;
    for (matrix_idx, forms) in ring_forms.iter().enumerate() {
        let mut row = vec![K::ZERO; claim.y_ring[matrix_idx].len()];
        for logical_col in 0..forms.len() {
            for rho in 0..D {
                row[rho] += forms[logical_col][rho] * z_coeffs[logical_col];
            }
        }
        if row != claim.y_ring[matrix_idx] {
            return Err(format!("y_ring[{matrix_idx}] mismatch"));
        }
        if claim.ct.get(matrix_idx).copied() != row.first().copied() {
            return Err(format!("ct[{matrix_idx}] mismatch"));
        }
    }

    Ok(())
}

fn check_dec_child_claim_consistency(
    params: &NeoParams,
    base_structure: &CcsStructure<F>,
    ring_structure: &CcsStructure<F>,
    log: &neo_ajtai::AjtaiSModule,
    claim: &neo_ccs::CeClaim<Commitment, F, K>,
    z_matrix: &neo_ccs::Mat<F>,
) -> Result<(), String> {
    if log.commit(z_matrix) != claim.c {
        return Err("c != L(Z)".into());
    }

    let z_coeffs =
        decode_superneo_coeffs_from_witness_mat(z_matrix, base_structure.m).map_err(|err| err.to_string())?;
    let max_digit = i128::from(params.b) - 1;
    for (logical_col, coeff) in z_coeffs.iter().enumerate() {
        let coeffs = coeff.as_coeffs();
        if coeffs[1] != F::ZERO {
            return Err(format!("child logical_col={logical_col} has non-base coefficient"));
        }
        let centered = to_balanced_i128(coeffs[0]);
        if centered.abs() > max_digit {
            return Err(format!(
                "child logical_col={logical_col} is outside the balanced digit alphabet"
            ));
        }
    }

    if !(claim.s_col.is_empty() && claim.y_zcol.is_empty()) {
        let chi_s = neo_ccs::tensor_point::<K>(&claim.s_col);
        let y_zcol = compute_y_zcol_from_witness(params, z_matrix, base_structure.m, &chi_s, claim.y_zcol.len())
            .map_err(|err| err.to_string())?;
        if y_zcol != claim.y_zcol {
            return Err("y_zcol != Z · χ_{s_col}".into());
        }
    }

    let ring_forms = build_superneo_ring_forms(ring_structure, &claim.r).map_err(|err| err.to_string())?;
    for (matrix_idx, forms) in ring_forms.iter().enumerate() {
        let mut row = vec![K::ZERO; claim.y_ring[matrix_idx].len()];
        for logical_col in 0..forms.len() {
            for rho in 0..D {
                row[rho] += forms[logical_col][rho] * z_coeffs[logical_col];
            }
        }
        if row != claim.y_ring[matrix_idx] {
            return Err(format!("y_ring[{matrix_idx}] mismatch"));
        }
        if claim.ct.get(matrix_idx).copied() != row.first().copied() {
            return Err(format!("ct[{matrix_idx}] mismatch"));
        }
    }

    Ok(())
}

struct DerivedReplayChallenges {
    public_challenges: neo_reductions::optimized_engine::Challenges,
    row_chals: Vec<K>,
    alpha_prime: Vec<K>,
    s_col: Vec<K>,
    alpha_prime_nc: Vec<K>,
}

#[allow(clippy::too_many_arguments)]
fn derive_replay_challenges_from_rounds(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    dims: Dims,
    mat_digest: &[Goldilocks; 4],
    fresh_claims: &[CcsClaim<Commitment, F>],
    me_inputs: &[neo_ccs::CeClaim<Commitment, F, K>],
    replay_proof: &neo_reductions::optimized_engine::PiCcsReplayProofWitness,
    public_instance_digest: [F; 4],
) -> Result<DerivedReplayChallenges, SimpleKernelError> {
    neo_reductions::engines::utils::bind_header_and_instance_digest_with_digest(
        tr,
        params,
        structure,
        dims,
        mat_digest,
        &public_instance_digest,
    )
    .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM replay challenge header binding failed: {err}")))?;
    neo_reductions::engines::utils::bind_me_inputs(tr, me_inputs)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM replay challenge ME binding failed: {err}")))?;
    let mut public_challenges = neo_reductions::engines::utils::sample_challenges(tr, dims.ell_d, dims.ell)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM replay challenge public sampling failed: {err}")))?;
    public_challenges.beta_m = neo_reductions::engines::utils::sample_beta_m(tr, dims.ell_m)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM replay challenge beta_m sampling failed: {err}")))?;

    tr.append_fields_raw(&[F::from_u64(PI_CCS_SUMCHECK_FE_RAW_DOMAIN_TAG)]);
    let initial_sum = neo_reductions::optimized_engine::claimed_initial_sum_from_inputs_with_k_mcs(
        structure,
        &public_challenges,
        fresh_claims.len(),
        me_inputs,
    );
    tr.append_fields_raw(&[F::from_u64(PI_CCS_SUMCHECK_INITIAL_RAW_TAG)]);
    tr.append_fields_raw(&initial_sum.as_coeffs());
    tr.append_fields_raw(&[F::from_u64(
        neo_reductions::sumcheck::SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG,
    )]);
    let (fe_all, _, fe_ok) = neo_reductions::sumcheck::verify_sumcheck_rounds_poseidon_v3(
        tr,
        dims.d_sc,
        initial_sum,
        &replay_proof.sumcheck_rounds,
    );
    if !fe_ok {
        return Err(SimpleKernelError::Bridge(
            "RV64IM replay challenge derivation failed: FE rounds invalid".into(),
        ));
    }
    let (row_chals, alpha_prime) = fe_all.split_at(dims.ell_n);

    tr.append_fields_raw(&[F::from_u64(PI_CCS_SUMCHECK_NC_RAW_DOMAIN_TAG)]);
    let initial_sum_nc = K::ZERO;
    tr.append_fields_raw(&[F::from_u64(PI_CCS_SUMCHECK_INITIAL_RAW_TAG)]);
    tr.append_fields_raw(&initial_sum_nc.as_coeffs());
    tr.append_fields_raw(&[F::from_u64(
        neo_reductions::sumcheck::SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG,
    )]);
    let (nc_all, _, nc_ok) = neo_reductions::sumcheck::verify_sumcheck_rounds_poseidon_v3(
        tr,
        dims.d_sc,
        initial_sum_nc,
        &replay_proof.sumcheck_rounds_nc,
    );
    if !nc_ok {
        return Err(SimpleKernelError::Bridge(
            "RV64IM replay challenge derivation failed: NC rounds invalid".into(),
        ));
    }
    let (s_col, alpha_prime_nc) = nc_all.split_at(dims.ell_m);
    let fold_digest = tr.digest32();
    if fold_digest != replay_proof.header_digest {
        return Err(SimpleKernelError::Bridge(
            "RV64IM replay challenge derivation failed: replay header digest mismatch".into(),
        ));
    }

    Ok(DerivedReplayChallenges {
        public_challenges,
        row_chals: row_chals.to_vec(),
        alpha_prime: alpha_prime.to_vec(),
        s_col: s_col.to_vec(),
        alpha_prime_nc: alpha_prime_nc.to_vec(),
    })
}

fn validate_final_statement_against_kernel_export(
    statement: &Rv64imFinalStatement,
    kernel_export: &Rv64imKernelExportProof,
) -> Result<Rv64imKernelExportRelationResult, SimpleKernelError> {
    if statement.folded.digest != folded_statement_digest(&statement.folded) {
        return Err(SimpleKernelError::Bridge(
            "RV64IM folded statement digest mismatch".into(),
        ));
    }
    if statement.digest != final_statement_digest(statement) {
        return Err(SimpleKernelError::Bridge(
            "RV64IM final statement digest mismatch".into(),
        ));
    }
    if statement.public_statement_digest != kernel_export.public_statement_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM final statement public digest does not match the kernel export proof".into(),
        ));
    }
    let verified_kernel =
        verify_rv64im_kernel_export_proof_with_output(statement.folded.kernel_relation_digest, kernel_export)?;
    if statement.folded.fold_schedule != verified_kernel.fold_schedule {
        return Err(SimpleKernelError::Bridge(
            "RV64IM folded statement schedule does not match the verified kernel export".into(),
        ));
    }
    if statement.folded.chunk_count as usize != verified_kernel.chunk_handoffs.len() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM folded statement chunk count does not match the verified kernel export".into(),
        ));
    }
    let semantic_step_count = verified_kernel
        .chunk_handoffs
        .iter()
        .map(|handoff| handoff.public_chunk.steps.len() as u64)
        .sum::<u64>();
    if statement.folded.semantic_step_count != semantic_step_count {
        return Err(SimpleKernelError::Bridge(
            "RV64IM folded statement semantic step count does not match the verified kernel export".into(),
        ));
    }
    Ok(verified_kernel)
}

fn verify_public_io(
    public_statement_digest: [u8; 32],
    relation: &Rv64imDeciderRelation,
    public_values: &[SpartanF],
) -> Result<(), SimpleKernelError> {
    let mut expected = Vec::new();
    expected.extend(
        digest32_as_fields(public_statement_digest)
            .into_iter()
            .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64())),
    );
    for summary in &relation.chunk_summaries {
        expected.extend(
            digest32_as_fields(summary.chunk_relation_digest)
                .into_iter()
                .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64())),
        );
    }
    if expected != public_values {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main relation public IO mismatch".into(),
        ));
    }
    Ok(())
}

fn alloc_rounds<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    rounds: &[Vec<K>],
    label: &str,
) -> Result<Vec<Vec<KNumVar>>, SynthesisError> {
    rounds
        .iter()
        .enumerate()
        .map(|(round_idx, round)| {
            round
                .iter()
                .enumerate()
                .map(|(coeff_idx, coeff)| {
                    alloc_constant_k(
                        cs,
                        KNum::from_neo_k(*coeff),
                        &format!("{label}_{round_idx}_{coeff_idx}"),
                    )
                })
                .collect()
        })
        .collect()
}

fn max_degree(rounds: &[Vec<K>]) -> usize {
    rounds
        .iter()
        .map(|round| round.len().saturating_sub(1))
        .max()
        .unwrap_or(0)
}

fn chunk_sumcheck_challenges(prefix: &[K], suffix: &[K]) -> Vec<K> {
    let mut out = Vec::with_capacity(prefix.len() + suffix.len());
    out.extend_from_slice(prefix);
    out.extend_from_slice(suffix);
    out
}

fn append_chunk_meta<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    transcript: &mut Poseidon2TranscriptCircuit,
    handoff: &Rv64imMainRelationHandoff,
) -> Result<(), SynthesisError> {
    if handoff.public_chunk.steps.len() == 1 {
        transcript.append_const_fields_raw(
            cs.namespace(|| "step_index"),
            &[
                SpartanF::from_canonical_u64(STEP_INDEX_RAW_TAG),
                SpartanF::from_canonical_u64(handoff.public_chunk.start_index as u64),
            ],
        )
    } else {
        transcript.append_const_fields_raw(
            cs.namespace(|| "chunk_meta"),
            &[
                SpartanF::from_canonical_u64(CHUNK_META_RAW_TAG),
                SpartanF::from_canonical_u64(handoff.public_chunk.start_index as u64),
                SpartanF::from_canonical_u64(handoff.public_chunk.steps.len() as u64),
            ],
        )
    }
}

fn next_public_digest(
    public_inputs: &[AllocatedNum<SpartanF>],
    cursor: &mut usize,
    label: &str,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    if *cursor + 4 > public_inputs.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    let out = core::array::from_fn(|idx| public_inputs[*cursor + idx].clone());
    *cursor += 4;
    let _ = label;
    Ok(out)
}

fn enforce_digest_eq<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    actual: &[AllocatedNum<SpartanF>; 4],
    expected: &[AllocatedNum<SpartanF>; 4],
    label: &str,
) -> Result<(), SynthesisError> {
    for (idx, (lhs, rhs)) in actual.iter().zip(expected.iter()).enumerate() {
        cs.enforce(
            || format!("{label}_{idx}"),
            |lc| lc + lhs.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + rhs.get_variable(),
        );
    }
    Ok(())
}

fn split_vec<T: Clone>(values: &[T], prefix_len: usize) -> Result<(Vec<T>, Vec<T>), SynthesisError> {
    if prefix_len > values.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    Ok((values[..prefix_len].to_vec(), values[prefix_len..].to_vec()))
}

fn chunk_relation_digest_circuit<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    public_chunk_digest: [u8; 32],
    main_relation_digest: &[AllocatedNum<SpartanF>; 4],
    _main_relation_digest_values: &[SpartanF; 4],
    bridge_handoff_digest: [u8; 32],
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let mut preimage = Vec::with_capacity(1 + 3 * 4);
    preimage.extend(alloc_const_field_values(
        cs,
        &[SpartanF::from_canonical_u64(RV64IM_CHUNK_RELATION_DIGEST_RAW_TAG)],
        "chunk_relation_digest_domain",
    )?);
    preimage.extend(alloc_const_field_values(
        cs,
        &digest32_as_spartan_fields(public_chunk_digest),
        "chunk_relation_digest_public_chunk",
    )?);
    preimage.extend(main_relation_digest.iter().cloned());
    preimage.extend(alloc_const_field_values(
        cs,
        &digest32_as_spartan_fields(bridge_handoff_digest),
        "chunk_relation_digest_bridge",
    )?);
    hash_packed_goldilocks_fields(cs.namespace(|| "chunk_relation_digest_hash"), &preimage)
}

fn alloc_const_field_values<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    values: &[SpartanF],
    label: &str,
) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
    values
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            let out = AllocatedNum::alloc(cs.namespace(|| format!("{label}_{idx}")), || Ok(*value))?;
            cs.enforce(
                || format!("{label}_{idx}_const"),
                |lc| lc + out.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + (*value, CS::one()),
            );
            Ok(out)
        })
        .collect()
}

fn digest32_as_spartan_fields(digest: [u8; 32]) -> [SpartanF; 4] {
    digest32_as_fields(digest).map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()))
}

fn chunk_relation_digest_values(
    public_chunk_digest: [u8; 32],
    main_relation_digest: [SpartanF; 4],
    bridge_handoff_digest: [u8; 32],
) -> [SpartanF; 4] {
    let mut preimage = Vec::with_capacity(1 + 3 * 4);
    preimage.push(F::from_u64(RV64IM_CHUNK_RELATION_DIGEST_RAW_TAG));
    preimage.extend(digest32_as_fields(public_chunk_digest));
    preimage.extend(main_relation_digest.map(|value| F::from_u64(value.to_canonical_u64())));
    preimage.extend(digest32_as_fields(bridge_handoff_digest));
    neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash(&preimage)
        .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()))
}
