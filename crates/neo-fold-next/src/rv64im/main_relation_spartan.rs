//! Owns the direct Spartan proof for the full RV64IM main relation.
//!
//! This module compiles the route-owned `R_main^SN` witness relation directly.
//! It does not route theorem meaning through the generic shell target.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use neo_ccs::{CcsStructure, CeClaim, Mat};
use neo_math::{KExtensions, F, K};
use neo_params::NeoParams;
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
use crate::rv64im::chunk_relation::RV64IM_CHUNK_RELATION_DIGEST_RAW_TAG;
use crate::rv64im::final_relation::{
    Rv64imChunkTransitionWitness, Rv64imFinalBuildProof, Rv64imFinalProofComponentDigests, Rv64imFinalStatement,
    RV64IM_CHUNK_DONE_RAW_TAG, RV64IM_SESSION_RAW_DOMAIN_TAG,
};
use crate::rv64im::kernel::{
    rv64im_cached_root_main_lane_context, rv64im_cached_root_main_lane_optimized_cache, Rv64imKernelExportProof,
    SimpleKernelError,
};
use crate::rv64im::main_relation::{validate_rv64im_decider_relation_surface, Rv64imDeciderRelation};
use crate::rv64im::main_relation_circuit::claim::{
    alloc_ce_claim, alloc_ce_claim_public_surface_with_shared_point, alloc_ce_claim_with_shared_point, CeClaimVar,
};
use crate::rv64im::main_relation_circuit::initial_sum::claimed_initial_sum_from_me_inputs;
use crate::rv64im::main_relation_circuit::k_field::{alloc_constant_k, alloc_k, KNum, KNumVar};
use crate::rv64im::main_relation_circuit::output_binding::enforce_me_outputs_against_inputs;
use crate::rv64im::main_relation_circuit::pi_ccs::{
    bind_header_and_instance_digest, bind_me_inputs, sample_challenges,
};
use crate::rv64im::main_relation_circuit::pi_dec::enforce_dec_public;
use crate::rv64im::main_relation_circuit::pi_rlc::{
    enforce_rlc_dec_public_with_rho_coeffs_for_last_chunk, enforce_rlc_public_with_rho_vars_constant_prefix,
};
use crate::rv64im::main_relation_circuit::rho_sampling::{
    alloc_zero_rot_rho_matrices, alloc_zero_rot_rhos, materialize_goldilocks_rot_matrices, sample_goldilocks_rot_rhos,
};
use crate::rv64im::main_relation_circuit::sumcheck::{sumcheck_eval_gadget, sumcheck_round_gadget};
use crate::rv64im::main_relation_circuit::sumcheck_replay::verify_sumcheck_rounds;
use crate::rv64im::main_relation_circuit::terminal_identity::{
    enforce_terminal_identity_fe, enforce_terminal_identity_nc,
};
use crate::rv64im::main_relation_circuit::transcript::Poseidon2TranscriptCircuit;
use crate::rv64im::main_relation_trace::{
    build_rv64im_main_circuit_trace_from_setup_shape, build_rv64im_main_circuit_trace_from_step_components,
    build_rv64im_main_relation_setup_shape_from_step_components, Rv64imMainCircuitCeClaimShape,
    Rv64imMainCircuitChunkCover, Rv64imMainCircuitChunkReplaySurface, Rv64imMainCircuitChunkTrace,
    Rv64imMainCircuitHandoff, Rv64imMainCircuitTrace, CHUNK_META_RAW_TAG, STEP_INDEX_RAW_TAG,
};
mod chunk_diagnostics;
mod chunk_step_ivc;
mod chunk_step_recursive;
mod debug;
mod fingerprint_cs;
mod fixed_transcript;
mod nifs_v_stages;
mod recursive_cover;
mod recursive_step;
mod step_statement;

const RV64IM_MAIN_RELATION_DELTA: u64 = 7;
pub type Rv64imSpartan2DeciderEngine = GoldilocksP3MerkleMleEngine;
pub type Rv64imSpartan2DeciderSnark = R1CSSNARK<Rv64imSpartan2DeciderEngine>;
pub type Rv64imSpartan2DeciderProverKey = spartan2::spartan::SpartanProverKey<Rv64imSpartan2DeciderEngine>;
pub type Rv64imSpartan2DeciderVerifierKey = spartan2::spartan::SpartanVerifierKey<Rv64imSpartan2DeciderEngine>;
pub type Rv64imSpartan2DeciderKeyPair = Arc<(Rv64imSpartan2DeciderProverKey, Rv64imSpartan2DeciderVerifierKey)>;

static RV64IM_MAIN_RELATION_SETUP_CACHE: OnceLock<Mutex<HashMap<[u8; 32], Rv64imSpartan2DeciderKeyPair>>> =
    OnceLock::new();

#[allow(unused_imports)]
pub use chunk_diagnostics::debug_measure_rv64im_main_relation_state_in_prefix_fingerprints;
pub(crate) use chunk_diagnostics::{
    debug_locate_rv64im_main_relation_chunk_stage, debug_profile_rv64im_main_relation_chunk_stage_progress,
};

pub use crate::rv64im::main_relation_trace::Rv64imMainRelationSetupShape;
pub use chunk_step_ivc::{
    build_rv64im_chunk_step_ivc_recursive_step_cover_shape, build_rv64im_chunk_step_ivc_recursive_step_padding,
    build_rv64im_chunk_step_ivc_recursive_step_padding_from_shape, build_rv64im_chunk_step_ivc_shape,
    prove_rv64im_chunk_step_ivc_spartan, prove_rv64im_chunk_step_ivc_spartan_chain,
    setup_rv64im_chunk_step_ivc_spartan, setup_rv64im_chunk_step_ivc_spartan_cached,
    verify_rv64im_chunk_step_ivc_spartan, verify_rv64im_chunk_step_ivc_spartan_chain,
    Rv64imChunkStepIvcRecursiveStepPadding, Rv64imChunkStepIvcShape, Rv64imChunkStepIvcSpartanChainProof,
    Rv64imChunkStepIvcSpartanError, Rv64imChunkStepIvcSpartanKeyPair, Rv64imChunkStepIvcSpartanProof,
    Rv64imChunkStepIvcSpartanProverKey, Rv64imChunkStepIvcSpartanVerifierKey,
};
pub(crate) use chunk_step_ivc::{
    prove_rv64im_chunk_step_ivc_spartan_compressed_chain, verify_rv64im_chunk_step_ivc_spartan_compressed_chain,
    Rv64imChunkStepIvcSpartanCompressedChainProof,
};
pub use chunk_step_recursive::{
    build_rv64im_main_recursion_f_prime_backend_relations,
    build_rv64im_main_recursion_f_prime_backend_relations_with_spartan_shape,
    build_rv64im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices,
    build_rv64im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices_and_perf,
    build_rv64im_main_recursion_f_prime_claim_cover, build_rv64im_main_recursion_f_prime_payload,
    build_rv64im_main_recursion_f_prime_payloads, build_rv64im_main_recursion_f_prime_payloads_with_spartan_shape,
    build_rv64im_main_recursion_step_spartan_shape,
    debug_check_rv64im_chunk_step_recursive_effective_chunk_trace_matches_native,
    debug_check_rv64im_main_recursion_f_prime_backend_relation_semantics,
    debug_trace_rv64im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices, Rv64imCcsClaimShape,
    Rv64imCcsWitnessShape, Rv64imCeClaimDigestShape, Rv64imMainRecursionFPrimeBackendRelation,
    Rv64imMainRecursionFPrimeBackendRelationBuildPerf, Rv64imMainRecursionFPrimeClaimCover,
    Rv64imMainRecursionFPrimePayload, Rv64imMainRecursionStepSpartanShape,
};
use debug::append_k_to_transcript;
pub use debug::{
    debug_check_rv64im_spartan2_decider_circuit, inspect_rv64im_spartan2_decider_trace,
    measure_rv64im_spartan2_decider_circuit, Rv64imMainRelationCircuitMetrics, Rv64imMainRelationCountBucket,
    Rv64imMainRelationHotspotDetail, Rv64imMainRelationPhaseBucket, Rv64imMainRelationSurfaceFamilyBucket,
    Rv64imMainRelationSurfaceMetrics, Rv64imMainRelationTraceStats,
};
use nifs_v_stages::{
    enforce_outer_chunk_relation_public_io, enforce_synthetic_outer_chunk_relation_public_io, synthesize_pi_ccs_stage,
    synthesize_pi_dec_stage, synthesize_rv64im_chunk_nifs_verifier_body,
    synthesize_rv64im_chunk_nifs_verifier_body_with_synthetic_chunk_relation_io, Rv64imChunkNifsVerifierCtx,
    Rv64imPiRlcStageOutput,
};
#[allow(unused_imports)]
pub use recursive_step::Rv64imMainRecursionStepChunkReplayFingerprint;
pub use recursive_step::{
    build_rv64im_main_recursion_step_authoritative_chunk_surface,
    build_rv64im_main_recursion_step_spartan_compressed_chain_shape,
    build_rv64im_main_recursion_step_spartan_published_target,
    debug_check_rv64im_main_recursion_step_authoritative_chunk_surface_matches_native,
    debug_check_rv64im_main_recursion_step_spartan_chunk_replay_surface,
    debug_check_rv64im_main_recursion_step_spartan_circuit,
    debug_check_rv64im_main_recursion_step_spartan_compressed_chain_circuit,
    debug_check_rv64im_main_recursion_step_spartan_compressed_chain_parity,
    debug_check_rv64im_main_recursion_step_spartan_compressed_chain_public_io,
    debug_check_rv64im_main_recursion_step_spartan_compressed_chain_shape_only_circuit,
    debug_check_rv64im_main_recursion_step_spartan_compressed_chain_shape_only_setup,
    debug_check_rv64im_main_recursion_step_spartan_compressed_chain_statement_binding,
    debug_check_rv64im_main_recursion_step_spartan_compressed_chain_wrapper_only,
    debug_check_rv64im_main_recursion_step_spartan_embedded_body,
    debug_check_rv64im_main_recursion_step_spartan_fresh_output_accumulator_digest_parity,
    debug_check_rv64im_main_recursion_step_spartan_inactive_side_lane_constraints,
    debug_check_rv64im_main_recursion_step_spartan_live_claim_me_digest_parity,
    debug_check_rv64im_main_recursion_step_spartan_pi_ccs_replay_lengths,
    debug_check_rv64im_main_recursion_step_spartan_shape_only_chain_parity,
    debug_check_rv64im_main_recursion_x_out_gadget_parity,
    debug_compare_rv64im_main_recursion_step_spartan_shape_only_skeleton,
    debug_measure_rv64im_main_recursion_step_chunk_replay_fingerprint,
    debug_measure_rv64im_main_recursion_step_shape_only_circuit_shape,
    debug_measure_rv64im_main_recursion_step_spartan_circuit_shape,
    debug_measure_rv64im_main_recursion_step_spartan_commitment_key,
    debug_measure_rv64im_main_recursion_step_spartan_compressed_chain_circuit_shape,
    debug_measure_rv64im_main_recursion_step_spartan_setup_equivalence,
    debug_measure_rv64im_main_recursion_step_spartan_shape_synthesis,
    debug_profile_rv64im_main_recursion_step_chunk_replay_stages,
    debug_profile_rv64im_main_recursion_step_spartan_compressed_chain_prove_stages,
    debug_trace_rv64im_main_recursion_step_spartan_shape_synthesis, prove_rv64im_main_recursion_step_spartan,
    prove_rv64im_main_recursion_step_spartan_chain, prove_rv64im_main_recursion_step_spartan_compressed_chain,
    setup_rv64im_main_recursion_step_spartan_cached, setup_rv64im_main_recursion_step_spartan_shape_cached,
    validate_rv64im_main_recursion_step_spartan_chain_shape, verify_rv64im_main_recursion_step_spartan,
    verify_rv64im_main_recursion_step_spartan_and_extract_published_target,
    verify_rv64im_main_recursion_step_spartan_chain,
    verify_rv64im_main_recursion_step_spartan_chain_and_extract_published_targets,
    verify_rv64im_main_recursion_step_spartan_compressed_chain,
    verify_rv64im_main_recursion_step_spartan_published_target,
    verify_rv64im_main_recursion_step_spartan_published_target_chain, Rv64imMainRecursionStepAuthoritativeChunkSurface,
    Rv64imMainRecursionStepSpartanChainProof, Rv64imMainRecursionStepSpartanCircuitShape,
    Rv64imMainRecursionStepSpartanCompressedChainProof, Rv64imMainRecursionStepSpartanCompressedChainProveMetrics,
    Rv64imMainRecursionStepSpartanCompressedChainShape, Rv64imMainRecursionStepSpartanError,
    Rv64imMainRecursionStepSpartanKeyPair, Rv64imMainRecursionStepSpartanProof,
    Rv64imMainRecursionStepSpartanProverKey, Rv64imMainRecursionStepSpartanPublishedTarget,
    Rv64imMainRecursionStepSpartanSetupEquivalence, Rv64imMainRecursionStepSpartanVerifierKey,
};
pub use step_statement::Rv64imMainRecursionStepSpartanStatement;

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
struct Rv64imMainRelationCircuit {
    public_statement_digest: [u8; 32],
    params: NeoParams,
    structure: CcsStructure<F>,
    dims: Dims,
    mat_digest: [Goldilocks; 4],
    trace: Rv64imMainCircuitTrace,
}

#[derive(Clone)]
pub(crate) struct Rv64imClaimBundle {
    claims: Vec<CeClaimVar>,
    effective_count: usize,
}

impl Rv64imClaimBundle {
    pub(crate) fn empty() -> Self {
        Self {
            claims: Vec::new(),
            effective_count: 0,
        }
    }

    pub(crate) fn from_effective_claims(claims: Vec<CeClaimVar>) -> Self {
        let effective_count = claims.len();
        Self {
            claims,
            effective_count,
        }
    }

    pub(crate) fn from_padded_claims(claims: Vec<CeClaimVar>, effective_count: usize) -> Self {
        debug_assert!(effective_count <= claims.len());
        Self {
            claims,
            effective_count,
        }
    }

    pub(crate) fn effective_claims(&self) -> &[CeClaimVar] {
        &self.claims[..self.effective_count]
    }

    pub(crate) fn effective_count(&self) -> usize {
        self.effective_count
    }

    pub(crate) fn into_effective_claims(self) -> Vec<CeClaimVar> {
        self.claims.into_iter().take(self.effective_count).collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Rv64imChunkBoundaryMode {
    Interior,
    TerminalPreserveIncoming,
    TerminalCarryChildren,
}

impl Rv64imChunkBoundaryMode {
    pub(crate) fn from_terminal_flags(is_terminal_chunk: bool, carry_terminal_children: bool) -> Self {
        match (is_terminal_chunk, carry_terminal_children) {
            (false, _) => Self::Interior,
            (true, false) => Self::TerminalPreserveIncoming,
            (true, true) => Self::TerminalCarryChildren,
        }
    }

    fn is_terminal(self) -> bool {
        !matches!(self, Self::Interior)
    }

    fn preserves_incoming_carry(self) -> bool {
        matches!(self, Self::TerminalPreserveIncoming)
    }

    fn uses_last_chunk_rlc_dec_shortcut(
        self,
        effective_fresh_claim_count: usize,
        effective_output_count: usize,
    ) -> bool {
        self.preserves_incoming_carry() && effective_fresh_claim_count == effective_output_count
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Rv64imChunkChildClaimSource {
    ReplayedChildren,
    TerminalFinalClaims,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Rv64imChunkNextCarryMode {
    ReplaceWithEffectiveChildren,
    PreserveIncoming,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Rv64imChunkRlcMode {
    Standard { constant_child_prefix: usize },
    TerminalLastChunkShortcut,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Rv64imChunkBoundaryPlan {
    pub(crate) child_claim_source: Rv64imChunkChildClaimSource,
    pub(crate) next_carry_mode: Rv64imChunkNextCarryMode,
    pub(crate) rlc_mode: Rv64imChunkRlcMode,
}

impl Rv64imChunkBoundaryPlan {
    pub(crate) fn from_boundary_mode(
        boundary_mode: Rv64imChunkBoundaryMode,
        effective_fresh_claim_count: usize,
        effective_output_count: usize,
    ) -> Self {
        let child_claim_source = if boundary_mode.is_terminal() {
            Rv64imChunkChildClaimSource::TerminalFinalClaims
        } else {
            Rv64imChunkChildClaimSource::ReplayedChildren
        };
        let next_carry_mode = if boundary_mode.preserves_incoming_carry() {
            Rv64imChunkNextCarryMode::PreserveIncoming
        } else {
            Rv64imChunkNextCarryMode::ReplaceWithEffectiveChildren
        };
        let rlc_mode =
            if boundary_mode.uses_last_chunk_rlc_dec_shortcut(effective_fresh_claim_count, effective_output_count) {
                Rv64imChunkRlcMode::TerminalLastChunkShortcut
            } else {
                Rv64imChunkRlcMode::Standard {
                    constant_child_prefix: 0,
                }
            };
        Self {
            child_claim_source,
            next_carry_mode,
            rlc_mode,
        }
    }
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

    fn synthesize_chunk<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        chunk_index: usize,
        cover_chunk: &Rv64imMainCircuitChunkCover,
        replay_chunk: &Rv64imMainCircuitChunkReplaySurface,
        public_inputs: &[AllocatedNum<SpartanF>],
        public_cursor: &mut usize,
        transcript: &mut Poseidon2TranscriptCircuit,
        carried_claims: Rv64imClaimBundle,
        boundary_plan: Rv64imChunkBoundaryPlan,
    ) -> Result<Rv64imClaimBundle, SynthesisError> {
        synthesize_rv64im_main_relation_chunk(
            &self.params,
            &self.structure,
            self.dims,
            &self.mat_digest,
            &self
                .trace
                .statement
                .folded
                .final_accumulator
                .final_main_claims,
            cs,
            chunk_index,
            cover_chunk,
            replay_chunk,
            public_inputs,
            public_cursor,
            transcript,
            carried_claims,
            boundary_plan,
            true,
            true,
        )
    }
}

pub(crate) fn synthesize_rv64im_main_relation_chunk<CS: ConstraintSystem<SpartanF>>(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    dims: Dims,
    mat_digest: &[Goldilocks; 4],
    terminal_final_claims: &[neo_ccs::CeClaim<neo_ajtai::Commitment, F, K>],
    cs: &mut CS,
    chunk_index: usize,
    cover_chunk: &Rv64imMainCircuitChunkCover,
    chunk: &Rv64imMainCircuitChunkReplaySurface,
    public_inputs: &[AllocatedNum<SpartanF>],
    public_cursor: &mut usize,
    transcript: &mut Poseidon2TranscriptCircuit,
    carried_claims: Rv64imClaimBundle,
    boundary_plan: Rv64imChunkBoundaryPlan,
    enforce_chunk_relation_public_io: bool,
    append_chunk_done: bool,
) -> Result<Rv64imClaimBundle, SynthesisError> {
    let next_carried_claims = synthesize_rv64im_chunk_nifs_verifier_body(
        params,
        structure,
        dims,
        mat_digest,
        terminal_final_claims,
        cs,
        chunk_index,
        cover_chunk,
        chunk,
        transcript,
        carried_claims,
        None,
        boundary_plan,
    )?;
    let ctx = Rv64imChunkNifsVerifierCtx {
        params,
        structure,
        dims,
        mat_digest,
        terminal_final_claims,
        chunk_index,
        cover_chunk,
        chunk,
        logical_me_input_claims: None,
        boundary_plan,
    };
    if enforce_chunk_relation_public_io {
        // The standalone chunk theorem binds the relation digest as public IO.
        // Recursive F' replay uses only the inner verifier body and must skip
        // this outer theorem wrapper.
        enforce_outer_chunk_relation_public_io(&ctx, cs, transcript, public_inputs, public_cursor)?;
    }
    if append_chunk_done {
        transcript.append_const_fields_raw(
            cs.namespace(|| format!("chunk_done_{chunk_index}")),
            &[
                SpartanF::from_canonical_u64(RV64IM_CHUNK_DONE_RAW_TAG),
                SpartanF::from_canonical_u64(1),
            ],
        )?;
    }
    Ok(next_carried_claims)
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
        let mut carried_claims = Rv64imClaimBundle::empty();

        for (chunk_idx, chunk) in self.trace.chunk_traces.iter().enumerate() {
            let cover_chunk = Rv64imMainCircuitChunkCover::from_trace(chunk);
            let replay_chunk = chunk
                .replay_surface()
                .map_err(|_| SynthesisError::Unsatisfiable)?;
            let boundary_plan = Rv64imChunkBoundaryPlan::from_boundary_mode(
                Rv64imChunkBoundaryMode::from_terminal_flags(chunk_idx + 1 == self.trace.chunk_traces.len(), false),
                chunk.fresh_claims.len(),
                chunk.ccs_trace.ccs_outputs.len(),
            );
            carried_claims = self.synthesize_chunk(
                &mut cs.namespace(|| format!("chunk_{chunk_idx}")),
                chunk_idx,
                &cover_chunk,
                &replay_chunk,
                &public_inputs,
                &mut public_cursor,
                &mut transcript,
                carried_claims,
                boundary_plan,
            )?;
        }

        if public_cursor != public_inputs.len() {
            return Err(SynthesisError::Unsatisfiable);
        }
        Ok(())
    }
}

pub fn build_rv64im_spartan2_decider_setup_shape_from_components(
    statement: &Rv64imFinalStatement,
    proof_digest: [u8; 32],
    kernel_export: &Rv64imKernelExportProof,
    chunk_summaries: &[crate::finalize::FixedShapeChunkSummary],
    steps: &[Rv64imChunkTransitionWitness],
) -> Result<Rv64imMainRelationSetupShape, SimpleKernelError> {
    let component_digests =
        crate::rv64im::final_relation::final_proof_component_digests_from_parts(kernel_export, steps);
    build_rv64im_main_relation_setup_shape_from_step_components(
        statement,
        proof_digest,
        kernel_export,
        chunk_summaries,
        steps,
        &component_digests,
    )
}

pub fn setup_rv64im_spartan2_decider_from_shape(
    shape: &Rv64imMainRelationSetupShape,
) -> Result<(Rv64imSpartan2DeciderProverKey, Rv64imSpartan2DeciderVerifierKey), SimpleKernelError> {
    let circuit = build_main_relation_circuit_from_setup_shape(shape)?;
    Rv64imSpartan2DeciderSnark::setup(circuit)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM main relation setup failed: {err}")))
}

pub fn setup_rv64im_spartan2_decider_cached_from_shape(
    shape: &Rv64imMainRelationSetupShape,
) -> Result<Rv64imSpartan2DeciderKeyPair, SimpleKernelError> {
    let circuit = build_main_relation_circuit_from_setup_shape(shape)?;
    let cache_key = rv64im_main_relation_setup_cache_key_from_shape(shape)?;
    let cache = RV64IM_MAIN_RELATION_SETUP_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(keys) = cache
        .lock()
        .map_err(|_| SimpleKernelError::Bridge("RV64IM main relation setup cache poisoned".into()))?
        .get(&cache_key)
        .cloned()
    {
        return Ok(keys);
    }
    let keys = Arc::new(
        Rv64imSpartan2DeciderSnark::setup(circuit)
            .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM main relation setup failed: {err}")))?,
    );
    cache
        .lock()
        .map_err(|_| SimpleKernelError::Bridge("RV64IM main relation setup cache poisoned".into()))?
        .insert(cache_key, keys.clone());
    Ok(keys)
}

pub fn prove_rv64im_spartan2_decider(
    pk: &Rv64imSpartan2DeciderProverKey,
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalBuildProof,
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
    proof: &Rv64imFinalBuildProof,
) -> Result<Rv64imMainRelationCircuit, SimpleKernelError> {
    build_main_relation_circuit_from_components(
        statement,
        proof.proof_digest,
        &proof.kernel_export,
        &proof.chunk_summaries,
        &proof.steps,
        &crate::rv64im::final_relation::final_proof_component_digests(proof),
    )
}

fn build_main_relation_circuit_from_components(
    statement: &Rv64imFinalStatement,
    proof_digest: [u8; 32],
    kernel_export: &Rv64imKernelExportProof,
    chunk_summaries: &[crate::finalize::FixedShapeChunkSummary],
    steps: &[Rv64imChunkTransitionWitness],
    component_digests: &Rv64imFinalProofComponentDigests,
) -> Result<Rv64imMainRelationCircuit, SimpleKernelError> {
    let trace = build_rv64im_main_circuit_trace_from_step_components(
        statement,
        proof_digest,
        kernel_export,
        chunk_summaries,
        steps,
        component_digests,
    )?;
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

fn build_main_relation_circuit_from_setup_shape(
    shape: &Rv64imMainRelationSetupShape,
) -> Result<Rv64imMainRelationCircuit, SimpleKernelError> {
    let trace = build_rv64im_main_circuit_trace_from_setup_shape(shape)?;
    let (params, _, structure) = rv64im_cached_root_main_lane_context()?;
    let optimized_cache = rv64im_cached_root_main_lane_optimized_cache()?;
    let dims = build_dims_and_policy(params, structure)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM main relation dims failed: {err}")))?;
    let mat_digest_vec = digest_ccs_matrices_with_sparse_cache(structure, Some(optimized_cache.sparse()));
    let mat_digest: [Goldilocks; 4] = mat_digest_vec
        .try_into()
        .map_err(|_| SimpleKernelError::Bridge("RV64IM main relation matrix digest length mismatch".into()))?;
    Ok(Rv64imMainRelationCircuit {
        public_statement_digest: trace.statement.public_statement_digest,
        params: params.clone(),
        structure: structure.clone(),
        dims,
        mat_digest,
        trace,
    })
}

fn rv64im_main_relation_setup_cache_key_from_shape(
    shape: &Rv64imMainRelationSetupShape,
) -> Result<[u8; 32], SimpleKernelError> {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/main_relation_spartan/setup_cache_key");
    let shape_bytes = bincode::serialize(shape)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM main relation setup cache key failed: {err}")))?;
    tr.append_message(
        b"neo.fold.next/rv64im/main_relation_spartan/setup_cache_key/shape",
        &shape_bytes,
    );
    Ok(tr.digest32())
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

fn cover_ce_claim_with_shared_point(
    shape: &Rv64imMainCircuitCeClaimShape,
    effective: Option<&CeClaim<neo_ajtai::Commitment, F, K>>,
    shared_r_values: &[K],
    shared_s_col_values: &[K],
) -> Result<CeClaim<neo_ajtai::Commitment, F, K>, SynthesisError> {
    let mut claim = if let Some(claim) = effective {
        pad_ce_claim_to_cover_shape(shape, claim)?
    } else {
        shape.zero_claim()
    };
    claim.r = shared_r_values.to_vec();
    claim.s_col = shared_s_col_values.to_vec();
    Ok(claim)
}

fn cover_ce_claim(
    shape: &Rv64imMainCircuitCeClaimShape,
    effective: Option<&CeClaim<neo_ajtai::Commitment, F, K>>,
) -> Result<CeClaim<neo_ajtai::Commitment, F, K>, SynthesisError> {
    if let Some(claim) = effective {
        return pad_ce_claim_to_cover_shape(shape, claim);
    }
    Ok(shape.zero_claim())
}

fn pad_f_matrix_to_shape(matrix: &Mat<F>, rows: usize, cols: usize) -> Result<Mat<F>, SynthesisError> {
    if matrix.rows() > rows || matrix.cols() > cols {
        return Err(SynthesisError::Unsatisfiable);
    }
    let mut out = Mat::zero(rows, cols, F::ZERO);
    for row in 0..matrix.rows() {
        for col in 0..matrix.cols() {
            out[(row, col)] = matrix[(row, col)];
        }
    }
    Ok(out)
}

fn pad_k_row_to_len(row: &[K], target_len: usize) -> Result<Vec<K>, SynthesisError> {
    if row.len() > target_len {
        return Err(SynthesisError::Unsatisfiable);
    }
    let mut out = row.to_vec();
    out.resize(target_len, K::ZERO);
    Ok(out)
}

fn pad_ce_claim_to_cover_shape(
    shape: &Rv64imMainCircuitCeClaimShape,
    claim: &CeClaim<neo_ajtai::Commitment, F, K>,
) -> Result<CeClaim<neo_ajtai::Commitment, F, K>, SynthesisError> {
    if !shape.covers_claim(claim) {
        return Err(SynthesisError::Unsatisfiable);
    }
    let y_ring_row_count = shape.y_ring_row_count as usize;
    if y_ring_row_count < shape.ct_len as usize {
        return Err(SynthesisError::Unsatisfiable);
    }
    let mut y_ring = Vec::with_capacity(y_ring_row_count);
    for row_idx in 0..y_ring_row_count {
        let mut target_len = shape.y_ring_row_lens.get(row_idx).copied().unwrap_or(0) as usize;
        if row_idx < shape.ct_len as usize {
            target_len = target_len.max(1);
        }
        let row = claim.y_ring.get(row_idx).map(Vec::as_slice).unwrap_or(&[]);
        y_ring.push(pad_k_row_to_len(row, target_len)?);
    }
    let mut c_step_coords = claim.c_step_coords.clone();
    c_step_coords.resize(shape.c_step_coords_len as usize, F::ZERO);
    Ok(CeClaim {
        c: claim.c.clone(),
        X: pad_f_matrix_to_shape(&claim.X, shape.x_rows as usize, shape.x_cols as usize)?,
        r: pad_k_row_to_len(&claim.r, shape.r_len as usize)?,
        s_col: pad_k_row_to_len(&claim.s_col, shape.s_col_len as usize)?,
        y_ring,
        ct: pad_k_row_to_len(&claim.ct, shape.ct_len as usize)?,
        aux_openings: pad_k_row_to_len(&claim.aux_openings, shape.aux_openings_len as usize)?,
        y_zcol: pad_k_row_to_len(&claim.y_zcol, shape.y_zcol_len as usize)?,
        m_in: claim.m_in,
        fold_digest: claim.fold_digest,
        c_step_coords,
        u_offset: claim.u_offset,
        u_len: claim.u_len,
    })
}

fn cover_ccs_claim(
    shape: &crate::rv64im::main_relation_trace::Rv64imMainCircuitCcsClaimShape,
    effective: Option<&neo_ccs::CcsClaim<neo_ajtai::Commitment, F>>,
) -> Result<neo_ccs::CcsClaim<neo_ajtai::Commitment, F>, SynthesisError> {
    if let Some(claim) = effective {
        if !shape.covers_claim(claim) {
            return Err(SynthesisError::Unsatisfiable);
        }
        let mut out = claim.clone();
        out.x.resize(shape.x_len as usize, F::ZERO);
        return Ok(out);
    }
    Ok(shape.zero_claim())
}

fn alloc_rounds<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    cover_round_lengths: &[u64],
    effective_rounds: &[Vec<K>],
    label: &str,
) -> Result<Vec<Vec<KNumVar>>, SynthesisError> {
    if cover_round_lengths.len() < effective_rounds.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    cover_round_lengths
        .iter()
        .enumerate()
        .map(|(round_idx, cover_len)| {
            let effective = effective_rounds
                .get(round_idx)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            if effective.len() > *cover_len as usize {
                return Err(SynthesisError::Unsatisfiable);
            }
            (0..(*cover_len as usize))
                .map(|coeff_idx| {
                    let coeff = effective.get(coeff_idx).copied().unwrap_or(K::ZERO);
                    alloc_k(
                        cs,
                        Some(KNum::from_neo_k(coeff)),
                        &format!("{label}_{round_idx}_{coeff_idx}"),
                    )
                })
                .collect()
        })
        .collect()
}

fn pad_round_values(cover_round_lengths: &[u64], effective_rounds: &[Vec<K>]) -> Result<Vec<Vec<K>>, SynthesisError> {
    if cover_round_lengths.len() < effective_rounds.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    cover_round_lengths
        .iter()
        .enumerate()
        .map(|(round_idx, cover_len)| {
            let effective = effective_rounds
                .get(round_idx)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            if effective.len() > *cover_len as usize {
                return Err(SynthesisError::Unsatisfiable);
            }
            let mut out = effective.to_vec();
            out.resize(*cover_len as usize, K::ZERO);
            Ok(out)
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

fn max_degree_from_cover_round_lengths(round_lengths: &[u64]) -> usize {
    round_lengths
        .iter()
        .copied()
        .map(|len| len.saturating_sub(1) as usize)
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
    handoff: &Rv64imMainCircuitHandoff,
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

pub(crate) fn next_public_digest(
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

pub(crate) fn enforce_digest_eq<CS: ConstraintSystem<SpartanF>>(
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
    bridge_handoff_digest: [u8; 32],
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let mut preimage = Vec::with_capacity(1 + 3 * 4);
    preimage.extend(alloc_const_field_values(
        cs,
        &[SpartanF::from_canonical_u64(RV64IM_CHUNK_RELATION_DIGEST_RAW_TAG)],
        "chunk_relation_digest_domain",
    )?);
    preimage.extend(alloc_private_field_values(
        cs,
        &digest32_as_spartan_fields(public_chunk_digest),
        "chunk_relation_digest_public_chunk",
    )?);
    preimage.extend(main_relation_digest.iter().cloned());
    preimage.extend(alloc_private_field_values(
        cs,
        &digest32_as_spartan_fields(bridge_handoff_digest),
        "chunk_relation_digest_bridge",
    )?);
    hash_packed_goldilocks_fields(cs.namespace(|| "chunk_relation_digest_hash"), &preimage)
}

pub(crate) fn alloc_const_field_values<CS: ConstraintSystem<SpartanF>>(
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

pub(crate) fn alloc_private_field_values<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    values: &[SpartanF],
    label: &str,
) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
    values
        .iter()
        .enumerate()
        .map(|(idx, value)| AllocatedNum::alloc(cs.namespace(|| format!("{label}_{idx}")), || Ok(*value)))
        .collect()
}

pub(crate) fn digest32_as_spartan_fields(digest: [u8; 32]) -> [SpartanF; 4] {
    digest32_as_fields(digest).map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()))
}
