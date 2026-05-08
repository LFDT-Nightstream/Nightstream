mod prove;
mod state;
mod terminal_circuit;
mod verify;

use neo_ajtai::Commitment;
use neo_ccs::{CcsStructure, CcsWitness, CeClaim, Mat};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_reductions::engines::optimized_engine::OptimizedStructureCache;
use neo_reductions::engines::utils::{build_dims_and_policy, digest_ccs_matrices_with_sparse_cache, Dims};
use p3_goldilocks::Goldilocks;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use prove::prove_direct_ccs_f_prime_circuit;
pub use verify::verify_direct_ccs_ivc_snark;

use super::circuit_util::direct_accumulator_digest_from_claims;
use super::construction2_fold::DirectCcsConstruction2FoldContext;
use super::final_ce::final_carry_witnesses;
use super::ivc_helpers::{superneo_ivc_states_match, validate_direct_ajtai_context};
use super::public_image::{
    direct_boundary_update_digest, direct_initial_boundary_digest, direct_public_trace_seed_digest,
    direct_public_trace_update_digest, direct_state_x_out, direct_vk_fs_digest, DIRECT_CCS_TRIVIAL_PC,
};
use super::snark::{DirectCcsIvcSnark, DirectCcsIvcSnarkVerifierKey};
use super::surface::build_direct_ccs_chunk_surface_from_ivc_relation;
use super::terminal_committed::{
    DirectCcsTerminalCommittedConstraintBreakdown, DirectCcsTerminalCommittedProof, DirectCcsTerminalCommittedRelation,
};
use super::zero_carry::build_direct_canonical_zero_carry;
use crate::construction2::{Construction2EncodedPublicInput, Construction2FreshInstance, Construction2PublicBoundary};
use crate::ivc::{SuperNeoIvcState, SuperNeoIvcStepRelation, SuperNeoIvcTranscriptSnapshot};
use crate::proof::{Carry, ChunkInput, StepInput};
use crate::superneo_circuit::ce_consistency::PaperCeRelationConstraintBreakdown;
use crate::superneo_nifs_circuit::{SuperNeoChunkCover, SuperNeoChunkReplaySurface};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DirectCcsFPrimeSnarkProof {
    pub construction2_u_i: Construction2PublicBoundary,
    pub(crate) terminal_f_prime_committed_step_proof: DirectCcsTerminalCommittedProof,
}

impl DirectCcsFPrimeSnarkProof {
    pub fn snark_bytes_len(&self) -> usize {
        self.terminal_f_prime_committed_step_proof.snark_data.len()
    }
}
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DirectCcsFPrimeSnarkPerf {
    pub setup_ms: f64,
    pub prep_ms: f64,
    pub prove_ms: f64,
    pub encode_ms: f64,
    pub total_prove_ms: f64,
    pub total_verify_ms: f64,
    pub r1cs_sizes: [usize; 10],
    pub r1cs_nnz: usize,
    pub pcs_ms: f64,
    pub final_proof_bytes: usize,
    pub snark_bytes: usize,
    pub public_inputs: usize,
    pub chunk_constraints_first4: [usize; 4],
    pub chunk_constraints_by_chunk: Vec<usize>,
    pub chunk_count: usize,
    pub public_link_constraints: usize,
    pub construction2_fold_constraints: usize,
    pub construction2_fold_final_ce_consistency_constraints: usize,
    pub chunk_done_constraints: usize,
    pub final_ce_relation_constraints: usize,
    pub final_ce_relation_breakdown: PaperCeRelationConstraintBreakdown,
    pub final_ce_bundle_constraints: usize,
    pub final_ce_bundle_digest_constraints: usize,
    pub final_ce_bundle_digest_match_constraints: usize,
    pub terminal_f_prime_constraints: usize,
    pub terminal_committed_width: usize,
    pub terminal_commitment_words: usize,
    pub terminal_source_values: usize,
    pub terminal_source_bit_values: usize,
    pub terminal_source_u32_values: usize,
    pub terminal_source_u64_values: usize,
    pub terminal_unclassified_private_values: usize,
    pub terminal_committed_breakdown: DirectCcsTerminalCommittedConstraintBreakdown,
    pub final_ce_r1cs_sizes: [usize; 10],
}

#[derive(Debug, Error)]
pub enum DirectCcsFPrimeSnarkError {
    #[error("direct CCS F' input error: {0}")]
    Input(String),
    #[error("direct CCS F' synthesis failed: {0}")]
    Synthesis(String),
    #[error("direct CCS F' setup failed: {0}")]
    Setup(String),
    #[error("direct CCS F' prepare failed: {0}")]
    Prepare(String),
    #[error("direct CCS F' prove failed: {0}")]
    Prove(String),
    #[error("direct CCS F' verify failed: {0}")]
    Verify(String),
    #[error("direct CCS F' proof encoding failed: {0}")]
    Encode(String),
    #[error("direct CCS F' proof decoding failed: {0}")]
    Decode(String),
    #[error("direct CCS F' public IO mismatch")]
    PublicIoMismatch,
}

#[derive(Clone)]
pub struct DirectCcsProgram {
    params: NeoParams,
    structure: CcsStructure<F>,
    public_input_len: Option<usize>,
}

impl DirectCcsProgram {
    pub fn new(params: &NeoParams, structure: &CcsStructure<F>) -> Self {
        Self {
            params: params.clone(),
            structure: structure.clone(),
            public_input_len: None,
        }
    }

    pub fn new_with_public_input_len(
        params: &NeoParams,
        structure: &CcsStructure<F>,
        public_input_len: usize,
    ) -> Result<Self, DirectCcsFPrimeSnarkError> {
        if public_input_len > structure.m {
            return Err(DirectCcsFPrimeSnarkError::Input(format!(
                "direct CCS public input len {public_input_len} exceeds CCS column count {}",
                structure.m
            )));
        }
        Ok(Self {
            params: params.clone(),
            structure: structure.clone(),
            public_input_len: Some(public_input_len),
        })
    }

    pub fn params(&self) -> &NeoParams {
        &self.params
    }
    pub fn structure(&self) -> &CcsStructure<F> {
        &self.structure
    }
    pub fn public_input_len(&self) -> Option<usize> {
        self.public_input_len
    }
    pub fn canonical_zero_carry(&self) -> Result<Carry, DirectCcsFPrimeSnarkError> {
        let public_input_len = self.public_input_len.ok_or_else(|| {
            DirectCcsFPrimeSnarkError::Input(
                "direct CCS canonical zero carry requires a fixed program public input len".into(),
            )
        })?;
        build_direct_canonical_zero_carry(&self.params, &self.structure, public_input_len)
    }
}

#[derive(Clone, Debug)]
pub struct DirectCcsStep {
    step: StepInput,
}

impl DirectCcsStep {
    pub fn new(step: StepInput) -> Self {
        Self { step }
    }
    pub fn into_step_input(self) -> StepInput {
        self.step
    }
}

#[derive(Clone, Debug)]
pub struct DirectCcsLatestFPrimeSummary {
    pub chunk_index: u64,
    pub fresh_claims: usize,
    pub incoming_ce_claims: usize,
    pub output_ce_claims: usize,
    pub final_ce_claims: usize,
    pub construction2_x_in: Construction2EncodedPublicInput,
    pub construction2_x_out: Construction2EncodedPublicInput,
}

#[derive(Clone)]
pub struct DirectCcsIvcState {
    params: NeoParams,
    structure: CcsStructure<F>,
    pub(crate) public_input_len: Option<usize>,
    dims: Dims,
    pub(crate) mat_digest: [Goldilocks; 4],
    pub(crate) vk_fs_digest: [u8; 32],
    pub(crate) initial_boundary_digest: [u8; 32],
    current_boundary_digest: [u8; 32],
    optimized_cache: OptimizedStructureCache,
    pub(crate) state: SuperNeoIvcState,
    accumulator_digest: [u8; 32],
    pub(crate) construction2_accumulator_digest: [u8; 32],
    public_trace_digest: [u8; 32],
    x_i: Construction2EncodedPublicInput,
    pub(crate) construction2_u_i: Construction2FreshInstance,
    pub(crate) last_step: Option<DirectCcsIvcStepRecord>,
}

#[derive(Clone)]
pub(crate) struct DirectCcsIvcStepRecord {
    pub(crate) relation: SuperNeoIvcStepRelation,
    pub(crate) surface: DirectCcsChunkCircuitSurface,
    pub(crate) x_i: Construction2EncodedPublicInput,
    pub(crate) construction2_u_i: Construction2FreshInstance,
    pub(crate) x_out: Construction2EncodedPublicInput,
    pub(crate) accumulator_out_digest: [u8; 32],
    pub(crate) accumulator_in_digest: [u8; 32],
    pub(crate) construction2_accumulator_in_digest: [u8; 32],
    pub(crate) construction2_accumulator_out_digest: [u8; 32],
    pub(crate) public_trace_in_digest: [u8; 32],
    pub(crate) current_boundary_in_digest: [u8; 32],
    pub(crate) public_trace_out_digest: [u8; 32],
    pub(crate) current_boundary_out_digest: [u8; 32],
    pub(crate) construction2_fold: Option<DirectCcsConstruction2FoldContext>,
}

#[derive(Clone)]
pub(crate) struct DirectCcsChunkCircuitSurface {
    pub(crate) cover: SuperNeoChunkCover,
    pub(crate) replay: SuperNeoChunkReplaySurface,
}

#[derive(Clone)]
pub(crate) struct DirectCcsFPrimeCircuit {
    params: NeoParams,
    structure: CcsStructure<F>,
    dims: Dims,
    mat_digest: [Goldilocks; 4],
    vk_fs_digest: [u8; 32],
    initial_boundary_digest: [u8; 32],
    chunks: Vec<DirectCcsChunkCircuitSurface>,
    initial_claims: Vec<CeClaim<Commitment, F, K>>,
    initial_transcript: Option<SuperNeoIvcTranscriptSnapshot>,
    chunk_count_in: u64,
    step_count_in: u64,
    x_in: Construction2EncodedPublicInput,
    construction2_input_u_i: Construction2FreshInstance,
    accumulator_in_digest: [u8; 32],
    construction2_accumulator_in_digest: [u8; 32],
    public_trace_in_digest: [u8; 32],
    current_boundary_in_digest: [u8; 32],
    chunk_count_out: u64,
    step_count_out: u64,
    x_out: Construction2EncodedPublicInput,
    accumulator_out_digest: [u8; 32],
    construction2_accumulator_out_digest: [u8; 32],
    public_trace_out_digest: [u8; 32],
    current_boundary_out_digest: [u8; 32],
    construction2_fold: Option<DirectCcsConstruction2FoldContext>,
    final_claims: Vec<CeClaim<Commitment, F, K>>,
    final_witnesses: Vec<CcsWitness<F>>,
}

#[derive(Clone)]
pub(crate) struct DirectCcsTerminalFPrimeCircuit {
    pub(crate) params: NeoParams,
    pub(crate) structure: CcsStructure<F>,
    pub(crate) dims: Dims,
    pub(crate) mat_digest: [Goldilocks; 4],
    pub(crate) vk_fs_digest: [u8; 32],
    pub(crate) initial_boundary_digest: [u8; 32],
    pub(crate) chunks: Vec<DirectCcsChunkCircuitSurface>,
    pub(crate) initial_claims: Vec<CeClaim<Commitment, F, K>>,
    pub(crate) initial_transcript: Option<SuperNeoIvcTranscriptSnapshot>,
    pub(crate) chunk_count_in: u64,
    pub(crate) step_count_in: u64,
    pub(crate) x_in: Construction2EncodedPublicInput,
    pub(crate) construction2_input_u_i: Construction2FreshInstance,
    pub(crate) accumulator_in_digest: [u8; 32],
    pub(crate) construction2_accumulator_in_digest: [u8; 32],
    pub(crate) public_trace_in_digest: [u8; 32],
    pub(crate) current_boundary_in_digest: [u8; 32],
    pub(crate) chunk_count_out: u64,
    pub(crate) step_count_out: u64,
    pub(crate) x_out: Construction2EncodedPublicInput,
    pub(crate) accumulator_out_digest: [u8; 32],
    pub(crate) construction2_accumulator_out_digest: [u8; 32],
    pub(crate) public_trace_out_digest: [u8; 32],
    pub(crate) current_boundary_out_digest: [u8; 32],
    pub(crate) construction2_fold: Option<DirectCcsConstruction2FoldContext>,
    pub(crate) final_witnesses: Vec<CcsWitness<F>>,
    pub(crate) prove_final_ce: bool,
}
