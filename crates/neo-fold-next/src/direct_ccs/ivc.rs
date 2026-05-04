use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use neo_ajtai::Commitment;
use neo_ccs::{CcsStructure, CcsWitness, CeClaim, Mat};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_reductions::engines::optimized_engine::OptimizedStructureCache;
use neo_reductions::engines::utils::{build_dims_and_policy, digest_ccs_matrices_with_sparse_cache, Dims};
use p3_goldilocks::Goldilocks;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use thiserror::Error;

use super::circuit_util::{
    alloc_digest_constant, digest32_as_spartan_fields, direct_accumulator_digest_circuit_from_claims,
    direct_accumulator_digest_from_claims, direct_terminal_accumulator_digest_range,
    direct_terminal_construction2_accumulator_digest_range, enforce_digest_eq_constant,
    enforce_digest_fields_public_io, enforce_direct_construction2_input_u_i,
    enforce_direct_current_boundary_transition, enforce_direct_public_trace_transition,
    enforce_direct_state_x_in_digest, enforce_direct_state_x_out_public_digest,
    enforce_direct_terminal_final_ce_consistency, field_to_spartan, public_digest_input, u64_halves_as_spartan_fields,
};
use super::construction2_fold::{synthesize_direct_construction2_fold, DirectCcsConstruction2FoldContext};
use super::final_ce::{final_carry_witnesses, measure_direct_final_ce_relation_breakdown};
use super::ivc_helpers::{
    alloc_initial_claim_bundle, alloc_initial_transcript, superneo_ivc_states_match, validate_direct_ajtai_context,
};
use super::public_image::{
    direct_boundary_update_digest, direct_initial_boundary_digest, direct_public_trace_seed_digest,
    direct_public_trace_update_digest, direct_state_x_out, direct_vk_fs_digest, DirectCcsIvcPublicImage,
    DIRECT_CCS_TRIVIAL_PC,
};
use super::snark::{DirectCcsIvcSnark, DirectCcsIvcSnarkVerifierKey};
use super::surface::build_direct_ccs_chunk_surface_from_ivc_relation;
use super::terminal_committed::{
    prove_direct_ccs_terminal_committed_relation, setup_direct_ccs_terminal_committed_relation_cached,
    verify_direct_ccs_terminal_committed_relation, DirectCcsTerminalCommittedConstraintBreakdown,
    DirectCcsTerminalCommittedProof, DirectCcsTerminalCommittedRelation, DirectCcsTerminalError,
};
use super::terminal_measure::measure_direct_ccs_f_prime_constraints;
use super::zero_carry::build_direct_canonical_zero_carry;
use crate::construction2::{Construction2EncodedPublicInput, Construction2FreshInstance, Construction2PublicBoundary};
use crate::ivc::{SuperNeoIvcState, SuperNeoIvcStepRelation, SuperNeoIvcTranscriptSnapshot};
use crate::proof::{Carry, ChunkInput, StepInput};
use crate::spartan_backend::{NeoFoldDeciderEngine, SpartanCircuit, SpartanF};
use crate::superneo_circuit::ce_consistency::PaperCeRelationConstraintBreakdown;
use crate::superneo_nifs_circuit::{synthesize_superneo_nifs_chunk, SuperNeoChunkCover, SuperNeoChunkReplaySurface};

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

impl DirectCcsFPrimeCircuit {
    pub(crate) fn terminal_circuit(&self, prove_final_ce: bool) -> DirectCcsTerminalFPrimeCircuit {
        DirectCcsTerminalFPrimeCircuit {
            params: self.params.clone(),
            structure: self.structure.clone(),
            dims: self.dims,
            mat_digest: self.mat_digest,
            vk_fs_digest: self.vk_fs_digest,
            initial_boundary_digest: self.initial_boundary_digest,
            chunks: self.chunks.clone(),
            initial_claims: self.initial_claims.clone(),
            initial_transcript: self.initial_transcript.clone(),
            chunk_count_in: self.chunk_count_in,
            step_count_in: self.step_count_in,
            x_in: self.x_in.clone(),
            construction2_input_u_i: self.construction2_input_u_i.clone(),
            accumulator_in_digest: self.accumulator_in_digest,
            construction2_accumulator_in_digest: self.construction2_accumulator_in_digest,
            public_trace_in_digest: self.public_trace_in_digest,
            current_boundary_in_digest: self.current_boundary_in_digest,
            chunk_count_out: self.chunk_count_out,
            step_count_out: self.step_count_out,
            x_out: self.x_out.clone(),
            accumulator_out_digest: self.accumulator_out_digest,
            construction2_accumulator_out_digest: self.construction2_accumulator_out_digest,
            public_trace_out_digest: self.public_trace_out_digest,
            current_boundary_out_digest: self.current_boundary_out_digest,
            construction2_fold: self.construction2_fold.clone(),
            final_witnesses: self.final_witnesses.clone(),
            prove_final_ce,
        }
    }
}

impl DirectCcsTerminalFPrimeCircuit {
    fn public_image(&self, construction2_u_i: Construction2PublicBoundary) -> DirectCcsIvcPublicImage {
        DirectCcsIvcPublicImage {
            mat_digest: self.mat_digest,
            vk_fs_digest: self.vk_fs_digest,
            initial_boundary_digest: self.initial_boundary_digest,
            current_boundary_digest: self.current_boundary_out_digest,
            pc: DIRECT_CCS_TRIVIAL_PC,
            chunk_count_out: self.chunk_count_out,
            step_count_out: self.step_count_out,
            x_out: self.x_out.clone(),
            accumulator_out_digest: self.accumulator_out_digest,
            public_trace_out_digest: self.public_trace_out_digest,
            construction2_accumulator_digest: self.construction2_accumulator_out_digest,
            construction2_u_i,
        }
    }

    fn terminal_public_values(&self) -> Vec<SpartanF> {
        let mut values = Vec::with_capacity(4 + 2 + 2 + 4 + 4 + 4 + 4 + 256 + 4 + 4 + 4);
        values.extend(self.mat_digest.iter().copied().map(field_to_spartan));
        values.extend(u64_halves_as_spartan_fields(self.chunk_count_out));
        values.extend(u64_halves_as_spartan_fields(self.step_count_out));
        values.extend(digest32_as_spartan_fields(self.vk_fs_digest));
        values.extend(digest32_as_spartan_fields(self.initial_boundary_digest));
        values.extend(digest32_as_spartan_fields(self.current_boundary_out_digest));
        values.extend(digest32_as_spartan_fields(self.x_out.bytes()));
        values.extend(self.x_out.field_image().into_iter().map(field_to_spartan));
        values.extend(digest32_as_spartan_fields(self.accumulator_out_digest));
        values.extend(digest32_as_spartan_fields(self.public_trace_out_digest));
        values.extend(digest32_as_spartan_fields(self.construction2_accumulator_out_digest));
        values
    }

    pub(crate) fn construction2_x_bit_range(&self) -> std::ops::Range<usize> {
        let start = 4 + 2 + 2 + 4 + 4 + 4 + 4;
        start..start + crate::construction2::CONSTRUCTION2_ENC_INST_BITS
    }

    pub(crate) fn construction2_x_i(&self) -> Result<Construction2EncodedPublicInput, DirectCcsTerminalError> {
        if self.chunks.last().is_none() {
            return Err(DirectCcsTerminalError::Bridge(
                "direct CCS terminal F' requires one latest chunk for Construction-2 x_i".into(),
            ));
        }
        Ok(self.x_out.clone())
    }

    pub(crate) fn synthesize_body_with_public_inputs<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        public_inputs: &[AllocatedNum<SpartanF>],
    ) -> Result<(), SynthesisError> {
        if self.chunk_count_in.checked_add(1) != Some(self.chunk_count_out) || self.step_count_in >= self.step_count_out
        {
            return Err(SynthesisError::Unsatisfiable);
        }
        enforce_direct_construction2_input_u_i(
            &mut cs.namespace(|| "direct_terminal_construction2_input_u_i"),
            &self.construction2_input_u_i,
            &self.x_in,
            self.chunk_count_in,
            self.params.kappa as usize,
        )?;
        let mut transcript = alloc_initial_transcript(cs, self.initial_transcript.as_ref())?;
        let mut carried = alloc_initial_claim_bundle(cs, &self.initial_claims)?;
        let mut last_chunk_digest = None;

        let accumulator_in_digest = direct_accumulator_digest_circuit_from_claims(
            &mut cs.namespace(|| "direct_terminal_accumulator_in_digest"),
            &self.params,
            carried.effective_claims(),
        )?;
        enforce_digest_eq_constant(
            &mut cs.namespace(|| "direct_terminal_accumulator_in_digest_private"),
            &accumulator_in_digest,
            self.accumulator_in_digest,
            "direct_terminal_accumulator_in_digest_private",
        )?;
        let construction2_accumulator_in_digest = alloc_digest_constant(
            &mut cs.namespace(|| "direct_terminal_construction2_accumulator_in_digest"),
            self.construction2_accumulator_in_digest,
            "direct_terminal_construction2_accumulator_in_digest",
        )?;
        enforce_direct_state_x_in_digest(
            &mut cs.namespace(|| "direct_terminal_x_in_digest"),
            self.vk_fs_digest,
            &self.mat_digest,
            self.chunk_count_in,
            self.step_count_in,
            self.initial_boundary_digest,
            self.current_boundary_in_digest,
            DIRECT_CCS_TRIVIAL_PC,
            &accumulator_in_digest,
            &construction2_accumulator_in_digest,
            self.public_trace_in_digest,
            self.x_in.bytes(),
            "direct_terminal_x_in_digest",
        )?;

        for (chunk_index, chunk) in self.chunks.iter().enumerate() {
            let (next, chunk_digest) = synthesize_superneo_nifs_chunk(
                &self.params,
                &self.structure,
                self.dims,
                &self.mat_digest,
                &mut cs.namespace(|| format!("chunk_{chunk_index}")),
                chunk_index,
                &chunk.cover,
                &chunk.replay,
                &mut transcript,
                carried,
                Some((
                    &accumulator_in_digest,
                    digest32_as_spartan_fields(self.accumulator_in_digest),
                )),
            )?;
            transcript.append_message(
                cs.namespace(|| format!("chunk_{chunk_index}_done")),
                b"neo.fold.next/chunk_done",
                &[1],
            )?;
            carried = next;
            last_chunk_digest = Some(chunk_digest);
        }

        let accumulator_digest = direct_accumulator_digest_circuit_from_claims(
            &mut cs.namespace(|| "direct_terminal_accumulator_digest"),
            &self.params,
            carried.effective_claims(),
        )?;
        enforce_digest_fields_public_io(
            &mut cs.namespace(|| "direct_terminal_accumulator_digest_public"),
            &accumulator_digest,
            public_inputs,
            direct_terminal_accumulator_digest_range(),
            "direct_terminal_accumulator_digest_public",
        )?;
        let last_chunk_digest = last_chunk_digest.ok_or(SynthesisError::Unsatisfiable)?;
        let current_boundary_out_digest = enforce_direct_current_boundary_transition(
            &mut cs.namespace(|| "direct_terminal_current_boundary_transition"),
            public_inputs,
            self.current_boundary_in_digest,
            &last_chunk_digest,
        )?;
        let public_trace_out_digest = enforce_direct_public_trace_transition(
            &mut cs.namespace(|| "direct_terminal_public_trace_transition"),
            public_inputs,
            self.public_trace_in_digest,
            &last_chunk_digest,
        )?;
        let construction2_accumulator_out_digest =
            public_digest_input(public_inputs, direct_terminal_construction2_accumulator_digest_range())?;
        synthesize_direct_construction2_fold(
            &mut cs.namespace(|| "direct_terminal_construction2_fold"),
            self.construction2_fold.as_ref(),
            public_inputs,
            self.construction2_accumulator_in_digest,
        )?;
        enforce_direct_state_x_out_public_digest(
            &mut cs.namespace(|| "direct_terminal_x_out_digest"),
            public_inputs,
            self.vk_fs_digest,
            &self.mat_digest,
            self.chunk_count_out,
            self.step_count_out,
            self.initial_boundary_digest,
            &current_boundary_out_digest,
            DIRECT_CCS_TRIVIAL_PC,
            &accumulator_digest,
            &construction2_accumulator_out_digest,
            &public_trace_out_digest,
            "direct_terminal_x_out_digest",
        )?;
        if self.prove_final_ce {
            enforce_direct_terminal_final_ce_consistency(
                &mut cs.namespace(|| "direct_terminal_final_ce"),
                &self.params,
                &self.structure,
                carried.effective_claims(),
                &self.final_witnesses,
            )?;
        }
        Ok(())
    }
}

impl DirectCcsIvcState {
    pub fn new(program: DirectCcsProgram) -> Result<Self, DirectCcsFPrimeSnarkError> {
        let mut state = Self::from_parts(program.params(), program.structure())?;
        state.public_input_len = program.public_input_len();
        state.reset_base_public_image();
        Ok(state)
    }

    pub fn new_with_canonical_zero_carry(program: DirectCcsProgram) -> Result<Self, DirectCcsFPrimeSnarkError> {
        let carry = program.canonical_zero_carry()?;
        let mut state = Self::from_parts(program.params(), program.structure())?;
        state.public_input_len = program.public_input_len();
        state.state = SuperNeoIvcState::seed_with_carry(carry);
        state.accumulator_digest = direct_accumulator_digest_from_claims(&state.params, &state.state.carry.claims);
        state.construction2_accumulator_digest = state.accumulator_digest;
        state.reset_base_public_image();
        Ok(state)
    }

    fn reset_base_public_image(&mut self) {
        self.vk_fs_digest = direct_vk_fs_digest(&self.params, &self.mat_digest, self.public_input_len);
        self.initial_boundary_digest = direct_initial_boundary_digest(&self.mat_digest, self.public_input_len);
        self.current_boundary_digest = self.initial_boundary_digest;
        self.public_trace_digest = direct_public_trace_seed_digest(&self.mat_digest);
        self.x_i = direct_state_x_out(
            self.vk_fs_digest,
            &self.mat_digest,
            self.state.chunk_count,
            self.state.step_count,
            self.initial_boundary_digest,
            self.current_boundary_digest,
            DIRECT_CCS_TRIVIAL_PC,
            self.accumulator_digest,
            self.construction2_accumulator_digest,
            self.public_trace_digest,
        );
        self.construction2_u_i =
            Construction2FreshInstance::canonical_zero(self.params.kappa as usize, self.x_i.clone());
    }

    pub fn from_parts(params: &NeoParams, structure: &CcsStructure<F>) -> Result<Self, DirectCcsFPrimeSnarkError> {
        validate_direct_ajtai_context(params, structure)?;
        let dims = build_dims_and_policy(params, structure)
            .map_err(|err| DirectCcsFPrimeSnarkError::Input(err.to_string()))?;
        let optimized_cache = OptimizedStructureCache::build(structure)
            .map_err(|err| DirectCcsFPrimeSnarkError::Input(err.to_string()))?;
        let mat_digest: [Goldilocks; 4] = digest_ccs_matrices_with_sparse_cache(structure, None)
            .try_into()
            .map_err(|digest: Vec<Goldilocks>| {
                DirectCcsFPrimeSnarkError::Input(format!("expected 4 matrix digest limbs, got {}", digest.len()))
            })?;
        let state = SuperNeoIvcState::seed();
        let accumulator_digest = direct_accumulator_digest_from_claims(params, &state.carry.claims);
        let construction2_accumulator_digest = accumulator_digest;
        let vk_fs_digest = direct_vk_fs_digest(params, &mat_digest, None);
        let initial_boundary_digest = direct_initial_boundary_digest(&mat_digest, None);
        let current_boundary_digest = initial_boundary_digest;
        let public_trace_digest = direct_public_trace_seed_digest(&mat_digest);
        let x_i = direct_state_x_out(
            vk_fs_digest,
            &mat_digest,
            state.chunk_count,
            state.step_count,
            initial_boundary_digest,
            current_boundary_digest,
            DIRECT_CCS_TRIVIAL_PC,
            accumulator_digest,
            construction2_accumulator_digest,
            public_trace_digest,
        );
        let construction2_u_i = Construction2FreshInstance::canonical_zero(params.kappa as usize, x_i.clone());
        Ok(Self {
            params: params.clone(),
            structure: structure.clone(),
            public_input_len: None,
            dims,
            mat_digest,
            vk_fs_digest,
            initial_boundary_digest,
            current_boundary_digest,
            optimized_cache,
            state,
            accumulator_digest,
            construction2_accumulator_digest,
            public_trace_digest,
            x_i,
            construction2_u_i,
            last_step: None,
        })
    }

    pub fn append_step<L, MR, MB>(
        &self,
        step: DirectCcsStep,
        log: &L,
        mixers: crate::prover::CommitmentMixers<MR, MB>,
    ) -> Result<Self, DirectCcsFPrimeSnarkError>
    where
        L: neo_ccs::traits::SModuleHomomorphism<F, Commitment> + Sync,
        MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        let chunk = ChunkInput {
            start_index: self.state.step_count as usize,
            steps: vec![step.into_step_input()],
        };
        self.append_chunk(chunk, log, mixers)
    }

    pub(crate) fn append_step_with_construction2_accumulator_digest<L, MR, MB>(
        &self,
        step: DirectCcsStep,
        log: &L,
        mixers: crate::prover::CommitmentMixers<MR, MB>,
        construction2_accumulator_digest_out: [u8; 32],
    ) -> Result<Self, DirectCcsFPrimeSnarkError>
    where
        L: neo_ccs::traits::SModuleHomomorphism<F, Commitment> + Sync,
        MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        let chunk = ChunkInput {
            start_index: self.state.step_count as usize,
            steps: vec![step.into_step_input()],
        };
        self.append_chunk_with_construction2_accumulator_digest(
            chunk,
            log,
            mixers,
            construction2_accumulator_digest_out,
        )
    }

    pub fn append_chunk<L, MR, MB>(
        &self,
        chunk: ChunkInput,
        log: &L,
        mixers: crate::prover::CommitmentMixers<MR, MB>,
    ) -> Result<Self, DirectCcsFPrimeSnarkError>
    where
        L: neo_ccs::traits::SModuleHomomorphism<F, Commitment> + Sync,
        MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        self.validate_current_surface()?;
        self.validate_chunk_shape(&chunk)?;
        let (next_state, relation) = self
            .state
            .append_chunk_with_perf_and_accumulator_handle(
                &self.params,
                &self.structure,
                chunk,
                log,
                mixers,
                &self.optimized_cache,
            )
            .map_err(|err| DirectCcsFPrimeSnarkError::Input(err.to_string()))?;
        relation
            .verify_with_accumulator_handle(&self.params, &self.structure, log, mixers, &self.optimized_cache)
            .map_err(|err| DirectCcsFPrimeSnarkError::Input(err.to_string()))?;
        self.append_verified_relation_with_state(next_state, &relation, log, mixers)
    }

    fn append_chunk_with_construction2_accumulator_digest<L, MR, MB>(
        &self,
        chunk: ChunkInput,
        log: &L,
        mixers: crate::prover::CommitmentMixers<MR, MB>,
        construction2_accumulator_digest_out: [u8; 32],
    ) -> Result<Self, DirectCcsFPrimeSnarkError>
    where
        L: neo_ccs::traits::SModuleHomomorphism<F, Commitment> + Sync,
        MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        self.validate_current_surface()?;
        self.validate_chunk_shape(&chunk)?;
        let (next_state, relation) = self
            .state
            .append_chunk_with_perf_and_accumulator_handle(
                &self.params,
                &self.structure,
                chunk,
                log,
                mixers,
                &self.optimized_cache,
            )
            .map_err(|err| DirectCcsFPrimeSnarkError::Input(err.to_string()))?;
        relation
            .verify_with_accumulator_handle(&self.params, &self.structure, log, mixers, &self.optimized_cache)
            .map_err(|err| DirectCcsFPrimeSnarkError::Input(err.to_string()))?;
        self.append_verified_relation_with_state_with_construction2_accumulator_digest(
            next_state,
            &relation,
            log,
            mixers,
            construction2_accumulator_digest_out,
        )
    }

    pub fn append_relation<L, MR, MB>(
        &self,
        relation: &SuperNeoIvcStepRelation,
        log: &L,
        mixers: crate::prover::CommitmentMixers<MR, MB>,
    ) -> Result<Self, DirectCcsFPrimeSnarkError>
    where
        L: neo_ccs::traits::SModuleHomomorphism<F, Commitment> + Sync,
        MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        if !superneo_ivc_states_match(&self.state, &relation.state_in) {
            return Err(DirectCcsFPrimeSnarkError::Input(
                "direct CCS IVC append relation does not start from the carried state".into(),
            ));
        }
        self.validate_current_surface()?;
        self.validate_chunk_shape(&relation.chunk)?;
        relation
            .verify_with_accumulator_handle(&self.params, &self.structure, log, mixers, &self.optimized_cache)
            .map_err(|err| DirectCcsFPrimeSnarkError::Input(err.to_string()))?;
        self.append_verified_relation_with_state(relation.state_out.clone(), relation, log, mixers)
    }

    pub(crate) fn append_relation_with_construction2_accumulator_digest<L, MR, MB>(
        &self,
        relation: &SuperNeoIvcStepRelation,
        log: &L,
        mixers: crate::prover::CommitmentMixers<MR, MB>,
        construction2_accumulator_digest_out: [u8; 32],
    ) -> Result<Self, DirectCcsFPrimeSnarkError>
    where
        L: neo_ccs::traits::SModuleHomomorphism<F, Commitment> + Sync,
        MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        if !superneo_ivc_states_match(&self.state, &relation.state_in) {
            return Err(DirectCcsFPrimeSnarkError::Input(
                "direct CCS IVC append relation does not start from the carried state".into(),
            ));
        }
        self.validate_current_surface()?;
        self.validate_chunk_shape(&relation.chunk)?;
        relation
            .verify_with_accumulator_handle(&self.params, &self.structure, log, mixers, &self.optimized_cache)
            .map_err(|err| DirectCcsFPrimeSnarkError::Input(err.to_string()))?;
        self.append_verified_relation_with_state_with_construction2_accumulator_digest(
            relation.state_out.clone(),
            relation,
            log,
            mixers,
            construction2_accumulator_digest_out,
        )
    }

    fn append_verified_relation_with_state<L, MR, MB>(
        &self,
        state_out: SuperNeoIvcState,
        relation: &SuperNeoIvcStepRelation,
        log: &L,
        mixers: crate::prover::CommitmentMixers<MR, MB>,
    ) -> Result<Self, DirectCcsFPrimeSnarkError>
    where
        L: neo_ccs::traits::SModuleHomomorphism<F, Commitment> + Sync,
        MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        let accumulator_digest = direct_accumulator_digest_from_claims(&self.params, &state_out.carry.claims);
        self.append_verified_relation_with_state_with_construction2_accumulator_digest(
            state_out,
            relation,
            log,
            mixers,
            accumulator_digest,
        )
    }

    fn append_verified_relation_with_state_with_construction2_accumulator_digest<L, MR, MB>(
        &self,
        state_out: SuperNeoIvcState,
        relation: &SuperNeoIvcStepRelation,
        log: &L,
        mixers: crate::prover::CommitmentMixers<MR, MB>,
        construction2_accumulator_digest_out: [u8; 32],
    ) -> Result<Self, DirectCcsFPrimeSnarkError>
    where
        L: neo_ccs::traits::SModuleHomomorphism<F, Commitment> + Sync,
        MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        let surface = build_direct_ccs_chunk_surface_from_ivc_relation(
            &self.params,
            &self.structure,
            self.dims,
            relation,
            log,
            mixers,
            &self.optimized_cache,
        )?;
        let accumulator_digest = direct_accumulator_digest_from_claims(&self.params, &state_out.carry.claims);
        let public_trace_digest = direct_public_trace_update_digest(
            self.public_trace_digest,
            surface.replay.handoff.public_chunk_instance_digest,
        );
        let current_boundary_digest = direct_boundary_update_digest(
            self.current_boundary_digest,
            surface.replay.handoff.public_chunk_instance_digest,
        );
        let x_out = direct_state_x_out(
            self.vk_fs_digest,
            &self.mat_digest,
            state_out.chunk_count,
            state_out.step_count,
            self.initial_boundary_digest,
            current_boundary_digest,
            DIRECT_CCS_TRIVIAL_PC,
            accumulator_digest,
            construction2_accumulator_digest_out,
            public_trace_digest,
        );
        let construction2_u_i = self.derive_next_construction2_u_i(
            relation,
            &surface,
            state_out.chunk_count,
            state_out.step_count,
            &x_out,
            accumulator_digest,
            construction2_accumulator_digest_out,
            current_boundary_digest,
            &state_out.carry.claims,
            &state_out.carry.witnesses,
        )?;
        Ok(Self {
            params: self.params.clone(),
            structure: self.structure.clone(),
            public_input_len: self.public_input_len,
            dims: self.dims,
            mat_digest: self.mat_digest,
            vk_fs_digest: self.vk_fs_digest,
            initial_boundary_digest: self.initial_boundary_digest,
            current_boundary_digest,
            optimized_cache: self.optimized_cache.clone(),
            state: state_out,
            accumulator_digest,
            construction2_accumulator_digest: construction2_accumulator_digest_out,
            public_trace_digest,
            x_i: x_out.clone(),
            construction2_u_i,
            last_step: Some(DirectCcsIvcStepRecord {
                relation: relation.clone(),
                surface,
                x_i: self.x_i.clone(),
                construction2_u_i: self.construction2_u_i.clone(),
                x_out,
                accumulator_in_digest: self.accumulator_digest,
                accumulator_out_digest: accumulator_digest,
                construction2_accumulator_in_digest: self.construction2_accumulator_digest,
                construction2_accumulator_out_digest: construction2_accumulator_digest_out,
                public_trace_in_digest: self.public_trace_digest,
                public_trace_out_digest: public_trace_digest,
                current_boundary_in_digest: self.current_boundary_digest,
                current_boundary_out_digest: current_boundary_digest,
                construction2_fold: None,
            }),
        })
    }

    fn derive_next_construction2_u_i(
        &self,
        relation: &SuperNeoIvcStepRelation,
        surface: &DirectCcsChunkCircuitSurface,
        chunk_count_out: u64,
        step_count_out: u64,
        x_out: &Construction2EncodedPublicInput,
        accumulator_out_digest: [u8; 32],
        construction2_accumulator_out_digest: [u8; 32],
        current_boundary_out_digest: [u8; 32],
        final_claims: &[CeClaim<Commitment, F, K>],
        final_witnesses: &[Mat<F>],
    ) -> Result<Construction2FreshInstance, DirectCcsFPrimeSnarkError> {
        let circuit = DirectCcsFPrimeCircuit {
            params: self.params.clone(),
            structure: self.structure.clone(),
            dims: self.dims,
            mat_digest: self.mat_digest,
            vk_fs_digest: self.vk_fs_digest,
            initial_boundary_digest: self.initial_boundary_digest,
            chunks: vec![surface.clone()],
            initial_claims: relation.state_in.carry.claims.clone(),
            initial_transcript: Some(relation.state_in.transcript.clone()),
            chunk_count_in: relation.state_in.chunk_count,
            step_count_in: relation.state_in.step_count,
            x_in: self.x_i.clone(),
            construction2_input_u_i: self.construction2_u_i.clone(),
            accumulator_in_digest: self.accumulator_digest,
            construction2_accumulator_in_digest: self.construction2_accumulator_digest,
            public_trace_in_digest: self.public_trace_digest,
            current_boundary_in_digest: self.current_boundary_digest,
            chunk_count_out,
            step_count_out,
            x_out: x_out.clone(),
            accumulator_out_digest,
            construction2_accumulator_out_digest,
            public_trace_out_digest: direct_public_trace_update_digest(
                self.public_trace_digest,
                surface.replay.handoff.public_chunk_instance_digest,
            ),
            current_boundary_out_digest,
            construction2_fold: None,
            final_claims: final_claims.to_vec(),
            final_witnesses: final_carry_witnesses(final_witnesses)?,
        };
        let relation = DirectCcsTerminalCommittedRelation::from_terminal_circuit(circuit.terminal_circuit(false))
            .map_err(|err| DirectCcsFPrimeSnarkError::Synthesis(err.to_string()))?;
        Construction2FreshInstance::from_public_boundary(relation.public_boundary())
            .map_err(DirectCcsFPrimeSnarkError::Input)
    }
    pub fn append_all<L, MR, MB>(
        params: &NeoParams,
        structure: &CcsStructure<F>,
        relations: &[SuperNeoIvcStepRelation],
        log: &L,
        mixers: crate::prover::CommitmentMixers<MR, MB>,
    ) -> Result<Self, DirectCcsFPrimeSnarkError>
    where
        L: neo_ccs::traits::SModuleHomomorphism<F, Commitment> + Sync,
        MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        let mut state = Self::from_parts(params, structure)?;
        for relation in relations {
            state = state.append_relation(relation, log, mixers)?;
        }
        Ok(state)
    }

    pub fn final_state(&self) -> &SuperNeoIvcState {
        &self.state
    }
    pub fn params(&self) -> &NeoParams {
        &self.params
    }
    pub fn structure(&self) -> &CcsStructure<F> {
        &self.structure
    }
    pub fn construction2_public_boundary(&self) -> Construction2PublicBoundary {
        Construction2PublicBoundary::from_fresh_instance(&self.construction2_u_i)
    }

    pub fn latest_relation_and_advice(&self) -> Result<DirectCcsLatestFPrimeSummary, DirectCcsFPrimeSnarkError> {
        let last = self.last_step.as_ref().ok_or_else(|| {
            DirectCcsFPrimeSnarkError::Input(
                "direct CCS folded compression requires at least one appended SuperNeo relation".into(),
            )
        })?;
        Ok(DirectCcsLatestFPrimeSummary {
            chunk_index: last.relation.chunk_index,
            fresh_claims: last.relation.chunk.steps.len(),
            incoming_ce_claims: last.relation.state_in.carry.claims.len(),
            output_ce_claims: last.relation.replay_witness.ccs_outputs.len(),
            final_ce_claims: self.state.carry.claims.len(),
            construction2_x_in: last.x_i.clone(),
            construction2_x_out: last.x_out.clone(),
        })
    }

    pub(crate) fn latest_construction2_fold_context(
        &self,
    ) -> Result<DirectCcsConstruction2FoldContext, DirectCcsFPrimeSnarkError> {
        let last = self.last_step.as_ref().ok_or_else(|| {
            DirectCcsFPrimeSnarkError::Input("direct CCS Construction-2 fold context requires a latest step".into())
        })?;
        Ok(DirectCcsConstruction2FoldContext {
            params: self.params.clone(),
            structure: self.structure.clone(),
            dims: self.dims,
            mat_digest: self.mat_digest,
            initial_claims: last.relation.state_in.carry.claims.clone(),
            initial_transcript: Some(last.relation.state_in.transcript.clone()),
            surface: last.surface.clone(),
            accumulator_in_digest: last.accumulator_in_digest,
            accumulator_out_digest: last.accumulator_out_digest,
        })
    }

    pub(crate) fn with_latest_construction2_fold_context(
        mut self,
        context: Option<DirectCcsConstruction2FoldContext>,
    ) -> Result<Self, DirectCcsFPrimeSnarkError> {
        let Some(context) = context else {
            return Ok(self);
        };
        let last = self.last_step.as_mut().ok_or_else(|| {
            DirectCcsFPrimeSnarkError::Input("direct CCS Construction-2 fold context requires a latest step".into())
        })?;
        context.validate_digest_linkage(
            last.construction2_accumulator_in_digest,
            last.construction2_accumulator_out_digest,
        )?;
        last.construction2_fold = Some(context);
        let relation =
            DirectCcsTerminalCommittedRelation::from_terminal_circuit(self.latest_circuit()?.terminal_circuit(false))
                .map_err(|err| DirectCcsFPrimeSnarkError::Synthesis(err.to_string()))?;
        let construction2_u_i = Construction2FreshInstance::from_public_boundary(relation.public_boundary())
            .map_err(DirectCcsFPrimeSnarkError::Input)?;
        if construction2_u_i.x_i() != &self.x_i {
            return Err(DirectCcsFPrimeSnarkError::Input(
                "direct CCS Construction-2 folded output u_i does not match current x_i".into(),
            ));
        }
        self.construction2_u_i = construction2_u_i;
        Ok(self)
    }

    pub fn compress_with_trace(
        &self,
        emit: &mut dyn FnMut(&str),
    ) -> Result<(DirectCcsFPrimeSnarkProof, DirectCcsFPrimeSnarkPerf), DirectCcsFPrimeSnarkError> {
        let (snark, _vk, perf) = self.compress_snark_with_trace(emit)?;
        Ok((snark.proof().clone(), perf))
    }

    pub fn compress_snark_with_trace(
        &self,
        emit: &mut dyn FnMut(&str),
    ) -> Result<
        (
            DirectCcsIvcSnark,
            DirectCcsIvcSnarkVerifierKey,
            DirectCcsFPrimeSnarkPerf,
        ),
        DirectCcsFPrimeSnarkError,
    > {
        self.ensure_terminal_compression_is_proof_complete()?;
        emit("direct_ccs_ivc.phase=latest_relation_and_advice.start");
        let circuit = self.latest_circuit()?;
        emit("direct_ccs_ivc.phase=latest_relation_and_advice.done");
        emit("direct_ccs_ivc.phase=spartan_terminal_prove.start");
        let proved = prove_direct_ccs_f_prime_circuit(circuit, emit);
        emit("direct_ccs_ivc.phase=spartan_terminal_prove.done");
        proved
    }

    pub fn compress(&self) -> Result<(DirectCcsFPrimeSnarkProof, DirectCcsFPrimeSnarkPerf), DirectCcsFPrimeSnarkError> {
        let mut emit = |_message: &str| {};
        self.compress_with_trace(&mut emit)
    }

    pub fn compress_snark(
        &self,
    ) -> Result<
        (
            DirectCcsIvcSnark,
            DirectCcsIvcSnarkVerifierKey,
            DirectCcsFPrimeSnarkPerf,
        ),
        DirectCcsFPrimeSnarkError,
    > {
        let mut emit = |_message: &str| {};
        self.compress_snark_with_trace(&mut emit)
    }

    pub(crate) fn latest_circuit(&self) -> Result<DirectCcsFPrimeCircuit, DirectCcsFPrimeSnarkError> {
        let last = self.last_step.as_ref().ok_or_else(|| {
            DirectCcsFPrimeSnarkError::Input(
                "direct CCS folded compression requires at least one appended SuperNeo relation".into(),
            )
        })?;
        Ok(DirectCcsFPrimeCircuit {
            params: self.params.clone(),
            structure: self.structure.clone(),
            dims: self.dims,
            mat_digest: self.mat_digest,
            vk_fs_digest: self.vk_fs_digest,
            initial_boundary_digest: self.initial_boundary_digest,
            chunks: vec![last.surface.clone()],
            initial_claims: last.relation.state_in.carry.claims.clone(),
            initial_transcript: Some(last.relation.state_in.transcript.clone()),
            chunk_count_in: last.relation.state_in.chunk_count,
            step_count_in: last.relation.state_in.step_count,
            x_in: last.x_i.clone(),
            construction2_input_u_i: last.construction2_u_i.clone(),
            accumulator_in_digest: last.accumulator_in_digest,
            construction2_accumulator_in_digest: last.construction2_accumulator_in_digest,
            public_trace_in_digest: last.public_trace_in_digest,
            current_boundary_in_digest: last.current_boundary_in_digest,
            chunk_count_out: self.state.chunk_count,
            step_count_out: self.state.step_count,
            x_out: last.x_out.clone(),
            accumulator_out_digest: last.accumulator_out_digest,
            construction2_accumulator_out_digest: last.construction2_accumulator_out_digest,
            public_trace_out_digest: last.public_trace_out_digest,
            current_boundary_out_digest: last.current_boundary_out_digest,
            construction2_fold: last.construction2_fold.clone(),
            final_claims: self.state.carry.claims.clone(),
            final_witnesses: final_carry_witnesses(&self.state.carry.witnesses)?,
        })
    }

    fn ensure_terminal_compression_is_proof_complete(&self) -> Result<(), DirectCcsFPrimeSnarkError> {
        let last = self.last_step.as_ref().ok_or_else(|| {
            DirectCcsFPrimeSnarkError::Input(
                "direct CCS folded compression requires at least one appended SuperNeo relation".into(),
            )
        })?;
        if self.state.chunk_count > 1 && last.construction2_fold.is_none() {
            return Err(DirectCcsFPrimeSnarkError::Input(
                "plain direct CCS terminal compression is latest-only and disabled for multi-step runs".into(),
            ));
        }
        Ok(())
    }

    fn validate_current_surface(&self) -> Result<(), DirectCcsFPrimeSnarkError> {
        let expected_accumulator_digest = direct_accumulator_digest_from_claims(&self.params, &self.state.carry.claims);
        if self.accumulator_digest != expected_accumulator_digest {
            return Err(DirectCcsFPrimeSnarkError::Input(
                "direct CCS IVC accumulator digest does not match carried CE state".into(),
            ));
        }
        let expected_x = direct_state_x_out(
            self.vk_fs_digest,
            &self.mat_digest,
            self.state.chunk_count,
            self.state.step_count,
            self.initial_boundary_digest,
            self.current_boundary_digest,
            DIRECT_CCS_TRIVIAL_PC,
            self.accumulator_digest,
            self.construction2_accumulator_digest,
            self.public_trace_digest,
        );
        if self.x_i != expected_x || self.construction2_u_i.x_i() != &self.x_i {
            return Err(DirectCcsFPrimeSnarkError::Input(
                "direct CCS IVC Construction-2 current instance does not bind to carried x_i".into(),
            ));
        }
        if self.state.chunk_count == 0 {
            if self.state.step_count != 0 || self.last_step.is_some() {
                return Err(DirectCcsFPrimeSnarkError::Input(
                    "direct CCS IVC base state cannot carry non-zero progress".into(),
                ));
            }
            if !self
                .construction2_u_i
                .is_canonical_zero_for(self.params.kappa as usize, &self.x_i)
            {
                return Err(DirectCcsFPrimeSnarkError::Input(
                    "direct CCS IVC base state must carry a canonical Construction-2 default instance".into(),
                ));
            }
        } else {
            let boundary = Construction2PublicBoundary::from_fresh_instance(&self.construction2_u_i);
            if boundary.commitment_digest != boundary.expected_commitment_digest()
                || boundary.fresh_instance_digest != boundary.expected_fresh_instance_digest()
                || !boundary.has_canonical_commitment_shape()
            {
                return Err(DirectCcsFPrimeSnarkError::Input(
                    "direct CCS IVC carried Construction-2 boundary is not canonical".into(),
                ));
            }
        }
        Ok(())
    }

    fn validate_chunk_shape(&self, chunk: &ChunkInput) -> Result<(), DirectCcsFPrimeSnarkError> {
        let expected_cols = self.structure.m.div_ceil(D);
        for step in &chunk.steps {
            if let Some(expected_m_in) = self.public_input_len {
                if step.mcs.m_in != expected_m_in {
                    return Err(DirectCcsFPrimeSnarkError::Input(format!(
                        "direct CCS step {} has m_in={}, expected fixed program public input len {}",
                        step.label, step.mcs.m_in, expected_m_in
                    )));
                }
            }
            if step.mcs.m_in != step.mcs.x.len() {
                return Err(DirectCcsFPrimeSnarkError::Input(format!(
                    "direct CCS step {} has m_in={} but {} public inputs",
                    step.label,
                    step.mcs.m_in,
                    step.mcs.x.len()
                )));
            }
            if step.mcs.m_in > self.structure.m {
                return Err(DirectCcsFPrimeSnarkError::Input(format!(
                    "direct CCS step {} has m_in={} beyond CCS columns {}",
                    step.label, step.mcs.m_in, self.structure.m
                )));
            }
            let expected_w = self.structure.m - step.mcs.m_in;
            if step.witness.w.len() != expected_w {
                return Err(DirectCcsFPrimeSnarkError::Input(format!(
                    "direct CCS step {} witness tail has len {}, expected {}",
                    step.label,
                    step.witness.w.len(),
                    expected_w
                )));
            }
            if step.witness.Z.rows() != D || step.witness.Z.cols() != expected_cols {
                return Err(DirectCcsFPrimeSnarkError::Input(format!(
                    "direct CCS step {} packed witness shape is {}x{}, expected {}x{}",
                    step.label,
                    step.witness.Z.rows(),
                    step.witness.Z.cols(),
                    D,
                    expected_cols
                )));
            }
        }
        Ok(())
    }
}

pub fn verify_direct_ccs_ivc_snark(
    state: &DirectCcsIvcState,
    proof: &DirectCcsFPrimeSnarkProof,
) -> Result<(), DirectCcsFPrimeSnarkError> {
    let circuit = state.latest_circuit()?;
    let terminal_circuit = circuit.terminal_circuit(true);
    let terminal_relation = DirectCcsTerminalCommittedRelation::from_terminal_circuit(terminal_circuit.clone())
        .map_err(|err| DirectCcsFPrimeSnarkError::Synthesis(err.to_string()))?;
    let terminal_committed_perf = terminal_relation
        .measure()
        .map_err(|err| DirectCcsFPrimeSnarkError::Synthesis(err.to_string()))?;
    let keys = setup_direct_ccs_terminal_committed_relation_cached(&terminal_relation, terminal_committed_perf)
        .map_err(|err| DirectCcsFPrimeSnarkError::Setup(err.to_string()))?;
    let expected_x_i = direct_state_x_out(
        terminal_circuit.vk_fs_digest,
        &terminal_circuit.mat_digest,
        terminal_circuit.chunk_count_out,
        terminal_circuit.step_count_out,
        terminal_circuit.initial_boundary_digest,
        terminal_circuit.current_boundary_out_digest,
        DIRECT_CCS_TRIVIAL_PC,
        terminal_circuit.accumulator_out_digest,
        terminal_circuit.construction2_accumulator_out_digest,
        terminal_circuit.public_trace_out_digest,
    );
    if proof.construction2_u_i.x_i != expected_x_i {
        return Err(DirectCcsFPrimeSnarkError::PublicIoMismatch);
    }
    verify_direct_ccs_terminal_committed_relation(
        &keys.verifier,
        &terminal_circuit.terminal_public_values(),
        &proof.construction2_u_i,
        &proof.terminal_f_prime_committed_step_proof,
    )
    .map_err(|err| DirectCcsFPrimeSnarkError::Verify(err.to_string()))
}

impl SpartanCircuit<NeoFoldDeciderEngine> for DirectCcsTerminalFPrimeCircuit {
    fn public_values(&self) -> Result<Vec<SpartanF>, SynthesisError> {
        Ok(self.terminal_public_values())
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
        let public_values = self.public_values()?;
        let public_inputs = public_values
            .into_iter()
            .enumerate()
            .map(|(idx, value)| AllocatedNum::alloc_input(cs.namespace(|| format!("public_{idx}")), || Ok(value)))
            .collect::<Result<Vec<_>, _>>()?;
        self.synthesize_body_with_public_inputs(cs, &public_inputs)
    }
}

fn prove_direct_ccs_f_prime_circuit(
    circuit: DirectCcsFPrimeCircuit,
    emit: &mut dyn FnMut(&str),
) -> Result<
    (
        DirectCcsIvcSnark,
        DirectCcsIvcSnarkVerifierKey,
        DirectCcsFPrimeSnarkPerf,
    ),
    DirectCcsFPrimeSnarkError,
> {
    let terminal_circuit = circuit.terminal_circuit(true);
    emit("direct_ccs_ivc.phase=terminal_shape_measure.start");
    let breakdown = measure_direct_ccs_f_prime_constraints(&terminal_circuit)?;
    emit("direct_ccs_ivc.phase=terminal_shape_measure.done");
    emit("direct_ccs_ivc.phase=terminal_committed_relation.start");
    let terminal_relation = DirectCcsTerminalCommittedRelation::from_terminal_circuit(terminal_circuit.clone())
        .map_err(|err| DirectCcsFPrimeSnarkError::Synthesis(err.to_string()))?;
    emit("direct_ccs_ivc.phase=terminal_committed_relation.done");
    emit("direct_ccs_ivc.phase=final_ce_measure.start");
    let final_ce_breakdown = measure_direct_final_ce_relation_breakdown(
        &circuit.params,
        &circuit.structure,
        &circuit.final_claims,
        &circuit.final_witnesses,
    )?;
    emit("direct_ccs_ivc.phase=final_ce_measure.done");

    let setup_started = Instant::now();
    emit("direct_ccs_ivc.phase=terminal_committed_measure.start");
    let terminal_committed_perf = terminal_relation
        .measure()
        .map_err(|err| DirectCcsFPrimeSnarkError::Synthesis(err.to_string()))?;
    let measure_msg = format!(
        "direct_ccs_ivc.terminal_committed_shape constraints={} public_inputs={} committed_width={} source_values={} commitment_words={}",
        terminal_committed_perf.constraints,
        terminal_committed_perf.public_inputs,
        terminal_committed_perf.committed_width,
        terminal_committed_perf.source_values,
        terminal_committed_perf.commitment_words
    );
    emit(&measure_msg);
    let log_lines = terminal_committed_perf
        .breakdown_log_lines()
        .into_iter()
        .chain(breakdown.chunk_stage_log_lines())
        .chain(breakdown.construction2_fold_breakdown.log_lines());
    for line in log_lines {
        emit(&line);
    }
    emit("direct_ccs_ivc.phase=terminal_committed_measure.done");
    emit("direct_ccs_ivc.phase=terminal_committed_setup.start");
    let setup_keys = setup_direct_ccs_terminal_committed_relation_cached(&terminal_relation, terminal_committed_perf)
        .map_err(|err| DirectCcsFPrimeSnarkError::Setup(err.to_string()))?;
    emit("direct_ccs_ivc.phase=terminal_committed_setup.done");
    let setup_ms = setup_started.elapsed().as_secs_f64() * 1_000.0;
    let terminal_committed_perf = setup_keys.perf.clone();
    let r1cs_sizes = terminal_committed_perf.sizes;
    let r1cs_nnz = terminal_committed_perf.nnz;
    let prep_ms = 0.0;
    let prove_started = Instant::now();
    emit("direct_ccs_ivc.phase=terminal_committed_prove.start");
    let (terminal_f_prime_committed_step_proof, pcs_ms) =
        prove_direct_ccs_terminal_committed_relation(&setup_keys.prover, &terminal_relation)
            .map_err(|err| DirectCcsFPrimeSnarkError::Prove(err.to_string()))?;
    emit("direct_ccs_ivc.phase=terminal_committed_prove.done");
    let prove_ms = prove_started.elapsed().as_secs_f64() * 1_000.0;
    let encode_ms = 0.0;
    let proof = DirectCcsFPrimeSnarkProof {
        construction2_u_i: terminal_relation.public_boundary().clone(),
        terminal_f_prime_committed_step_proof,
    };
    let public_image = terminal_circuit.public_image(proof.construction2_u_i.clone());
    let final_proof_bytes = bincode::serialize(&proof)
        .map_err(|err| DirectCcsFPrimeSnarkError::Encode(err.to_string()))?
        .len();
    let snark_bytes = proof.snark_bytes_len();
    let verifier_key = DirectCcsIvcSnarkVerifierKey::from_terminal_f_prime(setup_keys.verifier.clone());
    let snark = DirectCcsIvcSnark::from_parts(proof, public_image);
    Ok((
        snark,
        verifier_key,
        DirectCcsFPrimeSnarkPerf {
            setup_ms,
            prep_ms,
            prove_ms,
            encode_ms,
            total_prove_ms: prep_ms + prove_ms + encode_ms,
            total_verify_ms: 0.0,
            r1cs_sizes,
            r1cs_nnz,
            pcs_ms,
            final_proof_bytes,
            snark_bytes,
            public_inputs: terminal_committed_perf.public_inputs,
            chunk_constraints_first4: breakdown.chunk_constraints_first4,
            chunk_constraints_by_chunk: breakdown.chunk_constraints_by_chunk,
            chunk_count: breakdown.chunk_count,
            public_link_constraints: breakdown.public_link_constraints,
            construction2_fold_constraints: breakdown.construction2_fold_constraints,
            construction2_fold_final_ce_consistency_constraints: 0,
            chunk_done_constraints: breakdown.chunk_done_constraints,
            final_ce_relation_constraints: final_ce_breakdown.total_relation_constraints,
            final_ce_relation_breakdown: final_ce_breakdown.relation_breakdown,
            final_ce_bundle_constraints: 0,
            final_ce_bundle_digest_constraints: 0,
            final_ce_bundle_digest_match_constraints: 0,
            terminal_f_prime_constraints: terminal_committed_perf.constraints,
            terminal_committed_width: terminal_committed_perf.committed_width,
            terminal_commitment_words: terminal_committed_perf.commitment_words,
            terminal_source_values: terminal_committed_perf.source_values,
            terminal_source_bit_values: terminal_committed_perf.source_bit_values,
            terminal_source_u32_values: terminal_committed_perf.source_u32_values,
            terminal_source_u64_values: terminal_committed_perf.source_u64_values,
            terminal_unclassified_private_values: terminal_committed_perf.unclassified_private_values,
            terminal_committed_breakdown: terminal_committed_perf.breakdown,
            final_ce_r1cs_sizes: [0; 10],
        },
    ))
}
