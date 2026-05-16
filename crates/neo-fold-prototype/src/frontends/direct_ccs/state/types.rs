//! Core direct-CCS state and proof data structures.
//!
//! This file owns data shape only. Flow lives in sibling files such as
//! `init`, `append`, `compress`, and `summary`.

use super::*;

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
    pub timing: DirectCcsFPrimeTimingPerf,
    pub proof: DirectCcsFPrimeProofSizePerf,
    pub r1cs: DirectCcsFPrimeR1csPerf,
    pub chunks: DirectCcsFPrimeChunkPerf,
    pub constraints: DirectCcsFPrimeConstraintPerf,
    pub committed: DirectCcsFPrimeCommittedPerf,
    pub final_ce: DirectCcsFPrimeFinalCePerf,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct DirectCcsFPrimeTimingPerf {
    pub setup_ms: f64,
    pub prep_ms: f64,
    pub prove_ms: f64,
    pub encode_ms: f64,
    pub total_prove_ms: f64,
    pub total_verify_ms: f64,
    pub pcs_ms: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DirectCcsFPrimeProofSizePerf {
    pub final_proof_bytes: usize,
    pub snark_bytes: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DirectCcsFPrimeR1csPerf {
    pub sizes: [usize; 10],
    pub nnz: usize,
    pub public_inputs: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DirectCcsFPrimeChunkPerf {
    pub constraints_first4: [usize; 4],
    pub constraints_by_chunk: Vec<usize>,
    pub count: usize,
    pub done_constraints: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DirectCcsFPrimeConstraintPerf {
    pub public_link: usize,
    pub construction2_fold: usize,
    pub construction2_fold_final_ce_consistency: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DirectCcsFPrimeCommittedPerf {
    pub constraints: usize,
    pub width: usize,
    pub commitment_words: usize,
    pub source: DirectCcsFPrimeCommittedSourcePerf,
    pub breakdown: DirectCcsTerminalCommittedConstraintBreakdown,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DirectCcsFPrimeCommittedSourcePerf {
    pub values: usize,
    pub bit_values: usize,
    pub u32_values: usize,
    pub u64_values: usize,
    pub unclassified_private_values: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DirectCcsFPrimeFinalCePerf {
    pub relation_constraints: usize,
    pub relation_breakdown: PaperCeRelationConstraintBreakdown,
    pub bundle_constraints: usize,
    pub bundle_digest_constraints: usize,
    pub bundle_digest_match_constraints: usize,
    pub r1cs_sizes: [usize; 10],
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
    pub(crate) params: NeoParams,
    pub(crate) structure: CcsStructure<F>,
    pub(crate) public_input_len: Option<usize>,
    pub(crate) dims: Dims,
    pub(crate) mat_digest: [Goldilocks; 4],
    pub(crate) vk_fs_digest: [u8; 32],
    pub(crate) initial_boundary_digest: [u8; 32],
    pub(crate) current_boundary_digest: [u8; 32],
    pub(crate) optimized_cache: OptimizedStructureCache,
    pub(crate) state: SuperNeoIvcState,
    pub(crate) accumulator_digest: [u8; 32],
    pub(crate) construction2_accumulator_digest: [u8; 32],
    pub(crate) public_trace_digest: [u8; 32],
    pub(crate) x_i: Construction2EncodedPublicInput,
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
    pub(crate) final_claims: Vec<CeClaim<Commitment, F, K>>,
    pub(crate) final_witnesses: Vec<CcsWitness<F>>,
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
