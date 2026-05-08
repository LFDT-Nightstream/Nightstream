//! Owns the generic session proof boundary.
//!
//! Ownership:
//! - the active SuperNeo backend spine types (`StepInput`, `RunProof`, `PackagedProof`)
//!
//! It does not own:
//! - the `Π_CCS -> Π_RLC -> Π_DEC` protocol logic
//! - frontend step-build records
//! - time-opening summary surfaces
//! - VM-specific trace execution

use neo_ajtai::Commitment;
use neo_ccs::{CcsClaim, CcsWitness, CeClaim, Mat};
use neo_math::{F, K};
use neo_reductions::api::PiCcsProof;
use neo_reductions::error::PiCcsError;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StepInput {
    pub label: String,
    pub mcs: CcsClaim<Commitment, F>,
    pub witness: CcsWitness<F>,
}

impl StepInput {
    pub fn instance(&self) -> PublicStep {
        PublicStep {
            label: self.label.clone(),
            mcs: self.mcs.clone(),
        }
    }

    pub fn public(&self) -> PublicStep {
        self.instance()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PublicStep {
    pub label: String,
    pub mcs: CcsClaim<Commitment, F>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum FoldSchedule {
    WholeTrace,
    RowsPerChunk(usize),
}

impl FoldSchedule {
    pub fn validate(self) -> Result<(), PiCcsError> {
        match self {
            Self::WholeTrace => Ok(()),
            Self::RowsPerChunk(0) => Err(PiCcsError::InvalidInput(
                "RowsPerChunk(0) is not a valid fold schedule".into(),
            )),
            Self::RowsPerChunk(_) => Ok(()),
        }
    }

    pub fn meta_words(self) -> [u64; 2] {
        match self {
            Self::WholeTrace => [0, 0],
            Self::RowsPerChunk(rows) => [1, rows as u64],
        }
    }

    pub fn chunk_count(self, step_count: usize) -> Result<usize, PiCcsError> {
        self.validate()?;
        Ok(match self {
            Self::WholeTrace => usize::from(step_count != 0),
            Self::RowsPerChunk(rows) => {
                if step_count == 0 {
                    0
                } else {
                    step_count.div_ceil(rows)
                }
            }
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChunkInput {
    pub start_index: usize,
    pub steps: Vec<StepInput>,
}

impl ChunkInput {
    pub fn public(&self) -> PublicChunk {
        PublicChunk {
            start_index: self.start_index,
            steps: self.steps.iter().map(StepInput::instance).collect(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PublicChunk {
    pub start_index: usize,
    pub steps: Vec<PublicStep>,
}

#[derive(Clone, Debug)]
pub(crate) struct ProverChunkInput {
    pub public_chunk: PublicChunk,
    pub fresh_claims: Vec<CcsClaim<Commitment, F>>,
    pub fresh_witnesses: Vec<CcsWitness<F>>,
}

impl ProverChunkInput {
    pub fn start_index(&self) -> usize {
        self.public_chunk.start_index
    }

    pub fn fresh_step_count(&self) -> usize {
        self.public_chunk.steps.len()
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Carry {
    pub claims: Vec<CeClaim<Commitment, F, K>>,
    pub witnesses: Vec<Mat<F>>,
}

impl Carry {
    pub fn is_empty(&self) -> bool {
        self.claims.is_empty() && self.witnesses.is_empty()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiRlcArtifact {
    pub parent: CeClaim<Commitment, F, K>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiDecArtifact {
    pub children: Vec<CeClaim<Commitment, F, K>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChunkProof {
    pub chunk: PublicChunk,
    pub relation_digest: [u8; 32],
    pub ccs_outputs: Vec<CeClaim<Commitment, F, K>>,
    pub ccs_proof: PiCcsProof,
    pub rlc: PiRlcArtifact,
    pub dec: PiDecArtifact,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChunkResult {
    pub proof: ChunkProof,
    pub next_main: Carry,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ChunkProvePerf {
    pub start_index: usize,
    pub fresh_steps: usize,
    pub incoming_main_claims: usize,
    pub ccs_outputs: usize,
    pub dec_children: usize,
    pub prepare_inputs_ms: f64,
    pub ccs_bind_ms: f64,
    pub ccs_sample_challenges_ms: f64,
    pub ccs_fe_sumcheck_ms: f64,
    pub ccs_nc_sumcheck_ms: f64,
    pub ccs_output_materialize_ms: f64,
    pub ccs_ms: f64,
    pub dims_ms: f64,
    pub rlc_prepare_ms: f64,
    pub rlc_ms: f64,
    pub dec_split_ms: f64,
    pub dec_commit_ms: f64,
    pub dec_ms: f64,
    pub total_ms: f64,
}

#[derive(Clone, Debug, Default)]
pub struct RunProvePerf {
    pub chunks: Vec<ChunkProvePerf>,
    pub total_ms: f64,
}

impl RunProvePerf {
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    pub fn fresh_steps(&self) -> usize {
        self.chunks.iter().map(|chunk| chunk.fresh_steps).sum()
    }

    pub fn incoming_main_claims(&self) -> usize {
        self.chunks
            .iter()
            .map(|chunk| chunk.incoming_main_claims)
            .sum()
    }

    pub fn ccs_outputs(&self) -> usize {
        self.chunks.iter().map(|chunk| chunk.ccs_outputs).sum()
    }

    pub fn dec_children(&self) -> usize {
        self.chunks.iter().map(|chunk| chunk.dec_children).sum()
    }

    pub fn prepare_inputs_ms(&self) -> f64 {
        self.chunks
            .iter()
            .map(|chunk| chunk.prepare_inputs_ms)
            .sum()
    }

    pub fn ccs_ms(&self) -> f64 {
        self.chunks.iter().map(|chunk| chunk.ccs_ms).sum()
    }

    pub fn ccs_bind_ms(&self) -> f64 {
        self.chunks.iter().map(|chunk| chunk.ccs_bind_ms).sum()
    }

    pub fn ccs_sample_challenges_ms(&self) -> f64 {
        self.chunks
            .iter()
            .map(|chunk| chunk.ccs_sample_challenges_ms)
            .sum()
    }

    pub fn ccs_fe_sumcheck_ms(&self) -> f64 {
        self.chunks
            .iter()
            .map(|chunk| chunk.ccs_fe_sumcheck_ms)
            .sum()
    }

    pub fn ccs_nc_sumcheck_ms(&self) -> f64 {
        self.chunks
            .iter()
            .map(|chunk| chunk.ccs_nc_sumcheck_ms)
            .sum()
    }

    pub fn ccs_output_materialize_ms(&self) -> f64 {
        self.chunks
            .iter()
            .map(|chunk| chunk.ccs_output_materialize_ms)
            .sum()
    }

    pub fn dims_ms(&self) -> f64 {
        self.chunks.iter().map(|chunk| chunk.dims_ms).sum()
    }

    pub fn rlc_prepare_ms(&self) -> f64 {
        self.chunks.iter().map(|chunk| chunk.rlc_prepare_ms).sum()
    }

    pub fn rlc_ms(&self) -> f64 {
        self.chunks.iter().map(|chunk| chunk.rlc_ms).sum()
    }

    pub fn dec_split_ms(&self) -> f64 {
        self.chunks.iter().map(|chunk| chunk.dec_split_ms).sum()
    }

    pub fn dec_commit_ms(&self) -> f64 {
        self.chunks.iter().map(|chunk| chunk.dec_commit_ms).sum()
    }

    pub fn dec_ms(&self) -> f64 {
        self.chunks.iter().map(|chunk| chunk.dec_ms).sum()
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ChunkVerifyPerf {
    pub start_index: usize,
    pub fresh_steps: usize,
    pub incoming_main_claims: usize,
    pub ccs_outputs: usize,
    pub dec_children: usize,
    pub prepare_inputs_ms: f64,
    pub ccs_bind_ms: f64,
    pub ccs_bind_header_instances_ms: f64,
    pub ccs_bind_header_prefix_ms: f64,
    pub ccs_bind_header_poly_ms: f64,
    pub ccs_bind_header_public_instances_ms: f64,
    pub ccs_bind_me_inputs_ms: f64,
    pub ccs_bind_sample_challenges_ms: f64,
    pub ccs_fe_sumcheck_ms: f64,
    pub ccs_nc_sumcheck_ms: f64,
    pub ccs_output_checks_ms: f64,
    pub ccs_terminal_ms: f64,
    pub ccs_ms: f64,
    pub digest_checks_ms: f64,
    pub dims_ms: f64,
    pub rlc_challenge_ms: f64,
    pub rlc_rho_mats_ms: f64,
    pub rlc_rho_k_lift_ms: f64,
    pub rlc_x_ms: f64,
    pub rlc_y_ms: f64,
    pub rlc_y_zcol_ms: f64,
    pub rlc_aux_ms: f64,
    pub rlc_commitment_collect_ms: f64,
    pub rlc_commitment_mix_ms: f64,
    pub rlc_commitment_ms: f64,
    pub rlc_ms: f64,
    pub dec_ms: f64,
    pub total_ms: f64,
}

#[derive(Clone, Debug, Default)]
pub struct RunVerifyPerf {
    pub chunks: Vec<ChunkVerifyPerf>,
    pub total_ms: f64,
}

impl RunVerifyPerf {
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    pub fn fresh_steps(&self) -> usize {
        self.chunks.iter().map(|chunk| chunk.fresh_steps).sum()
    }

    pub fn incoming_main_claims(&self) -> usize {
        self.chunks
            .iter()
            .map(|chunk| chunk.incoming_main_claims)
            .sum()
    }

    pub fn ccs_outputs(&self) -> usize {
        self.chunks.iter().map(|chunk| chunk.ccs_outputs).sum()
    }

    pub fn dec_children(&self) -> usize {
        self.chunks.iter().map(|chunk| chunk.dec_children).sum()
    }

    pub fn prepare_inputs_ms(&self) -> f64 {
        self.chunks
            .iter()
            .map(|chunk| chunk.prepare_inputs_ms)
            .sum()
    }

    pub fn ccs_ms(&self) -> f64 {
        self.chunks.iter().map(|chunk| chunk.ccs_ms).sum()
    }

    pub fn ccs_bind_ms(&self) -> f64 {
        self.chunks.iter().map(|chunk| chunk.ccs_bind_ms).sum()
    }

    pub fn ccs_bind_header_instances_ms(&self) -> f64 {
        self.chunks
            .iter()
            .map(|chunk| chunk.ccs_bind_header_instances_ms)
            .sum()
    }

    pub fn ccs_bind_header_prefix_ms(&self) -> f64 {
        self.chunks
            .iter()
            .map(|chunk| chunk.ccs_bind_header_prefix_ms)
            .sum()
    }

    pub fn ccs_bind_header_poly_ms(&self) -> f64 {
        self.chunks
            .iter()
            .map(|chunk| chunk.ccs_bind_header_poly_ms)
            .sum()
    }

    pub fn ccs_bind_header_public_instances_ms(&self) -> f64 {
        self.chunks
            .iter()
            .map(|chunk| chunk.ccs_bind_header_public_instances_ms)
            .sum()
    }

    pub fn ccs_bind_me_inputs_ms(&self) -> f64 {
        self.chunks
            .iter()
            .map(|chunk| chunk.ccs_bind_me_inputs_ms)
            .sum()
    }

    pub fn ccs_bind_sample_challenges_ms(&self) -> f64 {
        self.chunks
            .iter()
            .map(|chunk| chunk.ccs_bind_sample_challenges_ms)
            .sum()
    }

    pub fn ccs_fe_sumcheck_ms(&self) -> f64 {
        self.chunks
            .iter()
            .map(|chunk| chunk.ccs_fe_sumcheck_ms)
            .sum()
    }

    pub fn ccs_nc_sumcheck_ms(&self) -> f64 {
        self.chunks
            .iter()
            .map(|chunk| chunk.ccs_nc_sumcheck_ms)
            .sum()
    }

    pub fn ccs_output_checks_ms(&self) -> f64 {
        self.chunks
            .iter()
            .map(|chunk| chunk.ccs_output_checks_ms)
            .sum()
    }

    pub fn ccs_terminal_ms(&self) -> f64 {
        self.chunks.iter().map(|chunk| chunk.ccs_terminal_ms).sum()
    }

    pub fn digest_checks_ms(&self) -> f64 {
        self.chunks.iter().map(|chunk| chunk.digest_checks_ms).sum()
    }

    pub fn dims_ms(&self) -> f64 {
        self.chunks.iter().map(|chunk| chunk.dims_ms).sum()
    }

    pub fn rlc_challenge_ms(&self) -> f64 {
        self.chunks.iter().map(|chunk| chunk.rlc_challenge_ms).sum()
    }

    pub fn rlc_x_ms(&self) -> f64 {
        self.chunks.iter().map(|chunk| chunk.rlc_x_ms).sum()
    }

    pub fn rlc_rho_mats_ms(&self) -> f64 {
        self.chunks.iter().map(|chunk| chunk.rlc_rho_mats_ms).sum()
    }

    pub fn rlc_rho_k_lift_ms(&self) -> f64 {
        self.chunks
            .iter()
            .map(|chunk| chunk.rlc_rho_k_lift_ms)
            .sum()
    }

    pub fn rlc_y_ms(&self) -> f64 {
        self.chunks.iter().map(|chunk| chunk.rlc_y_ms).sum()
    }

    pub fn rlc_y_zcol_ms(&self) -> f64 {
        self.chunks.iter().map(|chunk| chunk.rlc_y_zcol_ms).sum()
    }

    pub fn rlc_aux_ms(&self) -> f64 {
        self.chunks.iter().map(|chunk| chunk.rlc_aux_ms).sum()
    }

    pub fn rlc_commitment_collect_ms(&self) -> f64 {
        self.chunks
            .iter()
            .map(|chunk| chunk.rlc_commitment_collect_ms)
            .sum()
    }

    pub fn rlc_commitment_mix_ms(&self) -> f64 {
        self.chunks
            .iter()
            .map(|chunk| chunk.rlc_commitment_mix_ms)
            .sum()
    }

    pub fn rlc_commitment_ms(&self) -> f64 {
        self.chunks
            .iter()
            .map(|chunk| chunk.rlc_commitment_ms)
            .sum()
    }

    pub fn rlc_ms(&self) -> f64 {
        self.chunks.iter().map(|chunk| chunk.rlc_ms).sum()
    }

    pub fn dec_ms(&self) -> f64 {
        self.chunks.iter().map(|chunk| chunk.dec_ms).sum()
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RunProof {
    pub fold_schedule: FoldSchedule,
    pub chunks: Vec<ChunkProof>,
    pub final_main_claims: Vec<CeClaim<Commitment, F, K>>,
}

impl Default for FoldSchedule {
    fn default() -> Self {
        Self::RowsPerChunk(1)
    }
}

impl RunProof {
    pub fn public_step_count(&self) -> usize {
        self.chunks
            .iter()
            .map(|chunk| chunk.chunk.steps.len())
            .sum()
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PublicStatement {
    pub fold_schedule: FoldSchedule,
    pub chunk_count: u64,
    pub chunks: Vec<PublicChunk>,
    pub final_main_claims: Vec<CeClaim<Commitment, F, K>>,
    pub digest: [u8; 32],
}

impl PublicStatement {
    pub fn public_step_count(&self) -> usize {
        self.chunks.iter().map(|chunk| chunk.steps.len()).sum()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FinalProof {
    pub session: RunProof,
    pub proof_digest: [u8; 32],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PackagedProof {
    pub statement: PublicStatement,
    pub proof: FinalProof,
}

pub fn partition_step_inputs(schedule: FoldSchedule, steps: Vec<StepInput>) -> Result<Vec<ChunkInput>, PiCcsError> {
    partition_items(schedule, steps, |start_index, steps| ChunkInput { start_index, steps })
}

pub fn partition_public_steps(schedule: FoldSchedule, steps: Vec<PublicStep>) -> Result<Vec<PublicChunk>, PiCcsError> {
    partition_items(schedule, steps, |start_index, steps| PublicChunk { start_index, steps })
}

pub(crate) fn partition_prover_step_inputs(
    schedule: FoldSchedule,
    steps: Vec<StepInput>,
) -> Result<Vec<ProverChunkInput>, PiCcsError> {
    partition_items(schedule, steps, |start_index, steps| {
        let mut public_steps = Vec::with_capacity(steps.len());
        let mut fresh_claims = Vec::with_capacity(steps.len());
        let mut fresh_witnesses = Vec::with_capacity(steps.len());
        for StepInput { label, mcs, witness } in steps {
            public_steps.push(PublicStep {
                label,
                mcs: mcs.clone(),
            });
            fresh_claims.push(mcs);
            fresh_witnesses.push(witness);
        }
        ProverChunkInput {
            public_chunk: PublicChunk {
                start_index,
                steps: public_steps,
            },
            fresh_claims,
            fresh_witnesses,
        }
    })
}

fn partition_items<T, C, FBuild>(schedule: FoldSchedule, items: Vec<T>, build: FBuild) -> Result<Vec<C>, PiCcsError>
where
    FBuild: Fn(usize, Vec<T>) -> C,
{
    schedule.validate()?;
    if items.is_empty() {
        return Ok(Vec::new());
    }

    let chunk_len = match schedule {
        FoldSchedule::WholeTrace => items.len(),
        FoldSchedule::RowsPerChunk(rows) => rows,
    };

    let mut chunks = Vec::with_capacity(schedule.chunk_count(items.len())?);
    let mut next_start = 0usize;
    let mut cursor = items.into_iter();
    loop {
        let mut chunk_items = Vec::with_capacity(chunk_len);
        for _ in 0..chunk_len {
            match cursor.next() {
                Some(item) => chunk_items.push(item),
                None => break,
            }
        }
        if chunk_items.is_empty() {
            break;
        }
        let chunk_size = chunk_items.len();
        chunks.push(build(next_start, chunk_items));
        next_start += chunk_size;
    }
    Ok(chunks)
}
