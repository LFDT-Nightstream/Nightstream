//! Owns public/private step inputs, chunk inputs, carry state, and partitioning.

use neo_ajtai::Commitment;
use neo_ccs::{CcsClaim, CcsWitness, CeClaim, Mat};
use neo_math::{F, K};
use neo_reductions::error::PiCcsError;
use serde::{Deserialize, Serialize};

use super::FoldSchedule;

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
