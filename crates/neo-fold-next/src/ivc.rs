//! Owns the generic native SuperNeo IVC/NIFS carrier.
//!
//! This module threads the algebraic SuperNeo accumulator through chunks:
//! `CE(b)^k + CCS^K -> CE(b)^k`. It intentionally does not own HyperNova
//! Construction-2 hash images, application step semantics, or Spartan
//! compression circuits.

use std::time::Instant;

use neo_ajtai::Commitment;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsStructure, Mat};
use neo_math::F;
use neo_params::NeoParams;
use neo_reductions::error::PiCcsError;
use neo_reductions::optimized_engine::OptimizedStructureCache;
use neo_transcript::{Poseidon2Transcript, Transcript};

use crate::chunk_relation::{
    compute_chunk_replay_witness_and_relation_with_instance_digest_and_perf,
    verify_chunk_relation_with_witness_and_instance_digest_with_perf, ChunkReplayWitness,
};
use crate::proof::{partition_step_inputs, Carry, ChunkInput, ChunkProvePerf, FoldSchedule, StepInput};
use crate::prover::CommitmentMixers;

#[derive(Clone, Debug, PartialEq)]
pub struct SuperNeoIvcTranscriptSnapshot {
    pub state: [F; neo_params::poseidon2_goldilocks::WIDTH],
    pub absorbed: usize,
}

#[derive(Clone, Debug)]
pub struct SuperNeoIvcState {
    pub chunk_count: u64,
    pub step_count: u64,
    pub carry: Carry,
    pub transcript: SuperNeoIvcTranscriptSnapshot,
}

#[derive(Clone, Debug)]
pub struct SuperNeoIvcStepRelation {
    pub chunk_index: u64,
    pub chunk: ChunkInput,
    pub state_in: SuperNeoIvcState,
    pub state_out: SuperNeoIvcState,
    pub replay_witness: ChunkReplayWitness,
    pub fold_digest: [u8; 32],
    pub chunk_relation_digest: [u8; 32],
    pub perf: ChunkProvePerf,
}

#[derive(Clone, Debug)]
pub struct SuperNeoIvcBuild {
    pub relations: Vec<SuperNeoIvcStepRelation>,
    pub final_state: SuperNeoIvcState,
    pub cache_build_ms: f64,
    pub total_ms: f64,
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

fn session_transcript() -> Poseidon2Transcript {
    Poseidon2Transcript::new(b"neo.fold.next/session")
}

fn transcript_from_snapshot(snapshot: &SuperNeoIvcTranscriptSnapshot) -> Poseidon2Transcript {
    Poseidon2Transcript::from_state_and_absorbed(snapshot.state, snapshot.absorbed)
}

fn transcript_snapshot(transcript: &Poseidon2Transcript) -> SuperNeoIvcTranscriptSnapshot {
    SuperNeoIvcTranscriptSnapshot {
        state: transcript.state(),
        absorbed: transcript.absorbed(),
    }
}

fn append_chunk_done(transcript: &mut Poseidon2Transcript) {
    transcript.append_message(b"neo.fold.next/chunk_done", &[1]);
}

fn carry_matches(left: &Carry, right: &Carry) -> bool {
    left.claims == right.claims && left.witnesses == right.witnesses
}

fn ivc_protocol_error(message: impl Into<String>) -> PiCcsError {
    PiCcsError::ProtocolError(message.into())
}

impl SuperNeoIvcState {
    pub fn seed() -> Self {
        let transcript = session_transcript();
        Self {
            chunk_count: 0,
            step_count: 0,
            carry: Carry::default(),
            transcript: transcript_snapshot(&transcript),
        }
    }

    pub fn append_chunk_with_perf<L, MR, MB>(
        &self,
        params: &NeoParams,
        structure: &CcsStructure<F>,
        chunk: ChunkInput,
        log: &L,
        mixers: CommitmentMixers<MR, MB>,
        optimized_cache: &OptimizedStructureCache,
    ) -> Result<(Self, SuperNeoIvcStepRelation), PiCcsError>
    where
        L: SModuleHomomorphism<F, Commitment> + Sync,
        MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        if chunk.start_index as u64 != self.step_count {
            return Err(ivc_protocol_error(format!(
                "SuperNeo IVC chunk start {} does not match carried step_count {}",
                chunk.start_index, self.step_count
            )));
        }

        let mut transcript = transcript_from_snapshot(&self.transcript);
        let ((replay_witness, relation_result, fold_digest), perf) =
            compute_chunk_replay_witness_and_relation_with_instance_digest_and_perf(
                &mut transcript,
                params,
                structure,
                &chunk,
                &self.carry,
                log,
                mixers,
                optimized_cache,
                None,
            )?;
        append_chunk_done(&mut transcript);

        let next = Self {
            chunk_count: self.chunk_count + 1,
            step_count: self
                .step_count
                .checked_add(chunk.steps.len() as u64)
                .ok_or_else(|| ivc_protocol_error("SuperNeo IVC step_count overflow"))?,
            carry: relation_result.next_main,
            transcript: transcript_snapshot(&transcript),
        };
        let relation = SuperNeoIvcStepRelation {
            chunk_index: self.chunk_count,
            chunk,
            state_in: self.clone(),
            state_out: next.clone(),
            replay_witness,
            fold_digest,
            chunk_relation_digest: relation_result.artifacts.relation_digest,
            perf,
        };
        Ok((next, relation))
    }
}

impl SuperNeoIvcStepRelation {
    pub fn verify<L, MR, MB>(
        &self,
        params: &NeoParams,
        structure: &CcsStructure<F>,
        log: &L,
        mixers: CommitmentMixers<MR, MB>,
        optimized_cache: &OptimizedStructureCache,
    ) -> Result<ChunkProvePerf, PiCcsError>
    where
        L: SModuleHomomorphism<F, Commitment> + Sync,
        MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        if self.chunk_index != self.state_in.chunk_count {
            return Err(ivc_protocol_error(
                "SuperNeo IVC relation chunk_index does not match state_in chunk_count",
            ));
        }
        if self.chunk.start_index as u64 != self.state_in.step_count {
            return Err(ivc_protocol_error(
                "SuperNeo IVC relation chunk start does not match state_in step_count",
            ));
        }

        let mut transcript = transcript_from_snapshot(&self.state_in.transcript);
        let ((relation_result, fold_digest), perf) = verify_chunk_relation_with_witness_and_instance_digest_with_perf(
            &mut transcript,
            params,
            structure,
            &self.chunk,
            &self.state_in.carry,
            &self.replay_witness,
            log,
            mixers,
            optimized_cache,
            None,
        )?;
        if fold_digest != self.fold_digest {
            return Err(ivc_protocol_error(
                "SuperNeo IVC relation fold digest does not match verified transcript",
            ));
        }
        if relation_result.artifacts.relation_digest != self.chunk_relation_digest {
            return Err(ivc_protocol_error(
                "SuperNeo IVC relation digest does not match verified chunk relation",
            ));
        }
        append_chunk_done(&mut transcript);
        let expected_state_out = SuperNeoIvcState {
            chunk_count: self.state_in.chunk_count + 1,
            step_count: self.state_in.step_count + self.chunk.steps.len() as u64,
            carry: relation_result.next_main,
            transcript: transcript_snapshot(&transcript),
        };
        if expected_state_out.chunk_count != self.state_out.chunk_count
            || expected_state_out.step_count != self.state_out.step_count
            || expected_state_out.transcript != self.state_out.transcript
            || !carry_matches(&expected_state_out.carry, &self.state_out.carry)
        {
            return Err(ivc_protocol_error(
                "SuperNeo IVC relation state_out does not match verified NIFS.V output",
            ));
        }
        Ok(perf)
    }
}

pub fn build_superneo_ivc_relations_with_perf<L, MR, MB>(
    schedule: FoldSchedule,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    steps: impl IntoIterator<Item = StepInput>,
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
) -> Result<SuperNeoIvcBuild, PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    let total_started = Instant::now();
    let cache_started = Instant::now();
    let optimized_cache = OptimizedStructureCache::build(structure)?;
    let cache_build_ms = elapsed_ms(cache_started);

    let mut state = SuperNeoIvcState::seed();
    let mut relations = Vec::new();
    for chunk in partition_step_inputs(schedule, steps.into_iter().collect())? {
        let (next_state, relation) =
            state.append_chunk_with_perf(params, structure, chunk, log, mixers, &optimized_cache)?;
        relation.verify(params, structure, log, mixers, &optimized_cache)?;
        state = next_state;
        relations.push(relation);
    }

    Ok(SuperNeoIvcBuild {
        relations,
        final_state: state,
        cache_build_ms,
        total_ms: elapsed_ms(total_started),
    })
}
