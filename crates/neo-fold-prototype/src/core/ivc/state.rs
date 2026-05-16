//! Owns native SuperNeo IVC state append operations.

use neo_ajtai::Commitment;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsStructure, Mat};
use neo_math::F;
use neo_params::NeoParams;
use neo_reductions::error::PiCcsError;
use neo_reductions::optimized_engine::OptimizedStructureCache;

use super::support::ivc_protocol_error;
use super::transcript::{
    accumulator_handle_fields, append_chunk_done, session_transcript, transcript_from_snapshot, transcript_snapshot,
};
use super::types::{SuperNeoIvcState, SuperNeoIvcStepRelation};
use crate::chunk_folding::{prove_superneo_chunk_step, SuperNeoChunkStep};
use crate::finalize::public_chunk_digest;
use crate::proof::{Carry, ChunkInput};
use crate::prover::CommitmentMixers;

impl SuperNeoIvcState {
    pub fn seed() -> Self {
        Self::seed_with_carry(Carry::default())
    }

    pub fn seed_with_carry(carry: Carry) -> Self {
        let transcript = session_transcript();
        Self {
            chunk_count: 0,
            step_count: 0,
            carry,
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
        let superneo = prove_superneo_chunk_step(
            &mut transcript,
            params,
            structure,
            &chunk,
            &self.carry,
            log,
            mixers,
            optimized_cache,
            None,
            None,
        )?;
        append_chunk_done(&mut transcript);

        let SuperNeoChunkStep {
            replay_witness,
            relation_result,
            fold_digest,
            perf,
        } = superneo;
        let next_main = relation_result.next_main;
        let chunk_relation_digest = relation_result.artifacts.relation_digest;

        let next = Self {
            chunk_count: self.chunk_count + 1,
            step_count: self
                .step_count
                .checked_add(chunk.steps.len() as u64)
                .ok_or_else(|| ivc_protocol_error("SuperNeo IVC step_count overflow"))?,
            carry: next_main,
            transcript: transcript_snapshot(&transcript),
        };
        let relation = SuperNeoIvcStepRelation {
            chunk_index: self.chunk_count,
            chunk,
            state_in: self.clone(),
            state_out: next.clone(),
            replay_witness,
            fold_digest,
            chunk_relation_digest,
            perf,
        };
        Ok((next, relation))
    }

    pub fn append_chunk_with_perf_and_accumulator_handle<L, MR, MB>(
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
        let chunk_digest = public_chunk_digest(&chunk.public());
        let accumulator_handle = accumulator_handle_fields(params, &self.carry);
        let superneo = prove_superneo_chunk_step(
            &mut transcript,
            params,
            structure,
            &chunk,
            &self.carry,
            log,
            mixers,
            optimized_cache,
            Some(chunk_digest),
            Some(accumulator_handle),
        )?;
        append_chunk_done(&mut transcript);

        let SuperNeoChunkStep {
            replay_witness,
            relation_result,
            fold_digest,
            perf,
        } = superneo;
        let next_main = relation_result.next_main;
        let chunk_relation_digest = relation_result.artifacts.relation_digest;

        let next = Self {
            chunk_count: self.chunk_count + 1,
            step_count: self
                .step_count
                .checked_add(chunk.steps.len() as u64)
                .ok_or_else(|| ivc_protocol_error("SuperNeo IVC step_count overflow"))?,
            carry: next_main,
            transcript: transcript_snapshot(&transcript),
        };
        let relation = SuperNeoIvcStepRelation {
            chunk_index: self.chunk_count,
            chunk,
            state_in: self.clone(),
            state_out: next.clone(),
            replay_witness,
            fold_digest,
            chunk_relation_digest,
            perf,
        };
        Ok((next, relation))
    }
}
