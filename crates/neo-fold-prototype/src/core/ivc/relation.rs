//! Owns native SuperNeo IVC relation verification.

use neo_ajtai::Commitment;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsStructure, Mat};
use neo_math::F;
use neo_params::NeoParams;
use neo_reductions::error::PiCcsError;
use neo_reductions::optimized_engine::OptimizedStructureCache;

use super::support::{carry_matches, ivc_protocol_error};
use super::transcript::{accumulator_handle_fields, append_chunk_done, transcript_from_snapshot, transcript_snapshot};
use super::types::{SuperNeoIvcState, SuperNeoIvcStepRelation};
use crate::chunk_folding::{
    verify_chunk_relation_with_witness_and_instance_digest_and_me_input_handle_with_perf,
    verify_chunk_relation_with_witness_and_instance_digest_with_perf,
};
use crate::finalize::public_chunk_digest;
use crate::proof::ChunkProvePerf;
use crate::prover::CommitmentMixers;

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

    pub fn verify_with_accumulator_handle<L, MR, MB>(
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
        let chunk_digest = public_chunk_digest(&self.chunk.public());
        let accumulator_handle = accumulator_handle_fields(params, &self.state_in.carry);
        let ((relation_result, fold_digest), perf) =
            verify_chunk_relation_with_witness_and_instance_digest_and_me_input_handle_with_perf(
                &mut transcript,
                params,
                structure,
                &self.chunk,
                &self.state_in.carry,
                &self.replay_witness,
                log,
                mixers,
                optimized_cache,
                chunk_digest,
                accumulator_handle,
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
