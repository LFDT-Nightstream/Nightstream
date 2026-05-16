//! Owns the one-step RV32IM chunk-fold relation surface.
//!
//! This module exposes the verified one-step boundary needed by a future
//! fixed-shape recursive consumer. It does not own final-proof packaging or
//! Spartan circuit synthesis.

use serde::{Deserialize, Serialize};

use crate::chunk_folding::ChunkReplayWitness;
use crate::finalize::FixedShapeChunkSummary;
use crate::proof::Carry;
use crate::rv32im::final_relation::{
    build_rv32im_chunk_fold_step_traces, Rv32imChunkFoldState, Rv32imChunkFoldTranscriptSnapshot,
    Rv32imFinalBuildProof, Rv32imFinalStatement,
};
use crate::rv32im::kernel::{
    rv32im_cached_root_main_lane_optimized_cache, Rv32imVerifiedKernelChunkHandoff, SimpleKernelError,
};
use neo_transcript::Poseidon2Transcript;

use super::fold::{
    verify_rv32im_chunk_fold_verifier_step, Rv32imAccumulatorHandle, Rv32imChunkFoldCarry, Rv32imChunkStepPublic,
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rv32imChunkStepRelationStatement {
    pub step_public: Rv32imChunkStepPublic,
    pub chunk_summary: FixedShapeChunkSummary,
    pub transcript_in: Rv32imChunkFoldTranscriptSnapshot,
    pub transcript_out: Rv32imChunkFoldTranscriptSnapshot,
}

#[derive(Clone, Debug)]
pub struct Rv32imChunkStepRelationWitness {
    pub handoff: Rv32imVerifiedKernelChunkHandoff,
    pub carry_in: Carry,
    pub carry_out: Carry,
    pub replay_witness: ChunkReplayWitness,
}

#[derive(Clone, Debug)]
pub struct Rv32imChunkStepRelation {
    pub statement: Rv32imChunkStepRelationStatement,
    pub witness: Rv32imChunkStepRelationWitness,
}

pub fn build_rv32im_chunk_step_relations(
    statement: &Rv32imFinalStatement,
    proof: &Rv32imFinalBuildProof,
) -> Result<Vec<Rv32imChunkStepRelation>, SimpleKernelError> {
    Ok(build_rv32im_chunk_fold_step_traces(statement, proof)?
        .into_iter()
        .map(|trace| Rv32imChunkStepRelation {
            statement: Rv32imChunkStepRelationStatement {
                step_public: trace.step_public,
                chunk_summary: trace.chunk_summary,
                transcript_in: trace.transcript_in,
                transcript_out: trace.transcript_out,
            },
            witness: Rv32imChunkStepRelationWitness {
                handoff: trace.handoff,
                carry_in: trace.carry_in.main,
                carry_out: trace.carry_out.main,
                replay_witness: trace.replay_witness,
            },
        })
        .collect())
}

pub fn verify_rv32im_chunk_step_relation(
    statement: &Rv32imChunkStepRelationStatement,
    witness: &Rv32imChunkStepRelationWitness,
) -> Result<Rv32imChunkFoldState, SimpleKernelError> {
    validate_rv32im_chunk_step_relation_surface(statement, witness)?;
    let (params, log, structure) =
        crate::rv32im::kernel::rv32im_root_main_lane_context_for_claim_count(witness.carry_in.claims.len())?;
    let optimized_cache = rv32im_cached_root_main_lane_optimized_cache()?;
    let mut transcript =
        Poseidon2Transcript::from_state_and_absorbed(statement.transcript_in.state, statement.transcript_in.absorbed);
    let carry_in = Rv32imChunkFoldCarry::from_main(
        witness.carry_in.clone(),
        Rv32imAccumulatorHandle(statement.step_public.state_in),
    );
    let step = verify_rv32im_chunk_fold_verifier_step(
        statement.step_public.program_digest,
        statement.step_public.chunk_index as usize,
        statement.step_public.halted_out,
        &witness.handoff,
        &carry_in,
        &witness.replay_witness,
        &mut transcript,
        &params,
        structure,
        log,
        &optimized_cache,
    )?;
    if step.step_public != statement.step_public {
        return Err(SimpleKernelError::Bridge(
            "RV32IM chunk-step relation public step does not match the verified one-step fold output".into(),
        ));
    }
    let expected_summary = FixedShapeChunkSummary::from_public_chunk(
        &witness.handoff.public_chunk,
        step.public_chunk_digest,
        step.chunk_relation_digest,
    );
    if expected_summary != statement.chunk_summary {
        return Err(SimpleKernelError::Bridge(
            "RV32IM chunk-step relation summary does not match the verified one-step fold output".into(),
        ));
    }
    if step.next_carry.main.claims != witness.carry_out.claims
        || step.next_carry.main.witnesses != witness.carry_out.witnesses
    {
        return Err(SimpleKernelError::Bridge(
            "RV32IM chunk-step relation next carry does not match the carried witness state".into(),
        ));
    }
    let transcript_out = Rv32imChunkFoldTranscriptSnapshot {
        state: transcript.state(),
        absorbed: transcript.absorbed(),
    };
    if transcript_out != statement.transcript_out {
        return Err(SimpleKernelError::Bridge(
            "RV32IM chunk-step relation transcript_out does not match the verified one-step fold output".into(),
        ));
    }
    Ok(Rv32imChunkFoldState {
        carry: Rv32imChunkFoldCarry::from_main(witness.carry_out.clone(), step.next_carry.terminal_handle),
        transcript: transcript_out,
    })
}

pub fn validate_rv32im_chunk_step_relation_surface(
    statement: &Rv32imChunkStepRelationStatement,
    witness: &Rv32imChunkStepRelationWitness,
) -> Result<(), SimpleKernelError> {
    let expected_step_count = statement
        .step_public
        .step_hi
        .checked_sub(statement.step_public.step_lo)
        .ok_or_else(|| SimpleKernelError::Bridge("RV32IM chunk-step relation step bounds are not monotone".into()))?;
    if statement.chunk_summary.start_index != statement.step_public.step_lo {
        return Err(SimpleKernelError::Bridge(
            "RV32IM chunk-step relation summary start index does not match step_public.step_lo".into(),
        ));
    }
    if statement.chunk_summary.public_step_count != expected_step_count {
        return Err(SimpleKernelError::Bridge(
            "RV32IM chunk-step relation summary public_step_count does not match step_public span".into(),
        ));
    }
    if witness.handoff.public_chunk.start_index as u64 != statement.step_public.step_lo {
        return Err(SimpleKernelError::Bridge(
            "RV32IM chunk-step relation handoff start index does not match step_public.step_lo".into(),
        ));
    }
    if witness.handoff.public_chunk.steps.len() as u64 != expected_step_count {
        return Err(SimpleKernelError::Bridge(
            "RV32IM chunk-step relation handoff step count does not match step_public span".into(),
        ));
    }
    if statement.transcript_in.absorbed > neo_params::poseidon2_goldilocks::RATE
        || statement.transcript_out.absorbed > neo_params::poseidon2_goldilocks::RATE
    {
        return Err(SimpleKernelError::Bridge(
            "RV32IM chunk-step relation transcript snapshot absorbed count exceeds the Poseidon2 rate".into(),
        ));
    }
    Ok(())
}
