//! Owns audit helpers for the RV64IM terminal main-decider surface.

use crate::rv64im::chunk_fold_step::verify_rv64im_chunk_fold_verifier_step;
use crate::rv64im::chunk_relation::rv64im_chunk_relation_digest_from_fold_digest;
use crate::rv64im::chunk_step_ivc::{
    build_rv64im_chunk_step_ivc_statement_from_authoritative_parts, validate_rv64im_chunk_step_ivc_surface,
    Rv64imChunkStepIvcRelation, Rv64imChunkStepIvcStatement, Rv64imChunkStepIvcWitness,
};
use crate::rv64im::final_relation::{rv64im_chunk_fold_carried_transcript_snapshot, Rv64imChunkFoldState};
use crate::rv64im::kernel::{rv64im_cached_root_main_lane_context, rv64im_cached_root_main_lane_optimized_cache};
use crate::rv64im::main_proof::Rv64imAccumulatorPublicStatement;
use crate::rv64im::recursion_spartan::{
    validate_rv64im_main_recursion_accumulator_witness_against_published_statement,
    Rv64imMainRecursionAccumulatorWitness,
};
use crate::rv64im::SimpleKernelError;

pub use crate::rv64im::decider::{
    build_rv64im_published_proof_seam, build_rv64im_published_proof_seam_with_perf,
    prove_rv64im_public_proof_and_published_seam_with_perf, Rv64imPublicProofAndSeamBuildPerf,
    Rv64imPublishedProofSeam, Rv64imPublishedProofSeamBuildPerf,
};
pub use crate::rv64im::ivc_snark::{
    build_rv64im_terminal_decider_setup_shape_from_components, debug_check_rv64im_terminal_decider_circuit,
    Rv64imTerminalDeciderSetupShape,
};

fn build_rv64im_terminal_step_witness_from_accumulator_witness(
    accumulator_witness: &Rv64imMainRecursionAccumulatorWitness,
) -> Rv64imChunkStepIvcWitness {
    Rv64imChunkStepIvcWitness {
        handoff: accumulator_witness.handoff().clone(),
        state_in: Rv64imChunkFoldState {
            carry: accumulator_witness.running_last().clone(),
            transcript: accumulator_witness.transcript_in().clone(),
        },
        state_out: Rv64imChunkFoldState {
            carry: accumulator_witness.running_final().clone(),
            transcript: rv64im_chunk_fold_carried_transcript_snapshot(accumulator_witness.transcript_out()),
        },
        replay_witness: accumulator_witness.final_fold_witness().clone(),
        terminal_step: accumulator_witness.halted_out(),
    }
}

pub fn build_rv64im_terminal_step_statement_from_accumulator_witness_transport(
    accumulator_witness: &Rv64imMainRecursionAccumulatorWitness,
) -> Rv64imChunkStepIvcStatement {
    let chunk_relation_digest = rv64im_chunk_relation_digest_from_fold_digest(
        accumulator_witness.handoff().public_chunk_digest,
        accumulator_witness
            .final_fold_witness()
            .ccs_replay_proof
            .header_digest,
        accumulator_witness.handoff().bridge_handoff.digest,
    );
    let witness = build_rv64im_terminal_step_witness_from_accumulator_witness(accumulator_witness);
    build_rv64im_chunk_step_ivc_statement_from_authoritative_parts(
        accumulator_witness.public_statement_digest(),
        &witness,
        chunk_relation_digest,
    )
}

pub fn build_rv64im_terminal_decider_relation_from_accumulator_witness(
    published_statement: &Rv64imAccumulatorPublicStatement,
    accumulator_witness: &Rv64imMainRecursionAccumulatorWitness,
) -> Result<Rv64imChunkStepIvcRelation, SimpleKernelError> {
    validate_rv64im_main_recursion_accumulator_witness_against_published_statement(
        published_statement,
        accumulator_witness,
    )?;
    let step_program_digest = accumulator_witness.public_statement_digest();
    let (params, log, structure) = rv64im_cached_root_main_lane_context()?;
    let optimized_cache = rv64im_cached_root_main_lane_optimized_cache()?;
    let mut transcript = neo_transcript::Poseidon2Transcript::from_state_and_absorbed(
        accumulator_witness.transcript_in().state,
        accumulator_witness.transcript_in().absorbed,
    );
    let step = verify_rv64im_chunk_fold_verifier_step(
        step_program_digest,
        accumulator_witness.step_public().chunk_index as usize,
        accumulator_witness.halted_out(),
        accumulator_witness.handoff(),
        accumulator_witness.running_last(),
        accumulator_witness.final_fold_witness(),
        &mut transcript,
        params,
        structure,
        log,
        &optimized_cache,
    )?;
    if step.next_carry.main.claims != accumulator_witness.running_final().main.claims
        || step.next_carry.terminal_handle != accumulator_witness.running_final().terminal_handle
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM terminal decider witness final carry does not match the native terminal fold replay".into(),
        ));
    }
    if &step.step_public != accumulator_witness.step_public() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM terminal decider witness step public does not match the native terminal fold replay".into(),
        ));
    }
    let transcript_out = crate::rv64im::final_relation::Rv64imChunkFoldTranscriptSnapshot {
        state: transcript.state(),
        absorbed: transcript.absorbed(),
    };
    if &transcript_out != accumulator_witness.transcript_out() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM terminal decider witness transcript_out does not match the native terminal fold replay".into(),
        ));
    }
    let witness = build_rv64im_terminal_step_witness_from_accumulator_witness(accumulator_witness);
    let statement = build_rv64im_terminal_step_statement_from_accumulator_witness_transport(accumulator_witness);
    validate_rv64im_chunk_step_ivc_surface(&statement, &witness)?;
    Ok(Rv64imChunkStepIvcRelation { statement, witness })
}
