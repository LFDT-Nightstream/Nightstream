//! Owns the RV64IM one-step recursive boundary above the chunk-fold verifier.
//!
//! The public side stays application-visible and fixed-shape. The running
//! transcript and carried CE state stay private recursive state.

use serde::{Deserialize, Serialize};

use crate::chunk_relation::ChunkReplayWitness;
use crate::finalize::{digest32_as_fields, digest_fields_as_digest32, FixedShapeChunkSummary};
use crate::rv64im::chunk_fold_step::{
    verify_rv64im_chunk_fold_verifier_step, Rv64imChunkFoldCarry, Rv64imChunkStepPublic,
};
use crate::rv64im::construction2::build_rv64im_main_recursion_construction2_canonical_step_statement_digest_from_relation;
use crate::rv64im::final_relation::{
    build_rv64im_chunk_fold_step_traces, rv64im_chunk_fold_carried_transcript_snapshot,
    rv64im_chunk_fold_initial_transcript_snapshot, Rv64imChunkFoldState, Rv64imFinalBuildProof, Rv64imFinalStatement,
};
use crate::rv64im::kernel::{
    rv64im_cached_root_main_lane_context, rv64im_cached_root_main_lane_optimized_cache,
    Rv64imVerifiedKernelChunkHandoff, SimpleKernelError,
};
use neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash;
use neo_math::F;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

const RV64IM_STEP_STATEMENT_CHAIN_RAW_TAG: u64 = 0x7276_3634_7374_6d74;
const RV64IM_BRIDGE_HANDOFF_CHAIN_RAW_TAG: u64 = 0x7276_3634_62686467;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rv64imChunkStepIvcStatement {
    pub step_public: Rv64imChunkStepPublic,
    pub chunk_summary: FixedShapeChunkSummary,
}

impl Rv64imChunkStepIvcStatement {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/chunk_step_ivc_statement");
        tr.append_message(
            b"neo.fold.next/rv64im/chunk_step_ivc_statement/step_public",
            &self.step_public.expected_digest(),
        );
        tr.append_message(
            b"neo.fold.next/rv64im/chunk_step_ivc_statement/chunk_summary",
            &self.chunk_summary.digest(),
        );
        tr.digest32()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv64imChunkStepIvcPublishedTarget {
    pub program_digest: [u8; 32],
    pub chunk_index: u64,
    pub step_lo: u64,
    pub step_hi: u64,
    pub halted_out: bool,
    pub state_in: [u8; 32],
    pub state_out: [u8; 32],
    pub summary_start: u64,
    pub summary_step_count: u64,
    pub public_chunk_digest: [u8; 32],
    pub chunk_relation_digest: [u8; 32],
}

impl Rv64imChunkStepIvcPublishedTarget {
    pub fn public_values(&self) -> Vec<F> {
        let mut out = Vec::with_capacity(26);
        out.extend(digest32_as_fields(self.program_digest));
        out.push(F::from_u64(self.chunk_index));
        out.push(F::from_u64(self.step_lo));
        out.push(F::from_u64(self.step_hi));
        out.push(F::from_u64(u64::from(self.halted_out)));
        out.extend(digest32_as_fields(self.state_in));
        out.extend(digest32_as_fields(self.state_out));
        out.push(F::from_u64(self.summary_start));
        out.push(F::from_u64(self.summary_step_count));
        out.extend(digest32_as_fields(self.public_chunk_digest));
        out.extend(digest32_as_fields(self.chunk_relation_digest));
        out
    }

    pub fn chunk_summary(&self) -> FixedShapeChunkSummary {
        FixedShapeChunkSummary {
            start_index: self.summary_start,
            public_step_count: self.summary_step_count,
            public_chunk_digest: self.public_chunk_digest,
            chunk_relation_digest: self.chunk_relation_digest,
        }
    }

    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/chunk_step_ivc_statement");
        tr.append_message(
            b"neo.fold.next/rv64im/chunk_step_ivc_statement/step_public",
            &self.step_public().expected_digest(),
        );
        tr.append_message(
            b"neo.fold.next/rv64im/chunk_step_ivc_statement/chunk_summary",
            &self.chunk_summary().digest(),
        );
        tr.digest32()
    }

    pub fn step_public(&self) -> Rv64imChunkStepPublic {
        Rv64imChunkStepPublic {
            program_digest: self.program_digest,
            chunk_index: self.chunk_index,
            step_lo: self.step_lo,
            step_hi: self.step_hi,
            state_in: self.state_in,
            state_out: self.state_out,
            halted_out: self.halted_out,
        }
    }
}

pub(crate) fn build_rv64im_chunk_step_ivc_statement_from_authoritative_parts(
    program_digest: [u8; 32],
    witness: &Rv64imChunkStepIvcWitness,
    chunk_relation_digest: [u8; 32],
) -> Rv64imChunkStepIvcStatement {
    let step_lo = witness.handoff.public_chunk.start_index as u64;
    let public_step_count = witness.handoff.public_chunk.steps.len() as u64;
    Rv64imChunkStepIvcStatement {
        step_public: Rv64imChunkStepPublic {
            program_digest,
            chunk_index: witness.handoff.bridge_handoff.chunk_index as u64,
            step_lo,
            step_hi: step_lo + public_step_count,
            state_in: witness.state_in.carry.terminal_handle.0,
            state_out: witness.state_out.carry.terminal_handle.0,
            halted_out: witness.terminal_step,
        },
        chunk_summary: FixedShapeChunkSummary::from_public_chunk(
            &witness.handoff.public_chunk,
            witness.handoff.public_chunk_digest,
            chunk_relation_digest,
        ),
    }
}

#[derive(Clone, Debug)]
pub struct Rv64imChunkStepIvcWitness {
    pub handoff: Rv64imVerifiedKernelChunkHandoff,
    pub state_in: Rv64imChunkFoldState,
    pub state_out: Rv64imChunkFoldState,
    pub replay_witness: ChunkReplayWitness,
    pub terminal_step: bool,
}

#[derive(Clone, Debug)]
pub struct Rv64imChunkStepIvcRelation {
    pub statement: Rv64imChunkStepIvcStatement,
    pub witness: Rv64imChunkStepIvcWitness,
}

pub fn rv64im_chunk_step_ivc_initial_state() -> Rv64imChunkFoldState {
    Rv64imChunkFoldState {
        carry: Rv64imChunkFoldCarry::seed(),
        transcript: rv64im_chunk_fold_initial_transcript_snapshot(),
    }
}

fn rv64im_digest_chain_initial(raw_tag: u64) -> [u8; 32] {
    digest_fields_as_digest32(poseidon2_hash(&[F::from_u64(raw_tag)]))
}

fn rv64im_digest_chain_step(raw_tag: u64, current: [u8; 32], item: [u8; 32]) -> [u8; 32] {
    let mut preimage = Vec::with_capacity(1 + 4 + 4);
    preimage.push(F::from_u64(raw_tag));
    preimage.extend(digest32_as_fields(current));
    preimage.extend(digest32_as_fields(item));
    digest_fields_as_digest32(poseidon2_hash(&preimage))
}

fn rv64im_digest_chain(raw_tag: u64, digests: &[[u8; 32]]) -> [u8; 32] {
    let mut current = rv64im_digest_chain_initial(raw_tag);
    for digest in digests {
        current = rv64im_digest_chain_step(raw_tag, current, *digest);
    }
    current
}

pub(crate) fn rv64im_step_statement_chain_digest_init() -> [u8; 32] {
    rv64im_digest_chain_initial(RV64IM_STEP_STATEMENT_CHAIN_RAW_TAG)
}

pub(crate) fn rv64im_step_statement_chain_digest_step(current: [u8; 32], digest: [u8; 32]) -> [u8; 32] {
    rv64im_digest_chain_step(RV64IM_STEP_STATEMENT_CHAIN_RAW_TAG, current, digest)
}

pub(crate) fn rv64im_step_statement_chain_digest(relations: &[Rv64imChunkStepIvcRelation]) -> [u8; 32] {
    rv64im_step_statement_chain_digest_from_digests(
        &relations
            .iter()
            .map(|relation| {
                build_rv64im_chunk_step_ivc_published_target(&relation.statement)
                    .expect("validated chunk-step relation must derive a canonical published target")
                    .expected_digest()
            })
            .collect::<Vec<_>>(),
    )
}

pub(crate) fn rv64im_recursion_step_statement_chain_digest(relations: &[Rv64imChunkStepIvcRelation]) -> [u8; 32] {
    rv64im_step_statement_chain_digest_from_digests(
        &relations
            .iter()
            .map(|relation| {
                build_rv64im_main_recursion_construction2_canonical_step_statement_digest_from_relation(relation)
                    .expect(
                        "validated chunk-step relation must derive a canonical Construction-2 step statement digest",
                    )
            })
            .collect::<Vec<_>>(),
    )
}

pub(crate) fn rv64im_step_statement_chain_digest_from_digests(digests: &[[u8; 32]]) -> [u8; 32] {
    rv64im_digest_chain(RV64IM_STEP_STATEMENT_CHAIN_RAW_TAG, digests)
}

pub(crate) fn rv64im_bridge_handoff_chain_digest_init() -> [u8; 32] {
    rv64im_digest_chain_initial(RV64IM_BRIDGE_HANDOFF_CHAIN_RAW_TAG)
}

pub(crate) fn rv64im_bridge_handoff_chain_digest_step(current: [u8; 32], digest: [u8; 32]) -> [u8; 32] {
    rv64im_digest_chain_step(RV64IM_BRIDGE_HANDOFF_CHAIN_RAW_TAG, current, digest)
}

pub(crate) fn rv64im_bridge_handoff_chain_digest(relations: &[Rv64imChunkStepIvcRelation]) -> [u8; 32] {
    rv64im_bridge_handoff_chain_digest_from_digests(
        &relations
            .iter()
            .map(|relation| relation.witness.handoff.bridge_handoff.digest)
            .collect::<Vec<_>>(),
    )
}

pub(crate) fn rv64im_bridge_handoff_chain_digest_from_digests(digests: &[[u8; 32]]) -> [u8; 32] {
    rv64im_digest_chain(RV64IM_BRIDGE_HANDOFF_CHAIN_RAW_TAG, digests)
}

pub fn build_rv64im_chunk_step_ivc_relations(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalBuildProof,
) -> Result<Vec<Rv64imChunkStepIvcRelation>, SimpleKernelError> {
    Ok(build_rv64im_chunk_fold_step_traces(statement, proof)?
        .into_iter()
        .map(|trace| {
            let state_in = trace.state_in();
            let state_out = trace.state_out();
            let witness = Rv64imChunkStepIvcWitness {
                handoff: trace.handoff,
                state_in,
                state_out,
                replay_witness: trace.replay_witness,
                terminal_step: trace.halted_out,
            };
            Rv64imChunkStepIvcRelation {
                statement: build_rv64im_chunk_step_ivc_statement_from_authoritative_parts(
                    trace.step_public.program_digest,
                    &witness,
                    trace.chunk_summary.chunk_relation_digest,
                ),
                witness,
            }
        })
        .collect())
}

pub fn validate_rv64im_chunk_step_ivc_published_statement(
    statement: &Rv64imChunkStepIvcStatement,
) -> Result<(), SimpleKernelError> {
    let expected_step_count = statement
        .step_public
        .step_hi
        .checked_sub(statement.step_public.step_lo)
        .ok_or_else(|| SimpleKernelError::Bridge("RV64IM chunk-step IVC step bounds are not monotone".into()))?;
    if statement.chunk_summary.start_index != statement.step_public.step_lo {
        return Err(SimpleKernelError::Bridge(
            "RV64IM chunk-step IVC summary start index does not match step_public.step_lo".into(),
        ));
    }
    if statement.chunk_summary.public_step_count != expected_step_count {
        return Err(SimpleKernelError::Bridge(
            "RV64IM chunk-step IVC summary public_step_count does not match step_public span".into(),
        ));
    }
    Ok(())
}

pub fn build_rv64im_chunk_step_ivc_published_target(
    statement: &Rv64imChunkStepIvcStatement,
) -> Result<Rv64imChunkStepIvcPublishedTarget, SimpleKernelError> {
    validate_rv64im_chunk_step_ivc_published_statement(statement)?;
    Ok(Rv64imChunkStepIvcPublishedTarget {
        program_digest: statement.step_public.program_digest,
        chunk_index: statement.step_public.chunk_index,
        step_lo: statement.step_public.step_lo,
        step_hi: statement.step_public.step_hi,
        halted_out: statement.step_public.halted_out,
        state_in: statement.step_public.state_in,
        state_out: statement.step_public.state_out,
        summary_start: statement.chunk_summary.start_index,
        summary_step_count: statement.chunk_summary.public_step_count,
        public_chunk_digest: statement.chunk_summary.public_chunk_digest,
        chunk_relation_digest: statement.chunk_summary.chunk_relation_digest,
    })
}

pub fn validate_rv64im_chunk_step_ivc_surface(
    statement: &Rv64imChunkStepIvcStatement,
    witness: &Rv64imChunkStepIvcWitness,
) -> Result<(), SimpleKernelError> {
    validate_rv64im_chunk_step_ivc_published_statement(statement)?;
    let expected_step_count = statement.step_public.step_hi - statement.step_public.step_lo;
    if witness.handoff.public_chunk.start_index as u64 != statement.step_public.step_lo {
        return Err(SimpleKernelError::Bridge(
            "RV64IM chunk-step IVC handoff start index does not match step_public.step_lo".into(),
        ));
    }
    if witness.handoff.public_chunk.steps.len() as u64 != expected_step_count {
        return Err(SimpleKernelError::Bridge(
            "RV64IM chunk-step IVC handoff step count does not match step_public span".into(),
        ));
    }
    if witness.handoff.bridge_handoff.chunk_index as u64 != statement.step_public.chunk_index {
        return Err(SimpleKernelError::Bridge(
            "RV64IM chunk-step IVC bridge handoff chunk_index does not match step_public.chunk_index".into(),
        ));
    }
    if witness.state_in.carry.terminal_handle.0 != statement.step_public.state_in {
        return Err(SimpleKernelError::Bridge(
            "RV64IM chunk-step IVC state_in terminal handle does not match step_public.state_in".into(),
        ));
    }
    if witness.state_out.carry.terminal_handle.0 != statement.step_public.state_out {
        return Err(SimpleKernelError::Bridge(
            "RV64IM chunk-step IVC state_out terminal handle does not match step_public.state_out".into(),
        ));
    }
    if witness.state_in.transcript.absorbed > neo_params::poseidon2_goldilocks::RATE
        || witness.state_out.transcript.absorbed > neo_params::poseidon2_goldilocks::RATE
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM chunk-step IVC transcript snapshot absorbed count exceeds the Poseidon2 rate".into(),
        ));
    }
    Ok(())
}

pub fn verify_rv64im_chunk_step_ivc_chain(
    relations: &[Rv64imChunkStepIvcRelation],
) -> Result<Rv64imChunkFoldState, SimpleKernelError> {
    let mut expected_state = rv64im_chunk_step_ivc_initial_state();
    for (chain_index, relation) in relations.iter().enumerate() {
        if relation.witness.handoff.bridge_handoff.chunk_index as usize != chain_index {
            return Err(SimpleKernelError::Bridge(
                "RV64IM chunk-step IVC chain chunk_index does not match the recursive chain position".into(),
            ));
        }
        if !rv64im_chunk_step_ivc_states_match(&relation.witness.state_in, &expected_state) {
            return Err(SimpleKernelError::Bridge(
                "RV64IM chunk-step IVC chain state_in does not match the carried private recursive state".into(),
            ));
        }
        expected_state = verify_rv64im_chunk_step_ivc(&relation.statement, &relation.witness)?;
    }
    Ok(expected_state)
}

pub fn verify_rv64im_chunk_step_ivc(
    statement: &Rv64imChunkStepIvcStatement,
    witness: &Rv64imChunkStepIvcWitness,
) -> Result<Rv64imChunkFoldState, SimpleKernelError> {
    validate_rv64im_chunk_step_ivc_surface(statement, witness)?;
    let (params, log, structure) = rv64im_cached_root_main_lane_context()?;
    let optimized_cache = rv64im_cached_root_main_lane_optimized_cache()?;
    let mut transcript = Poseidon2Transcript::from_state_and_absorbed(
        witness.state_in.transcript.state,
        witness.state_in.transcript.absorbed,
    );
    let step = verify_rv64im_chunk_fold_verifier_step(
        statement.step_public.program_digest,
        witness.handoff.bridge_handoff.chunk_index as usize,
        witness.terminal_step,
        &witness.handoff,
        &witness.state_in.carry,
        &witness.replay_witness,
        &mut transcript,
        params,
        structure,
        log,
        &optimized_cache,
    )?;
    if step.step_public != statement.step_public {
        return Err(SimpleKernelError::Bridge(
            "RV64IM chunk-step IVC public step does not match the verified one-step fold output".into(),
        ));
    }
    let expected_summary = FixedShapeChunkSummary::from_public_chunk(
        &witness.handoff.public_chunk,
        step.public_chunk_digest,
        step.chunk_relation_digest,
    );
    if expected_summary != statement.chunk_summary {
        return Err(SimpleKernelError::Bridge(
            "RV64IM chunk-step IVC summary does not match the verified one-step fold output".into(),
        ));
    }
    if step.next_carry.main.claims != witness.state_out.carry.main.claims
        || step.next_carry.main.witnesses != witness.state_out.carry.main.witnesses
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM chunk-step IVC next carry does not match the carried private next-main state".into(),
        ));
    }
    if step.next_carry.terminal_handle != witness.state_out.carry.terminal_handle {
        return Err(SimpleKernelError::Bridge(
            "RV64IM chunk-step IVC next terminal handle does not match the carried private state".into(),
        ));
    }
    let transcript_out = rv64im_chunk_fold_carried_transcript_snapshot(
        &crate::rv64im::final_relation::Rv64imChunkFoldTranscriptSnapshot {
            state: transcript.state(),
            absorbed: transcript.absorbed(),
        },
    );
    if transcript_out != witness.state_out.transcript {
        return Err(SimpleKernelError::Bridge(
            "RV64IM chunk-step IVC transcript_out does not match the carried private transcript state".into(),
        ));
    }
    Ok(Rv64imChunkFoldState {
        carry: Rv64imChunkFoldCarry {
            main: witness.state_out.carry.main.clone(),
            terminal_handle: witness.state_out.carry.terminal_handle,
        },
        transcript: transcript_out,
    })
}

fn rv64im_chunk_step_ivc_states_match(lhs: &Rv64imChunkFoldState, rhs: &Rv64imChunkFoldState) -> bool {
    lhs.carry.main.claims == rhs.carry.main.claims
        && lhs.carry.main.witnesses == rhs.carry.main.witnesses
        && lhs.carry.terminal_handle == rhs.carry.terminal_handle
        && lhs.transcript == rhs.transcript
}
