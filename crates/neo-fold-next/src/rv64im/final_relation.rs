//! Owns the RV64IM folded/final relation replay above the accepted/export seam.

use neo_ajtai::Commitment;
use neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash;
use neo_ccs::{CcsStructure, CeClaim};
use neo_math::{F, K};
use neo_params::NeoParams;
use neo_reductions::engines::utils::me_digest_poseidon_into;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::chunk_relation::ChunkReplayWitness;
use crate::finalize::{
    digest32_as_fields, digest_fields_as_digest32, digest_fixed_shape_final_proof, FixedShapeChunkSummary,
};
use crate::proof::{Carry, ChunkInput, ChunkProvePerf, FoldSchedule};
use crate::rv64im::chunk_fold_step::{
    adapt_rv64im_chunk_to_fresh_ccs, prove_rv64im_chunk_fold_verifier_step_with_perf,
    verify_rv64im_chunk_fold_verifier_step, Rv64imAccumulatorHandle, Rv64imChunkFoldCarry, Rv64imChunkFoldFresh,
    Rv64imChunkStepPublic,
};
use crate::rv64im::chunk_relation::rv64im_chunk_replay_witness_digest;
use crate::rv64im::kernel::{
    build_rv64im_kernel_export_build_output_from_carried_accepted_artifact_with_source_and_chunk_inputs,
    build_rv64im_kernel_export_proof_from_carried_accepted_artifact, rv64im_cached_root_main_lane_context,
    rv64im_cached_root_main_lane_optimized_cache, verify_rv64im_kernel_export_proof_with_output,
    verify_rv64im_kernel_export_proof_with_relation_output, Rv64imAcceptedProofArtifact, Rv64imKernelExportProof,
    Rv64imKernelExportRelationResult, Rv64imKernelExportSource, Rv64imVerifiedKernelChunkHandoff, SimpleKernelError,
};

pub(crate) const RV64IM_SESSION_RAW_DOMAIN_TAG: u64 = 17;
pub(crate) const RV64IM_CHUNK_DONE_RAW_TAG: u64 = 16;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rv64imRecursiveAccumulator {
    pub final_main_claims: Vec<CeClaim<Commitment, F, K>>,
    pub terminal_handle: Rv64imAccumulatorHandle,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rv64imFoldedStatement {
    pub fold_schedule: FoldSchedule,
    pub chunk_count: u64,
    pub semantic_step_count: u64,
    pub kernel_relation_digest: [u8; 32],
    pub final_accumulator: Rv64imRecursiveAccumulator,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug)]
pub struct Rv64imChunkTransitionWitness {
    pub replay_witness: ChunkReplayWitness,
}

#[derive(Clone, Debug)]
pub struct Rv64imFoldedProof {
    pub kernel_export: Rv64imKernelExportProof,
    pub steps: Vec<Rv64imChunkTransitionWitness>,
}

#[derive(Clone, Debug)]
pub struct Rv64imChunkFoldStepTrace {
    pub handoff: Rv64imVerifiedKernelChunkHandoff,
    pub fresh: Rv64imChunkFoldFresh,
    pub chunk_summary: FixedShapeChunkSummary,
    pub carry_in: Rv64imChunkFoldCarry,
    pub carry_out: Rv64imChunkFoldCarry,
    pub transcript_in: Rv64imChunkFoldTranscriptSnapshot,
    pub transcript_out: Rv64imChunkFoldTranscriptSnapshot,
    pub step_public: Rv64imChunkStepPublic,
    pub replay_witness: ChunkReplayWitness,
    pub replay_witness_digest: [u8; 32],
    pub halted_out: bool,
}

#[derive(Clone, Debug)]
pub struct Rv64imTerminalChunkFoldWitness {
    pub public_statement_digest: [u8; 32],
    pub handoff: Rv64imVerifiedKernelChunkHandoff,
    pub running_last: Rv64imChunkFoldCarry,
    pub transcript_in: Rv64imChunkFoldTranscriptSnapshot,
    pub fresh_last: Rv64imChunkFoldFresh,
    pub final_fold_witness: ChunkReplayWitness,
    pub running_final: Rv64imChunkFoldCarry,
    pub transcript_out: Rv64imChunkFoldTranscriptSnapshot,
    pub step_public: Rv64imChunkStepPublic,
    pub halted_out: bool,
}

impl Rv64imTerminalChunkFoldWitness {
    pub fn accumulator_final(&self) -> Rv64imRecursiveAccumulator {
        recursive_accumulator_from_carry(&self.running_final)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rv64imChunkFoldTranscriptSnapshot {
    pub state: [F; neo_params::poseidon2_goldilocks::WIDTH],
    pub absorbed: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rv64imChunkFoldState {
    pub carry: Rv64imChunkFoldCarry,
    pub transcript: Rv64imChunkFoldTranscriptSnapshot,
}

#[inline]
fn extend_packed_bytes_as_fields(dst: &mut Vec<F>, bytes: &[u8]) {
    const BYTES_PER_LIMB: usize = 7;
    dst.push(F::from_u64(bytes.len() as u64));
    for chunk in bytes.chunks(BYTES_PER_LIMB) {
        let mut limb = [0u8; 8];
        limb[..chunk.len()].copy_from_slice(chunk);
        dst.push(F::from_u64(u64::from_le_bytes(limb)));
    }
}

pub(crate) fn rv64im_chunk_fold_transcript_snapshot_digest(snapshot: &Rv64imChunkFoldTranscriptSnapshot) -> [u8; 32] {
    let mut preimage = Vec::with_capacity(1 + 8 + neo_params::poseidon2_goldilocks::WIDTH);
    extend_packed_bytes_as_fields(
        &mut preimage,
        b"neo.fold.next/rv64im/main_recursion_transcript_snapshot/v2",
    );
    preimage.push(F::from_u64(snapshot.absorbed as u64));
    preimage.extend(snapshot.state);
    digest_fields_as_digest32(poseidon2_hash(&preimage))
}

pub(crate) fn rv64im_chunk_fold_state_instance_digest(state: &Rv64imChunkFoldState) -> [u8; 32] {
    let mut scratch = Vec::<F>::with_capacity(2048);
    let claim_digests = state
        .carry
        .main
        .claims
        .iter()
        .map(|claim| me_digest_poseidon_into(&mut scratch, claim))
        .collect::<Vec<_>>();

    let mut preimage = Vec::with_capacity(32 + claim_digests.len() * 4);
    extend_packed_bytes_as_fields(
        &mut preimage,
        b"neo.fold.next/rv64im/main_recursion_accumulator_instance/v2",
    );
    preimage.push(F::from_u64(claim_digests.len() as u64));
    preimage.extend(
        claim_digests
            .iter()
            .flat_map(|digest| digest.iter().copied()),
    );
    preimage.extend(digest32_as_fields(rv64im_chunk_fold_transcript_snapshot_digest(
        &state.transcript,
    )));
    preimage.extend(digest32_as_fields(state.carry.terminal_handle.0));
    digest_fields_as_digest32(poseidon2_hash(&preimage))
}

pub(crate) fn rv64im_recursive_accumulator_instance_digest_from_parts(
    final_main_claims: &[CeClaim<Commitment, F, K>],
    _terminal_handle_digest: [u8; 32],
) -> [u8; 32] {
    let final_main_claim_digests = final_main_claim_digests(final_main_claims);
    let mut preimage = Vec::with_capacity(32 + final_main_claim_digests.len() * 4);
    extend_packed_bytes_as_fields(
        &mut preimage,
        b"neo.fold.next/rv64im/main_recursion_recursive_accumulator_instance/v2",
    );
    preimage.push(F::from_u64(final_main_claim_digests.len() as u64));
    preimage.extend(
        final_main_claim_digests
            .iter()
            .flat_map(|digest| digest.iter().copied()),
    );
    digest_fields_as_digest32(poseidon2_hash(&preimage))
}

pub(crate) fn rv64im_chunk_fold_carry_recursive_accumulator_digest(carry: &Rv64imChunkFoldCarry) -> [u8; 32] {
    rv64im_recursive_accumulator_instance_digest_from_parts(&carry.main.claims, carry.terminal_handle.0)
}

impl Rv64imChunkFoldStepTrace {
    pub fn state_in(&self) -> Rv64imChunkFoldState {
        Rv64imChunkFoldState {
            carry: self.carry_in.clone(),
            transcript: self.transcript_in.clone(),
        }
    }

    pub fn state_out(&self) -> Rv64imChunkFoldState {
        Rv64imChunkFoldState {
            carry: self.carry_out.clone(),
            transcript: rv64im_chunk_fold_carried_transcript_snapshot(&self.transcript_out),
        }
    }
}

pub(crate) fn rv64im_chunk_fold_carried_transcript_snapshot(
    transcript_out: &Rv64imChunkFoldTranscriptSnapshot,
) -> Rv64imChunkFoldTranscriptSnapshot {
    let mut transcript = Poseidon2Transcript::from_state_and_absorbed(transcript_out.state, transcript_out.absorbed);
    transcript.append_fields_raw(&[F::from_u64(RV64IM_CHUNK_DONE_RAW_TAG), F::ONE]);
    Rv64imChunkFoldTranscriptSnapshot {
        state: transcript.state(),
        absorbed: transcript.absorbed(),
    }
}

/// Build-time replay bundle for the final seam.
///
/// This is not a published proof surface: it still carries per-chunk replay
/// witnesses needed by internal relation builders, audits, and decider prep.
#[derive(Clone, Debug)]
pub struct Rv64imFinalBuildProof {
    pub proof_digest: [u8; 32],
    pub kernel_export: Rv64imKernelExportProof,
    pub chunk_summaries: Vec<FixedShapeChunkSummary>,
    pub steps: Vec<Rv64imChunkTransitionWitness>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Rv64imFinalProofComponentDigests {
    pub kernel_export_proof_digest: [u8; 32],
    pub chunk_transition_digests: Vec<[u8; 32]>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rv64imFinalStatement {
    pub public_statement_digest: [u8; 32],
    pub folded: Rv64imFoldedStatement,
    pub digest: [u8; 32],
}

struct Rv64imFoldedBuildOutput {
    folded: Rv64imFoldedStatement,
    chunk_summaries: Vec<FixedShapeChunkSummary>,
    proof: Rv64imFoldedProof,
    verified_kernel: Rv64imKernelExportRelationResult,
}

pub(crate) struct Rv64imFinalBuildOutput {
    pub statement: Rv64imFinalStatement,
    pub proof: Rv64imFinalBuildProof,
    pub verified_kernel: Rv64imKernelExportRelationResult,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Rv64imFoldedBuildPerf {
    pub kernel_export_ms: f64,
    pub recursive: Rv64imRecursiveBuildPerf,
    pub folded_digest_ms: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Rv64imFinalBuildPerf {
    pub folded: Rv64imFoldedBuildPerf,
    pub final_proof_ms: f64,
    pub statement_digest_ms: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Rv64imRecursiveBuildPerf {
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

impl Rv64imRecursiveBuildPerf {
    fn record_chunk(&mut self, chunk: &ChunkProvePerf) {
        self.prepare_inputs_ms += chunk.prepare_inputs_ms;
        self.ccs_bind_ms += chunk.ccs_bind_ms;
        self.ccs_sample_challenges_ms += chunk.ccs_sample_challenges_ms;
        self.ccs_fe_sumcheck_ms += chunk.ccs_fe_sumcheck_ms;
        self.ccs_nc_sumcheck_ms += chunk.ccs_nc_sumcheck_ms;
        self.ccs_output_materialize_ms += chunk.ccs_output_materialize_ms;
        self.ccs_ms += chunk.ccs_ms;
        self.dims_ms += chunk.dims_ms;
        self.rlc_prepare_ms += chunk.rlc_prepare_ms;
        self.rlc_ms += chunk.rlc_ms;
        self.dec_split_ms += chunk.dec_split_ms;
        self.dec_commit_ms += chunk.dec_commit_ms;
        self.dec_ms += chunk.dec_ms;
        self.total_ms += chunk.total_ms;
    }
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

pub fn prove_rv64im_folded_statement_from_accepted(
    artifact: &Rv64imAcceptedProofArtifact,
) -> Result<(Rv64imFoldedStatement, Rv64imFoldedProof), SimpleKernelError> {
    let built = build_rv64im_folded_statement_from_accepted(artifact)?;
    Ok((built.folded, built.proof))
}

pub fn verify_rv64im_folded_statement(
    folded: &Rv64imFoldedStatement,
    proof: &Rv64imFoldedProof,
) -> Result<(), SimpleKernelError> {
    verify_folded_statement_components_with_output(folded, &proof.kernel_export, &proof.steps)?;
    Ok(())
}

pub fn prove_rv64im_final_statement_from_accepted(
    artifact: &Rv64imAcceptedProofArtifact,
) -> Result<(Rv64imFinalStatement, Rv64imFinalBuildProof), SimpleKernelError> {
    let built = prove_rv64im_final_statement_from_accepted_with_output(artifact)?;
    Ok((built.statement, built.proof))
}

pub fn verify_rv64im_final_statement(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalBuildProof,
) -> Result<(), SimpleKernelError> {
    verify_rv64im_final_statement_with_output(statement, proof)?;
    Ok(())
}

pub fn build_rv64im_chunk_step_publics(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalBuildProof,
) -> Result<Vec<Rv64imChunkStepPublic>, SimpleKernelError> {
    Ok(build_rv64im_chunk_fold_step_traces(statement, proof)?
        .into_iter()
        .map(|step| step.step_public)
        .collect())
}

pub fn build_rv64im_chunk_fold_freshs(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalBuildProof,
) -> Result<Vec<Rv64imChunkFoldFresh>, SimpleKernelError> {
    Ok(build_rv64im_chunk_fold_step_traces(statement, proof)?
        .into_iter()
        .map(|step| step.fresh)
        .collect())
}

pub fn build_rv64im_chunk_fold_step_traces(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalBuildProof,
) -> Result<Vec<Rv64imChunkFoldStepTrace>, SimpleKernelError> {
    validate_rv64im_final_statement_surface(statement, proof)?;
    let verified_kernel =
        verify_rv64im_kernel_export_proof_with_output(statement.folded.kernel_relation_digest, &proof.kernel_export)?;
    let (traces, accumulator) = build_chunk_fold_step_traces_from_verified_kernel(
        statement.public_statement_digest,
        &verified_kernel,
        &proof.steps,
        Some(&proof.chunk_summaries),
    )?;
    let expected_final_accumulator = recursive_accumulator_from_carry(&accumulator);
    if expected_final_accumulator != statement.folded.final_accumulator {
        return Err(SimpleKernelError::Bridge(
            "RV64IM chunk-fold step trace final accumulator does not match the carried folded statement".into(),
        ));
    }
    Ok(traces)
}

pub fn build_rv64im_terminal_chunk_fold_witness(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalBuildProof,
) -> Result<Rv64imTerminalChunkFoldWitness, SimpleKernelError> {
    let traces = build_rv64im_chunk_fold_step_traces(statement, proof)?;
    let last = traces.last().cloned().ok_or_else(|| {
        SimpleKernelError::Bridge("RV64IM terminal chunk-fold witness requires a non-empty chunk replay chain".into())
    })?;
    let accumulator_final = recursive_accumulator_from_carry(&last.carry_out);
    if accumulator_final != statement.folded.final_accumulator {
        return Err(SimpleKernelError::Bridge(
            "RV64IM terminal chunk-fold witness final accumulator does not match the carried folded statement".into(),
        ));
    }
    Ok(Rv64imTerminalChunkFoldWitness {
        public_statement_digest: statement.public_statement_digest,
        handoff: last.handoff,
        running_last: last.carry_in,
        transcript_in: last.transcript_in,
        fresh_last: last.fresh,
        final_fold_witness: last.replay_witness,
        running_final: last.carry_out,
        transcript_out: last.transcript_out,
        step_public: last.step_public,
        halted_out: last.halted_out,
    })
}

pub fn verify_rv64im_terminal_chunk_fold_witness(
    witness: &Rv64imTerminalChunkFoldWitness,
) -> Result<(), SimpleKernelError> {
    let (params, log, structure) = rv64im_cached_root_main_lane_context()?;
    let optimized_cache = rv64im_cached_root_main_lane_optimized_cache()?;
    let mut transcript =
        Poseidon2Transcript::from_state_and_absorbed(witness.transcript_in.state, witness.transcript_in.absorbed);
    let step = verify_rv64im_chunk_fold_verifier_step(
        witness.public_statement_digest,
        witness.step_public.chunk_index as usize,
        witness.halted_out,
        &witness.handoff,
        &witness.running_last,
        &witness.final_fold_witness,
        &mut transcript,
        params,
        structure,
        log,
        &optimized_cache,
    )?;
    if step.next_carry.main.claims != witness.running_final.main.claims
        || step.next_carry.terminal_handle != witness.running_final.terminal_handle
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM terminal chunk-fold witness final carry does not match the native final fold replay".into(),
        ));
    }
    if step.step_public != witness.step_public {
        return Err(SimpleKernelError::Bridge(
            "RV64IM terminal chunk-fold witness step public does not match the native final fold replay".into(),
        ));
    }
    let transcript_out = Rv64imChunkFoldTranscriptSnapshot {
        state: transcript.state(),
        absorbed: transcript.absorbed(),
    };
    if transcript_out != witness.transcript_out {
        return Err(SimpleKernelError::Bridge(
            "RV64IM terminal chunk-fold witness transcript_out does not match the native final fold replay".into(),
        ));
    }
    Ok(())
}

pub fn rv64im_chunk_fold_initial_transcript_snapshot() -> Rv64imChunkFoldTranscriptSnapshot {
    let transcript = Poseidon2Transcript::new_raw_fields(&[F::from_u64(RV64IM_SESSION_RAW_DOMAIN_TAG)]);
    Rv64imChunkFoldTranscriptSnapshot {
        state: transcript.state(),
        absorbed: transcript.absorbed(),
    }
}

pub(crate) fn verify_rv64im_final_statement_with_output(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalBuildProof,
) -> Result<Rv64imKernelExportRelationResult, SimpleKernelError> {
    validate_rv64im_final_statement_surface(statement, proof)?;
    let (verified_kernel, expected_chunk_summaries, _) = verify_folded_statement_components_with_output_and_main_carry(
        &statement.folded,
        &proof.kernel_export,
        &proof.steps,
    )?;
    if proof.chunk_summaries != expected_chunk_summaries {
        return Err(SimpleKernelError::Bridge(
            "RV64IM final proof chunk summaries do not match the verified export seam".into(),
        ));
    }
    Ok(verified_kernel)
}

pub(crate) fn validate_rv64im_final_statement_surface(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalBuildProof,
) -> Result<(), SimpleKernelError> {
    let component_digests = final_proof_component_digests(proof);
    validate_rv64im_final_statement_surface_with_component_digests(statement, proof, &component_digests)
}

pub(crate) fn validate_rv64im_final_statement_surface_with_parts(
    statement: &Rv64imFinalStatement,
    proof_digest: [u8; 32],
    kernel_export: &Rv64imKernelExportProof,
    chunk_summaries: &[FixedShapeChunkSummary],
    component_digests: &Rv64imFinalProofComponentDigests,
) -> Result<(), SimpleKernelError> {
    if statement.folded.digest != folded_statement_digest(&statement.folded) {
        return Err(SimpleKernelError::Bridge(
            "RV64IM folded statement digest mismatch".into(),
        ));
    }
    if statement.digest != final_statement_digest(statement) {
        return Err(SimpleKernelError::Bridge(
            "RV64IM final statement digest mismatch".into(),
        ));
    }
    if statement.public_statement_digest != kernel_export.public_statement_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM final statement public digest does not match the carried accepted artifact".into(),
        ));
    }
    if proof_digest != final_proof_digest_from_component_digests(&statement.folded, chunk_summaries, component_digests)
    {
        return Err(SimpleKernelError::Bridge("RV64IM final proof digest mismatch".into()));
    }
    Ok(())
}

pub(crate) fn validate_rv64im_final_statement_surface_with_component_digests(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalBuildProof,
    component_digests: &Rv64imFinalProofComponentDigests,
) -> Result<(), SimpleKernelError> {
    validate_rv64im_final_statement_surface_with_parts(
        statement,
        proof.proof_digest,
        &proof.kernel_export,
        &proof.chunk_summaries,
        component_digests,
    )
}

pub(crate) fn prove_rv64im_final_statement_from_accepted_with_output(
    artifact: &Rv64imAcceptedProofArtifact,
) -> Result<Rv64imFinalBuildOutput, SimpleKernelError> {
    let (built, _) = prove_rv64im_final_statement_from_accepted_with_output_and_perf(artifact)?;
    Ok(built)
}

pub(crate) fn prove_rv64im_final_statement_from_accepted_with_output_and_perf(
    artifact: &Rv64imAcceptedProofArtifact,
) -> Result<(Rv64imFinalBuildOutput, Rv64imFinalBuildPerf), SimpleKernelError> {
    prove_rv64im_final_statement_from_accepted_with_output_and_perf_and_source(artifact, None, None)
}

pub(crate) fn prove_rv64im_final_statement_from_accepted_with_output_and_perf_and_source(
    artifact: &Rv64imAcceptedProofArtifact,
    kernel_export_source: Option<Rv64imKernelExportSource>,
    chunk_inputs: Option<Vec<ChunkInput>>,
) -> Result<(Rv64imFinalBuildOutput, Rv64imFinalBuildPerf), SimpleKernelError> {
    let (built, folded_perf) =
        build_rv64im_folded_statement_from_accepted_with_perf_and_source(artifact, kernel_export_source, chunk_inputs)?;

    let started = Instant::now();
    let (final_proof, _) = build_final_proof(&built.folded, built.chunk_summaries, built.proof)?;
    let final_proof_ms = elapsed_ms(started);

    let started = Instant::now();
    let mut statement = Rv64imFinalStatement {
        public_statement_digest: artifact.statement.digest,
        folded: built.folded,
        digest: [0; 32],
    };
    statement.digest = final_statement_digest(&statement);
    let statement_digest_ms = elapsed_ms(started);

    Ok((
        Rv64imFinalBuildOutput {
            statement,
            proof: final_proof,
            verified_kernel: built.verified_kernel,
        },
        Rv64imFinalBuildPerf {
            folded: folded_perf,
            final_proof_ms,
            statement_digest_ms,
        },
    ))
}

pub(crate) fn folded_statement_digest(folded: &Rv64imFoldedStatement) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/folded_statement");
    tr.append_message(b"neo.fold.next/rv64im/folded_statement/version", b"v1");
    tr.append_u64s(
        b"neo.fold.next/rv64im/folded_statement/meta",
        &[folded.chunk_count, folded.semantic_step_count],
    );
    tr.append_u64s(
        b"neo.fold.next/rv64im/folded_statement/schedule",
        &folded.fold_schedule.meta_words(),
    );
    tr.append_message(
        b"neo.fold.next/rv64im/folded_statement/kernel_relation_digest",
        &folded.kernel_relation_digest,
    );
    append_recursive_accumulator(&mut tr, &folded.final_accumulator);
    tr.digest32()
}

pub(crate) fn final_statement_digest(statement: &Rv64imFinalStatement) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/final_statement");
    tr.append_message(b"neo.fold.next/rv64im/final_statement/version", b"v1");
    tr.append_message(
        b"neo.fold.next/rv64im/final_statement/public_statement_digest",
        &statement.public_statement_digest,
    );
    tr.append_message(
        b"neo.fold.next/rv64im/final_statement/folded_digest",
        &statement.folded.digest,
    );
    tr.digest32()
}

fn build_rv64im_folded_statement_from_accepted(
    artifact: &Rv64imAcceptedProofArtifact,
) -> Result<Rv64imFoldedBuildOutput, SimpleKernelError> {
    let (built, _) = build_rv64im_folded_statement_from_accepted_with_perf(artifact)?;
    Ok(built)
}

fn build_rv64im_folded_statement_from_accepted_with_perf(
    artifact: &Rv64imAcceptedProofArtifact,
) -> Result<(Rv64imFoldedBuildOutput, Rv64imFoldedBuildPerf), SimpleKernelError> {
    build_rv64im_folded_statement_from_accepted_with_perf_and_source(artifact, None, None)
}

fn build_rv64im_folded_statement_from_accepted_with_perf_and_source(
    artifact: &Rv64imAcceptedProofArtifact,
    kernel_export_source: Option<Rv64imKernelExportSource>,
    chunk_inputs: Option<Vec<ChunkInput>>,
) -> Result<(Rv64imFoldedBuildOutput, Rv64imFoldedBuildPerf), SimpleKernelError> {
    let started = Instant::now();
    let (relation, kernel_export, verified_kernel) = match kernel_export_source {
        Some(source) => {
            let built =
                build_rv64im_kernel_export_build_output_from_carried_accepted_artifact_with_source_and_chunk_inputs(
                    artifact,
                    source,
                    chunk_inputs,
                )?;
            (built.relation, built.proof, built.result)
        }
        None => build_rv64im_kernel_export_proof_from_carried_accepted_artifact(artifact)?,
    };
    let kernel_export_ms = elapsed_ms(started);

    let started = Instant::now();
    let (params, log, structure) = rv64im_cached_root_main_lane_context()?;
    let (steps, chunk_summaries, final_accumulator, mut recursive_perf) =
        build_recursive_proof(&verified_kernel.chunk_handoffs, params, structure, log)?;
    let recursive_proof_ms = elapsed_ms(started);
    recursive_perf.total_ms = recursive_proof_ms;

    let started = Instant::now();
    let mut folded = Rv64imFoldedStatement {
        fold_schedule: verified_kernel.fold_schedule,
        chunk_count: verified_kernel.chunk_handoffs.len() as u64,
        semantic_step_count: verified_kernel
            .chunk_handoffs
            .iter()
            .map(|handoff| handoff.chunk_input.steps.len() as u64)
            .sum(),
        kernel_relation_digest: relation.digest,
        final_accumulator,
        digest: [0; 32],
    };
    folded.digest = folded_statement_digest(&folded);
    let folded_digest_ms = elapsed_ms(started);

    Ok((
        Rv64imFoldedBuildOutput {
            folded,
            chunk_summaries,
            proof: Rv64imFoldedProof { kernel_export, steps },
            verified_kernel,
        },
        Rv64imFoldedBuildPerf {
            kernel_export_ms,
            recursive: recursive_perf,
            folded_digest_ms,
        },
    ))
}

fn verify_folded_statement_components_with_output(
    folded: &Rv64imFoldedStatement,
    kernel_export: &Rv64imKernelExportProof,
    steps: &[Rv64imChunkTransitionWitness],
) -> Result<(Rv64imKernelExportRelationResult, Vec<FixedShapeChunkSummary>), SimpleKernelError> {
    let (verified_kernel, chunk_summaries, _) =
        verify_folded_statement_components_with_output_and_main_carry(folded, kernel_export, steps)?;
    Ok((verified_kernel, chunk_summaries))
}

fn verify_folded_statement_components_with_output_and_main_carry(
    folded: &Rv64imFoldedStatement,
    kernel_export: &Rv64imKernelExportProof,
    steps: &[Rv64imChunkTransitionWitness],
) -> Result<(Rv64imKernelExportRelationResult, Vec<FixedShapeChunkSummary>, Carry), SimpleKernelError> {
    if folded.digest != folded_statement_digest(folded) {
        return Err(SimpleKernelError::Bridge(
            "RV64IM folded statement digest mismatch".into(),
        ));
    }
    let verified_kernel = verify_rv64im_kernel_export_proof_with_output(folded.kernel_relation_digest, kernel_export)?;
    if folded.fold_schedule != verified_kernel.fold_schedule {
        return Err(SimpleKernelError::Bridge(
            "RV64IM folded statement schedule does not match the verified export relation".into(),
        ));
    }
    if folded.chunk_count as usize != verified_kernel.chunk_handoffs.len() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM folded statement chunk count does not match the verified export relation".into(),
        ));
    }
    let verified_semantic_step_count: usize = verified_kernel
        .chunk_handoffs
        .iter()
        .map(|handoff| handoff.chunk_input.steps.len())
        .sum();
    if folded.semantic_step_count as usize != verified_semantic_step_count {
        return Err(SimpleKernelError::Bridge(
            "RV64IM folded statement semantic step count does not match the verified export relation".into(),
        ));
    }
    if steps.len() != verified_kernel.chunk_handoffs.len() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM folded proof chunk replay count does not match the verified export relation".into(),
        ));
    }
    let (chunk_summaries, final_main) = verify_recursive_steps(folded, &verified_kernel, steps)?;
    Ok((verified_kernel, chunk_summaries, final_main))
}

fn verify_recursive_steps(
    folded: &Rv64imFoldedStatement,
    verified_kernel: &Rv64imKernelExportRelationResult,
    steps: &[Rv64imChunkTransitionWitness],
) -> Result<(Vec<FixedShapeChunkSummary>, Carry), SimpleKernelError> {
    let (chunk_summaries, final_state, _) = replay_recursive_steps_with_state([0; 32], verified_kernel, steps)?;
    let final_accumulator = recursive_accumulator_from_carry(&final_state);
    if final_accumulator != folded.final_accumulator {
        return Err(SimpleKernelError::Bridge(
            "RV64IM folded statement final accumulator mismatch".into(),
        ));
    }
    Ok((chunk_summaries, final_state.main))
}

fn replay_recursive_steps(
    verified_kernel: &Rv64imKernelExportRelationResult,
    steps: &[Rv64imChunkTransitionWitness],
) -> Result<(Vec<FixedShapeChunkSummary>, Rv64imRecursiveAccumulator), SimpleKernelError> {
    let (chunk_summaries, final_state, _) = replay_recursive_steps_with_state([0; 32], verified_kernel, steps)?;
    Ok((chunk_summaries, recursive_accumulator_from_carry(&final_state)))
}

fn replay_recursive_steps_with_state(
    public_statement_digest: [u8; 32],
    verified_kernel: &Rv64imKernelExportRelationResult,
    steps: &[Rv64imChunkTransitionWitness],
) -> Result<
    (
        Vec<FixedShapeChunkSummary>,
        Rv64imChunkFoldCarry,
        Vec<Rv64imChunkStepPublic>,
    ),
    SimpleKernelError,
> {
    let (traces, accumulator) =
        build_chunk_fold_step_traces_from_verified_kernel(public_statement_digest, verified_kernel, steps, None)?;
    Ok((
        traces
            .iter()
            .map(|step| step.chunk_summary.clone())
            .collect(),
        accumulator,
        traces.into_iter().map(|step| step.step_public).collect(),
    ))
}

fn build_chunk_fold_step_traces_from_verified_kernel(
    public_statement_digest: [u8; 32],
    verified_kernel: &Rv64imKernelExportRelationResult,
    steps: &[Rv64imChunkTransitionWitness],
    expected_chunk_summaries: Option<&[FixedShapeChunkSummary]>,
) -> Result<(Vec<Rv64imChunkFoldStepTrace>, Rv64imChunkFoldCarry), SimpleKernelError> {
    if steps.len() != verified_kernel.chunk_handoffs.len() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM chunk-fold step trace replay count does not match the verified export relation".into(),
        ));
    }
    if let Some(summaries) = expected_chunk_summaries {
        if summaries.len() != verified_kernel.chunk_handoffs.len() {
            return Err(SimpleKernelError::Bridge(
                "RV64IM chunk-fold step trace summary count does not match the verified export relation".into(),
            ));
        }
    }

    let (params, log, structure) = rv64im_cached_root_main_lane_context()?;
    let optimized_cache = rv64im_cached_root_main_lane_optimized_cache()?;
    let mut transcript = Poseidon2Transcript::new_raw_fields(&[F::from_u64(RV64IM_SESSION_RAW_DOMAIN_TAG)]);
    let mut accumulator = Rv64imChunkFoldCarry::seed();
    let mut traces = Vec::with_capacity(steps.len());

    for (chunk_index, step_witness) in steps.iter().enumerate() {
        let handoff = verified_kernel
            .chunk_handoffs
            .get(chunk_index)
            .ok_or_else(|| {
                SimpleKernelError::Bridge(format!(
                    "RV64IM chunk transition {chunk_index} missing a verified export handoff"
                ))
            })?
            .clone();
        let fresh = adapt_rv64im_chunk_to_fresh_ccs(&handoff);
        let carry_in = accumulator.clone();
        let transcript_in = Rv64imChunkFoldTranscriptSnapshot {
            state: transcript.state(),
            absorbed: transcript.absorbed(),
        };
        let halted_out = verified_kernel.halted && chunk_index + 1 == verified_kernel.chunk_handoffs.len();
        let step = verify_rv64im_chunk_fold_verifier_step(
            public_statement_digest,
            chunk_index,
            halted_out,
            &handoff,
            &carry_in,
            &step_witness.replay_witness,
            &mut transcript,
            params,
            structure,
            log,
            &optimized_cache,
        )?;
        let transcript_out = Rv64imChunkFoldTranscriptSnapshot {
            state: transcript.state(),
            absorbed: transcript.absorbed(),
        };
        let chunk_summary = FixedShapeChunkSummary::from_public_chunk(
            &handoff.public_chunk,
            step.public_chunk_digest,
            step.chunk_relation_digest,
        );
        if let Some(expected) = expected_chunk_summaries {
            if expected[chunk_index] != chunk_summary {
                return Err(SimpleKernelError::Bridge(format!(
                    "RV64IM chunk-fold step trace {chunk_index} summary does not match the carried final proof summary"
                )));
            }
        }
        accumulator = step.next_carry.clone();
        traces.push(Rv64imChunkFoldStepTrace {
            handoff,
            fresh,
            chunk_summary,
            carry_in,
            carry_out: step.next_carry,
            transcript_in,
            transcript_out,
            step_public: step.step_public,
            replay_witness: step_witness.replay_witness.clone(),
            replay_witness_digest: chunk_transition_witness_digest(step_witness),
            halted_out,
        });
        transcript.append_fields_raw(&[F::from_u64(RV64IM_CHUNK_DONE_RAW_TAG), F::ONE]);
    }

    Ok((traces, accumulator))
}

pub(crate) fn build_rv64im_chunk_fold_step_traces_from_components(
    statement: &Rv64imFinalStatement,
    proof_digest: [u8; 32],
    kernel_export: &Rv64imKernelExportProof,
    chunk_summaries: &[FixedShapeChunkSummary],
    steps: &[Rv64imChunkTransitionWitness],
    component_digests: &Rv64imFinalProofComponentDigests,
) -> Result<Vec<Rv64imChunkFoldStepTrace>, SimpleKernelError> {
    validate_rv64im_final_statement_surface_with_parts(
        statement,
        proof_digest,
        kernel_export,
        chunk_summaries,
        component_digests,
    )?;
    let (verified_kernel, expected_chunk_summaries, _) =
        verify_folded_statement_components_with_output_and_main_carry(&statement.folded, kernel_export, steps)?;
    if expected_chunk_summaries.as_slice() != chunk_summaries {
        return Err(SimpleKernelError::Bridge(
            "RV64IM final proof chunk summaries do not match the verified export seam".into(),
        ));
    }
    let (traces, accumulator) = build_chunk_fold_step_traces_from_verified_kernel(
        statement.public_statement_digest,
        &verified_kernel,
        steps,
        Some(chunk_summaries),
    )?;
    let expected_final_accumulator = recursive_accumulator_from_carry(&accumulator);
    if expected_final_accumulator != statement.folded.final_accumulator {
        return Err(SimpleKernelError::Bridge(
            "RV64IM chunk-fold step trace final accumulator does not match the carried folded statement".into(),
        ));
    }
    Ok(traces)
}

fn build_recursive_proof(
    chunk_handoffs: &[Rv64imVerifiedKernelChunkHandoff],
    params: &NeoParams,
    structure: &CcsStructure<F>,
    log: &neo_ajtai::AjtaiSModule,
) -> Result<
    (
        Vec<Rv64imChunkTransitionWitness>,
        Vec<FixedShapeChunkSummary>,
        Rv64imRecursiveAccumulator,
        Rv64imRecursiveBuildPerf,
    ),
    SimpleKernelError,
> {
    let optimized_cache = rv64im_cached_root_main_lane_optimized_cache()?;
    let mut transcript = Poseidon2Transcript::new_raw_fields(&[F::from_u64(RV64IM_SESSION_RAW_DOMAIN_TAG)]);
    let mut accumulator = Rv64imChunkFoldCarry::seed();
    let mut steps = Vec::with_capacity(chunk_handoffs.len());
    let mut chunk_summaries = Vec::with_capacity(chunk_handoffs.len());
    let mut perf = Rv64imRecursiveBuildPerf::default();

    for (chunk_index, handoff) in chunk_handoffs.iter().enumerate() {
        let halted_out = false;
        let ((replay_witness, step), chunk_perf) = prove_rv64im_chunk_fold_verifier_step_with_perf(
            [0; 32],
            chunk_index,
            halted_out,
            handoff,
            &accumulator,
            &mut transcript,
            params,
            structure,
            log,
            &optimized_cache,
        )?;
        perf.record_chunk(&chunk_perf);
        chunk_summaries.push(FixedShapeChunkSummary::from_public_chunk(
            &handoff.public_chunk,
            step.public_chunk_digest,
            step.chunk_relation_digest,
        ));
        accumulator = step.next_carry;
        steps.push(Rv64imChunkTransitionWitness { replay_witness });
        transcript.append_fields_raw(&[F::from_u64(RV64IM_CHUNK_DONE_RAW_TAG), F::ONE]);
    }

    Ok((
        steps,
        chunk_summaries,
        recursive_accumulator_from_carry(&accumulator),
        perf,
    ))
}

fn build_final_proof(
    folded: &Rv64imFoldedStatement,
    chunk_summaries: Vec<FixedShapeChunkSummary>,
    proof: Rv64imFoldedProof,
) -> Result<(Rv64imFinalBuildProof, Rv64imFinalProofComponentDigests), SimpleKernelError> {
    let component_digests = final_proof_component_digests_from_parts(&proof.kernel_export, &proof.steps);
    let proof_digest = final_proof_digest_from_component_digests(folded, &chunk_summaries, &component_digests);
    Ok((
        Rv64imFinalBuildProof {
            proof_digest,
            kernel_export: proof.kernel_export,
            chunk_summaries,
            steps: proof.steps,
        },
        component_digests,
    ))
}

pub(crate) fn final_proof_digest_from_component_digests(
    folded: &Rv64imFoldedStatement,
    chunk_summaries: &[FixedShapeChunkSummary],
    component_digests: &Rv64imFinalProofComponentDigests,
) -> [u8; 32] {
    digest_fixed_shape_final_proof(
        &folded.digest,
        folded.chunk_count,
        chunk_summaries,
        &[component_digests.kernel_export_proof_digest],
        &component_digests.chunk_transition_digests,
    )
}

pub(crate) fn final_proof_component_digests(proof: &Rv64imFinalBuildProof) -> Rv64imFinalProofComponentDigests {
    final_proof_component_digests_from_parts(&proof.kernel_export, &proof.steps)
}

pub fn reconstruct_rv64im_final_statement_from_export_and_replay(
    public_statement_digest: [u8; 32],
    kernel_export: &Rv64imKernelExportProof,
    steps: &[Rv64imChunkTransitionWitness],
) -> Result<(Rv64imFinalStatement, Rv64imFinalBuildProof), SimpleKernelError> {
    if public_statement_digest != kernel_export.public_statement_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM reconstructed final seam public statement digest does not match the carried kernel export proof"
                .into(),
        ));
    }
    let (kernel_relation, verified_kernel) = verify_rv64im_kernel_export_proof_with_relation_output(kernel_export)?;
    if steps.len() != verified_kernel.chunk_handoffs.len() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM reconstructed final seam chunk replay count does not match the verified export relation".into(),
        ));
    }
    let (chunk_summaries, final_accumulator) = replay_recursive_steps(&verified_kernel, steps)?;
    let mut folded = Rv64imFoldedStatement {
        fold_schedule: verified_kernel.fold_schedule,
        chunk_count: verified_kernel.chunk_handoffs.len() as u64,
        semantic_step_count: verified_kernel
            .chunk_handoffs
            .iter()
            .map(|handoff| handoff.chunk_input.steps.len() as u64)
            .sum(),
        kernel_relation_digest: kernel_relation.digest,
        final_accumulator,
        digest: [0; 32],
    };
    folded.digest = folded_statement_digest(&folded);
    let (final_proof, _) = build_final_proof(
        &folded,
        chunk_summaries,
        Rv64imFoldedProof {
            kernel_export: kernel_export.clone(),
            steps: steps.to_vec(),
        },
    )?;
    let mut final_statement = Rv64imFinalStatement {
        public_statement_digest,
        folded,
        digest: [0; 32],
    };
    final_statement.digest = final_statement_digest(&final_statement);
    Ok((final_statement, final_proof))
}

pub(crate) fn final_proof_component_digests_from_parts(
    kernel_export: &Rv64imKernelExportProof,
    steps: &[Rv64imChunkTransitionWitness],
) -> Rv64imFinalProofComponentDigests {
    Rv64imFinalProofComponentDigests {
        kernel_export_proof_digest: kernel_export.digest,
        chunk_transition_digests: steps.iter().map(chunk_transition_witness_digest).collect(),
    }
}

fn append_recursive_accumulator(tr: &mut Poseidon2Transcript, accumulator: &Rv64imRecursiveAccumulator) {
    let final_main_claim_digests = final_main_claim_digests(&accumulator.final_main_claims);
    tr.append_u64s(
        b"neo.fold.next/rv64im/final_accumulator/claim_count",
        &[final_main_claim_digests.len() as u64],
    );
    tr.append_fields_iter(
        b"neo.fold.next/rv64im/final_accumulator/final_main_claim_digest",
        final_main_claim_digests.len() * 4,
        final_main_claim_digests
            .iter()
            .flat_map(|digest| digest.iter().copied()),
    );
    tr.append_message(
        b"neo.fold.next/rv64im/final_accumulator/terminal_handle",
        &accumulator.terminal_handle.0,
    );
}

pub(crate) fn final_main_claim_digests(final_main_claims: &[CeClaim<Commitment, F, K>]) -> Vec<[F; 4]> {
    let mut digests = Vec::with_capacity(final_main_claims.len());
    let mut scratch = Vec::<F>::with_capacity(2048);
    for claim in final_main_claims {
        digests.push(me_digest_poseidon_into(&mut scratch, claim));
    }
    digests
}

pub(crate) fn chunk_transition_witness_digest(step: &Rv64imChunkTransitionWitness) -> [u8; 32] {
    rv64im_chunk_replay_witness_digest(&step.replay_witness)
}

fn recursive_accumulator_from_carry(carry: &Rv64imChunkFoldCarry) -> Rv64imRecursiveAccumulator {
    Rv64imRecursiveAccumulator {
        final_main_claims: carry.main.claims.clone(),
        terminal_handle: carry.terminal_handle,
    }
}
