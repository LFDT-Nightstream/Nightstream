//! Owns RV32IM root main-lane step construction and folding-session prove/verify flow.

use crate::finalize::package_session_proof;
use crate::proof::{
    partition_public_steps, Carry, ChunkInput, FoldSchedule, PackagedProof, PublicChunk, PublicStep, RunProof,
    RunProvePerf, RunVerifyPerf, StepInput,
};
use crate::prover::ShardProver;
use crate::rv32im::ccs::{semantic_row_from_execution_row, RV32IM_ROOT_ROW_WIDTH};
use crate::rv32im::lower::Rv32ExpandedRow;
use crate::session::{prove_chunks_from_slice_with_perf_and_cache, verify_packaged_with_detailed_perf_and_cache};
use crate::verifier::ShardVerifier;
use crate::witness_layout::encode_vector_for_full_width;
use neo_ccs::{traits::SModuleHomomorphism, Mat};
use neo_math::F;
use neo_reductions::api::FoldingMode;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use rayon::prelude::*;
use std::time::Instant;

use super::super::{
    perf_diagnostics::{
        RootMainLanePackagedProofProvePerf, RootMainLanePackagedProofVerifyPerf, RootMainLaneRunProofProvePerf,
        RootMainLaneRunProofVerifyPerf,
    },
    root_lane_witness::RootLaneWitness,
};
use super::ajtai::rv32im_ajtai_mixers;
use super::context::{
    cached_root_main_lane_ccs, cached_root_main_lane_optimized_cache, cached_simple_kernel_root_context,
    rv32im_root_step_cap_for_schedule, SimpleKernelRootContext,
};
use super::support::{allow_parallel_step_build, millis_since};
use super::types::SimpleKernelError;

const ROOT_MAIN_LANE_STEP_LABEL: &str = "";

fn root_encode_semantic_row(
    root_context: &SimpleKernelRootContext,
    trace_index: usize,
    semantic_row: &[F; RV32IM_ROOT_ROW_WIDTH],
) -> Result<(Vec<F>, Mat<F>), SimpleKernelError> {
    let witness = semantic_row[1..].to_vec();
    let packed = encode_vector_for_full_width(root_context.params(), RV32IM_ROOT_ROW_WIDTH, semantic_row)
        .map_err(|err| SimpleKernelError::Bridge(format!("root encoding failed for row {trace_index}: {err}")))?;
    Ok((witness, packed))
}

fn build_prepared_step_from_semantic_row(
    root_context: &SimpleKernelRootContext,
    trace_index: usize,
    semantic_row: &[F; RV32IM_ROOT_ROW_WIDTH],
) -> Result<StepInput, SimpleKernelError> {
    let (witness, z_mat) = root_encode_semantic_row(root_context, trace_index, semantic_row)?;
    Ok(StepInput {
        // Root main-lane position is already bound by chunk ordering, so per-row labels only bloat traces.
        label: ROOT_MAIN_LANE_STEP_LABEL.into(),
        mcs: neo_ccs::CcsClaim {
            c: root_context.log().commit(&z_mat),
            x: vec![F::ONE],
            m_in: 1,
        },
        witness: neo_ccs::CcsWitness { w: witness, Z: z_mat },
    })
}

fn build_public_step_from_semantic_row(
    root_context: &SimpleKernelRootContext,
    trace_index: usize,
    semantic_row: &[F; RV32IM_ROOT_ROW_WIDTH],
) -> Result<PublicStep, SimpleKernelError> {
    let z_mat = encode_vector_for_full_width(root_context.params(), RV32IM_ROOT_ROW_WIDTH, semantic_row)
        .map_err(|err| SimpleKernelError::Bridge(format!("root encoding failed for row {trace_index}: {err}")))?;
    Ok(PublicStep {
        label: ROOT_MAIN_LANE_STEP_LABEL.into(),
        mcs: neo_ccs::CcsClaim {
            c: root_context.log().commit(&z_mat),
            x: vec![F::ONE],
            m_in: 1,
        },
    })
}

pub(super) fn build_prepared_steps_from_root_lane_witness(
    root_context: &SimpleKernelRootContext,
    rows: &[Rv32ExpandedRow],
    root_lane_witness: &RootLaneWitness,
) -> Result<Vec<StepInput>, SimpleKernelError> {
    if rows.len() != root_lane_witness.semantic_rows.len() {
        return Err(SimpleKernelError::Bridge(format!(
            "root lane semantic row count {} != execution row count {}",
            root_lane_witness.semantic_rows.len(),
            rows.len()
        )));
    }
    if allow_parallel_step_build(rows.len()) {
        return rows
            .par_iter()
            .zip(root_lane_witness.semantic_rows.par_iter())
            .map(|(row, semantic_row)| {
                build_prepared_step_from_semantic_row(root_context, row.trace_index, semantic_row)
            })
            .collect();
    }
    rows.iter()
        .zip(root_lane_witness.semantic_rows.iter())
        .map(|(row, semantic_row)| build_prepared_step_from_semantic_row(root_context, row.trace_index, semantic_row))
        .collect()
}

fn build_prepared_steps_from_execution_rows_with_root_context(
    root_context: &SimpleKernelRootContext,
    rows: &[Rv32ExpandedRow],
) -> Result<Vec<StepInput>, SimpleKernelError> {
    if allow_parallel_step_build(rows.len()) {
        return rows
            .par_iter()
            .map(|row| {
                let semantic_row = semantic_row_from_execution_row(row);
                build_prepared_step_from_semantic_row(root_context, row.trace_index, &semantic_row)
            })
            .collect();
    }
    let mut steps = Vec::with_capacity(rows.len());
    for row in rows {
        let semantic_row = semantic_row_from_execution_row(row);
        steps.push(build_prepared_step_from_semantic_row(
            root_context,
            row.trace_index,
            &semantic_row,
        )?);
    }
    Ok(steps)
}

pub(crate) fn build_prepared_steps_from_execution_rows(
    rows: &[Rv32ExpandedRow],
) -> Result<Vec<StepInput>, SimpleKernelError> {
    let root_context = cached_simple_kernel_root_context()?;
    build_prepared_steps_from_execution_rows_with_root_context(root_context, rows)
}

fn build_public_steps_from_execution_rows_with_root_context(
    root_context: &SimpleKernelRootContext,
    rows: &[Rv32ExpandedRow],
) -> Result<Vec<PublicStep>, SimpleKernelError> {
    if allow_parallel_step_build(rows.len()) {
        return rows
            .par_iter()
            .map(|row| {
                let semantic_row = semantic_row_from_execution_row(row);
                build_public_step_from_semantic_row(root_context, row.trace_index, &semantic_row)
            })
            .collect();
    }
    let mut steps = Vec::with_capacity(rows.len());
    for row in rows {
        let semantic_row = semantic_row_from_execution_row(row);
        steps.push(build_public_step_from_semantic_row(
            root_context,
            row.trace_index,
            &semantic_row,
        )?);
    }
    Ok(steps)
}

fn build_public_steps_from_execution_rows(rows: &[Rv32ExpandedRow]) -> Result<Vec<PublicStep>, SimpleKernelError> {
    let root_context = cached_simple_kernel_root_context()?;
    build_public_steps_from_execution_rows_with_root_context(root_context, rows)
}

fn same_public_step(lhs: &PublicStep, rhs: &PublicStep) -> bool {
    lhs.label == rhs.label
        && lhs.mcs.m_in == rhs.mcs.m_in
        && lhs.mcs.x == rhs.mcs.x
        && lhs.mcs.c.d == rhs.mcs.c.d
        && lhs.mcs.c.kappa == rhs.mcs.c.kappa
        && lhs.mcs.c.data == rhs.mcs.c.data
}

fn same_public_chunk(lhs: &PublicChunk, rhs: &PublicChunk) -> bool {
    lhs.start_index == rhs.start_index
        && lhs.steps.len() == rhs.steps.len()
        && lhs
            .steps
            .iter()
            .zip(rhs.steps.iter())
            .all(|(lhs, rhs)| same_public_step(lhs, rhs))
}

fn root_main_lane_packaged_verify_perf(
    prepare_public_steps_ms: f64,
    public_chunk_match_ms: f64,
    packaged_verify: crate::finalize::PackagedVerifyPerf,
    total_ms: f64,
) -> RootMainLanePackagedProofVerifyPerf {
    RootMainLanePackagedProofVerifyPerf {
        prepare_public_steps_ms,
        public_chunk_match_ms,
        packaged_statement_digest_ms: packaged_verify.statement_digest_ms,
        packaged_chunk_digests_ms: packaged_verify.chunk_digests_ms,
        packaged_final_main_claim_digests_ms: packaged_verify.final_main_claim_digests_ms,
        packaged_statement_hash_ms: packaged_verify.statement_hash_ms,
        packaged_schedule_checks_ms: packaged_verify.schedule_checks_ms,
        packaged_proof_digest_ms: packaged_verify.proof_digest_ms,
        packaged_final_claim_match_ms: packaged_verify.final_claim_match_ms,
        packaged_total_ms: packaged_verify.total_ms,
        session: packaged_verify.session,
        total_ms,
    }
}

fn root_main_lane_chunk_len(schedule: FoldSchedule, row_count: usize) -> Result<usize, SimpleKernelError> {
    rv32im_root_step_cap_for_schedule(schedule, row_count)
}

pub fn prove_root_main_lane_packaged_proof_with_perf(
    rows: &[Rv32ExpandedRow],
    schedule: FoldSchedule,
) -> Result<(PackagedProof, RootMainLanePackagedProofProvePerf), SimpleKernelError> {
    let (packaged, _, perf) = prove_root_main_lane_packaged_proof_with_inputs_and_perf(rows, schedule)?;
    Ok((packaged, perf))
}

pub(crate) fn prove_root_main_lane_packaged_proof_with_inputs_and_perf(
    rows: &[Rv32ExpandedRow],
    schedule: FoldSchedule,
) -> Result<(PackagedProof, Vec<ChunkInput>, RootMainLanePackagedProofProvePerf), SimpleKernelError> {
    let total_started = Instant::now();
    let chunk_len = root_main_lane_chunk_len(schedule, rows.len())?;
    let root_context = SimpleKernelRootContext::new_for_step_cap(chunk_len)?;
    let ccs = cached_root_main_lane_ccs()?;
    let prepare_steps_started = Instant::now();
    let steps = build_prepared_steps_from_execution_rows_with_root_context(&root_context, rows)?;
    let prepare_steps_ms = millis_since(prepare_steps_started);
    let chunk_inputs = crate::proof::partition_step_inputs(schedule, steps)?;
    let public_chunks = chunk_inputs
        .iter()
        .map(ChunkInput::public)
        .collect::<Vec<_>>();
    let (session, session_perf) = prove_chunks_from_slice_with_perf_and_cache(
        FoldingMode::Optimized,
        schedule,
        root_context.params(),
        ccs,
        &chunk_inputs,
        root_context.log(),
        rv32im_ajtai_mixers(),
        None,
    )?;
    let packaged = package_session_proof(public_chunks, session)?;
    Ok((
        packaged,
        chunk_inputs,
        RootMainLanePackagedProofProvePerf {
            prepare_steps_ms,
            session: session_perf,
            total_ms: millis_since(total_started),
        },
    ))
}

pub fn prove_root_main_lane_run_proof_with_perf(
    rows: &[Rv32ExpandedRow],
    schedule: FoldSchedule,
) -> Result<(RunProof, RootMainLaneRunProofProvePerf), SimpleKernelError> {
    let total_started = Instant::now();
    let chunk_len = root_main_lane_chunk_len(schedule, rows.len())?;
    let root_context = SimpleKernelRootContext::new_for_step_cap(chunk_len)?;
    let ccs = cached_root_main_lane_ccs()?;
    let optimized_cache = cached_root_main_lane_optimized_cache()?;
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/session");
    let mut main_carry = Carry::default();
    let mut proof = RunProof {
        fold_schedule: schedule,
        ..RunProof::default()
    };
    let mut session = RunProvePerf::default();
    let mut prepare_steps_ms = 0.0;
    let mut start_index = 0usize;
    while start_index < rows.len() {
        let end_index = (start_index + chunk_len).min(rows.len());
        let prepare_steps_started = Instant::now();
        let steps =
            build_prepared_steps_from_execution_rows_with_root_context(&root_context, &rows[start_index..end_index])?;
        prepare_steps_ms += millis_since(prepare_steps_started);
        let chunk = ChunkInput { start_index, steps };
        let (proved, chunk_perf) = ShardProver::prove_chunk_with_perf(
            FoldingMode::Optimized,
            &mut tr,
            root_context.params(),
            ccs,
            &chunk,
            &main_carry,
            root_context.log(),
            rv32im_ajtai_mixers(),
            Some(&optimized_cache),
        )?;
        main_carry = proved.next_main;
        proof.chunks.push(proved.proof);
        session.chunks.push(chunk_perf);
        tr.append_message(b"neo.fold.next/chunk_done", &[1]);
        start_index = end_index;
    }
    proof.final_main_claims = main_carry.claims;
    session.total_ms = millis_since(total_started);
    Ok((
        proof,
        RootMainLaneRunProofProvePerf {
            prepare_steps_ms,
            session,
            total_ms: millis_since(total_started),
        },
    ))
}

pub fn verify_root_main_lane_packaged_proof_with_public_rows(
    rows: &[Rv32ExpandedRow],
    packaged: &PackagedProof,
) -> Result<RootMainLanePackagedProofVerifyPerf, SimpleKernelError> {
    let total_started = Instant::now();
    let chunk_len = root_main_lane_chunk_len(packaged.statement.fold_schedule, rows.len())?;
    let root_context = SimpleKernelRootContext::new_for_step_cap(chunk_len)?;
    let ccs = cached_root_main_lane_ccs()?;
    let prepare_public_steps_started = Instant::now();
    let public_steps = build_public_steps_from_execution_rows_with_root_context(&root_context, rows)?;
    let prepare_public_steps_ms = millis_since(prepare_public_steps_started);
    let public_chunk_match_started = Instant::now();
    let expected_chunks = partition_public_steps(packaged.statement.fold_schedule, public_steps)?;
    if packaged.statement.chunks.len() != expected_chunks.len() {
        return Err(SimpleKernelError::Bridge(format!(
            "RV32IM root main-lane packaged proof chunk count {} != expected chunk count {}",
            packaged.statement.chunks.len(),
            expected_chunks.len()
        )));
    }
    for (idx, (actual, expected)) in packaged
        .statement
        .chunks
        .iter()
        .zip(expected_chunks.iter())
        .enumerate()
    {
        if !same_public_chunk(actual, expected) {
            return Err(SimpleKernelError::Bridge(format!(
                "RV32IM root main-lane packaged proof public chunk {idx} mismatch"
            )));
        }
    }
    let public_chunk_match_ms = millis_since(public_chunk_match_started);
    let (_, packaged_verify) = verify_packaged_with_detailed_perf_and_cache(
        FoldingMode::Optimized,
        root_context.params(),
        ccs,
        packaged,
        rv32im_ajtai_mixers(),
        None,
    )?;
    Ok(root_main_lane_packaged_verify_perf(
        prepare_public_steps_ms,
        public_chunk_match_ms,
        packaged_verify,
        millis_since(total_started),
    ))
}

pub(crate) fn verify_root_main_lane_packaged_proof_with_verified_public_statement_with_perf(
    packaged: &PackagedProof,
) -> Result<RootMainLanePackagedProofVerifyPerf, SimpleKernelError> {
    let total_started = Instant::now();
    let step_cap =
        rv32im_root_step_cap_for_schedule(packaged.statement.fold_schedule, packaged.statement.public_step_count())?;
    let root_context = SimpleKernelRootContext::new_for_step_cap(step_cap)?;
    let ccs = cached_root_main_lane_ccs()?;
    let optimized_cache = cached_root_main_lane_optimized_cache()?;
    let (_, packaged_verify) = verify_packaged_with_detailed_perf_and_cache(
        FoldingMode::Optimized,
        root_context.params(),
        ccs,
        packaged,
        rv32im_ajtai_mixers(),
        Some(optimized_cache),
    )?;
    Ok(root_main_lane_packaged_verify_perf(
        0.0,
        0.0,
        packaged_verify,
        millis_since(total_started),
    ))
}

pub fn verify_root_main_lane_run_proof_with_public_rows(
    rows: &[Rv32ExpandedRow],
    proof: &RunProof,
) -> Result<RootMainLaneRunProofVerifyPerf, SimpleKernelError> {
    let total_started = Instant::now();
    let root_context = cached_simple_kernel_root_context()?;
    let ccs = cached_root_main_lane_ccs()?;
    let chunk_len = root_main_lane_chunk_len(proof.fold_schedule, rows.len())?;
    let optimized_cache = cached_root_main_lane_optimized_cache()?;
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/session");
    let mut main_carry = &[][..];
    let mut session = RunVerifyPerf::default();
    let mut prepare_public_steps_ms = 0.0;
    let mut start_index = 0usize;
    for (chunk_index, chunk_proof) in proof.chunks.iter().enumerate() {
        let end_index = (start_index + chunk_len).min(rows.len());
        let prepare_public_steps_started = Instant::now();
        let steps = build_public_steps_from_execution_rows(&rows[start_index..end_index])?;
        prepare_public_steps_ms += millis_since(prepare_public_steps_started);
        let chunk = PublicChunk { start_index, steps };
        let (next_main, chunk_perf) = ShardVerifier::verify_chunk_with_perf(
            FoldingMode::Optimized,
            &mut tr,
            root_context.params(),
            ccs,
            &chunk,
            main_carry,
            chunk_proof,
            rv32im_ajtai_mixers(),
            Some(&optimized_cache),
        )?;
        main_carry = next_main;
        session.chunks.push(chunk_perf);
        tr.append_message(b"neo.fold.next/chunk_done", &[1]);
        start_index = end_index;
        if chunk_index + 1 == proof.chunks.len() && start_index != rows.len() {
            return Err(SimpleKernelError::Bridge(
                "RV32IM root main-lane run proof ended before covering all rows".into(),
            ));
        }
    }
    if start_index != rows.len() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM root main-lane run proof chunk count does not cover the provided rows".into(),
        ));
    }
    if main_carry != proof.final_main_claims.as_slice() {
        return Err(SimpleKernelError::Proof(
            "RV32IM root main-lane run proof final carried claims mismatch".into(),
        ));
    }
    session.total_ms = millis_since(total_started);
    Ok(RootMainLaneRunProofVerifyPerf {
        prepare_public_steps_ms,
        session,
        total_ms: millis_since(total_started),
    })
}
