//! Owns the RV32IM per-chunk replay surface above the verified export seam.

use neo_ajtai::AjtaiSModule;
use neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash;
use neo_ccs::CcsStructure;
use neo_math::{KExtensions, F, K};
use neo_reductions::optimized_engine::OptimizedStructureCache;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

use crate::chunk_folding::{
    claim_digests, compute_chunk_replay_witness_and_relation_with_instance_digest_and_me_input_handle_and_perf,
    trace_chunk_relation_with_witness_and_instance_digest_and_me_input_handle,
    verify_chunk_relation_with_witness_and_instance_digest_and_me_input_handle_with_perf, ChunkReplayTrace,
    ChunkReplayWitness,
};
use crate::finalize::fixed_shape_recursive_step_handle;
use crate::finalize::{digest32_as_fields, digest_fields_as_digest32};
use crate::proof::{Carry, ChunkInput, ChunkProvePerf};
use crate::rv32im::final_relation::Rv32imChunkFoldTranscriptSnapshot;
use crate::rv32im::kernel::{
    prepared_step_digest, rv32im_ajtai_mixers, Rv32imChunkBridgeHandoff, Rv32imVerifiedKernelChunkHandoff,
    SimpleKernelError,
};

pub(crate) const RV32IM_CHUNK_RELATION_DIGEST_RAW_TAG: u64 = 13;

#[derive(Clone, Debug)]
pub(crate) struct Rv32imChunkRelationTrace {
    pub chunk_relation_digest: [u8; 32],
    pub ccs_outputs: Vec<neo_ccs::CeClaim<neo_ajtai::Commitment, F, K>>,
    pub ccs_replay_proof: neo_reductions::optimized_engine::PiCcsReplayProofWitness,
    pub pi_ccs_post_transcript: Rv32imChunkFoldTranscriptSnapshot,
    pub terminal_state: neo_reductions::optimized_engine::PiCcsReplayTerminalState,
    pub parent: neo_ccs::CeClaim<neo_ajtai::Commitment, F, K>,
    pub children: Vec<neo_ccs::CeClaim<neo_ajtai::Commitment, F, K>>,
    pub z_split: Vec<neo_ccs::Mat<F>>,
}

pub(crate) fn prove_rv32im_chunk_transition_with_perf(
    chunk_index: usize,
    handoff: &Rv32imVerifiedKernelChunkHandoff,
    incoming_main: &Carry,
    me_input_accumulator_handle: [F; 4],
    transcript: &mut Poseidon2Transcript,
    params: &neo_params::NeoParams,
    structure: &CcsStructure<F>,
    log: &AjtaiSModule,
    optimized_cache: &OptimizedStructureCache,
) -> Result<((ChunkReplayWitness, Carry, [u8; 32], [u8; 32]), ChunkProvePerf), SimpleKernelError> {
    // This builder only accepts a verified export handoff, so the prover hot path
    // does not replay structural bridge validation here. Verification still does.
    let ((replay_witness, proved, verified_fold_digest), perf) =
        compute_chunk_replay_witness_and_relation_with_instance_digest_and_me_input_handle_and_perf(
            transcript,
            params,
            structure,
            &handoff.chunk_input,
            incoming_main,
            log,
            rv32im_ajtai_mixers(),
            optimized_cache,
            handoff.public_chunk_instance_digest,
            me_input_accumulator_handle,
        )
        .map_err(|err| {
            SimpleKernelError::Proof(format!("RV32IM chunk transition {chunk_index} prove failed: {err}"))
        })?;
    let public_chunk_digest = handoff.public_chunk_digest;
    let chunk_relation_digest = rv32im_chunk_relation_digest_from_fold_digest(
        public_chunk_digest,
        verified_fold_digest,
        handoff.bridge_handoff.digest,
    );
    Ok((
        (
            replay_witness,
            proved.next_main,
            public_chunk_digest,
            chunk_relation_digest,
        ),
        perf,
    ))
}

pub(crate) fn verify_rv32im_chunk_relation_with_replay(
    chunk_index: usize,
    handoff: &Rv32imVerifiedKernelChunkHandoff,
    incoming_main: &Carry,
    me_input_accumulator_handle: [F; 4],
    replay_witness: &ChunkReplayWitness,
    transcript: &mut Poseidon2Transcript,
    params: &neo_params::NeoParams,
    structure: &CcsStructure<F>,
    log: &AjtaiSModule,
    optimized_cache: &OptimizedStructureCache,
) -> Result<(Carry, [u8; 32], [u8; 32]), SimpleKernelError> {
    validate_rv32im_chunk_bridge_handoff(chunk_index, handoff)?;
    let ((proved, verified_fold_digest), _) =
        verify_chunk_relation_with_witness_and_instance_digest_and_me_input_handle_with_perf(
            transcript,
            params,
            structure,
            &handoff.chunk_input,
            incoming_main,
            replay_witness,
            log,
            rv32im_ajtai_mixers(),
            optimized_cache,
            handoff.public_chunk_instance_digest,
            me_input_accumulator_handle,
        )
        .map_err(|err| {
            SimpleKernelError::Proof(format!("RV32IM chunk transition {chunk_index} verify failed: {err}"))
        })?;
    let public_chunk_digest = handoff.public_chunk_digest;
    let chunk_relation_digest = rv32im_chunk_relation_digest_from_fold_digest(
        public_chunk_digest,
        verified_fold_digest,
        handoff.bridge_handoff.digest,
    );
    Ok((proved.next_main, public_chunk_digest, chunk_relation_digest))
}

pub(crate) fn trace_rv32im_chunk_relation_with_replay(
    chunk_index: usize,
    handoff: &Rv32imVerifiedKernelChunkHandoff,
    incoming_main: &Carry,
    me_input_accumulator_handle: [F; 4],
    replay_witness: &ChunkReplayWitness,
    transcript: &mut Poseidon2Transcript,
    params: &neo_params::NeoParams,
    structure: &CcsStructure<F>,
    log: &AjtaiSModule,
    optimized_cache: &OptimizedStructureCache,
) -> Result<Rv32imChunkRelationTrace, SimpleKernelError> {
    validate_rv32im_chunk_bridge_handoff(chunk_index, handoff)?;
    trace_rv32im_chunk_relation_parts_with_replay(
        chunk_index,
        &handoff.chunk_input,
        &handoff.bridge_handoff,
        handoff.public_chunk_digest,
        handoff.public_chunk_instance_digest,
        incoming_main,
        me_input_accumulator_handle,
        replay_witness,
        transcript,
        params,
        structure,
        log,
        optimized_cache,
    )
}

pub(crate) fn trace_rv32im_chunk_relation_parts_with_replay(
    chunk_index: usize,
    chunk_input: &ChunkInput,
    bridge_handoff: &Rv32imChunkBridgeHandoff,
    public_chunk_digest_bytes: [u8; 32],
    public_chunk_instance_digest: [F; 4],
    incoming_main: &Carry,
    me_input_accumulator_handle: [F; 4],
    replay_witness: &ChunkReplayWitness,
    transcript: &mut Poseidon2Transcript,
    params: &neo_params::NeoParams,
    structure: &CcsStructure<F>,
    log: &AjtaiSModule,
    optimized_cache: &OptimizedStructureCache,
) -> Result<Rv32imChunkRelationTrace, SimpleKernelError> {
    validate_rv32im_chunk_replay_input(chunk_index, chunk_input, bridge_handoff)?;
    let trace = trace_chunk_relation_with_witness_and_instance_digest_and_me_input_handle(
        transcript,
        params,
        structure,
        chunk_input,
        incoming_main,
        replay_witness,
        log,
        rv32im_ajtai_mixers(),
        optimized_cache,
        public_chunk_instance_digest,
        me_input_accumulator_handle,
    )
    .map_err(|err| SimpleKernelError::Proof(format!("RV32IM chunk transition {chunk_index} trace failed: {err}")))?;
    Ok(trace_into_rv32im(
        trace,
        public_chunk_digest_bytes,
        bridge_handoff.digest,
    ))
}

pub(crate) fn rv32im_step_handle(
    previous_handle: [u8; 32],
    chunk_index: usize,
    chunk_start_index: usize,
    chunk_len: usize,
    chunk_relation_digest: [u8; 32],
) -> [u8; 32] {
    fixed_shape_recursive_step_handle(
        previous_handle,
        chunk_index,
        chunk_start_index,
        chunk_len,
        chunk_relation_digest,
    )
}

pub(crate) fn rv32im_chunk_replay_witness_digest(replay_witness: &ChunkReplayWitness) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/chunk_transition_witness");
    tr.append_message(b"neo.fold.next/rv32im/chunk_transition_witness/version", b"v2");
    tr.append_u64s(
        b"neo.fold.next/rv32im/chunk_transition_witness/counts",
        &[
            replay_witness.ccs_outputs.len() as u64,
            replay_witness.ccs_replay_proof.sumcheck_rounds.len() as u64,
            replay_witness.ccs_replay_proof.sumcheck_rounds_nc.len() as u64,
        ],
    );
    append_output_digests(&mut tr, &replay_witness.ccs_outputs);
    append_k_rounds(
        &mut tr,
        b"neo.fold.next/rv32im/chunk_transition_witness/fe_round",
        &replay_witness.ccs_replay_proof.sumcheck_rounds,
    );
    append_k_rounds(
        &mut tr,
        b"neo.fold.next/rv32im/chunk_transition_witness/nc_round",
        &replay_witness.ccs_replay_proof.sumcheck_rounds_nc,
    );
    tr.digest32()
}

pub(crate) fn rv32im_chunk_relation_digest_from_fold_digest(
    public_chunk_digest: [u8; 32],
    verified_fold_digest: [u8; 32],
    bridge_handoff_digest: [u8; 32],
) -> [u8; 32] {
    let mut preimage = Vec::with_capacity(1 + 3 * 4);
    preimage.push(F::from_u64(RV32IM_CHUNK_RELATION_DIGEST_RAW_TAG));
    preimage.extend_from_slice(&digest32_as_fields(public_chunk_digest));
    preimage.extend_from_slice(&digest32_as_fields(verified_fold_digest));
    preimage.extend_from_slice(&digest32_as_fields(bridge_handoff_digest));
    digest_fields_as_digest32(poseidon2_hash(&preimage))
}

fn trace_into_rv32im(
    trace: ChunkReplayTrace,
    public_chunk_digest: [u8; 32],
    bridge_handoff_digest: [u8; 32],
) -> Rv32imChunkRelationTrace {
    Rv32imChunkRelationTrace {
        chunk_relation_digest: rv32im_chunk_relation_digest_from_fold_digest(
            public_chunk_digest,
            trace.terminal_state.fold_digest,
            bridge_handoff_digest,
        ),
        ccs_outputs: trace.ccs_outputs,
        ccs_replay_proof: trace.ccs_replay_proof,
        pi_ccs_post_transcript: Rv32imChunkFoldTranscriptSnapshot {
            state: trace.ccs_post_transcript_state,
            absorbed: trace.ccs_post_transcript_absorbed,
        },
        terminal_state: trace.terminal_state,
        parent: trace.parent,
        children: trace.children,
        z_split: trace.z_split,
    }
}

fn validate_rv32im_chunk_bridge_handoff(
    chunk_index: usize,
    handoff: &Rv32imVerifiedKernelChunkHandoff,
) -> Result<(), SimpleKernelError> {
    if handoff.public_chunk.start_index != handoff.chunk_input.start_index {
        return Err(SimpleKernelError::Bridge(format!(
            "RV32IM chunk transition {chunk_index} public chunk start does not match the carried chunk input"
        )));
    }
    if handoff.public_chunk.steps.len() != handoff.chunk_input.steps.len() {
        return Err(SimpleKernelError::Bridge(format!(
            "RV32IM chunk transition {chunk_index} public chunk step count does not match the carried chunk input"
        )));
    }
    if handoff.bridge_handoff.digest != handoff.bridge_handoff.expected_digest() {
        return Err(SimpleKernelError::Bridge(format!(
            "RV32IM chunk transition {chunk_index} bridge handoff digest mismatch"
        )));
    }
    if handoff.bridge_handoff.chunk_index != chunk_index as u64 {
        return Err(SimpleKernelError::Bridge(format!(
            "RV32IM chunk transition {chunk_index} bridge handoff chunk index mismatch"
        )));
    }
    if handoff.bridge_handoff.chunk_start_index != handoff.chunk_input.start_index as u64 {
        return Err(SimpleKernelError::Bridge(format!(
            "RV32IM chunk transition {chunk_index} bridge handoff chunk start mismatch"
        )));
    }
    if handoff.bridge_handoff.public_step_count != handoff.chunk_input.steps.len() as u64 {
        return Err(SimpleKernelError::Bridge(format!(
            "RV32IM chunk transition {chunk_index} bridge handoff step count mismatch"
        )));
    }
    if handoff.bridge_handoff.step_bindings.len() != handoff.chunk_input.steps.len() {
        return Err(SimpleKernelError::Bridge(format!(
            "RV32IM chunk transition {chunk_index} bridge handoff binding count mismatch"
        )));
    }
    for (chunk_local_index, (binding, step)) in handoff
        .bridge_handoff
        .step_bindings
        .iter()
        .zip(handoff.chunk_input.steps.iter())
        .enumerate()
    {
        let public_step = handoff
            .public_chunk
            .steps
            .get(chunk_local_index)
            .ok_or_else(|| {
                SimpleKernelError::Bridge(format!(
                    "RV32IM chunk transition {chunk_index}:{chunk_local_index} public chunk step missing"
                ))
            })?;
        if public_step.label != step.label
            || public_step.mcs.m_in != step.mcs.m_in
            || public_step.mcs.x != step.mcs.x
            || public_step.mcs.c != step.mcs.c
        {
            return Err(SimpleKernelError::Bridge(format!(
                "RV32IM chunk transition {chunk_index}:{chunk_local_index} public step does not match the carried chunk input"
            )));
        }
        if binding.digest != binding.expected_digest() {
            return Err(SimpleKernelError::Bridge(format!(
                "RV32IM chunk transition {chunk_index}:{chunk_local_index} bridge binding digest mismatch"
            )));
        }
        if binding.logical_index != (handoff.chunk_input.start_index + chunk_local_index) as u64 {
            return Err(SimpleKernelError::Bridge(format!(
                "RV32IM chunk transition {chunk_index}:{chunk_local_index} bridge binding logical index mismatch"
            )));
        }
        if binding.prepared_step_digest != prepared_step_digest(step) {
            return Err(SimpleKernelError::Bridge(format!(
                "RV32IM chunk transition {chunk_index}:{chunk_local_index} bridge binding does not match the carried chunk step"
            )));
        }
    }
    Ok(())
}

fn validate_rv32im_chunk_replay_input(
    chunk_index: usize,
    chunk_input: &ChunkInput,
    bridge_handoff: &Rv32imChunkBridgeHandoff,
) -> Result<(), SimpleKernelError> {
    if bridge_handoff.digest != bridge_handoff.expected_digest() {
        return Err(SimpleKernelError::Bridge(format!(
            "RV32IM chunk transition {chunk_index} bridge handoff digest mismatch"
        )));
    }
    if bridge_handoff.chunk_index != chunk_index as u64 {
        return Err(SimpleKernelError::Bridge(format!(
            "RV32IM chunk transition {chunk_index} bridge handoff chunk index mismatch"
        )));
    }
    if bridge_handoff.chunk_start_index != chunk_input.start_index as u64 {
        return Err(SimpleKernelError::Bridge(format!(
            "RV32IM chunk transition {chunk_index} bridge handoff chunk start mismatch"
        )));
    }
    if bridge_handoff.public_step_count != chunk_input.steps.len() as u64 {
        return Err(SimpleKernelError::Bridge(format!(
            "RV32IM chunk transition {chunk_index} bridge handoff step count mismatch"
        )));
    }
    if bridge_handoff.step_bindings.len() != chunk_input.steps.len() {
        return Err(SimpleKernelError::Bridge(format!(
            "RV32IM chunk transition {chunk_index} bridge handoff binding count mismatch"
        )));
    }
    for (chunk_local_index, (binding, step)) in bridge_handoff
        .step_bindings
        .iter()
        .zip(chunk_input.steps.iter())
        .enumerate()
    {
        if binding.digest != binding.expected_digest() {
            return Err(SimpleKernelError::Bridge(format!(
                "RV32IM chunk transition {chunk_index}:{chunk_local_index} bridge binding digest mismatch"
            )));
        }
        if binding.logical_index != (chunk_input.start_index + chunk_local_index) as u64 {
            return Err(SimpleKernelError::Bridge(format!(
                "RV32IM chunk transition {chunk_index}:{chunk_local_index} bridge binding logical index mismatch"
            )));
        }
        if binding.prepared_step_digest != prepared_step_digest(step) {
            return Err(SimpleKernelError::Bridge(format!(
                "RV32IM chunk transition {chunk_index}:{chunk_local_index} bridge binding does not match the carried chunk step"
            )));
        }
    }
    Ok(())
}

fn append_output_digests(tr: &mut Poseidon2Transcript, ccs_outputs: &[neo_ccs::CeClaim<neo_ajtai::Commitment, F, K>]) {
    for digest in claim_digests(ccs_outputs) {
        tr.append_fields_iter(
            b"neo.fold.next/rv32im/chunk_transition_witness/output_digest",
            digest.len(),
            digest,
        );
    }
}

fn append_k_rounds(tr: &mut Poseidon2Transcript, label: &'static [u8], rounds: &[Vec<K>]) {
    tr.append_u64s(
        b"neo.fold.next/rv32im/chunk_transition_witness/round_count",
        &[rounds.len() as u64],
    );
    for round in rounds {
        append_k_values(tr, label, round);
    }
}

fn append_k_values(tr: &mut Poseidon2Transcript, label: &'static [u8], values: &[K]) {
    tr.append_u64s(
        b"neo.fold.next/rv32im/chunk_transition_witness/k_len",
        &[values.len() as u64],
    );
    let coeffs_per_elem = values
        .first()
        .map(|value| value.as_coeffs().len())
        .unwrap_or(0);
    tr.append_fields_iter(
        label,
        values.len().saturating_mul(coeffs_per_elem),
        values.iter().flat_map(|value| value.as_coeffs()),
    );
}
