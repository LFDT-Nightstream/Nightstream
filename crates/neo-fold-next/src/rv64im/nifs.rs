//! Owns the native RV64IM fold-step boundary between a running recursive state and one fresh chunk step.

use crate::chunk_relation::ChunkReplayWitness;
use crate::finalize::FixedShapeChunkSummary;
use crate::proof::Carry;
use crate::rv64im::chunk_fold_step::{
    adapt_rv64im_chunk_to_fresh_ccs, build_rv64im_chunk_step_public, Rv64imAccumulatorHandle, Rv64imChunkFoldCarry,
    Rv64imChunkStepPublic,
};
use crate::rv64im::chunk_relation::{
    prove_rv64im_chunk_transition_with_perf, rv64im_step_handle, trace_rv64im_chunk_relation_with_replay,
    Rv64imChunkRelationTrace,
};
use crate::rv64im::final_relation::{
    rv64im_chunk_fold_carried_transcript_snapshot, Rv64imChunkFoldState, Rv64imChunkFoldTranscriptSnapshot,
};
use crate::rv64im::kernel::{
    rv64im_cached_root_main_lane_context, rv64im_cached_root_main_lane_optimized_cache,
    Rv64imVerifiedKernelChunkHandoff, SimpleKernelError,
};
use neo_ajtai::Commitment;
use neo_ccs::CeClaim;
use neo_math::{F, K};
use neo_reductions::optimized_engine::PiCcsReplayProofWitness;
use neo_transcript::Poseidon2Transcript;

#[derive(Clone, Debug)]
pub(crate) struct Rv64imNifsRunningWitness {
    pub state: Rv64imChunkFoldState,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Rv64imNifsFreshInstance {
    pub step_public: Rv64imChunkStepPublic,
    pub chunk_summary: FixedShapeChunkSummary,
}

#[derive(Clone, Debug)]
pub(crate) struct Rv64imNifsFreshWitness {
    pub handoff: Rv64imVerifiedKernelChunkHandoff,
    pub state_out: Rv64imChunkFoldState,
}

#[derive(Clone, Debug)]
pub(crate) struct Rv64imNifsProof {
    ccs_outputs: Vec<CeClaim<Commitment, F, K>>,
    // Lower Pi_CCS replay transport. RV64IM NIFS derives its carried fold state from the
    // verified trace terminal fold digest, not from any digest bytes carried inside this witness.
    replay_transport: PiCcsReplayProofWitness,
    parent: CeClaim<Commitment, F, K>,
    children: Vec<CeClaim<Commitment, F, K>>,
}

impl Rv64imNifsProof {
    pub(crate) fn from_replay_parts(
        ccs_outputs: Vec<CeClaim<Commitment, F, K>>,
        replay_transport: PiCcsReplayProofWitness,
        parent: CeClaim<Commitment, F, K>,
        children: Vec<CeClaim<Commitment, F, K>>,
    ) -> Self {
        Self {
            ccs_outputs,
            replay_transport,
            parent,
            children,
        }
    }

    fn replay_witness(&self) -> ChunkReplayWitness {
        ChunkReplayWitness {
            ccs_outputs: self.ccs_outputs.clone(),
            ccs_replay_proof: self.replay_transport.clone(),
        }
    }
}

fn build_rv64im_nifs_proof_from_trace(trace: Rv64imChunkRelationTrace) -> Rv64imNifsProof {
    Rv64imNifsProof::from_replay_parts(trace.ccs_outputs, trace.ccs_replay_proof, trace.parent, trace.children)
}

fn validate_rv64im_nifs_ce_claim_surface(
    claim: &CeClaim<Commitment, F, K>,
    label: &str,
) -> Result<(), SimpleKernelError> {
    if claim.ct.len() != claim.y_ring.len() {
        return Err(SimpleKernelError::Proof(format!(
            "{label} scalar view length does not match y_ring"
        )));
    }
    for (row_idx, (row, ct)) in claim.y_ring.iter().zip(claim.ct.iter()).enumerate() {
        let constant_term = row
            .first()
            .copied()
            .ok_or_else(|| SimpleKernelError::Proof(format!("{label} y_ring[{row_idx}] is empty")))?;
        if constant_term != *ct {
            return Err(SimpleKernelError::Proof(format!(
                "{label} ct[{row_idx}] does not match y_ring[{row_idx}][0]"
            )));
        }
    }
    Ok(())
}

fn verify_rv64im_nifs_pi_ccs_trace(
    running: &Rv64imNifsRunningWitness,
    fresh_instance: &Rv64imNifsFreshInstance,
    fresh_witness: &Rv64imNifsFreshWitness,
    proof: &Rv64imNifsProof,
) -> Result<(Rv64imChunkRelationTrace, Poseidon2Transcript), SimpleKernelError> {
    validate_rv64im_nifs_surface(running, fresh_instance, fresh_witness)?;
    let (params, log, structure) = rv64im_cached_root_main_lane_context()?;
    let optimized_cache = rv64im_cached_root_main_lane_optimized_cache()?;
    let mut transcript =
        Poseidon2Transcript::from_state_and_absorbed(running.state.transcript.state, running.state.transcript.absorbed);
    let trace = trace_rv64im_chunk_relation_with_replay(
        fresh_instance.step_public.chunk_index as usize,
        &fresh_witness.handoff,
        &running.state.carry.main,
        &proof.replay_witness(),
        &mut transcript,
        params,
        structure,
        log,
        optimized_cache,
    )
    .map_err(|err| SimpleKernelError::Proof(format!("RV64IM NIFS proof Pi_CCS replay failed: {err}")))?;
    Ok((trace, transcript))
}

fn verify_rv64im_nifs_pi_rlc_parent(
    trace: &Rv64imChunkRelationTrace,
    proof: &Rv64imNifsProof,
) -> Result<(), SimpleKernelError> {
    validate_rv64im_nifs_ce_claim_surface(&trace.parent, "RV64IM NIFS proof Pi_RLC parent claim")?;
    if trace.parent != proof.parent {
        return Err(SimpleKernelError::Proof(
            "RV64IM NIFS proof RLC parent claim does not match the verified chunk relation trace".into(),
        ));
    }
    Ok(())
}

fn verify_rv64im_nifs_pi_dec_children(
    trace: &Rv64imChunkRelationTrace,
    proof: &Rv64imNifsProof,
) -> Result<(), SimpleKernelError> {
    for (child_index, claim) in trace.children.iter().enumerate() {
        validate_rv64im_nifs_ce_claim_surface(claim, &format!("RV64IM NIFS proof Pi_DEC child {child_index}"))?;
    }
    if trace.children != proof.children {
        return Err(SimpleKernelError::Proof(
            "RV64IM NIFS proof DEC child claims do not match the verified chunk relation trace".into(),
        ));
    }
    Ok(())
}

fn derive_rv64im_nifs_state_from_trace(
    running: &Rv64imNifsRunningWitness,
    fresh_instance: &Rv64imNifsFreshInstance,
    fresh_witness: &Rv64imNifsFreshWitness,
    trace: &Rv64imChunkRelationTrace,
    transcript: &Poseidon2Transcript,
) -> Result<Rv64imChunkFoldState, SimpleKernelError> {
    let chunk_relation_digest = trace.chunk_relation_digest;
    let next_carry = Rv64imChunkFoldCarry {
        main: Carry {
            claims: trace.children.clone(),
            witnesses: trace.z_split.clone(),
        },
        terminal_handle: Rv64imAccumulatorHandle(rv64im_step_handle(
            running.state.carry.terminal_handle.0,
            fresh_instance.step_public.chunk_index as usize,
            fresh_witness.handoff.public_chunk.start_index,
            fresh_witness.handoff.public_chunk.steps.len(),
            chunk_relation_digest,
        )),
    };
    let fresh = adapt_rv64im_chunk_to_fresh_ccs(&fresh_witness.handoff);
    let expected_step_public = build_rv64im_chunk_step_public(
        fresh_instance.step_public.program_digest,
        fresh_instance.step_public.chunk_index as usize,
        &fresh,
        &running.state.carry,
        &next_carry,
        fresh_instance.step_public.halted_out,
    );
    if expected_step_public != fresh_instance.step_public {
        return Err(SimpleKernelError::Bridge(
            "RV64IM NIFS fresh instance public step does not match the verified fold-step output".into(),
        ));
    }
    let expected_summary = FixedShapeChunkSummary::from_public_chunk(
        &fresh_witness.handoff.public_chunk,
        fresh_witness.handoff.public_chunk_digest,
        chunk_relation_digest,
    );
    if expected_summary != fresh_instance.chunk_summary {
        return Err(SimpleKernelError::Bridge(
            "RV64IM NIFS fresh instance chunk summary does not match the verified fold-step output".into(),
        ));
    }
    if next_carry.main.claims != fresh_witness.state_out.carry.main.claims
        || next_carry.main.witnesses != fresh_witness.state_out.carry.main.witnesses
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM NIFS carried next-main state does not match the verified fold-step output".into(),
        ));
    }
    if next_carry.terminal_handle != fresh_witness.state_out.carry.terminal_handle {
        return Err(SimpleKernelError::Bridge(
            "RV64IM NIFS carried terminal handle does not match the verified fold-step output".into(),
        ));
    }
    let transcript_out = rv64im_chunk_fold_carried_transcript_snapshot(&Rv64imChunkFoldTranscriptSnapshot {
        state: transcript.state(),
        absorbed: transcript.absorbed(),
    });
    if transcript_out != fresh_witness.state_out.transcript {
        return Err(SimpleKernelError::Bridge(
            "RV64IM NIFS carried transcript_out does not match the verified fold-step output".into(),
        ));
    }
    Ok(Rv64imChunkFoldState {
        carry: next_carry,
        transcript: transcript_out,
    })
}

pub(crate) fn prove_rv64im_nifs_step(
    running: &Rv64imNifsRunningWitness,
    fresh_instance: &Rv64imNifsFreshInstance,
    fresh_witness: &Rv64imNifsFreshWitness,
) -> Result<Rv64imNifsProof, SimpleKernelError> {
    validate_rv64im_nifs_surface(running, fresh_instance, fresh_witness)?;
    let (params, log, structure) = rv64im_cached_root_main_lane_context()?;
    let optimized_cache = rv64im_cached_root_main_lane_optimized_cache()?;
    let mut prove_transcript =
        Poseidon2Transcript::from_state_and_absorbed(running.state.transcript.state, running.state.transcript.absorbed);
    let ((replay_witness, _, _, _), _) = prove_rv64im_chunk_transition_with_perf(
        fresh_instance.step_public.chunk_index as usize,
        &fresh_witness.handoff,
        &running.state.carry.main,
        &mut prove_transcript,
        params,
        structure,
        log,
        optimized_cache,
    )?;
    let mut trace_transcript =
        Poseidon2Transcript::from_state_and_absorbed(running.state.transcript.state, running.state.transcript.absorbed);
    let trace = trace_rv64im_chunk_relation_with_replay(
        fresh_instance.step_public.chunk_index as usize,
        &fresh_witness.handoff,
        &running.state.carry.main,
        &replay_witness,
        &mut trace_transcript,
        params,
        structure,
        log,
        optimized_cache,
    )?;
    let _ = derive_rv64im_nifs_state_from_trace(running, fresh_instance, fresh_witness, &trace, &trace_transcript)?;
    Ok(build_rv64im_nifs_proof_from_trace(trace))
}

fn verify_rv64im_nifs_structured_surfaces(
    running: &Rv64imNifsRunningWitness,
    fresh_instance: &Rv64imNifsFreshInstance,
    fresh_witness: &Rv64imNifsFreshWitness,
    proof: &Rv64imNifsProof,
) -> Result<Rv64imChunkFoldState, SimpleKernelError> {
    let (trace, transcript) = verify_rv64im_nifs_pi_ccs_trace(running, fresh_instance, fresh_witness, proof)?;
    verify_rv64im_nifs_pi_rlc_parent(&trace, proof)?;
    verify_rv64im_nifs_pi_dec_children(&trace, proof)?;
    derive_rv64im_nifs_state_from_trace(running, fresh_instance, fresh_witness, &trace, &transcript)
}

pub(crate) fn verify_rv64im_nifs_step(
    running: &Rv64imNifsRunningWitness,
    fresh_instance: &Rv64imNifsFreshInstance,
    fresh_witness: &Rv64imNifsFreshWitness,
    proof: &Rv64imNifsProof,
) -> Result<Rv64imNifsRunningWitness, SimpleKernelError> {
    Ok(Rv64imNifsRunningWitness {
        state: verify_rv64im_nifs_structured_surfaces(running, fresh_instance, fresh_witness, proof)?,
    })
}

fn validate_rv64im_nifs_surface(
    running: &Rv64imNifsRunningWitness,
    fresh_instance: &Rv64imNifsFreshInstance,
    fresh_witness: &Rv64imNifsFreshWitness,
) -> Result<(), SimpleKernelError> {
    let expected_step_count = fresh_instance
        .step_public
        .step_hi
        .checked_sub(fresh_instance.step_public.step_lo)
        .ok_or_else(|| SimpleKernelError::Bridge("RV64IM NIFS step bounds are not monotone".into()))?;
    if fresh_instance.chunk_summary.start_index != fresh_instance.step_public.step_lo {
        return Err(SimpleKernelError::Bridge(
            "RV64IM NIFS summary start index does not match step_public.step_lo".into(),
        ));
    }
    if fresh_instance.chunk_summary.public_step_count != expected_step_count {
        return Err(SimpleKernelError::Bridge(
            "RV64IM NIFS summary public_step_count does not match step_public span".into(),
        ));
    }
    if fresh_witness.handoff.public_chunk.start_index as u64 != fresh_instance.step_public.step_lo {
        return Err(SimpleKernelError::Bridge(
            "RV64IM NIFS handoff start index does not match step_public.step_lo".into(),
        ));
    }
    if fresh_witness.handoff.public_chunk.steps.len() as u64 != expected_step_count {
        return Err(SimpleKernelError::Bridge(
            "RV64IM NIFS handoff step count does not match step_public span".into(),
        ));
    }
    if running.state.carry.terminal_handle.0 != fresh_instance.step_public.state_in {
        return Err(SimpleKernelError::Bridge(
            "RV64IM NIFS running state terminal handle does not match step_public.state_in".into(),
        ));
    }
    if fresh_witness.state_out.carry.terminal_handle.0 != fresh_instance.step_public.state_out {
        return Err(SimpleKernelError::Bridge(
            "RV64IM NIFS carried state_out terminal handle does not match step_public.state_out".into(),
        ));
    }
    if running.state.transcript.absorbed > neo_params::poseidon2_goldilocks::RATE
        || fresh_witness.state_out.transcript.absorbed > neo_params::poseidon2_goldilocks::RATE
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM NIFS transcript snapshot absorbed count exceeds the Poseidon2 rate".into(),
        ));
    }
    Ok(())
}
