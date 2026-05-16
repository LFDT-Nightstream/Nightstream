use neo_ajtai::Commitment;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, CeClaim};
use neo_math::{F, K};
use neo_params::NeoParams;
use neo_reductions::api::{prove, FoldingMode, PiCcsProof};
use neo_reductions::error::PiCcsError;
use neo_reductions::optimized_engine::{
    optimized_prove_with_cache_and_instance_digest_and_perf,
    optimized_replay_outputs_with_cache_and_instance_digest_and_perf,
    optimized_replay_trace_with_cache_and_instance_digest_and_perf,
    optimized_replay_trace_with_cache_instance_digest_and_me_input_handle_and_perf,
    optimized_replay_witness_with_cache_and_instance_digest_and_perf,
    optimized_verify_with_cache_and_instance_digest_and_me_input_handle_and_perf, OptimizedStructureCache,
    PiCcsProvePerf, PiCcsReplayProofWitness, PiCcsReplayTerminalState, PiCcsReplayWitnessOutputs,
};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use std::time::Instant;

use super::types::ChunkReplayWitness;
use crate::proof::Carry;

pub(super) struct ProvedPiCcs {
    pub(super) outputs: Vec<CeClaim<Commitment, F, K>>,
    pub(super) proof: PiCcsProof,
    pub(super) perf: PiCcsProvePerf,
    pub(super) elapsed_ms: f64,
    pub(super) fold_digest: [u8; 32],
}

pub(super) struct ReplayedPiCcsOutputs {
    pub(super) outputs: Vec<CeClaim<Commitment, F, K>>,
    pub(super) perf: PiCcsProvePerf,
    pub(super) elapsed_ms: f64,
    pub(super) fold_digest: [u8; 32],
}

pub(super) struct ReplayWitnessPiCcs {
    pub(super) outputs: Vec<CeClaim<Commitment, F, K>>,
    pub(super) replay_proof: PiCcsReplayProofWitness,
    pub(super) perf: PiCcsProvePerf,
    pub(super) elapsed_ms: f64,
    pub(super) fold_digest: [u8; 32],
}

pub(super) struct VerifiedPiCcs {
    pub(super) perf: PiCcsProvePerf,
    pub(super) elapsed_ms: f64,
    pub(super) fold_digest: [u8; 32],
}

pub(super) struct TracedPiCcs {
    pub(super) terminal_state: PiCcsReplayTerminalState,
    pub(super) replay_proof: PiCcsReplayProofWitness,
    pub(super) post_transcript_state: [F; neo_params::poseidon2_goldilocks::WIDTH],
    pub(super) post_transcript_absorbed: usize,
}

pub fn build_inert_chunk_replay_proof_witness(
    fe_round_lengths: &[u64],
    nc_round_lengths: &[u64],
) -> PiCcsReplayProofWitness {
    PiCcsReplayProofWitness {
        sumcheck_rounds: fe_round_lengths
            .iter()
            .map(|len| vec![K::ZERO; *len as usize])
            .collect(),
        sumcheck_rounds_nc: nc_round_lengths
            .iter()
            .map(|len| vec![K::ZERO; *len as usize])
            .collect(),
        header_digest: [0; 32],
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prove_pi_ccs<L>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Commitment, F>],
    fresh_witnesses: &[CcsWitness<F>],
    incoming_main: &Carry,
    public_chunk_digest: [F; 4],
    log: &L,
    optimized_cache: Option<&OptimizedStructureCache>,
) -> Result<ProvedPiCcs, PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
{
    let started = Instant::now();
    let (outputs, proof, perf) = if matches!(mode, FoldingMode::Optimized) {
        let cache = optimized_cache.ok_or_else(|| {
            PiCcsError::InvalidInput("missing optimized structure cache for optimized chunk relation".into())
        })?;
        optimized_prove_with_cache_and_instance_digest_and_perf(
            tr,
            params,
            s,
            fresh_claims,
            fresh_witnesses,
            &incoming_main.claims,
            &incoming_main.witnesses,
            public_chunk_digest,
            log,
            cache,
        )?
    } else {
        let (outputs, proof) = prove(
            mode,
            tr,
            params,
            s,
            fresh_claims,
            fresh_witnesses,
            &incoming_main.claims,
            &incoming_main.witnesses,
            log,
        )?;
        (outputs, proof, PiCcsProvePerf::default())
    };
    let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let fold_digest = fold_digest_from_proof(&proof)?;
    Ok(ProvedPiCcs {
        outputs,
        proof,
        perf,
        elapsed_ms,
        fold_digest,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn replay_pi_ccs_outputs<L>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Commitment, F>],
    fresh_witnesses: &[CcsWitness<F>],
    incoming_main: &Carry,
    public_chunk_digest: [F; 4],
    log: &L,
    optimized_cache: &OptimizedStructureCache,
) -> Result<ReplayedPiCcsOutputs, PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
{
    let started = Instant::now();
    let replay = optimized_replay_outputs_with_cache_and_instance_digest_and_perf(
        tr,
        params,
        s,
        fresh_claims,
        fresh_witnesses,
        &incoming_main.claims,
        &incoming_main.witnesses,
        public_chunk_digest,
        log,
        optimized_cache,
    )?;
    Ok(ReplayedPiCcsOutputs {
        outputs: replay.me_outputs,
        perf: replay.perf,
        elapsed_ms: started.elapsed().as_secs_f64() * 1_000.0,
        fold_digest: replay.fold_digest,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn compute_pi_ccs_replay_witness<L>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Commitment, F>],
    fresh_witnesses: &[CcsWitness<F>],
    incoming_main: &Carry,
    public_chunk_digest: [F; 4],
    me_input_accumulator_handle: Option<[F; 4]>,
    log: &L,
    optimized_cache: &OptimizedStructureCache,
) -> Result<ReplayWitnessPiCcs, PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
{
    let started = Instant::now();
    let replay = if let Some(handle) = me_input_accumulator_handle {
        let (terminal_state, replay_proof) =
            optimized_replay_trace_with_cache_instance_digest_and_me_input_handle_and_perf(
                tr,
                params,
                s,
                fresh_claims,
                fresh_witnesses,
                &incoming_main.claims,
                &incoming_main.witnesses,
                public_chunk_digest,
                handle,
                log,
                optimized_cache,
            )?;
        PiCcsReplayWitnessOutputs {
            me_outputs: terminal_state.me_outputs,
            replay_proof,
            perf: terminal_state.perf,
        }
    } else {
        optimized_replay_witness_with_cache_and_instance_digest_and_perf(
            tr,
            params,
            s,
            fresh_claims,
            fresh_witnesses,
            &incoming_main.claims,
            &incoming_main.witnesses,
            public_chunk_digest,
            log,
            optimized_cache,
        )?
    };
    let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let fold_digest = replay.replay_proof.header_digest;
    Ok(ReplayWitnessPiCcs {
        outputs: replay.me_outputs,
        replay_proof: replay.replay_proof,
        perf: replay.perf,
        elapsed_ms,
        fold_digest,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn verify_pi_ccs_replay_witness(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Commitment, F>],
    incoming_main: &Carry,
    replay_witness: &ChunkReplayWitness,
    public_chunk_digest: [F; 4],
    me_input_accumulator_handle: Option<[F; 4]>,
    optimized_cache: &OptimizedStructureCache,
) -> Result<VerifiedPiCcs, PiCcsError> {
    let started = Instant::now();
    let ccs_proof = replay_witness.ccs_replay_proof.to_pi_ccs_proof();
    let (ok, verify_perf) = if let Some(handle) = me_input_accumulator_handle {
        optimized_verify_with_cache_and_instance_digest_and_me_input_handle_and_perf(
            tr,
            params,
            s,
            fresh_claims,
            &incoming_main.claims,
            &replay_witness.ccs_outputs,
            &ccs_proof,
            optimized_cache,
            public_chunk_digest,
            handle,
        )?
    } else {
        neo_reductions::optimized_engine::optimized_verify_with_cache_and_instance_digest_and_perf(
            tr,
            params,
            s,
            fresh_claims,
            &incoming_main.claims,
            &replay_witness.ccs_outputs,
            &ccs_proof,
            optimized_cache,
            public_chunk_digest,
        )?
    };
    if !ok {
        return Err(PiCcsError::ProtocolError(
            "optimized replay witness does not verify against chunk relation".into(),
        ));
    }
    let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let expected_fold_digest = replay_witness.ccs_replay_proof.header_digest;
    let fold_digest = tr.digest32();
    if fold_digest != expected_fold_digest {
        return Err(PiCcsError::ProtocolError(
            "optimized replay witness header digest does not match transcript replay".into(),
        ));
    }
    Ok(VerifiedPiCcs {
        perf: PiCcsProvePerf {
            bind_ms: verify_perf.bind_ms,
            sample_challenges_ms: verify_perf.bind_sample_challenges_ms,
            fe_sumcheck_ms: verify_perf.fe_sumcheck_ms,
            nc_sumcheck_ms: verify_perf.nc_sumcheck_ms,
            output_materialize_ms: verify_perf.output_checks_ms + verify_perf.terminal_ms,
            total_ms: verify_perf.total_ms,
        },
        elapsed_ms,
        fold_digest,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn trace_pi_ccs_replay<L>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Commitment, F>],
    fresh_witnesses: &[CcsWitness<F>],
    incoming_main: &Carry,
    replay_witness: &ChunkReplayWitness,
    public_chunk_digest: [F; 4],
    me_input_accumulator_handle: Option<[F; 4]>,
    log: &L,
    optimized_cache: &OptimizedStructureCache,
) -> Result<TracedPiCcs, PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
{
    let (terminal_state, derived_replay_proof) = if let Some(handle) = me_input_accumulator_handle {
        optimized_replay_trace_with_cache_instance_digest_and_me_input_handle_and_perf(
            tr,
            params,
            s,
            fresh_claims,
            fresh_witnesses,
            &incoming_main.claims,
            &incoming_main.witnesses,
            public_chunk_digest,
            handle,
            log,
            optimized_cache,
        )?
    } else {
        optimized_replay_trace_with_cache_and_instance_digest_and_perf(
            tr,
            params,
            s,
            fresh_claims,
            fresh_witnesses,
            &incoming_main.claims,
            &incoming_main.witnesses,
            public_chunk_digest,
            log,
            optimized_cache,
        )?
    };
    let post_transcript_state = tr.state();
    let post_transcript_absorbed = tr.absorbed();
    if terminal_state.me_outputs != replay_witness.ccs_outputs {
        return Err(PiCcsError::ProtocolError(
            "optimized replay outputs do not match the carried chunk replay witness outputs".into(),
        ));
    }
    if derived_replay_proof != replay_witness.ccs_replay_proof {
        return Err(PiCcsError::ProtocolError(
            "optimized replay proof rounds do not match the carried chunk replay witness".into(),
        ));
    }
    Ok(TracedPiCcs {
        terminal_state,
        replay_proof: replay_witness.ccs_replay_proof.clone(),
        post_transcript_state,
        post_transcript_absorbed,
    })
}

fn fold_digest_from_proof(ccs_proof: &PiCcsProof) -> Result<[u8; 32], PiCcsError> {
    ccs_proof
        .header_digest
        .as_slice()
        .try_into()
        .map_err(|_| PiCcsError::ProtocolError("Π_CCS header digest must be 32 bytes".into()))
}
