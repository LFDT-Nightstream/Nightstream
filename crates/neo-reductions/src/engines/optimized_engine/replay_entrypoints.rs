//! Public replay entrypoints for optimized Π_CCS.
//!
//! This file owns API wrappers that ask the core prover loop for terminal
//! state, replay witnesses, or cross-check traces. The hot proving schedule
//! remains in `prove.rs`.

use crate::error::PiCcsError;
use crate::optimized_engine::OptimizedStructureCache;
use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, CeClaim, Mat};
use neo_math::{F, K};
use neo_params::NeoParams;
use neo_transcript::Poseidon2Transcript;

use super::backend::{BackendTranscriptMode, PiCcsPhaseBackend};
use super::legacy_types::{
    PiCcsReplayOutputs, PiCcsReplayProofWitness, PiCcsReplayTerminalState, PiCcsReplayWitnessOutputs,
};
use super::prove::{run_optimized_replay_with_cache_and_perf, ReplayTraceMode};
use super::replay_binding::ReplayBinding;
use super::replay_validation::validate_replay_terminal_state;

pub fn optimized_replay_terminal_state_with_cache_and_perf<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    mcs_witnesses: &[CcsWitness<F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    me_witnesses: &[Mat<F>],
    log: &L,
    cache: &OptimizedStructureCache,
) -> Result<PiCcsReplayTerminalState, PiCcsError> {
    let (terminal_state, _rounds) = run_optimized_replay_with_cache_and_perf(
        tr,
        params,
        s,
        mcs_list,
        mcs_witnesses,
        me_inputs,
        me_witnesses,
        log,
        cache,
        ReplayBinding::claims(),
        ReplayTraceMode::TerminalState,
        true,
        None,
        None,
        None,
        BackendTranscriptMode::Replay,
    )?;
    validate_replay_terminal_state(params, s, mcs_list, me_inputs, &terminal_state)?;
    Ok(terminal_state)
}

pub fn optimized_replay_terminal_state_with_cache_and_instance_digest_and_perf<
    L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>,
>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    mcs_witnesses: &[CcsWitness<F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    me_witnesses: &[Mat<F>],
    public_instance_digest: [F; 4],
    log: &L,
    cache: &OptimizedStructureCache,
) -> Result<PiCcsReplayTerminalState, PiCcsError> {
    let (terminal_state, _rounds) = run_optimized_replay_with_cache_and_perf(
        tr,
        params,
        s,
        mcs_list,
        mcs_witnesses,
        me_inputs,
        me_witnesses,
        log,
        cache,
        ReplayBinding::instance_digest(public_instance_digest),
        ReplayTraceMode::TerminalState,
        true,
        None,
        None,
        None,
        BackendTranscriptMode::Replay,
    )?;
    validate_replay_terminal_state(params, s, mcs_list, me_inputs, &terminal_state)?;
    Ok(terminal_state)
}

pub fn optimized_replay_terminal_state_with_cache_instance_digest_and_me_input_handle_and_perf<
    L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>,
>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    mcs_witnesses: &[CcsWitness<F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    me_witnesses: &[Mat<F>],
    public_instance_digest: [F; 4],
    me_input_accumulator_handle: [F; 4],
    log: &L,
    cache: &OptimizedStructureCache,
) -> Result<PiCcsReplayTerminalState, PiCcsError> {
    let (terminal_state, _rounds) = run_optimized_replay_with_cache_and_perf(
        tr,
        params,
        s,
        mcs_list,
        mcs_witnesses,
        me_inputs,
        me_witnesses,
        log,
        cache,
        ReplayBinding::legacy_handle(public_instance_digest, me_input_accumulator_handle),
        ReplayTraceMode::TerminalState,
        true,
        None,
        None,
        None,
        BackendTranscriptMode::Replay,
    )?;
    validate_replay_terminal_state(params, s, mcs_list, me_inputs, &terminal_state)?;
    Ok(terminal_state)
}

#[allow(clippy::too_many_arguments)]
pub fn optimized_replay_terminal_state_with_phase_backend_and_transcript_mode<
    L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>,
>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    mcs_witnesses: &[CcsWitness<F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    me_witnesses: &[Mat<F>],
    public_instance_digest: [F; 4],
    me_input_accumulator_handle: [F; 4],
    log: &L,
    cache: &OptimizedStructureCache,
    phase_backend: Option<&mut dyn PiCcsPhaseBackend>,
    transcript_mode: BackendTranscriptMode,
) -> Result<PiCcsReplayTerminalState, PiCcsError> {
    let (terminal_state, _rounds) = run_optimized_replay_with_cache_and_perf(
        tr,
        params,
        s,
        mcs_list,
        mcs_witnesses,
        me_inputs,
        me_witnesses,
        log,
        cache,
        ReplayBinding::legacy_handle(public_instance_digest, me_input_accumulator_handle),
        ReplayTraceMode::TerminalState,
        false,
        phase_backend,
        None,
        None,
        transcript_mode,
    )?;
    validate_replay_terminal_state(params, s, mcs_list, me_inputs, &terminal_state)?;
    Ok(terminal_state)
}

pub fn optimized_replay_outputs_with_cache_and_perf<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    mcs_witnesses: &[CcsWitness<F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    me_witnesses: &[Mat<F>],
    log: &L,
    cache: &OptimizedStructureCache,
) -> Result<PiCcsReplayOutputs, PiCcsError> {
    let terminal_state = optimized_replay_terminal_state_with_cache_and_perf(
        tr,
        params,
        s,
        mcs_list,
        mcs_witnesses,
        me_inputs,
        me_witnesses,
        log,
        cache,
    )?;
    Ok(PiCcsReplayOutputs {
        me_outputs: terminal_state.me_outputs,
        fold_digest: terminal_state.fold_digest,
        perf: terminal_state.perf,
    })
}

pub fn optimized_replay_outputs_with_cache_and_instance_digest_and_perf<
    L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>,
>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    mcs_witnesses: &[CcsWitness<F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    me_witnesses: &[Mat<F>],
    public_instance_digest: [F; 4],
    log: &L,
    cache: &OptimizedStructureCache,
) -> Result<PiCcsReplayOutputs, PiCcsError> {
    let (terminal_state, _rounds) = run_optimized_replay_with_cache_and_perf(
        tr,
        params,
        s,
        mcs_list,
        mcs_witnesses,
        me_inputs,
        me_witnesses,
        log,
        cache,
        ReplayBinding::instance_digest(public_instance_digest),
        ReplayTraceMode::TerminalState,
        false,
        None,
        None,
        None,
        BackendTranscriptMode::Replay,
    )?;
    Ok(PiCcsReplayOutputs {
        me_outputs: terminal_state.me_outputs,
        fold_digest: terminal_state.fold_digest,
        perf: terminal_state.perf,
    })
}

pub fn optimized_replay_witness_with_cache_and_perf<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    mcs_witnesses: &[CcsWitness<F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    me_witnesses: &[Mat<F>],
    log: &L,
    cache: &OptimizedStructureCache,
) -> Result<PiCcsReplayWitnessOutputs, PiCcsError> {
    let (terminal_state, rounds) = run_optimized_replay_with_cache_and_perf(
        tr,
        params,
        s,
        mcs_list,
        mcs_witnesses,
        me_inputs,
        me_witnesses,
        log,
        cache,
        ReplayBinding::claims(),
        ReplayTraceMode::Prove,
        false,
        None,
        None,
        None,
        BackendTranscriptMode::Replay,
    )?;
    let rounds = rounds
        .expect("optimized replay-witness trace must capture proof rounds")
        .into_owned("optimized replay-witness trace unexpectedly returned deferred proof rounds")?;
    Ok(PiCcsReplayWitnessOutputs {
        me_outputs: terminal_state.me_outputs,
        replay_proof: PiCcsReplayProofWitness {
            sumcheck_rounds: rounds.sumcheck_rounds,
            sumcheck_rounds_nc: rounds.sumcheck_rounds_nc,
            header_digest: terminal_state.fold_digest,
        },
        perf: terminal_state.perf,
    })
}

pub fn optimized_replay_witness_with_cache_and_instance_digest_and_perf<
    L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>,
>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    mcs_witnesses: &[CcsWitness<F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    me_witnesses: &[Mat<F>],
    public_instance_digest: [F; 4],
    log: &L,
    cache: &OptimizedStructureCache,
) -> Result<PiCcsReplayWitnessOutputs, PiCcsError> {
    let (terminal_state, rounds) = run_optimized_replay_with_cache_and_perf(
        tr,
        params,
        s,
        mcs_list,
        mcs_witnesses,
        me_inputs,
        me_witnesses,
        log,
        cache,
        ReplayBinding::instance_digest(public_instance_digest),
        ReplayTraceMode::Prove,
        false,
        None,
        None,
        None,
        BackendTranscriptMode::Replay,
    )?;
    let rounds = rounds
        .expect("optimized replay-witness trace must capture proof rounds")
        .into_owned("optimized replay-witness trace unexpectedly returned deferred proof rounds")?;
    Ok(PiCcsReplayWitnessOutputs {
        me_outputs: terminal_state.me_outputs,
        replay_proof: PiCcsReplayProofWitness {
            sumcheck_rounds: rounds.sumcheck_rounds,
            sumcheck_rounds_nc: rounds.sumcheck_rounds_nc,
            header_digest: terminal_state.fold_digest,
        },
        perf: terminal_state.perf,
    })
}

pub fn optimized_replay_trace_with_cache_and_instance_digest_and_perf<
    L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>,
>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    mcs_witnesses: &[CcsWitness<F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    me_witnesses: &[Mat<F>],
    public_instance_digest: [F; 4],
    log: &L,
    cache: &OptimizedStructureCache,
) -> Result<(PiCcsReplayTerminalState, PiCcsReplayProofWitness), PiCcsError> {
    let (terminal_state, rounds) = run_optimized_replay_with_cache_and_perf(
        tr,
        params,
        s,
        mcs_list,
        mcs_witnesses,
        me_inputs,
        me_witnesses,
        log,
        cache,
        ReplayBinding::instance_digest(public_instance_digest),
        ReplayTraceMode::Prove,
        true,
        None,
        None,
        None,
        BackendTranscriptMode::Replay,
    )?;
    validate_replay_terminal_state(params, s, mcs_list, me_inputs, &terminal_state)?;
    let rounds = rounds
        .expect("optimized replay trace must capture proof rounds")
        .into_owned("optimized replay trace unexpectedly returned deferred proof rounds")?;
    Ok((
        terminal_state.clone(),
        PiCcsReplayProofWitness {
            sumcheck_rounds: rounds.sumcheck_rounds,
            sumcheck_rounds_nc: rounds.sumcheck_rounds_nc,
            header_digest: terminal_state.fold_digest,
        },
    ))
}

pub fn optimized_replay_trace_with_cache_instance_digest_and_me_input_handle_and_perf<
    L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>,
>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    mcs_witnesses: &[CcsWitness<F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    me_witnesses: &[Mat<F>],
    public_instance_digest: [F; 4],
    me_input_accumulator_handle: [F; 4],
    log: &L,
    cache: &OptimizedStructureCache,
) -> Result<(PiCcsReplayTerminalState, PiCcsReplayProofWitness), PiCcsError> {
    let (terminal_state, rounds) = run_optimized_replay_with_cache_and_perf(
        tr,
        params,
        s,
        mcs_list,
        mcs_witnesses,
        me_inputs,
        me_witnesses,
        log,
        cache,
        ReplayBinding::legacy_handle(public_instance_digest, me_input_accumulator_handle),
        ReplayTraceMode::Prove,
        true,
        None,
        None,
        None,
        BackendTranscriptMode::Replay,
    )?;
    validate_replay_terminal_state(params, s, mcs_list, me_inputs, &terminal_state)?;
    let rounds = rounds
        .expect("optimized replay trace must capture proof rounds")
        .into_owned("optimized replay trace unexpectedly returned deferred proof rounds")?;
    Ok((
        terminal_state.clone(),
        PiCcsReplayProofWitness {
            sumcheck_rounds: rounds.sumcheck_rounds,
            sumcheck_rounds_nc: rounds.sumcheck_rounds_nc,
            header_digest: terminal_state.fold_digest,
        },
    ))
}
