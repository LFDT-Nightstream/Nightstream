//! Optimized prove implementation for PiCcsEngine.

#![allow(non_snake_case)]

use crate::error::PiCcsError;
use crate::optimized_engine::{
    PiCcsDeferredProof, PiCcsProof, PiCcsProvePerf, PiCcsReplayTerminalState, PiCcsTerminalOutputShell,
};
use crate::sumcheck::RoundOracle;
use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, CeClaim, Mat};
use neo_math::{KExtensions, F, K};
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

use super::backend::{
    BackendTranscriptMode, FePhaseTraceRequest, FeSumcheckBackend, NcColTraceRequest, NcSumcheckBackend,
    PiCcsPhaseBackend, PiCcsPhaseTraceRequest,
};
use super::phase_trace::{
    apply_fe_backend_summary, apply_fe_backend_trace, apply_nc_backend_trace, apply_pi_ccs_phase_summary,
    apply_pi_ccs_phase_trace, AppliedPiCcsPhase, FeBackendSummaryApply, FeBackendTraceApply, NcBackendTraceApply,
    PiCcsPhaseApply, PiCcsPhaseSummaryApply,
};
use super::proof_assembly::{
    proof_from_terminal_state, DeferredFeRowRounds, DeferredProofRounds, OptimizedProofRounds,
};
use super::replay_binding::ReplayBinding;
use super::terminal_outputs::build_me_outputs_from_terminal_surfaces;
use super::transcript_segments::{append_nc_sumcheck_prolog, sample_public_challenges_with_backend};
use super::OptimizedStructureCache;
use crate::engines::utils;

#[derive(Clone, Copy, Eq, PartialEq)]
pub(super) enum ReplayTraceMode {
    Prove,
    DeferredProof,
    TerminalState,
}

impl ReplayTraceMode {
    fn captures_host_rounds(self) -> bool {
        matches!(self, Self::Prove | Self::DeferredProof)
    }

    fn exports_summary_rounds_immediately(self) -> bool {
        matches!(self, Self::Prove)
    }
}

pub(super) fn owned_rounds(rounds: DeferredProofRounds) -> Result<OptimizedProofRounds, PiCcsError> {
    match rounds {
        DeferredProofRounds::Owned(rounds) => Ok(rounds),
        DeferredProofRounds::PhaseBackend | DeferredProofRounds::FeRows(_) => Err(PiCcsError::InvalidInput(
            "optimized prove expected immediately owned proof rounds".into(),
        )),
    }
}

#[cfg(feature = "perf-timers")]
fn perf_epoch_nanos() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

/// Optimized prove implementation.
pub fn optimized_prove<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    mcs_witnesses: &[CcsWitness<F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    me_witnesses: &[Mat<F>],
    log: &L,
) -> Result<(Vec<CeClaim<Cmt, F, K>>, PiCcsProof), PiCcsError> {
    let cache = OptimizedStructureCache::build(s)?;
    optimized_prove_with_cache(
        tr,
        params,
        s,
        mcs_list,
        mcs_witnesses,
        me_inputs,
        me_witnesses,
        log,
        &cache,
    )
}

pub fn optimized_prove_with_cache<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    mcs_witnesses: &[CcsWitness<F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    me_witnesses: &[Mat<F>],
    log: &L,
    cache: &OptimizedStructureCache,
) -> Result<(Vec<CeClaim<Cmt, F, K>>, PiCcsProof), PiCcsError> {
    let (me_outputs, proof, _perf) = optimized_prove_with_cache_and_perf(
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
    Ok((me_outputs, proof))
}

pub fn optimized_prove_with_cache_and_perf<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    mcs_witnesses: &[CcsWitness<F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    me_witnesses: &[Mat<F>],
    log: &L,
    cache: &OptimizedStructureCache,
) -> Result<(Vec<CeClaim<Cmt, F, K>>, PiCcsProof, PiCcsProvePerf), PiCcsError> {
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
    let rounds = owned_rounds(rounds.expect("optimized prove trace must capture proof rounds"))?;
    let proof = proof_from_terminal_state(&terminal_state, rounds);

    Ok((terminal_state.me_outputs, proof, terminal_state.perf))
}

pub fn optimized_prove_with_cache_and_instance_digest_and_perf<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
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
) -> Result<(Vec<CeClaim<Cmt, F, K>>, PiCcsProof, PiCcsProvePerf), PiCcsError> {
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
    let rounds = owned_rounds(rounds.expect("optimized prove trace must capture proof rounds"))?;
    let proof = proof_from_terminal_state(&terminal_state, rounds);

    Ok((terminal_state.me_outputs, proof, terminal_state.perf))
}

/// Variant of [`optimized_prove_with_cache_and_instance_digest_and_perf`] that
/// binds the ME-input accumulator handle into the transcript instead of the
/// per-claim ME-input projection digests. The caller supplies a 4-lane handle
/// (e.g. the running-accumulator digest) which must be recomputed from
/// authoritative claim data on the verify side via the matching `_me_input_handle_`
/// verify entry. Body is identical to the non-handle variant except for the
/// `Some(me_input_accumulator_handle)` argument passed into the replay driver.
pub fn optimized_prove_with_cache_and_instance_digest_and_me_input_handle_and_perf<
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
) -> Result<
    (
        Vec<CeClaim<Cmt, F, K>>,
        PiCcsProof,
        PiCcsProvePerf,
        super::PiDecProverPrecompute,
    ),
    PiCcsError,
> {
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
    let rounds = owned_rounds(rounds.expect("optimized prove trace must capture proof rounds"))?;
    let proof = proof_from_terminal_state(&terminal_state, rounds);
    let pi_dec_precompute = terminal_state
        .pi_dec_precompute
        .clone()
        .ok_or_else(|| PiCcsError::InvalidInput("CPU Pi_CCS prove did not produce Pi_DEC precomputation".into()))?;

    Ok((terminal_state.me_outputs, proof, terminal_state.perf, pi_dec_precompute))
}

/// `optimized_prove_with_cache_and_instance_digest_and_me_input_handle_and_perf`
/// with an active [`FeSumcheckBackend`] driving the FE row rounds. Output is
/// field-identical to the CPU path by the backend contract.
#[allow(clippy::too_many_arguments)]
pub fn optimized_prove_with_device_backends<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
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
    fe_backend: Option<&mut dyn FeSumcheckBackend>,
    nc_backend: Option<&mut dyn NcSumcheckBackend>,
) -> Result<(Vec<CeClaim<Cmt, F, K>>, PiCcsProof, PiCcsProvePerf), PiCcsError> {
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
        false,
        None,
        fe_backend,
        nc_backend,
        BackendTranscriptMode::Replay,
    )?;
    let rounds = owned_rounds(rounds.expect("optimized prove trace must capture proof rounds"))?;
    let proof = proof_from_terminal_state(&terminal_state, rounds);

    Ok((terminal_state.me_outputs, proof, terminal_state.perf))
}

/// [`optimized_prove_with_device_backends`] with explicit control over
/// whether device transcript segments are replayed online or adopted from
/// device snapshots. `Replay` is for parity/debug gates; `DeviceSnapshot`
/// is for the timed prover path.
#[allow(clippy::too_many_arguments)]
pub fn optimized_prove_with_device_backends_and_transcript_mode<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
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
    fe_backend: Option<&mut dyn FeSumcheckBackend>,
    nc_backend: Option<&mut dyn NcSumcheckBackend>,
    transcript_mode: BackendTranscriptMode,
) -> Result<(Vec<CeClaim<Cmt, F, K>>, PiCcsProof, PiCcsProvePerf), PiCcsError> {
    optimized_prove_with_phase_backend_and_transcript_mode(
        tr,
        params,
        s,
        mcs_list,
        mcs_witnesses,
        me_inputs,
        me_witnesses,
        public_instance_digest,
        me_input_accumulator_handle,
        log,
        cache,
        None,
        fe_backend,
        nc_backend,
        transcript_mode,
    )
}

/// Whole-phase-capable Π_CCS prove entrypoint.
///
/// `phase_backend` is the intended CUDA migration seam: it can eventually own
/// FE rows + Ajtai tail + NC prolog/columns as one device transcript segment.
/// When absent, or when it declines a shape, the existing FE/NC hooks remain
/// the canonical path.
#[allow(clippy::too_many_arguments)]
pub fn optimized_prove_with_phase_backend_and_transcript_mode<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
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
    fe_backend: Option<&mut dyn FeSumcheckBackend>,
    nc_backend: Option<&mut dyn NcSumcheckBackend>,
    transcript_mode: BackendTranscriptMode,
) -> Result<(Vec<CeClaim<Cmt, F, K>>, PiCcsProof, PiCcsProvePerf), PiCcsError> {
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
        false,
        phase_backend,
        fe_backend,
        nc_backend,
        transcript_mode,
    )?;
    let rounds = owned_rounds(rounds.expect("optimized prove trace must capture proof rounds"))?;
    let proof = proof_from_terminal_state(&terminal_state, rounds);

    Ok((terminal_state.me_outputs, proof, terminal_state.perf))
}

/// Run Pi_CCS to terminal state while leaving proof-round logs backend-owned.
///
/// The returned object exposes the terminal CE claims immediately and can
/// later assemble the public proof by asking the same phase backend to export
/// resident FE/NC coefficient logs.
#[allow(clippy::too_many_arguments)]
pub fn optimized_defer_prove_with_phase_backend_and_transcript_mode<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
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
    phase_backend: &mut dyn PiCcsPhaseBackend,
    transcript_mode: BackendTranscriptMode,
) -> Result<PiCcsDeferredProof, PiCcsError> {
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
        ReplayTraceMode::DeferredProof,
        false,
        Some(phase_backend),
        None,
        None,
        transcript_mode,
    )?;
    Ok(PiCcsDeferredProof::new(
        terminal_state,
        _rounds.unwrap_or(DeferredProofRounds::PhaseBackend),
    ))
}

/// Run Pi_CCS to terminal state while leaving FE row proof coefficients
/// backend-owned. This preserves the fast row-trace execution grain while
/// deferring proof-log materialization to proof assembly.
#[allow(clippy::too_many_arguments)]
pub fn optimized_defer_prove_with_device_backends_and_transcript_mode<
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
    fe_backend: &mut dyn FeSumcheckBackend,
    nc_backend: Option<&mut dyn NcSumcheckBackend>,
    transcript_mode: BackendTranscriptMode,
) -> Result<PiCcsDeferredProof, PiCcsError> {
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
        ReplayTraceMode::DeferredProof,
        false,
        None,
        Some(fe_backend),
        nc_backend,
        transcript_mode,
    )?;
    let rounds = rounds.ok_or_else(|| {
        PiCcsError::InvalidInput("deferred device-backend Pi_CCS proof did not return proof source".into())
    })?;
    Ok(PiCcsDeferredProof::new(terminal_state, rounds))
}

pub(super) fn run_optimized_replay_with_cache_and_perf<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    mcs_witnesses: &[CcsWitness<F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    me_witnesses: &[Mat<F>],
    log: &L,
    cache: &OptimizedStructureCache,
    binding: ReplayBinding,
    mode: ReplayTraceMode,
    capture_pi_dec_precompute: bool,
    mut phase_backend: Option<&mut dyn PiCcsPhaseBackend>,
    mut fe_backend: Option<&mut dyn FeSumcheckBackend>,
    mut nc_backend: Option<&mut dyn NcSumcheckBackend>,
    backend_transcript_mode: BackendTranscriptMode,
) -> Result<(PiCcsReplayTerminalState, Option<DeferredProofRounds>), PiCcsError> {
    let total_started = std::time::Instant::now();
    if mcs_list.is_empty() {
        return Err(PiCcsError::InvalidInput("optimized_prove: empty mcs_list".into()));
    }
    if mcs_list.len() != mcs_witnesses.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "optimized_prove: |mcs_list| mismatch (expected {}, got {})",
            mcs_list.len(),
            mcs_witnesses.len()
        )));
    }
    if me_inputs.len() != me_witnesses.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "optimized_prove: |me_inputs| mismatch (expected {}, got {})",
            me_inputs.len(),
            me_witnesses.len()
        )));
    }

    // Dims + transcript binding
    let bind_started = std::time::Instant::now();
    let dims = utils::build_dims_and_policy(params, s)?;
    let transcript_variant = binding.transcript_variant();
    if let Some(public_instance_digest) = binding.public_instance_digest {
        utils::bind_header_and_instance_digest_with_digest_for_variant(
            tr,
            params,
            s,
            dims,
            cache.mat_digest(),
            &public_instance_digest,
            transcript_variant,
        )?;
    } else {
        utils::bind_header_and_instances_with_digest_for_variant(
            tr,
            params,
            s,
            mcs_list,
            dims,
            cache.mat_digest(),
            transcript_variant,
        )?;
    }
    if let Some(handle) = binding.me_input_accumulator_handle {
        utils::bind_me_inputs_accumulator_handle(tr, me_inputs.len(), &handle)?;
    } else {
        utils::bind_me_inputs(tr, me_inputs)?;
    }
    let bind_ms = bind_started.elapsed().as_secs_f64() * 1_000.0;
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "optimized_prove: 1. bind/header        {bind_ms:>9.2}ms @{}",
        perf_epoch_nanos()
    );

    // Sample challenges
    let sample_started = std::time::Instant::now();
    let block_pending = binding.block_pending();
    if block_pending.is_some() && (phase_backend.is_some() || nc_backend.is_some()) {
        return Err(PiCcsError::InvalidInput(
            "block-lane NC currently requires the CPU NC path".into(),
        ));
    }
    let mut ch = if block_pending.is_some() {
        utils::sample_challenges(tr, dims.ell_d, dims.ell)?
    } else {
        sample_public_challenges_with_backend(
            tr,
            &mut phase_backend,
            backend_transcript_mode,
            dims.ell_d,
            dims.ell,
            dims.ell_m,
        )?
    };
    let block_challenges = block_pending
        .as_ref()
        .map(|_| super::block_lane_replay::sample_challenges(tr, dims, &mut ch))
        .transpose()?;
    let sample_challenges_ms = sample_started.elapsed().as_secs_f64() * 1_000.0;
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "optimized_prove: 2. sample challenges  {sample_challenges_ms:>9.2}ms @{}",
        perf_epoch_nanos()
    );

    // Initial sum: use the public T computed from ME inputs and α
    // (not the full hypercube sum Q, which includes MCS/NC terms).
    // This ensures invalid witnesses fail the first sumcheck invariant.
    let r_inputs = utils::shared_me_input_r(me_inputs, dims.ell_n)?;
    let initial_sum = phase_backend
        .as_deref_mut()
        .and_then(|backend| backend.claimed_initial_sum(&ch, mcs_witnesses.len(), me_inputs.len(), s.t()))
        .or_else(|| {
            fe_backend
                .as_deref_mut()
                .and_then(|backend| backend.claimed_initial_sum(&ch, mcs_witnesses.len(), me_inputs.len(), s.t()))
        })
        .unwrap_or_else(|| super::claimed_initial_sum_from_inputs_with_k_mcs(s, &ch, mcs_witnesses.len(), me_inputs));

    #[cfg(feature = "debug-logs")]
    {
        eprintln!("\n========== OPTIMIZED PROVE ==========");
        eprintln!(
            "[prove] k_total = {} (mcs_witnesses={}, me_witnesses={}, me_inputs={})",
            mcs_witnesses.len() + me_witnesses.len(),
            mcs_witnesses.len(),
            me_witnesses.len(),
            me_inputs.len()
        );
        eprintln!(
            "[prove] dims: ell_d={}, ell_n={}, d_sc={}",
            dims.ell_d, dims.ell_n, dims.d_sc
        );
        eprintln!("[prove] gamma = {:?}", ch.gamma);
        eprintln!("[prove] initial_sum (public T) = {:?}", initial_sum);

        // For debugging: compute the full hypercube sum to compare
        let full_sum = super::sum_q_over_hypercube_paper_exact(
            s,
            params,
            mcs_witnesses,
            me_witnesses,
            &ch,
            dims.ell_d,
            dims.ell_n,
            r_inputs,
        );
        let diff = full_sum - initial_sum;
        eprintln!("[prove] full Q sum = {:?}", full_sum);
        eprintln!("[prove] difference (full - T) = {:?}", diff);
        eprintln!("[prove] breakdown:");
        eprintln!("[prove]   T (Eval block) = {:?}", initial_sum);
        eprintln!("[prove]   eq(X,β)·(F+NC) = {:?}", diff);
        if full_sum != initial_sum {
            eprintln!("[prove] WARNING: Full sum != T! This means eq(X,β)·(F+NC) ≠ 0");
            eprintln!("[prove]   For valid witnesses, this should be zero!");
            eprintln!("[prove]   Either:");
            eprintln!("[prove]     - F(CCS constraints) doesn't hold → circuit witness is invalid");
            eprintln!("[prove]     - NC(norm constraints) doesn't hold → X doesn't match Z columns");
        }
    }

    // Optimized oracles with cached sparse formats and factored algebra
    #[cfg(feature = "perf-timers")]
    let oracle_started = std::time::Instant::now();
    let phase_shape_candidate =
        block_challenges.is_none() && phase_backend.is_some() && mcs_witnesses.len() + me_witnesses.len() > 1;
    let mut oracle = if phase_shape_candidate {
        let backend = phase_backend
            .as_deref_mut()
            .expect("phase candidate requires a phase backend");
        super::oracle::OptimizedOracle::new_with_sparse_and_superneo_cache_and_backend(
            s,
            params,
            mcs_witnesses,
            me_witnesses,
            ch.clone(),
            dims.ell_d,
            dims.ell_n,
            dims.d_sc,
            r_inputs,
            cache.sparse_arc(),
            cache.superneo_arc(),
            backend.fe_backend_for_oracle(),
        )
    } else {
        super::oracle::OptimizedOracle::new_with_sparse_and_superneo_cache_and_backend(
            s,
            params,
            mcs_witnesses,
            me_witnesses,
            ch.clone(),
            dims.ell_d,
            dims.ell_n,
            dims.d_sc,
            r_inputs,
            cache.sparse_arc(),
            cache.superneo_arc(),
            fe_backend.as_deref_mut(),
        )
    };
    #[cfg(feature = "perf-timers")]
    {
        let oracle_build_ms = oracle_started.elapsed().as_secs_f64() * 1_000.0;
        eprintln!(
            "optimized_prove: 3. oracle build       {oracle_build_ms:>9.2}ms @{}",
            perf_epoch_nanos()
        );
    }

    // ---------------------------------------------------------------------
    // FE sumcheck channel (SplitNcV1).
    // ---------------------------------------------------------------------
    tr.append_fields_raw(&[F::from_u64(crate::engines::utils::PI_CCS_SUMCHECK_FE_RAW_DOMAIN_TAG)]);
    tr.append_fields_raw(&[F::from_u64(crate::engines::utils::PI_CCS_SUMCHECK_INITIAL_RAW_TAG)]);
    tr.append_fields_raw(&initial_sum.as_coeffs());
    tr.append_fields_raw(&[F::from_u64(crate::sumcheck::SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG)]);

    let mut running_sum = initial_sum;
    let mut sumcheck_rounds = mode
        .captures_host_rounds()
        .then(|| Vec::with_capacity(oracle.num_rounds()));
    let mut sumcheck_chals: Vec<K> = Vec::with_capacity(oracle.num_rounds());
    let mut oracle_nc_from_phase = phase_shape_candidate.then(|| {
        let backend = phase_backend
            .as_ref()
            .expect("phase candidate requires a phase backend");
        if backend.defers_nc_digit_tables() {
            super::oracle::NcOracle::new_with_deferred_digit_tables(
                s,
                params,
                mcs_witnesses,
                me_witnesses,
                ch.clone(),
                dims.ell_d,
                dims.ell_m,
                dims.d_sc,
            )
        } else {
            super::oracle::NcOracle::new(
                s,
                params,
                mcs_witnesses,
                me_witnesses,
                ch.clone(),
                dims.ell_d,
                dims.ell_m,
                dims.d_sc,
            )
        }
    });
    let mut initial_sum_nc = K::ZERO;
    let mut running_sum_nc = initial_sum_nc;
    let mut sumcheck_rounds_nc = mode
        .captures_host_rounds()
        .then(|| Vec::with_capacity(dims.ell_m + dims.ell_d));
    let mut sumcheck_chals_nc: Vec<K> = Vec::with_capacity(dims.ell_m + dims.ell_d);

    let fe_sumcheck_started = std::time::Instant::now();
    #[cfg(feature = "perf-timers")]
    let mut fe_eval_ms = 0.0f64;
    #[cfg(feature = "perf-timers")]
    let mut fe_interp_ms = 0.0f64;
    #[cfg(feature = "perf-timers")]
    let mut fe_fold_ms = 0.0f64;
    #[cfg(feature = "perf-timers")]
    let mut fe_largest_eval_ms = 0.0f64;
    #[cfg(feature = "perf-timers")]
    let mut fe_largest_eval_round = 0usize;
    #[cfg(feature = "perf-timers")]
    let mut fe_largest_fold_ms = 0.0f64;
    #[cfg(feature = "perf-timers")]
    let mut fe_largest_fold_round = 0usize;
    let mut first_host_round = 0usize;
    let mut first_nc_host_round_from_phase = None;
    let mut phase_terminal_surfaces = None;
    let mut phase_summary_export_pending = false;
    let mut fe_row_summary_export_pending = None;
    if let (Some(backend), Some(oracle_nc)) = (phase_backend.as_deref_mut(), oracle_nc_from_phase.as_mut()) {
        if backend.start(&oracle.row_phase_snapshot(), &oracle_nc.col_phase_snapshot()) {
            if let Some((cache, n_eff, witnesses)) = oracle.ajtai_backend_trace_context() {
                let transcript_state = tr.state();
                let transcript_absorbed = tr.absorbed();
                let fe_row_rounds = dims.ell_n.min(oracle.num_rounds());
                let nc_rounds = dims.ell_m.min(oracle_nc.num_rounds());
                let summary_allowed = mode == ReplayTraceMode::TerminalState
                    || (mode.captures_host_rounds() && !backend_transcript_mode.replays());
                let summary = if summary_allowed {
                    backend.summarize_pi_ccs_phase(PiCcsPhaseTraceRequest {
                        fe: FePhaseTraceRequest {
                            transcript_state,
                            transcript_absorbed,
                            row_rounds: fe_row_rounds,
                            tail_rounds: dims.ell_d,
                            cache,
                            n_eff,
                            witnesses: witnesses.clone(),
                            alpha: &ch.alpha,
                            beta_a: &ch.beta_a,
                            beta_r: &ch.beta_r,
                            r_inputs,
                            gamma: ch.gamma,
                            k_mcs: mcs_witnesses.len(),
                        },
                        fe_initial_sum: initial_sum,
                        nc_col_rounds: nc_rounds,
                        nc_tail_rounds: dims.ell_d,
                        nc_tail_coeff_count: dims.d_sc + 1,
                        nc_initial_sum: initial_sum_nc,
                    })
                } else {
                    None
                };
                match summary {
                    Some(summary) => {
                        let applied = apply_pi_ccs_phase_summary(PiCcsPhaseSummaryApply {
                            tr,
                            oracle: &mut oracle,
                            oracle_nc,
                            summary,
                            fe_row_rounds,
                            nc_col_rounds: nc_rounds,
                            nc_tail_rounds: dims.ell_d,
                            transcript_mode: backend_transcript_mode,
                            sumcheck_chals: &mut sumcheck_chals,
                            running_sum: &mut running_sum,
                            sumcheck_chals_nc: &mut sumcheck_chals_nc,
                            running_sum_nc: &mut running_sum_nc,
                        })?;
                        let AppliedPiCcsPhase {
                            first_host_round: next_fe_round,
                            first_nc_host_round,
                            terminal_surfaces,
                        } = applied;
                        first_host_round = next_fe_round;
                        first_nc_host_round_from_phase = Some(first_nc_host_round);
                        phase_terminal_surfaces = terminal_surfaces;
                        phase_summary_export_pending = mode.captures_host_rounds();
                    }
                    None if mode != ReplayTraceMode::TerminalState => {
                        let trace = backend.prove_pi_ccs_phase(PiCcsPhaseTraceRequest {
                            fe: FePhaseTraceRequest {
                                transcript_state,
                                transcript_absorbed,
                                row_rounds: fe_row_rounds,
                                tail_rounds: dims.ell_d,
                                cache,
                                n_eff,
                                witnesses,
                                alpha: &ch.alpha,
                                beta_a: &ch.beta_a,
                                beta_r: &ch.beta_r,
                                r_inputs,
                                gamma: ch.gamma,
                                k_mcs: mcs_witnesses.len(),
                            },
                            fe_initial_sum: initial_sum,
                            nc_col_rounds: nc_rounds,
                            nc_tail_rounds: dims.ell_d,
                            nc_tail_coeff_count: dims.d_sc + 1,
                            nc_initial_sum: initial_sum_nc,
                        });
                        if let Some(trace) = trace {
                            let applied = apply_pi_ccs_phase_trace(PiCcsPhaseApply {
                                tr,
                                oracle: &mut oracle,
                                oracle_nc,
                                trace,
                                fe_row_rounds,
                                nc_col_rounds: nc_rounds,
                                nc_tail_rounds: dims.ell_d,
                                transcript_mode: backend_transcript_mode,
                                sumcheck_rounds: &mut sumcheck_rounds,
                                sumcheck_chals: &mut sumcheck_chals,
                                running_sum: &mut running_sum,
                                sumcheck_rounds_nc: &mut sumcheck_rounds_nc,
                                sumcheck_chals_nc: &mut sumcheck_chals_nc,
                                running_sum_nc: &mut running_sum_nc,
                            })?;
                            let AppliedPiCcsPhase {
                                first_host_round: next_fe_round,
                                first_nc_host_round,
                                terminal_surfaces,
                            } = applied;
                            first_host_round = next_fe_round;
                            first_nc_host_round_from_phase = Some(first_nc_host_round);
                            phase_terminal_surfaces = terminal_surfaces;
                        }
                    }
                    None => {}
                }
            }
        }
    }

    let backend_active = if first_host_round == oracle.num_rounds() {
        false
    } else {
        match fe_backend.as_deref_mut() {
            Some(backend) => backend.start(&oracle.row_phase_snapshot()),
            None => false,
        }
    };
    if !backend_active && first_host_round != oracle.num_rounds() {
        oracle.materialize_deferred_row_equality_tables();
    }
    if oracle.row_phase_requires_backend() && first_host_round != oracle.num_rounds() && !backend_active {
        return Err(PiCcsError::InvalidInput(
            "FE backend deferred row data but did not accept the row-phase snapshot".into(),
        ));
    }

    if backend_active && dims.ell_n > 0 {
        let row_rounds = dims.ell_n.min(oracle.num_rounds());
        let transcript_state = tr.state();
        let transcript_absorbed = tr.absorbed();
        let full_trace = if row_rounds == dims.ell_n {
            oracle
                .ajtai_backend_trace_context()
                .and_then(|(cache, n_eff, witnesses)| {
                    fe_backend
                        .as_deref_mut()
                        .expect("fe backend is active")
                        .fe_phase_trace_from_transcript(FePhaseTraceRequest {
                            transcript_state,
                            transcript_absorbed,
                            row_rounds,
                            tail_rounds: dims.ell_d,
                            cache,
                            n_eff,
                            witnesses,
                            alpha: &ch.alpha,
                            beta_a: &ch.beta_a,
                            beta_r: &ch.beta_r,
                            r_inputs,
                            gamma: ch.gamma,
                            k_mcs: mcs_witnesses.len(),
                        })
                })
        } else {
            None
        };
        if let Some(mut trace) = full_trace {
            if trace.ajtai_y_eval.is_none() {
                if let Some((cache, row_challenges, n_eff, witnesses)) = oracle.ajtai_backend_challenge_context() {
                    let backend = fe_backend.as_deref_mut().expect("fe backend is active");
                    trace.ajtai_y_eval =
                        backend.ajtai_y_eval_from_row_challenges(cache, row_challenges, n_eff, &witnesses);
                    if trace.ajtai_y_eval.is_none() {
                        let chi_r = neo_ccs::utils::tensor_point_parallel::<K>(row_challenges);
                        trace.ajtai_y_eval = backend.ajtai_y_eval(cache, &chi_r, n_eff, &witnesses);
                    }
                }
            }
            let expected_rounds = oracle.num_rounds();
            first_host_round = apply_fe_backend_trace(FeBackendTraceApply {
                tr,
                oracle: &mut oracle,
                trace,
                expected_rounds,
                row_rounds: dims.ell_n,
                transcript_mode: backend_transcript_mode,
                sumcheck_rounds: &mut sumcheck_rounds,
                sumcheck_chals: &mut sumcheck_chals,
                running_sum: &mut running_sum,
                trace_name: "FE backend phase trace",
            })?;
        } else if mode == ReplayTraceMode::DeferredProof
            && !backend_transcript_mode.replays()
            && row_rounds == dims.ell_n
        {
            if let Some(summary) = fe_backend
                .as_deref_mut()
                .expect("fe backend is active")
                .row_round_summary_from_transcript(transcript_state, transcript_absorbed, row_rounds, running_sum)
            {
                first_host_round = apply_fe_backend_summary(FeBackendSummaryApply {
                    tr,
                    oracle: &mut oracle,
                    summary,
                    expected_rounds: row_rounds,
                    transcript_mode: backend_transcript_mode,
                    sumcheck_chals: &mut sumcheck_chals,
                    running_sum: &mut running_sum,
                    trace_name: "FE backend row summary",
                })?;
                fe_row_summary_export_pending = Some(row_rounds);
            }
        } else if let Some(trace) = fe_backend
            .as_deref_mut()
            .expect("fe backend is active")
            .row_round_trace_from_transcript(transcript_state, transcript_absorbed, row_rounds)
        {
            first_host_round = apply_fe_backend_trace(FeBackendTraceApply {
                tr,
                oracle: &mut oracle,
                trace,
                expected_rounds: row_rounds,
                row_rounds,
                transcript_mode: backend_transcript_mode,
                sumcheck_rounds: &mut sumcheck_rounds,
                sumcheck_chals: &mut sumcheck_chals,
                running_sum: &mut running_sum,
                trace_name: "FE backend row trace",
            })?;
        }
    }

    for round_idx in first_host_round..oracle.num_rounds() {
        let backend_row_round = backend_active && round_idx < dims.ell_n;
        if round_idx == dims.ell_n {
            if let Some((cache, row_challenges, n_eff, witnesses)) = oracle.ajtai_backend_challenge_context() {
                if let Some(backend) = fe_backend.as_deref_mut() {
                    let y_eval = backend
                        .ajtai_y_eval_from_row_challenges(cache, row_challenges, n_eff, &witnesses)
                        .or_else(|| {
                            let chi_r = neo_ccs::utils::tensor_point_parallel::<K>(row_challenges);
                            backend.ajtai_y_eval(cache, &chi_r, n_eff, &witnesses)
                        });
                    if let Some(y_eval) = y_eval {
                        oracle.inject_ajtai_y_eval(y_eval);
                    }
                }
            }
        }
        #[cfg(feature = "perf-timers")]
        let eval_started = std::time::Instant::now();
        let deg = oracle.degree_bound();
        let xs: Vec<K> = (0..=deg).map(|t| K::from(F::from_u64(t as u64))).collect();
        let ys = if backend_row_round {
            let backend = fe_backend.as_deref_mut().expect("fe backend is active");
            let coeffs = backend.round_coeffs();
            xs.iter()
                .map(|&x| crate::sumcheck::poly_eval_k(&coeffs, x))
                .collect()
        } else {
            oracle.evals_at(&xs)
        };
        #[cfg(feature = "perf-timers")]
        {
            let eval_ms = eval_started.elapsed().as_secs_f64() * 1_000.0;
            fe_eval_ms += eval_ms;
            if eval_ms > fe_largest_eval_ms {
                fe_largest_eval_ms = eval_ms;
                fe_largest_eval_round = round_idx;
            }
        }

        #[cfg(feature = "debug-logs")]
        if round_idx == 0 {
            eprintln!("\n[prove] === Round 0 ===");
            eprintln!("[prove] p(0) = {:?}", ys[0]);
            eprintln!("[prove] p(1) = {:?}", ys[1]);
            eprintln!("[prove] p(0) + p(1) = {:?}", ys[0] + ys[1]);
            eprintln!("[prove] running_sum (should equal T) = {:?}", running_sum);
            if ys[0] + ys[1] != running_sum {
                eprintln!("[prove] ERROR: Sumcheck invariant violated!");
                eprintln!("[prove]   This means the witness is invalid or T is computed incorrectly");
            } else {
                eprintln!("[prove] OK: p(0) + p(1) == running_sum");
            }
        }

        if ys[0] + ys[1] != running_sum {
            #[cfg(feature = "debug-logs")]
            {
                eprintln!("\n[prove] SUMCHECK FAILED at round {}", round_idx);
                eprintln!("[prove] p(0)+p(1) = {:?}", ys[0] + ys[1]);
                eprintln!("[prove] running_sum = {:?}", running_sum);
                eprintln!("[prove] difference = {:?}", (ys[0] + ys[1]) - running_sum);
            }
            return Err(PiCcsError::SumcheckError(format!(
                "round {} invariant failed: p(0)+p(1) ≠ running_sum (paper-exact)",
                round_idx
            )));
        }
        // Sumcheck requires coefficients in low→high order (c0, c1, ..., cn) so that
        // poly_eval_k(coeffs, ·) reproduces ys at x=0,1 and the verifier invariant
        // p(0)+p(1) == running_sum holds.
        #[cfg(feature = "perf-timers")]
        let interp_started = std::time::Instant::now();
        let coeffs = crate::sumcheck::interpolate_from_evals(&xs, &ys);
        #[cfg(feature = "perf-timers")]
        {
            fe_interp_ms += interp_started.elapsed().as_secs_f64() * 1_000.0;
        }

        debug_assert_eq!(crate::sumcheck::poly_eval_k(&coeffs, K::ZERO), ys[0]);
        debug_assert_eq!(crate::sumcheck::poly_eval_k(&coeffs, K::ONE), ys[1]);

        let coeff_fields = crate::sumcheck::round_coeff_fields(&coeffs);
        tr.append_fields_raw(&coeff_fields);
        let c = tr.challenge_fields_raw(2);
        let r_i = neo_math::from_complex(c[0], c[1]);
        sumcheck_chals.push(r_i);

        // Evaluate at challenge using poly_eval_k (low→high) for consistency.
        running_sum = crate::sumcheck::poly_eval_k(&coeffs, r_i);

        #[cfg(feature = "perf-timers")]
        let fold_started = std::time::Instant::now();
        if backend_row_round {
            fe_backend
                .as_deref_mut()
                .expect("fe backend is active")
                .fold(r_i);
            oracle.advance_row_round_without_fold(r_i);
        } else {
            oracle.fold(r_i);
        }
        #[cfg(feature = "perf-timers")]
        {
            let fold_ms = fold_started.elapsed().as_secs_f64() * 1_000.0;
            fe_fold_ms += fold_ms;
            if fold_ms > fe_largest_fold_ms {
                fe_largest_fold_ms = fold_ms;
                fe_largest_fold_round = round_idx;
            }
        }
        if let Some(rounds) = sumcheck_rounds.as_mut() {
            rounds.push(coeffs);
        }
    }
    let fe_sumcheck_ms = fe_sumcheck_started.elapsed().as_secs_f64() * 1_000.0;
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "optimized_prove: 4. FE sumcheck        {fe_sumcheck_ms:>9.2}ms ({} rounds) @{}",
        sumcheck_chals.len(),
        perf_epoch_nanos()
    );
    #[cfg(feature = "perf-timers")]
    eprintln!("optimized_prove: 4b. FE eval          {fe_eval_ms:>9.2}ms");
    #[cfg(feature = "perf-timers")]
    eprintln!("optimized_prove: 4c. FE interpolate   {fe_interp_ms:>9.2}ms");
    #[cfg(feature = "perf-timers")]
    eprintln!("optimized_prove: 4d. FE fold          {fe_fold_ms:>9.2}ms");
    #[cfg(feature = "perf-timers")]
    eprintln!("optimized_prove: 4e. FE largest eval  {fe_largest_eval_ms:>9.2}ms (round {fe_largest_eval_round})");
    #[cfg(feature = "perf-timers")]
    eprintln!("optimized_prove: 4f. FE largest fold  {fe_largest_fold_ms:>9.2}ms (round {fe_largest_fold_round})");

    // ---------------------------------------------------------------------
    // NC-only sumcheck (split-NC scaffolding; claimed sum is 0)
    // ---------------------------------------------------------------------
    let nc_sumcheck_started = std::time::Instant::now();
    #[cfg(feature = "perf-timers")]
    let mut nc_col_coeff_ms = 0.0f64;
    #[cfg(feature = "perf-timers")]
    let mut nc_ajtai_coeff_ms = 0.0f64;
    #[cfg(feature = "perf-timers")]
    let mut nc_col_fold_ms = 0.0f64;
    #[cfg(feature = "perf-timers")]
    let mut nc_ajtai_fold_ms = 0.0f64;
    #[cfg(feature = "perf-timers")]
    let mut nc_largest_coeff_ms = 0.0f64;
    #[cfg(feature = "perf-timers")]
    let mut nc_largest_coeff_round = 0usize;
    #[cfg(feature = "perf-timers")]
    let mut nc_largest_fold_ms = 0.0f64;
    #[cfg(feature = "perf-timers")]
    let mut nc_largest_fold_round = 0usize;
    let y_zcol_digits;
    if let Some(challenges) = block_challenges {
        let trace = super::block_lane_replay::run(
            tr,
            s,
            mcs_witnesses,
            me_witnesses,
            challenges,
            block_pending.expect("block mode must be present"),
            mode.captures_host_rounds(),
        )?;
        initial_sum_nc = trace.initial_sum;
        running_sum_nc = trace.final_sum;
        sumcheck_rounds_nc = trace.rounds;
        sumcheck_chals_nc = trace.challenges;
        y_zcol_digits = Some(trace.block_rows);
    } else {
        #[cfg(feature = "perf-timers")]
        let nc_oracle_new_started = std::time::Instant::now();
        // With a device backend, skip the digit-table build — the backend
        // sources them from resident planes (materialized below if it declines).
        let mut oracle_nc = oracle_nc_from_phase.take().unwrap_or_else(|| {
            if nc_backend.is_some() {
                super::oracle::NcOracle::new_with_deferred_digit_tables(
                    s,
                    params,
                    mcs_witnesses,
                    me_witnesses,
                    ch.clone(),
                    dims.ell_d,
                    dims.ell_m,
                    dims.d_sc,
                )
            } else {
                super::oracle::NcOracle::new(
                    s,
                    params,
                    mcs_witnesses,
                    me_witnesses,
                    ch.clone(),
                    dims.ell_d,
                    dims.ell_m,
                    dims.d_sc,
                )
            }
        });
        #[cfg(feature = "perf-timers")]
        let nc_oracle_new_ms = nc_oracle_new_started.elapsed().as_secs_f64() * 1_000.0;
        #[cfg(feature = "perf-timers")]
        eprintln!("optimized_prove: 5a. NC oracle new     {nc_oracle_new_ms:>9.2}ms");

        let phase_nc_trace_applied = first_nc_host_round_from_phase.is_some();
        let nc_backend_active = if phase_nc_trace_applied {
            false
        } else {
            match nc_backend.as_deref_mut() {
                Some(backend) => backend.start(&oracle_nc.col_phase_snapshot()),
                None => false,
            }
        };
        if !nc_backend_active && !phase_nc_trace_applied {
            oracle_nc.materialize_deferred_col_tables();
        }

        let mut first_nc_host_round = first_nc_host_round_from_phase.unwrap_or(0);
        let mut nc_prolog_on_host = false;
        if nc_backend_active && dims.ell_m > 0 {
            let col_rounds = dims.ell_m.min(oracle_nc.num_rounds());
            let request = NcColTraceRequest {
                transcript_state: tr.state(),
                transcript_absorbed: tr.absorbed(),
                rounds: col_rounds,
                initial_sum: initial_sum_nc,
            };
            if let Some(trace) = nc_backend
                .as_deref_mut()
                .expect("nc backend is active")
                .col_round_trace_with_prolog(request)
            {
                first_nc_host_round = apply_nc_backend_trace(NcBackendTraceApply {
                    tr,
                    oracle_nc: &mut oracle_nc,
                    trace,
                    expected_rounds: col_rounds,
                    transcript_mode: backend_transcript_mode,
                    append_prolog: true,
                    initial_sum: initial_sum_nc,
                    sumcheck_rounds_nc: &mut sumcheck_rounds_nc,
                    sumcheck_chals_nc: &mut sumcheck_chals_nc,
                    running_sum_nc: &mut running_sum_nc,
                })?;
            }
        }
        if first_nc_host_round == 0 && !phase_nc_trace_applied {
            append_nc_sumcheck_prolog(tr, initial_sum_nc);
            nc_prolog_on_host = true;
        }

        if nc_prolog_on_host && nc_backend_active && dims.ell_m > 0 {
            let col_rounds = dims.ell_m.min(oracle_nc.num_rounds());
            let transcript_state = tr.state();
            let transcript_absorbed = tr.absorbed();
            if let Some(trace) = nc_backend
                .as_deref_mut()
                .expect("nc backend is active")
                .col_round_trace_from_transcript(transcript_state, transcript_absorbed, col_rounds)
            {
                first_nc_host_round = apply_nc_backend_trace(NcBackendTraceApply {
                    tr,
                    oracle_nc: &mut oracle_nc,
                    trace,
                    expected_rounds: col_rounds,
                    transcript_mode: backend_transcript_mode,
                    append_prolog: false,
                    initial_sum: initial_sum_nc,
                    sumcheck_rounds_nc: &mut sumcheck_rounds_nc,
                    sumcheck_chals_nc: &mut sumcheck_chals_nc,
                    running_sum_nc: &mut running_sum_nc,
                })?;
            }
        }

        for _round_idx in first_nc_host_round..oracle_nc.num_rounds() {
            let backend_col_round = nc_backend_active && oracle_nc.round_idx < dims.ell_m;
            #[cfg(feature = "perf-timers")]
            let is_col_round = oracle_nc.round_idx < dims.ell_m;
            #[cfg(feature = "perf-timers")]
            let coeff_started = std::time::Instant::now();
            let coeffs = if backend_col_round {
                nc_backend
                    .as_deref_mut()
                    .expect("nc backend is active")
                    .round_coeffs()
            } else if let Some(coeffs) = oracle_nc.optimized_col_phase_round_coeffs() {
                coeffs
            } else {
                let deg = oracle_nc.degree_bound();
                let xs: Vec<K> = (0..=deg).map(|t| K::from(F::from_u64(t as u64))).collect();
                let ys = oracle_nc.evals_at(&xs);
                crate::sumcheck::interpolate_from_evals(&xs, &ys)
            };
            #[cfg(feature = "perf-timers")]
            {
                let coeff_ms = coeff_started.elapsed().as_secs_f64() * 1_000.0;
                if is_col_round {
                    nc_col_coeff_ms += coeff_ms;
                } else {
                    nc_ajtai_coeff_ms += coeff_ms;
                }
                if coeff_ms > nc_largest_coeff_ms {
                    nc_largest_coeff_ms = coeff_ms;
                    nc_largest_coeff_round = _round_idx;
                }
            }

            let p0 = coeffs[0];
            let p1 = crate::sumcheck::poly_eval_k_base(&coeffs, F::ONE);
            if p0 + p1 != running_sum_nc {
                return Err(PiCcsError::SumcheckError(format!(
                    "NC sumcheck invariant failed at round {_round_idx}: p(0)+p(1) ≠ running_sum"
                )));
            }

            let coeff_fields = crate::sumcheck::round_coeff_fields(&coeffs);
            tr.append_fields_raw(&coeff_fields);
            let c = tr.challenge_fields_raw(2);
            let r_i = neo_math::from_complex(c[0], c[1]);
            sumcheck_chals_nc.push(r_i);

            running_sum_nc = crate::sumcheck::poly_eval_k(&coeffs, r_i);
            #[cfg(feature = "perf-timers")]
            let fold_started = std::time::Instant::now();
            if backend_col_round {
                let backend = nc_backend.as_deref_mut().expect("nc backend is active");
                backend.fold(r_i);
                oracle_nc.advance_col_round_without_fold(r_i);
                if oracle_nc.round_idx == dims.ell_m {
                    let state = backend.finalized_col_state();
                    oracle_nc.inject_finalized_col_state(state.digit_rows, state.eq_beta_m0);
                }
            } else {
                oracle_nc.fold(r_i);
            }
            #[cfg(feature = "perf-timers")]
            {
                let fold_ms = fold_started.elapsed().as_secs_f64() * 1_000.0;
                if is_col_round {
                    nc_col_fold_ms += fold_ms;
                } else {
                    nc_ajtai_fold_ms += fold_ms;
                }
                if fold_ms > nc_largest_fold_ms {
                    nc_largest_fold_ms = fold_ms;
                    nc_largest_fold_round = _round_idx;
                }
            }
            if let Some(rounds) = sumcheck_rounds_nc.as_mut() {
                rounds.push(coeffs);
            }
        }
        y_zcol_digits = Some(oracle_nc.finalized_y_zcol_digits());
    }
    let nc_sumcheck_ms = nc_sumcheck_started.elapsed().as_secs_f64() * 1_000.0;
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "optimized_prove: 5. NC sumcheck        {nc_sumcheck_ms:>9.2}ms ({} rounds) @{}",
        sumcheck_chals_nc.len(),
        perf_epoch_nanos()
    );
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "optimized_prove: 5b. NC coeff col      {nc_col_coeff_ms:>9.2}ms ({} rounds)",
        dims.ell_m
    );
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "optimized_prove: 5c. NC coeff ajtai    {nc_ajtai_coeff_ms:>9.2}ms ({} rounds)",
        dims.ell_d
    );
    #[cfg(feature = "perf-timers")]
    eprintln!("optimized_prove: 5d. NC fold col       {nc_col_fold_ms:>9.2}ms");
    #[cfg(feature = "perf-timers")]
    eprintln!("optimized_prove: 5e. NC fold ajtai     {nc_ajtai_fold_ms:>9.2}ms");
    #[cfg(feature = "perf-timers")]
    eprintln!("optimized_prove: 5f. NC largest coeff  {nc_largest_coeff_ms:>9.2}ms (round {nc_largest_coeff_round})");
    #[cfg(feature = "perf-timers")]
    eprintln!("optimized_prove: 5g. NC largest fold   {nc_largest_fold_ms:>9.2}ms (round {nc_largest_fold_round})");

    // Build outputs at r′ using the oracle's r′-only precomputation (no dense scan).
    let output_started = std::time::Instant::now();
    let fold_digest = tr.digest32();
    let nc_point_variables = binding.nc_point_variables(dims);
    let (s_col, _alpha_nc) = sumcheck_chals_nc.split_at(nc_point_variables);
    let out_me = if let Some(surfaces) = phase_terminal_surfaces.take() {
        build_me_outputs_from_terminal_surfaces(
            params,
            s,
            mcs_list,
            mcs_witnesses,
            me_inputs,
            &sumcheck_chals[..dims.ell_n],
            s_col,
            surfaces,
            fold_digest,
        )?
    } else {
        oracle.build_me_outputs_from_ajtai_precomp(
            mcs_list,
            me_inputs,
            s_col,
            y_zcol_digits.as_deref(),
            fold_digest,
            log,
        )
    };
    let pi_dec_precompute = capture_pi_dec_precompute
        .then(|| oracle.take_pi_dec_precompute())
        .flatten();
    let output_materialize_ms = output_started.elapsed().as_secs_f64() * 1_000.0;
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "optimized_prove: 6. output             {output_materialize_ms:>9.2}ms @{}",
        perf_epoch_nanos()
    );

    let perf = PiCcsProvePerf {
        bind_ms,
        sample_challenges_ms,
        fe_sumcheck_ms,
        nc_sumcheck_ms,
        output_materialize_ms,
        total_ms: total_started.elapsed().as_secs_f64() * 1_000.0,
    };
    #[cfg(feature = "perf-timers")]
    eprintln!("optimized_prove: TOTAL                {:>9.2}ms", perf.total_ms);

    let row_chals = sumcheck_chals[..dims.ell_n].to_vec();
    let alpha_prime = sumcheck_chals[dims.ell_n..].to_vec();
    let s_col_chals = sumcheck_chals_nc[..nc_point_variables].to_vec();
    let alpha_prime_nc = sumcheck_chals_nc[nc_point_variables..].to_vec();
    let output_shell = PiCcsTerminalOutputShell {
        count: out_me.len(),
        m_in: out_me
            .first()
            .map(|claim| claim.m_in)
            .ok_or_else(|| PiCcsError::InvalidInput("Pi_CCS terminal output shell is empty".into()))?,
        row_chals: row_chals.clone(),
        s_col: s_col_chals.clone(),
        has_y_zcol: !s_col_chals.is_empty(),
        fold_digest,
    };

    let terminal_state = PiCcsReplayTerminalState {
        variant: binding.proof_variant(),
        me_outputs: out_me,
        output_shell,
        sc_initial_sum: initial_sum,
        sc_initial_sum_nc: initial_sum_nc,
        challenges_public: ch,
        row_chals,
        alpha_prime,
        s_col: s_col_chals,
        alpha_prime_nc,
        sumcheck_final: running_sum,
        sumcheck_final_nc: running_sum_nc,
        fold_digest,
        perf,
        pi_dec_precompute,
    };
    let rounds = if mode.captures_host_rounds() {
        let rounds = if phase_summary_export_pending && mode.exports_summary_rounds_immediately() {
            let log = phase_backend
                .as_deref_mut()
                .and_then(PiCcsPhaseBackend::export_pi_ccs_phase_rounds)
                .ok_or_else(|| {
                    PiCcsError::InvalidInput(
                        "Pi_CCS phase backend summarized proof state but did not export proof rounds".into(),
                    )
                })?;
            OptimizedProofRounds {
                sumcheck_rounds: log.fe_coeffs,
                initial_sum,
                sumcheck_rounds_nc: log.nc_coeffs,
                initial_sum_nc,
            }
        } else if phase_summary_export_pending {
            return Ok((terminal_state, None));
        } else if let Some(row_rounds) = fe_row_summary_export_pending {
            return Ok((
                terminal_state,
                Some(DeferredProofRounds::FeRows(DeferredFeRowRounds {
                    row_rounds,
                    sumcheck_tail_rounds: sumcheck_rounds.expect("prove mode must capture FE tail rounds"),
                    initial_sum,
                    sumcheck_rounds_nc: sumcheck_rounds_nc.expect("prove mode must capture NC rounds"),
                    initial_sum_nc,
                })),
            ));
        } else {
            OptimizedProofRounds {
                sumcheck_rounds: sumcheck_rounds.expect("prove mode must capture FE rounds"),
                initial_sum,
                sumcheck_rounds_nc: sumcheck_rounds_nc.expect("prove mode must capture NC rounds"),
                initial_sum_nc,
            }
        };
        Some(DeferredProofRounds::Owned(rounds))
    } else {
        None
    };
    Ok((terminal_state, rounds))
}

pub fn optimized_prove_simple<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    mcs_witnesses: &[CcsWitness<F>],
    log: &L,
) -> Result<(Vec<CeClaim<Cmt, F, K>>, PiCcsProof), PiCcsError> {
    optimized_prove(tr, params, s, mcs_list, mcs_witnesses, &[], &[], log)
}
