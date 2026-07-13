//! The optimized engine seam — paper layer ↔ `neo-reductions`.
//!
//! Owns: thin wrappers that present `neo-reductions`'s API under paper-named
//! entry points. **No protocol logic here.** If a wrapper does anything
//! beyond split arrays and forward arguments, it's in the wrong place.
//!
//! Does not own: any of the math. Soundness lives in `neo-reductions` and is
//! audited there. This file's job is to keep the paper-layer ↔ engine seam
//! mechanically obvious.

use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_math::{F, K};
use neo_reductions::api as nr;
use neo_reductions::api::FoldingMode;
use neo_reductions::common::{sample_rot_rhos_n_typed, split_b_matrix_k_with_nonzero_flags, RotRho};
use neo_reductions::optimized_engine::{
    optimized_defer_prove_with_device_backends_and_transcript_mode,
    optimized_defer_prove_with_phase_backend_and_transcript_mode,
    optimized_prove_with_cache_and_instance_digest_and_me_input_handle_and_perf,
    optimized_prove_with_phase_backend_and_transcript_mode,
    optimized_verify_with_cache_and_instance_digest_and_me_input_handle_and_perf, BackendTranscriptMode,
    FeSumcheckBackend, NcSumcheckBackend, OptimizedStructureCache, PiCcsDeferredProof, PiCcsPhaseBackend,
    PiDecProverPrecompute,
};
use thiserror::Error;

use crate::paper::construction2::RunningInstance;
use crate::paper::digest::{
    pi_ccs_instance_digest_from_parent_digest, pi_ccs_instance_digest_parent_authority, AccumulatorHandle,
};
use crate::paper::params::Params;
use crate::paper::relations::{CcsClaim, CcsInstance, CcsWitness, CeClaim, Structure};

/// `ell_d = log2(next_power_of_two(D))`, the padded Ajtai-side ring-degree
/// exponent. Const for our preset; computed here so callers don't repeat the
/// formula.
fn ell_d() -> usize {
    neo_math::D.next_power_of_two().trailing_zeros() as usize
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("engine.optimized: {0}")]
    Reductions(#[from] neo_reductions::error::PiCcsError),
    #[error("engine.optimized: \u{03A0}_DEC engine returned an empty child set")]
    PiDecFailed,
    #[error("engine.optimized: \u{03A0}_DEC public checks failed at prove time (y={ok_y}, X={ok_x}, c={ok_c})")]
    PiDecPublicCheckFailed { ok_y: bool, ok_x: bool, ok_c: bool },
    #[error("engine.optimized: running accumulator is missing its \u{03A0}_RLC parent authority")]
    MissingParentAuthority,
    #[error("engine.optimized: empty running accumulator unexpectedly carries a parent authority")]
    UnexpectedParentAuthority,
}

/// Π_CCS (§7.3) prove — wrapper over the optimized engine's
/// instance-digest-bound entry.
///
/// **Why instance-digest binding**: without it, `nr::prove` and `nr::verify`
/// can produce different transcript states for non-trivial polynomial CCS
/// shapes (anything beyond the empty-`f` toy fixture) and Π_RLC's verifier
/// rejects. Binding a public-instance digest into both prover and verifier
/// transcripts keeps them in lockstep.
///
/// **Soundness boundary**: the digest is *recomputed by both sides* from
/// `(fresh_claims, running_claims)` via [`pi_ccs_instance_digest`]. It is
/// never carried prover→verifier on the wire — that would let the prover
/// pick a digest that hides their input. Both sides hash the same
/// authoritative public data and compare transcript states implicitly.
pub fn prove_pi_ccs<L>(
    tr: &mut neo_transcript::Poseidon2Transcript,
    pp: &Params,
    s: &Structure,
    cache: &OptimizedStructureCache,
    fresh: Vec<CcsInstance>,
    running: &RunningInstance,
    log: &L,
) -> Result<(Vec<CeClaim>, nr::PiCcsProof, PiDecProverPrecompute), Error>
where
    L: neo_ccs::traits::SModuleHomomorphism<neo_math::F, neo_ajtai::Commitment> + Sync,
{
    let (mcs, mcs_witnesses) = split_fresh_for_engine(fresh);
    prove_pi_ccs_parts(tr, pp, s, cache, &mcs, &mcs_witnesses, running, log)
}

pub fn prove_pi_ccs_parts<L>(
    tr: &mut neo_transcript::Poseidon2Transcript,
    pp: &Params,
    s: &Structure,
    cache: &OptimizedStructureCache,
    fresh_claims: &[CcsClaim],
    fresh_witnesses: &[CcsWitness],
    running: &RunningInstance,
    log: &L,
) -> Result<(Vec<CeClaim>, nr::PiCcsProof, PiDecProverPrecompute), Error>
where
    L: neo_ccs::traits::SModuleHomomorphism<neo_math::F, neo_ajtai::Commitment> + Sync,
{
    let parent_authority = running_parent_authority(running)?;
    let instance_digest = pi_ccs_instance_digest_parent_authority(fresh_claims, running.claims.len(), parent_authority);
    let me_handle = running_parent_accumulator_handle(running)?;
    let (outputs, proof, perf, pi_dec_precompute) =
        optimized_prove_with_cache_and_instance_digest_and_me_input_handle_and_perf(
            tr,
            pp.inner(),
            s,
            fresh_claims,
            fresh_witnesses,
            &running.claims,
            &running.witnesses,
            instance_digest,
            me_handle,
            log,
            cache,
        )?;
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "[pi-ccs/prove] bind={:.2}ms sample={:.2}ms fe={:.2}ms nc={:.2}ms outputs={:.2}ms total={:.2}ms inputs=fresh:{}+running:{} outputs:{}",
        perf.bind_ms,
        perf.sample_challenges_ms,
        perf.fe_sumcheck_ms,
        perf.nc_sumcheck_ms,
        perf.output_materialize_ms,
        perf.total_ms,
        fresh_claims.len(),
        running.claims.len(),
        outputs.len(),
    );
    #[cfg(not(feature = "perf-timers"))]
    let _ = perf;
    Ok((outputs, proof, pi_dec_precompute))
}

/// [`prove_pi_ccs_parts`] with optional device sumcheck backends threaded
/// through to the engine. `(None, None)` is exactly the CPU path.
#[allow(clippy::too_many_arguments)]
pub fn prove_pi_ccs_parts_with_backends<L>(
    tr: &mut neo_transcript::Poseidon2Transcript,
    pp: &Params,
    s: &Structure,
    cache: &OptimizedStructureCache,
    fresh_claims: &[CcsClaim],
    fresh_witnesses: &[CcsWitness],
    running: &RunningInstance,
    log: &L,
    fe_backend: Option<&mut dyn FeSumcheckBackend>,
    nc_backend: Option<&mut dyn NcSumcheckBackend>,
) -> Result<(Vec<CeClaim>, nr::PiCcsProof), Error>
where
    L: neo_ccs::traits::SModuleHomomorphism<neo_math::F, neo_ajtai::Commitment> + Sync,
{
    prove_pi_ccs_parts_with_backends_and_transcript_mode(
        tr,
        pp,
        s,
        cache,
        fresh_claims,
        fresh_witnesses,
        running,
        log,
        fe_backend,
        nc_backend,
        BackendTranscriptMode::Replay,
        None,
        None,
    )
}

/// [`prove_pi_ccs_parts_with_backends`] with explicit control over whether
/// device transcript segments are replayed into the host transcript online.
#[allow(clippy::too_many_arguments)]
pub fn prove_pi_ccs_parts_with_backends_and_transcript_mode<L>(
    tr: &mut neo_transcript::Poseidon2Transcript,
    pp: &Params,
    s: &Structure,
    cache: &OptimizedStructureCache,
    fresh_claims: &[CcsClaim],
    fresh_witnesses: &[CcsWitness],
    running: &RunningInstance,
    log: &L,
    fe_backend: Option<&mut dyn FeSumcheckBackend>,
    nc_backend: Option<&mut dyn NcSumcheckBackend>,
    transcript_mode: BackendTranscriptMode,
    running_parent_digest: Option<[F; 4]>,
    running_accumulator_handle: Option<[F; 4]>,
) -> Result<(Vec<CeClaim>, nr::PiCcsProof), Error>
where
    L: neo_ccs::traits::SModuleHomomorphism<neo_math::F, neo_ajtai::Commitment> + Sync,
{
    prove_pi_ccs_parts_with_phase_backend_and_transcript_mode(
        tr,
        pp,
        s,
        cache,
        fresh_claims,
        fresh_witnesses,
        running,
        log,
        None,
        fe_backend,
        nc_backend,
        transcript_mode,
        running_parent_digest,
        running_accumulator_handle,
    )
}

/// Whole-phase-capable Π_CCS wrapper. The paper layer still owns shape and
/// digest binding; `phase_backend`, when provided, owns only device scheduling
/// for the FE+NC transcript chain.
#[allow(clippy::too_many_arguments)]
pub fn prove_pi_ccs_parts_with_phase_backend_and_transcript_mode<L>(
    tr: &mut neo_transcript::Poseidon2Transcript,
    pp: &Params,
    s: &Structure,
    cache: &OptimizedStructureCache,
    fresh_claims: &[CcsClaim],
    fresh_witnesses: &[CcsWitness],
    running: &RunningInstance,
    log: &L,
    phase_backend: Option<&mut dyn PiCcsPhaseBackend>,
    fe_backend: Option<&mut dyn FeSumcheckBackend>,
    nc_backend: Option<&mut dyn NcSumcheckBackend>,
    transcript_mode: BackendTranscriptMode,
    running_parent_digest: Option<[F; 4]>,
    running_accumulator_handle: Option<[F; 4]>,
) -> Result<(Vec<CeClaim>, nr::PiCcsProof), Error>
where
    L: neo_ccs::traits::SModuleHomomorphism<neo_math::F, neo_ajtai::Commitment> + Sync,
{
    // Validate inputs and compute the instance digest BEFORE moving `fresh`
    // into engine arrays — both sides hash the same public claims.
    let instance_digest = prover_instance_digest(fresh_claims, running, running_parent_digest)?;
    // Accumulator-handle ME-input binding: bind the same Π_RLC parent
    // authority as the public-instance digest. The Π_DEC children remain the
    // algebraic running inputs, but they do not steer this Fiat-Shamir absorb.
    let me_handle = match running_accumulator_handle {
        Some(handle) => handle,
        None => running_parent_accumulator_handle(running)?,
    };

    let (outputs, proof, _perf) = optimized_prove_with_phase_backend_and_transcript_mode(
        tr,
        pp.inner(),
        s,
        fresh_claims,
        fresh_witnesses,
        &running.claims,
        &running.witnesses,
        instance_digest,
        me_handle,
        log,
        cache,
        phase_backend,
        fe_backend,
        nc_backend,
        transcript_mode,
    )?;
    Ok((outputs, proof))
}

/// Pi_CCS terminal-state first path.
///
/// This keeps `neo-reductions` as the protocol owner while allowing a CUDA
/// phase backend to hold proof-round logs until the caller actually exports
/// proof bytes.
#[allow(clippy::too_many_arguments)]
pub fn defer_pi_ccs_parts_with_phase_backend_and_transcript_mode<L>(
    tr: &mut neo_transcript::Poseidon2Transcript,
    pp: &Params,
    s: &Structure,
    cache: &OptimizedStructureCache,
    fresh_claims: &[CcsClaim],
    fresh_witnesses: &[CcsWitness],
    running: &RunningInstance,
    log: &L,
    phase_backend: &mut dyn PiCcsPhaseBackend,
    transcript_mode: BackendTranscriptMode,
    running_parent_digest: Option<[F; 4]>,
    running_accumulator_handle: Option<[F; 4]>,
) -> Result<PiCcsDeferredProof, Error>
where
    L: neo_ccs::traits::SModuleHomomorphism<neo_math::F, neo_ajtai::Commitment> + Sync,
{
    let instance_digest = prover_instance_digest(fresh_claims, running, running_parent_digest)?;
    let me_handle = match running_accumulator_handle {
        Some(handle) => handle,
        None => running_parent_accumulator_handle(running)?,
    };

    Ok(optimized_defer_prove_with_phase_backend_and_transcript_mode(
        tr,
        pp.inner(),
        s,
        fresh_claims,
        fresh_witnesses,
        &running.claims,
        &running.witnesses,
        instance_digest,
        me_handle,
        log,
        cache,
        phase_backend,
        transcript_mode,
    )?)
}

/// Pi_CCS terminal-state first path for the default device row/NC backends.
///
/// This is the row-trace execution-grain companion to
/// [`defer_pi_ccs_parts_with_phase_backend_and_transcript_mode`]: FE row
/// proof logs remain backend-owned, while the protocol terminal state is
/// available for Π_RLC immediately.
#[allow(clippy::too_many_arguments)]
pub fn defer_pi_ccs_parts_with_device_backends_and_transcript_mode<L>(
    tr: &mut neo_transcript::Poseidon2Transcript,
    pp: &Params,
    s: &Structure,
    cache: &OptimizedStructureCache,
    fresh_claims: &[CcsClaim],
    fresh_witnesses: &[CcsWitness],
    running: &RunningInstance,
    log: &L,
    fe_backend: &mut dyn FeSumcheckBackend,
    nc_backend: Option<&mut dyn NcSumcheckBackend>,
    transcript_mode: BackendTranscriptMode,
    running_parent_digest: Option<[F; 4]>,
    running_accumulator_handle: Option<[F; 4]>,
) -> Result<PiCcsDeferredProof, Error>
where
    L: neo_ccs::traits::SModuleHomomorphism<neo_math::F, neo_ajtai::Commitment> + Sync,
{
    let instance_digest = prover_instance_digest(fresh_claims, running, running_parent_digest)?;
    let me_handle = match running_accumulator_handle {
        Some(handle) => handle,
        None => running_parent_accumulator_handle(running)?,
    };

    Ok(optimized_defer_prove_with_device_backends_and_transcript_mode(
        tr,
        pp.inner(),
        s,
        fresh_claims,
        fresh_witnesses,
        &running.claims,
        &running.witnesses,
        instance_digest,
        me_handle,
        log,
        cache,
        fe_backend,
        nc_backend,
        transcript_mode,
    )?)
}

fn prover_instance_digest(
    fresh_claims: &[CcsClaim],
    running: &RunningInstance,
    running_parent_digest: Option<[F; 4]>,
) -> Result<[F; 4], Error> {
    let parent_authority = running_parent_authority(running)?;
    Ok(match running_parent_digest {
        Some(digest) => pi_ccs_instance_digest_from_parent_digest(fresh_claims, running.claims.len(), Some(digest)),
        None => pi_ccs_instance_digest_parent_authority(fresh_claims, running.claims.len(), parent_authority),
    })
}

/// Π_CCS (§7.3) verify — mirror of [`prove_pi_ccs`] using the optimized
/// engine's instance-digest-bound verify entry.
///
/// Recomputes `instance_digest` from `(fresh_claims, running_claims)` —
/// the verifier never trusts a prover-supplied digest.
///
/// **Transcript symmetry**: the optimized engine's prove path internally
/// squeezes the transcript to produce its `header_digest` (the same value
/// that flows out as `proof.header_digest`). The verify path does not
/// squeeze internally — its caller must catch up by calling
/// `tr.digest32()` so the post-Π_CCS transcript states match on both
/// sides. We do that here AND check the squeezed digest against the
/// prover's recorded `proof.header_digest` (so a tampered header gets
/// rejected before Π_RLC samples its ρ_i).
pub fn verify_pi_ccs(
    tr: &mut neo_transcript::Poseidon2Transcript,
    pp: &Params,
    s: &Structure,
    cache: &OptimizedStructureCache,
    fresh_claims: &[CcsClaim],
    running: &RunningInstance,
    fold_outputs: &[CeClaim],
    proof: &nr::PiCcsProof,
) -> Result<bool, Error> {
    use neo_transcript::Transcript as _;
    let parent_authority = running_parent_authority(running)?;
    let instance_digest = pi_ccs_instance_digest_parent_authority(fresh_claims, running.claims.len(), parent_authority);
    // Same parent-authority handle the prover bound.
    let me_handle = running_parent_accumulator_handle(running)?;
    let (ok, perf) = optimized_verify_with_cache_and_instance_digest_and_me_input_handle_and_perf(
        tr,
        pp.inner(),
        s,
        fresh_claims,
        &running.claims,
        fold_outputs,
        proof,
        cache,
        instance_digest,
        me_handle,
    )?;
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "[pi-ccs/verify] bind={:.2}ms header={:.2}ms me={:.2}ms sample={:.2}ms fe={:.2}ms nc={:.2}ms outputs={:.2}ms terminal={:.2}ms total={:.2}ms",
        perf.bind_ms,
        perf.bind_header_instances_ms,
        perf.bind_me_inputs_ms,
        perf.bind_sample_challenges_ms,
        perf.fe_sumcheck_ms,
        perf.nc_sumcheck_ms,
        perf.output_checks_ms,
        perf.terminal_ms,
        perf.total_ms,
    );
    #[cfg(not(feature = "perf-timers"))]
    let _ = perf;
    if !ok {
        return Ok(false);
    }
    // Catch-up squeeze: bring the verifier's transcript to the same state
    // the prover's transcript reaches at the end of Π_CCS prove. The
    // squeezed digest must match `proof.header_digest`, otherwise the
    // prover lied about the transcript and we reject before any Π_RLC work.
    let observed = tr.digest32();
    if proof.header_digest.as_slice() != observed {
        return Ok(false);
    }
    for output in fold_outputs {
        if output.fold_digest != observed {
            return Ok(false);
        }
    }
    let _ = FoldingMode::Optimized; // keep the explicit folding-mode dependency visible
    Ok(true)
}

// ──────────────────────────────────────────────────────────────────────────
// Marshaling helpers — ONLY array splits, no math.
// ──────────────────────────────────────────────────────────────────────────

/// Split paper-layer (claim, witness) pairs into the engine's parallel arrays
/// by *moving* — no cloning. Consumes `fresh`.
fn split_fresh_for_engine(fresh: Vec<CcsInstance>) -> (Vec<CcsClaim>, Vec<CcsWitness>) {
    let mut claims = Vec::with_capacity(fresh.len());
    let mut witnesses = Vec::with_capacity(fresh.len());
    for instance in fresh {
        claims.push(instance.claim);
        witnesses.push(instance.witness);
    }
    (claims, witnesses)
}

// `carry.witnesses` is a `Vec<WitnessMat>` already; there is no separate
// `carry_witnesses(...)` helper because the field accessor is the helper.

fn running_parent_authority(running: &RunningInstance) -> Result<Option<&CeClaim>, Error> {
    if running.claims.is_empty() {
        if running.parent_authority.is_some() {
            return Err(Error::UnexpectedParentAuthority);
        }
        Ok(None)
    } else {
        running
            .parent_authority
            .as_ref()
            .map(Some)
            .ok_or(Error::MissingParentAuthority)
    }
}

fn running_parent_accumulator_handle(running: &RunningInstance) -> Result<[F; 4], Error> {
    let handle = match running_parent_authority(running)? {
        Some(parent) => AccumulatorHandle::from_running_parts(&running.claims, Some(parent)),
        None => AccumulatorHandle::empty(),
    };
    Ok(handle.digest_fields())
}

// ──────────────────────────────────────────────────────────────────────────
// Π_RLC (§7.4) — wrappers around `neo_reductions::api::rlc_with_commit` and
// `rlc_public_matches_verified_inputs_with_perf`.
// ──────────────────────────────────────────────────────────────────────────

/// Sample N rotation-matrix challenges from the transcript with the
/// `count·T·(b−1) < B` bound check baked in.
pub fn sample_rho_n(
    tr: &mut neo_transcript::Poseidon2Transcript,
    pp: &Params,
    count: usize,
) -> Result<Vec<RotRho>, Error> {
    sample_rot_rhos_n_typed(tr, pp.inner(), &pp.ring(), count).map_err(Into::into)
}

/// Π_RLC prove. Combines K+k CE claims into one CE claim of norm B,
/// using the caller-supplied commitment mixer.
pub fn prove_pi_rlc<MR>(
    pp: &Params,
    s: &Structure,
    rhos: &[RotRho],
    me_inputs: &[CeClaim],
    witnesses: &[Mat<F>],
    mix_rhos_commits: MR,
) -> Result<(CeClaim, Mat<F>), Error>
where
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment,
{
    nr::rlc_with_commit(
        FoldingMode::Optimized,
        s,
        pp.inner(),
        rhos,
        me_inputs,
        witnesses,
        ell_d(),
        mix_rhos_commits,
    )
    .map_err(Into::into)
}

pub fn prove_pi_rlc_refs<MR>(
    pp: &Params,
    s: &Structure,
    rhos: &[RotRho],
    me_inputs: &[CeClaim],
    witnesses: &[&Mat<F>],
    mix_rhos_commits: MR,
) -> Result<(CeClaim, Mat<F>), Error>
where
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment,
{
    nr::rlc_with_commit_refs(
        FoldingMode::Optimized,
        s,
        pp.inner(),
        rhos,
        me_inputs,
        witnesses,
        ell_d(),
        mix_rhos_commits,
    )
    .map_err(Into::into)
}

pub fn prove_pi_rlc_refs_with_witness_mixer<MR, MW>(
    pp: &Params,
    s: &Structure,
    rhos: &[RotRho],
    me_inputs: &[CeClaim],
    witnesses: &[&Mat<F>],
    mix_rhos_commits: MR,
    mix_witnesses: MW,
) -> Result<(CeClaim, Mat<F>), Error>
where
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment,
    MW: Fn(&[Mat<F>], &[&Mat<F>]) -> Mat<F>,
{
    nr::rlc_with_commit_refs_and_witness_mix(
        FoldingMode::Optimized,
        s,
        pp.inner(),
        rhos,
        me_inputs,
        witnesses,
        ell_d(),
        mix_rhos_commits,
        mix_witnesses,
    )
    .map_err(Into::into)
}

pub fn prove_pi_rlc_refs_with_resident_witness<MR, MW, Resident>(
    pp: &Params,
    s: &Structure,
    rhos: &[RotRho],
    me_inputs: &[CeClaim],
    witnesses: &[&Mat<F>],
    mix_rhos_commits: MR,
    mix_witnesses: MW,
) -> Result<(CeClaim, Resident), Error>
where
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment,
    MW: Fn(&[Mat<F>], &[&Mat<F>]) -> Resident,
{
    nr::rlc_with_commit_refs_and_resident_witness(
        FoldingMode::Optimized,
        s,
        pp.inner(),
        rhos,
        me_inputs,
        witnesses,
        ell_d(),
        mix_rhos_commits,
        mix_witnesses,
    )
    .map_err(Into::into)
}

/// Π_RLC verify. Re-derives `expected = Σρ_i · me_inputs[i]` and checks
/// against the prover's claimed combined CE claim. The prover's parent is on
/// the wire and the verifier asserts
/// equality before feeding Π_DEC. The bit-identical match matters because
/// Π_DEC's children were committed against the prover's exact parent.
pub fn verify_pi_rlc<MR>(
    pp: &Params,
    s: &Structure,
    rhos: &[RotRho],
    me_inputs: &[CeClaim],
    expected: &CeClaim,
    mix_rhos_commits: MR,
) -> Result<bool, Error>
where
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment,
{
    let (ok, perf) = nr::rlc_public_matches_verified_inputs_with_perf(
        s,
        pp.inner(),
        rhos,
        me_inputs,
        expected,
        mix_rhos_commits,
        ell_d(),
    )?;
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "[pi-rlc/verify] rho={:.2}ms X={:.2}ms y={:.2}ms y_zcol={:.2}ms aux={:.2}ms commit={:.2}ms total={:.2}ms inputs={}",
        perf.rho_mats_ms + perf.rho_k_lift_ms,
        perf.x_ms,
        perf.y_ms,
        perf.y_zcol_ms,
        perf.aux_ms,
        perf.commitment_ms,
        perf.total_ms,
        me_inputs.len(),
    );
    #[cfg(not(feature = "perf-timers"))]
    let _ = perf;
    Ok(ok)
}

// ──────────────────────────────────────────────────────────────────────────
// Π_DEC (§7.5) — wrappers around `neo_reductions::api::dec_children_with_commit`
// and `verify_dec_public`. Splitting the witness via `split_b_matrix_k` is
// part of the prover's job here.
// ──────────────────────────────────────────────────────────────────────────

/// Π_DEC prove. Splits the parent witness via `split_b`, commits each child
/// via the user's Ajtai homomorphism, then runs the engine to produce the k
/// children CE claims.
pub fn prove_pi_dec<L, MB>(
    pp: &Params,
    s: &Structure,
    cache: &OptimizedStructureCache,
    log: &L,
    parent: &CeClaim,
    parent_witness: &Mat<F>,
    precompute: Option<&PiDecProverPrecompute>,
    combine_b_pows: MB,
) -> Result<(Vec<CeClaim>, Vec<Mat<F>>), Error>
where
    L: neo_ccs::traits::SModuleHomomorphism<F, Commitment> + Sync,
    MB: Fn(&[Commitment], u32) -> Commitment,
{
    if let Some(precompute) = precompute {
        assert_eq!(
            precompute.row_chals, parent.r,
            "Π_DEC prover precompute must belong to the parent claim's row point"
        );
    }
    let k = pp.k_rho() as usize;
    #[cfg(feature = "perf-timers")]
    let t_split = std::time::Instant::now();
    let (z_split, digit_nonzero) = split_b_matrix_k_with_nonzero_flags(parent_witness, k, pp.b())?;
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "[pi-dec] split_b                         {:>7.2}s",
        t_split.elapsed().as_secs_f64()
    );

    #[cfg(feature = "perf-timers")]
    let t_commit = std::time::Instant::now();
    let nonzero_refs: Vec<&Mat<F>> = z_split
        .iter()
        .zip(digit_nonzero.iter())
        .filter_map(|(z, &nonzero)| nonzero.then_some(z))
        .collect();
    let nonzero_commitments = log.commit_many(&nonzero_refs);
    let mut nonzero_iter = nonzero_commitments.into_iter();
    let child_commitments: Vec<Commitment> = digit_nonzero
        .iter()
        .map(|&nonzero| {
            if nonzero {
                nonzero_iter
                    .next()
                    .expect("Π_DEC: nonzero commitment count must match nonzero digit planes")
            } else {
                Commitment::zeros(parent.c.d, parent.c.kappa)
            }
        })
        .collect();
    debug_assert!(
        nonzero_iter.next().is_none(),
        "Π_DEC: unused nonzero commitments after digit-plane assignment"
    );
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "[pi-dec] child commitments               {:>7.2}s (nonzero {}/{k})",
        t_commit.elapsed().as_secs_f64(),
        nonzero_refs.len()
    );

    #[cfg(feature = "perf-timers")]
    let t_children = std::time::Instant::now();
    let (children, ok_y, ok_x, ok_c) = nr::dec_children_with_commit_superneo_cached_from_trusted_split_digits(
        FoldingMode::Optimized,
        s,
        pp.inner(),
        parent,
        &z_split,
        &digit_nonzero,
        ell_d(),
        &child_commitments,
        combine_b_pows,
        cache.superneo(),
        None,
        None,
    );
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "[pi-dec] dec_children_with_commit        {:>7.2}s",
        t_children.elapsed().as_secs_f64()
    );
    if children.is_empty() {
        return Err(Error::PiDecFailed);
    }
    // The engine returns these flags so the prover can fail fast instead of
    // emitting unverifiable children.
    if !(ok_y && ok_x && ok_c) {
        return Err(Error::PiDecPublicCheckFailed { ok_y, ok_x, ok_c });
    }
    Ok((children, z_split))
}

/// Π_DEC claim construction from accelerator-produced, host-validated digit
/// planes and their canonical Ajtai commitments.
#[allow(clippy::too_many_arguments)]
pub fn prove_pi_dec_from_split<MB>(
    pp: &Params,
    s: &Structure,
    cache: &OptimizedStructureCache,
    parent: &CeClaim,
    z_split: Vec<Mat<F>>,
    digit_nonzero: Vec<bool>,
    child_commitments: Vec<Commitment>,
    precomputed_y_ring: &[Vec<[K; neo_math::D]>],
    precompute: Option<&PiDecProverPrecompute>,
    combine_b_pows: MB,
) -> Result<(Vec<CeClaim>, Vec<Mat<F>>), Error>
where
    MB: Fn(&[Commitment], u32) -> Commitment,
{
    if let Some(precompute) = precompute {
        assert_eq!(
            precompute.row_chals, parent.r,
            "Pi_DEC prover precompute must belong to the parent claim's row point"
        );
    }
    let (children, ok_y, ok_x, ok_c) = nr::dec_children_with_commit_superneo_cached_from_trusted_split_digits(
        FoldingMode::Optimized,
        s,
        pp.inner(),
        parent,
        &z_split,
        &digit_nonzero,
        ell_d(),
        &child_commitments,
        combine_b_pows,
        cache.superneo(),
        None,
        Some(precomputed_y_ring),
    );
    if children.is_empty() {
        return Err(Error::PiDecFailed);
    }
    if !(ok_y && ok_x && ok_c) {
        return Err(Error::PiDecPublicCheckFailed { ok_y, ok_x, ok_c });
    }
    Ok((children, z_split))
}

/// Π_DEC verify. Re-derives parent commitments and y-evaluations from the
/// children using `combine_b_pows` and checks they match.
pub fn verify_pi_dec<MB>(pp: &Params, s: &Structure, parent: &CeClaim, children: &[CeClaim], combine_b_pows: MB) -> bool
where
    MB: Fn(&[Commitment], u32) -> Commitment,
{
    nr::verify_dec_public(s, pp.inner(), parent, children, combine_b_pows, ell_d())
}
