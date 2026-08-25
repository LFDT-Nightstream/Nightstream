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
    optimized_prove_with_cache_and_precompute_and_backend_and_perf, optimized_prove_with_cache_and_precompute_and_perf,
    optimized_verify_with_cache_and_perf, OptimizedStructureCache, PaperJointOracleBackend, PiDecProverPrecompute,
};
use thiserror::Error;

use crate::paper::construction2::running::RunningInstanceError;
use crate::paper::construction2::RunningInstance;
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
    #[error(transparent)]
    Running(#[from] RunningInstanceError),
}

/// Π_CCS (§7.3) prove through the selected one-joint transcript.
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
    prove_pi_ccs_parts_inner(tr, pp, s, cache, fresh_claims, fresh_witnesses, running, log, None)
}

#[allow(clippy::too_many_arguments)]
pub fn prove_pi_ccs_parts_with_backend<L>(
    tr: &mut neo_transcript::Poseidon2Transcript,
    pp: &Params,
    s: &Structure,
    cache: &OptimizedStructureCache,
    fresh_claims: &[CcsClaim],
    fresh_witnesses: &[CcsWitness],
    running: &RunningInstance,
    log: &L,
    backend: &mut dyn PaperJointOracleBackend,
) -> Result<(Vec<CeClaim>, nr::PiCcsProof, PiDecProverPrecompute), Error>
where
    L: neo_ccs::traits::SModuleHomomorphism<neo_math::F, neo_ajtai::Commitment> + Sync,
{
    prove_pi_ccs_parts_inner(
        tr,
        pp,
        s,
        cache,
        fresh_claims,
        fresh_witnesses,
        running,
        log,
        Some(backend),
    )
}

#[allow(clippy::too_many_arguments)]
fn prove_pi_ccs_parts_inner<L>(
    tr: &mut neo_transcript::Poseidon2Transcript,
    pp: &Params,
    s: &Structure,
    cache: &OptimizedStructureCache,
    fresh_claims: &[CcsClaim],
    fresh_witnesses: &[CcsWitness],
    running: &RunningInstance,
    log: &L,
    backend: Option<&mut dyn PaperJointOracleBackend>,
) -> Result<(Vec<CeClaim>, nr::PiCcsProof, PiDecProverPrecompute), Error>
where
    L: neo_ccs::traits::SModuleHomomorphism<neo_math::F, neo_ajtai::Commitment> + Sync,
{
    let (outputs, proof, perf, pi_dec_precompute) = match backend {
        Some(backend) => optimized_prove_with_cache_and_precompute_and_backend_and_perf(
            tr,
            pp.inner(),
            s,
            fresh_claims,
            fresh_witnesses,
            &running.claims,
            &running.witnesses,
            log,
            cache,
            backend,
        )?,
        None => optimized_prove_with_cache_and_precompute_and_perf(
            tr,
            pp.inner(),
            s,
            fresh_claims,
            fresh_witnesses,
            &running.claims,
            &running.witnesses,
            log,
            cache,
        )?,
    };
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "[pi-ccs/prove] bind={:.2}ms sample={:.2}ms sumcheck={:.2}ms outputs={:.2}ms total={:.2}ms inputs=fresh:{}+running:{} outputs:{}",
        perf.bind_ms,
        perf.sample_challenges_ms,
        perf.sumcheck_ms,
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

/// Π_CCS (§7.3) verify — mirror of [`prove_pi_ccs`] using the optimized
/// engine's instance-digest-bound verify entry.
///
/// Recomputes `instance_digest` from `(fresh_claims, running_claims)` —
/// the verifier never trusts a prover-supplied digest.
///
/// The selected verifier replays the complete PaddedRowIdentity transcript.
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
    let (ok, perf) = optimized_verify_with_cache_and_perf(
        tr,
        pp.inner(),
        s,
        fresh_claims,
        &running.claims,
        fold_outputs,
        proof,
        cache,
    )?;
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "[pi-ccs/verify] bind={:.2}ms header={:.2}ms me={:.2}ms sample={:.2}ms sumcheck={:.2}ms outputs={:.2}ms terminal={:.2}ms total={:.2}ms",
        perf.bind_ms,
        perf.bind_header_instances_ms,
        perf.bind_me_inputs_ms,
        perf.bind_sample_challenges_ms,
        perf.sumcheck_ms,
        perf.output_checks_ms,
        perf.terminal_ms,
        perf.total_ms,
    );
    #[cfg(not(feature = "perf-timers"))]
    let _ = perf;
    Ok(ok)
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
        "[pi-rlc/verify] rho={:.2}ms X={:.2}ms y={:.2}ms commit={:.2}ms total={:.2}ms inputs={}",
        perf.rho_mats_ms + perf.rho_k_lift_ms,
        perf.x_ms,
        perf.y_ms,
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
    prove_pi_dec_inner(
        pp,
        s,
        cache,
        log,
        parent,
        parent_witness,
        precompute,
        combine_b_pows,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn prove_pi_dec_with_backend<L, MB>(
    pp: &Params,
    s: &Structure,
    cache: &OptimizedStructureCache,
    log: &L,
    parent: &CeClaim,
    parent_witness: &Mat<F>,
    precompute: Option<&PiDecProverPrecompute>,
    combine_b_pows: MB,
    backend: &mut dyn PaperJointOracleBackend,
) -> Result<(Vec<CeClaim>, Vec<Mat<F>>), Error>
where
    L: neo_ccs::traits::SModuleHomomorphism<F, Commitment> + Sync,
    MB: Fn(&[Commitment], u32) -> Commitment,
{
    prove_pi_dec_inner(
        pp,
        s,
        cache,
        log,
        parent,
        parent_witness,
        precompute,
        combine_b_pows,
        Some(backend),
    )
}

#[allow(clippy::too_many_arguments)]
fn prove_pi_dec_inner<L, MB>(
    pp: &Params,
    s: &Structure,
    cache: &OptimizedStructureCache,
    log: &L,
    parent: &CeClaim,
    parent_witness: &Mat<F>,
    precompute: Option<&PiDecProverPrecompute>,
    combine_b_pows: MB,
    backend: Option<&mut dyn PaperJointOracleBackend>,
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
    let t_openings = std::time::Instant::now();
    let precomputed_y_ring = match backend {
        Some(backend) => backend.dec_openings(cache, &z_split, &parent.r, s.m)?,
        None => None,
    };
    #[cfg(feature = "perf-timers")]
    if precomputed_y_ring.is_some() {
        eprintln!(
            "[pi-dec] child openings                  {:>7.2}s",
            t_openings.elapsed().as_secs_f64()
        );
    }

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
        precomputed_y_ring.as_deref(),
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
    precomputed_openings: &[neo_ccs::V1_1Evaluations<K>],
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
        Some(precomputed_openings),
    );
    if children.is_empty() {
        return Err(Error::PiDecFailed);
    }
    if !(ok_y && ok_x && ok_c) {
        return Err(Error::PiDecPublicCheckFailed { ok_y, ok_x, ok_c });
    }
    Ok((children, z_split))
}

/// Π_DEC verify. Computes the canonical public-X split from the parent, then
/// re-derives parent commitments and y-evaluations from the children.
pub fn verify_pi_dec<MB>(pp: &Params, s: &Structure, parent: &CeClaim, children: &[CeClaim], combine_b_pows: MB) -> bool
where
    MB: Fn(&[Commitment], u32) -> Commitment,
{
    nr::verify_dec_public(s, pp.inner(), parent, children, combine_b_pows, ell_d())
}
