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
use neo_math::F;
use neo_reductions::api as nr;
use neo_reductions::api::FoldingMode;
use neo_reductions::common::{sample_rot_rhos_n_typed, split_b_matrix_k, RotRho};
use neo_reductions::optimized_engine::{
    optimized_prove_with_cache_and_instance_digest_and_perf, optimized_verify_with_cache_and_instance_digest_and_perf,
    OptimizedStructureCache,
};
use thiserror::Error;

use crate::paper::construction2::RunningInstance;
use crate::paper::digest::pi_ccs_instance_digest;
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
}

/// Π_CCS (§7.3) prove — wrapper over the optimized engine's
/// instance-digest-bound entry.
///
/// **Why instance-digest binding**: without it, `nr::prove` and `nr::verify`
/// can produce different transcript states for non-trivial polynomial CCS
/// shapes (anything beyond the empty-`f` toy fixture) and Π_RLC's verifier
/// rejects. neo-fold-next solved this by binding a public-instance digest
/// into both prover and verifier transcripts; we mirror that contract.
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
    fresh: Vec<CcsInstance>,
    running: &RunningInstance,
    log: &L,
) -> Result<(Vec<CeClaim>, nr::PiCcsProof), Error>
where
    L: neo_ccs::traits::SModuleHomomorphism<neo_math::F, neo_ajtai::Commitment> + Sync,
{
    // Validate inputs and compute the instance digest BEFORE moving `fresh`
    // into engine arrays — both sides hash the same public claims.
    let fresh_claims_for_digest: Vec<CcsClaim> = fresh.iter().map(|i| i.claim.clone()).collect();
    let instance_digest = pi_ccs_instance_digest(&fresh_claims_for_digest, &running.claims);

    let (mcs, mcs_witnesses) = split_fresh_for_engine(fresh);
    let cache = OptimizedStructureCache::build(s)?;
    let (outputs, proof, _perf) = optimized_prove_with_cache_and_instance_digest_and_perf(
        tr,
        pp.inner(),
        s,
        &mcs,
        &mcs_witnesses,
        &running.claims,
        &running.witnesses,
        instance_digest,
        log,
        &cache,
    )?;
    Ok((outputs, proof))
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
    fresh_claims: &[CcsClaim],
    running_claims: &[CeClaim],
    fold_outputs: &[CeClaim],
    proof: &nr::PiCcsProof,
) -> Result<bool, Error> {
    use neo_transcript::Transcript as _;
    let instance_digest = pi_ccs_instance_digest(fresh_claims, running_claims);
    let cache = OptimizedStructureCache::build(s)?;
    let (ok, _perf) = optimized_verify_with_cache_and_instance_digest_and_perf(
        tr,
        pp.inner(),
        s,
        fresh_claims,
        running_claims,
        fold_outputs,
        proof,
        &cache,
        instance_digest,
    )?;
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

/// Π_RLC verify. Re-derives `expected = Σρ_i · me_inputs[i]` and checks
/// against the prover's claimed combined CE claim. Mirrors `neo-fold-next`'s
/// contract: the prover's parent is on the wire and the verifier asserts
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
    let (ok, _perf) = nr::rlc_public_matches_verified_inputs_with_perf(
        s,
        pp.inner(),
        rhos,
        me_inputs,
        expected,
        mix_rhos_commits,
        ell_d(),
    )?;
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
    log: &L,
    parent: &CeClaim,
    parent_witness: &Mat<F>,
    combine_b_pows: MB,
) -> Result<(Vec<CeClaim>, Vec<Mat<F>>), Error>
where
    L: neo_ccs::traits::SModuleHomomorphism<F, Commitment> + Sync,
    MB: Fn(&[Commitment], u32) -> Commitment,
{
    let k = pp.k_rho() as usize;
    let z_split = split_b_matrix_k(parent_witness, k, pp.b())?;
    let child_commitments: Vec<Commitment> = z_split.iter().map(|z| log.commit(z)).collect();
    let (children, ok_y, ok_x, ok_c) = nr::dec_children_with_commit(
        FoldingMode::Optimized,
        s,
        pp.inner(),
        parent,
        &z_split,
        ell_d(),
        &child_commitments,
        combine_b_pows,
    );
    if children.is_empty() {
        return Err(Error::PiDecFailed);
    }
    // The engine returns these flags so the prover can fail fast instead of
    // emitting unverifiable children. Mirrors `neo-fold-next`'s contract.
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
