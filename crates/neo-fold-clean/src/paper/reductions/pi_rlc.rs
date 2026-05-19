//! Π_RLC — SuperNeo §7.4. Random Linear Combination.
//!
//! Reduction:  CE(b, ℒ)^{K+k}   →   CE(B, ℒ)   where B = b^k
//!
//! Soundness: **weak** wrt φ projecting commitments (Lemma 4, proof in §D.5).
//! Composes with Π_CCS (strong wrt the same φ) via Theorem 6.
//!
//! ## What this file owns
//!
//! - The `prove` and `verify` step-down flows in paper-step order.
//! - The shape contract: K+k input CE claims, K+k witnesses (prover-only).
//!
//! ## What this file does *not* own
//!
//! - The bound check `(K+k)·T·(b−1) < B` — that's in `paper/sampling.rs`.
//! - The actual Σρ_i mix — that's `engine::optimized::prove_pi_rlc`
//!   (prover) and `engine::optimized::verify_pi_rlc` (verifier).
//!
//! ## Why `combined` is on the wire
//!
//! In principle the verifier can recompute `combined = Σ ρ_i · u_i` from
//! public-coin ρ and the K+k Π_CCS outputs. We mirror `neo-fold-prototype`'s
//! contract and have the prover send its parent: the verifier *checks* the
//! recomputation matches before feeding Π_DEC. Why not just recompute and
//! drop the field?
//!
//! Π_DEC's children are committed against the prover's exact `parent`. The
//! parent's CE claim is built by the engine and then patched with
//! `out.c = mix_commits(rhos, &inputs_c)`. Recomputing on the verifier
//! side could produce an equivalent CE claim but not necessarily a
//! bit-identical one (e.g., on the `c` field if the mix closure on the
//! verifier side has a subtle implementation drift). Comparing the
//! prover-supplied parent against the verifier's recomputation is the safer
//! contract; the wire cost is one CE claim per IVC step.

use neo_ccs::Mat;
use neo_math::F;
use thiserror::Error;

use crate::engine::optimized as engine;
use crate::engine::transcript::Transcript;
use crate::paper::params::Params;
use crate::paper::relations::{CeClaim, RlcMixer};
use crate::paper::sampling::check_rlc_bound;

#[derive(Debug, Error)]
pub enum Error {
    #[error("\u{03A0}_RLC: input claims must be \u{2265} 1 (got 0)")]
    Shape,
    #[error("\u{03A0}_RLC: |claims| ({claims}) \u{2260} |witnesses| ({witnesses})")]
    WitnessMismatch { claims: usize, witnesses: usize },
    #[error("\u{03A0}_RLC: verifier rejected the prover's combined CE claim")]
    VerifyRejected,
    #[error(transparent)]
    Sampling(#[from] crate::paper::sampling::SamplingError),
    #[error(transparent)]
    Engine(#[from] engine::Error),
}

/// Output of one Π_RLC step — a single CE claim of norm B plus its
/// combined witness `Z_mix = Σρ_i Z_i`. Witness is prover-only; the
/// verifier reconstructs only the claim.
#[derive(Clone, Debug)]
pub struct Output {
    pub claim: CeClaim,
    pub witness: Mat<F>,
}

/// Wire-format proof: the prover's combined CE claim of norm B.
///
/// The ρ-rotation challenges are not serialized here; prover and verifier
/// both resample them from the Fiat-Shamir transcript at this phase.
#[derive(Clone, Debug)]
pub struct Proof {
    pub combined: CeClaim,
}

// ──────────────────────────────────────────────────────────────────────────
// Prover  (§7.4 step order)
// ──────────────────────────────────────────────────────────────────────────

pub fn prove(
    tr: &mut Transcript,
    pp: &Params,
    s: &crate::paper::relations::Structure,
    mix: RlcMixer,
    claims: &[CeClaim],
    witnesses: &[Mat<F>],
) -> Result<(Output, Proof), Error> {
    validate_input_shape(claims, witnesses)?;
    enforce_rlc_bound(pp, claims.len())?;
    let rhos = engine::sample_rho_n(tr.inner_mut(), pp, claims.len())?;
    let (combined, z_mix) = engine::prove_pi_rlc(pp, s, &rhos, claims, witnesses, |zs, cs| mix(zs, cs))?;
    Ok((
        Output {
            claim: combined.clone(),
            witness: z_mix,
        },
        Proof { combined },
    ))
}

// ──────────────────────────────────────────────────────────────────────────
// Verifier (§7.4)
// ──────────────────────────────────────────────────────────────────────────

/// Verify the prover's combined CE claim against `Σρ_i · u_i` recomputed
/// from `(transcript, claims)`. Returns the verified parent for Π_DEC.
pub fn verify(
    tr: &mut Transcript,
    pp: &Params,
    s: &crate::paper::relations::Structure,
    mix: RlcMixer,
    claims: &[CeClaim],
    proof: &Proof,
) -> Result<CeClaim, Error> {
    if claims.is_empty() {
        return Err(Error::Shape);
    }
    enforce_rlc_bound(pp, claims.len())?;
    let rhos = engine::sample_rho_n(tr.inner_mut(), pp, claims.len())?;
    let ok = engine::verify_pi_rlc(pp, s, &rhos, claims, &proof.combined, |zs, cs| mix(zs, cs))?;
    if !ok {
        return Err(Error::VerifyRejected);
    }
    Ok(proof.combined.clone())
}

// ──────────────────────────────────────────────────────────────────────────
// Step bodies
// ──────────────────────────────────────────────────────────────────────────

fn validate_input_shape(claims: &[CeClaim], witnesses: &[Mat<F>]) -> Result<(), Error> {
    if claims.is_empty() {
        return Err(Error::Shape);
    }
    if claims.len() != witnesses.len() {
        return Err(Error::WitnessMismatch {
            claims: claims.len(),
            witnesses: witnesses.len(),
        });
    }
    Ok(())
}

/// Definition 14 norm bound: `count · T · (b−1) < B`. Fails loudly here
/// so the caller cannot reach the engine with a count that violates it.
fn enforce_rlc_bound(pp: &Params, count: usize) -> Result<(), Error> {
    check_rlc_bound(pp, count, pp.T() as u128).map_err(Into::into)
}
