//! Π_CCS — SuperNeo §7.3.
//!
//! Reduction:  CCS(b, ℒ)^K  ×  CE(b, ℒ)^k   →   CE(b, ℒ)^{K+k}
//!
//! Soundness: **strong** wrt φ projecting commitments (Lemma 3, proof in §D.4).
//! Composes with Π_RLC (weak wrt the same φ) via Theorem 6.
//!
//! ## What this file owns
//!
//! - The `prove` and `verify` step-down flows in paper-step order.
//! - The shape contract: K fresh CCS instances, k carried CE claims.
//! - The wire-format `Proof` bundle: `(sumcheck, outputs)`. The K+k output
//!   claims **must** be in the wire format because the verifier needs them
//!   both to feed `engine::verify_pi_ccs` and to feed the next reduction
//!   (Π_RLC) downstream.
//!
//! ## What this file does *not* own
//!
//! - The Q polynomial, the sumcheck, the terminal identity check. All of
//!   that math lives in `engine::optimized`, which wraps `neo-reductions`.

use thiserror::Error;

use neo_ajtai::AjtaiSModule;
use neo_math::{D, K};
use neo_reductions::optimized_engine::OptimizedStructureCache;

use crate::engine::optimized as engine;
use crate::engine::transcript::Transcript;
use crate::paper::construction2::RunningInstance;
use crate::paper::params::Params;
use crate::paper::relations::{superneo_inactive_x_zero, CcsClaim, CcsInstance, CcsWitness, CeClaim, Structure};

/// Engine-level sumcheck transcript, opaque at the paper layer.
pub use neo_reductions::api::PiCcsProof as SumcheckProof;

#[derive(Debug, Error)]
pub enum Error {
    #[error("\u{03A0}_CCS: input shape mismatch ({0})")]
    Shape(&'static str),
    #[error(transparent)]
    Engine(#[from] engine::Error),
}

/// Wire-format Π_CCS proof: the sumcheck transcript plus the K+k output CE
/// claims at point r'. Both are required to verify and to feed Π_RLC.
#[derive(Clone, Debug)]
pub struct Proof {
    pub sumcheck: SumcheckProof,
    pub outputs: Vec<CeClaim>,
}

// ──────────────────────────────────────────────────────────────────────────
// Prover  (§7.3, paper step order)
// ──────────────────────────────────────────────────────────────────────────

/// Π_CCS prover. Top-down:
///
/// 1. Validate the input shape against `pp` (paper Definition 14).
/// 2. Delegate the sumcheck-driven fold to the engine.
/// 3. Bundle the K+k output claims and the sumcheck transcript as the proof.
pub fn prove(
    tr: &mut Transcript,
    pp: &Params,
    s: &Structure,
    cache: &OptimizedStructureCache,
    log: &AjtaiSModule,
    fresh: Vec<CcsInstance>,
    running: &RunningInstance,
) -> Result<Proof, Error> {
    let (fresh_claims, fresh_witnesses) = split_fresh_instances(fresh);
    prove_from_parts(tr, pp, s, cache, log, &fresh_claims, &fresh_witnesses, running)
}

pub(crate) fn prove_from_parts(
    tr: &mut Transcript,
    pp: &Params,
    s: &Structure,
    cache: &OptimizedStructureCache,
    log: &AjtaiSModule,
    fresh_claims: &[CcsClaim],
    fresh_witnesses: &[CcsWitness],
    running: &RunningInstance,
) -> Result<Proof, Error> {
    validate_input_shape(pp, s, fresh_claims, fresh_witnesses, running)?;
    let (outputs, sumcheck) = engine::prove_pi_ccs_parts(
        tr.inner_mut(),
        pp,
        s,
        cache,
        fresh_claims,
        fresh_witnesses,
        running,
        log,
    )?;
    validate_clean_split_nc_claims(s, &outputs)?;
    Ok(Proof { sumcheck, outputs })
}

// ──────────────────────────────────────────────────────────────────────────
// Verifier  (§7.3 step 4; mirrors `prove`)
// ──────────────────────────────────────────────────────────────────────────

/// Π_CCS verifier. Top-down:
///
/// 1. Validate the input shape (claims-only; verifier never sees witnesses).
/// 2. Delegate the sumcheck and terminal-identity check to the engine,
///    using the K+k output claims carried inside the proof bundle.
/// 3. Return the K+k claims so the next reduction (Π_RLC) can consume them.
///
/// The verifier receives public commitments `c`, not openings `z`. It does
/// not know which setup the prover used internally. It fixes `pp` locally;
/// if this verifier accepts, the proof is treated as a proof of knowledge of
/// openings under that fixed `pp`.
pub fn verify(
    tr: &mut Transcript,
    pp: &Params,
    s: &Structure,
    cache: &OptimizedStructureCache,
    fresh_claims: &[CcsClaim],
    running: &RunningInstance,
    proof: &Proof,
) -> Result<Vec<CeClaim>, Error> {
    validate_verifier_shape(pp, s, fresh_claims, &running.claims, &proof.outputs)?;
    let ok = engine::verify_pi_ccs(
        tr.inner_mut(),
        pp,
        s,
        cache,
        fresh_claims,
        running,
        &proof.outputs,
        &proof.sumcheck,
    )?;
    if !ok {
        return Err(Error::Shape("engine returned false on verify"));
    }
    Ok(proof.outputs.clone())
}

// ──────────────────────────────────────────────────────────────────────────
// Step bodies — short, named, paper-referenced.
// ──────────────────────────────────────────────────────────────────────────

/// Reject CE claims whose `X` has non-zero entries in columns
/// `[ceil(m_in / D), x.cols())`. The circuit-side verifier enforces the
/// same invariant and the v2 `ce_claim_digest` skips inactive columns —
/// so without this guard, a malicious prover could smuggle data into
/// inactive columns where it is not transcript-bound.
fn validate_inactive_x_zero(claims: &[CeClaim], label: &'static str) -> Result<(), Error> {
    for claim in claims {
        if !superneo_inactive_x_zero(&claim.X, claim.m_in) {
            return Err(Error::Shape(label));
        }
    }
    Ok(())
}

/// Step 0 (prover): K fresh ≥ 1, running length equals `pp.k_rho()` after step 1.
fn validate_input_shape(
    pp: &Params,
    s: &Structure,
    fresh_claims: &[CcsClaim],
    fresh_witnesses: &[CcsWitness],
    running: &RunningInstance,
) -> Result<(), Error> {
    if fresh_claims.is_empty() {
        return Err(Error::Shape("K (fresh) must be \u{2265} 1"));
    }
    validate_fresh_count_within_rlc_guard(pp, fresh_claims.len())?;
    if fresh_claims.len() != fresh_witnesses.len() {
        return Err(Error::Shape("|fresh_claims| \u{2260} |fresh_witnesses|"));
    }
    if !running.shape_ok() {
        return Err(Error::Shape("running: |claims| \u{2260} |witnesses|"));
    }
    if !running.is_empty() && running.claims.len() as u32 != pp.k_rho() {
        return Err(Error::Shape("running length does not match params.k_rho()"));
    }
    for (idx, claim) in fresh_claims.iter().enumerate() {
        if claim.m_in > s.m {
            return Err(Error::Shape("fresh m_in exceeds structure.m"));
        }
        if claim.x.len() != claim.m_in {
            return Err(Error::Shape("fresh x length does not match m_in"));
        }
        let z_len = claim
            .m_in
            .checked_add(fresh_witnesses[idx].w.len())
            .ok_or(Error::Shape("fresh m_in + witness length overflow"))?;
        if z_len != s.m {
            return Err(Error::Shape("fresh m_in + witness length must equal structure.m"));
        }
    }
    validate_inactive_x_zero(&running.claims, "running inactive X columns must be zero")?;
    Ok(())
}

fn split_fresh_instances(fresh: Vec<CcsInstance>) -> (Vec<CcsClaim>, Vec<CcsWitness>) {
    let mut claims = Vec::with_capacity(fresh.len());
    let mut witnesses = Vec::with_capacity(fresh.len());
    for instance in fresh {
        claims.push(instance.claim);
        witnesses.push(instance.witness);
    }
    (claims, witnesses)
}

/// Step 0 (verifier): mirror of the prover shape check, on claims only.
fn validate_verifier_shape(
    pp: &Params,
    s: &Structure,
    fresh_claims: &[CcsClaim],
    running_claims: &[CeClaim],
    fold_outputs: &[CeClaim],
) -> Result<(), Error> {
    if fresh_claims.is_empty() {
        return Err(Error::Shape("K (fresh) must be \u{2265} 1"));
    }
    validate_fresh_count_within_rlc_guard(pp, fresh_claims.len())?;
    if !running_claims.is_empty() && running_claims.len() as u32 != pp.k_rho() {
        return Err(Error::Shape("running length does not match params.k_rho()"));
    }
    let expected_outputs = fresh_claims.len() + running_claims.len();
    if fold_outputs.len() != expected_outputs {
        return Err(Error::Shape("|fold_outputs| \u{2260} K + k"));
    }
    validate_inactive_x_zero(running_claims, "running inactive X columns must be zero")?;
    validate_inactive_x_zero(fold_outputs, "fold output inactive X columns must be zero")?;
    validate_clean_split_nc_claims(s, running_claims)?;
    validate_clean_split_nc_claims(s, fold_outputs)?;
    Ok(())
}

fn validate_fresh_count_within_rlc_guard(pp: &Params, fresh_len: usize) -> Result<(), Error> {
    if fresh_len > pp.max_fresh_count() {
        return Err(Error::Shape("K (fresh) exceeds params.max_fresh_count()"));
    }
    Ok(())
}

fn validate_clean_split_nc_claims(s: &Structure, claims: &[CeClaim]) -> Result<(), Error> {
    for claim in claims {
        validate_clean_split_nc_claim(s, claim)?;
    }
    Ok(())
}

fn validate_clean_split_nc_claim(s: &Structure, claim: &CeClaim) -> Result<(), Error> {
    let d_pad = D.next_power_of_two();
    let ell_n = s.n.next_power_of_two().max(2).trailing_zeros() as usize;
    let ell_m = s.m.next_power_of_two().max(2).trailing_zeros() as usize;

    if claim.r.len() != ell_n {
        return Err(Error::Shape("CE r length must match SplitNc row point"));
    }
    if claim.s_col.len() != ell_m {
        return Err(Error::Shape("CE s_col length must match SplitNc column point"));
    }
    if claim.y_ring.len() != s.t() {
        return Err(Error::Shape("CE y_ring length must match structure.t"));
    }
    if claim.ct.len() != s.t() {
        return Err(Error::Shape("CE ct length must match structure.t"));
    }
    if claim.y_zcol.len() != d_pad {
        return Err(Error::Shape("CE y_zcol length must match padded ring degree"));
    }
    if !claim.aux_openings.is_empty() || !claim.c_step_coords.is_empty() || claim.u_offset != 0 || claim.u_len != 0 {
        return Err(Error::Shape("CE claim carries unsupported SplitNc sidecars"));
    }

    for (ct, row) in claim.ct.iter().zip(&claim.y_ring) {
        let Some(&constant_term) = row.first() else {
            return Err(Error::Shape("CE y_ring row must expose a constant term"));
        };
        if *ct != constant_term {
            return Err(Error::Shape("CE ct must equal y_ring constant term"));
        }
        if row.len() != d_pad {
            return Err(Error::Shape("CE y_ring rows must use padded ring degree"));
        }
        if row.iter().skip(D).any(|&lane| lane != K::default()) {
            return Err(Error::Shape("CE y_ring padding lanes must be zero"));
        }
    }
    if claim
        .y_zcol
        .iter()
        .skip(D)
        .any(|&lane| lane != K::default())
    {
        return Err(Error::Shape("CE y_zcol padding lanes must be zero"));
    }

    Ok(())
}
