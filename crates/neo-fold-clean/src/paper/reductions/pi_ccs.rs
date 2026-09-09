//! Native Pi_CCS reduction from fresh CCS and carried CE claims to CE outputs.
//!
//! Owns: paper-level shape checks, prover/verifier orchestration, wire-format
//! outputs, and the recomputable output-message digest.
//!
//! Does not own: Q-polynomial construction, SumCheck arithmetic, terminal
//! identities, or in-circuit verification.
//!
//! Emits constraints: no.
//!
//! Authority boundary: output claims are authenticated by the engine verifier;
//! `outputs_digest` is recomputed compression for the Pi_RLC handoff and is never
//! authority by itself.
//!
//! | Obligation | Local owner | Emits constraints? | Authority source |
//! |---|---|---|---|
//! | Proof/output shape | [`Proof`] | no | Fixed fresh and running claim counts |
//! | Incoming running claim | Exact identity-first paper shape | no | Engine Pi_CCS prover |
//! | Prover reduction | [`prove`] | no | Engine Pi_CCS prover |
//! | Verifier reduction | [`verify`] | no | Engine SumCheck and terminal checks |

use thiserror::Error;

use neo_ajtai::AjtaiSModule;
use neo_math::{D, K};
use neo_reductions::optimized_engine::{OptimizedStructureCache, PaperJointOracleBackend, PiDecProverPrecompute};

use crate::engine::optimized as engine;
use crate::engine::paper_exact as reference_engine;
use crate::engine::transcript::Transcript;
use crate::paper::construction2::RunningInstance;
use crate::paper::params::Params;
use crate::paper::relations::{superneo_has_canonical_x_shape, CcsClaim, CcsInstance, CcsWitness, CeClaim, Structure};

/// Engine-level sumcheck transcript, opaque at the paper layer.
pub use neo_reductions::api::PiCcsProof as SumcheckProof;

#[derive(Debug, Error)]
pub enum Error {
    #[error("\u{03A0}_CCS: input shape mismatch ({0})")]
    Shape(&'static str),
    #[error("\u{03A0}_CCS: output adv must equal its input claim's adv (Π_CCS forwards commitments unchanged)")]
    AdvForwarding,
    #[error(transparent)]
    Engine(#[from] engine::Error),
    #[error(transparent)]
    PaperExactEngine(#[from] reference_engine::Error),
}

/// One PiCCS proof and its complete Section 7.3 output message.
#[derive(Clone, Debug, PartialEq)]
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
    prove_from_parts(tr, pp, s, cache, log, &fresh_claims, &fresh_witnesses, running).map(|(proof, _)| proof)
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
) -> Result<(Proof, PiDecProverPrecompute), Error> {
    prove_from_parts_inner(tr, pp, s, cache, log, fresh_claims, fresh_witnesses, running, None)
}

/// Run the canonical PiCCS prover with a protocol-neutral round evaluator.
///
/// This is an accelerator seam. The normal verifier and proof format do not
/// depend on the selected evaluator.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn prove_from_parts_with_backend(
    tr: &mut Transcript,
    pp: &Params,
    s: &Structure,
    cache: &OptimizedStructureCache,
    log: &AjtaiSModule,
    fresh_claims: &[CcsClaim],
    fresh_witnesses: &[CcsWitness],
    running: &RunningInstance,
    backend: &mut dyn PaperJointOracleBackend,
) -> Result<(Proof, PiDecProverPrecompute), Error> {
    prove_from_parts_inner(
        tr,
        pp,
        s,
        cache,
        log,
        fresh_claims,
        fresh_witnesses,
        running,
        Some(backend),
    )
}

#[allow(clippy::too_many_arguments)]
fn prove_from_parts_inner(
    tr: &mut Transcript,
    pp: &Params,
    s: &Structure,
    cache: &OptimizedStructureCache,
    log: &AjtaiSModule,
    fresh_claims: &[CcsClaim],
    fresh_witnesses: &[CcsWitness],
    running: &RunningInstance,
    backend: Option<&mut dyn PaperJointOracleBackend>,
) -> Result<(Proof, PiDecProverPrecompute), Error> {
    validate_input_shape(pp, s, fresh_claims, fresh_witnesses, running)?;
    let (mut outputs, sumcheck, pi_dec_precompute) = match backend {
        Some(backend) => engine::prove_pi_ccs_parts_with_backend(
            tr.inner_mut(),
            pp,
            s,
            cache,
            fresh_claims,
            fresh_witnesses,
            running,
            log,
            backend,
        )?,
        None => engine::prove_pi_ccs_parts(
            tr.inner_mut(),
            pp,
            s,
            cache,
            fresh_claims,
            fresh_witnesses,
            running,
            log,
        )?,
    };
    forward_adv(fresh_claims, &running.claims, &mut outputs)?;
    validate_v1_1_claims(s, &outputs)?;
    Ok((Proof { sumcheck, outputs }, pi_dec_precompute))
}

/// Independent PaperExact PiCCS prover used only by the PaperExact NIFS
/// reference. It retains the paper-layer shape and output checks while its
/// polynomial, transcript, and SumCheck work use the direct engine.
pub(crate) fn prove_from_parts_paper_exact(
    tr: &mut Transcript,
    pp: &Params,
    s: &Structure,
    log: &AjtaiSModule,
    fresh_claims: &[CcsClaim],
    fresh_witnesses: &[CcsWitness],
    running: &RunningInstance,
) -> Result<Proof, Error> {
    validate_input_shape(pp, s, fresh_claims, fresh_witnesses, running)?;
    let (mut outputs, sumcheck) =
        reference_engine::prove_pi_ccs_parts(tr.inner_mut(), pp, s, fresh_claims, fresh_witnesses, running, log)?;
    forward_adv(fresh_claims, &running.claims, &mut outputs)?;
    validate_v1_1_claims(s, &outputs)?;
    Ok(Proof { sumcheck, outputs })
}

/// the auxiliary-commitment flow (Π_CCS side): the reduction changes evaluation claims, not
/// commitments — each output carries its input's `c` unchanged, so it
/// carries its input's `adv` unchanged too. Outputs are ordered
/// [fresh…, running…], mirroring the paper's i ∈ [K+k] indexing. This
/// identity forwarding is load-bearing: it is what connects the deposited
/// (fresh) claims' tuples — bound by the F′ `D_seen` chain — to the tuples
/// Π_RLC mixes and the terminal decider opens.
fn forward_adv(fresh: &[CcsClaim], running: &[CeClaim], outputs: &mut [CeClaim]) -> Result<(), Error> {
    if outputs.len() != fresh.len() + running.len() {
        return Err(Error::Shape("|outputs| \u{2260} K + k in adv forwarding"));
    }
    let inputs = fresh
        .iter()
        .map(|c| &c.adv)
        .chain(running.iter().map(|c| &c.adv));
    for (output, adv) in outputs.iter_mut().zip(inputs) {
        output.adv = adv.clone();
    }
    Ok(())
}

/// Verifier twin of [`forward_adv`]: outputs are prover-supplied, so the
/// identity must be *checked*, not installed.
fn validate_adv_forwarding(fresh: &[CcsClaim], running: &[CeClaim], outputs: &[CeClaim]) -> Result<(), Error> {
    if outputs.len() != fresh.len() + running.len() {
        return Err(Error::Shape("|outputs| \u{2260} K + k in adv forwarding"));
    }
    let inputs = fresh
        .iter()
        .map(|c| &c.adv)
        .chain(running.iter().map(|c| &c.adv));
    for (output, adv) in outputs.iter().zip(inputs) {
        if output.adv != *adv {
            return Err(Error::AdvForwarding);
        }
    }
    Ok(())
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
    validate_verifier_shape(pp, s, fresh_claims, running, &proof.outputs)?;
    validate_adv_forwarding(fresh_claims, &running.claims, &proof.outputs)?;
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

/// Independent verifier twin for [`prove_from_parts_paper_exact`].
pub(crate) fn verify_paper_exact(
    tr: &mut Transcript,
    pp: &Params,
    s: &Structure,
    fresh_claims: &[CcsClaim],
    running: &RunningInstance,
    proof: &Proof,
) -> Result<Vec<CeClaim>, Error> {
    validate_verifier_shape(pp, s, fresh_claims, running, &proof.outputs)?;
    validate_adv_forwarding(fresh_claims, &running.claims, &proof.outputs)?;
    let ok = reference_engine::verify_pi_ccs(
        tr.inner_mut(),
        pp,
        s,
        fresh_claims,
        running,
        &proof.outputs,
        &proof.sumcheck,
    )?;
    if !ok {
        return Err(Error::Shape("PaperExact engine returned false on verify"));
    }
    Ok(proof.outputs.clone())
}

// ──────────────────────────────────────────────────────────────────────────
// Step bodies — short, named, paper-referenced.
// ──────────────────────────────────────────────────────────────────────────

/// Reject CE claims that are not exact whole-ring coefficient embeddings.
fn validate_canonical_x_shape(claims: &[CeClaim], label: &'static str) -> Result<(), Error> {
    for claim in claims {
        if !superneo_has_canonical_x_shape(&claim.X, claim.m_in) {
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
    if !running.prover_shape_is_valid() {
        return Err(Error::Shape("running: |claims| \u{2260} |witnesses|"));
    }
    if !running.is_empty() && running.claims.len() as u32 != pp.k_rho() {
        return Err(Error::Shape("running length does not match params.k_rho()"));
    }
    for (idx, claim) in fresh_claims.iter().enumerate() {
        if claim.m_in > s.m {
            return Err(Error::Shape("fresh m_in exceeds structure.m"));
        }
        if claim.m_in % D != 0 {
            return Err(Error::Shape("fresh m_in must contain whole degree-D ring elements"));
        }
        if claim.x.len() != claim.m_in {
            return Err(Error::Shape("fresh x length does not match m_in"));
        }
        if fresh_witnesses[idx].private_len(claim.m_in, s.m).is_none() {
            return Err(Error::Shape("fresh m_in + witness length must equal structure.m"));
        }
    }
    validate_canonical_x_shape(
        &running.claims,
        "running X must use the canonical coefficient embedding",
    )?;
    validate_v1_1_claims(s, &running.claims)?;
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
    running: &RunningInstance,
    fold_outputs: &[CeClaim],
) -> Result<(), Error> {
    let running_claims = &running.claims;
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
    validate_canonical_x_shape(running_claims, "running X must use the canonical coefficient embedding")?;
    validate_canonical_x_shape(
        fold_outputs,
        "fold output X must use the canonical coefficient embedding",
    )?;
    validate_v1_1_claims(s, running_claims)?;
    validate_v1_1_claims(s, fold_outputs)?;
    Ok(())
}

fn validate_fresh_count_within_rlc_guard(pp: &Params, fresh_len: usize) -> Result<(), Error> {
    if fresh_len > pp.max_fresh_count() {
        return Err(Error::Shape("K (fresh) exceeds params.max_fresh_count()"));
    }
    Ok(())
}

fn validate_v1_1_claims(s: &Structure, claims: &[CeClaim]) -> Result<(), Error> {
    for claim in claims {
        validate_v1_1_claim(s, claim)?;
    }
    Ok(())
}

fn validate_v1_1_claim(s: &Structure, claim: &CeClaim) -> Result<(), Error> {
    let d_pad = D.next_power_of_two();
    let assignment_width = neo_reductions::common::superneo_carrier_width(s.m);
    let ell_n =
        s.n.max(assignment_width)
            .next_power_of_two()
            .max(2)
            .trailing_zeros() as usize;

    if claim.r.len() != ell_n {
        return Err(Error::Shape("CE r length must match the joint row point"));
    }
    if claim.eval_k.len() != d_pad {
        return Err(Error::Shape("CE Eval_K must use the padded ring degree"));
    }
    if claim
        .eval_k
        .iter()
        .skip(D)
        .any(|&lane| lane != K::default())
    {
        return Err(Error::Shape("CE Eval_K padding lanes must be zero"));
    }
    if claim.eval_a.len() != s.t() {
        return Err(Error::Shape("CE Eval_A count must equal the CCS matrix count"));
    }
    for row in &claim.eval_a {
        if row.len() != d_pad {
            return Err(Error::Shape("CE Eval_A rows must use the padded ring degree"));
        }
        if row.iter().skip(D).any(|&lane| lane != K::default()) {
            return Err(Error::Shape("CE Eval_A padding lanes must be zero"));
        }
    }
    Ok(())
}
