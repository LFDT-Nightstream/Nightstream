//! Prover-side lifecycle: `prove` (top-level loop) + `extend` (one step) +
//! `start_proof` (base-case `UncompressedAudit` constructor).
//!
//! No session-wide transcript lives here. Each F' step owns its own per-step
//! transcript inside `paper::f_prime::prove`; the terminal fold owns its own
//! inside `paper::construction2::prove_final_fold`.

use crate::lifecycle::{Error, Preprocessing, Uncompressed, UncompressedAudit};
use crate::paper::construction2::{self, SemanticStateAdvance, State};
use crate::paper::relations::{CcsClaim, CcsInstance};

/// Drive the IVC over a sequence of batches, top-down. Each batch is
/// `Vec<CcsInstance>` — typically produced by
/// [`crate::lifecycle::FoldSchedule::partition`].
///
/// Returns the **pre-finalize** [`UncompressedAudit`]: per-step
/// `StepProof`s + public batches accumulated, terminal fold not yet run
/// (`audit.proof.final_fold == None`, trailing `latest` non-empty).
pub fn prove<I>(prep: &Preprocessing, batches: I) -> Result<UncompressedAudit, Error>
where
    I: IntoIterator<Item = Vec<CcsInstance>>,
{
    let mut in_flight = start_proof(prep);
    for batch in batches {
        in_flight = extend(prep, in_flight, batch)?;
    }
    Ok(in_flight)
}

/// Extend an in-flight proof by one step. The batch is the K instances the
/// next step will fold into running (i.e., what becomes `state.proof.latest`).
///
/// Stateless chains call this directly; stateful frontends (e.g. R1CS-F'
/// with an app-state plan) use [`extend_with_semantic_state`] so the
/// advanced `semantic_state_digest` is bound to actual app-state wires.
pub fn extend(
    prep: &Preprocessing,
    audit: UncompressedAudit,
    batch: Vec<CcsInstance>,
) -> Result<UncompressedAudit, Error> {
    extend_inner(prep, audit, batch, SemanticStateAdvance::Stateless)
}

/// Begin a stateful proof: seed the base state with
/// `semantic_state_digest_initial` (typically `H(initial_app_state)`) and
/// fold one batch.
///
/// Stateful chains require a `Preprocessing` whose verifier-owned
/// `semantic_state_mode == Stateful`. The mode is **structure-derived
/// and not externally settable** — only in-crate frontends whose plan
/// declares `semantic_state_in/out_var_indices` (e.g. R1CS-F') produce
/// such a `Preprocessing`. Calling this against a Stateless
/// `Preprocessing` produces a proof that every verifier rejects with
/// `StatelessSemanticInvariantViolated`.
pub fn prove_one_with_semantic_state(
    prep: &Preprocessing,
    batch: Vec<CcsInstance>,
    semantic_state_digest_initial: [u8; 32],
    semantic_state_digest_next: [u8; 32],
) -> Result<UncompressedAudit, Error> {
    let audit = start_proof_with_semantic_state(prep, semantic_state_digest_initial);
    extend_inner(
        prep,
        audit,
        batch,
        SemanticStateAdvance::Stateful(semantic_state_digest_next),
    )
}

/// Extend an in-flight proof with an app-supplied
/// `semantic_state_digest_next`. The digest MUST equal
/// `H(state_out_vars)` under the same Poseidon2 binding rows that the
/// F' image's CCS structure enforces (see
/// `frontends/f_prime/recursive_plan::semantic_state_preimage_sources`).
/// See [`prove_one_with_semantic_state`] for the structure-derived
/// `SemanticStateMode::Stateful` requirement on `prep`.
pub fn extend_with_semantic_state(
    prep: &Preprocessing,
    audit: UncompressedAudit,
    batch: Vec<CcsInstance>,
    semantic_state_digest_next: [u8; 32],
) -> Result<UncompressedAudit, Error> {
    extend_inner(
        prep,
        audit,
        batch,
        SemanticStateAdvance::Stateful(semantic_state_digest_next),
    )
}

fn extend_inner(
    prep: &Preprocessing,
    mut audit: UncompressedAudit,
    batch: Vec<CcsInstance>,
    semantic_advance: SemanticStateAdvance,
) -> Result<UncompressedAudit, Error> {
    if audit.proof.final_fold.is_some() {
        return Err(Error::AlreadyFinalized);
    }
    let public_batch: Vec<CcsClaim> = batch.iter().map(|i| i.claim.clone()).collect();
    super::validate_public_input_len(prep, &public_batch)?;
    let (next_state, step_proof) = construction2::step_with_semantic_state(
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.structure_digest(),
        &prep.log,
        prep.mix_rhos_commits,
        prep.combine_b_pows,
        &prep.vk,
        audit.proof.state,
        batch,
        semantic_advance,
    )?;
    audit.proof.state = next_state;
    audit.steps.push(step_proof);
    audit.public_batches.push(public_batch);
    Ok(audit)
}

/// Base-case `UncompressedAudit`: empty steps, empty `public_batches`,
/// base `State`, no terminal fold.
pub(super) fn start_proof(prep: &Preprocessing) -> UncompressedAudit {
    let acc_digest = crate::paper::digest::accumulator_digest_from_claims(prep.params.b(), &[]);
    start_proof_with_semantic_state(prep, acc_digest)
}

fn start_proof_with_semantic_state(prep: &Preprocessing, semantic_state_digest: [u8; 32]) -> UncompressedAudit {
    let structure = *prep.structure_digest();
    let z_0 = crate::paper::digest::initial_boundary_digest(&structure, prep.public_input_len);
    let public_trace = crate::paper::digest::public_trace_seed_digest(&structure);
    let acc_digest = crate::paper::digest::accumulator_digest_from_claims(prep.params.b(), &[]);
    UncompressedAudit {
        proof: Uncompressed {
            state: State::base(z_0, public_trace, acc_digest, semantic_state_digest),
            final_fold: None,
        },
        steps: Vec::new(),
        public_batches: Vec::new(),
    }
}
