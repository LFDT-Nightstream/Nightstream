//! Prover-side lifecycle: `prove` (top-level loop) + `extend` (one step) +
//! `start_proof` (base-case `UncompressedAudit` constructor).
//!
//! No session-wide transcript lives here. Each F' step owns its own per-step
//! transcript inside `paper::f_prime::prove`; the terminal fold owns its own
//! inside `paper::construction2::prove_final_fold`.

use crate::lifecycle::{Error, Preprocessing, Uncompressed, UncompressedAudit};
use crate::paper::construction2::{
    self, FoldProof, NebulaAdvance, NebulaLane, ProofState, SemanticStateAdvance, State, StateCoordinates,
};
use crate::paper::nifs::NifsProverAdapter;
use crate::paper::relations::{CcsClaim, CcsInstance, CeClaim};
use neo_math::{D, F};

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
    prep.validate_verifier_key_binding()?;
    let mut in_flight = start_proof(prep);
    for batch in batches {
        in_flight = extend(prep, in_flight, batch)?;
    }
    Ok(in_flight)
}

pub fn prove_with_nifs_adapter<I>(
    prep: &Preprocessing,
    adapter: &mut dyn NifsProverAdapter,
    batches: I,
) -> Result<UncompressedAudit, Error>
where
    I: IntoIterator<Item = Vec<CcsInstance>>,
{
    prep.validate_verifier_key_binding()?;
    let mut in_flight = start_proof(prep);
    for batch in batches {
        in_flight = extend_with_nifs_adapter(prep, adapter, in_flight, batch)?;
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
    extend_inner(prep, audit, batch, SemanticStateAdvance::Stateless, None)
}

/// Extend a Nebula chain with the step that **opens a segment**: `d_pre`
/// is the prover's claimed per-lane chain digests over the segment's
/// forthcoming lane-commitment leaves, computed by the
/// segment prover's precommit pass). γ is squeezed inside the lane
/// transition; the payload rides `StepProof.nebula_open` so the verifier
/// replays the identical open. Mid-segment continuation steps use plain
/// [`extend`].
pub fn extend_nebula_open(
    prep: &Preprocessing,
    audit: UncompressedAudit,
    batch: Vec<CcsInstance>,
    d_pre: [[F; 4]; 3],
) -> Result<UncompressedAudit, Error> {
    extend_inner(prep, audit, batch, SemanticStateAdvance::Stateless, Some(d_pre))
}

/// Adapter-backed variant of [`extend_nebula_open`]. The Nebula transition
/// remains lifecycle-owned; only the NIFS fold is delegated to `adapter`.
pub fn extend_nebula_open_with_nifs_adapter(
    prep: &Preprocessing,
    adapter: &mut dyn NifsProverAdapter,
    audit: UncompressedAudit,
    batch: Vec<CcsInstance>,
    d_pre: [[F; 4]; 3],
) -> Result<UncompressedAudit, Error> {
    extend_inner_with_adapter(
        prep,
        adapter,
        audit,
        batch,
        SemanticStateAdvance::Stateless,
        Some(d_pre),
    )
}

pub fn extend_with_nifs_adapter(
    prep: &Preprocessing,
    adapter: &mut dyn NifsProverAdapter,
    audit: UncompressedAudit,
    batch: Vec<CcsInstance>,
) -> Result<UncompressedAudit, Error> {
    extend_inner_with_adapter(prep, adapter, audit, batch, SemanticStateAdvance::Stateless, None)
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
    super::validate_semantic_state_digest_canonical("initial_semantic_state_digest", semantic_state_digest_initial)?;
    super::validate_semantic_state_digest_canonical("semantic_state_digest_next", semantic_state_digest_next)?;
    let audit = start_proof_with_semantic_state(prep, semantic_state_digest_initial);
    extend_inner(
        prep,
        audit,
        batch,
        SemanticStateAdvance::Stateful(semantic_state_digest_next),
        None,
    )
}

pub fn prove_one_with_semantic_state_and_nifs_adapter(
    prep: &Preprocessing,
    adapter: &mut dyn NifsProverAdapter,
    batch: Vec<CcsInstance>,
    semantic_state_digest_initial: [u8; 32],
    semantic_state_digest_next: [u8; 32],
) -> Result<UncompressedAudit, Error> {
    super::validate_semantic_state_digest_canonical("initial_semantic_state_digest", semantic_state_digest_initial)?;
    super::validate_semantic_state_digest_canonical("semantic_state_digest_next", semantic_state_digest_next)?;
    let audit = start_proof_with_semantic_state(prep, semantic_state_digest_initial);
    extend_inner_with_adapter(
        prep,
        adapter,
        audit,
        batch,
        SemanticStateAdvance::Stateful(semantic_state_digest_next),
        None,
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
    super::validate_semantic_state_digest_canonical("semantic_state_digest_next", semantic_state_digest_next)?;
    extend_inner(
        prep,
        audit,
        batch,
        SemanticStateAdvance::Stateful(semantic_state_digest_next),
        None,
    )
}

pub fn extend_with_semantic_state_and_nifs_adapter(
    prep: &Preprocessing,
    adapter: &mut dyn NifsProverAdapter,
    audit: UncompressedAudit,
    batch: Vec<CcsInstance>,
    semantic_state_digest_next: [u8; 32],
) -> Result<UncompressedAudit, Error> {
    super::validate_semantic_state_digest_canonical("semantic_state_digest_next", semantic_state_digest_next)?;
    extend_inner_with_adapter(
        prep,
        adapter,
        audit,
        batch,
        SemanticStateAdvance::Stateful(semantic_state_digest_next),
        None,
    )
}

/// Fold the current `latest` and derive the next public coordinates before
/// the recursive relation synthesizes its real next instance.
pub(crate) struct PreparedRecursiveStep {
    audit: UncompressedAudit,
    prepared: crate::paper::f_prime::PreparedFPrimeStep,
    pre: StateCoordinates,
    fresh: Vec<CcsClaim>,
    running: Vec<CeClaim>,
    running_parent_authority: Option<CeClaim>,
}

impl PreparedRecursiveStep {
    pub(crate) fn pre(&self) -> &StateCoordinates {
        &self.pre
    }

    pub(crate) fn post(&self) -> &StateCoordinates {
        self.prepared.coordinates()
    }

    pub(crate) fn fresh(&self) -> &[CcsClaim] {
        &self.fresh
    }

    pub(crate) fn running(&self) -> &[CeClaim] {
        &self.running
    }

    pub(crate) fn running_parent_authority(&self) -> Option<&CeClaim> {
        self.running_parent_authority.as_ref()
    }

    pub(crate) fn nifs_proof(&self) -> Result<crate::paper::nifs::NifsProof, Error> {
        match &self.prepared.proof().fold {
            FoldProof::Recursive(proof) => Ok(proof.clone()),
            FoldProof::NoFold => Err(Error::RecursivePreparationRequiresActiveState),
        }
    }

    pub(crate) fn complete(mut self, instance: CcsInstance) -> Result<UncompressedAudit, Error> {
        let claim = instance.claim.clone();
        let (state, proof) = self.prepared.complete(vec![instance])?;
        self.audit.proof.state = state;
        self.audit.steps.push(proof);
        self.audit.public_batches.push(vec![claim]);
        Ok(self.audit)
    }
}

pub(crate) fn prepare_recursive_step(
    prep: &Preprocessing,
    audit: UncompressedAudit,
    semantic_advance: SemanticStateAdvance,
) -> Result<PreparedRecursiveStep, Error> {
    prepare_recursive_step_inner(prep, None, audit, semantic_advance)
}

pub(crate) fn prepare_recursive_step_with_nifs_adapter(
    prep: &Preprocessing,
    adapter: &mut dyn NifsProverAdapter,
    audit: UncompressedAudit,
    semantic_advance: SemanticStateAdvance,
) -> Result<PreparedRecursiveStep, Error> {
    prepare_recursive_step_inner(prep, Some(adapter), audit, semantic_advance)
}

fn prepare_recursive_step_inner(
    prep: &Preprocessing,
    adapter: Option<&mut dyn NifsProverAdapter>,
    audit: UncompressedAudit,
    semantic_advance: SemanticStateAdvance,
) -> Result<PreparedRecursiveStep, Error> {
    prep.validate_verifier_key_binding()?;
    if audit.proof.final_fold.is_some() {
        return Err(Error::AlreadyFinalized);
    }
    if let SemanticStateAdvance::Stateful(digest) = semantic_advance {
        super::validate_semantic_state_digest_canonical("semantic_state_digest_next", digest)?;
    }
    if prep.params.max_fresh_count() < 1 {
        return Err(Error::BatchTooLarge {
            got: 1,
            max: prep.params.max_fresh_count(),
        });
    }

    let current_state = &audit.proof.state;
    let (fresh, running, running_parent_authority) = match &current_state.proof {
        ProofState::Active { running, latest } => (
            latest.claims(),
            running.claims.clone(),
            running.parent_authority.clone(),
        ),
        ProofState::Initial => return Err(Error::RecursivePreparationRequiresActiveState),
    };
    let pre = StateCoordinates::from(current_state);
    let nebula_advance = delayed_nebula_advance(prep, current_state, None)?;
    let fresh_shape = crate::paper::f_prime::FreshClaimShape {
        d: D,
        kappa: prep.params.kappa() as usize,
        m_in: prep
            .public_input_len
            .expect("authoritative recursive F' preprocessing fixes public input length"),
    };
    let prepared = match adapter {
        Some(adapter) => crate::paper::f_prime::prepare_single_with_adapter_and_semantic_state(
            adapter,
            &prep.params,
            prep.structure(),
            prep.optimized_cache(),
            prep.structure_digest(),
            &prep.log,
            prep.mix_rhos_commits,
            prep.combine_b_pows,
            &prep.vk,
            current_state.clone(),
            fresh_shape,
            semantic_advance,
            prep.nebula().map(|cfg| &cfg.scheme),
            nebula_advance,
        )?,
        None => crate::paper::f_prime::prepare_single_with_semantic_state(
            &prep.params,
            prep.structure(),
            prep.optimized_cache(),
            prep.structure_digest(),
            &prep.log,
            prep.mix_rhos_commits,
            prep.combine_b_pows,
            &prep.vk,
            current_state.clone(),
            fresh_shape,
            semantic_advance,
            prep.nebula().map(|cfg| &cfg.scheme),
            nebula_advance,
        )?,
    };
    Ok(PreparedRecursiveStep {
        audit,
        prepared,
        pre,
        fresh,
        running,
        running_parent_authority,
    })
}

fn extend_inner(
    prep: &Preprocessing,
    audit: UncompressedAudit,
    batch: Vec<CcsInstance>,
    semantic_advance: SemanticStateAdvance,
    nebula_open: Option<[[F; 4]; 3]>,
) -> Result<UncompressedAudit, Error> {
    extend_inner_with_nifs_prover(prep, None, audit, batch, semantic_advance, nebula_open)
}

fn extend_inner_with_adapter(
    prep: &Preprocessing,
    adapter: &mut dyn NifsProverAdapter,
    audit: UncompressedAudit,
    batch: Vec<CcsInstance>,
    semantic_advance: SemanticStateAdvance,
    nebula_open: Option<[[F; 4]; 3]>,
) -> Result<UncompressedAudit, Error> {
    extend_inner_with_nifs_prover(prep, Some(adapter), audit, batch, semantic_advance, nebula_open)
}

fn extend_inner_with_nifs_prover(
    prep: &Preprocessing,
    adapter: Option<&mut dyn NifsProverAdapter>,
    mut audit: UncompressedAudit,
    batch: Vec<CcsInstance>,
    semantic_advance: SemanticStateAdvance,
    nebula_open: Option<[[F; 4]; 3]>,
) -> Result<UncompressedAudit, Error> {
    extend_in_place_inner_with_nifs_prover(prep, adapter, &mut audit, batch, semantic_advance, nebula_open)?;
    Ok(audit)
}

fn extend_in_place_inner_with_nifs_prover(
    prep: &Preprocessing,
    adapter: Option<&mut dyn NifsProverAdapter>,
    audit: &mut UncompressedAudit,
    batch: Vec<CcsInstance>,
    semantic_advance: SemanticStateAdvance,
    nebula_open: Option<[[F; 4]; 3]>,
) -> Result<(), Error> {
    prep.validate_verifier_key_binding()?;
    if audit.proof.final_fold.is_some() {
        return Err(Error::AlreadyFinalized);
    }
    if batch.is_empty() {
        return Err(Error::EmptyBatch);
    }
    if prep.enforces_terminal_induction() && batch.len() != 1 {
        return Err(Error::TerminalInductionArity { got: batch.len() });
    }
    let max_fresh = prep.params.max_fresh_count();
    if batch.len() > max_fresh {
        return Err(Error::BatchTooLarge {
            got: batch.len(),
            max: max_fresh,
        });
    }
    let public_batch: Vec<CcsClaim> = batch.iter().map(|i| i.claim.clone()).collect();
    super::validate_public_input_len(prep, &public_batch)?;
    let current_state = audit.proof.state.clone();
    // The prover runs the same shared Nebula lane transition
    // decode-and-advance the verifiers replay, so a malformed segment
    // fails here at the named transition check instead of at verification.
    let nebula_advance = if prep.enforces_terminal_induction() {
        delayed_nebula_advance(prep, &current_state, nebula_open)?
    } else {
        match (prep.nebula(), &current_state.nebula) {
            (Some(cfg), Some(lane)) => {
                let mut lane_out = lane.clone();
                lane_out.advance_for_batch(
                    cfg,
                    prep.vk.digest(),
                    current_state.z_i,
                    current_state.acc_digest,
                    nebula_open,
                    &public_batch,
                )?;
                Some(NebulaAdvance {
                    lane_out,
                    open: nebula_open,
                })
            }
            (None, None) => {
                if nebula_open.is_some() {
                    return Err(Error::NebulaNotConfigured);
                }
                None
            }
            _ => return Err(Error::NebulaLanePresenceMismatch),
        }
    };
    let (next_state, step_proof) = if let Some(adapter) = adapter {
        construction2::step_with_adapter_and_semantic_state(
            adapter,
            &prep.params,
            prep.structure(),
            prep.optimized_cache(),
            prep.structure_digest(),
            &prep.log,
            prep.mix_rhos_commits,
            prep.combine_b_pows,
            &prep.vk,
            current_state,
            batch,
            semantic_advance,
            prep.nebula().map(|cfg| &cfg.scheme),
            nebula_advance,
        )?
    } else {
        construction2::step_with_semantic_state(
            &prep.params,
            prep.structure(),
            prep.optimized_cache(),
            prep.structure_digest(),
            &prep.log,
            prep.mix_rhos_commits,
            prep.combine_b_pows,
            &prep.vk,
            current_state,
            batch,
            semantic_advance,
            prep.nebula().map(|cfg| &cfg.scheme),
            nebula_advance,
        )?
    };
    audit.proof.state = next_state;
    audit.steps.push(step_proof);
    audit.public_batches.push(public_batch);
    Ok(())
}

fn delayed_nebula_advance(
    prep: &Preprocessing,
    state: &State,
    external_open: Option<[[F; 4]; 3]>,
) -> Result<Option<NebulaAdvance>, Error> {
    if external_open.is_some() {
        return Err(Error::TerminalInductionExternalNebulaOpen);
    }
    match (prep.nebula(), &state.nebula) {
        (Some(cfg), Some(lane)) => {
            let mut lane_out = lane.clone();
            if let crate::paper::construction2::ProofState::Active { latest, .. } = &state.proof {
                let claims = latest.claims();
                if !claims.is_empty() {
                    lane_out.advance_for_delayed_claims(
                        cfg,
                        prep.vk.digest(),
                        state.z_i,
                        state.acc_digest,
                        crate::paper::f_prime::r1cs::F_PRIME_PUBLIC_INPUT_LEN,
                        &claims,
                    )?;
                }
            }
            Ok(Some(NebulaAdvance { lane_out, open: None }))
        }
        (None, None) => Ok(None),
        _ => Err(Error::NebulaLanePresenceMismatch),
    }
}

/// Base-case `UncompressedAudit`: empty steps, empty `public_batches`,
/// base `State`, no terminal fold.
pub(super) fn start_proof(prep: &Preprocessing) -> UncompressedAudit {
    let acc_digest = crate::paper::digest::AccumulatorHandle::empty().digest();
    let semantic_state_digest = match prep.semantic_state_mode() {
        construction2::SemanticStateMode::Stateless => acc_digest,
        construction2::SemanticStateMode::Stateful => prep.initial_semantic_state_digest(),
    };
    start_proof_with_semantic_state(prep, semantic_state_digest)
}

fn start_proof_with_semantic_state(prep: &Preprocessing, semantic_state_digest: [u8; 32]) -> UncompressedAudit {
    let structure = *prep.structure_digest();
    let z_0 = crate::paper::digest::initial_boundary_digest(&structure, prep.public_input_len);
    let public_trace = crate::paper::digest::public_trace_seed_digest(&structure);
    let acc_digest = crate::paper::digest::AccumulatorHandle::empty().digest();
    let mut state = State::base(z_0, public_trace, acc_digest, semantic_state_digest);
    // A Nebula preprocessing carries the lane from the very first state:
    // counters at zero, products at 1_K, memory bound to the plan's D_init.
    if let Some(cfg) = prep.nebula() {
        state.nebula = Some(NebulaLane::base(cfg));
    }
    UncompressedAudit {
        proof: Uncompressed {
            state,
            final_fold: None,
        },
        steps: Vec::new(),
        public_batches: Vec::new(),
    }
}
