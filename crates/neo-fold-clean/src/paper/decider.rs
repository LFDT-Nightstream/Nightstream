//! Terminal-compression contract.
//!
//! Owns the *statement* a compact terminal proof must bind and a
//! non-SNARK `validate_witness` preflight that runs the **chain-replay**
//! authority path — walking every per-step NIFS.V plus the terminal fold.
//! This is a superset of `lifecycle::verify_uncompressed` (non-replay IVC,
//! which authenticates only the terminal fold), but it is not the whole proof
//! relation: the future compact proof must reproduce the decider R1CS in
//! `crate::engine::decider`, including terminal CE rows. `prove` / `verify`
//! are `Unsupported` placeholders until that verifier is implemented.
//!
//! ## Authority boundary
//!
//! - **Public**: `PublicImage` — the chain-binding coordinates the
//!   verifier recomputes from preprocessing (vk_fs_digest, counters, z_0,
//!   z_i, pc, acc_digest, public_trace, x_out).
//! - **Witness**: the prover-side material the verifier walks to derive
//!   the same public image — step proofs, public batches, the terminal
//!   fold proof, and the post-finalization state (with its final running
//!   accumulator and witness matrices).
//!
//! `validate_witness` ties the two together before circuit synthesis:
//! replay every step + the final fold, recompute the public image from
//! the resulting verifier state, and assert it matches
//! `statement.public`. It also checks that the final running's witness
//! matrices commit to the claims' commitments, so a prover that supplies
//! a public image disconnected from any witness is rejected before the
//! decider R1CS is even emitted. The remaining terminal CE obligations
//! (`X`, low-norm, `y_ring`, `ct`, and NC sidecars) are circuit rows in
//! `engine::decider` / `paper::decider_ce_relation`; do not replace them
//! with this Rust preflight.

use neo_ajtai::AjtaiSModule;
use neo_ccs::traits::SModuleHomomorphism;
use neo_math::F;
use neo_reductions::common::validate_superneo_witness_mat;
use neo_reductions::optimized_engine::OptimizedStructureCache;
use thiserror::Error;

use crate::paper::construction2::{
    self, EncInst, FinalFoldProof, ProofState, RunningInstance, SemanticStateMode, State, StepProof,
    TerminalFoldInputs, VerifierKey,
};
use crate::paper::digest::{initial_boundary_digest, public_trace_seed_digest, AccumulatorHandle};
use crate::paper::f_prime::nebula_lane_circuit::delayed_nebula_public_suffix_len;
use crate::paper::f_prime::r1cs::{f_prime_public_input_link_matches, FPrimePublicInputLayout};
use crate::paper::params::Params;
use crate::paper::relations::{CcsClaim, DecMixer, RlcMixer, Structure};
use crate::paper::terminal_ce::TerminalCeProof;

#[derive(Debug, Error)]
pub enum Error {
    #[error("decider: terminal compression is not implemented yet")]
    Unsupported,
    #[error("decider: validation walk failed: {0}")]
    WalkFailed(String),
    #[error("decider: public image derived from witness ≠ statement.public")]
    PublicImageMismatch,
    #[error("decider: witness shape mismatch (proof.state must be Active with empty latest after finalization)")]
    WitnessShape,
    #[error("decider: final running claims/witnesses length mismatch")]
    WitnessLengthMismatch,
    #[error("decider: final running claim {index} witness commitment ≠ claim.c")]
    WitnessCommitmentMismatch { index: usize },
    #[error("decider: public_batches / steps length mismatch (got {batches} batches, {steps} steps)")]
    StepsBatchesLengthMismatch { batches: usize, steps: usize },
    #[error("decider: public batch has {got} fresh instances, but this parameter profile supports at most {max}")]
    BatchTooLarge { got: usize, max: usize },
    #[error("decider: terminal latest claim {index} public input does not encode the pre-final state x_out")]
    TerminalLatestPublicInputMismatch { index: usize },
    #[error(
        "decider: compact terminal CE proof verification is not implemented; direct terminal CE rows are required"
    )]
    TerminalCeProofUnsupported,
}

/// Public coordinates the compact terminal proof binds — same fields the verifier
/// recomputes from preprocessing. Paper-named; matches the absorb order
/// of [`paper::construction2::compute_x_out`].
///
/// **Content authority**: under direct-CCS with
/// `f_prime_chunk_public_digest` dropping `claim.x` and `claim.c.data`,
/// `z_i` and `public_trace` are step/shape digests only — they bind step
/// indices and protocol shape, not chunk contents. `acc_digest` is the
/// content-binding public coordinate, independently recomputed by
/// [`validate_witness`] from the NIFS.V chain over `Witness.public_batches`
/// + the final fold.
///
/// [`paper::construction2::compute_x_out`]: crate::paper::construction2
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub struct PublicImage {
    pub vk_fs_digest: [u8; 32],
    pub chunk_count: u64,
    pub step_count: u64,
    pub z_0: [u8; 32],
    pub z_i: [u8; 32],
    pub pc: u64,
    pub initial_semantic_state_digest: [u8; 32],
    pub semantic_state_digest: [u8; 32],
    pub acc_digest: [u8; 32],
    pub public_trace: [u8; 32],
    pub x_out: EncInst,
}

/// Prover-side witness for one compressed proof. Drives
/// [`validate_witness`]'s authority walk; the resulting verifier state's
/// public image must equal `Statement.public`.
#[derive(Clone, Debug)]
pub struct Witness {
    /// Per-step F' proofs, one per `extend` call.
    pub steps: Vec<StepProof>,
    /// Public claims of each step's deposited batch. Length must match `steps`.
    pub public_batches: Vec<Vec<CcsClaim>>,
    /// Final NIFS proof that flushes the trailing latest. The decider
    /// statement always requires this proof: `validate_witness` is the
    /// chain-replay gatekeeper for the compressed verifier, so it must not
    /// accept an already-empty recorded state as authority without a
    /// verifier-driven terminal fold.
    pub final_fold: Option<FinalFoldProof>,
    /// Post-finalization state. Carries the final running accumulator's
    /// claims and witness matrices; `validate_witness` requires
    /// `proof = Active { running, latest: empty }`.
    pub final_state: State,
    /// Future compact terminal-CE proof material.
    ///
    /// Current decider synthesis rejects `Some(_)` and keeps using the direct
    /// terminal CE rows. This field exists so the eventual compact verifier has
    /// an explicit data-flow slot instead of treating terminal-child digests as
    /// authority.
    pub terminal_ce_proof: Option<TerminalCeProof>,
}

/// What the compact terminal proof proves. Bundles the public coordinates and the
/// prover-side witness. The verifier-side `compress::verify` consumes only
/// `public` (the witness stays prover-private).
#[derive(Clone, Debug)]
pub struct Statement {
    pub public: PublicImage,
    pub witness: Witness,
}

/// Non-SNARK preflight for one compressed-proof statement.
///
/// Replays every step (`construction2::verify_step`) and the terminal
/// fold (`construction2::verify_final_fold`) starting from
/// preprocessing-derived base state, derives the resulting
/// [`PublicImage`], and asserts it equals `statement.public`. Also
/// verifies the final running's witness matrices commit to the claims'
/// commitments under `log`.
///
/// This is the preflight for the contract the compact proof must reproduce.
/// The full contract is the decider R1CS emitted by `crate::engine::decider`;
/// it adds the terminal CE closure rows that a Rust preflight cannot stand in
/// for. Building a proof around only this function would be underspecified.
#[allow(clippy::too_many_arguments)]
pub fn validate_witness(
    params: &Params,
    structure: &Structure,
    cache: &OptimizedStructureCache,
    structure_digest_v: &[neo_math::F; 4],
    log: &AjtaiSModule,
    mix_rhos_commits: RlcMixer,
    combine_b_pows: DecMixer,
    vk: &VerifierKey,
    public_input_len: Option<usize>,
    f_prime_recursive_link: bool,
    terminal_induction: bool,
    semantic_mode: SemanticStateMode,
    // Verifier-owned initial app/VM semantic-state seed. Pulled from
    // `prep.initial_semantic_state_digest()` at the lifecycle layer;
    // MUST equal `statement.public.initial_semantic_state_digest` or
    // `validate_witness` returns `Error::PublicImageMismatch`.
    initial_semantic_state_digest_anchor: [u8; 32],
    nebula: Option<&crate::paper::construction2::NebulaConfig>,
    statement: &Statement,
) -> Result<(), Error> {
    // (0) Pin the prover's claimed initial app-state to the verifier's
    //     preprocessing-derived anchor. `vk_fs_digest` already absorbs
    //     `initial_semantic_state_digest_anchor`, so a mismatched
    //     `statement.public.initial_semantic_state_digest` would also
    //     fail the chain-x_out check downstream — but surfacing the
    //     dedicated error here gives the caller a precise diagnostic.
    if statement.public.initial_semantic_state_digest != initial_semantic_state_digest_anchor {
        return Err(Error::PublicImageMismatch);
    }
    let Witness {
        steps,
        public_batches,
        final_fold,
        final_state,
        terminal_ce_proof,
    } = &statement.witness;

    if terminal_ce_proof.is_some() {
        return Err(Error::TerminalCeProofUnsupported);
    }

    if steps.len() != public_batches.len() {
        return Err(Error::StepsBatchesLengthMismatch {
            batches: public_batches.len(),
            steps: steps.len(),
        });
    }
    if final_fold.is_none() {
        return Err(Error::WalkFailed(
            "decider witness must carry a terminal final_fold".into(),
        ));
    }

    // Rebuild verifier state from preprocessing.
    let z_0 = initial_boundary_digest(structure_digest_v, public_input_len);
    let public_trace = public_trace_seed_digest(structure_digest_v);
    let acc_digest = AccumulatorHandle::empty().digest();
    let mut state = State::base(z_0, public_trace, acc_digest, initial_semantic_state_digest_anchor);
    if let Some(cfg) = nebula {
        state.nebula = Some(crate::paper::construction2::NebulaLane::base(cfg));
    }

    // Walk each step through F'.verify. Before every recursive fold, pin
    // the currently pending `latest` claim's public input to the current
    // verifier-derived `state.x_out`. This is the native replay version
    // of HyperNova's recursive-link check `u_i.x == enc_inst(prior_x_out)`;
    // without it, only the final trailing latest would be linked.
    //
    // The mode discriminates whether the stateless invariant
    // (`StepProof.semantic_state_digest == new accumulator digest`) is
    // enforced — for stateful chains the F' image's binding rows
    // authenticate the digest instead.
    for (step_index, (public_batch, step_proof)) in public_batches.iter().zip(steps).enumerate() {
        let max_fresh = params.max_fresh_count();
        if public_batch.len() > max_fresh {
            return Err(Error::BatchTooLarge {
                got: public_batch.len(),
                max: max_fresh,
            });
        }
        check_terminal_latest_link(
            params,
            structure_digest_v,
            vk,
            public_input_len,
            f_prime_recursive_link,
            nebula,
            &state,
            semantic_mode,
        )?;
        // Nebula lane replay (the lane transition): recompute the advanced lane
        // from the deposited claims and the step's segment-open payload —
        // the same shared transition the prover ran. Divergence surfaces
        // as the specific lane-transition check that failed, before x_out.
        let nebula_advance = match (nebula, &state.nebula) {
            (Some(cfg), Some(lane)) => {
                let mut lane_out = lane.clone();
                if terminal_induction {
                    if step_proof.nebula_open.is_some() {
                        return Err(Error::WalkFailed(
                            "folded F' carries Nebula open data in the delayed claim suffix".into(),
                        ));
                    }
                    if let ProofState::Active { latest, .. } = &state.proof {
                        lane_out
                            .advance_for_delayed_claims(
                                cfg,
                                vk.digest(),
                                state.z_i,
                                state.acc_digest,
                                crate::paper::f_prime::r1cs::F_PRIME_PUBLIC_INPUT_LEN,
                                &latest.claims(),
                            )
                            .map_err(|e| Error::WalkFailed(format!("nebula lane: {e}")))?;
                    }
                } else {
                    lane_out
                        .advance_for_batch(
                            cfg,
                            vk.digest(),
                            state.z_i,
                            state.acc_digest,
                            step_proof.nebula_open,
                            public_batch,
                        )
                        .map_err(|e| Error::WalkFailed(format!("nebula lane: {e}")))?;
                }
                Some(crate::paper::construction2::NebulaAdvance {
                    lane_out,
                    open: if terminal_induction {
                        None
                    } else {
                        step_proof.nebula_open
                    },
                })
            }
            (None, None) => None,
            _ => {
                return Err(Error::WalkFailed(
                    "nebula config/lane presence mismatch between preprocessing and chain state".into(),
                ))
            }
        };
        state = construction2::verify_step(
            params,
            structure,
            cache,
            structure_digest_v,
            mix_rhos_commits,
            combine_b_pows,
            vk,
            state,
            public_batch,
            step_proof,
            semantic_mode,
            nebula_advance,
        )
        .map_err(|e| Error::WalkFailed(format!("step {step_index}: {e}")))?;
    }

    check_terminal_latest_link(
        params,
        structure_digest_v,
        vk,
        public_input_len,
        f_prime_recursive_link,
        nebula,
        &state,
        semantic_mode,
    )?;

    let terminal_fold = final_fold
        .as_ref()
        .ok_or_else(|| Error::WalkFailed("decider witness must carry a terminal final_fold".into()))?;
    validate_terminal_fold_snapshot(&state, &terminal_fold.terminal_inputs)?;

    // Flush trailing latest through the terminal fold.
    state = construction2::verify_final_fold(
        params,
        structure,
        cache,
        structure_digest_v,
        mix_rhos_commits,
        combine_b_pows,
        vk,
        terminal_induction.then_some(nebula).flatten(),
        state,
        final_fold.as_ref(),
        semantic_mode,
    )
    .map_err(|e| Error::WalkFailed(format!("final_fold: {e}")))?;

    // Derive the public image from the walked state and compare to the
    // statement's declared public.
    let x_out = construction2::compute_x_out(vk, params, structure_digest_v, &state, semantic_mode);
    let derived = PublicImage {
        vk_fs_digest: vk.digest(),
        chunk_count: state.chunk_count,
        step_count: state.step_count,
        z_0: state.z_0,
        z_i: state.z_i,
        pc: state.pc,
        initial_semantic_state_digest: state.initial_semantic_state_digest,
        semantic_state_digest: state.semantic_state_digest,
        acc_digest: state.acc_digest,
        public_trace: state.public_trace,
        x_out,
    };
    if derived != statement.public {
        return Err(Error::PublicImageMismatch);
    }

    // Bind `statement.witness.final_state` to the walked state. Without
    // this, a prover could supply self-consistent witness matrices for an
    // unrelated running accumulator that happens to share the public image
    // — the witness openings would type-check but they would not open
    // the *walked* commitments. Three checks pin the binding:
    //
    //   (a) Canonical fields equal: chunk/step counters, z_0, z_i, pc,
    //       acc_digest, public_trace.
    //   (b) `final_state.proof` is Active with empty `latest` (same shape
    //       as `verify_final_fold`'s output).
    //   (c) `final_state.proof.running.claims == walked.proof.running.claims`
    //       — only then do the witness openings authenticate the walked
    //       accumulator.
    if final_state.chunk_count != state.chunk_count
        || final_state.step_count != state.step_count
        || final_state.z_0 != state.z_0
        || final_state.z_i != state.z_i
        || final_state.pc != state.pc
        || final_state.initial_semantic_state_digest != state.initial_semantic_state_digest
        || final_state.semantic_state_digest != state.semantic_state_digest
        || final_state.acc_digest != state.acc_digest
        || final_state.public_trace != state.public_trace
    {
        return Err(Error::WitnessShape);
    }
    let walked_running = final_running(&state)?;
    let prover_running = final_running(final_state)?;
    if prover_running.claims != walked_running.claims
        || prover_running.parent_authority != walked_running.parent_authority
    {
        return Err(Error::WitnessShape);
    }

    // Now check that the prover's witness matrices open the walked CE
    // claims' commitments under `log`. This preflight only checks
    // chain/public-image binding plus commitment openings for fast
    // diagnostics. The decider R1CS, not this Rust preflight, owns the
    // full terminal CE closure: X projection, low-norm, y_ring, ct, and
    // NC sidecars (see `paper::decider_ce_relation` / `engine::decider`).
    if prover_running.claims.len() != prover_running.witnesses.len() {
        return Err(Error::WitnessLengthMismatch);
    }
    for (index, (claim, witness)) in prover_running
        .claims
        .iter()
        .zip(&prover_running.witnesses)
        .enumerate()
    {
        validate_superneo_witness_mat(witness, structure.m).map_err(|_| Error::WitnessShape)?;
        if log.commit(witness) != claim.c {
            return Err(Error::WitnessCommitmentMismatch { index });
        }
    }
    Ok(())
}

/// Bind the proof-carried terminal snapshot to the state reconstructed by the
/// audit walk. The snapshot is public input to the compact terminal verifier;
/// it must not select a second running/latest pair or carry private witnesses.
fn validate_terminal_fold_snapshot(state: &State, snapshot: &TerminalFoldInputs) -> Result<(), Error> {
    let ProofState::Active { running, latest } = &state.proof else {
        return Err(Error::WitnessShape);
    };
    let expected_running = running
        .materialize_prover_input()
        .map_err(|error| Error::WalkFailed(format!("terminal running snapshot: {error}")))?
        .claims_only();
    if !snapshot.pre_final_running.witnesses.is_empty()
        || snapshot.pre_final_running.claims != expected_running.claims
        || snapshot.pre_final_running.parent_authority != expected_running.parent_authority
        || snapshot.pre_nebula != state.nebula
        || snapshot.latest.instances.len() != latest.instances.len()
    {
        return Err(Error::WitnessShape);
    }
    for (provided, expected) in snapshot.latest.instances.iter().zip(&latest.instances) {
        if provided.claim.c != expected.claim.c
            || provided.claim.x != expected.claim.x
            || provided.claim.m_in != expected.claim.m_in
            || provided.claim.adv != expected.claim.adv
            || !provided.witness.w.is_empty()
            || provided.witness.Z.rows() != 0
            || provided.witness.Z.cols() != 0
        {
            return Err(Error::WitnessShape);
        }
    }
    Ok(())
}

/// Extract the running accumulator from a post-finalization state: must be
/// `Active { running, latest: empty }`, anything else is a witness-shape
/// error.
fn final_running(state: &State) -> Result<RunningInstance, Error> {
    match &state.proof {
        ProofState::Active { running, latest } if latest.instances.is_empty() => {
            running.materialize().map_err(|_| Error::WitnessShape)
        }
        _ => Err(Error::WitnessShape),
    }
}

fn check_terminal_latest_link(
    params: &Params,
    structure_digest: &[F; 4],
    vk: &VerifierKey,
    public_input_len: Option<usize>,
    f_prime_recursive_link: bool,
    nebula: Option<&crate::paper::construction2::NebulaConfig>,
    state: &State,
    semantic_mode: SemanticStateMode,
) -> Result<(), Error> {
    if !f_prime_recursive_link {
        return Ok(());
    }
    let ProofState::Active { latest, .. } = &state.proof else {
        return Ok(());
    };
    if latest.instances.is_empty() {
        return Ok(());
    }

    let expected = construction2::compute_x_out(vk, params, structure_digest, state, semantic_mode);
    let layout = match nebula {
        None => FPrimePublicInputLayout::plain(),
        Some(config) => FPrimePublicInputLayout::with_suffix(delayed_nebula_public_suffix_len(config.stacks)),
    };
    let expected_public_input_len = public_input_len.unwrap_or(layout.total_len());
    for (index, instance) in latest.instances.iter().enumerate() {
        let claim = &instance.claim;
        if !f_prime_public_input_link_matches(layout, &expected, expected_public_input_len, claim.m_in, &claim.x) {
            return Err(Error::TerminalLatestPublicInputMismatch { index });
        }
    }
    Ok(())
}

/// The compressed proof handed to the verifier.
///
/// Future code will populate this with compact terminal proof bytes (and any
/// auxiliary public-IO fields the decider's R1CS exposes). Today it is a
/// placeholder type so the lifecycle wiring compiles end-to-end.
#[derive(Clone, Debug, Default)]
pub struct Proof;

/// Verifier key digest (32 bytes). Compared by the caller against an expected
/// value, never trusted as authority.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifierKeyDigest(pub [u8; 32]);

/// Run terminal compression on the IVC statement. Placeholder until a compact
/// verifier is wired; the contract `validate_witness` enforces is only the
/// native preflight for the relation the proof must reproduce.
pub fn prove(_statement: &Statement) -> Result<(Proof, VerifierKeyDigest), Error> {
    Err(Error::Unsupported)
}

/// Verify a compact terminal proof against the expected public image.
/// Placeholder until a compact verifier is wired.
pub fn verify(_public: &PublicImage, _vk_digest: &VerifierKeyDigest, _proof: &Proof) -> Result<(), Error> {
    Err(Error::Unsupported)
}
