//! Finalization — flush the trailing `latest` before Spartan compression.
//!
//! Each `extend` records a new `latest` for the *next* step's fold, so the
//! final extend leaves one batch unfolded sitting in `state.proof.latest`.
//! Compression has to fold it via one last NIFS.P call so the running
//! accumulator handed to Spartan covers every batch the user passed.
//!
//! Final folding is **not** an F' step (no chunk counter advance, no F' R1CS
//! contract). It owns its own terminal NIFS transcript here so callers never
//! mix it with the per-step F' transcripts used by `paper::f_prime`.

use neo_ajtai::AjtaiSModule;
use neo_ccs::matrix::Mat;
use neo_reductions::optimized_engine::OptimizedStructureCache;

use crate::engine::transcript::Transcript;
use crate::paper::construction2::latest::LatestInstance;
use crate::paper::construction2::proof_state::ProofState;
use crate::paper::construction2::running::RunningInstance;
use crate::paper::construction2::state::State;
use crate::paper::construction2::step_proof::{FinalFoldProof, TerminalFoldInputs};
use crate::paper::construction2::verifier_key::VerifierKey;
use crate::paper::construction2::{transition, Error, SemanticStateMode};
use crate::paper::digest;
use crate::paper::nifs;
use crate::paper::params::Params;
use crate::paper::relations::{CcsInstance, CcsWitness, DecMixer, LaneScheme, RlcMixer, Structure};

/// Init label for the terminal finalization NIFS transcript. Distinct from
/// the F'-step label so an auditor sees finalization as its own slot in the
/// transcript namespace.
///
/// `pub` so the in-circuit decider mirror (`engine::decider`) can replay
/// the terminal fold under the same label — without this, the in-circuit
/// transcript prefix would diverge from the native one and NIFS.V would
/// reject lifecycle-produced final-fold proofs.
pub const FINAL_FOLD_TRANSCRIPT_LABEL: &[u8] = b"neo.fold.clean/finalization/v1";

fn final_fold_transcript() -> Transcript {
    Transcript::with_label(FINAL_FOLD_TRANSCRIPT_LABEL)
}

/// One final NIFS.P call to fold any trailing `latest` into `running`.
///
/// Returns the post-flush `State` (with `latest = empty`) and the optional
/// flush proof. `Ok((state, None))` means there was nothing to flush —
/// either the state was `Initial` (no extends ever ran) or the trailing
/// latest was already empty.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_final_fold_with_backend(
    nifs_backend: nifs::NifsProverBackend,
    pp: &Params,
    s: &Structure,
    cache: &OptimizedStructureCache,
    structure_digest: &[neo_math::F; 4],
    log: &AjtaiSModule,
    mix_rhos_commits: RlcMixer,
    combine_b_pows: DecMixer,
    vk: &VerifierKey,
    lanes: Option<&LaneScheme>,
    delayed_nebula: Option<&crate::paper::construction2::NebulaConfig>,
    state: State,
    semantic_mode: SemanticStateMode,
) -> Result<(State, Option<FinalFoldProof>), Error> {
    prove_final_fold_with_nifs_prover(
        FinalFoldNifsProver::Backend(nifs_backend),
        pp,
        s,
        cache,
        structure_digest,
        log,
        mix_rhos_commits,
        combine_b_pows,
        vk,
        lanes,
        delayed_nebula,
        state,
        semantic_mode,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_final_fold_with_adapter(
    adapter: &mut dyn nifs::NifsProverAdapter,
    pp: &Params,
    s: &Structure,
    cache: &OptimizedStructureCache,
    structure_digest: &[neo_math::F; 4],
    log: &AjtaiSModule,
    mix_rhos_commits: RlcMixer,
    combine_b_pows: DecMixer,
    vk: &VerifierKey,
    lanes: Option<&LaneScheme>,
    delayed_nebula: Option<&crate::paper::construction2::NebulaConfig>,
    state: State,
    semantic_mode: SemanticStateMode,
) -> Result<(State, Option<FinalFoldProof>), Error> {
    prove_final_fold_with_nifs_prover(
        FinalFoldNifsProver::Adapter(adapter),
        pp,
        s,
        cache,
        structure_digest,
        log,
        mix_rhos_commits,
        combine_b_pows,
        vk,
        lanes,
        delayed_nebula,
        state,
        semantic_mode,
    )
}

enum FinalFoldNifsProver<'a> {
    Backend(nifs::NifsProverBackend),
    Adapter(&'a mut dyn nifs::NifsProverAdapter),
}

#[allow(clippy::too_many_arguments)]
fn prove_final_fold_with_nifs_prover(
    mut nifs_prover: FinalFoldNifsProver<'_>,
    pp: &Params,
    s: &Structure,
    cache: &OptimizedStructureCache,
    structure_digest: &[neo_math::F; 4],
    log: &AjtaiSModule,
    mix_rhos_commits: RlcMixer,
    combine_b_pows: DecMixer,
    vk: &VerifierKey,
    lanes: Option<&LaneScheme>,
    delayed_nebula: Option<&crate::paper::construction2::NebulaConfig>,
    state: State,
    semantic_mode: SemanticStateMode,
) -> Result<(State, Option<FinalFoldProof>), Error> {
    let State {
        chunk_count,
        step_count,
        z_0,
        z_i,
        pc,
        initial_semantic_state_digest,
        semantic_state_digest,
        acc_digest,
        public_trace,
        proof,
        nebula,
    } = state;

    let pre_nebula = nebula.clone();
    let mut terminal_nebula = nebula;
    let (post_running, nifs_with_inputs, post_acc_digest_override) = match proof {
        ProofState::Initial => {
            return Ok((
                State {
                    chunk_count,
                    step_count,
                    z_0,
                    z_i,
                    pc,
                    initial_semantic_state_digest,
                    semantic_state_digest,
                    acc_digest: digest::AccumulatorHandle::empty().digest(),
                    public_trace,
                    proof: ProofState::Initial,
                    nebula: terminal_nebula,
                },
                None,
            ));
        }
        ProofState::Active { running, latest } if latest.instances.is_empty() => {
            (running.into_materialized()?, None, None)
        }
        ProofState::Active {
            running: running_carrier,
            latest,
        } => {
            let running = running_carrier.materialize()?;
            if let Some(cfg) = delayed_nebula {
                let lane = terminal_nebula.as_mut().ok_or(Error::BaseCaseMismatch)?;
                lane.advance_for_delayed_claims(
                    cfg,
                    vk.digest(),
                    z_i,
                    acc_digest,
                    crate::paper::f_prime::r1cs::F_PRIME_PUBLIC_INPUT_LEN,
                    &latest.claims(),
                )?;
            }
            // Snapshot the pre-fold (running, latest) **claims** for the
            // non-replay IVC verifier. Witnesses are stripped here so the
            // proof never carries prover-private data across the trust
            // boundary; see [`crate::paper::construction2::TerminalFoldInputs`].
            let terminal_inputs = TerminalFoldInputs {
                pre_final_running: strip_running_witnesses(&running),
                latest: strip_latest_witnesses(&latest),
                pre_nebula: pre_nebula.clone(),
            };

            let mut tr = final_fold_transcript();
            let (post_running, nifs_proof, post_acc_digest_override) = match &mut nifs_prover {
                FinalFoldNifsProver::Backend(backend) => {
                    let (running, proof) = nifs::prove_with_backend(
                        *backend,
                        &mut tr,
                        pp,
                        s,
                        cache,
                        log,
                        lanes,
                        mix_rhos_commits,
                        combine_b_pows,
                        latest.instances,
                        &running,
                    )?;
                    (running, proof, None)
                }
                FinalFoldNifsProver::Adapter(adapter) => {
                    let output = nifs::prove_terminal_with_adapter_output_from_carrier(
                        *adapter,
                        &mut tr,
                        pp,
                        s,
                        cache,
                        log,
                        lanes,
                        mix_rhos_commits,
                        combine_b_pows,
                        latest.instances,
                        &running_carrier,
                        &running,
                    )?;
                    let (running, proof, post_summary) = output.into_materialized_parts_with_summary()?;
                    let acc_digest = post_summary.and_then(|summary| summary.acc_digest_override());
                    (running, proof, acc_digest)
                }
            };
            (
                post_running,
                Some((nifs_proof, terminal_inputs)),
                post_acc_digest_override,
            )
        }
    };

    // Re-derive acc_digest from the post-flush running's Π_RLC parent.
    let post_acc_digest = post_acc_digest_override.unwrap_or_else(|| {
        if post_running.claims.is_empty() {
            digest::AccumulatorHandle::empty().digest()
        } else {
            let parent = post_running
                .parent_authority
                .as_ref()
                .expect("post-flush running must carry its Pi_RLC parent authority");
            digest::AccumulatorHandle::from_running_parts(&post_running.claims, Some(parent)).digest()
        }
    });

    let state_after = State {
        chunk_count,
        step_count,
        z_0,
        z_i,
        pc,
        initial_semantic_state_digest,
        semantic_state_digest,
        acc_digest: post_acc_digest,
        public_trace,
        nebula: terminal_nebula,
        proof: ProofState::active(post_running, LatestInstance::from_instances(Vec::new())),
    };
    let final_proof = nifs_with_inputs.map(|(nifs, terminal_inputs)| FinalFoldProof {
        x_out: transition::compute_x_out(vk, pp, structure_digest, &state_after, semantic_mode),
        nifs,
        terminal_inputs,
    });
    Ok((state_after, final_proof))
}

/// Return a witness-stripped clone of a `RunningInstance`. The verifier
/// never sees the prover's `W_i` matrices; stripping them at finalization
/// keeps `TerminalFoldInputs.pre_final_running` purely public.
fn strip_running_witnesses(running: &RunningInstance) -> RunningInstance {
    RunningInstance {
        claims: running.claims.clone(),
        // Wire-compatible empty placeholders: zero-shape `Mat`s carry no data,
        // satisfy `RunningInstance::shape_ok()` semantically only if `claims`
        // is empty. The verifier path does not invoke `shape_ok` on this
        // value, so empty `witnesses` is the right snapshot here.
        witnesses: Vec::new(),
        parent_authority: running.parent_authority.clone(),
    }
}

/// Return a witness-stripped clone of a `LatestInstance`. The verifier
/// only needs `(c, x, m_in)` from each CCS instance; the witness `(w, Z)`
/// is prover-private and is replaced with zero-shape placeholders.
fn strip_latest_witnesses(latest: &LatestInstance) -> LatestInstance {
    LatestInstance::from_instances(
        latest
            .instances
            .iter()
            .map(|inst| CcsInstance {
                claim: inst.claim.clone(),
                witness: CcsWitness {
                    w: Vec::new(),
                    Z: Mat::zero(0, 0, neo_math::F::default()),
                },
            })
            .collect(),
    )
}

pub(crate) fn verify_final_fold(
    pp: &Params,
    s: &Structure,
    cache: &OptimizedStructureCache,
    structure_digest: &[neo_math::F; 4],
    mix_rhos_commits: RlcMixer,
    combine_b_pows: DecMixer,
    vk: &VerifierKey,
    delayed_nebula: Option<&crate::paper::construction2::NebulaConfig>,
    state: State,
    proof: Option<&FinalFoldProof>,
    semantic_mode: SemanticStateMode,
) -> Result<State, Error> {
    let State {
        chunk_count,
        step_count,
        z_0,
        z_i,
        pc,
        initial_semantic_state_digest,
        semantic_state_digest,
        acc_digest,
        public_trace,
        proof: prev_proof,
        nebula,
    } = state;

    let mut terminal_nebula = nebula;
    let post_running = match prev_proof {
        ProofState::Initial => {
            if proof.is_some() {
                return Err(Error::UnexpectedFinalFoldProof);
            }
            return Ok(State {
                chunk_count,
                step_count,
                z_0,
                z_i,
                pc,
                initial_semantic_state_digest,
                semantic_state_digest,
                acc_digest: digest::AccumulatorHandle::empty().digest(),
                public_trace,
                nebula: terminal_nebula,
                proof: ProofState::Initial,
            });
        }
        ProofState::Active { running, latest } if latest.instances.is_empty() => {
            if proof.is_some() {
                return Err(Error::UnexpectedFinalFoldProof);
            }
            running.into_materialized()?
        }
        ProofState::Active { running, latest } => {
            let running = running.into_materialized()?;
            let proof = proof.ok_or(Error::MissingFinalFoldProof)?;
            if let Some(cfg) = delayed_nebula {
                let lane = terminal_nebula.as_mut().ok_or(Error::BaseCaseMismatch)?;
                lane.advance_for_delayed_claims(
                    cfg,
                    vk.digest(),
                    z_i,
                    acc_digest,
                    crate::paper::f_prime::r1cs::F_PRIME_PUBLIC_INPUT_LEN,
                    &latest.claims(),
                )?;
            }
            let mut tr = final_fold_transcript();
            nifs::verify(
                &mut tr,
                pp,
                s,
                cache,
                mix_rhos_commits,
                combine_b_pows,
                &latest.claims(),
                &running,
                &proof.nifs,
            )?
        }
    };

    let post_acc_digest = if post_running.claims.is_empty() {
        digest::AccumulatorHandle::empty().digest()
    } else {
        let parent = post_running
            .parent_authority
            .as_ref()
            .expect("post-flush running must carry its Pi_RLC parent authority");
        digest::AccumulatorHandle::from_running_parts(&post_running.claims, Some(parent)).digest()
    };
    let state_after = State {
        chunk_count,
        step_count,
        z_0,
        z_i,
        pc,
        initial_semantic_state_digest,
        semantic_state_digest,
        acc_digest: post_acc_digest,
        public_trace,
        nebula: terminal_nebula,
        proof: ProofState::active(post_running, LatestInstance::from_instances(Vec::new())),
    };

    if let Some(proof) = proof {
        let x_out = transition::compute_x_out(vk, pp, structure_digest, &state_after, semantic_mode);
        if x_out != proof.x_out {
            return Err(Error::XOutMismatch);
        }
    }
    Ok(state_after)
}
