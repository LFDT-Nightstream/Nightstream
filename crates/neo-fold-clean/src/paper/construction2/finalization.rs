//! Finalization — flush the trailing `latest` before Spartan compression.
//!
//! Each `extend` records a new `latest` for the *next* step's fold, so the
//! final extend leaves one batch unfolded sitting in `state.proof.latest`.
//! Compression has to fold it via one last NIFS.P call so the running
//! accumulator handed to Spartan covers every batch the user passed.

use neo_ajtai::AjtaiSModule;

use crate::engine::transcript::Transcript;
use crate::paper::construction2::latest::LatestInstance;
use crate::paper::construction2::proof_state::ProofState;
use crate::paper::construction2::running::RunningInstance;
use crate::paper::construction2::state::State;
use crate::paper::construction2::step_proof::FinalFoldProof;
use crate::paper::construction2::verifier_key::VerifierKey;
use crate::paper::construction2::{transition, Error};
use crate::paper::digest;
use crate::paper::nifs;
use crate::paper::params::Params;
use crate::paper::relations::{DecMixer, RlcMixer, Structure};

/// One final NIFS.P call to fold any trailing `latest` into `running`.
///
/// Returns the post-flush `State` (with `latest = empty`) and the optional
/// flush proof. `Ok((state, None))` means there was nothing to flush —
/// either the state was `Initial` (no extends ever ran) or the trailing
/// latest was already empty.
pub(crate) fn prove_final_fold(
    tr: &mut Transcript,
    pp: &Params,
    s: &Structure,
    log: &AjtaiSModule,
    mix_rhos_commits: RlcMixer,
    combine_b_pows: DecMixer,
    vk: &VerifierKey,
    state: State,
) -> Result<(State, Option<FinalFoldProof>), Error> {
    let State {
        chunk_count,
        step_count,
        z_0,
        z_i,
        pc,
        acc_digest: _, // recomputed from post-flush running below
        public_trace,
        proof,
    } = state;

    let (post_running, nifs_proof) = match proof {
        ProofState::Initial => {
            return Ok((
                State {
                    chunk_count,
                    step_count,
                    z_0,
                    z_i,
                    pc,
                    acc_digest: digest::accumulator_digest_from_claims(pp.b(), &[]),
                    public_trace,
                    proof: ProofState::Initial,
                },
                None,
            ));
        }
        ProofState::Active { running, latest } if latest.instances.is_empty() => (running, None),
        ProofState::Active { running, latest } => {
            let (post_running, nifs_proof) = nifs::prove(
                tr,
                pp,
                s,
                log,
                mix_rhos_commits,
                combine_b_pows,
                latest.instances,
                &running,
            )?;
            (post_running, Some(nifs_proof))
        }
    };

    // Re-derive acc_digest from the post-flush running.
    let post_acc_digest = digest::accumulator_digest_from_claims(pp.b(), &post_running.claims);

    let state_after = State {
        chunk_count,
        step_count,
        z_0,
        z_i,
        pc,
        acc_digest: post_acc_digest,
        public_trace,
        proof: ProofState::Active {
            running: post_running,
            latest: LatestInstance::from_instances(Vec::new()),
        },
    };
    let final_proof = nifs_proof.map(|nifs| FinalFoldProof {
        x_out: transition::compute_x_out(vk, pp, s, &state_after),
        nifs,
    });
    Ok((state_after, final_proof))
}

pub(crate) fn verify_final_fold(
    tr: &mut Transcript,
    pp: &Params,
    s: &Structure,
    mix_rhos_commits: RlcMixer,
    combine_b_pows: DecMixer,
    vk: &VerifierKey,
    state: State,
    proof: Option<&FinalFoldProof>,
) -> Result<State, Error> {
    let State {
        chunk_count,
        step_count,
        z_0,
        z_i,
        pc,
        acc_digest: _,
        public_trace,
        proof: prev_proof,
    } = state;

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
                acc_digest: digest::accumulator_digest_from_claims(pp.b(), &[]),
                public_trace,
                proof: ProofState::Initial,
            });
        }
        ProofState::Active { running, latest } if latest.instances.is_empty() => {
            if proof.is_some() {
                return Err(Error::UnexpectedFinalFoldProof);
            }
            running
        }
        ProofState::Active { running, latest } => {
            let proof = proof.ok_or(Error::MissingFinalFoldProof)?;
            let next_running_claims = nifs::verify(
                tr,
                pp,
                s,
                mix_rhos_commits,
                combine_b_pows,
                &latest.claims(),
                &running.claims,
                &proof.nifs,
            )?;
            RunningInstance {
                claims: next_running_claims,
                witnesses: Vec::new(),
            }
        }
    };

    let post_acc_digest = digest::accumulator_digest_from_claims(pp.b(), &post_running.claims);
    let state_after = State {
        chunk_count,
        step_count,
        z_0,
        z_i,
        pc,
        acc_digest: post_acc_digest,
        public_trace,
        proof: ProofState::Active {
            running: post_running,
            latest: LatestInstance::from_instances(Vec::new()),
        },
    };

    if let Some(proof) = proof {
        let x_out = transition::compute_x_out(vk, pp, s, &state_after);
        if x_out != proof.x_out {
            return Err(Error::XOutMismatch);
        }
    }
    Ok(state_after)
}
