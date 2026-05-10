//! Verifier-side lifecycle: `verify_uncompressed`.
//!
//! `verify` (the compressed variant) lives in `compress.rs` next to
//! `compress` because they share the decider statement-builder helpers.
//!
//! Contract:
//! - Authority comes from running the reduction verifiers:
//!   `F' → NIFS.V → Π_CCS.V → Π_RLC.V → Π_DEC.V`.
//! - Digests in `State` are derived chain context for `x_out` / decider
//!   public images. They are not accepted as evidence.
//! - After verification, the verifier compares the actual final accumulator
//!   claims produced by those verifier calls, not just a digest of them.

use neo_ccs::traits::SModuleHomomorphism;

use crate::engine::transcript::Transcript;
use crate::lifecycle::{Error, Preprocessing, Uncompressed};
use crate::paper::construction2::{self, ProofState, RunningInstance, State};
use crate::paper::digest::{
    accumulator_digest_from_claims, initial_boundary_digest, public_trace_seed_digest, structure_digest,
};

/// Verify an uncompressed proof by running the public verifier path.
///
/// IVC proofs are valid at every step; `compress` is just Spartan
/// compression for non-interactivity and size. This function checks the
/// same property without compressing, useful for:
///
/// - prover-side self-checks before paying for the final Spartan prove,
/// - external verification when compression isn't required,
/// - debugging.
pub fn verify_uncompressed(prep: &Preprocessing, proof: &Uncompressed) -> Result<(), Error> {
    check_proof_shape(proof)?;

    // Rebuild verifier state from preprocessing. Nothing from the prover's
    // recorded `proof.state` is used as input to the verifier path.
    let mut transcript = Transcript::session();
    let structure = structure_digest(&prep.structure);
    let z_0 = initial_boundary_digest(&structure, prep.public_input_len);
    let public_trace = public_trace_seed_digest(&structure);
    let acc_digest = accumulator_digest_from_claims(prep.params.b(), &[]);
    let mut state = State::base(z_0, public_trace, acc_digest);

    // Each step runs the public verifier for that batch. The verifier sees
    // only claims and proof messages; witnesses stay prover-side.
    for (public_batch, step_proof) in proof.public_batches.iter().zip(&proof.steps) {
        super::validate_public_input_len(prep, public_batch)?;
        state = construction2::verify_step(
            &mut transcript,
            &prep.params,
            &prep.structure,
            prep.mix_rhos_commits,
            prep.combine_b_pows,
            &prep.vk,
            state,
            public_batch,
            step_proof,
        )?;
    }

    // `extend` stores each new batch as `latest` for the next step, so a
    // finished proof must include one terminal NIFS proof for the trailing
    // latest. This verifies that final fold and returns the verifier-derived
    // public accumulator.
    state = construction2::verify_final_fold(
        &mut transcript,
        &prep.params,
        &prep.structure,
        prep.mix_rhos_commits,
        prep.combine_b_pows,
        &prep.vk,
        state,
        proof.final_fold.as_ref(),
    )?;

    // Final acceptance compares verifier-derived public state and final CE claims.
    // In particular, `recorded.acc_digest` is intentionally ignored below:
    // a digest is compact derived context, not verifier authority.
    check_final_state_matches(prep, &state, &proof.state)?;
    Ok(())
}

fn check_proof_shape(proof: &Uncompressed) -> Result<(), Error> {
    if proof.steps.len() != proof.public_batches.len() {
        return Err(Error::UncompressedShapeMismatch {
            steps: proof.steps.len(),
            batches: proof.public_batches.len(),
        });
    }
    Ok(())
}

fn check_final_state_matches(prep: &Preprocessing, verified: &State, recorded: &State) -> Result<(), Error> {
    // Compare chain coordinates that are independently recomputed. Do not
    // compare `acc_digest`: the concrete final accumulator claims are
    // checked in `check_final_proof_state_matches`.
    if verified.chunk_count != recorded.chunk_count
        || verified.step_count != recorded.step_count
        || verified.z_0 != recorded.z_0
        || verified.z_i != recorded.z_i
        || verified.pc != recorded.pc
        || verified.public_trace != recorded.public_trace
    {
        return Err(Error::UncompressedStateMismatch);
    }
    check_final_proof_state_matches(prep, &verified.proof, &recorded.proof)?;
    Ok(())
}

fn check_final_proof_state_matches(
    prep: &Preprocessing,
    verified: &ProofState,
    recorded: &ProofState,
) -> Result<(), Error> {
    match (verified, recorded) {
        (ProofState::Initial, ProofState::Initial) => Ok(()),
        (
            ProofState::Active {
                running: verified_running,
                latest: verified_latest,
            },
            ProofState::Active {
                running: recorded_running,
                latest: recorded_latest,
            },
        ) => {
            if !verified_latest.instances.is_empty() || !recorded_latest.instances.is_empty() {
                return Err(Error::UncompressedStateMismatch);
            }
            // This is the trust-bearing accumulator check for the
            // uncompressed verifier: NIFS.V's computed output must equal
            // the final public CE claims recorded in the proof object.
            if verified_running.claims != recorded_running.claims {
                return Err(Error::UncompressedStateMismatch);
            }
            check_final_running_witnesses(prep, recorded_running)
        }
        _ => Err(Error::UncompressedStateMismatch),
    }
}

fn check_final_running_witnesses(prep: &Preprocessing, running: &RunningInstance) -> Result<(), Error> {
    if !running.shape_ok() {
        return Err(Error::FinalAccumulatorWitnessShapeMismatch);
    }
    for (index, (claim, witness)) in running.claims.iter().zip(&running.witnesses).enumerate() {
        if prep.log.commit(witness) != claim.c {
            return Err(Error::FinalAccumulatorWitnessCommitmentMismatch { index });
        }
    }
    Ok(())
}
