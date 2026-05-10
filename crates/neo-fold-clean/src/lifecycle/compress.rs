//! Compression seam: `compress` (Spartan terminal SNARK) + `verify` (Spartan
//! verifier) + the public-image / decider-statement builders both share.
//!
//! The seam is wired, but the PR5 decider is not implemented yet, so public
//! `compress` / compressed `verify` return `decider::Error::Unsupported`.

use crate::lifecycle::{Compressed, Error, Preprocessing, PublicImage, Uncompressed};
use crate::paper::construction2::{self, EncInst, ProofState, State};
use crate::paper::decider;
use crate::paper::digest;

/// Compress the uncompressed proof to a Spartan SNARK.
///
/// **Flushes the trailing latest** before handing to Spartan: the last
/// `extend` left `state.proof.latest` un-folded (it'd be the input to the
/// *next* extend, but there is no next). Compression folds it now via
/// one final NIFS.P call so the final `running` accumulator covers every
/// batch the user passed.
pub fn compress(prep: &Preprocessing, proof: Uncompressed) -> Result<Compressed, Error> {
    let post_proof = finish_uncompressed(prep, proof)?;
    super::verify::verify_uncompressed(prep, &post_proof)?;
    let public_image = build_public_image(prep, &post_proof);
    let statement = build_decider_statement(prep, &post_proof, &public_image);
    let (snark_proof, vk_digest) = decider::prove(&statement)?;
    Ok(Compressed {
        proof: snark_proof,
        vk: vk_digest,
        public_image,
    })
}

/// Finalize an uncompressed proof by folding the trailing `latest` into the
/// running accumulator and retaining the terminal NIFS proof.
pub fn finish_uncompressed(prep: &Preprocessing, proof: Uncompressed) -> Result<Uncompressed, Error> {
    let Uncompressed {
        state,
        steps,
        public_batches,
        final_fold,
        mut transcript,
    } = proof;
    if final_fold.is_some() {
        let proof = Uncompressed {
            state,
            steps,
            public_batches,
            final_fold,
            transcript,
        };
        check_already_finalized_consistency(prep, &proof)?;
        return Ok(proof);
    }
    let (post_state, final_fold) = construction2::prove_final_fold(
        &mut transcript,
        &prep.params,
        &prep.structure,
        &prep.log,
        prep.mix_rhos_commits,
        prep.combine_b_pows,
        &prep.vk,
        state,
    )?;
    Ok(Uncompressed {
        state: post_state,
        steps,
        public_batches,
        final_fold,
        transcript,
    })
}

fn check_already_finalized_consistency(prep: &Preprocessing, proof: &Uncompressed) -> Result<(), Error> {
    let final_fold = proof
        .final_fold
        .as_ref()
        .ok_or(Error::FinalizedProofInconsistent)?;
    let ProofState::Active { running, latest } = &proof.state.proof else {
        return Err(Error::FinalizedProofInconsistent);
    };
    if !latest.instances.is_empty() {
        return Err(Error::FinalizedProofInconsistent);
    }

    let expected_acc_digest = digest::accumulator_digest_from_claims(prep.params.b(), &running.claims);
    if proof.state.acc_digest != expected_acc_digest {
        return Err(Error::FinalizedProofInconsistent);
    }

    let expected_x_out = construction2::compute_x_out(&prep.vk, &prep.params, &prep.structure, &proof.state);
    if final_fold.x_out != expected_x_out {
        return Err(Error::FinalizedProofInconsistent);
    }

    Ok(())
}

/// Verify a compressed proof against the expected public image.
pub fn verify(prep: &Preprocessing, compressed: &Compressed) -> Result<(), Error> {
    let statement = build_decider_statement_for_verify(prep, &compressed.public_image);
    decider::verify(&statement, &compressed.vk, &compressed.proof).map_err(Into::into)
}

// ──────────────────────────────────────────────────────────────────────────
// Statement / public-image builders.
// ──────────────────────────────────────────────────────────────────────────

fn build_public_image(prep: &Preprocessing, proof: &Uncompressed) -> PublicImage {
    let x_out = proof
        .final_fold
        .as_ref()
        .map(|f| f.x_out.clone())
        .or_else(|| proof.steps.last().map(|s| s.x_out.clone()))
        .unwrap_or_else(|| EncInst::from_digest([0u8; 32]));
    PublicImage {
        chunk_count: proof.state.chunk_count,
        step_count: proof.state.step_count,
        z_0: proof.state.z_0,
        z_i: proof.state.z_i,
        pc: proof.state.pc,
        acc_digest: proof.state.acc_digest,
        public_trace: proof.state.public_trace,
        x_out,
        vk_fs_digest: prep.vk.digest(),
    }
}

fn build_decider_statement(
    prep: &Preprocessing,
    proof: &Uncompressed,
    public_image: &PublicImage,
) -> decider::Statement {
    decider::Statement {
        vk: prep.vk.clone(),
        state: proof.state.clone(),
        final_fold: proof.final_fold.clone(),
        x_out: public_image.x_out.clone(),
    }
}

fn build_decider_statement_for_verify(prep: &Preprocessing, public_image: &PublicImage) -> decider::Statement {
    // Verifier rebuilds the State view it expects from the public image
    // alone. ProofState::Initial because U_i is bound through the SNARK, not
    // carried here.
    let state = State {
        chunk_count: public_image.chunk_count,
        step_count: public_image.step_count,
        z_0: public_image.z_0,
        z_i: public_image.z_i,
        pc: public_image.pc,
        acc_digest: public_image.acc_digest,
        public_trace: public_image.public_trace,
        proof: ProofState::Initial,
    };
    decider::Statement {
        vk: prep.vk.clone(),
        state,
        final_fold: None,
        x_out: public_image.x_out.clone(),
    }
}
