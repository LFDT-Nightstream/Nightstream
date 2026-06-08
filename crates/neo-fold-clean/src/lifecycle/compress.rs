//! Compression seam: `compress` (Spartan terminal SNARK) + `verify` (Spartan
//! verifier) + the public-image / decider-statement builders both share.
//!
//! The seam is wired, but the PR5 decider is not implemented yet, so public
//! `compress` / compressed `verify` return `decider::Error::Unsupported`.
//!
//! `build_decider_statement` is `pub` so tests can exercise
//! `decider::validate_witness` directly against the lifecycle's output
//! without re-deriving the statement shape.

use crate::lifecycle::{Compressed, Error, Preprocessing, PublicImage, Uncompressed, UncompressedAudit};
use crate::paper::construction2::{self, EncInst, ProofState};
use crate::paper::decider;
use crate::paper::digest;

/// Compress the uncompressed proof to a Spartan SNARK.
///
/// **Flushes the trailing latest** before handing to Spartan: the last
/// `extend` left `state.proof.latest` un-folded (it'd be the input to the
/// *next* extend, but there is no next). Compression folds it now via
/// one final NIFS.P call so the final `running` accumulator covers every
/// batch the user passed.
///
/// Takes an [`UncompressedAudit`] because Spartan's terminal-compression
/// statement consumes the chain audit trail (`steps`, `public_batches`)
/// to bind the public image to a verifiable history.
pub fn compress(prep: &Preprocessing, audit: UncompressedAudit) -> Result<Compressed, Error> {
    let post_audit = finish_uncompressed_with_audit(prep, audit)?;
    super::verify::verify_uncompressed_audit(prep, &post_audit)?;
    let statement = build_decider_statement(prep, &post_audit);
    let public_image = statement.public.clone();
    let (snark_proof, vk_digest) = decider::prove(&statement)?;
    Ok(Compressed {
        proof: snark_proof,
        vk: vk_digest,
        public_image,
    })
}

/// Finalize an [`UncompressedAudit`] into a **terminal-only**
/// [`Uncompressed`] by folding the trailing `latest` into the running
/// accumulator and dropping the per-step audit trail.
///
/// Pass the result to [`super::verify::verify_uncompressed`] (the
/// non-replay IVC verifier). If you also need the audit trail —
/// e.g. for the Spartan decider or for chain-replay debugging — use
/// [`finish_uncompressed_with_audit`] instead.
pub fn finish_uncompressed(prep: &Preprocessing, audit: UncompressedAudit) -> Result<Uncompressed, Error> {
    Ok(finish_uncompressed_with_audit(prep, audit)?.proof)
}

/// Diagnostic / decider variant of [`finish_uncompressed`] — finalizes
/// while **keeping** the per-step audit trail.
///
/// Same finalization work as [`finish_uncompressed`] (one terminal NIFS.P
/// call to flush the trailing `latest`); the difference is the return
/// type. Use this when you need the audit trail downstream — concretely
/// only these three call sites should reach for it:
///
/// 1. The Spartan compressed-decider statement (via
///    [`build_decider_statement`] → [`compress`]).
/// 2. The chain-replay verifier
///    [`super::verify::verify_uncompressed_audit`] (debugging / red-team
///    coverage of audit-trail tampers).
/// 3. Tests that intentionally mutate `steps` / `public_batches` to
///    exercise the chain-replay verifier.
///
/// Terminal-only callers use [`finish_uncompressed`] +
/// [`super::verify::verify_uncompressed`]; the audit trail is dropped
/// because that verifier never reads it. Multi-chunk F' callers
/// must keep this audit-bearing form until the compressed decider is
/// wired, because the per-step recursive-link evidence lives here.
pub fn finish_uncompressed_with_audit(
    prep: &Preprocessing,
    audit: UncompressedAudit,
) -> Result<UncompressedAudit, Error> {
    let UncompressedAudit {
        proof: Uncompressed { state, final_fold },
        steps,
        public_batches,
    } = audit;

    if final_fold.is_some() {
        let result = UncompressedAudit {
            proof: Uncompressed { state, final_fold },
            steps,
            public_batches,
        };
        check_already_finalized_consistency(prep, &result.proof)?;
        return Ok(result);
    }
    check_trailing_latest_batch_size(prep, &state)?;

    let (post_state, final_fold) = construction2::prove_final_fold(
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.structure_digest(),
        &prep.log,
        prep.mix_rhos_commits,
        prep.combine_b_pows,
        &prep.vk,
        state,
        prep.semantic_state_mode,
    )?;
    Ok(UncompressedAudit {
        proof: Uncompressed {
            state: post_state,
            final_fold,
        },
        steps,
        public_batches,
    })
}

fn check_trailing_latest_batch_size(
    prep: &Preprocessing,
    state: &crate::paper::construction2::State,
) -> Result<(), Error> {
    let ProofState::Active { latest, .. } = &state.proof else {
        return Ok(());
    };
    let max = prep.params.max_fresh_count();
    if latest.instances.len() > max {
        return Err(Error::BatchTooLarge {
            got: latest.instances.len(),
            max,
        });
    }
    Ok(())
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

    let expected_acc_digest = if running.claims.is_empty() {
        digest::AccumulatorHandle::empty().digest()
    } else {
        let parent = running
            .parent_authority
            .as_ref()
            .ok_or(Error::FinalizedProofInconsistent)?;
        digest::AccumulatorHandle::from_running_parts(&running.claims, Some(parent)).digest()
    };
    if proof.state.acc_digest != expected_acc_digest {
        return Err(Error::FinalizedProofInconsistent);
    }

    let expected_x_out = construction2::compute_x_out(
        &prep.vk,
        &prep.params,
        prep.structure_digest(),
        &proof.state,
        prep.semantic_state_mode,
    );
    if final_fold.x_out != expected_x_out {
        return Err(Error::FinalizedProofInconsistent);
    }

    Ok(())
}

/// Verify a compressed proof against the expected public image.
pub fn verify(_prep: &Preprocessing, compressed: &Compressed) -> Result<(), Error> {
    decider::verify(&compressed.public_image, &compressed.vk, &compressed.proof).map_err(Into::into)
}

// ──────────────────────────────────────────────────────────────────────────
// Statement / public-image builders.
// ──────────────────────────────────────────────────────────────────────────

fn build_public_image(prep: &Preprocessing, audit: &UncompressedAudit) -> PublicImage {
    let x_out = audit
        .proof
        .final_fold
        .as_ref()
        .map(|f| f.x_out.clone())
        .or_else(|| audit.steps.last().map(|s| s.x_out.clone()))
        .unwrap_or_else(|| EncInst::from_digest([0u8; 32]));
    PublicImage {
        vk_fs_digest: prep.vk.digest(),
        chunk_count: audit.proof.state.chunk_count,
        step_count: audit.proof.state.step_count,
        z_0: audit.proof.state.z_0,
        z_i: audit.proof.state.z_i,
        pc: audit.proof.state.pc,
        initial_semantic_state_digest: audit.proof.state.initial_semantic_state_digest,
        semantic_state_digest: audit.proof.state.semantic_state_digest,
        acc_digest: audit.proof.state.acc_digest,
        public_trace: audit.proof.state.public_trace,
        x_out,
    }
}

/// Build the decider statement from a finalized [`UncompressedAudit`].
/// Public for tests that exercise `decider::validate_witness` against
/// real lifecycle output.
///
/// `audit` is expected to be the output of [`finish_uncompressed_with_audit`]:
/// trailing `latest` flushed away, `audit.proof.state.proof` Active with
/// empty `latest`.
pub fn build_decider_statement(prep: &Preprocessing, audit: &UncompressedAudit) -> decider::Statement {
    let public = build_public_image(prep, audit);
    let witness = decider::Witness {
        steps: audit.steps.clone(),
        public_batches: audit.public_batches.clone(),
        final_fold: audit.proof.final_fold.clone(),
        final_state: audit.proof.state.clone(),
        terminal_ce_proof: None,
    };
    decider::Statement { public, witness }
}
