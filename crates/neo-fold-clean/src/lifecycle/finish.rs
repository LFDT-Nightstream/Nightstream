//! Terminal finalization and decider-statement construction.
//!
//! `build_decider_statement` is `pub` so tests can exercise
//! `decider::validate_witness` directly against the lifecycle's output
//! without re-deriving the statement shape.

use crate::lifecycle::{Error, Preprocessing, PublicImage, Uncompressed, UncompressedAudit};
use crate::paper::construction2::{self, EncInst, ProofState};
use crate::paper::decider;
use crate::paper::nifs::NifsProverAdapter;

/// Finalize an [`UncompressedAudit`] into a compact [`Uncompressed`] proof.
/// Plain authoritative F' keeps HyperNova's running/latest pair; Nebula and
/// one-chunk terminal relations use the terminal-fold representation.
///
/// Pass the result to [`super::verify::verify_uncompressed`] (the
/// non-replay IVC verifier). If you also need the audit trail —
/// e.g. for the checked decider relation or chain-replay debugging — use
/// [`finish_uncompressed_with_audit`] instead.
pub fn finish_uncompressed(prep: &Preprocessing, audit: UncompressedAudit) -> Result<Uncompressed, Error> {
    prep.validate_verifier_key_binding()?;
    // HyperNova Construction 2 verifies the running accumulator and newest
    // fresh F' instance separately. Once preprocessing certifies that F'
    // itself constrains the preceding NIFS.V fold, a plain chain needs no
    // extra terminal fold: preserve `(running, latest)` for that verifier.
    // Nebula remains on the final-fold path because its one-step-delayed
    // memory claim must still be consumed and the segment closed.
    if prep.enforces_terminal_induction() && prep.nebula().is_none() {
        check_trailing_latest_batch_size(prep, &audit.proof.state)?;
        if audit.proof.final_fold.is_some() {
            return Err(Error::FinalizedProofInconsistent);
        }
        return Ok(audit.proof);
    }
    Ok(finish_uncompressed_with_audit(prep, audit)?.proof)
}

/// Diagnostic / decider variant of [`finish_uncompressed`] — finalizes
/// while **keeping** the per-step audit trail.
///
/// Same finalization work as [`finish_uncompressed`] (one terminal NIFS.P
/// call to flush the trailing `latest`); the difference is the return
/// type. Use it to build the checked decider statement, replay the chain, or
/// test audit-trail tampering.
///
/// Terminal-only callers use [`finish_uncompressed`] +
/// [`super::verify::verify_uncompressed`]; the audit trail is dropped
/// because that verifier never reads it. Authoritative generic and Nebula F'
/// relations carry multi-chunk induction inside the folded relation;
/// historical image-only callers must keep this audit-bearing form for replay.
pub fn finish_uncompressed_with_audit(
    prep: &Preprocessing,
    audit: UncompressedAudit,
) -> Result<UncompressedAudit, Error> {
    finish_uncompressed_with_audit_inner(prep, None, audit)
}

pub fn finish_uncompressed_with_audit_and_nifs_adapter(
    prep: &Preprocessing,
    adapter: &mut dyn NifsProverAdapter,
    audit: UncompressedAudit,
) -> Result<UncompressedAudit, Error> {
    finish_uncompressed_with_audit_inner(prep, Some(adapter), audit)
}

fn finish_uncompressed_with_audit_inner(
    prep: &Preprocessing,
    adapter: Option<&mut dyn NifsProverAdapter>,
    audit: UncompressedAudit,
) -> Result<UncompressedAudit, Error> {
    prep.validate_verifier_key_binding()?;
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

    let lanes = prep.nebula().map(|cfg| &cfg.scheme);
    let delayed_nebula = prep
        .enforces_terminal_induction()
        .then(|| prep.nebula())
        .flatten();
    let (post_state, final_fold) = if let Some(adapter) = adapter {
        construction2::prove_final_fold_with_adapter(
            adapter,
            &prep.params,
            prep.structure(),
            prep.optimized_cache(),
            prep.structure_digest(),
            &prep.log,
            prep.mix_rhos_commits,
            prep.combine_b_pows,
            &prep.vk,
            lanes,
            delayed_nebula,
            state,
            prep.semantic_state_mode,
        )?
    } else {
        construction2::prove_final_fold(
            &prep.params,
            prep.structure(),
            prep.optimized_cache(),
            prep.structure_digest(),
            &prep.log,
            prep.mix_rhos_commits,
            prep.combine_b_pows,
            &prep.vk,
            lanes,
            delayed_nebula,
            state,
            prep.semantic_state_mode,
        )?
    };
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
    let expected_acc_digest = running
        .accumulator_digest(prep.params.b(), prep.structure())
        .map_err(|_| Error::FinalizedProofInconsistent)?;
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
    };
    decider::Statement { public, witness }
}
